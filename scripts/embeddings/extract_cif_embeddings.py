#!/usr/bin/env python3
"""
extract_cif_embeddings.py

Forward-pass each CIF through a model once and save mean-pooled hidden states for all
(or selected) transformer layers in parallel.

Two model families are supported, selected with --model:

    crystallm  CrystaLLM v1 large. 16 blocks, 1024-dim, nanoGPT-style checkpoint
               (ckpt.pt), and CrystaLLM's own CIF tokenizer -- roughly one token per
               CIF field.
    llamat2    LLaMat-2 (m3rg-iitd), LLaMA-2 7B continued-pretrained on materials
               text. 32 blocks, 4096-dim, HuggingFace format, general BPE tokenizer.

The CIF TEXT fed to both is identical -- the same preprocess_cif / strip_symmetry pass
below -- so the comparison is between models rather than between inputs. Everything
downstream of the forward pass (length-sorted batching, masked mean pooling, the
checkpoint/resume scheme, the parquet schema) is shared.

What is NOT comparable across the two: a layer index, and any artifact fitted on one.
Layer 7 is 7/16 of the way through crystallm and 7/32 through llamat2, the hidden
sizes differ, and there is no shared basis, so a steering vector, PCA basis or manifold
belongs to exactly one model. See utils.MODELS.

Setup:
    crystallm:  cd CrystaLLM && python bin/download.py crystallm_v1_large.tar.gz
                tar -xzf crystallm_v1_large.tar.gz
    llamat2:    huggingface-cli download m3rg-iitd/<checkpoint> --local-dir <dir>
                Needs a venv with `transformers`; none of the three existing venvs
                has it (crystallm_venv is nanoGPT-only, relax/megnet are for M3GNet
                and MEGNet). See README.

Usage:
    source CrystaLLM/crystallm_venv/bin/activate
    python scripts/embeddings/extract_cif_embeddings.py \
        --model crystallm \
        --ckpt-dir CrystaLLM/crystallm_v1_large \
        --pkl CrystaLLM/cifs_v1_prep.pkl.gz \
        [--layers 0,7,14]   # comma-separated; omit for all layers \
        [--batch-size 32] \
        [--limit 1000]

Outputs: embeddings/<dataset>/<model>/<variant>/cif_layer{N}/checkpoint_XXXXX.parquet
for each layer N, where <dataset> is --dataset (inferred from the pkl stem if omitted:
cifs_v1_mp_prep -> v1_mp, else v1_all), <model> is --model, and <variant> is --variant:

    full   CIFs exactly as stored (default)
    nosym  the two lines that state the symmetry outright are removed first:
             _symmetry_space_group_name_H-M   P4/mmm
             _symmetry_Int_Tables_number      123
           Needed because those lines put the label directly in the input, so a probe
           on `full` embeddings recovers the space group by reading a copied token
           rather than by finding a learned representation. With them gone the model
           only has the cell parameters and coordinates to go on.
"""

import argparse
import gzip
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import DEFAULT_MODEL, MODELS, embeddings_paths


# ---------------------------------------------------------------------------------
# CIF text preparation -- shared by every backend, so both models read the same bytes
# ---------------------------------------------------------------------------------

def load_cifs(pkl_path: str, limit: int = None):
    print(f"Loading {pkl_path} ...")
    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if limit:
        data = data[:limit]
    print(f"Loaded {len(data):,} entries")
    return data  # list of (id, cif_string)


def preprocess_cif(cif: str) -> str:
    """Strip comment lines, empty lines, and pymatgen metadata — matching tokenize_cifs.py.

    Not optional: bin/tokenize_cifs.py:92 runs this before building train.bin, so the
    model only ever saw stripped CIFs. cifs_v1_prep.pkl.gz is "prep" in a different
    sense (semisymmetrize + atomic props + rounding, bin/preprocess.py:51-53) and still
    carries the '# MP Entry mp-1217888 ...' / '# generated using pymatgen' header. Feeding
    that in adds ~17% more tokens than training ever had (400 vs 343 tokens/CIF), which
    tokenize to <unk> plus the material id spelled out digit by digit -- and mean pooling
    mixes all of it into every embedding.
    """
    lines = cif.split("\n")
    cleaned = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#") and "pymatgen" not in l]
    cleaned.append("\n")
    return "\n".join(cleaned)


# The two keys that name the symmetry outright. Both are dropped for --variant nosym;
# either one alone would leave the label fully recoverable.
SYMMETRY_KEYS = ("_symmetry_space_group_name_H-M", "_symmetry_Int_Tables_number")


def strip_symmetry(cif: str) -> str:
    """Remove the lines stating the space group, leaving cell and coordinates intact.

    Note what is deliberately NOT removed: _symmetry_equiv_pos_as_xyz (already collapsed
    to a single 'x, y, z' by the CrystaLLM preprocessing) and
    _atom_site_symmetry_multiplicity, which leaks the Wyckoff signature but is part of
    the structure description rather than a restatement of the space group.
    """
    return "\n".join(l for l in cif.split("\n")
                     if not l.strip().startswith(SYMMETRY_KEYS))


def mean_pool(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean pool hidden states over true (non-padding) token positions.

    Upcast to float32 first: llamat2 runs in float16, and summing ~1500 half-precision
    positions accumulates visible error (half has ~3 decimal digits, and the running
    sum grows past where its steps stay exact). crystallm already computes in float32,
    so this is a no-op there.
    """
    hidden = hidden.float()
    B, T, D = hidden.shape
    mask = torch.arange(T, device=hidden.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
    mask_f = mask.unsqueeze(2).float()  # (B, T, 1)
    pooled = (hidden * mask_f).sum(dim=1) / lengths.unsqueeze(1).float()  # (B, D)
    return pooled


# ---------------------------------------------------------------------------------
# Backends -- the only architecture-specific code
# ---------------------------------------------------------------------------------
#
# Each backend exposes exactly what the extraction loop needs:
#
#   n_layer, n_embd, block_size   ints
#   blocks                        the per-layer nn.Modules to hook, in order
#   encode(cif) -> list[int]      token ids for one CIF, already truncated
#   forward(input_ids)            one no-grad forward; hooks do the capturing
#
# The heavy imports live INSIDE the loaders on purpose. crystallm pulls in omegaconf and
# a pinned pymatgen; transformers pulls in its own stack; and the two will not be
# installed in the same venv. A top-level import of either would make this file
# unimportable for the other model -- and seven scripts import load_model from here.


def load_model(model_dir: str, device: torch.device):
    """Load CrystaLLM from a nanoGPT-style ckpt.pt. Returns (model, config).

    Kept at this name and signature because steer_generate_cif.py, test_kv_cache.py,
    layer_causal_probe.py, layernorm_survival.py, manifold_distance.py,
    injection_magnitude.py and analyze_steering_norms.py all import it from here.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM"))
    from crystallm import GPTConfig, GPT

    ckpt = torch.load(os.path.join(model_dir, "ckpt.pt"), map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    # Disable dropout for inference. The checkpoint ships dropout=0.1, and the functional
    # SDPA dropout_p (_model.py) is NOT gated by model.eval(), so leaving it on drops ~10%
    # of attention weights on every forward -> nondeterministic generation. This loader is
    # inference-only, so force dropout to 0 here (covers SDPA + all nn.Dropout modules).
    config.dropout = 0.0
    model = GPT(config)
    state_dict = ckpt["model"]
    # strip compile prefix if present
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded model: {config.n_layer} layers, {config.n_embd} dim, block_size {config.block_size}")
    return model, config


class CrystaLLMBackend:
    """CrystaLLM v1: nanoGPT blocks at model.transformer.h, CIFTokenizer."""

    def __init__(self, ckpt_dir: str, device: torch.device, torch_dtype: str = None):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM"))
        from crystallm import CIFTokenizer

        self.model, config = load_model(ckpt_dir, device)
        self.tokenizer = CIFTokenizer()
        self.device = device
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.blocks = self.model.transformer.h
        self.pad_id = 0

    def encode(self, cif: str) -> list:
        return self.tokenizer.encode(self.tokenizer.tokenize_cif(cif))[:self.block_size]

    def forward(self, input_ids):
        self.model(input_ids)


class LlamatBackend:
    """LLaMat-2: HuggingFace LLaMA-2, blocks at model.model.layers, BPE tokenizer.

    Loaded in half precision by default -- a 7B model is ~27 GB in float32 and ~13 GB in
    float16, and only the latter leaves room for activations on a 32 GB card. bfloat16
    needs sm_80 (Ampere); the V100s here are sm_70, so float16 is the default.
    """

    def __init__(self, ckpt_dir: str, device: torch.device, torch_dtype: str = "float16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, torch_dtype=getattr(torch, torch_dtype), low_cpu_mem_usage=True)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        config = self.model.config
        self.device = device
        self.n_layer = config.num_hidden_layers
        self.n_embd = config.hidden_size
        self.block_size = config.max_position_embeddings
        self.blocks = self.model.model.layers
        # Padding is masked out of the pooled mean and, because attention is causal and
        # padding sits at the END of each row, no real token ever attends to it. So the
        # pad id only has to be a valid index. Llama tokenizers often ship without a pad
        # token, hence the fallback chain rather than tokenizer.pad_token_id alone.
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        self.pad_id = 0 if pad is None else pad
        print(f"Loaded model: {self.n_layer} layers, {self.n_embd} dim, "
              f"block_size {self.block_size}, dtype {torch_dtype}")

    def encode(self, cif: str) -> list:
        # add_special_tokens keeps the BOS the model was trained with.
        return self.tokenizer(cif, add_special_tokens=True,
                              truncation=True, max_length=self.block_size)["input_ids"]

    def forward(self, input_ids):
        # No attention_mask: right-padding plus causal attention means real tokens never
        # see the pads, and mean_pool drops them from the average.
        self.model(input_ids)


BACKENDS = {"crystallm": CrystaLLMBackend, "llamat2": LlamatBackend}
assert set(BACKENDS) == set(MODELS), "every registered model needs a backend"


def tokenize_batch(cif_strings, backend, device: torch.device):
    """Tokenize a list of CIF strings, truncate, pad to batch max length."""
    encoded = [backend.encode(cif) for cif in cif_strings]
    lengths = [len(ids) for ids in encoded]

    max_len = max(lengths)
    padded = torch.full((len(encoded), max_len), backend.pad_id,
                        dtype=torch.long, device=device)
    for i, ids in enumerate(encoded):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    return padded, torch.tensor(lengths, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------------

# float32 is what both models actually compute in, so storing float64 -- which is what
# .tolist() produced, since Python floats are C doubles -- wrote four bytes of zeros per
# dimension. float16 halves it again and is the practical choice for llamat2: 32 layers
# of 4096 dims over 2.29M structures is ~19 GB per layer even at half precision.
DTYPES = {"float16": np.float16, "float32": np.float32, "float64": np.float64}


def existing_dtype(ckpt_dir: Path):
    """The embedding dtype already used in a layer dir, or None if it is empty.

    Resuming with a different dtype writes shards that cannot be concatenated into one
    parquet, and consolidate_embeddings.py only discovers that at the end of a long job.
    Adopting what is already there makes resume always consistent.
    """
    for f in sorted(ckpt_dir.glob("checkpoint_*.parquet")):
        try:
            t = pq.read_table(f, columns=["embedding"]).schema.field("embedding").type
            return {"halffloat": "float16", "float": "float32", "double": "float64"} \
                .get(str(t.value_type), None)
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS),
                        help="Which model to run, and the subdir it is stored under "
                             "(see utils.MODELS). Distinct from --ckpt-dir.")
    parser.add_argument("--ckpt-dir", required=True,
                        help="Path to the checkpoint directory holding the weights.")
    parser.add_argument("--pkl", required=True, help="Path to cifs pkl.gz")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices to extract (default: all layers)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Write to disk every N batches (default: 100)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", default=None,
                        help="Embeddings subdir under embeddings/ to write to. "
                             "Default: inferred from the pkl stem "
                             "(cifs_v1_mp_prep -> v1_mp, else v1_all).")
    parser.add_argument("--variant", default="full", choices=["full", "nosym"],
                        help="full: CIFs as-is. nosym: drop the _symmetry_space_group_name_H-M "
                             "and _symmetry_Int_Tables_number lines first, so the space group "
                             "is not handed to the model verbatim (default: full).")
    parser.add_argument("--dtype", default="float32", choices=list(DTYPES),
                        help="Stored precision of the embedding vectors. float32 is what "
                             "the models compute in. Use float16 for llamat2, where a "
                             "single layer is ~19 GB over the full corpus even so. "
                             "Ignored if the layer dir already holds shards, whose dtype "
                             "is adopted instead so a resume stays concatenable.")
    parser.add_argument("--torch-dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Precision the model RUNS in (llamat2 only; crystallm "
                             "ignores it). bfloat16 needs sm_80+, the V100s are sm_70.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # which embeddings/<dataset>/<model>/<variant>/ subdir to write to
    dataset = args.dataset or (
        "v1_mp" if "mp" in Path(args.pkl).stem.lower().split("_") else "v1_all")
    print(f"Writing embeddings to embeddings/{dataset}/{args.model}/{args.variant}/")

    backend = BACKENDS[args.model](args.ckpt_dir, device, args.torch_dtype)

    layers = list(range(backend.n_layer)) if args.layers is None \
             else [int(x) for x in args.layers.split(",")]
    assert all(0 <= l < backend.n_layer for l in layers), \
        f"Some layers out of range (model has {backend.n_layer} layers)"

    out_dirs = {l: embeddings_paths(l, dataset, args.variant, args.model)[1] for l in layers}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Adopt the dtype already on disk, if any, so a resumed run stays concatenable.
    found = {existing_dtype(d) for d in out_dirs.values()} - {None}
    if len(found) > 1:
        raise SystemExit(f"Layer dirs already hold mixed embedding dtypes {sorted(found)} "
                         f"-- consolidate or clear them before resuming.")
    dtype_name = found.pop() if found else args.dtype
    if dtype_name != args.dtype:
        print(f"Existing shards are {dtype_name}; using that instead of --dtype {args.dtype}")
    store_dtype = DTYPES[dtype_name]
    bytes_per_vec = np.dtype(store_dtype).itemsize * backend.n_embd
    print(f"Storing embeddings as {dtype_name} "
          f"({bytes_per_vec:,} bytes/structure/layer, {len(layers)} layers)")

    done_ids_per_layer = {}
    for l, d in out_dirs.items():
        done = set()
        for f in d.glob("checkpoint_*.parquet"):
            try:
                done.update(pd.read_parquet(f, columns=["id"])["id"].tolist())
            except Exception:
                pass
        done_ids_per_layer[l] = done
        if done:
            print(f"Layer {l}: resuming — {len(done):,} entries already done")

    data = load_cifs(args.pkl, args.limit)

    # filter to entries not yet done (use layer 0 as reference)
    ref_done = done_ids_per_layer[layers[0]]
    data = [(id_, cif) for id_, cif in data if id_ not in ref_done]

    # Batch similar-length CIFs together. Lengths are median 317 tokens but reach 4126,
    # so in file order one long CIF pads its whole batch up to its length: 2.33x more
    # compute than the real tokens need. Sorting drops that to ~1.04x. Raw string length
    # is the sort key (corr 0.998 with token count) to avoid a full tokenization pass.
    #
    # Safe to reorder: each (id, cif) pair moves together and ids are written alongside
    # their own embeddings, so rows stay correctly paired; every downstream consumer
    # joins on id rather than row position. Results are unchanged, not just close --
    # attention is causal and padding sits at the end, so real tokens never attend to it,
    # and mean_pool masks it out.
    data.sort(key=lambda entry: len(entry[1]))
    total = len(data)
    print(f"Extracting {len(layers)} layers for {total:,} remaining entries, batch size {args.batch_size} ...")
    print(f"Layers: {layers}")

    captured = {}
    hooks = []
    for l in layers:
        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                # HF blocks return a tuple (hidden_states, ...); nanoGPT returns the
                # tensor. Taking [0] of a tensor would silently keep one row.
                captured[layer_idx] = out[0] if isinstance(out, tuple) else out
            return hook_fn
        hooks.append(backend.blocks[l].register_forward_hook(make_hook(l)))

    pending = {l: {"ids": [], "embeddings": []} for l in layers}
    checkpoint_num = {l: len(list(out_dirs[l].glob("checkpoint_*.parquet"))) for l in layers}
    completed = 0
    start_time = time.time()

    for i in range(0, total, args.batch_size):
        batch = data[i: i + args.batch_size]
        ids = [entry[0] for entry in batch]
        # preprocess_cif first so strip_symmetry sees stripped/normalized lines.
        cifs = [preprocess_cif(entry[1]) for entry in batch]
        if args.variant == "nosym":
            cifs = [strip_symmetry(c) for c in cifs]

        try:
            input_ids, lengths = tokenize_batch(cifs, backend, device)
            captured.clear()
            with torch.no_grad():
                backend.forward(input_ids)

            for l in layers:
                embeddings = mean_pool(captured[l], lengths).cpu().numpy().astype(store_dtype)
                pending[l]["ids"].extend(ids)
                pending[l]["embeddings"].extend(list(embeddings))

        except Exception as e:
            print(f"  WARNING: batch at {i} failed: {e}", flush=True)
            for l in layers:
                pending[l]["ids"].extend(ids)
                pending[l]["embeddings"].extend(
                    [np.zeros(backend.n_embd, dtype=store_dtype)] * len(ids))

        completed += len(batch)
        batch_num = i // args.batch_size + 1

        if batch_num % args.checkpoint_every == 0 or completed == total:
            for l in layers:
                if not pending[l]["ids"]:
                    continue
                ckpt_file = out_dirs[l] / f"checkpoint_{checkpoint_num[l]:05d}.parquet"
                pd.DataFrame({
                    "id": pending[l]["ids"],
                    "embedding": pending[l]["embeddings"],
                }).to_parquet(ckpt_file, index=False)
                checkpoint_num[l] += 1
                pending[l] = {"ids": [], "embeddings": []}

        elapsed = time.time() - start_time
        rate = completed / elapsed * 3600
        remaining = (total - completed) / (completed / elapsed) if completed else 0
        print(f"  {completed:,} / {total:,}  ({rate:,.0f}/hr, ~{remaining/3600:.1f}h left)", flush=True)

    for h in hooks:
        h.remove()
    print(f"\nDone. Outputs in embeddings/{dataset}/{args.model}/{args.variant}/"
          f"cif_layer{{N}}/ for layers {layers}")


if __name__ == "__main__":
    main()
