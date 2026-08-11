#!/usr/bin/env python3
"""
extract_cif_embeddings.py

Forward-pass each CIF through CrystaLLM once and save mean-pooled hidden states
for all (or selected) transformer layers in parallel.

Setup:
    cd CrystaLLM && python bin/download.py crystallm_v1_large.tar.gz
    tar -xzf crystallm_v1_large.tar.gz

Usage:
    source CrystaLLM/crystallm_venv/bin/activate
    python extract_cif_embeddings.py \
        --model CrystaLLM/crystallm_v1_large \
        --pkl CrystaLLM/cifs_v1_prep.pkl.gz \
        [--layers 0,7,14]   # comma-separated; omit for all layers \
        [--batch-size 16] \
        [--limit 1000]

Outputs: embeddings/<dataset>/<variant>/cif_layer{N}/checkpoint_XXXXX.parquet for each
layer N, where <dataset> is --dataset (inferred from the pkl stem if omitted:
cifs_v1_mp_prep -> v1_mp, else v1_all) and <variant> is --variant:

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

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM"))
from crystallm import CIFTokenizer, GPTConfig, GPT


def load_model(model_dir: str, device: torch.device):
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


def tokenize_batch(cif_strings, tokenizer: CIFTokenizer, block_size: int, device: torch.device):
    """Tokenize a list of CIF strings, truncate, pad to batch max length."""
    encoded = []
    lengths = []
    for cif in cif_strings:
        tokens = tokenizer.tokenize_cif(cif)
        ids = tokenizer.encode(tokens)[:block_size]
        encoded.append(ids)
        lengths.append(len(ids))

    max_len = max(lengths)
    # pad with 0 (will be masked out during pooling)
    padded = torch.zeros(len(encoded), max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(encoded):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    return padded, torch.tensor(lengths, dtype=torch.long, device=device)


def mean_pool(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean pool hidden states over true (non-padding) token positions."""
    # hidden: (B, T, D), lengths: (B,)
    B, T, D = hidden.shape
    mask = torch.arange(T, device=hidden.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
    mask_f = mask.unsqueeze(2).float()  # (B, T, 1)
    pooled = (hidden * mask_f).sum(dim=1) / lengths.unsqueeze(1).float()  # (B, D)
    return pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model dir containing ckpt.pt")
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # which embeddings/<dataset>/ subdir to write to
    dataset = args.dataset or (
        "v1_mp" if "mp" in Path(args.pkl).stem.lower().split("_") else "v1_all")
    print(f"Writing embeddings to embeddings/{dataset}/{args.variant}/")

    model, config = load_model(args.model, device)

    layers = list(range(config.n_layer)) if args.layers is None \
             else [int(x) for x in args.layers.split(",")]
    assert all(0 <= l < config.n_layer for l in layers), \
        f"Some layers out of range (model has {config.n_layer} layers)"

    out_dirs = {l: Path(f"embeddings/{dataset}/{args.variant}/cif_layer{l}") for l in layers}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

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

    tokenizer = CIFTokenizer()
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
                captured[layer_idx] = out
            return hook_fn
        hooks.append(model.transformer.h[l].register_forward_hook(make_hook(l)))

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
            input_ids, lengths = tokenize_batch(cifs, tokenizer, config.block_size, device)
            captured.clear()
            with torch.no_grad():
                model(input_ids)

            for l in layers:
                embeddings = mean_pool(captured[l], lengths).cpu().float()
                pending[l]["ids"].extend(ids)
                pending[l]["embeddings"].extend(embeddings.tolist())

        except Exception as e:
            print(f"  WARNING: batch at {i} failed: {e}", flush=True)
            for l in layers:
                pending[l]["ids"].extend(ids)
                pending[l]["embeddings"].extend([[0.0] * config.n_embd] * len(ids))

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
    print(f"\nDone. Outputs in embeddings/{dataset}/{args.variant}/cif_layer{{N}}/ for layers {layers}")


if __name__ == "__main__":
    main()
