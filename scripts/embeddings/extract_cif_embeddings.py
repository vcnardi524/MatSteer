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
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
import llamat_prompts
from utils import DEFAULT_MODEL, MODELS, VARIANTS, embeddings_paths


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


def mean_pool(hidden: torch.Tensor, starts: torch.Tensor,
              ends: torch.Tensor) -> torch.Tensor:
    """Mean pool hidden states over positions [start, end) of each row.

    WHY A SPAN AND NOT JUST A LENGTH. For crystallm the whole sequence is the structure,
    so start is 0 and this is the old whole-sequence mean. For llamat2_cif the sequence
    is a constant ~200-token prompt followed by the structure's crystal string, and
    attention is causal -- so every prompt position holds the IDENTICAL hidden state for
    all 154,871 structures. Pooling those in would make roughly half of every embedding
    one shared constant vector. start therefore skips the prompt.

    Padding sits beyond `end` and is excluded by the same mask.

    Upcast to float32 first: llamat2 runs in float16, and summing ~1500 half-precision
    positions accumulates visible error (half has ~3 decimal digits, and the running
    sum grows past where its steps stay exact). crystallm already computes in float32,
    so this is a no-op there.
    """
    hidden = hidden.float()
    B, T, D = hidden.shape
    pos = torch.arange(T, device=hidden.device).unsqueeze(0)             # (1, T)
    mask = (pos >= starts.unsqueeze(1)) & (pos < ends.unsqueeze(1))      # (B, T)
    n = mask.sum(1)
    if (n == 0).any():
        raise ValueError("a row has an empty pooling span -- nothing to average")
    pooled = (hidden * mask.unsqueeze(2).float()).sum(dim=1) / n.unsqueeze(1).float()
    return pooled


# ---------------------------------------------------------------------------------
# Backends -- the only architecture-specific code, and it lives in scripts/backends.py
# ---------------------------------------------------------------------------------
#
# They moved there because steered generation hooks the SAME `blocks` attribute this
# script does -- one to capture a hidden state, one to modify it -- and two copies of
# `model.transformer.h` vs `model.model.layers` would drift. See scripts/backends.py.
#
# Re-exported here, not just imported, because seven scripts do
# `from extract_cif_embeddings import load_model, load_cifs`.
#
# The heavy imports live INSIDE the loaders on purpose. crystallm pulls in omegaconf and
# a pinned pymatgen; transformers pulls in its own stack; and the two will not be
# installed in the same venv. A top-level import of either would make this file
# unimportable for the other model.

from backends import load_model, BACKENDS, CrystaLLMBackend, LlamatBackend   # noqa: F401,E402


# ---------------------------------------------------------------------------------
# Text builders -- what text a structure becomes, and which part of it gets pooled
# ---------------------------------------------------------------------------------
#
# A builder returns (prompt, answer). The embedding is pooled over the ANSWER only;
# the prompt is context the model reads first. Returning None skips the structure.
#
# This is a separate axis from the backend: the same LLaMA weights could be fed a raw
# CIF or a crystal string, and it is the TEXT that decides what a pooled vector means.


class RawCifBuilder:
    """The CIF itself, no prompt. What crystallm has always been given.

    prompt is "" so the pooled span is the whole sequence -- byte-for-byte the old
    behaviour, which is the regression test.
    """

    def __init__(self, variant: str):
        self.variant = variant

    def build(self, cif: str, cid: str):
        text = preprocess_cif(cif)
        if self.variant == "nosym":
            text = strip_symmetry(text)
        return "", text


class CrystalStringBuilder:
    """The crystal string, in the answer slot of the unconditional generation prompt.

    This is what LLaMat-2-CIF was tuned to WRITE. See scripts/llamat_prompts.py for why
    a CIF would be the wrong text to embed for steering work.

    Every structure is checked against its own `_chemical_formula_sum` before encoding.
    Symmetry expansion can emit more sites than the formula states -- measured at 1.73%
    on the prep pickle and 0.07% on the originals -- and since the crystal string lists
    one line per site, a mismatch is a structurally wrong input, not a cosmetic one.
    Those are skipped and logged rather than embedded.
    """

    def __init__(self, system_index: int = 0, wrapper: str = "notebook",
                 on_overflow: str = "skip"):
        from pymatgen.core.structure import Structure
        self._Structure = Structure
        self.prompt = llamat_prompts.unconditional_prompt(system_index, wrapper)
        self.on_overflow = on_overflow
        self.max_sites = None          # set once the backend's block_size is known

    def build(self, cif: str, cid: str):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # pymatgen is loud about occupancies
                s = self._Structure.from_str(cif, fmt="cif")
        except Exception as e:
            raise SkipStructure(f"parse failed: {type(e).__name__}") from e
        stated = stated_composition(cif)
        got = {str(el): int(n) for el, n in s.composition.get_el_amt_dict().items()}
        if stated is not None and got != stated:
            raise SkipStructure(f"composition {got} != stated {stated}")
        if self.on_overflow == "skip" and self.max_sites and len(s) > self.max_sites:
            raise SkipStructure(f"{len(s)} sites exceeds the context budget "
                                f"({self.max_sites})")
        return self.prompt, llamat_prompts.crystal_string(s)

    def set_budget(self, block_size: int, n_prompt_tokens: int,
                   tokens_per_site: float = 18.0):
        """Largest site count that fits, from the measured 18.0 tokens/site on v1_mp.

        A cheap pre-filter so an oversized structure is rejected before it is tokenised
        and before it inflates a batch's padding. The exact check still happens at
        tokenisation; this only avoids the obvious cases.
        """
        self.max_sites = int((block_size - n_prompt_tokens - 1) / tokens_per_site)


class SkipStructure(Exception):
    """Raised by a builder for a structure that must not be embedded."""


_FORMULA_SUM = re.compile(r"_chemical_formula_sum\s+(?:'([^']+)'|(\S+))")
_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")


def stated_composition(cif: str):
    """{element: count} from the CIF's own _chemical_formula_sum, or None if absent."""
    m = _FORMULA_SUM.search(cif)
    if not m:
        return None
    out = {}
    for el, cnt in _ELEMENT.findall(m.group(1) or m.group(2)):
        if el:
            out[el] = out.get(el, 0) + int(cnt or 1)
    return out


def build_text_builder(args):
    if args.variant == "crystal_uncond":
        return CrystalStringBuilder(args.system_index, args.wrapper, args.on_overflow)
    return RawCifBuilder(args.variant)


def pad_batch(encoded, pad_id: int, device: torch.device):
    """Pad a list of (ids, answer_start) to the batch max. Returns ids, starts, ends.

    starts/ends bound the ANSWER span of each row, which is what mean_pool averages.
    """
    ends = [len(ids) for ids, _ in encoded]
    starts = [start for _, start in encoded]
    if any(s >= e for s, e in zip(starts, ends)):
        raise ValueError("a row has no answer tokens -- the span is empty")

    max_len = max(ends)
    padded = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
    for i, (ids, _) in enumerate(encoded):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    return (padded,
            torch.tensor(starts, dtype=torch.long, device=device),
            torch.tensor(ends, dtype=torch.long, device=device))


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


def flush(out_dirs, layers, pending, checkpoint_num, skipped, truncated,
          skip_path, trunc_path):
    """Write whatever is buffered. Safe to call with nothing pending.

    Split out so the end of the run can call it unconditionally. Relying on the loop
    alone lost the tail once already: the last batches were entirely skipped, so the
    in-loop call never ran.
    """
    for l in layers:
        if not pending[l]["id"]:
            continue
        ckpt_file = out_dirs[l] / f"checkpoint_{checkpoint_num[l]:05d}.parquet"
        pd.DataFrame(pending[l]).to_parquet(ckpt_file, index=False)
        checkpoint_num[l] += 1
        pending[l] = {"id": [], "embedding": [], "n_answer_tokens": []}
    if skipped:
        pd.DataFrame(skipped).to_csv(skip_path, index=False)
    if truncated:
        pd.DataFrame(truncated).to_csv(trunc_path, index=False)


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
    parser.add_argument("--variant", default="full", choices=list(VARIANTS),
                        help="What text a structure becomes. full: the CIF as-is. nosym: "
                             "the same with _symmetry_space_group_name_H-M and "
                             "_symmetry_Int_Tables_number dropped, so the space group is "
                             "not handed to the model verbatim. crystal_uncond: the "
                             "compact crystal string in the answer slot of LLaMat's "
                             "unconditional generation prompt -- only the answer tokens "
                             "are pooled (default: full).")
    parser.add_argument("--system-index", type=int, default=0,
                        help="Which of the 10 GENERATION_SYSTEMS messages to use for "
                             "crystal_uncond. Training picked one at random per example; "
                             "extraction fixes one so only the structure varies.")
    parser.add_argument("--on-overflow", default="skip", choices=("skip", "truncate"),
                        help="What to do when prompt+crystal string exceeds the model "
                             "context (2048 for llamat-2-cif). skip: drop the structure "
                             "and log it -- a truncated crystal string describes a cell "
                             "with FEWER ATOMS, so its embedding is not that structure's. "
                             "truncate: keep it, flagged by n_answer_tokens. Measured on "
                             "v1_mp: 8.1%% overflow, and they are the large cells "
                             "(median 142 sites vs 26 kept), so this choice biases the "
                             "corpus either way -- report which was used.")
    parser.add_argument("--wrapper", default="notebook",
                        choices=list(llamat_prompts.WRAPPERS),
                        help="Prompt wrapper for crystal_uncond. Unresolved which the "
                             "released checkpoint was trained with: the committed "
                             "pipeline uses chatml, every inference script the authors "
                             "wrote uses notebook (default: notebook).")
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

    builder = build_text_builder(args)
    if isinstance(builder, CrystalStringBuilder):
        print(f"Prompt ({args.wrapper} wrapper, system {args.system_index}, "
              f"{len(builder.prompt)} chars) is constant; pooling the answer span only.")

    backend = BACKENDS[args.model](args.ckpt_dir, device, args.torch_dtype)
    if isinstance(builder, CrystalStringBuilder):
        n_prompt = len(backend.encode_pair(builder.prompt, "")[0])
        builder.set_budget(backend.block_size, n_prompt)
        print(f"  prompt is {n_prompt} tokens of the {backend.block_size} context; "
              f"--on-overflow={args.on_overflow} above ~{builder.max_sites} sites")

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
    n_corpus = len(data)          # before the resume filter, for the accounting check

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

    # keys are the parquet column names: the dict is written straight to DataFrame
    pending = {l: {"id": [], "embedding": [], "n_answer_tokens": []} for l in layers}
    skip_path = out_dirs[layers[0]].parent / "skipped.csv"
    trunc_path = out_dirs[layers[0]].parent / "truncated.csv"
    skipped, truncated = [], []
    checkpoint_num = {l: len(list(out_dirs[l].glob("checkpoint_*.parquet"))) for l in layers}
    completed = 0
    start_time = time.time()

    for i in range(0, total, args.batch_size):
        batch = data[i: i + args.batch_size]

        # Build text per structure. A builder may reject one (unparseable, or a
        # composition that disagrees with the CIF's own formula), and a rejected
        # structure is dropped from the batch rather than embedded -- a missing row
        # falls out of every downstream join, a wrong one does not.
        ids, encoded = [], []
        for cid, cif in batch:
            try:
                prompt, answer = builder.build(cif, cid)
                seq, start = backend.encode_pair(prompt, answer)
                # EXACT length check. The builder's site-count filter is an estimate
                # (18.0 tokens/site); this is the real number, so nothing is truncated
                # without a decision being recorded.
                if len(seq) > backend.block_size:
                    if args.on_overflow == "skip":
                        raise SkipStructure(
                            f"{len(seq)} tokens exceeds context {backend.block_size}")
                    seq, n_over = seq[:backend.block_size], len(seq) - backend.block_size
                    truncated.append({"id": cid, "dropped_tokens": n_over})
                    if start >= len(seq):
                        raise SkipStructure("truncation left no answer tokens")
            except SkipStructure as e:
                skipped.append({"id": cid, "reason": str(e)})
                continue
            ids.append(cid)
            encoded.append((seq, start))
        # NO `continue` when the whole batch was skipped. An earlier version bailed out
        # here, which jumped over the checkpoint block below -- and that block is where
        # BOTH the embeddings and the skip log get written. The corpus is sorted
        # shortest-first, so the tail is entirely over-context structures and every tail
        # batch took that branch: neither buffer was flushed again, and 3,584 structures
        # vanished from both outputs at once. Missing from the skip log too, they could
        # not even be accounted for. Guard the forward pass instead of the loop body.
        if encoded:
            try:
                input_ids, starts, ends = pad_batch(encoded, backend.pad_id, device)
                captured.clear()
                with torch.no_grad():
                    backend.forward(input_ids)

                n_answer = (ends - starts).cpu().numpy()
                for l in layers:
                    embeddings = mean_pool(captured[l], starts,
                                           ends).cpu().numpy().astype(store_dtype)
                    pending[l]["id"].extend(ids)
                    pending[l]["embedding"].extend(list(embeddings))
                    pending[l]["n_answer_tokens"].extend(n_answer.tolist())

            except Exception as e:
                # A failed batch is recorded as skipped, NOT written as zero vectors: a
                # zero row looks like a real embedding to every downstream consumer.
                print(f"  WARNING: batch at {i} failed, skipping {len(ids)}: {e}",
                      flush=True)
                skipped.extend({"id": cid, "reason": f"forward failed: {type(e).__name__}"}
                               for cid in ids)

        completed += len(batch)
        batch_num = i // args.batch_size + 1

        if batch_num % args.checkpoint_every == 0 or completed == total:
            flush(out_dirs, layers, pending, checkpoint_num, skipped, truncated,
                  skip_path, trunc_path)

        elapsed = time.time() - start_time
        rate = completed / elapsed * 3600
        remaining = (total - completed) / (completed / elapsed) if completed else 0
        print(f"  {completed:,} / {total:,}  ({rate:,.0f}/hr, ~{remaining/3600:.1f}h left)", flush=True)

    # Unconditional: the in-loop call is conditional on batch_num/completed and has
    # already been shown to miss the tail. Cheap when there is nothing buffered.
    flush(out_dirs, layers, pending, checkpoint_num, skipped, truncated,
          skip_path, trunc_path)

    # Every structure must end up either embedded or logged as skipped. Silently losing
    # some is exactly the failure this run hit, and it was only caught by counting rows
    # afterwards -- so count them here instead.
    #
    # Compare against the WHOLE corpus, not `total`. `total` is only what this run had
    # left to do, while the counts below span every run: the shards hold earlier runs'
    # embeddings too, and skipped ids are never written to a shard, so a resume always
    # re-processes and re-logs them and skipped.csv ends up complete either way. Using
    # `total` here made a correct resume report a negative shortfall.
    n_done = sum(len(pd.read_parquet(f, columns=["id"]))
                 for f in out_dirs[layers[0]].glob("checkpoint_*.parquet"))
    n_seen = n_done + len(skipped)
    if n_seen != n_corpus:
        print(f"  WARNING: {n_corpus:,} structures in the corpus, but {n_done:,} embedded "
              f"+ {len(skipped):,} skipped = {n_seen:,}. "
              f"{n_corpus - n_seen:,} unaccounted for.", flush=True)
    else:
        print(f"  accounted for: {n_done:,} embedded + {len(skipped):,} skipped "
              f"= {n_corpus:,} (the whole corpus)")

    for h in hooks:
        h.remove()
    print(f"\nDone. Outputs in embeddings/{dataset}/{args.model}/{args.variant}/"
          f"cif_layer{{N}}/ for layers {layers}")


if __name__ == "__main__":
    main()
