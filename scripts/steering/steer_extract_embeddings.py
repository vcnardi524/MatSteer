#!/usr/bin/env python3
"""Capture hidden states at layers 7-15 while steering at one layer, to see how far the
injection travels.

Steering is injected once, at a single layer. Nothing else in this repo measures what the
blocks AFTER that layer do with the perturbation -- whether the model amplifies it, damps
it, or absorbs it. That is the mechanism question behind every steering result here.

THIS SCRIPT WRITES NO CIFs. The generated structures are held in memory only long enough
to score their validity, then discarded. There is no validation parquet, no relaxation and
no property prediction for these runs; they are a measurement, not another sweep arm.

HOOK ORDER MATTERS
------------------
The capture hook on the steered layer is registered AFTER the steering hook. PyTorch feeds
each forward hook the output as modified by the previous one, so in that order the capture
sees the POST-injection state. Registered the other way round it would silently record the
pre-injection state and the steered layer's distance from baseline would come out ~0.
`--check-hook-order` verifies this on one prompt and exits.

HOW A SEQUENCE IS ASSEMBLED
---------------------------
With the KV cache the hook fires once at prefill with T = prompt length
(CrystaLLM/crystallm/_model.py:382) and once per decode step with T = 1 (_model.py:405).
Concatenating the captured chunks along T therefore reconstructs the whole prompt +
generated sequence, which is what extract_cif_embeddings.py:136 pools over for the corpus
embeddings. Taking the mean over T gives one vector per layer per sample, on the same
footing as those.

The captured count is ONE SHORT of the returned sequence, and that is correct: the decode
loop samples a token, appends it, then breaks on the double-newline end marker BEFORE
forwarding it (_model.py:396-405). So the final token -- the terminating newline -- never
gets a forward pass and has no hidden state to capture. A gap larger than one means
something was actually dropped.

WHY VALIDITY IS RECORDED
------------------------
Measured 2026-09-09: across 106 density arms, corr(validity, median density) = -0.91.
Degradation alone moves the property being steered, so an effect measured without a
validity flag cannot be told apart from generation falling apart. The same trap applies
one level down here -- a propagation curve computed over broken generations measures
breakage. The flags cost two columns and no CIF storage.

Output: <results-dir>/embeddings/<stem>.parquet
  id, sample, layer, embedding, is_valid, is_sensible

Usage:
    python scripts/steering/steer_extract_embeddings.py \
        --model CrystaLLM/crystallm_v1_large --pkl CrystaLLM/cifs_v1_test_sample1000.pkl.gz \
        --method manifold --steering-property density_atomic \
        --manifold steering_vectors/manifolds/density_atomic_layer7_k64_w1_max40.parquet \
        --layer 7 --k 64 --delta 2 --variant residual --scale 4 \
        --results-dir steering_results/density_atomic
"""
import argparse
import importlib.util as _ilu
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # neighbours (compute_pca_basis, ...)
sys.path.insert(0, str(HERE.parent))               # scripts/ -> utils.py
sys.path.insert(0, str(HERE.parent / "embeddings"))
sys.path.insert(0, str(HERE.parent / "eval"))


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The steering hooks are IMPORTED, never reimplemented, so the injection here is bit for
# bit the one the sweep used.
_sg = _load("steer_generate_cif", HERE / "steer_generate_cif.py")
# Imported normally, NOT via _load: multiprocessing pickles eval_one by module
# name, and a spec_from_file_location module cannot be re-imported in the child.
import validate_steered_cifs as _val

CHECKPOINT_EVERY = 50          # prompts between validation+write passes


def capture_hooks(model, layers, buffer):
    """Read-only hooks appending each forward's hidden states. Returns the handles.

    Must be registered AFTER any steering hook on a shared layer -- see module docstring.
    """
    handles = []
    for l in layers:
        def make(layer_idx):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                buffer[layer_idx].append(h.detach()[0].float().cpu())
                return None                        # read-only: never modify the output
            return fn
        handles.append(model.transformer.h[l].register_forward_hook(make(l)))
    return handles


def pooled(buffer, layers):
    """{layer: (1024,) mean over the full sequence}, and the token count seen."""
    out, n_tok = {}, None
    for l in layers:
        seq = torch.cat(buffer[l], dim=0)          # (T_total, D)
        out[l] = seq.mean(0).numpy()
        n_tok = seq.shape[0] if n_tok is None else n_tok
    return out, n_tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pkl", required=True)
    p.add_argument("--method", choices=("linear", "manifold"), default="manifold")
    p.add_argument("--manifold", default=None)
    p.add_argument("--variant", choices=("residual", "project", "project_nomu"),
                   default="residual")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--layer", type=int, default=7, help="Layer the injection is applied at")
    p.add_argument("--capture-layers", default="7,8,9,10,11,12,13,14,15")
    p.add_argument("--steering-property", default="density_atomic")
    p.add_argument("--n-prompts", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--with-spacegroup", action="store_true")
    p.add_argument("--results-dir", default="steering_results")
    p.add_argument("--out", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--check-hook-order", action="store_true",
                   help="Verify capture-after-steering on one prompt, then exit")
    args = p.parse_args()

    np.random.seed(_sg.RANDOM_SEED)
    torch.manual_seed(_sg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = _sg.load_model(args.model, device)
    tokenizer = _sg.CIFTokenizer()
    layers = [int(x) for x in args.capture_layers.split(",")]
    assert all(0 <= l < config.n_layer for l in layers), f"layers outside 0..{config.n_layer-1}"

    if args.method == "linear":
        hook, run_tag = _sg.build_linear(args, device)
    else:
        hook, run_tag = _sg.build_manifold(args, device)
    print(f"KV cache=on  dropout={config.dropout}  capturing layers {layers}")

    # ---- output path: same stem convention as the generation runs ----
    split = next((s for s in ("train", "test", "val") if s in Path(args.pkl).stem),
                 Path(args.pkl).stem)
    sg_tag = "" if args.with_spacegroup else "_nosg"
    prefix = "steered" if args.method == "linear" else "steered_manifold"
    stem = f"{prefix}_{split}_{run_tag}_layer{args.layer}{sg_tag}"
    out_dir = Path(args.results_dir) / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{stem}.parquet"
    print(f"Writing {out_path}")

    data = _sg.load_cifs(args.pkl)
    pattern = _sg.PATTERN_COMP_SG if args.with_spacegroup else _sg.PATTERN_COMP
    prompts = []
    for id_, cif in data:
        try:
            prompts.append((id_, _sg.extract_prompt(cif, pattern)))
        except Exception:
            pass
        if args.n_prompts and len(prompts) >= args.n_prompts:
            break
    print(f"Extracted {len(prompts)} prompts")

    # ---- registration order is the whole correctness argument: steering first ----
    steer_handle = model.transformer.h[args.layer].register_forward_hook(hook)
    buffer = {l: [] for l in layers}
    cap_handles = capture_hooks(model, layers, buffer)

    if args.check_hook_order:
        id_, prompt = prompts[0]
        x = torch.tensor(tokenizer.encode(tokenizer.tokenize_cif(prompt)),
                         dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            model(x)
        steered = torch.cat(buffer[args.layer], 0).clone()
        for h in cap_handles + [steer_handle]:
            h.remove()
        buffer = {l: [] for l in layers}
        cap_handles = capture_hooks(model, layers, buffer)     # no steering hook now
        with torch.no_grad():
            model(x)
        clean = torch.cat(buffer[args.layer], 0)
        gap = (steered - clean).norm(dim=-1).mean().item()
        rel = gap / clean.norm(dim=-1).mean().item()
        print(f"\nHOOK ORDER CHECK on layer {args.layer}, prompt {id_}")
        print(f"  mean |captured_with_steering - captured_without| = {gap:.4f}  ({rel:.2%} of |h|)")
        print("  PASS: capture sees the post-injection state" if gap > 1e-6 else
              "  FAIL: capture is running BEFORE the steering hook -- fix the order")
        return

    done = set()
    if out_path.exists():
        prev = pd.read_parquet(out_path, columns=["id", "sample"])
        done = set(zip(prev["id"], prev["sample"]))
        print(f"Resuming — {len(done):,} (id, sample) rows already done")

    pending_rows, pending_cifs = [], []

    def flush():
        """Validate this chunk's CIFs, attach the flags, append, and drop the CIFs."""
        if not pending_rows:
            return
        with Pool(args.workers) as pool:
            res = pool.map(_val.eval_one, list(enumerate(pending_cifs)))
        flags = {r["idx"]: r for r in res}
        rows = []
        for i, rec in enumerate(pending_rows):
            f = flags[i]
            for l, emb in rec["emb"].items():
                rows.append({"id": rec["id"], "sample": rec["sample"], "layer": l,
                             "embedding": emb,
                             "is_valid": bool(f["is_valid"]),
                             "is_sensible": bool(f["is_sensible"])})
        df = pd.DataFrame(rows)
        if out_path.exists():
            df = pd.concat([pd.read_parquet(out_path), df], ignore_index=True)
        df.to_parquet(out_path, index=False)
        print(f"  wrote {len(df):,} rows total "
              f"({sum(r['is_valid'] for r in flags.values())}/{len(flags)} valid this chunk)",
              flush=True)
        pending_rows.clear()
        pending_cifs.clear()

    for i, (id_, prompt) in enumerate(prompts):
        if all((id_, j + 1) in done for j in range(args.n_samples)):
            continue
        print(f"[{i+1}/{len(prompts)}] {id_}", flush=True)
        x = torch.tensor(tokenizer.encode(tokenizer.tokenize_cif(prompt)),
                         dtype=torch.long, device=device).unsqueeze(0)
        for j in range(args.n_samples):
            if (id_, j + 1) in done:
                continue
            for l in layers:
                buffer[l].clear()
            with torch.no_grad():
                y = model.generate_cached(x, args.max_new_tokens,
                                          temperature=args.temperature, top_k=args.top_k)
            emb, n_tok = pooled(buffer, layers)
            if i == 0 and j == 0:
                print(f"  captured {n_tok} token states vs {y.shape[1]} generated tokens")
            pending_rows.append({"id": id_, "sample": j + 1, "emb": emb})
            pending_cifs.append(tokenizer.decode(y[0].tolist()))
        if len(pending_rows) >= CHECKPOINT_EVERY * args.n_samples:
            flush()

    flush()
    for h in cap_handles + [steer_handle]:
        h.remove()
    final = pd.read_parquet(out_path, columns=["id", "sample", "layer", "is_valid"])
    print(f"\nSaved {out_path}")
    print(f"  {final['id'].nunique():,} prompts, {len(final):,} rows, "
          f"{final.layer.nunique()} layers, "
          f"{100*final.drop_duplicates(['id','sample']).is_valid.mean():.1f}% valid")


if __name__ == "__main__":
    main()
