#!/usr/bin/env python3
"""
Steered CIF generation using CrystaLLM.

Two steering methods share the same forward hook on model.transformer.h[layer], applied
to every token position at every generation step (ChemSteer's inject_timestep='all'):

  linear        h <- h + alpha * v
                v is the normalized high-class minus low-class mean difference from
                compute_steering_vector.py. Assumes the property is a straight line in
                the residual stream.

  pca_centroid  z <- (h - mu) @ W.T ;  h <- h + t * (centroid_pca - z) @ W
                Project into the top-K PCA subspace of the training activations, move
                the coordinates a fraction t of the way to a target centroid, and map
                the change back. There is no low class: the centroid alone sets the
                destination, and everything outside the K principal directions passes
                through untouched, so the residual stream keeps supplying the context
                the centroid does not specify. t=0 is no steering, t=1 snaps the
                subspace coordinates onto the centroid.
                Needs compute_pca_basis.py and compute_centroid_target.py.

Prompting follows CrystaLLM (composition + space group header from CIF).
Outputs a single parquet with columns: id, sample, cif_steered. Both methods write into
the same directory; the filename carries the method so the two never collide.

Usage:
    python steer_generate_cif.py --model CrystaLLM/crystallm_v1_large \
        --pkl CrystaLLM/cifs_v1_test.pkl.gz --alpha 40 --layer 14 --n-samples 3 \
        --with-spacegroup --steering-property density_atomic

    python steer_generate_cif.py --model CrystaLLM/crystallm_v1_large \
        --pkl CrystaLLM/cifs_v1_test.pkl.gz --method pca_centroid \
        --target 30 --t 0.5 --k 64 --layer 14 --n-samples 3 --with-spacegroup
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM", "bin"))

from crystallm import CIFTokenizer
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "embeddings"))   # -> extract_cif_embeddings.py
from make_prompts import PATTERN_COMP, PATTERN_COMP_SG, extract_prompt
from extract_cif_embeddings import load_model, load_cifs
# sys.path[0] is this script's own dir, so its neighbour imports directly.
from compute_pca_basis import METHOD_DIR as PCA_DIR
from compute_centroid_target import load_pca

RANDOM_SEED = 42
CHECKPOINT_EVERY = 100  # write to parquet every N prompts


def rewrap(out, hidden):
    """Put a modified hidden state back into whatever Block.forward returned.

    With the KV cache it returns (hidden, present_kv); the cache passes through
    untouched.
    """
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden


def linear_hook(steer_vec, alpha, device):
    vec = torch.tensor(steer_vec * alpha, dtype=torch.float32, device=device).view(1, 1, -1)

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        return rewrap(out, h + vec.to(h.dtype))

    return hook


def pca_centroid_hook(mean, components, centroid_pca, t, device):
    mu = torch.tensor(mean, dtype=torch.float32, device=device)            # (1024,)
    W = torch.tensor(components, dtype=torch.float32, device=device)       # (k, 1024)
    c = torch.tensor(centroid_pca, dtype=torch.float32, device=device)     # (k,)

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        z = (h.float() - mu) @ W.T                    # (B, T, k) subspace coordinates
        return rewrap(out, h + (t * (c - z) @ W).to(h.dtype))

    return hook


def prompt_embedding(model, tokenizer, device, prompt_str, layer):
    """Mean-pooled layer-L hidden state of the prompt, matching how the training
    embeddings were built -- except those pooled a whole CIF and this pools a header,
    which is the only view available before generation starts."""
    captured = {}

    def grab(module, inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = model.transformer.h[layer].register_forward_hook(grab)
    tokens = tokenizer.encode(tokenizer.tokenize_cif(prompt_str))
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        model(x)
    handle.remove()
    return captured["h"][0].float().mean(0)          # (1024,)


def local_centroid(z_prompt, bank, n_neighbours):
    """Centroid of the n_neighbours class members closest to this prompt.

    The global centroid averages the whole class, so every prompt is pulled toward the
    same point no matter where it starts. If the class does not sit in one place, that
    average is a point between the pieces rather than inside any of them.
    """
    d = torch.cdist(z_prompt.view(1, -1), bank).view(-1)
    idx = torch.topk(d, min(n_neighbours, bank.shape[0]), largest=False).indices
    return bank[idx].mean(0), float(d[idx].mean())


def generate(model, tokenizer, device, prompt_str, max_new_tokens, temperature, top_k,
             hook=None, layer=None, use_cache=False):
    handle = model.transformer.h[layer].register_forward_hook(hook) if hook else None

    tokens = tokenizer.encode(tokenizer.tokenize_cif(prompt_str))
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        if use_cache:
            y = model.generate_cached(x, max_new_tokens, temperature=temperature, top_k=top_k)
        else:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    if handle:
        handle.remove()

    return tokenizer.decode(y[0].tolist())


def build_linear(args, device):
    """(hook, filename stem suffix) for the mean-difference method."""
    sv_path = Path("steering_vectors") / args.steering_property / f"layer{args.layer}.parquet"
    if not sv_path.exists():
        # legacy flat location (pre per-property dirs)
        legacy = Path("steering_vectors") / f"{args.steering_property}_layer{args.layer}.parquet"
        if not legacy.exists():
            raise FileNotFoundError(f"No steering vector at {sv_path} or {legacy}")
        sv_path = legacy
    row = pd.read_parquet(sv_path).iloc[0]   # single clean low-vs-high vector
    steer_vec = np.array(row["steering_vector"], dtype=np.float32)
    lo = row.get("low_thresh", row.get("low_thresh_ev"))   # new / legacy column names
    hi = row.get("high_thresh", row.get("high_thresh_ev"))
    print(f"Steering vector [{args.steering_property}] {sv_path}: low<={lo} "
          f"(n={int(row['n_low']):,}) vs high>={hi} (n={int(row['n_high']):,})  "
          f"raw_norm={row['raw_norm']:.2f}")
    print(f"Method=linear  alpha={args.alpha}  layer={args.layer}")
    return linear_hook(steer_vec, args.alpha, device), f"alpha{args.alpha}"


def build_pca_centroid(args, device):
    """(hook, filename stem suffix) for the PCA-subspace centroid method."""
    if args.target is None:
        raise SystemExit("--method pca_centroid needs --target")
    mean, comps = load_pca(args.layer, args.k)
    cen_path = (PCA_DIR / args.steering_property /
                f"layer{args.layer}_k{args.k}_target{args.target:g}.parquet")
    if not cen_path.exists():
        raise FileNotFoundError(
            f"No centroid at {cen_path} — run compute_centroid_target.py --target "
            f"{args.target:g} --property {args.steering_property}")
    row = pd.read_parquet(cen_path).iloc[0]
    centroid_pca = np.asarray(row["centroid_pca"], dtype=np.float32)
    print(f"Centroid [{args.steering_property}] {cen_path}: target={row['target']:g}, "
          f"class n={int(row['class_size']):,} over [{row['class_lo']:.3f}, "
          f"{row['class_hi']:.3f}] (mean {row['class_mean']:.3f})")
    print(f"Method=pca_centroid  t={args.t}  k={args.k}  layer={args.layer}")
    return (pca_centroid_hook(mean, comps, centroid_pca, args.t, device),
            f"target{args.target:g}_t{args.t}_k{args.k}")


def build_pca_local(args, device):
    """(mean, components, bank) for the per-prompt local-centroid method."""
    if args.target is None:
        raise SystemExit("--method pca_local needs --target")
    mean, comps = load_pca(args.layer, args.k)
    stem = f"layer{args.layer}_k{args.k}_target{args.target:g}"
    bank_path = PCA_DIR / args.steering_property / f"{stem}_bank.parquet"
    if not bank_path.exists():
        raise FileNotFoundError(
            f"No class bank at {bank_path} -- rerun compute_centroid_target.py with "
            f"--save-bank")
    Z = np.vstack(pd.read_parquet(bank_path)["coord"].to_numpy()).astype(np.float32)
    print(f"Class bank [{args.steering_property}] {bank_path}: {Z.shape[0]:,} members "
          f"x {Z.shape[1]} dims")
    print(f"Method=pca_local  t={args.t}  k={args.k}  neighbours={args.neighbours}  "
          f"layer={args.layer}")
    return (torch.tensor(mean, device=device), torch.tensor(comps, device=device),
            torch.tensor(Z, device=device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--method", choices=("linear", "pca_centroid", "pca_local"),
                        default="linear",
                        help="linear: alpha * mean-difference vector. "
                             "pca_centroid: interpolate toward one global target centroid "
                             "inside the top-K PCA subspace. "
                             "pca_local: same, but the centroid is recomputed per prompt "
                             "from the class members nearest that prompt.")
    parser.add_argument("--neighbours", type=int, default=256,
                        help="[pca_local] class members averaged into each prompt's "
                             "local centroid")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="[linear] steering strength (positive = towards the high class)")
    parser.add_argument("--target", type=float, default=None,
                        help="[pca_centroid] target property value; picks the centroid file")
    parser.add_argument("--t", type=float, default=0.5,
                        help="[pca_centroid] interpolation fraction toward the centroid, "
                             "0 = none, 1 = snap onto it")
    parser.add_argument("--k", type=int, default=64,
                        help="[pca_centroid] size of the PCA subspace")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--steering-property", default="bandgap",
                        help="Property subdir under steering_vectors/ (linear) or "
                             "steering_vectors/pca_centroid/ (pca_centroid)")
    parser.add_argument("--n-prompts", type=int, default=0,
                        help="Number of prompts to use (0 = all)")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--with-spacegroup", action="store_true",
                        help="Include space group in prompt (recommended)")
    parser.add_argument("--results-dir", default="steering_results",
                        help="Base results dir; output goes to <results-dir>/generated_cifs "
                             "unless --out is given explicitly")
    parser.add_argument("--out", default=None,
                        help="Explicit output dir (overrides --results-dir/generated_cifs)")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Use KV-cached decoding (generate_cached). On by default; "
                             "verified byte-identical to uncached, ~1.9x faster at batch 1. "
                             "Pass --no-use-cache to fall back to the uncached path.")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = load_model(args.model, device)
    tokenizer = CIFTokenizer()

    local = None
    if args.method == "linear":
        hook, run_tag = build_linear(args, device)
    elif args.method == "pca_centroid":
        hook, run_tag = build_pca_centroid(args, device)
    else:
        local = build_pca_local(args, device)
        hook = None                      # rebuilt per prompt, once its centroid is known
        run_tag = f"target{args.target:g}_t{args.t}_k{args.k}_nb{args.neighbours}"
    print(f"KV cache={'on' if args.use_cache else 'off'}  dropout={config.dropout}")

    data = load_cifs(args.pkl)

    pattern = PATTERN_COMP_SG if args.with_spacegroup else PATTERN_COMP
    prompts = []
    for id_, cif in data:
        try:
            p = extract_prompt(cif, pattern)
            prompts.append((id_, p))
        except Exception:
            pass
        if args.n_prompts and len(prompts) >= args.n_prompts:
            break
    print(f"Extracted {len(prompts)} prompts")

    # infer split name from pkl filename (train/test/val)
    pkl_stem = Path(args.pkl).stem  # e.g. cifs_v1_test
    split = next((s for s in ("train", "test", "val") if s in pkl_stem), pkl_stem)

    # The property is encoded by the output directory (per-property <results-dir>), so
    # the filename carries method/split/strength/layer. The method prefix keeps the two
    # methods' runs distinguishable in a shared directory and by stem downstream.
    out_dir = Path(args.out) if args.out else Path(args.results_dir) / "generated_cifs"
    out_dir.mkdir(parents=True, exist_ok=True)
    sg_tag = "" if args.with_spacegroup else "_nosg"
    prefix = {"linear": "steered", "pca_centroid": "steered_pca",
              "pca_local": "steered_pcalocal"}[args.method]
    out_path = out_dir / f"{prefix}_{split}_{run_tag}_layer{args.layer}{sg_tag}.parquet"

    # resume: skip already-done ids
    done_ids = set()
    if out_path.exists():
        done_ids = set(pd.read_parquet(out_path, columns=["id"])["id"].tolist())
        print(f"Resuming — {len(done_ids):,} ids already done")

    pending = []

    for i, (id_, prompt) in enumerate(prompts):
        if id_ in done_ids:
            continue

        print(f"[{i+1}/{len(prompts)}] {id_}", flush=True)

        if local is not None:
            # This prompt's own neighbourhood of the class, not the class average.
            mean_t, comps_t, bank = local
            z = (prompt_embedding(model, tokenizer, device, prompt, args.layer)
                 - mean_t) @ comps_t.T
            c_local, dist = local_centroid(z, bank, args.neighbours)
            hook = pca_centroid_hook(mean_t.cpu().numpy(), comps_t.cpu().numpy(),
                                     c_local.cpu().numpy(), args.t, device)
            if i % 100 == 0:
                print(f"    local centroid {dist:.2f} from prompt in subspace", flush=True)

        for j in range(args.n_samples):
            steered = generate(model, tokenizer, device, prompt,
                               args.max_new_tokens, args.temperature, args.top_k,
                               hook=hook, layer=args.layer, use_cache=args.use_cache)
            pending.append({
                "id":          id_,
                "sample":      j + 1,
                "cif_steered": steered,
            })

        if len(pending) >= CHECKPOINT_EVERY * args.n_samples:
            chunk = pd.DataFrame(pending)
            if out_path.exists():
                chunk = pd.concat([pd.read_parquet(out_path), chunk], ignore_index=True)
            chunk.to_parquet(out_path, index=False)
            pending = []
            print(f"  Checkpointed at prompt {i+1}", flush=True)

    if pending:
        chunk = pd.DataFrame(pending)
        if out_path.exists():
            chunk = pd.concat([pd.read_parquet(out_path), chunk], ignore_index=True)
        chunk.to_parquet(out_path, index=False)

    total = len(pd.read_parquet(out_path))
    print(f"\nDone. {total:,} rows saved to {out_path}")


if __name__ == "__main__":
    main()
