#!/usr/bin/env python3
"""
Steered CIF generation using CrystaLLM + band gap steering vectors.

Injection pattern mirrors ChemSteer (inject_timestep='all'):
  - Forward hook on model.transformer.h[layer]
  - Adds alpha * steering_vector to every token position at every generation step

Prompting follows CrystaLLM (composition + space group header from CIF).
Outputs a single parquet file with columns: id, sample, cif_steered, cif_original.

Usage:
    python steer_generate_cif.py \
        --model CrystaLLM/crystallm_v1_large \
        --pkl CrystaLLM/cifs_v1_test.pkl.gz \
        --percentile 10 \
        --alpha 3.0 \
        --layer 14 \
        --n-samples 3 \
        --with-spacegroup \
        --out generated_cifs/
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CrystaLLM"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CrystaLLM", "bin"))

from crystallm import CIFTokenizer
from make_prompts import PATTERN_COMP, PATTERN_COMP_SG, extract_prompt
from extract_cif_embeddings import load_model, load_cifs

RANDOM_SEED = 42
CHECKPOINT_EVERY = 100  # write to parquet every N prompts


def generate(model, tokenizer, device, prompt_str, max_new_tokens, temperature, top_k,
             steer_vec=None, layer=None, alpha=1.0, use_cache=False):
    handle = None
    if steer_vec is not None:
        vec = torch.tensor(steer_vec * alpha, dtype=torch.float32, device=device).view(1, 1, -1)

        def _steer_hook(module, inp, out):
            # With the KV cache, Block.forward returns (hidden, present_kv); add the
            # steering vector to the hidden state and pass the cache through untouched.
            if isinstance(out, tuple):
                return (out[0] + vec,) + out[1:]
            return out + vec

        handle = model.transformer.h[layer].register_forward_hook(_steer_hook)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Steering strength (positive = towards high band gap)")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--n-prompts", type=int, default=0,
                        help="Number of prompts to use (0 = all)")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--with-spacegroup", action="store_true",
                        help="Include space group in prompt (recommended)")
    parser.add_argument("--steering-file", default=None,
                        help="Path to a single-row clean steering-vector parquet. "
                             "If set, overrides the percentile-based vector.")
    parser.add_argument("--out", default="generated_cifs")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use KV-cached decoding (generate_cached). "
                             "Verified byte-identical to uncached; ~1.9x faster at batch 1.")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = load_model(args.model, device)
    tokenizer = CIFTokenizer()

    sv_df = pd.read_parquet(f"steering_vectors/bandgap_layer{args.layer}.parquet")
    row = sv_df.iloc[0]   # single clean metal-vs-insulator vector (no percentiles)
    steer_vec = np.array(row["steering_vector"], dtype=np.float32)
    print(f"Clean steering vector: metals<={row['low_thresh_ev']}eV (n={int(row['n_low']):,}) vs "
          f"insulators>={row['high_thresh_ev']}eV (n={int(row['n_high']):,})  "
          f"raw_norm={row['raw_norm']:.2f}")
    print(f"Alpha={args.alpha}  Layer={args.layer}")

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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sg_tag = "" if args.with_spacegroup else "_nosg"
    out_path = out_dir / f"steered_{split}_clean_alpha{args.alpha}_layer{args.layer}{sg_tag}.parquet"

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

        for j in range(args.n_samples):
            steered = generate(model, tokenizer, device, prompt,
                               args.max_new_tokens, args.temperature, args.top_k,
                               steer_vec=steer_vec, layer=args.layer, alpha=args.alpha,
                               use_cache=args.use_cache)
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
