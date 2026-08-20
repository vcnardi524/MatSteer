#!/usr/bin/env python3
"""
Equivalence test for the KV-cached generation path.

Verifies, with the steering hook active, that:
  (1) per-position logits from the cached forward match a single full (uncached)
      forward over the same sequence   -> validates the cache math directly, and
  (2) greedy generation (top_k=1) produces identical token sequences with the
      cache on vs off                  -> validates the end-to-end decode path.

Also reports wall-clock speedup. Run on GPU for a realistic timing number; CPU
works for correctness (just slower). Exit code is non-zero if any check fails.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM", "bin"))

from crystallm import CIFTokenizer
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "embeddings"))   # -> extract_cif_embeddings.py
from make_prompts import PATTERN_COMP_SG, extract_prompt
from extract_cif_embeddings import load_model, load_cifs

LAYER = 14
ALPHA = 16.0


def build_hook(vec):
    def _steer_hook(module, inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec,) + out[1:]
        return out + vec
    return _steer_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CrystaLLM/crystallm_v1_large")
    ap.add_argument("--pkl", default="CrystaLLM/cifs_v1_test.pkl.gz")
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, _ = load_model(args.model, device)
    model.eval()
    tokenizer = CIFTokenizer()

    sv = pd.read_parquet(f"steering_vectors/bandgap_layer{LAYER}.parquet").iloc[0]
    steer_vec = np.array(sv["steering_vector"], dtype=np.float32)
    vec = torch.tensor(steer_vec * ALPHA, dtype=torch.float32, device=device).view(1, 1, -1)

    data = load_cifs(args.pkl)
    prompts = []
    for id_, cif in data:
        try:
            prompts.append((id_, extract_prompt(cif, PATTERN_COMP_SG)))
        except Exception:
            pass
        if len(prompts) >= args.n_prompts:
            break

    all_ok = True
    t_uncached = t_cached = 0.0

    for id_, prompt in prompts:
        tokens = tokenizer.encode(tokenizer.tokenize_cif(prompt))
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        handle = model.transformer.h[LAYER].register_forward_hook(build_hook(vec))
        try:
            # ---- greedy generation, cache off vs on (identical seed) ----
            torch.manual_seed(42)
            t0 = time.time()
            y_ref = model.generate(x, args.max_new_tokens, temperature=1.0, top_k=1)
            t_uncached += time.time() - t0

            torch.manual_seed(42)
            t0 = time.time()
            y_cache = model.generate_cached(x, args.max_new_tokens, temperature=1.0, top_k=1)
            t_cached += time.time() - t0

            seq_ref = y_ref[0].tolist()
            seq_cache = y_cache[0].tolist()
            seqs_match = seq_ref == seq_cache

            # ---- per-position logits: one full forward vs step-by-step cached ----
            seq = y_ref  # (1, T) full generated sequence
            T = seq.size(1)
            dummy = torch.zeros_like(seq)
            logits_full, _ = model(seq, targets=dummy)          # (1, T, vocab), all positions

            past = None
            cached_cols = []
            for i in range(T):
                lg, _, past = model(seq[:, i:i + 1], use_cache=True, past_kvs=past)
                cached_cols.append(lg[:, -1, :])
            logits_cached = torch.stack(cached_cols, dim=1)      # (1, T, vocab)

            max_diff = (logits_full - logits_cached).abs().max().item()
            logits_ok = max_diff < args.atol
        finally:
            handle.remove()

        ok = seqs_match and logits_ok
        all_ok &= ok
        print(f"[{ 'OK ' if ok else 'FAIL' }] {id_}  "
              f"greedy_match={seqs_match} (len {len(seq_ref)})  "
              f"max_logit_diff={max_diff:.2e} (<{args.atol:g})")

    print(f"\nTiming over {len(prompts)} prompts (greedy, {args.max_new_tokens} max tokens):")
    print(f"  uncached: {t_uncached:.1f}s   cached: {t_cached:.1f}s   "
          f"speedup: {t_uncached / max(t_cached, 1e-9):.2f}x")
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
