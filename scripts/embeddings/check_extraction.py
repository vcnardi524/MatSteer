#!/usr/bin/env python3
"""Exit non-zero if an embedding run's output is not usable.

WHY THIS EXISTS. SLURM's afterok only tests the exit code, and the failure modes that
matter here exit cleanly: a hook on the wrong module, an answer span off by a few
tokens, fp16 pooling going non-finite on long sequences. Any of those produce a full
set of embeddings that look fine and are quietly wrong. Chaining a long run behind a
short one with afterok alone therefore gates on almost nothing. This script is the
gate, so `smoke -> check -> full` only proceeds if the smoke output is actually sane.

Usage:
    python scripts/embeddings/check_extraction.py \
        --dataset v1_mp --model llamat2_cif --variant crystal_uncond \
        --layers 0,2,4,...,30 --min-rows 90 --expect-dim 4096
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import DEFAULT_MODEL, MODELS, VARIANTS, embedding_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v1_mp")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--variant", default="crystal_uncond", choices=list(VARIANTS))
    ap.add_argument("--layers", required=True, help="comma-separated, all must be present")
    ap.add_argument("--min-rows", type=int, default=90)
    ap.add_argument("--expect-dim", type=int, default=4096)
    ap.add_argument("--min-answer-tokens", type=float, default=20,
                    help="median n_answer_tokens below this means the span is wrong")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    failures, first = [], None

    for layer in layers:
        try:
            files = embedding_files(layer, args.dataset, args.variant, args.model)
        except FileNotFoundError as e:
            failures.append(f"layer {layer}: {e}")
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        X = np.vstack(df["embedding"].to_numpy()).astype(np.float32)

        if len(df) < args.min_rows:
            failures.append(f"layer {layer}: {len(df)} rows < {args.min_rows}")
        if X.shape[1] != args.expect_dim:
            failures.append(f"layer {layer}: dim {X.shape[1]} != {args.expect_dim}")
        if not np.isfinite(X).all():
            failures.append(f"layer {layer}: {(~np.isfinite(X)).sum():,} non-finite values")
        n_zero = int((X == 0).all(axis=1).sum())
        if n_zero:
            failures.append(f"layer {layer}: {n_zero} all-zero rows")
        if not df["id"].is_unique:
            failures.append(f"layer {layer}: duplicate ids")
        if "n_answer_tokens" in df:
            med = float(df["n_answer_tokens"].median())
            if med < args.min_answer_tokens:
                failures.append(f"layer {layer}: median n_answer_tokens {med:.0f} "
                                f"< {args.min_answer_tokens}")

        # Two different structures must not give the same vector. If the pooled span
        # were the constant prompt instead of the answer, every row would be identical.
        if len(X) >= 2:
            spread = float(np.abs(X[0] - X[1]).max())
            if spread == 0.0:
                failures.append(f"layer {layer}: rows 0 and 1 are IDENTICAL -- the "
                                f"pooled span is probably the prompt, not the answer")
        if first is None:
            first = (len(df), X.shape[1],
                     float(df.get("n_answer_tokens", pd.Series([np.nan])).median()),
                     float(np.linalg.norm(X, axis=1).mean()))

    if first:
        print(f"{len(layers)} layers | {first[0]:,} rows | dim {first[1]} | "
              f"median n_answer_tokens {first[2]:.0f} | mean |v| {first[3]:.1f}")
    for f in failures:
        print(f"  FAIL  {f}")
    if failures:
        print(f"\n{len(failures)} check(s) failed -- NOT safe to chain the full run")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
