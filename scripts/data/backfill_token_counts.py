#!/usr/bin/env python3
"""Add `n_new_tokens` to llamat generations that predate steer_generate_cif recording it.

WHY THIS IS EXACT, not a proxy. A generation that used the whole --max-new-tokens budget
was cut off mid-structure, and the CIF cannot show it: the file is internally consistent,
it just describes a smaller crystal than the model was writing. Token count is the only
direct evidence, and it survives in `raw_output` because decoding is deterministic.

Re-tokenising recovers the count exactly. `backend.decode(..., skip_special_tokens=True)`
strips the EOS a natural stop emits, so a finished generation re-tokenises to fewer
tokens than the cap while a truncated one lands exactly ON it. Measured over the 3,000
control generations at a 600-token cap: 245 sit at exactly 600, NOTHING sits at 599, and
nothing exceeds 600. That gap is what makes the test unambiguous. Guessing from
`raw_output` length instead found only 197 of those 245.

Needs the tokenizer only -- no model, no GPU -- but it lives in transformers, so run this
in llamat_venv. Validation itself runs in crystallm_venv and only reads the column.

Usage:
    ./llamat_venv/bin/python scripts/data/backfill_token_counts.py \
        steering_results/llamat2_cif/*/generated_cifs/*.parquet
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ckpt-dir", default="models/llamat2_cif",
                    help="only the tokenizer is loaded from here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=True)

    for path in args.files:
        df = pd.read_parquet(path)
        name = os.path.basename(path)
        if "raw_output" not in df.columns:
            print(f"{name:<52} skipped -- no raw_output")
            continue
        if "n_new_tokens" in df.columns:
            print(f"{name:<52} skipped -- already has n_new_tokens")
            continue

        n = [len(tok(t or "", add_special_tokens=False)["input_ids"])
             for t in df["raw_output"]]
        df["n_new_tokens"] = n
        top = max(n)
        at_cap = sum(1 for x in n if x == top)
        if not args.dry_run:
            df.to_parquet(path, index=False)
        print(f"{name:<52} {len(df):>6,} rows  max {top}  "
              f"{at_cap:>5,} at max ({at_cap/len(df):.1%})"
              + ("  [dry run]" if args.dry_run else ""))


if __name__ == "__main__":
    main()
