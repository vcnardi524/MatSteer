#!/usr/bin/env python3
"""Draw the 1,000-structure steering prompt set for llamat2-cif, and track it.

WHY A TRACKED CSV AND NOT THE CLONE. The conditions a prompt needs -- formula, element
list, space group -- live only in llamat/src/cifs/crystal-text-llm/data/test.csv, and
`llamat/` is an UNTRACKED clone. Generation would otherwise depend on a checkout that a
fresh clone does not have. This writes the four fields a prompt actually needs for the
1,000 sampled ids, which is small enough to commit, so steered generation never reads
the clone.

WHY THESE 1,000. Same contract as CrystaLLM's prompt set (scripts/data/make_test_sample.py):
a fixed `random.Random(42).sample` over the sorted id list, so every arm and the alpha=0
control draw the SAME prompts and pair on `id`. That pairing is the whole reason for
this file -- see CLAUDE.md, "That pairing does not exist for llamat2-cif", which this
prompt set is what fixes.

The source is llamat2-cif's own test split, the only held-out set we have for it.

WHY formula_sum AND NOT THE `elements` COLUMN. Training built the element list with
elements_from_formula_sum(_chemical_formula_sum), which follows CIF order; test.csv's
`elements` column is alphabetised and disagrees on 63.7% of structures. Storing the
formula_sum keeps one source of truth and reproduces what the model was trained on.

Usage:
    python scripts/data/make_llamat_test_sample.py
    python scripts/data/make_llamat_test_sample.py --verify
"""

import argparse
import hashlib
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llamat_prompts

SOURCE = "llamat/src/cifs/crystal-text-llm/data/test.csv"
OUT = "data/llamat_test_sample1000.csv"
SEED = 42
N = 1000


def fingerprint(ids):
    return hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def build(source, seed, n):
    t = pd.read_csv(source)
    t["id"] = "MP_" + t.material_id
    t = t.sort_values("id").reset_index(drop=True)       # order-independent of the file
    print(f"source {source}: {len(t):,} structures")

    idx = random.Random(seed).sample(range(len(t)), n)
    rows = []
    for j in sorted(idx):
        r = t.iloc[j]
        formula_sum = llamat_prompts.formula_sum_from_cif(r.cif)
        if not formula_sum:
            raise SystemExit(f"{r.id} has no _chemical_formula_sum; cannot build a prompt")
        rows.append({"id": r.id,
                     "pretty_formula": r.pretty_formula,
                     "formula_sum": formula_sum,
                     "spacegroup_number": int(r["spacegroup.number"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--verify", action="store_true",
                    help="Compare against the existing file and exit non-zero on a "
                         "mismatch. Writes nothing.")
    args = ap.parse_args()

    df = build(args.source, args.seed, args.n)
    ids = df["id"].tolist()
    print(f"drew {len(ids):,} with random.Random({args.seed}).sample")
    print(f"  fingerprint (sha1 of sorted ids): {fingerprint(ids)}")
    print(f"  distinct formulas {df.pretty_formula.nunique():,}, "
          f"distinct space groups {df.spacegroup_number.nunique()}")

    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        same = old["id"].tolist() == ids
        print(f"  existing {args.out}: {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            print(f"    existing fingerprint {fingerprint(old['id'].tolist())}")
            if not args.verify:
                raise SystemExit(
                    "refusing to overwrite a different sample -- results generated "
                    "against the old one would not be comparable. Delete it first if "
                    "that is really intended.")
        if args.verify:
            raise SystemExit(0 if same else 1)
        return

    if args.verify:
        raise SystemExit(f"{args.out} does not exist yet")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
