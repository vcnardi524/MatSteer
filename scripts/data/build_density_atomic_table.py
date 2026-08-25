#!/usr/bin/env python3
"""
Volume per atom for every structure in the v1 corpus, straight from the CIF text.

MP's `density_atomic` column covers only the 58,650 v1 ids that carry a material_id,
which is too few to define a target class over the training set. The same quantity is
written into every CIF, though -- `_cell_volume` divided by the atom count summed from
`_chemical_formula_sum`, both full-cell values -- so it can be read for all 2.29M
structures with a regex and no pymatgen parse.

Output: density_atomic_v1.parquet
  columns: id, split, cell_volume, n_atoms, density_atomic

Usage:
    python scripts/data/build_density_atomic_table.py
"""
import argparse
import gzip
import pickle
from pathlib import Path

import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import cell_volume_from_text, natoms_from_text

SPLIT_PKLS = {split: f"CrystaLLM/cifs_v1_{split}.pkl.gz" for split in ("train", "val", "test")}


def read_split(split: str, path: str) -> pd.DataFrame:
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame({
        "id": [i for i, _ in data],
        "split": split,
        "cell_volume": [cell_volume_from_text(c) for _, c in data],
        "n_atoms": [natoms_from_text(c) for _, c in data],
    })
    print(f"  {split}: {len(df):,} structures")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="density_atomic_v1.parquet")
    args = ap.parse_args()

    print("Reading CIFs ...")
    df = pd.concat([read_split(s, p) for s, p in SPLIT_PKLS.items()], ignore_index=True)
    df["density_atomic"] = df["cell_volume"] / df["n_atoms"]

    v = df["density_atomic"]
    print(f"\n{len(df):,} structures, {v.isna().sum():,} unreadable")
    print(f"  min={v.min():.3f}  p1={v.quantile(.01):.3f}  median={v.median():.3f}  "
          f"p99={v.quantile(.99):.3f}  max={v.max():.3f}  mean={v.mean():.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
