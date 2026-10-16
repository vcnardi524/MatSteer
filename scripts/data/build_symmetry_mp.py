#!/usr/bin/env python3
"""Space group and point group for every structure in the MP corpus.

WHY THIS EXISTS. metadata_mp.parquet carries 56 columns and NOT ONE of them is a space
group, point group or Wyckoff field -- only metadata.parquet (NOMAD) has those, and it
covers a different corpus. So the symmetry separability test had no labels to run on for
v1_mp at all.

The CIFs themselves do carry it: `_symmetry_space_group_name_H-M` is written into every
one. The point group then follows deterministically from the space group via pymatgen,
so both labels come out of one pass over the corpus.

The symbol -> point group mapping is cached: there are ~230 space groups against 154,879
structures, and constructing a pymatgen SpaceGroup is far from free.

Output columns are (id, space_group_symbol, point_group) with id MP_-prefixed, matching
the embedding keys, so utils.load_labeled_embeddings joins it with no renaming:

    METADATA_PATH=symmetry_mp.parquet LABEL_COL=point_group ... symmetry_separability.py

Usage:
    python scripts/data/build_symmetry_mp.py
    python scripts/data/build_symmetry_mp.py --dataset v1_all --out symmetry_v1_all.parquet
"""
import argparse
import gzip
import os
import pickle
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import DATASET_PKL, DATASETS, extract_space_group_symbol


def point_group_of(symbol: str, cache: dict) -> str:
    """Point group for a H-M space group symbol, or None if pymatgen cannot place it."""
    if symbol in cache:
        return cache[symbol]
    pg = None
    try:
        from pymatgen.symmetry.groups import SpaceGroup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pg = SpaceGroup(symbol).point_group
    except Exception:
        pg = None
    cache[symbol] = pg
    return pg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v1_mp", choices=list(DATASETS))
    ap.add_argument("--out", default=None, help="default: symmetry_<dataset>.parquet")
    args = ap.parse_args()

    pkl = DATASET_PKL[args.dataset]
    out = args.out or f"symmetry_{args.dataset}.parquet"
    print(f"Reading {pkl} ...")
    with gzip.open(pkl, "rb") as f:
        cifs = pickle.load(f)
    print(f"  {len(cifs):,} CIFs")

    cache, rows, no_symbol = {}, [], 0
    for cid, cif in cifs:
        sym = extract_space_group_symbol(cif)
        if not sym:
            no_symbol += 1
            continue
        rows.append((cid, sym, point_group_of(sym, cache)))
    df = pd.DataFrame(rows, columns=["id", "space_group_symbol", "point_group"])
    df.to_parquet(out, index=False)

    n_pg = df.point_group.notna().sum()
    print(f"\nWrote {out}: {len(df):,} rows")
    print(f"  no _symmetry_space_group_name_H-M in the CIF : {no_symbol:,}")
    print(f"  space group symbols seen                     : {df.space_group_symbol.nunique():,}")
    print(f"  mapped to a point group                      : {n_pg:,} ({n_pg/len(df):.2%})")
    print(f"  distinct point groups                        : {df.point_group.nunique():,}")
    print("\n  most common space groups:")
    print(df.space_group_symbol.value_counts().head(6).to_string())


if __name__ == "__main__":
    main()
