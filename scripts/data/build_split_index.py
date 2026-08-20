#!/usr/bin/env python3
"""Build the id -> train/val/test lookup that analysis scripts filter on.

CrystaLLM's split lives in three gzipped pickles that take ~2 minutes to load, which
is far too slow to repeat in every analysis run. This flattens them once into a small
parquet so partition filtering is a cheap merge.

Ids present in the embeddings but in none of the three splits are simply absent here;
utils.load_split_index labels them "unknown". That is not an error -- the MP corpus
(cifs_v1_mp_prep) contains ~96k CIFs that the dedup step removed before the v1 split
was made, so they legitimately have no partition.

Usage:
    python scripts/data/build_split_index.py
"""
import gzip
import os
import pickle

import pandas as pd

SPLIT_PKLS = {
    "train": "CrystaLLM/cifs_v1_train.pkl.gz",
    "val": "CrystaLLM/cifs_v1_val.pkl.gz",
    "test": "CrystaLLM/cifs_v1_test.pkl.gz",
}
OUT_PATH = "splits_v1.parquet"


def main():
    ids, splits = [], []
    for split, path in SPLIT_PKLS.items():
        print(f"Loading {path} ...")
        with gzip.open(path, "rb") as f:
            entries = pickle.load(f)
        ids.extend(cif_id for cif_id, _ in entries)
        splits.extend([split] * len(entries))
        print(f"  {split}: {len(entries):,}")

    df = pd.DataFrame({"id": ids, "split": pd.Categorical(splits)})
    assert df["id"].is_unique, "an id appears in more than one split"
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} ids to {OUT_PATH}")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
