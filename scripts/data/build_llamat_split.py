#!/usr/bin/env python3
"""Build the id -> split lookup for the LLaMat models.

WHY THIS EXISTS. A split belongs to a MODEL, not to a corpus. CrystaLLM's train/val/test
was drawn for CrystaLLM and says nothing about what LLaMat held out, so scoring
llamat2_cif against CrystaLLM's "val" would evaluate it on rows it may well have trained
on.

The only evidence we have about LLaMat is the test set its own CIF pipeline shipped,
llamat/src/cifs/crystal-text-llm/data/test.csv -- 9,046 materials, inherited from the
Gruver et al. crystal-text-llm dataset. That is 5.6% of metadata_mp.parquet and 6.1% of
what we embedded, so the other ~94% is of UNKNOWN status, not known-trained-on.

WHAT THIS FILE CONTAINS, AND WHAT IT DELIBERATELY DOES NOT. Only those test ids, labelled
"test". No train rows, no val rows. utils.partition_id_sets defines `not_heldout` by
EXCLUSION -- everything that is not val or test -- so with only test rows listed,
`not_heldout` resolves to exactly "everything we have no evidence was held out". That is
the honest fitting pool, and it is available without claiming LLaMat trained on any of it.

Ids are written MP_-prefixed (MP_mp-10009) to match the embedding and metadata_mp keys;
test.csv itself stores them bare (mp-10009).

NOTE the labels in test.csv are NOT used anywhere. Checked against the live Materials
Project API on a random sample of 400: test.csv reproduces current MP formation energy
only 14.5% of the time against metadata_mp.parquet's 88.2%, so it is an older MP release.
This file takes the ID LIST only; every label comes from metadata_mp.parquet.

Usage:
    python scripts/data/build_llamat_split.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import SPLIT_FILES

TEST_CSV = "llamat/src/cifs/crystal-text-llm/data/test.csv"
OUT_PATH = SPLIT_FILES["llamat2_cif"]


def main():
    if not os.path.exists(TEST_CSV):
        raise SystemExit(f"{TEST_CSV} not found -- the llamat clone must be present")
    t = pd.read_csv(TEST_CSV, usecols=["material_id"])
    if not t.material_id.is_unique:
        raise SystemExit("material_id is not unique in test.csv")

    df = pd.DataFrame({"id": "MP_" + t.material_id.astype(str), "split": "test"})
    df["split"] = df["split"].astype("category")
    df.to_parquet(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH}: {len(df):,} ids, all split=test")
    print(f"  e.g. {list(df.id[:3])}")
    print("  train/val are intentionally empty -- `not_heldout` covers the rest by "
          "exclusion")


if __name__ == "__main__":
    main()
