#!/usr/bin/env python3
"""
Add a `cif_token_len` column to metadata_mp.parquet: the tokenized length of each
material's AUGMENTED CIF (what the model actually ingests), keyed by material_id.

Source of truth is CrystaLLM/cifs_v1_mp_prep.pkl.gz — the preprocessed/augmented
(id, cif) pairs that were tokenized into tokens_v1_mp. We re-tokenize each with the
same path tokenize_cifs.py uses (strip comment/pymatgen lines, then tokenize_cif),
so the count matches train.bin exactly. Rows dropped during preprocessing (8 of
154,879) get NaN.

Run inside CrystaLLM/crystallm_venv (needs the crystallm tokenizer).

Usage:
    python scripts/add_cif_token_len_column.py
"""
import gzip
import os
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CrystaLLM"))
from crystallm import CIFTokenizer

PREP = Path("CrystaLLM/cifs_v1_mp_prep.pkl.gz")
META = Path("metadata_mp.parquet")
COL = "cif_token_len"

_tok = None


def _init():
    global _tok
    _tok = CIFTokenizer()


def _clean(cif):
    # mirror tokenize_cifs.py:preprocess() so the length matches train.bin
    out = [l.strip() for l in cif.split("\n")
           if l.strip() and not l.strip().startswith("#") and "pymatgen" not in l]
    out.append("\n")
    return "\n".join(out)


def _len(item):
    mid, cif = item
    return mid, len(_tok.tokenize_cif(_clean(cif)))


def main():
    print(f"Loading {PREP} ...")
    with gzip.open(PREP, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data):,} augmented CIFs")

    print("Tokenizing (8 workers) ...")
    with Pool(8, initializer=_init) as pool:
        pairs = pool.map(_len, data, chunksize=200)
    lens = pd.DataFrame(pairs, columns=["material_id", COL])
    print(f"  lengths: min={lens[COL].min()} max={lens[COL].max():,} "
          f"mean={lens[COL].mean():.0f}  >2048={(lens[COL] > 2048).sum():,}")

    print(f"Merging into {META} ...")
    df = pd.read_parquet(META)
    if COL in df.columns:
        df = df.drop(columns=[COL])
    df = df.merge(lens, on="material_id", how="left")
    n_missing = int(df[COL].isna().sum())
    print(f"  {n_missing} rows without a token length (dropped in preprocessing)")

    bak = META.with_suffix(".parquet.bak_pretoklen")
    if not bak.exists():
        shutil.copy2(META, bak)
        print(f"  backed up -> {bak}")
    tmp = META.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, META)
    print(f"Wrote {META}  ({len(df):,} rows, {len(df.columns)} cols, added '{COL}')")


if __name__ == "__main__":
    main()
