#!/usr/bin/env python3
"""
Add a Wyckoff-letter column to metadata.parquet:
  wyckoff_letters  <- sorted DISTINCT occupied Wyckoff letters, space-separated

Read from results.material.topology[].symmetry.wyckoff_sets in
preparsed_metadata_nomad.parquet, preferring the "conventional cell" topology
entry (Wyckoff multiplicities are defined w.r.t. the conventional cell).

Letters only -- no counts. A letter is listed once no matter how many
element-sets occupy it, so Pnma with (c,Se)(c,Se)(c,Se)(c,Zr)(c,Ba) -> "c".
This deliberately avoids the "5c"-style notation, which reads as a multiplicity
but is really a count of sets (there is no 5c in Pnma).

Sorting is canonical (a..z then A, per the International Tables) so the same
occupied set always produces the same string.

metadata.parquet is rewritten IN PLACE; a copy is saved to metadata.parquet.bak2
first (.bak is the Jun-2 pre-bandgap snapshot).

Usage:
    python scripts/data/add_wyckoff_letters_column.py
"""
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PREPARSED = "preparsed_metadata_nomad.parquet"
METADATA = "metadata.parquet"
BACKUP = "metadata.parquet.bak2"
COL = "wyckoff_letters"
BATCH = 4000


def find_wyckoff_sets(results: dict):
    """wyckoff_sets from the conventional cell if present, else any topology entry."""
    topo = results.get("material", {}).get("topology")
    if not isinstance(topo, list):
        return None
    best = None
    for t in topo:
        if not isinstance(t, dict):
            continue
        sym = t.get("symmetry")
        if isinstance(sym, dict) and sym.get("wyckoff_sets"):
            if t.get("label") == "conventional cell":
                return sym["wyckoff_sets"]
            best = best or sym["wyckoff_sets"]
    return best


def letter_key(c: str):
    """ITA order: lowercase a..z first, then uppercase A (groups with >26 positions)."""
    return (c.isupper(), c)


def letters_signature(ws) -> str | None:
    letters = {s.get("wyckoff_letter") for s in ws
               if isinstance(s, dict) and s.get("wyckoff_letter") is not None}
    if not letters:
        return None
    return " ".join(sorted(letters, key=letter_key))


def main():
    print(f"Streaming {PREPARSED} ...")
    pf = pq.ParquetFile(PREPARSED)
    print(f"  {pf.metadata.num_rows:,} rows")

    sig = {}
    n_seen = n_nosets = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=["id", "results"]):
        ids = batch.column("id").to_pylist()
        res = batch.column("results").to_pylist()
        for id_, r in zip(ids, res):
            n_seen += 1
            if r is None:
                continue
            try:
                d = json.loads(r) if isinstance(r, str) else r
            except Exception:
                continue
            ws = find_wyckoff_sets(d)
            if not ws:
                n_nosets += 1
                continue
            s = letters_signature(ws)
            if s is not None:
                sig[id_] = s
        if n_seen % 200000 < BATCH:
            print(f"  {n_seen:,} scanned, {len(sig):,} with letters", flush=True)

    print(f"  scanned {n_seen:,};  with wyckoff letters: {len(sig):,};  no wyckoff_sets: {n_nosets:,}")

    print(f"\nLoading {METADATA} ...")
    meta = pd.read_parquet(METADATA)
    print(f"  {len(meta):,} rows, cols={list(meta.columns)}")

    print(f"Backing up -> {BACKUP}")
    shutil.copy2(METADATA, BACKUP)

    meta[COL] = meta["id"].map(sig)
    n_hit = int(meta[COL].notna().sum())
    print(f"  {COL}: {n_hit:,} populated  ({n_hit/len(meta):.1%}),  {len(meta)-n_hit:,} NaN")

    meta.to_parquet(METADATA, index=False)
    print(f"  Saved {METADATA} with new column '{COL}'.")

    v = meta[COL].dropna()
    print(f"\n=== {COL} ===")
    print(f"  distinct signatures : {v.nunique():,}")
    print(f"  most common:")
    print(v.value_counts().head(10).to_string())
    print(f"\n  letters per material: mean={v.str.split().str.len().mean():.2f}  "
          f"max={v.str.split().str.len().max()}")


if __name__ == "__main__":
    main()
