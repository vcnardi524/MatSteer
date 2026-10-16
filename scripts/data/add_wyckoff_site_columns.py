#!/usr/bin/env python3
"""Add multiplicity- and element-resolved Wyckoff columns to metadata.parquet.

`add_wyckoff_letters_column.py` keeps only the DISTINCT occupied letters, so Pnma with
(c,Se)(c,Se)(c,Se)(c,Zr)(c,Ba) collapses to just "c". That throws away two things NOMAD
already provides in the same `wyckoff_sets` records: how many atoms sit on each position,
and which element occupies it. Both are recoverable with no recomputation and no change
of convention, which matters -- recomputing letters with pymatgen instead agrees with
NOMAD only 71% of the time because Wyckoff letters are setting-dependent.

Adds ONE column, leaving `wyckoff_letters` untouched:

    wyckoff_sites   "4a 4c 4c 8d"   one token per SET, multiplicity + letter

ELEMENT-RESOLVED LABELS WERE TRIED AND DROPPED. Tagging each site with its element
("Ba:4c O:8d Se:4c") is more concrete but is not a usable category: measured over 251,056
structures it gives 230,386 distinct values, and NOT ONE of them reaches 30 members. It is
effectively a per-structure identifier, so every cocluster enrichment cell would hold ~1
row. Granularity that fine has to be modelled as composition x site, not as a categorical.

    label            distinct   categories >=30   rows they cover
    letters               441                80             99.4%
    sites               1,642               147             98.1%
    element-resolved  230,386                 0              0.0%

THE MULTIPLICITY IS PER SET, NOT PER LETTER. `len(indices)` is the multiplicity of that
one orbit -- a real 4c. Three separate sets on letter c give "4c 4c 4c", NOT "12c", and
certainly not "3c": Pnma has no 3c, and summing across sets would invent multiplicities
that do not exist in the International Tables. That trap is why the original script
avoided counts altogether; the fix is to keep the sets separate rather than to drop them.

Tokens are sorted canonically (by letter in ITA order, then multiplicity) so the same
occupied set always produces the same string.

metadata.parquet is rewritten IN PLACE after a backup to metadata.parquet.bak3.

Usage:
    python scripts/data/add_wyckoff_site_columns.py
    python scripts/data/add_wyckoff_site_columns.py --dry-run     # report, write nothing
"""
import argparse
import json
import shutil
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

PREPARSED = "preparsed_metadata_nomad.parquet"
METADATA = "metadata.parquet"
BACKUP = "metadata.parquet.bak3"
BATCH = 4000


def find_wyckoff_sets(results: dict):
    """wyckoff_sets from the conventional cell if present, else any topology entry.

    Multiplicities are defined w.r.t. the conventional cell, so preferring that entry
    is not cosmetic -- a primitive-cell set would give the wrong counts.
    """
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


def signature(ws):
    """wyckoff_sites for one structure, or None."""
    toks = []
    for s in ws:
        if not isinstance(s, dict):
            continue
        letter = s.get("wyckoff_letter")
        idx = s.get("indices")
        if letter is None or not idx:
            continue
        mult = len(idx)
        toks.append((letter, mult))
    if not toks:
        return None
    toks.sort(key=lambda t: (letter_key(t[0]), t[1]))
    return " ".join(f"{m}{l}" for l, m in toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Streaming {PREPARSED} ...")
    pf = pq.ParquetFile(PREPARSED)
    print(f"  {pf.metadata.num_rows:,} rows")

    sites = {}
    n_seen = n_nosets = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=["id", "results"]):
        for id_, r in zip(batch.column("id").to_pylist(),
                          batch.column("results").to_pylist()):
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
            a = signature(ws)
            if a is not None:
                sites[id_] = a
        if n_seen % 200000 < BATCH:
            print(f"  {n_seen:,} scanned, {len(sites):,} with sites", flush=True)
    print(f"  scanned {n_seen:,};  with sites: {len(sites):,};  no wyckoff_sets: {n_nosets:,}")

    print(f"\nLoading {METADATA} ...")
    meta = pd.read_parquet(METADATA)
    print(f"  {len(meta):,} rows")

    new = meta["id"].map(sites)
    n_hit = int(new.notna().sum())
    print(f"  populated: {n_hit:,} ({n_hit/len(meta):.1%})")

    # the new columns must agree with the existing letters column wherever both exist --
    # same source records, so a mismatch means the parse changed meaning
    both = meta["wyckoff_letters"].notna() & new.notna()
    derived = new[both].str.split().apply(
        lambda ts: " ".join(sorted({t.lstrip("0123456789") for t in ts}, key=letter_key)))
    agree = (derived == meta.loc[both, "wyckoff_letters"]).mean()
    print(f"  letters re-derived from wyckoff_sites match the existing column: "
          f"{100*agree:.2f}% of {int(both.sum()):,}")

    for col, v in (("wyckoff_letters", meta["wyckoff_letters"].dropna()),
                   ("wyckoff_sites", new.dropna())):
        vc = v.value_counts(); big = vc[vc >= 30]
        print(f"  {col:<20} {v.nunique():>8,} distinct, {len(big):>5,} categories with "
              f">=30 rows covering {100*big.sum()/len(v):.1f}%")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        print("\nexamples:")
        ex = pd.DataFrame({"letters": meta.loc[both, "wyckoff_letters"],
                           "sites": new[both]}).head(6)
        print(ex.to_string(index=False))
        return

    print(f"Backing up -> {BACKUP}")
    shutil.copy2(METADATA, BACKUP)
    meta["wyckoff_sites"] = new
    meta.to_parquet(METADATA, index=False)
    print(f"  Saved {METADATA} with wyckoff_sites")


if __name__ == "__main__":
    main()
