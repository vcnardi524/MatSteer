#!/usr/bin/env python3
"""
Two-part Wyckoff/space-group analysis over the NOMAD preparsed metadata.

For every material we extract:
    space_group_symbol   (material.symmetry.space_group_symbol)
    wyckoff signature    (anonymous: sorted multiset of occupied Wyckoff letters
                          from the conventional-cell topology entry, e.g. "a b c")

Then reports:
  (1) Per space group: how many DISTINCT wyckoff signatures occur.
  (2) Cross space group: how many DISTINCT space groups each signature occurs in,
      i.e. which signatures are shared across space groups.

Outputs CSVs under analysis/ and prints summaries.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

INP = "preparsed_metadata_nomad.parquet"
OUTDIR = Path("analysis")
BATCH = 2000


def find_wyckoff_sets(results: dict):
    topo = results.get("material", {}).get("topology")
    if not isinstance(topo, list):
        return None
    best = None
    for t in topo:
        if not isinstance(t, dict):
            continue
        sym = t.get("symmetry")
        if not isinstance(sym, dict):
            continue
        ws = sym.get("wyckoff_sets")
        if ws:
            if t.get("label") == "conventional cell":
                return ws
            best = best or ws
    return best


def signature(ws: list):
    letters = [s.get("wyckoff_letter") for s in ws
               if isinstance(s, dict) and s.get("wyckoff_letter") is not None]
    if not letters:
        return None
    lc = Counter(letters)
    return " ".join(f"{n}{l}" if n > 1 else l for l, n in sorted(lc.items()))


def main():
    OUTDIR.mkdir(exist_ok=True)
    pf = pq.ParquetFile(INP)
    total = pf.metadata.num_rows
    print(f"Streaming {total:,} rows ...", flush=True)

    # sg -> Counter(signature -> n_materials)
    sg_sig = defaultdict(Counter)
    # signature -> Counter(sg -> n_materials)
    sig_sg = defaultdict(Counter)

    seen = 0
    hit = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=["id", "results"]):
        for res in batch.column("results").to_pylist():
            seen += 1
            try:
                r = json.loads(res)
            except (json.JSONDecodeError, TypeError):
                continue
            ws = find_wyckoff_sets(r)
            if not ws:
                continue
            sig = signature(ws)
            if sig is None:
                continue
            sg = (r.get("material", {}).get("symmetry", {}) or {}).get("space_group_symbol")
            if not sg:
                continue
            sg_sig[sg][sig] += 1
            sig_sg[sig][sg] += 1
            hit += 1
        if seen % 100000 < BATCH:
            print(f"  {seen:,}/{total:,}  (with wyckoff+sg: {hit:,})", flush=True)

    print(f"\nScanned {seen:,}; usable {hit:,} ({hit/seen:.1%})")
    print(f"Distinct space groups: {len(sg_sig):,}")
    print(f"Distinct wyckoff signatures: {len(sig_sg):,}")

    # ---- (1) per space group: number of distinct signatures ----
    rows1 = []
    for sg, sigs in sg_sig.items():
        rows1.append({
            "space_group": sg,
            "n_materials": sum(sigs.values()),
            "n_unique_signatures": len(sigs),
            "top_signature": sigs.most_common(1)[0][0],
        })
    df1 = pd.DataFrame(rows1).sort_values("n_unique_signatures", ascending=False)
    df1.to_csv(OUTDIR / "wyckoff_per_sg_unique.csv", index=False)

    print("\n" + "=" * 70)
    print("(1) UNIQUE WYCKOFF SIGNATURES PER SPACE GROUP  (top 20 by # unique)")
    print("=" * 70)
    print(df1.head(20).to_string(index=False))
    print(f"\n  median unique signatures per SG: {df1['n_unique_signatures'].median():.0f}")
    print(f"  mean:   {df1['n_unique_signatures'].mean():.1f}")
    print(f"  max:    {df1['n_unique_signatures'].max():,} "
          f"(SG {df1.iloc[0]['space_group']})")

    # ---- (2) signatures shared across space groups ----
    rows2 = []
    for sig, sgs in sig_sg.items():
        rows2.append({
            "signature": sig,
            "n_space_groups": len(sgs),
            "n_materials": sum(sgs.values()),
            "example_sgs": ", ".join(s for s, _ in sgs.most_common(5)),
        })
    df2 = pd.DataFrame(rows2).sort_values(
        ["n_space_groups", "n_materials"], ascending=False)
    df2.to_csv(OUTDIR / "wyckoff_shared_signatures.csv", index=False)

    n_shared = (df2["n_space_groups"] > 1).sum()
    print("\n" + "=" * 70)
    print("(2) SIGNATURES SHARED ACROSS SPACE GROUPS  (top 20 by # space groups)")
    print("=" * 70)
    print(df2.head(20).to_string(index=False))
    print(f"\n  signatures total:                 {len(df2):,}")
    print(f"  signatures in >1 space group:     {n_shared:,} ({n_shared/len(df2):.1%})")
    print(f"  signatures unique to one SG:      {len(df2)-n_shared:,}")
    print(f"\nCSVs -> {OUTDIR}/wyckoff_per_sg_unique.csv, {OUTDIR}/wyckoff_shared_signatures.csv")


if __name__ == "__main__":
    main()
