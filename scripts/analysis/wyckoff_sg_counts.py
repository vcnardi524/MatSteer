#!/usr/bin/env python3
"""How many materials share each (space group + occupied Wyckoff sites) symmetry?

This is the label the cocluster enrichment actually tests. A cluster is called enriched
for a symmetry when it holds more of that label than chance predicts, so the label is only
testable if enough materials share it -- a category with three members cannot be enriched
in any meaningful sense.

Reports the same table for both resolutions so the cost of the finer label is visible:

    sg_letters   space group + DISTINCT occupied letters   ("Pnma | c")
    sg_sites     space group + one token per SET           ("Pnma | 4c 4c 4c 4c 4c")

Partition matters: 89.6% of labelled structures are in CrystaLLM's training set, so
counts over `all` describe the corpus, not what a held-out analysis can test.

    python scripts/analysis/wyckoff_sg_counts.py --partition all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import filter_partition, analysis_dir                      # noqa: E402

THRESHOLDS = (2, 5, 10, 30, 100, 1000)


def summarise(counts: pd.Series, name: str, n_rows: int) -> dict:
    row = {"label": name, "n_materials": n_rows, "n_categories": len(counts),
           "median_size": counts.median(), "largest": counts.iloc[0],
           "largest_share_pct": 100 * counts.iloc[0] / n_rows}
    for t in THRESHOLDS:
        big = counts[counts >= t]
        row[f"cats_ge_{t}"] = len(big)
        row[f"rows_ge_{t}_pct"] = 100 * big.sum() / n_rows
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="metadata.parquet")
    ap.add_argument("--partition", default="all")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cols = ["id", "space_group_symbol", "wyckoff_letters", "wyckoff_sites"]
    d = pd.read_parquet(args.metadata, columns=cols)
    print(f"{len(d):,} rows in {args.metadata}")
    if args.partition != "all":
        d = filter_partition(d, args.partition, verbose=True)

    d = d[d.space_group_symbol.notna()]
    d["sg_letters"] = np.where(d.wyckoff_letters.notna(),
                               d.space_group_symbol.astype(str) + " | "
                               + d.wyckoff_letters.astype(str), None)
    d["sg_sites"] = np.where(d.wyckoff_sites.notna(),
                             d.space_group_symbol.astype(str) + " | "
                             + d.wyckoff_sites.astype(str), None)

    out_dir = Path(args.out_dir or analysis_dir("v1_all", None, args.partition))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in ("sg_letters", "sg_sites"):
        v = d[name].dropna()
        counts = v.value_counts()
        rows.append(summarise(counts, name, len(v)))
        counts.rename("n_materials").rename_axis(name).to_csv(
            out_dir / f"wyckoff_{name}_counts_{args.partition}.csv")
        print(f"\n=== {name} ===  {len(v):,} materials with this label")
        print(f"  {len(counts):,} distinct symmetries")
        print(f"  size: median {counts.median():.0f}, largest {counts.iloc[0]:,} "
              f"({100*counts.iloc[0]/len(v):.1f}% of rows)")
        print(f"  {'members>=':<12}{'symmetries':>12}{'rows covered':>15}")
        for t in THRESHOLDS:
            big = counts[counts >= t]
            print(f"  {t:<12}{len(big):>12,}{100*big.sum()/len(v):>14.1f}%")
        print("  most common:")
        for k, n in counts.head(6).items():
            print(f"    {n:>8,}  {k}")

    summary = pd.DataFrame(rows)
    p = out_dir / f"wyckoff_sg_summary_{args.partition}.csv"
    summary.to_csv(p, index=False, float_format="%.4g")
    print(f"\nSaved per-symmetry counts and {p}")


if __name__ == "__main__":
    main()
