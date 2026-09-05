#!/usr/bin/env python3
"""How much real information does each column of a metadata parquet actually carry?

"What fraction is populated" is the obvious question and it is not quite sufficient: a
column can be 100% non-null and still carry nothing because every row holds the same
value. It is also worth seeing where the zeros are, because a large spike at zero
distorts bucketing and regression even when the zeros are perfectly real.

Zeros are REPORTED, not judged. band_gap = 0 means the material is a metal and
energy_above_hull = 0 means it sits on the hull -- both are genuine measurements, not
missing data. Whether a zero counts as signal is a per-property decision for the reader;
this file states the facts and leaves it.

Columns are read one at a time -- metadata_mp.parquet is 643 MB and pandas would want the
whole thing in memory otherwise. Null counts come from Arrow, which reads them out of the
file's own statistics rather than the data.

    pct_present   non-null, and for list columns also non-empty
    pct_zero      of the whole table, numerics only: exactly zero. A high value flags a
                  pile-up to handle before bucketing -- it does NOT mean invalid
    n_unique      distinct non-null values; 1 means the column is constant and carries
                  nothing regardless of how populated it looks
    verdict       empty / constant / sparse / ok -- from presence and uniqueness only

Usage:
    python scripts/data/property_coverage.py metadata_mp.parquet
    python scripts/data/property_coverage.py metadata.parquet --out analysis/foo.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def summarise(col: pa.ChunkedArray, name: str, n_rows: int) -> dict:
    t = col.type
    n_null = col.null_count
    n_present = n_rows - n_null
    row = dict(column=name, dtype=str(t)[:38], n_rows=n_rows,
               n_present=n_present, pct_present=100 * n_present / n_rows if n_rows else 0)

    # nested types: report presence only, and for lists treat [] as absent
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        lens = pa.compute.list_value_length(col).to_pandas()
        nonempty = int((lens.fillna(0) > 0).sum())
        row.update(n_zero=np.nan, pct_zero=np.nan, n_unique=np.nan,
                   verdict="empty" if nonempty == 0 else
                           "sparse" if nonempty < 0.5 * n_rows else "ok")
        row["n_present"], row["pct_present"] = nonempty, 100 * nonempty / n_rows
        return row
    if pa.types.is_struct(t):
        row.update(n_zero=np.nan, pct_zero=np.nan, n_unique=np.nan,
                   verdict="empty" if n_present == 0 else "ok")
        return row

    s = col.to_pandas()
    nn = s.dropna()
    n_unique = int(nn.nunique()) if len(nn) else 0
    numeric = pa.types.is_floating(t) or pa.types.is_integer(t)
    n_zero = int((nn == 0).sum()) if numeric else np.nan
    row.update(n_zero=n_zero, n_unique=n_unique,
               pct_zero=(100 * n_zero / n_rows) if numeric and n_rows else np.nan)
    # presence and uniqueness only -- a column full of real zeros is not "bad"
    row["verdict"] = ("empty" if n_present == 0 else
                      "constant" if n_unique <= 1 else
                      "sparse" if n_present < 0.5 * n_rows else "ok")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    f = pq.ParquetFile(args.path)
    n_rows = f.metadata.num_rows
    names = f.schema_arrow.names
    print(f"{args.path}: {n_rows:,} rows, {len(names)} columns\n")

    rows = []
    for i, name in enumerate(names, 1):
        col = f.read(columns=[name]).column(0)
        rows.append(summarise(col, name, n_rows))
        print(f"  [{i:>2}/{len(names)}] {name}", flush=True)

    d = pd.DataFrame(rows).sort_values(["pct_present", "column"], ascending=[False, True])
    out = Path(args.out or
               f"analysis/{Path(args.path).stem}_property_coverage.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, float_format="%.4g")

    print(f"\n{'column':<40}{'pct_present':>12}{'pct_zero':>10}{'n_unique':>10}  verdict")
    for _, r in d.iterrows():
        u = "" if pd.isna(r.n_unique) else f"{int(r.n_unique):,}"
        z = "" if pd.isna(r.pct_zero) else f"{r.pct_zero:.2f}%"
        print(f"  {r.column:<38}{r.pct_present:>11.2f}%{z:>10}{u:>10}  {r.verdict}")
    print(f"\nverdict counts: {d.verdict.value_counts().to_dict()}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
