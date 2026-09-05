#!/usr/bin/env python3
"""How much real information does each column of a metadata parquet actually carry?

"What fraction is populated" is the obvious question and it is not sufficient: a column
can be 100% non-null and still useless because every row holds the same value, or because
it is numeric and entirely zero. All three cases matter when deciding whether a property
is worth probing or steering, so all three are reported.

Columns are read one at a time -- metadata_mp.parquet is 643 MB and pandas would want the
whole thing in memory otherwise. Null counts come from Arrow, which reads them out of the
file's own statistics rather than the data.

    pct_present   non-null, and for list columns also non-empty
    pct_usable    of the whole table: non-null, and additionally non-zero for numerics.
                  This is the number to read when asking "can I use this property".
    n_unique      distinct non-null values; 1 means the column is constant and carries
                  nothing regardless of how populated it looks
    verdict       a one-word read: empty / constant / all-zero / sparse / ok

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
        row.update(n_usable=nonempty, pct_usable=100 * nonempty / n_rows,
                   n_unique=np.nan,
                   verdict="empty" if nonempty == 0 else
                           "sparse" if nonempty < 0.5 * n_rows else "ok")
        return row
    if pa.types.is_struct(t):
        row.update(n_usable=n_present, pct_usable=row["pct_present"], n_unique=np.nan,
                   verdict="empty" if n_present == 0 else "ok")
        return row

    s = col.to_pandas()
    nn = s.dropna()
    n_unique = int(nn.nunique()) if len(nn) else 0
    if pa.types.is_floating(t) or pa.types.is_integer(t):
        n_zero = int((nn == 0).sum())
        n_usable = len(nn) - n_zero
    else:
        n_zero = 0
        n_usable = len(nn)
    row.update(n_usable=n_usable, pct_usable=100 * n_usable / n_rows if n_rows else 0,
               n_unique=n_unique)
    row["verdict"] = ("empty" if n_present == 0 else
                      "constant" if n_unique <= 1 else
                      "all-zero" if n_usable == 0 else
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

    d = pd.DataFrame(rows).sort_values(["pct_usable", "column"], ascending=[False, True])
    out = Path(args.out or
               f"analysis/{Path(args.path).stem}_property_coverage.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, float_format="%.4g")

    print(f"\n{'column':<40}{'pct_present':>12}{'pct_usable':>11}{'n_unique':>10}  verdict")
    for _, r in d.iterrows():
        u = "" if pd.isna(r.n_unique) else f"{int(r.n_unique):,}"
        print(f"  {r.column:<38}{r.pct_present:>11.2f}%{r.pct_usable:>10.2f}%{u:>10}  "
              f"{r.verdict}")
    print(f"\nverdict counts: {d.verdict.value_counts().to_dict()}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
