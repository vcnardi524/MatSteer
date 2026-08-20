#!/usr/bin/env python3
"""
Usage:
    python parse_preparsed_metadata.py --in preparsed_metadata_nomad.parquet \
                                       --out metadata_nomad.parquet \
                                       --paths paths.txt

paths.txt: one dot-separated path per line, e.g.
    material.symmetry.point_group
    properties.electronic.dos_electronic.band_gap.energy_lowest_unoccupied
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd


def get_by_path(obj, path: str):
    for part in path.split("."):
        if isinstance(obj, list):
            obj = obj[0] if obj else None
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    if isinstance(obj, list):
        obj = obj[0] if obj else None
    return obj


def derive_col_names(paths: list[str]) -> dict[str, str]:
    """Map path → column name, using fewest trailing segments needed to stay unique."""
    col_to_path = {}
    path_to_col = {}
    for path in paths:
        parts = path.split(".")
        for depth in range(1, len(parts) + 1):
            candidate = "_".join(parts[-depth:])
            if candidate not in col_to_path:
                col_to_path[candidate] = path
                path_to_col[path] = candidate
                break
        else:
            path_to_col[path] = path.replace(".", "_")
    return path_to_col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",   dest="inp", default="preparsed_metadata_nomad.parquet")
    parser.add_argument("--out",  default="metadata_nomad.parquet")
    parser.add_argument("--paths", required=True, help="text file with one dot-path per line")
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    path_to_col = derive_col_names(paths)
    print("Column mapping:")
    for path, col in path_to_col.items():
        print(f"  {col:<40}  ← {path}")

    print(f"\nLoading {args.inp} ...")
    df = pd.read_parquet(args.inp, columns=["id", "results"])
    print(f"Rows: {len(df):,}")

    rows = []
    for _, row in df.iterrows():
        rec = {"id": row["id"]}
        try:
            r = json.loads(row["results"])
            for path, col in path_to_col.items():
                val = get_by_path(r, path)
                rec[col] = val if val is not None else np.nan
        except (json.JSONDecodeError, AttributeError):
            for col in path_to_col.values():
                rec[col] = np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)

    print(f"\nSaved {len(out):,} rows → {args.out}")
    print(f"\nNull counts:")
    print(out.isnull().sum().to_string())


if __name__ == "__main__":
    main()
