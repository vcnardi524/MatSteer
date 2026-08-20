#!/usr/bin/env python3
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "metadata_prep.parquet"
df = pd.read_parquet(path)

print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col}")

import json

print(f"\nFirst row:")
for col in df.columns:
    print(f"\n--- {col} ---")
    val = df.iloc[0][col]
    try:
        print(json.dumps(json.loads(val), indent=2))
    except (TypeError, ValueError):
        print(val)
