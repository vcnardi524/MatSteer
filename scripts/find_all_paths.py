#!/usr/bin/env python3
import json
import sys
import pandas as pd
from collections import Counter

def get_leaf_paths(data, current_path=""):
    paths = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            if isinstance(value, dict):
                if value:
                    paths.extend(get_leaf_paths(value, new_path))
            elif isinstance(value, list):
                if value:
                    paths.extend(get_leaf_paths(value, new_path))
            else:
                paths.append(new_path)

    elif isinstance(data, list):
        for item in data:
            paths.extend(get_leaf_paths(item, current_path))

    return paths

path = sys.argv[1] if len(sys.argv) > 1 else "preparsed_metadata_nomad.parquet"

print(f"Loading {path} ...")
df = pd.read_parquet(path, columns=["id", "results"])
print(f"Rows: {len(df):,}")

missing = 0
counter = Counter()
total = len(df)

for results_str in df["results"]:
    try:
        results = json.loads(results_str)
        paths = set(get_leaf_paths(results))
        counter.update(paths)
    except (json.JSONDecodeError, AttributeError):
        missing += 1

with open("all_leaf_paths.txt", "w") as f:
    f.write(f"{'path':<80}  {'count':>8}  {'pct':>7}\n")
    f.write("-" * 100 + "\n")
    for p, count in sorted(counter.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        f.write(f"{p:<80}  {count:>8,}  {pct:>6.1f}%\n")

print(f"Total unique paths: {len(counter):,}")
print(f"Rows missing/unparseable: {missing:,}")
print(f"Saved to all_leaf_paths.txt")

