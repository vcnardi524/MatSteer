#!/usr/bin/env python3
import numpy as np
import pandas as pd

J_TO_EV = 6.2415e18

df = pd.read_parquet("metadata.parquet", columns=["energy_lowest_unoccupied", "energy_highest_occupied"])
df = df.dropna()
gaps = (df["energy_lowest_unoccupied"] - df["energy_highest_occupied"]) * J_TO_EV
gaps = gaps.values

print(f"Total entries with band gap: {len(gaps):,}")

percentiles = [5, 10, 15, 20, 25]
rows = []
for p in percentiles:
    bottom_max = np.percentile(gaps, p)
    bottom = gaps[gaps <= bottom_max]
    bottom_mean = bottom.mean()
    top_min = np.percentile(gaps, 100 - p)
    top = gaps[gaps >= top_min]
    top_mean = top.mean()

    # middle     = gaps[(gaps >= bottom_max) & (gaps <= top_min)]
    # mean_mid   = middle.mean() if len(middle) > 0 else np.nan
    rows.append({
        "x=y (%)":        p,
        "top_min_ev":     round(top_min, 6),
        "bottom_max_ev":  round(bottom_max, 6),
        "top_mean":       top_mean,
        "bottom_mean":    bottom_mean, 
        "n_top":          len(top),
        "n_bottom":       len(bottom),
    })

results = pd.DataFrame(rows)
print(results.to_string(index=False))

results.to_csv("bandgap_percentile_stats.csv", index=False)
print("\nSaved bandgap_percentile_stats.csv")
