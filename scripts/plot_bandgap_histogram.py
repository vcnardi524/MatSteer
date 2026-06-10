#!/usr/bin/env python3
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "preparsed_metadata_nomad.parquet"

print(f"Loading {path} ...")
df = pd.read_parquet(path, columns=["results"])
print(f"Rows: {len(df):,}")

J_TO_EV = 6.2415e18

gaps = []
for results_str in df["results"]:
    try:
        r = json.loads(results_str)
        bg = r["properties"]["electronic"]["dos_electronic"]["band_gap"]
        if isinstance(bg, list):
            bg = bg[0]
        lo = bg.get("energy_lowest_unoccupied")
        hi = bg.get("energy_highest_occupied")
        if lo is not None and hi is not None:
            gap_ev = (float(lo) - float(hi)) * J_TO_EV
            gaps.append(gap_ev)
    except (KeyError, TypeError, IndexError, json.JSONDecodeError):
        pass

gaps = np.array(gaps)
print(f"Extracted band gaps: {len(gaps):,}")
print(f"  Positive (insulator/semiconductor): {(gaps > 0).sum():,}")
print(f"  Zero or negative (metal):           {(gaps <= 0).sum():,}")
print(f"  Min: {gaps.min():.4f} eV")
print(f"  Max: {gaps.max():.4f} eV")
print(f"  Mean (all):     {gaps.mean():.4f} eV")
print(f"  Mean (gap > 0): {gaps[gaps > 0].mean():.4f} eV")

pos = gaps[gaps > 0]
clipped = pos[pos <= 12]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# All values, log y
axes[0].hist(gaps, bins=300, color="steelblue", edgecolor="none")
axes[0].set_yscale("log")
axes[0].set_xlabel("Band gap (eV)")
axes[0].set_ylabel("Count (log scale)")
axes[0].set_title(f"All values — log y (n={len(gaps):,})")
axes[0].axvline(0, color="red", lw=1, linestyle="--")

# Positive only, log y
axes[1].hist(pos, bins=300, color="darkorange", edgecolor="none")
axes[1].set_yscale("log")
axes[1].set_xlabel("Band gap (eV)")
axes[1].set_ylabel("Count (log scale)")
axes[1].set_title(f"Positive only — log y (n={len(pos):,})")

# Clipped to 0–12 eV, linear
axes[2].hist(clipped, bins=200, color="seagreen", edgecolor="none")
axes[2].set_xlabel("Band gap (eV)")
axes[2].set_ylabel("Count")
axes[2].set_title(f"Clipped 0–12 eV (n={len(clipped):,}, {len(clipped)/len(pos)*100:.1f}% of positives)")
axes[2].set_xlim(0, 12)

plt.tight_layout()
out = "bandgap_histogram.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
