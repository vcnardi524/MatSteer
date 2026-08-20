#!/usr/bin/env python3
"""Histogram of the clean NOMAD band_gap.value column (log y axis).

--partition is required so each split can be eyeballed separately. There is no
--variant: this reads metadata only, never embeddings, so the CIF variant cannot
change the distribution.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import DATASETS, PARTITIONS, filter_partition, analysis_dir

COL = "dos_electronic.band_gap"  # identical to electronic.band_gap

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--dataset", default="v1_all", choices=list(DATASETS))
_p.add_argument("--partition", required=True, choices=list(PARTITIONS))
args = _p.parse_args()

out = analysis_dir(args.dataset, None, args.partition)
df = pd.read_parquet("metadata.parquet", columns=["id", COL])
df = filter_partition(df, args.partition)
v = df[COL].dropna().astype(float)
print(f"n={len(v):,}  metals(==0)={int((v==0).sum()):,}  >0={int((v>0).sum()):,}  "
      f">=0.5eV={int((v>=0.5).sum()):,}  >=1.0eV={int((v>=1.0).sum()):,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# full range, log y
axes[0].hist(v, bins=120, range=(0, 18.3), color="steelblue", edgecolor="none")
axes[0].set_yscale("log")
axes[0].set_title(f"Clean band gap (full range, log y)\nn={len(v):,}")
axes[0].set_xlabel("band gap (eV)"); axes[0].set_ylabel("count (log)")

# zoom 0-5 eV, log y
vz = v[v <= 5]
axes[1].hist(vz, bins=100, range=(0, 5), color="darkorange", edgecolor="none")
axes[1].set_yscale("log")
axes[1].set_title("Zoom 0-5 eV (log y)")
axes[1].set_xlabel("band gap (eV)"); axes[1].set_ylabel("count (log)")
for thr in (0.05, 0.5, 1.0):
    axes[1].axvline(thr, color="k", ls="--", lw=0.8)
    axes[1].text(thr, axes[1].get_ylim()[1]*0.5, f"{thr}", rotation=90, fontsize=8, va="top")

plt.tight_layout()
png = out / "clean_bandgap_histogram.png"
plt.savefig(png, dpi=130)
print(f"Saved {png}")
