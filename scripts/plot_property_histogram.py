#!/usr/bin/env python3
"""Histogram of any scalar property column from a metadata parquet.

Generalizes the old plot_clean_bandgap_histogram.py: the property to plot is a
parameter (--column), so it works for band_gap, volume, density, or any other
first-order (directly-read) scalar column in the file.

Two panels:
  - left:  full data range, log y (survives heavy tails / zero spikes)
  - right: zoom to the central [--zoom-lo, --zoom-hi] percentile range, linear y

Usage:
    # band gap from the MP metadata
    python scripts/plot_property_histogram.py --column band_gap

    # a first-order property (cell volume)
    python scripts/plot_property_histogram.py --column volume

    # any other file / column
    python scripts/plot_property_histogram.py --file metadata.parquet --column density
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# properties that want band-gap-style threshold markers on the zoom panel
BANDGAP_THRESHOLDS = (0.05, 0.5, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="metadata_mp.parquet",
                   help="Parquet file to read the column from")
    p.add_argument("--column", default="band_gap",
                   help="Scalar property column to histogram")
    p.add_argument("--out-dir", default="analysis")
    p.add_argument("--bins", type=int, default=120)
    p.add_argument("--zoom-lo", type=float, default=0.5,
                   help="Lower percentile for the zoom panel (0-100)")
    p.add_argument("--zoom-hi", type=float, default=99.0,
                   help="Upper percentile for the zoom panel (0-100)")
    p.add_argument("--zoom-logy", choices=("auto", "on", "off"), default="auto",
                   help="Zoom-panel y-scale. 'auto' uses log when one bin towers "
                        "over the rest (e.g. band_gap's spike at 0), else linear.")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.file, columns=[args.column])

    # Robust to nulls: NA/None and any non-numeric entry -> NaN, drop ±inf too,
    # then drop everything missing. Report how many rows were discarded so a
    # sparsely-populated column is obvious rather than silently plotted.
    n_total = len(df)
    v = pd.to_numeric(df[args.column], errors="coerce")
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    n_dropped = n_total - len(v)
    if v.empty:
        raise SystemExit(f"Column '{args.column}' has no numeric non-null values "
                         f"({n_total:,} rows, all null/non-numeric).")

    is_bandgap = "band_gap" in args.column.lower()
    print(f"{args.column}: n={len(v):,}  (dropped {n_dropped:,} null/non-numeric of "
          f"{n_total:,})  min={v.min():.4g}  median={v.median():.4g}  "
          f"mean={v.mean():.4g}  max={v.max():.4g}  zeros={int((v == 0).sum()):,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: full range, log y
    axes[0].hist(v, bins=args.bins, range=(v.min(), v.max()),
                 color="steelblue", edgecolor="none")
    axes[0].set_yscale("log")
    axes[0].set_title(f"{args.column} (full range, log y)\nn={len(v):,}  "
                      f"[{Path(args.file).name}]")
    axes[0].set_xlabel(args.column); axes[0].set_ylabel("count (log)")

    # right: zoom to central percentile band. Pick the y-scale from the data so a
    # dominating spike (band_gap's zeros) doesn't flatten the rest into an invisible
    # carpet, while a well-spread property (volume) still gets a readable linear axis.
    lo, hi = v.quantile(args.zoom_lo / 100), v.quantile(args.zoom_hi / 100)
    vz = v[(v >= lo) & (v <= hi)]
    counts, _ = np.histogram(vz, bins=args.bins, range=(lo, hi))
    if args.zoom_logy == "auto":
        pos = counts[counts > 0]
        use_log = pos.size > 0 and counts.max() > 30 * np.median(pos)
    else:
        use_log = args.zoom_logy == "on"

    axes[1].hist(vz, bins=args.bins, range=(lo, hi),
                 color="darkorange", edgecolor="none")
    if use_log:
        axes[1].set_yscale("log")
    axes[1].set_title(f"Zoom {args.zoom_lo:g}-{args.zoom_hi:g} percentile "
                      f"([{lo:.3g}, {hi:.3g}]){', log y' if use_log else ''}")
    axes[1].set_xlabel(args.column); axes[1].set_ylabel("count (log)" if use_log else "count")

    if is_bandgap:
        for thr in BANDGAP_THRESHOLDS:
            if lo <= thr <= hi:
                axes[1].axvline(thr, color="k", ls="--", lw=0.8)
                axes[1].text(thr, axes[1].get_ylim()[1] * 0.95, f"{thr}",
                             rotation=90, fontsize=8, va="top")

    plt.tight_layout()
    png = out / f"{Path(args.file).stem}_{args.column}_histogram.png"
    plt.savefig(png, dpi=130)
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
