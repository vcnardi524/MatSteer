#!/usr/bin/env python3
"""
Extract NOMAD band-gap values from the preparsed metadata JSON and histogram them.

Pulls two fields from results JSON:
  properties.electronic.band_gap.value
  properties.electronic.dos_electronic.band_gap.value
NOMAD stores energies in Joules (SI) -> convert to eV.

Output:
  analysis/nomad_bandgap_values.parquet   (id, band_gap_ev, dos_band_gap_ev)
  analysis/nomad_bandgap_histogram.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import DATASETS, PARTITIONS, filter_partition, analysis_dir

J_TO_EV = 6.241509074e18


def dig(d, path):
    for k in path.split("."):
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


def main():
    # No --variant: this reads metadata only, so the CIF variant cannot change it.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="v1_all", choices=list(DATASETS))
    ap.add_argument("--partition", required=True, choices=list(PARTITIONS))
    args = ap.parse_args()

    out = analysis_dir(args.dataset, None, args.partition)
    print("Loading preparsed NOMAD metadata ...")
    df = pd.read_parquet("preparsed_metadata_nomad.parquet", columns=["id", "results"])
    print(f"  {len(df):,} rows")
    df = filter_partition(df, args.partition)

    ids, bg, dosbg = [], [], []
    for id_, r in zip(df["id"].values, df["results"].values):
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:
                continue
        b = dig(r, "properties.electronic.band_gap.value")
        d = dig(r, "properties.electronic.dos_electronic.band_gap.value")
        if b is None and d is None:
            continue
        ids.append(id_)
        bg.append(b)
        dosbg.append(d)

    res = pd.DataFrame({"id": ids, "band_gap_ev": bg, "dos_band_gap_ev": dosbg})
    # convert J -> eV
    for c in ["band_gap_ev", "dos_band_gap_ev"]:
        res[c] = pd.to_numeric(res[c], errors="coerce") * J_TO_EV
    res.to_parquet(out / "nomad_bandgap_values.parquet", index=False)
    print(f"  extracted {len(res):,} rows with at least one band gap")

    for c in ["band_gap_ev", "dos_band_gap_ev"]:
        v = res[c].dropna()
        print(f"\n{c}:  n={len(v):,}")
        print(f"  min={v.min():.3f}  max={v.max():.3f}  mean={v.mean():.3f}  median={v.median():.3f}")
        print(f"  ==0 : {(v==0).mean():.1%}   >0 : {(v>0).mean():.1%}   <0 : {(v<0).mean():.1%}")
        for hi in [1, 2, 5]:
            print(f"  in (0,{hi}] eV: {((v>0)&(v<=hi)).sum():,}")

    # histogram (clip to a sensible physical window for visibility)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, c, title in zip(
        axes,
        ["band_gap_ev", "dos_band_gap_ev"],
        ["properties.electronic.band_gap", "dos_electronic.band_gap"],
    ):
        v = res[c].dropna()
        vc = v[(v >= 0) & (v <= 10)]
        ax.hist(vc, bins=100, color="steelblue", edgecolor="none")
        ax.set_title(f"{title}\nn={len(v):,} (showing 0-10 eV: {len(vc):,})")
        ax.set_xlabel("band gap (eV)")
        ax.set_ylabel("count")
        ax.axvline(0, color="k", lw=0.5)
    plt.tight_layout()
    png = out / "nomad_bandgap_histogram.png"
    plt.savefig(png, dpi=120)
    print(f"\nSaved histogram -> {png}")
    print(f"Saved values    -> {out / 'nomad_bandgap_values.parquet'}")


if __name__ == "__main__":
    main()
