#!/usr/bin/env python3
"""
Compute band gap steering vectors from CrystaLLM embeddings.

For each percentile interval p in [5, 10, 15, 20, 25]:
  - Negative set: materials with band_gap_ev <= bottom_max (bottom p%)
  - Positive set: materials with band_gap_ev >= top_min    (top p%)
  - steering_vector = normalize(mean(positive) - mean(negative))

Output: steering_vectors/bandgap_layer{N}.parquet
  columns: percentile, bottom_max_ev, top_min_ev, n_low, n_high, steering_vector

Usage:
    python compute_bandgap_steering.py [--layer 14] [--thresholds bandgap_percentile_stats.csv]
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


J_TO_EV = 6.2415e18


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
from utils import load_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--dataset", default="v1_all",
                        help="Embeddings subdir under embeddings/ (v1_all or v1_mp)")
    args = parser.parse_args()

    percentiles = [5, 10, 15, 20, 25]

    print("Loading metadata ...")
    meta = pd.read_parquet(
        "metadata.parquet",
        columns=["id", "energy_lowest_unoccupied", "energy_highest_occupied"]
    )
    meta = meta[meta["energy_lowest_unoccupied"].notna() & meta["energy_highest_occupied"].notna()].copy()
    meta["band_gap_ev"] = (meta["energy_lowest_unoccupied"] - meta["energy_highest_occupied"]) * J_TO_EV
    print(f"  Entries with band gap: {len(meta):,}")

    emb = load_embeddings(args.layer, dataset=args.dataset)
    print(f"  Embeddings: {len(emb):,}")

    df = emb.merge(meta[["id", "band_gap_ev"]], on="id", how="inner")
    print(f"  After join: {len(df):,}")

    X = np.vstack(df["embedding"].values)
    gaps = df["band_gap_ev"].values

    rows = []
    for p in percentiles:
        bottom_max = np.percentile(gaps, p)
        top_min    = np.percentile(gaps, 100 - p)

        mask_low  = gaps <= bottom_max
        mask_high = gaps >= top_min

        n_low  = mask_low.sum()
        n_high = mask_high.sum()
        print(f"  p={p}%  low={n_low:,} (gap <= {bottom_max:.6f} eV)  high={n_high:,} (gap >= {top_min:.6f} eV)")

        low_centroid  = X[mask_low].mean(axis=0)
        high_centroid = X[mask_high].mean(axis=0)

        steer = high_centroid - low_centroid
        norm = np.linalg.norm(steer)
        if norm == 0:
            print(f"    WARNING: zero norm at p={p}%, skipping")
            continue
        steer = steer / norm

        rows.append({
            "percentile":     p,
            "bottom_max_ev":  bottom_max,
            "top_min_ev":     top_min,
            "n_low":          int(n_low),
            "n_high":         int(n_high),
            "steering_vector": steer.tolist(),
        })
        print(f"    -> steering vector norm={np.linalg.norm(steer):.4f}, dim={steer.shape[0]}")

    out_dir = Path("steering_vectors")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"bandgap_layer{args.layer}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"\nSaved {len(rows)} steering vectors to {out_path}")


if __name__ == "__main__":
    main()
