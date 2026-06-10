#!/usr/bin/env python3
"""
Compute a band-gap steering vector from CLEAN NOMAD band gaps.

Uses metadata.parquet column `dos_electronic.band_gap` (NOMAD's properly-computed
band_gap.value in eV, min across spin channels) instead of the corrupt
LUMO-HOMO subtraction. Selection is by physical eV thresholds, not percentiles:

  Negative set (metals)      : gap <= --low   (default 0.05 eV)
  Positive set (insulators)  : gap >= --high  (default 1.0 eV)
  steering_vector = normalize(mean(positive) - mean(negative))

Output: steering_vectors/bandgap_clean_layer{N}.parquet
  columns: low_thresh_ev, high_thresh_ev, n_low, n_high, raw_norm, steering_vector
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BG_COL = "dos_electronic.band_gap"   # clean value (eV); identical to electronic.band_gap


def load_embeddings(layer: int) -> pd.DataFrame:
    single = Path(f"embeddings/cif_layer{layer}.parquet")
    if single.exists():
        return pd.read_parquet(single, columns=["id", "embedding"])
    ckpt = Path(f"embeddings/cif_layer{layer}")
    files = sorted(ckpt.glob("checkpoint_*.parquet")) + sorted(ckpt.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No embeddings for layer {layer}")
    return pd.concat([pd.read_parquet(f, columns=["id", "embedding"]) for f in files],
                     ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--low", type=float, default=0.05, help="metal threshold (eV)")
    ap.add_argument("--high", type=float, default=1.0, help="insulator threshold (eV)")
    args = ap.parse_args()

    print(f"Loading metadata clean band gap ({BG_COL}) ...")
    meta = pd.read_parquet("metadata.parquet", columns=["id", BG_COL])
    meta = meta.dropna(subset=[BG_COL]).copy()
    meta["gap"] = meta[BG_COL].astype(float)
    print(f"  {len(meta):,} entries with clean band gap")

    print(f"Loading layer-{args.layer} embeddings ...")
    emb = load_embeddings(args.layer)
    df = emb.merge(meta[["id", "gap"]], on="id", how="inner")
    print(f"  joined: {len(df):,}")

    X = np.vstack(df["embedding"].values).astype(np.float32)
    gaps = df["gap"].values

    mask_low = gaps <= args.low
    mask_high = gaps >= args.high
    n_low, n_high = int(mask_low.sum()), int(mask_high.sum())
    print(f"  metals (gap <= {args.low}): {n_low:,}")
    print(f"  insulators (gap >= {args.high}): {n_high:,}")
    if n_high < 50 or n_low < 50:
        print("  WARNING: a class is very small; consider adjusting thresholds")

    steer = X[mask_high].mean(0) - X[mask_low].mean(0)
    raw_norm = float(np.linalg.norm(steer))
    steer = steer / raw_norm
    print(f"  raw mean-diff norm = {raw_norm:.3f}  (vs ~11 corrupt p5, ~2 corrupt p10)")

    # Overwrite the canonical per-layer file with the single clean vector
    # (no percentiles). steer_generate_cif.py reads row 0 directly.
    out_dir = Path("steering_vectors"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"bandgap_layer{args.layer}.parquet"
    pd.DataFrame([{
        "low_thresh_ev": args.low,
        "high_thresh_ev": args.high,
        "n_low": n_low,
        "n_high": n_high,
        "raw_norm": raw_norm,
        "steering_vector": steer.tolist(),
    }]).to_parquet(out_path, index=False)
    print(f"Overwrote {out_path} with clean metal-vs-insulator vector.")


if __name__ == "__main__":
    main()
