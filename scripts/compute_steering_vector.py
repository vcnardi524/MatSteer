#!/usr/bin/env python3
"""
Compute a steering vector for an arbitrary scalar property (default: clean band gap).

The vector is the normalized difference of embedding means between a high-value and a
low-value set of the chosen property:

  Negative set (low)   : property <= --low
  Positive set (high)  : property >= --high
  steering_vector = normalize(mean(high) - mean(low))     # points low -> high

--property picks the metadata column; --metadata / --dataset / --id-col pick the file
it lives in, the embeddings subdir it joins, and the join key:
  * band gap (NOMAD, default): metadata.parquet  / dataset v1_all / id-col id
  * MP properties:             metadata_mp.parquet / dataset v1_mp  / id-col material_id
The metadata id column is joined against the embedding `id`, so it must match (the MP
material_id is stored MP_-prefixed to line up with the v1_mp embedding ids).

Output: steering_vectors/<name>/layer{N}.parquet   (one dir per property)
  columns: property, low_thresh, high_thresh, n_low, n_high, raw_norm, steering_vector
--name defaults to 'bandgap' for the default band-gap column, else a slug of --property.

The default property is NOMAD's clean band_gap.value in eV (min across spin channels),
not the corrupt LUMO-HOMO subtraction. Its default thresholds (0.05 / 1.0 eV) select
metals vs insulators; for any other property pass --low/--high in that property's units.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_embeddings

DEFAULT_PROPERTY = "dos_electronic.band_gap"  # clean band gap (eV); == electronic.band_gap


def slug(name: str) -> str:
    """Filesystem-safe dir label from a property column, e.g.
    'dos_electronic.band_gap' -> 'dos_electronic_band_gap'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default=DEFAULT_PROPERTY,
                    help="Metadata column to steer on (default: clean band gap)")
    ap.add_argument("--name", default=None,
                    help="Output subdir under steering_vectors/ (default: 'bandgap' for "
                         "the default band-gap column, else a slug of --property)")
    ap.add_argument("--metadata", default="metadata.parquet",
                    help="Metadata parquet holding --property")
    ap.add_argument("--dataset", default="v1_all",
                    help="Embeddings subdir under embeddings/ (v1_all or v1_mp)")
    ap.add_argument("--id-col", default="id",
                    help="Metadata join key against the embedding id (e.g. material_id for MP)")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--low", type=float, default=0.05,
                    help="Low/negative-set threshold, in the property's units")
    ap.add_argument("--high", type=float, default=1.0,
                    help="High/positive-set threshold, in the property's units")
    ap.add_argument("--pct", type=float, default=None,
                    help="Percentile bucket size: low set = bottom PCT%%, high set = "
                         "top PCT%%. Thresholds computed on the joined data; overrides "
                         "--low/--high. E.g. --pct 10 -> below p10 vs above p90.")
    args = ap.parse_args()

    prop = args.property
    name = args.name or ("bandgap" if prop == DEFAULT_PROPERTY else slug(prop))

    print(f"Loading '{prop}' from {args.metadata} ...")
    meta = pd.read_parquet(args.metadata, columns=[args.id_col, prop])
    meta = meta.dropna(subset=[prop]).copy()
    meta["val"] = pd.to_numeric(meta[prop], errors="coerce")
    meta = meta.dropna(subset=["val"])
    print(f"  {len(meta):,} entries with a numeric '{prop}'")

    print(f"Loading layer-{args.layer} embeddings (dataset={args.dataset}) ...")
    emb = load_embeddings(args.layer, dataset=args.dataset)
    join = meta[[args.id_col, "val"]].rename(columns={args.id_col: "id"})
    df = emb.merge(join, on="id", how="inner")
    print(f"  joined: {len(df):,}")
    if df.empty:
        raise SystemExit(
            f"No id overlap between embeddings/{args.dataset} and {args.metadata} "
            f"on '{args.id_col}'. Check that dataset/id-col match the property's source.")

    X = np.vstack(df["embedding"].values).astype(np.float32)
    vals = df["val"].values

    if args.pct is not None:
        low_thresh = float(np.percentile(vals, args.pct))
        high_thresh = float(np.percentile(vals, 100 - args.pct))
        print(f"  percentile buckets: bottom {args.pct:g}% (<= p{args.pct:g}={low_thresh:.4g}) "
              f"vs top {args.pct:g}% (>= p{100-args.pct:g}={high_thresh:.4g})")
    else:
        low_thresh, high_thresh = args.low, args.high

    mask_low = vals <= low_thresh
    mask_high = vals >= high_thresh
    n_low, n_high = int(mask_low.sum()), int(mask_high.sum())
    print(f"  low  ({prop} <= {low_thresh:.4g}): {n_low:,}")
    print(f"  high ({prop} >= {high_thresh:.4g}): {n_high:,}")
    if n_high < 50 or n_low < 50:
        print("  WARNING: a class is very small; consider adjusting --low/--high")
    if n_high == 0 or n_low == 0:
        raise SystemExit("A class is empty — adjust --low/--high for this property.")

    steer = X[mask_high].mean(0) - X[mask_low].mean(0)
    raw_norm = float(np.linalg.norm(steer))
    steer = steer / raw_norm
    print(f"  raw mean-diff norm = {raw_norm:.3f}")

    out_dir = Path("steering_vectors") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer{args.layer}.parquet"
    pd.DataFrame([{
        "property": prop,
        "low_thresh": low_thresh,
        "high_thresh": high_thresh,
        "pct": args.pct,
        "n_low": n_low,
        "n_high": n_high,
        "raw_norm": raw_norm,
        "steering_vector": steer.tolist(),
    }]).to_parquet(out_path, index=False)
    print(f"Wrote {out_path}  (property='{prop}', name='{name}')")


if __name__ == "__main__":
    main()
