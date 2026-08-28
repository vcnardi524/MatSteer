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

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
import pyarrow.parquet as pq
from utils import DEFAULT_VARIANT, embeddings_paths


def embedding_files(layer: int, dataset: str, variant: str) -> list:
    """The consolidated parquet if it exists, else the checkpoint shards."""
    single, ckpt = embeddings_paths(layer, dataset, variant)
    if single.exists():
        return [single]
    files = sorted(ckpt.glob("checkpoint_*.parquet")) + sorted(ckpt.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No embeddings for layer {layer} at {single} or {ckpt}/")
    return files


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

    join = meta[[args.id_col, "val"]].rename(columns={args.id_col: "id"})

    # Thresholds come from the labels alone, so they can be fixed before any embedding
    # is read -- which lets the class means be accumulated by streaming instead of
    # holding the layer in memory. A consolidated layer is 2.29M x 1024 floats (9.4 GB
    # raw, several times that once pandas has boxed the list column), so loading one
    # outright is what stopped this script running over every layer.
    if args.pct is not None:
        low_thresh = float(np.percentile(join["val"].to_numpy(), args.pct))
        high_thresh = float(np.percentile(join["val"].to_numpy(), 100 - args.pct))
        print(f"  percentile buckets: bottom {args.pct:g}% (<= p{args.pct:g}={low_thresh:.4g}) "
              f"vs top {args.pct:g}% (>= p{100-args.pct:g}={high_thresh:.4g})")
    else:
        low_thresh, high_thresh = args.low, args.high

    low_ids = set(join.loc[join["val"] <= low_thresh, "id"])
    high_ids = set(join.loc[join["val"] >= high_thresh, "id"])

    print(f"Streaming layer-{args.layer} embeddings (dataset={args.dataset}) ...")
    files = embedding_files(args.layer, args.dataset, DEFAULT_VARIANT)
    sums = {"low": np.zeros(1024), "high": np.zeros(1024)}
    counts = {"low": 0, "high": 0}
    for path in files:
        for rb in pq.ParquetFile(path).iter_batches(batch_size=50_000,
                                                    columns=["id", "embedding"]):
            df = rb.to_pandas()
            for side, keep in (("low", low_ids), ("high", high_ids)):
                sel = df[df["id"].isin(keep)]
                if sel.empty:
                    continue
                sums[side] += np.vstack(sel["embedding"].to_numpy()).astype(np.float64).sum(0)
                counts[side] += len(sel)

    n_low, n_high = counts["low"], counts["high"]
    print(f"  low  ({prop} <= {low_thresh:.4g}): {n_low:,}")
    print(f"  high ({prop} >= {high_thresh:.4g}): {n_high:,}")
    if n_high < 50 or n_low < 50:
        print("  WARNING: a class is very small; consider adjusting --low/--high")
    if n_high == 0 or n_low == 0:
        raise SystemExit("A class is empty — adjust --low/--high, or check that "
                         "dataset/id-col match the property's source.")

    steer = ((sums["high"] / n_high) - (sums["low"] / n_low)).astype(np.float32)
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
