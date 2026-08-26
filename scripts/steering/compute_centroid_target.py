#!/usr/bin/env python3
"""
Target centroid for pca_centroid steering, in the layer-L PCA subspace.

The linear method builds a direction from a high class minus a low class. This one has
no low class: it takes the structures whose property sits closest to a chosen target,
averages their layer-L embeddings, and stores that centroid's coordinates in the PCA
subspace. Generation then interpolates toward it, leaving the rest of the residual
stream to supply everything the centroid does not specify.

Class membership is "the --class-size structures nearest --target", so the class is a
band around the target rather than a threshold tail, and its width is reported.

Output: steering_vectors/pca_centroid/<property>/layer{L}_k{K}_target{T}.parquet
  columns: property, target, class_size, class_lo, class_hi, class_mean,
           layer, k, partition, centroid (1024), centroid_pca (K)

Usage:
    python scripts/steering/compute_centroid_target.py --target 30 --partition train
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py
# sys.path[0] is this script's own dir, so its neighbour imports directly.
from compute_pca_basis import METHOD_DIR, embedding_files, stream_batches
from utils import load_split_index, add_partition_args

DEFAULT_LABELS = "density_atomic_v1.parquet"
DEFAULT_PROPERTY = "density_atomic"


def load_pca(layer: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean (1024,), components (k, 1024)) from the shared basis file."""
    path = METHOD_DIR / f"pca_layer{layer}_k{k}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No PCA basis at {path} — run compute_pca_basis.py first")
    row = pd.read_parquet(path).iloc[0]
    mean = np.asarray(row["mean"], dtype=np.float32)
    comps = np.asarray(row["components"], dtype=np.float32).reshape(int(row["k"]), -1)
    print(f"PCA basis {path}: k={comps.shape[0]}, fitted on {int(row['n_samples']):,} "
          f"rows of {row['partition']}")
    return mean, comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, required=True,
                    help="Target property value the centroid is built around")
    ap.add_argument("--property", default=DEFAULT_PROPERTY,
                    help="Label column in --labels")
    ap.add_argument("--name", default=None,
                    help="Output subdir under steering_vectors/pca_centroid/ "
                         "(default: --property). Give it when the column name is not a "
                         "good directory name, e.g. dos_electronic.band_gap -> bandgap.")
    ap.add_argument("--labels", default=DEFAULT_LABELS,
                    help="Parquet with an `id` column and --property")
    ap.add_argument("--class-size", type=int, default=100_000,
                    help="How many nearest-to-target structures form the class")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=64)
    add_partition_args(ap)   # --dataset / --variant / --partition
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument("--save-bank", action="store_true",
                    help="Also write <stem>_bank.parquet: every class member's subspace "
                         "coordinate. pca_local steering searches it for the members "
                         "nearest each prompt, instead of using one global centroid.")
    args = ap.parse_args()

    mean, comps = load_pca(args.layer, args.k)

    labels = pd.read_parquet(args.labels, columns=["id", args.property])
    labels = labels.dropna(subset=[args.property])
    if args.partition != "all":
        splits = load_split_index()
        keep = set(splits.loc[splits["split"] == args.partition, "id"])
        labels = labels[labels["id"].isin(keep)]
    print(f"{len(labels):,} labelled structures in partition '{args.partition}'")

    if len(labels) < args.class_size:
        print(f"  ! only {len(labels):,} available; class is the whole partition")
    cls = labels.reindex(
        (labels[args.property] - args.target).abs().sort_values().index[:args.class_size])
    vals = cls[args.property]
    print(f"Class: {len(cls):,} nearest to {args.target:g}  "
          f"range [{vals.min():.3f}, {vals.max():.3f}]  mean {vals.mean():.3f}")

    class_ids = set(cls["id"])
    files = embedding_files(args.layer, args.dataset, args.variant)
    print(f"Averaging layer-{args.layer} embeddings over the class ...")
    total, n = np.zeros(len(mean), dtype=np.float64), 0
    bank = [] if args.save_bank else None
    for batch in stream_batches(files, class_ids, args.batch_size):
        total += batch.sum(0)
        n += len(batch)
        if bank is not None:
            # keep each member's subspace coordinate, not the 1024-d vector: that is
            # all a local neighbourhood search needs, and it is 16x smaller.
            bank.append(((batch - mean) @ comps.T).astype(np.float32))
        print(f"  {n:,} / {len(cls):,}", flush=True)
    if n != len(cls):
        print(f"  ! {len(cls) - n:,} class ids had no embedding and were dropped")
    if n == 0:
        raise SystemExit("No class embeddings found — check --dataset/--variant.")

    centroid = (total / n).astype(np.float32)
    centroid_pca = (centroid - mean) @ comps.T
    print(f"Centroid: |c|={np.linalg.norm(centroid):.3f}, "
          f"|c - mean|={np.linalg.norm(centroid - mean):.3f}, "
          f"of which {np.linalg.norm(centroid_pca):.3f} lies in the top-{args.k} subspace")

    out_dir = METHOD_DIR / (args.name or args.property)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer{args.layer}_k{args.k}_target{args.target:g}.parquet"
    pd.DataFrame([{
        "property": args.property,
        "name": args.name or args.property,
        "target": args.target,
        "class_size": n,
        "class_lo": float(vals.min()),
        "class_hi": float(vals.max()),
        "class_mean": float(vals.mean()),
        "layer": args.layer,
        "k": args.k,
        "partition": args.partition,
        "centroid": centroid.tolist(),
        "centroid_pca": centroid_pca.astype(np.float32).tolist(),
    }]).to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")

    if bank is not None:
        Z = np.concatenate(bank)
        bank_path = out_path.with_name(out_path.stem + "_bank.parquet")
        pd.DataFrame({"coord": list(Z)}).to_parquet(bank_path, index=False)
        spread = np.linalg.norm(Z - centroid_pca, axis=1)
        print(f"Wrote {bank_path}  ({Z.shape[0]:,} x {Z.shape[1]})")
        print(f"  members sit a median {np.median(spread):.2f} from the global centroid "
              f"(p10 {np.percentile(spread, 10):.2f}, p90 {np.percentile(spread, 90):.2f}) "
              f"-- the class is only a single destination if this is small")


if __name__ == "__main__":
    main()
