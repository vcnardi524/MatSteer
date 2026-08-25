#!/usr/bin/env python3
"""
Top-K PCA basis of the layer-L residual stream, fitted over the training corpus.

This is the subspace the pca_centroid steering method works in: at generation time a
hidden state is projected down to K coordinates, moved toward a target centroid there,
and mapped back, so only the K principal directions of the training distribution are
touched and the rest of the residual stream passes through untouched.

The basis is a property of the model's activations, not of any property being steered,
so it is fitted once per layer and shared by every target centroid.

Fitted with IncrementalPCA over the parquet row groups -- the full layer-14 train set is
2.05M x 1024 floats (8.4 GB), too big to hold as one array on most of these nodes.

Output: steering_vectors/pca_centroid/pca_layer{L}_k{K}.parquet   (single row)
  columns: layer, dataset, variant, partition, k, n_samples,
           mean (1024), components (K*1024, row-major), explained_variance_ratio (K)

Usage:
    python scripts/steering/compute_pca_basis.py --layer 14 --k 64 --partition train
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import embeddings_paths, load_split_index, add_partition_args

METHOD_DIR = Path("steering_vectors") / "pca_centroid"


def embedding_files(layer: int, dataset: str, variant: str) -> list[Path]:
    """The consolidated parquet if it exists, else the checkpoint shards."""
    single, ckpt = embeddings_paths(layer, dataset, variant)
    if single.exists():
        return [single]
    files = sorted(ckpt.glob("checkpoint_*.parquet")) + sorted(ckpt.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No embeddings for layer {layer} at {single} or {ckpt}/")
    return files


def stream_batches(files: list[Path], keep_ids: set | None, batch_size: int):
    """Yield (batch_size, 1024) float32 arrays of embeddings, filtered to keep_ids."""
    buf, n_buf = [], 0
    for path in files:
        for rb in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                    columns=["id", "embedding"]):
            df = rb.to_pandas()
            if keep_ids is not None:
                df = df[df["id"].isin(keep_ids)]
            if df.empty:
                continue
            buf.append(np.vstack(df["embedding"].to_numpy()).astype(np.float32))
            n_buf += len(df)
            if n_buf >= batch_size:
                yield np.concatenate(buf)
                buf, n_buf = [], 0
    if buf:
        yield np.concatenate(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=64, help="Number of principal directions to keep")
    add_partition_args(ap)   # --dataset / --variant / --partition
    ap.add_argument("--batch-size", type=int, default=50_000,
                    help="Rows per IncrementalPCA.partial_fit; must exceed --k")
    args = ap.parse_args()

    if args.batch_size <= args.k:
        raise SystemExit(f"--batch-size ({args.batch_size}) must exceed --k ({args.k})")

    keep_ids = None
    if args.partition != "all":
        splits = load_split_index()
        keep_ids = set(splits.loc[splits["split"] == args.partition, "id"])
        print(f"Partition '{args.partition}': {len(keep_ids):,} ids")

    files = embedding_files(args.layer, args.dataset, args.variant)
    print(f"Fitting IncrementalPCA(k={args.k}) over layer {args.layer} "
          f"[{args.dataset}/{args.variant}/{args.partition}] from {len(files)} file(s) ...")

    pca = IncrementalPCA(n_components=args.k)
    n_seen = 0
    for i, batch in enumerate(stream_batches(files, keep_ids, args.batch_size)):
        if len(batch) < args.k:
            print(f"  skipping final batch of {len(batch)} rows (< k)")
            break
        pca.partial_fit(batch)
        n_seen += len(batch)
        print(f"  batch {i+1}: {n_seen:,} rows", flush=True)

    if n_seen == 0:
        raise SystemExit("No embeddings matched -- check --dataset/--variant/--partition.")

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"\nFitted on {n_seen:,} rows. Top-{args.k} directions explain "
          f"{cum[-1]:.1%} of the variance.")
    for i in sorted({i for i in (0, 3, 7, 15, 31, args.k - 1) if i < args.k}):
        print(f"  pc{i+1:<3} {evr[i]:7.3%}   cumulative {cum[i]:6.1%}")

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METHOD_DIR / f"pca_layer{args.layer}_k{args.k}.parquet"
    pd.DataFrame([{
        "layer": args.layer,
        "dataset": args.dataset,
        "variant": args.variant,
        "partition": args.partition,
        "k": args.k,
        "n_samples": n_seen,
        "mean": pca.mean_.astype(np.float32).tolist(),
        "components": pca.components_.astype(np.float32).ravel().tolist(),
        "explained_variance_ratio": evr.astype(np.float32).tolist(),
    }]).to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
