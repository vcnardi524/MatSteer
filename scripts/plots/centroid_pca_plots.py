#!/usr/bin/env python3
"""
Bucket a property, average the embeddings in each bucket, and look at where those
centroids sit in PCA space.

Steering toward a single class centroid does nothing (see the pca_centroid results), and
one explanation is that the property does not lie along a straight line in activation
space. This draws the thing that question is about: one centroid per property bucket,
plotted in the PCA subspace and coloured in property order. If the property is a linear
direction the centroids fall on a straight line; if it curves, bends and folds show up
here directly, and in which principal directions.

Buckets are [x, x+width) with width 1 by default, so for volume per atom that is one
centroid per A^3/atom. Buckets thinner than --min-count are dropped -- a centroid over a
handful of structures is noise, and the tails of every property here are long.

Colour is sequential (property order) with a colorbar, never categorical: the buckets are
ordered magnitudes, not identities. The centroids are joined in bucket order so the path
is readable, and each panel is one triple of principal directions.

Output: analysis/<dataset>/<variant>/<partition>/plots/centroid_pca_<property>.png
        plus the centroid coordinates as .csv

Usage:
    python scripts/plots/centroid_pca_plots.py --partition train
    python scripts/plots/centroid_pca_plots.py --partition val
    python scripts/plots/centroid_pca_plots.py --partition train \
        --property dos_electronic.band_gap --labels metadata.parquet --width 0.25
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import analysis_dir, load_split_index, add_partition_args, embeddings_paths

DEFAULT_LABELS = "density_atomic_v1.parquet"
DEFAULT_PROPERTY = "density_atomic"
# Which principal directions each panel shows. The first is the subspace almost all of
# the steering displacement lives in; the later ones say whether the property keeps
# structure past the leading directions or dissolves into noise.
DEFAULT_DIMS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]


def embedding_files(layer, dataset, variant):
    single, ckpt = embeddings_paths(layer, dataset, variant)
    if single.exists():
        return [single]
    files = sorted(ckpt.glob("checkpoint_*.parquet")) + sorted(ckpt.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No embeddings for layer {layer} at {single} or {ckpt}/")
    return files


def bucket_centroids(labels, prop, width, layer, dataset, variant, batch_size):
    """{bucket_index: (centroid, count)} accumulated by streaming the layer."""
    idx = np.floor(labels[prop].to_numpy() / width).astype(np.int64)
    bucket_of = dict(zip(labels["id"].to_numpy(), idx))

    sums, counts = {}, {}
    for path in embedding_files(layer, dataset, variant):
        for rb in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                    columns=["id", "embedding"]):
            df = rb.to_pandas()
            b = df["id"].map(bucket_of)
            df = df[b.notna()]
            if df.empty:
                continue
            b = b[b.notna()].to_numpy().astype(np.int64)
            X = np.vstack(df["embedding"].to_numpy()).astype(np.float64)
            for u in np.unique(b):
                m = b == u
                sums[u] = sums.get(u, 0) + X[m].sum(0)
                counts[u] = counts.get(u, 0) + int(m.sum())
    return sums, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default=DEFAULT_PROPERTY, help="Label column to bucket")
    ap.add_argument("--labels", default=DEFAULT_LABELS,
                    help="Parquet with `id` and --property")
    ap.add_argument("--width", type=float, default=1.0,
                    help="Bucket width in the property's own units")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--basis", choices=["corpus", "centroids"], default="corpus",
                    help="corpus: the stored training-activation PCA, the same directions "
                         "the steering happens in. centroids: refit on the centroids "
                         "themselves, which maximises the spread of the path but is not "
                         "comparable across properties.")
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="Flat list of principal directions, in triples: 0 1 2 3 4 5")
    ap.add_argument("--min-count", type=int, default=30,
                    help="Drop buckets with fewer structures than this")
    add_partition_args(ap)   # --dataset / --variant / --partition
    ap.add_argument("--batch-size", type=int, default=50_000)
    args = ap.parse_args()

    dims = DEFAULT_DIMS
    if args.dims:
        if len(args.dims) % 3:
            raise SystemExit("--dims needs a multiple of three values")
        dims = [tuple(args.dims[i:i + 3]) for i in range(0, len(args.dims), 3)]

    labels = pd.read_parquet(args.labels, columns=["id", args.property]).dropna()
    labels[args.property] = pd.to_numeric(labels[args.property], errors="coerce")
    labels = labels.dropna(subset=[args.property])
    if args.partition != "all":
        splits = load_split_index()
        keep = set(splits.loc[splits["split"] == args.partition, "id"])
        labels = labels[labels["id"].isin(keep)]
    print(f"{len(labels):,} labelled structures in partition '{args.partition}'")

    print(f"Streaming layer-{args.layer} embeddings, bucketing '{args.property}' "
          f"by {args.width:g} ...")
    sums, counts = bucket_centroids(labels, args.property, args.width, args.layer,
                                    args.dataset, args.variant, args.batch_size)
    kept = sorted(b for b, n in counts.items() if n >= args.min_count)
    dropped = len(counts) - len(kept)
    if not kept:
        raise SystemExit("No bucket met --min-count.")
    print(f"  {len(kept)} buckets kept, {dropped} dropped below {args.min_count} structures")

    C = np.vstack([sums[b] / counts[b] for b in kept])          # (n_buckets, 1024)
    lo = np.array(kept, dtype=float) * args.width
    n = np.array([counts[b] for b in kept])
    print(f"  property range {lo.min():g} to {lo.max() + args.width:g}, "
          f"{n.min():,}-{n.max():,} structures per bucket")

    if args.basis == "corpus":
        bp = f"steering_vectors/pca_centroid/pca_layer{args.layer}_k{args.k}.parquet"
        if not os.path.exists(bp):
            raise SystemExit(f"No PCA basis at {bp} -- run compute_pca_basis.py, or pass "
                             f"--basis centroids")
        row = pd.read_parquet(bp).iloc[0]
        mean = np.asarray(row["mean"], dtype=np.float64)
        comps = np.asarray(row["components"], dtype=np.float64).reshape(int(row["k"]), -1)
        evr = np.asarray(row["explained_variance_ratio"], dtype=float)
        basis_note = f"corpus PCA (layer {args.layer}, fitted on {int(row['n_samples']):,} train)"
    else:
        from sklearn.decomposition import PCA
        p = PCA(n_components=min(args.k, len(C) - 1)).fit(C)
        mean, comps, evr = p.mean_, p.components_, p.explained_variance_ratio_
        basis_note = f"PCA refitted on the {len(C)} centroids"
    Y = (C - mean) @ comps.T
    print(f"  basis: {basis_note}")

    max_dim = max(max(d) for d in dims)
    if max_dim >= Y.shape[1]:
        raise SystemExit(f"--dims asks for pc{max_dim} but the basis has {Y.shape[1]}")

    # Sequential colour: the buckets are ordered magnitudes, so one ordered ramp with a
    # colorbar. Never categorical hues -- those would imply the buckets are identities.
    cmap = plt.get_cmap("viridis")
    cvals = (lo - lo.min()) / max(lo.max() - lo.min(), 1e-9)

    ncol = min(len(dims), 2)
    nrow = int(np.ceil(len(dims) / ncol))
    fig = plt.figure(figsize=(8.5 * ncol, 7.0 * nrow))
    for i, (a, b, c) in enumerate(dims, start=1):
        ax = fig.add_subplot(nrow, ncol, i, projection="3d")
        # the path in bucket order, so a bend is visible and not just a cloud
        ax.plot(Y[:, a], Y[:, b], Y[:, c], color="0.55", lw=1.0, zorder=1)
        s = ax.scatter(Y[:, a], Y[:, b], Y[:, c], c=cvals, cmap=cmap,
                       s=np.clip(18 + 30 * n / n.max(), 18, 70),
                       edgecolor="white", linewidth=0.5, depthshade=False, zorder=2)
        ax.set_xlabel(f"pc{a}  ({evr[a]:.1%})")
        ax.set_ylabel(f"pc{b}  ({evr[b]:.1%})")
        ax.set_zlabel(f"pc{c}  ({evr[c]:.1%})")
        ax.set_title(f"pc{a} / pc{b} / pc{c}", fontsize=11)
        ax.grid(alpha=0.25)

    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                            norm=plt.Normalize(lo.min(), lo.max() + args.width)),
                      ax=fig.axes, shrink=0.55, pad=0.02)
    cb.set_label(f"{args.property}  (bucket lower edge, width {args.width:g})")

    fig.suptitle(
        f"{args.property}: bucket centroids in PCA space — {args.dataset}/{args.variant}/"
        f"{args.partition}, layer {args.layer}\n{len(kept)} buckets of width "
        f"{args.width:g}, {n.min():,}-{n.max():,} structures each, {basis_note}. "
        f"Marker size = bucket count; grey path joins buckets in property order.",
        fontsize=12.5, y=0.99)

    out_dir = analysis_dir(args.dataset, args.variant, args.partition, subdir="plots")
    stem = f"centroid_pca_{args.property.replace('.', '_')}_layer{args.layer}"
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    out = pd.DataFrame({"bucket_lo": lo, "bucket_hi": lo + args.width, "n": n})
    for j in range(min(Y.shape[1], max_dim + 1)):
        out[f"pc{j}"] = Y[:, j]
    csv = out_dir / f"{stem}.csv"
    out.to_csv(csv, index=False, float_format="%.6g")
    print(f"\nSaved {png}\nSaved {csv}")


if __name__ == "__main__":
    main()
