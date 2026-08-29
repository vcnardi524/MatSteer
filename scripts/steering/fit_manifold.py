#!/usr/bin/env python3
"""
Fit a 1-D manifold through property-bucket centroids and save it for steering.

Bucket the property, average each bucket's embeddings, project into the PCA subspace,
and fit a smooth curve through the resulting centroids. The curve is saved as a dense
polyline that `Manifold.load()` reads at inference.

WHY A SMOOTHING FIT, NOT AN INTERPOLATING ONE
---------------------------------------------
Centroid precision varies enormously: for density at width 1 the mode bucket rests on
161,051 structures and the tail ones on 30, a ~70x range in sqrt(n). We already measured
that the sparse tail is what inflates the apparent curvature -- the count-weighted
turning angle is 28 degrees against an unweighted median of 78. And the buckets are not
even contiguous: density runs 3..83 then jumps to 89, and that last bucket (n=40) swings
pc1 from 6.6 to 11.8. An interpolating spline would chase it.

So each subspace dimension is fitted against arc length with weights ~ sqrt(count). The
reference implementation this borrows from has a global smoothness but no per-point
weights; with support this uneven, weights are the part that matters.

Usage:
    python scripts/steering/fit_manifold.py --property density_atomic --partition train
    python scripts/steering/fit_manifold.py --property band_gap --labels metadata_mp.parquet \
        --id-col material_id --dataset v1_mp --partition all --width 0.1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from manifold import Manifold, bucket_centroids
from utils import load_split_index, add_partition_args

OUT_DIR = "steering_vectors/manifolds"


def load_pca(layer, k):
    """(mean (1024,), components (k, 1024)) from the shared basis file."""
    path = f"steering_vectors/pca_centroid/pca_layer{layer}_k{k}.parquet"
    if not os.path.exists(path):
        raise SystemExit(f"No PCA basis at {path} -- run compute_pca_basis.py first")
    row = pd.read_parquet(path).iloc[0]
    return (np.asarray(row["mean"], dtype=np.float64),
            np.asarray(row["components"], dtype=np.float64).reshape(int(row["k"]), -1))


def fit_curve(Y, prop, counts, n_samples, smoothing):
    """Smooth Y (n_buckets, k) against arc length; return (samples, arc, prop) resampled.

    Arc length is cumulative distance between consecutive centroids in the subspace, so a
    step of a given size means the same distance everywhere along the curve -- which the
    property value does not, the buckets being dense at the mode and sparse in the tails.
    """
    from scipy.interpolate import make_smoothing_spline

    order = np.argsort(prop)
    Y, prop, counts = Y[order], prop[order], counts[order]

    seg = np.linalg.norm(np.diff(Y, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])

    # weight by sqrt(count): the mode centroid pins the curve, the tail barely moves it
    w = np.sqrt(counts.astype(np.float64))
    w = w / w.mean()

    grid = np.linspace(arc[0], arc[-1], n_samples)
    smoothed = np.empty((n_samples, Y.shape[1]))
    for j in range(Y.shape[1]):
        spline = make_smoothing_spline(arc, Y[:, j], w=w, lam=smoothing)
        smoothed[:, j] = spline(grid)

    prop_grid = np.interp(grid, arc, prop)

    # arc length of the SMOOTHED curve, so `decode` is parameterised by real distance
    seg2 = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    arc2 = np.concatenate([[0.0], np.cumsum(seg2)])
    return smoothed, arc2, prop_grid, arc[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="density_atomic")
    ap.add_argument("--labels", default="density_atomic_v1.parquet")
    ap.add_argument("--id-col", default="id",
                    help="Join key in --labels; metadata_mp.parquet needs material_id")
    ap.add_argument("--width", type=float, default=1.0, help="Bucket width")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--min-value", type=float, default=None,
                    help="Drop structures below this property value before bucketing")
    ap.add_argument("--max-value", type=float, default=None,
                    help="Drop structures above this. For a steerable curve, trim the "
                         "sparse tail: on density it holds ~1%% of structures but over "
                         "half the arc length, so a step mostly traverses empty space.")
    ap.add_argument("--min-count", type=int, default=30,
                    help="Drop buckets with fewer structures; their centroids are noise")
    ap.add_argument("--n-samples", type=int, default=2048,
                    help="Points along the saved polyline")
    ap.add_argument("--smoothing", type=float, default=None,
                    help="Spline lambda. None lets scipy pick by cross-validation.")
    add_partition_args(ap)
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels = pd.read_parquet(args.labels, columns=[args.id_col, args.property]).dropna()
    if args.id_col != "id":
        labels = labels.rename(columns={args.id_col: "id"})
    labels[args.property] = pd.to_numeric(labels[args.property], errors="coerce")
    labels = labels.dropna(subset=[args.property])
    if args.min_value is not None:
        labels = labels[labels[args.property] >= args.min_value]
    if args.max_value is not None:
        labels = labels[labels[args.property] <= args.max_value]
    if args.partition != "all":
        splits = load_split_index()
        keep = set(splits.loc[splits["split"] == args.partition, "id"])
        labels = labels[labels["id"].isin(keep)]
    print(f"{len(labels):,} labelled structures in partition '{args.partition}'")

    print(f"Streaming layer-{args.layer} embeddings, bucketing by {args.width:g} ...")
    sums, counts, _, _ = bucket_centroids(labels, args.property, args.width, args.layer,
                                          args.dataset, args.variant, args.batch_size)
    kept = sorted(b for b, n in counts.items() if n >= args.min_count)
    if len(kept) < 4:
        raise SystemExit(f"Only {len(kept)} buckets survived --min-count; need >= 4")
    C = np.vstack([sums[b] / counts[b] for b in kept])
    prop = np.array(kept, float) * args.width + args.width / 2      # bucket centre
    n = np.array([counts[b] for b in kept])
    print(f"  {len(kept)} buckets, {n.min():,}-{n.max():,} structures each "
          f"(sqrt-weight range {np.sqrt(n.min()):.0f}-{np.sqrt(n.max()):.0f})")

    mean, comps = load_pca(args.layer, args.k)
    Y = (C - mean) @ comps.T
    print(f"  projected into the layer-{args.layer} k={args.k} subspace")

    samples, arc, prop_grid, raw_len = fit_curve(Y, prop, n, args.n_samples, args.smoothing)
    print(f"  polyline arc length {arc[-1]:.2f} (through the raw centroids: {raw_len:.2f})")

    m = Manifold(samples, arc, prop_grid, meta={
        "property": args.property, "layer": args.layer, "k": args.k,
        "width": args.width, "partition": args.partition, "dataset": args.dataset,
        "n_buckets": len(kept), "min_count": args.min_count,
    })

    # How much of the data does the curve actually explain? Compare the residual against
    # the scatter of the centroids themselves -- if the curve leaves the residual at the
    # scatter, it explains nothing and steering along it is pointless.
    import torch
    Yt = torch.tensor(Y, dtype=torch.float32)
    u, residual = m.encode(Yt)
    resid_norm = residual.norm(dim=1)
    spread = float(np.linalg.norm(Y - Y.mean(0), axis=1).mean())
    line = np.polyfit(np.arange(len(Y)), Y, 1)
    line_fit = (np.arange(len(Y))[:, None] * line[0] + line[1])
    line_resid = float(np.linalg.norm(Y - line_fit, axis=1).mean())
    print(f"\n  centroid spread about their own mean : {spread:8.3f}")
    print(f"  mean residual, straight-line fit      : {line_resid:8.3f}")
    print(f"  mean residual, fitted curve           : {float(resid_norm.mean()):8.3f}"
          f"   (max {float(resid_norm.max()):.3f})")
    print(f"  curve explains {1 - float(resid_norm.mean())/spread:.1%} of the centroid "
          f"spread; a straight line explains {1 - line_resid/spread:.1%}")

    # Monotone, not linear: arc length and property value are deliberately NOT
    # proportional -- that is the reason for parameterising by arc at all. So the check
    # is rank correlation. A Pearson r below 1 here is the curve bending, not a fault.
    from scipy.stats import spearmanr
    rho = float(spearmanr(prop, u.flatten().numpy()).statistic)
    print(f"  arc vs property, rank correlation      : {rho:+.4f}"
          + ("" if rho > 0.999 else "   <- should be +1.0; the curve doubles back"))

    # How the curve's length is distributed over the data. If most of the arc sits where
    # almost no structures do, a step in arc length mostly moves through empty tail.
    dense_hi = float(np.percentile(np.repeat(prop, n // max(n.min(), 1)), 99))
    frac_arc = float((m.arc[m.prop > dense_hi].max() - m.arc[m.prop > dense_hi].min())
                     / m.arc[-1]) if (m.prop > dense_hi).any() else 0.0
    frac_n = float(n[prop > dense_hi].sum() / n.sum())
    print(f"  arc above {dense_hi:.0f} ({args.property})       : {frac_arc:8.1%} of the "
          f"curve, holding {frac_n:.2%} of structures")
    if frac_arc > 0.3 and frac_n < 0.05:
        print(f"  ! most of the curve lies where almost no data does. A --delta step will\n"
              f"    mostly traverse empty tail; consider --max-value or a higher "
              f"--min-count.")

    # the value range is part of what the curve IS, so it goes in the filename
    tag = f"_w{args.width:g}"
    if args.min_value is not None:
        tag += f"_min{args.min_value:g}"
    if args.max_value is not None:
        tag += f"_max{args.max_value:g}"
    out = args.out or (f"{OUT_DIR}/{args.property.replace('.', '_')}"
                       f"_layer{args.layer}_k{args.k}{tag}.parquet")
    print(f"\nSaved {m.save(out)}")
    print(f"  {m!r}")


if __name__ == "__main__":
    main()
