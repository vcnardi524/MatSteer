#!/usr/bin/env python3
"""The fitted manifold drawn on top of the bucket centroids it was fitted to.

The centroid plot shows where each property bucket's mean embedding sits. The manifold
is the smoothing spline through those points, and steering moves along IT, not through
the centroids themselves. This overlays the two so the approximation is visible: where
the curve tracks the centroids, a step along it is a step through real data; where it
cuts a corner, it is not.

Reads the centroid CSV that centroid_pca_plots.py already wrote rather than recomputing
it -- both live in the same corpus PCA basis, so the coordinates are directly comparable
and the expensive streaming pass is not repeated. Writes to its own filename; it never
touches the plot it reads.

Usage:
    python scripts/plots/plot_manifold_overlay.py --layer 7
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS = Path("analysis/v1_all/full/train/plots")
TRIPLES = [(0, 1, 2), (3, 4, 5), (1, 2, 3), (2, 3, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--property", default="density_atomic")
    ap.add_argument("--centroids", default=None)
    ap.add_argument("--manifold", default=None)
    args = ap.parse_args()

    cen_path = Path(args.centroids or
                    PLOTS / f"centroid_pca_{args.property}_layer{args.layer}"
                            f"_w{args.width:g}.csv")
    man_path = Path(args.manifold or
                    f"steering_vectors/manifolds/{args.property}_layer{args.layer}"
                    f"_k64_w{args.width:g}_max40.parquet")
    c = pd.read_csv(cen_path).sort_values("bucket_lo").reset_index(drop=True)
    m = pd.read_parquet(man_path)
    curve = np.vstack(m["point"].to_numpy()).astype(float)
    print(f"centroids {cen_path.name}: {len(c)} buckets, "
          f"{args.property} {c.bucket_lo.min():g}..{c.bucket_lo.max():g}")
    print(f"manifold  {man_path.name}: {len(curve)} polyline points, "
          f"{args.property} {m.prop.min():.1f}..{m.prop.max():.1f}")

    # the curve was fitted with --max-value 40, so it covers only part of the buckets
    lo, hi = float(m.prop.min()), float(m.prop.max())
    inside = (c.bucket_lo >= np.floor(lo)) & (c.bucket_lo <= hi)
    print(f"  {inside.sum()} of {len(c)} buckets lie in the curve's range; "
          f"{(~inside).sum()} sit beyond it and the curve does not model them")

    pcs = [x for x in c.columns if re.fullmatch(r"pc\d+", x)]
    P = c[pcs].to_numpy(float)
    fig, axes = plt.subplots(2, 2, figsize=(15, 13),
                             subplot_kw={"projection": "3d"})
    for ax, (a, b, d3) in zip(axes.ravel(), TRIPLES):
        # centroids, coloured by property, sized by how many structures back them
        sz = 18 + 130 * (c["n"] / c["n"].max()) ** 0.5
        ax.scatter(P[inside, a], P[inside, b], P[inside, d3], c=c.bucket_lo[inside],
                   cmap="viridis", s=sz[inside], edgecolor="k", linewidth=0.4,
                   depthshade=False, zorder=3)
        ax.scatter(P[~inside, a], P[~inside, b], P[~inside, d3], c="#bbbbbb",
                   s=sz[~inside] * 0.5, depthshade=False, zorder=2,
                   label="outside the curve's range")
        ax.plot(curve[:, a], curve[:, b], curve[:, d3], color="#D55E00", lw=2.6,
                zorder=4, label="fitted manifold")
        ax.set_xlabel(f"pc{a}"); ax.set_ylabel(f"pc{b}"); ax.set_zlabel(f"pc{d3}")
        ax.set_title(f"pc{a} / pc{b} / pc{d3}", fontsize=11)
    axes[0][0].legend(loc="upper left", fontsize=9)

    # how far is each in-range centroid from the curve it was fitted to?
    dist = np.linalg.norm(P[inside][:, None, :len(pcs)] -
                          curve[None, :, :len(pcs)], axis=2).min(axis=1)
    fig.suptitle(
        f"{args.property}: the fitted manifold over the centroids it was fitted to "
        f"\u2014 layer {args.layer}, width {args.width:g}\n"
        f"orange = the spline steering moves along; coloured points = bucket centroids "
        f"(size = count); grey = buckets past the curve's {hi:.1f} cutoff\n"
        f"median centroid-to-curve distance {np.median(dist):.3f}, "
        f"max {dist.max():.3f}, over a curve spanning "
        f"{np.linalg.norm(curve[-1] - curve[0]):.1f} end to end",
        fontsize=12)
    fig.tight_layout()
    out = PLOTS / (f"centroid_pca_{args.property}_layer{args.layer}"
                   f"_w{args.width:g}_with_manifold.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\nmedian centroid-to-curve distance {np.median(dist):.3f}  "
          f"max {dist.max():.3f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
