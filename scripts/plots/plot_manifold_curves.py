#!/usr/bin/env python3
"""Draw fitted manifolds in the PCA subspace they live in, one row per curve.

plot_manifold_overlay.py shows a curve against the centroids it was fitted to, but that
needs a matching centroid CSV. This needs only the manifold parquet, so several fits --
different properties, layers, widths, partitions -- can be put side by side and compared
directly. Useful for judging whether a refit actually changed the geometry.

Each row is one manifold across three PC triples; colour runs along the property, so a
curve that doubles back in property order is visible as colour repeating along its length.
The header carries the numbers that say whether the fit is usable: how much of the arc
sits in the sparse tail, and the property range actually covered.

    python scripts/plots/plot_manifold_curves.py                  # every manifold on disk
    python scripts/plots/plot_manifold_curves.py --match energy   # just the hull ones
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRIPLES = [(0, 1, 2), (1, 2, 3), (3, 4, 5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="steering_vectors/manifolds")
    ap.add_argument("--match", default="", help="substring filter on the filename")
    ap.add_argument("--threshold", type=float, default=None,
                    help="report what fraction of the arc lies above this property "
                         "value; defaults to the curve's own median, which is 50%% by "
                         "construction and therefore uninformative -- set it")
    ap.add_argument("--out", default="analysis/manifold_curves.png")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(f"{args.dir}/*.parquet") if args.match in f)
    if not files:
        raise SystemExit(f"no manifolds matching {args.match!r} in {args.dir}")
    print(f"{len(files)} manifolds\n")

    fig = plt.figure(figsize=(5.2 * len(TRIPLES), 4.6 * len(files)))
    for r, f in enumerate(files):
        d = pd.read_parquet(f)
        m = d.iloc[0]
        P = np.vstack(d["point"].to_numpy()).astype(float)
        prop, arc = d["prop"].to_numpy(), d["arc"].to_numpy()
        # Fraction of the curve's LENGTH spent above a fixed property value. For a
        # skewed property most of the arc buys almost no data, which is what makes a
        # --delta step traverse empty space. (Taking the median of the curve's own
        # samples would give 50% by construction and say nothing.)
        thr = args.threshold if args.threshold is not None else float(np.median(prop))
        tail = 100 * (arc[-1] - float(np.interp(thr, prop, arc))) / arc[-1]
        name = Path(f).stem
        for c, (a, b, z) in enumerate(TRIPLES):
            ax = fig.add_subplot(len(files), len(TRIPLES),
                                 r * len(TRIPLES) + c + 1, projection="3d")
            s = ax.scatter(P[:, a], P[:, b], P[:, z], c=prop, cmap="viridis", s=4)
            ax.plot(P[:, a], P[:, b], P[:, z], color="#888", lw=0.7, alpha=0.6)
            ax.scatter(*P[0, [a, b, z]], color="#2166ac", s=70, marker="o",
                       edgecolor="k", label=f"low ({prop[0]:.3g})")
            ax.scatter(*P[-1, [a, b, z]], color="#b2182b", s=70, marker="s",
                       edgecolor="k", label=f"high ({prop[-1]:.3g})")
            ax.set_xlabel(f"pc{a}", labelpad=-6); ax.set_ylabel(f"pc{b}", labelpad=-6)
            ax.set_zlabel(f"pc{z}", labelpad=-6)
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.legend(fontsize=7, loc="upper left")
                ax.set_title(f"{name}\n{m.meta_property} L{m.meta_layer} "
                             f"{m.meta_dataset}/{m.meta_partition} w{m.meta_width:g}  |  "
                             f"{int(m.meta_n_buckets)} buckets, arc {arc[-1]:.1f}, "
                             f"{tail:.0f}% of arc above {thr:g}",
                             fontsize=9, loc="left")
            else:
                ax.set_title(f"pc{a}/pc{b}/pc{z}", fontsize=9)
        print(f"  {name}: arc {arc[-1]:.1f}, property {prop[0]:.3g}..{prop[-1]:.3g}, "
              f"{tail:.0f}% of arc above property {thr:g}")

    fig.suptitle("Fitted manifolds in PCA space — colour runs low (blue) to high (yellow) "
                 "along the property", fontsize=13)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
