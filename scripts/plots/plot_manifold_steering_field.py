#!/usr/bin/env python3
"""Where would manifold steering push, at each point along the property axis?

The overlay plot shows the fitted curve sitting among the bucket centroids it was fitted
to. This adds the intervention itself: at every centroid, an arrow for the displacement
the manifold hook would inject there, for one (delta, scale).

The arrow is exactly what steer_generate_cif.py's manifold_hook computes, in the same
PCA coordinates the centroids are plotted in:

    injection(u) = scale * ( decode(u + delta) - decode(u) )

One ROW per delta, so the rows show how the push changes as the arc step grows. One
COLUMN per triple of principal directions, the same four the overlay plot uses, because a
3-D view of a 64-D displacement hides whatever is orthogonal to it.

WHY THE ARC POSITION COMES FROM THE PROPERTY, NOT FROM encode()
---------------------------------------------------------------
The hook calls Manifold.encode, which finds the nearest knot by distance in the full
k=64 subspace. The centroid CSV written by centroid_pca_plots.py keeps only the first 12
principal coordinates, so encode() cannot be called on it -- truncating to 12 would pick a
different knot. Each centroid is a property bucket, though, and the curve carries a
property value at every knot, so `property_to_arc(bucket centre)` places the bucket on the
curve exactly and without any truncation. For a picture of "what would steering do to a
state sitting at this property value", that is the right anchor.

Arrows are drawn at their true length in PCA units. --arrow-scale only stretches them for
legibility and is called out in the title whenever it is not 1.

Usage:
    python scripts/plots/plot_manifold_steering_field.py \
        --centroids analysis/v1_mp/full/not_heldout/plots/centroid_pca_formation_energy_per_atom_layer7_w0.25.csv \
        --manifold steering_vectors/manifolds/formation_energy_per_atom_layer7_k64_w0.25.parquet \
        --deltas 1 3 6 10 --scale 3
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from manifold import Manifold

TRIPLES = [(0, 1, 2), (3, 4, 5), (1, 2, 3), (2, 3, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--manifold", required=True)
    ap.add_argument("--deltas", type=float, nargs="+", default=[1, 3, 6, 10],
                    help="One row of plots per arc step")
    ap.add_argument("--scale", type=float, default=3.0,
                    help="The steering scale s. Arrow length is literally "
                         "s*|decode(u+d)-decode(u)|, so this sets both the intervention "
                         "and the arrow length.")
    ap.add_argument("--arrow-frac", type=float, default=0.13,
                    help="Auto-scale arrows so the LARGEST delta's median arrow is this "
                         "fraction of the panel's own extent. Scaling is per COLUMN and "
                         "constant down the column, so rows stay comparable to each "
                         "other; columns are not, because the principal directions have "
                         "very different ranges (pc0 spans ~20 units here, pc5 ~2).")
    ap.add_argument("--property-label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c = pd.read_csv(args.centroids).sort_values("bucket_lo").reset_index(drop=True)
    m = Manifold.load(args.manifold)
    pcs = [x for x in c.columns if re.fullmatch(r"pc\d+", x)]
    P = c[pcs].to_numpy(float)
    curve = np.vstack(m["point"].to_numpy() if hasattr(m, "columns")
                      else m.samples.numpy()).astype(float)
    prop_label = args.property_label or m.meta.get("property", "property")
    centres = ((c.bucket_lo + c.bucket_hi) / 2).to_numpy(float)
    print(f"centroids: {len(c)} buckets over {prop_label} "
          f"{centres.min():.2f}..{centres.max():.2f}, {len(pcs)} pc columns")
    print(f"manifold : {m!r}")

    # each bucket's place on the curve, from its property value
    u = torch.tensor([[m.property_to_arc(v)] for v in centres], dtype=torch.float32)
    clamped = int(((u.squeeze(1) <= float(m.arc[0])) |
                   (u.squeeze(1) >= float(m.arc[-1]))).sum())
    if clamped:
        print(f"  {clamped} buckets sit at a curve END; their step is truncated by the "
              f"clamp in decode()")

    rows = []
    for d in args.deltas:
        disp = (args.scale * (m.decode(u + d) - m.decode(u))).numpy()   # (n_buckets, k)
        rows.append((d, disp))
        n = np.linalg.norm(disp, axis=1)
        print(f"  delta {d:>5g}: |injection| median {np.median(n):7.3f}  "
              f"min {n.min():7.3f}  max {n.max():7.3f}  (PCA units)")

    nr, nc = len(rows), len(TRIPLES)
    fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc, 5.0 * nr),
                             subplot_kw={"projection": "3d"}, squeeze=False)
    sz = 16 + 110 * (c["n"] / c["n"].max()) ** 0.5
    big = rows[-1][1]                       # the largest delta sets the scale
    col_scale = {}
    for ci, (a, b, e) in enumerate(TRIPLES):
        pts = np.vstack([P[:, [a, b, e]], curve[:, [a, b, e]]])
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        med = float(np.median(np.linalg.norm(big[:, [a, b, e]], axis=1)))
        col_scale[ci] = (args.arrow_frac * extent / med) if med > 0 else 1.0

    for ri, (d, disp) in enumerate(rows):
        for ci, (a, b, e) in enumerate(TRIPLES):
            ax = axes[ri][ci]
            k = col_scale[ci]
            ax.plot(curve[:, a], curve[:, b], curve[:, e], color="#BBBBBB", lw=1.8,
                    zorder=1)
            ax.scatter(P[:, a], P[:, b], P[:, e], c=centres, cmap="viridis", s=sz,
                       edgecolor="k", linewidth=0.35, depthshade=False, zorder=3)
            ax.quiver(P[:, a], P[:, b], P[:, e],
                      disp[:, a] * k, disp[:, b] * k, disp[:, e] * k,
                      color="#D55E00", lw=1.3, arrow_length_ratio=0.3,
                      length=1.0, normalize=False, zorder=4)
            # pin the axes to the DATA, so long arrows are clipped rather than
            # zooming every panel out until the structure disappears
            pts = np.vstack([P[:, [a, b, e]], curve[:, [a, b, e]]])
            lo, hi = pts.min(0), pts.max(0)
            pad = 0.12 * (hi - lo + 1e-9)
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
            ax.set_xlabel(f"pc{a}", labelpad=-6, fontsize=8)
            ax.set_ylabel(f"pc{b}", labelpad=-6, fontsize=8)
            ax.set_zlabel(f"pc{e}", labelpad=-6, fontsize=8)
            ax.tick_params(labelsize=6, pad=-2)
            if ci == 0:
                ax.text2D(-0.08, 0.5, f"arc step d = {d:g}", transform=ax.transAxes,
                          rotation=90, va="center", ha="center", fontsize=13,
                          fontweight="bold")
            if ri == 0:
                ax.set_title(f"pc{a} / pc{b} / pc{e}", fontsize=10)

    # A key for the colour, so the reader can tell which end of the curve is which
    # property value rather than reading the ramp as decoration.
    # Its own axis on the right -- ax=fig.axes drops it on top of the panels.
    fig.subplots_adjust(right=0.91)
    cax = fig.add_axes([0.925, 0.3, 0.011, 0.4])
    cb = fig.colorbar(plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(centres.min(), centres.max())), cax=cax)
    cb.set_label(f"{prop_label}  (bucket centre)", fontsize=11)

    extra = (f"   arrows scaled per column for legibility (relative lengths WITHIN a "
             f"column are true)")
    fig.suptitle(
        f"Where manifold steering would push, at each {prop_label} bucket\n"
        f"arrow = scale x (decode(u+d) - decode(u)) at that bucket's place on the curve, "
        f"scale = {args.scale:g}{extra}\n"
        f"grey = the fitted curve; points = bucket centroids coloured by {prop_label}, "
        f"sized by how many structures back them",
        fontsize=13)

    out = Path(args.out or Path(args.centroids).with_name(
        Path(args.centroids).stem + f"_steering_field_s{args.scale:g}.png"))
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
