#!/usr/bin/env python3
"""Where do real embeddings land on the manifold, against where they should?

`encode` answers "which point on the fitted curve is this activation nearest". For
steering to work that has to recover the structure's actual property: a structure at
30 A^3/atom should land where the curve passes through 30. This plots whether it does.

Both axes are in property units, so a perfect encode is the identity line:

x  the structure's true property
y  the property of the curve point encode() picked for it

Left panel is a 2-D histogram (tens of thousands of points overplot as a scatter);
right panel is the same data as the distribution of encoded property for a few narrow
true-property bands, which shows how tightly each band lands where it belongs.

Usage:
    python scripts/plots/plot_manifold_encoding.py \
        --manifold steering_vectors/manifolds/density_atomic_layer14_k64_w1_max40.parquet
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from manifold import Manifold, embedding_files
from utils import analysis_dir, load_split_index, add_partition_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifold", required=True)
    ap.add_argument("--property", default="density_atomic")
    ap.add_argument("--labels", default="density_atomic_v1.parquet")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=40000, help="structures to sample")
    add_partition_args(ap)                      # --dataset / --variant / --partition
    ap.add_argument("--batch-size", type=int, default=50_000)
    args = ap.parse_args()

    m = Manifold.load(args.manifold)
    lo, hi = float(m.prop.min()), float(m.prop.max())
    print(f"{m!r}\n  curve spans {args.property} {lo:.2f}..{hi:.2f}, arc 0..{m.length:.2f}")

    row = pd.read_parquet(
        f"steering_vectors/pca_centroid/pca_layer{args.layer}_k{args.k}.parquet").iloc[0]
    mu = np.asarray(row["mean"], np.float32)
    W = np.asarray(row["components"], np.float32).reshape(int(row["k"]), -1)

    lab = pd.read_parquet(args.labels, columns=[args.id_col, args.property]).dropna()
    if args.id_col != "id":
        lab = lab.rename(columns={args.id_col: "id"})
    lab[args.property] = pd.to_numeric(lab[args.property], errors="coerce")
    lab = lab.dropna(subset=[args.property])
    # only the range the curve actually covers -- outside it encode can only clamp
    lab = lab[(lab[args.property] >= lo) & (lab[args.property] <= hi)]
    if args.partition != "all":
        sp = load_split_index()
        lab = lab[lab["id"].isin(set(sp.loc[sp["split"] == args.partition, "id"]))]
    rng = np.random.default_rng(0)
    if len(lab) > args.n:
        lab = lab.iloc[rng.choice(len(lab), args.n, replace=False)]
    want = dict(zip(lab["id"].to_numpy(), lab[args.property].to_numpy()))
    print(f"  {len(want):,} structures sampled from partition '{args.partition}'")

    vals, Z = [], []
    for path in embedding_files(args.layer, args.dataset, args.variant):
        for rb in pq.ParquetFile(path).iter_batches(batch_size=args.batch_size,
                                                    columns=["id", "embedding"]):
            d = rb.to_pandas()
            d = d[d["id"].isin(want)]
            if d.empty:
                continue
            X = np.vstack(d["embedding"].to_numpy()).astype(np.float32)
            Z.append((X - mu) @ W.T)
            vals.extend(d["id"].map(want).to_numpy())
    Z = np.vstack(Z); vals = np.asarray(vals, float)
    print(f"  embeddings {Z.shape}")

    u, _ = m.encode(torch.tensor(Z))
    u = u.flatten().numpy()
    # read the arc back out in property units, so both axes share a scale
    got = np.interp(u, m.arc.numpy(), m.prop.numpy())
    err = got - vals
    rho = float(pd.Series(vals).corr(pd.Series(got), method="spearman"))
    print(f"  spearman(true, encoded) = {rho:+.4f}   (a working encode is ~+1.0)")
    print(f"  error (encoded - true):  median {np.median(err):+.2f}  "
          f"MAE {np.abs(err).mean():.2f}  within +-2: {(np.abs(err) <= 2).mean():.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    hb = ax.hexbin(vals, got, gridsize=60, cmap="Blues", mincnt=1,
                   extent=(lo, hi, lo, hi))
    fig.colorbar(hb, ax=ax, label="structures")
    ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="a perfect encode")
    ax.set_xlabel(f"true {args.property}")
    ax.set_ylabel(f"{args.property} encode() assigns it")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(f"{len(vals):,} real embeddings\n"
                 f"spearman {rho:+.3f}   MAE {np.abs(err).mean():.2f}   "
                 f"within \u00b12: {(np.abs(err) <= 2).mean():.0%}")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    ax = axes[1]
    bands = [(6, 9), (12, 14), (16, 18), (21, 23), (28, 31)]
    cmap = plt.get_cmap("viridis")
    rows = []
    for i, (a, b) in enumerate(bands):
        sel = (vals >= a) & (vals < b)
        if sel.sum() < 30:
            continue
        c = cmap(i / max(len(bands) - 1, 1))
        ax.hist(got[sel], bins=70, range=(lo, hi), histtype="step", lw=2,
                density=True, color=c, label=f"{a:g}\u2013{b:g} ({sel.sum():,})")
        ax.axvline((a + b) / 2, color=c, ls=":", lw=1.2)
        rows.append({"band": f"{a:g}-{b:g}", "n": int(sel.sum()),
                     "median_encoded": float(np.median(got[sel])),
                     "mae": float(np.abs(err[sel]).mean()),
                     "frac_within_2": float((np.abs(err[sel]) <= 2).mean())})
    ax.set_xlabel(f"{args.property} encode() assigns it")
    ax.set_ylabel("share of band (area = 1)")
    ax.set_title("where each true-property band lands\n"
                 "dotted line = its true centre")
    ax.legend(fontsize=9, frameon=False, title=f"true {args.property}")
    print("\n  band        n  median  MAE  within2")
    for r in rows:
        print(f"  {r['band']:>7} {r['n']:6,} {r['median_encoded']:7.2f} "
              f"{r['mae']:5.2f} {r['frac_within_2']:7.0%}")

    fig.suptitle(f"{args.property}: does encode() recover the property? "
                 f"layer {args.layer}, k={args.k}, {args.dataset}/{args.partition}",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out = analysis_dir(args.dataset, args.variant, args.partition, subdir="plots") / \
        f"manifold_encoding_{args.property}_layer{args.layer}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    pd.DataFrame({args.property: vals, "encoded_arc": u,
                  "encoded_property": got}).to_csv(
        str(out).replace(".png", ".csv"), index=False, float_format="%.6g")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
