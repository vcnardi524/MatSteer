#!/usr/bin/env python3
"""
PCA and t-SNE of the MP (`v1_mp`) CrystaLLM embeddings coloured by a scalar MP
property, one figure per layer plus a cross-layer t-SNE comparison.

Default property: **`density_atomic`** from `metadata_mp.parquet`.

WHAT `density_atomic` ACTUALLY IS
--------------------------------
It is MP's **volume per atom in A^3/atom** — verified here to equal `volume / nsites`
exactly (max abs diff 1.4e-14 over 154,879 rows). So it runs the opposite way from the
word "dense": a LARGE value is a SPARSE, loosely packed structure. Range in this file is
4.36 - 3733.54 A^3/atom, median 14.97, p99 209.29. Pass `--invert` to plot the number
density 1/density_atomic (atoms/A^3) instead, or `--column density` for the mass density
in g/cm^3.

WHY THIS PROPERTY IS SAFE ON MP DATA (unlike `volume`)
------------------------------------------------------
`metadata_mp.parquet`'s scalar columns come from MP's `SummaryDoc`, which describes the
**primitive** cell, while the CIFs the model read — and therefore the `v1_mp` embeddings
— are **conventional** cells. Extensive columns (`volume`, `nsites`) are off by the
centering factor (2x/4x for I/F/C lattices) and are dirty labels. `density_atomic` is
volume/nsites, which is **intensive**, so it is identical in both cells and is a clean
label. Same for `density`. See tasks.md, "metadata_mp scalar cols are PRIMITIVE-cell".

The join key is **`material_id`** (e.g. `MP_mp-32493`), NOT `id` (`MP_mp-bwbt`) —
`material_id` is what matches the embedding ids. All 154,871 embedded ids join.

COLOUR SCALE
------------
`density_atomic` is heavily right-skewed (p99 is 14x the median, max is 250x), so the
default `--color-scale auto` switches to log10 when p99/p1 > 10 and every value is
positive. Linear scales are percentile-clipped (`--clip-lo/--clip-hi`) so one 3733
A^3/atom outlier cannot flatten the map to a single colour.

OUTPUT
------
Both land under analysis/<dataset>/<variant>/<partition>/plots/ :
    layer{N}/{prop}_tsne_pca_mp_layer{N}.png   per layer: PC1-2, PC3-4, t-SNE
    {prop}_tsne_mp_layers{A}_{B}.png           t-SNE side by side across layers

The dataset defaults to **v1_mp** here, not utils' v1_all. A t-SNE layout is only
comparable to another t-SNE layout built from the same point cloud, so a v1_mp map
cannot be laid next to a v1_all one (2.29M ids, 71% NOMAD) and read for shape.

Usage:
    python scripts/plots/plot_tsne_pca_mp_property.py --layers 5 14 --partition all
    python scripts/plots/plot_tsne_pca_mp_property.py --layers 5 14 --partition val --invert
    python scripts/plots/plot_tsne_pca_mp_property.py --layers 14 --column density --partition test
"""
import argparse

import numpy as np
RANDOM_SEED = 42
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import load_embeddings, add_partition_args, filter_partition, analysis_dir

METADATA_PATH = "metadata_mp.parquet"
JOIN_KEY = "material_id"   # NOT `id` -- see the module docstring
CMAP = "viridis"

# Axis/colourbar labels for the columns worth plotting. Anything else falls back to
# the bare column name.
UNITS = {
    "density_atomic": "Volume per atom (A^3/atom)",
    "density": "Mass density (g/cm^3)",
    "volume": "Cell volume (A^3, PRIMITIVE -- dirty label, see docstring)",
    "nsites": "Sites per cell (PRIMITIVE -- dirty label, see docstring)",
}


def load_joined(layer: int, column: str, invert: bool, dataset: str,
                variant: str, partition: str) -> pd.DataFrame:
    """v1_mp embeddings for one layer joined to an MP scalar column on material_id."""
    meta = pd.read_parquet(METADATA_PATH, columns=[JOIN_KEY, column]).dropna(subset=[column])
    meta = meta.rename(columns={JOIN_KEY: "id"})
    meta["value"] = meta[column].astype(float)
    if invert:
        meta = meta[meta["value"] != 0]
        meta["value"] = 1.0 / meta["value"]
    print(f"  metadata rows with {column}: {len(meta):,}")

    emb = load_embeddings(layer, dataset=dataset, variant=variant)
    emb = filter_partition(emb, partition)
    df = emb.merge(meta[["id", "value"]], on="id", how="inner")
    print(f"  layer-{layer} embeddings: {len(emb):,}  ->  after join: {len(df):,}")
    if len(df) < 10:
        raise SystemExit("Too few joined rows to embed -- check --column / dataset.")
    return df


def resolve_scale(y: np.ndarray, mode: str, clip_lo: float, clip_hi: float):
    """(values_to_colour, vmin, vmax, label_suffix) for the requested colour scale."""
    p_lo, p_hi = np.percentile(y, [clip_lo, clip_hi])
    if mode == "auto":
        ratio = p_hi / p_lo if p_lo > 0 else np.inf
        mode = "log" if (y.min() > 0 and ratio > 10) else "linear"
        print(f"  colour-scale auto -> {mode} (p{clip_hi:g}/p{clip_lo:g} ratio {ratio:.1f})")
    if mode == "log":
        if y.min() <= 0:
            raise SystemExit("--color-scale log needs strictly positive values.")
        c = np.log10(y)
        lo, hi = np.percentile(c, [clip_lo, clip_hi])
        return c, lo, hi, " [log10]"
    return y, p_lo, p_hi, f" [clipped p{clip_lo:g}-p{clip_hi:g}]"


def knn_smoothness(coords: np.ndarray, c: np.ndarray, k: int = 10) -> float:
    """Spearman between each point's value and the mean value of its k t-SNE
    neighbours. High = the property varies smoothly across the map (the layout is
    organized by it); ~0 = the property is scattered through the layout."""
    k = min(k, len(coords) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    return float(spearmanr(c, c[idx[:, 1:]].mean(axis=1)).statistic)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 14])
    ap.add_argument("--column", default="density_atomic",
                    help="Scalar column in metadata_mp.parquet (default: density_atomic, "
                         "= volume/nsites in A^3/atom)")
    ap.add_argument("--invert", action="store_true",
                    help="Plot 1/column -- turns density_atomic into a number density "
                         "in atoms/A^3, where bigger really does mean denser")
    add_partition_args(ap)   # --dataset / --variant / --partition
    # This script is MP-specific (metadata_mp.parquet, joined on material_id), so it
    # overrides add_partition_args' v1_all default. Pointing it at v1_all would join MP
    # metadata against the NOMAD+OQMD+MP corpus and silently keep only the ~59k shared
    # ids -- a different point cloud from the one the title claims.
    ap.set_defaults(dataset="v1_mp")
    ap.add_argument("--n-samples", type=int, default=15000)
    ap.add_argument("--color-scale", choices=["auto", "log", "linear"], default="auto")
    ap.add_argument("--clip-lo", type=float, default=1.0, help="Lower colour percentile")
    ap.add_argument("--clip-hi", type=float, default=99.0, help="Upper colour percentile")
    ap.add_argument("--tsne-perplexity", type=float, default=30)
    args = ap.parse_args()
    np.random.seed(RANDOM_SEED)

    prop = ("inv_" if args.invert else "") + args.column
    base_label = UNITS.get(args.column, args.column)
    label = f"1 / [{base_label}]" if args.invert else base_label
    print(f"Property: {prop}  ({label})\nJoin key: {JOIN_KEY}  dataset: {args.dataset}")

    results = {}
    for layer in args.layers:
        print(f"\n=== layer {layer} ===")
        df = load_joined(layer, args.column, args.invert, args.dataset,
                         args.variant, args.partition)

        # Same sample across layers: the id sets are identical, and seeding per layer
        # keeps the two maps comparable point-for-point.
        if len(df) > args.n_samples:
            df = df.sort_values("id").sample(n=args.n_samples, random_state=RANDOM_SEED)
        df = df.reset_index(drop=True)
        X = np.vstack(df["embedding"].values)
        y = df["value"].values
        print(f"  sampled {len(df):,}   value range {y.min():.4g} - {y.max():.4g}, "
              f"median {np.median(y):.4g}")

        c, vmin, vmax, suffix = resolve_scale(y, args.color_scale, args.clip_lo, args.clip_hi)

        print("  PCA (4 components) ...")
        pca = PCA(n_components=4, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_
        print("  explained variance:", " ".join(f"PC{i+1}={evr[i]:.3f}" for i in range(4)))
        for i in range(4):
            print(f"  Spearman(PC{i+1}, {prop}) = {spearmanr(X_pca[:, i], y).statistic:+.3f}")

        print("  t-SNE ...")
        X_tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity,
                      random_state=RANDOM_SEED, n_jobs=-1).fit_transform(X)
        smooth = knn_smoothness(X_tsne, c)
        print(f"  t-SNE 10-NN smoothness (Spearman value vs neighbour mean) = {smooth:+.3f}")

        # Draw low values first so the high tail is not buried under the bulk.
        order = np.argsort(c)
        results[layer] = dict(tsne=X_tsne, c=c, vmin=vmin, vmax=vmax, order=order,
                              smooth=smooth, n=len(df))

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        panels = [
            (axes[0], X_pca[:, 0], X_pca[:, 1], f"PCA PC1 vs PC2 (var {evr[:2].sum():.1%})", "PC1", "PC2"),
            (axes[1], X_pca[:, 2], X_pca[:, 3], f"PCA PC3 vs PC4 (var {evr[2:4].sum():.1%})", "PC3", "PC4"),
            (axes[2], X_tsne[:, 0], X_tsne[:, 1],
             f"t-SNE (10-NN smoothness {smooth:+.2f})", "Component 1", "Component 2"),
        ]
        for ax, cx, cy, title, xlab, ylab in panels:
            sc = ax.scatter(cx[order], cy[order], c=c[order], cmap=CMAP, s=6,
                            alpha=0.75, linewidths=0, vmin=vmin, vmax=vmax)
            plt.colorbar(sc, ax=ax, label=label + suffix)
            ax.set_title(f"{title} — layer {layer}")
            ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        fig.suptitle(f"MP embeddings ({args.dataset}) by {prop} — layer {layer}, "
                     f"n={len(df):,}   [{label}]", y=1.02, fontsize=13)
        fig.tight_layout()

        out_dir = analysis_dir(args.dataset, args.variant, args.partition,
                               subdir=f"plots/layer{layer}")
        out = out_dir / f"{prop}_tsne_pca_mp_layer{layer}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")

    if len(args.layers) > 1:
        fig, axes = plt.subplots(1, len(args.layers),
                                 figsize=(8.5 * len(args.layers), 7.5), squeeze=False)
        for ax, layer in zip(axes[0], args.layers):
            r = results[layer]
            sc = ax.scatter(r["tsne"][r["order"], 0], r["tsne"][r["order"], 1],
                            c=r["c"][r["order"]], cmap=CMAP, s=6, alpha=0.75,
                            linewidths=0, vmin=r["vmin"], vmax=r["vmax"])
            plt.colorbar(sc, ax=ax, label=label + suffix)
            ax.set_title(f"Layer {layer} — 10-NN smoothness {r['smooth']:+.2f}")
            ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
        fig.suptitle(f"t-SNE of MP embeddings by {prop} across layers "
                     f"(n={results[args.layers[0]]['n']:,})   [{label}]",
                     y=1.02, fontsize=13)
        fig.tight_layout()
        out_dir = analysis_dir(args.dataset, args.variant, args.partition, subdir="plots")
        out = out_dir / f"{prop}_tsne_mp_layers{'_'.join(map(str, args.layers))}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
