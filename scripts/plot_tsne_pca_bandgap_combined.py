#!/usr/bin/env python3
"""
Single combined PCA + t-SNE plot of CrystaLLM embeddings coloured by CLEAN band
gap (eV) -- metals and non-metals together, not split.

Samples uniformly across the band-gap range so rare wide-gap insulators are
represented, and clips the colour scale (--vmax) so structure in the dominant
low-gap range stays visible.

Usage:
    python scripts/plot_tsne_pca_bandgap_combined.py [--layer 14] \
        [--n-samples 12000] [--bins 60] [--vmax 4]
"""
import argparse
import numpy as np

RANDOM_SEED = 42
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

BG_COL = "dos_electronic.band_gap"


def load_embeddings(layer: int) -> pd.DataFrame:
    single = Path(f"embeddings/cif_layer{layer}.parquet")
    if single.exists():
        print(f"Loading {single} ...")
        return pd.read_parquet(single, columns=["id", "embedding"])
    ckpt = Path(f"embeddings/cif_layer{layer}")
    files = sorted(ckpt.glob("checkpoint_*.parquet")) + sorted(ckpt.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No embeddings for layer {layer}")
    return pd.concat([pd.read_parquet(f, columns=["id", "embedding"]) for f in files],
                     ignore_index=True)


def uniform_sample_by_value(values, n, bins):
    edges = np.linspace(values.min(), values.max(), bins + 1)
    bi = np.clip(np.digitize(values, edges, right=True), 1, bins)
    per_bin = max(1, n // bins)
    sel = []
    for b in range(1, bins + 1):
        idx = np.where(bi == b)[0]
        if len(idx):
            sel.append(np.random.choice(idx, size=min(per_bin, len(idx)), replace=False))
    sel = np.concatenate(sel)
    if len(sel) > n:
        sel = np.random.choice(sel, size=n, replace=False)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n-samples", type=int, default=12000)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--vmax", type=float, default=4.0,
                    help="Colour-scale cap (eV); gaps above saturate")
    ap.add_argument("--tsne-perplexity", type=float, default=30)
    args = ap.parse_args()

    np.random.seed(RANDOM_SEED)

    print("Loading metadata (clean band gap) ...")
    meta = pd.read_parquet("metadata.parquet", columns=["id", BG_COL])
    meta = meta.dropna(subset=[BG_COL]).reset_index(drop=True)
    meta["band_gap_ev"] = meta[BG_COL].astype(float)
    print(f"  {len(meta):,} entries with clean band gap")

    emb = load_embeddings(args.layer)
    df = emb.merge(meta[["id", "band_gap_ev"]], on="id", how="inner")
    print(f"  joined: {len(df):,}")

    # Dense sampling: take ALL insulators (gap > 0.05, rare) + a metal fill so
    # the plot is dense and the full gap gradient is represented.
    nonm = df[df["band_gap_ev"] > 0.05].reset_index(drop=True)
    metals = df[df["band_gap_ev"] <= 0.05].reset_index(drop=True)
    n_metal_fill = max(0, args.n_samples - len(nonm))
    m_idx = np.random.choice(len(metals), size=min(n_metal_fill, len(metals)), replace=False)
    g = pd.concat([metals.iloc[m_idx], nonm], ignore_index=True)
    g = g.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle overplot order
    X = np.vstack(g["embedding"].values)
    y = g["band_gap_ev"].values
    print(f"  sampled {len(g):,}  ({len(nonm):,} insulators + {len(m_idx):,} metals; "
          f"gap range {y.min():.3f}-{y.max():.3f} eV, colour capped at {args.vmax})")

    print("  PCA (6 components) ...")
    pca = PCA(n_components=6, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    print("  explained variance ratio:", " ".join(f"PC{i+1}={evr[i]:.3f}" for i in range(6)))
    print("  t-SNE ...")
    X_tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity,
                  random_state=RANDOM_SEED, n_jobs=-1).fit_transform(X)

    # 2x2 grid: PC1-2, PC3-4, PC5-6, t-SNE
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    panels = [
        (axes[0, 0], X_pca[:, 0], X_pca[:, 1], f"PCA PC1 vs PC2 (var {evr[0]+evr[1]:.1%})", "PC1", "PC2"),
        (axes[0, 1], X_pca[:, 2], X_pca[:, 3], f"PCA PC3 vs PC4 (var {evr[2]+evr[3]:.1%})", "PC3", "PC4"),
        (axes[1, 0], X_pca[:, 4], X_pca[:, 5], f"PCA PC5 vs PC6 (var {evr[4]+evr[5]:.1%})", "PC5", "PC6"),
        (axes[1, 1], X_tsne[:, 0], X_tsne[:, 1], "t-SNE", "Component 1", "Component 2"),
    ]
    for ax, cx, cy, title, xlab, ylab in panels:
        sc = ax.scatter(cx, cy, c=y, cmap="plasma", s=8, alpha=0.7,
                        linewidths=0, vmin=0, vmax=args.vmax)
        plt.colorbar(sc, ax=ax, label=f"Band gap (eV, capped {args.vmax})")
        ax.set_title(f"{title} - Layer {args.layer}")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    plt.suptitle(f"All materials by clean band gap - CrystaLLM Layer {args.layer} "
                 f"(n={len(g):,})", y=1.00)
    plt.tight_layout()
    out_dir = Path(f"plots/layer{args.layer}"); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bandgap_combined_layer{args.layer}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # 3D t-SNE rendered as the three pairwise projections (dims are unordered,
    # so 1-2 / 1-3 / 2-3 are all equally valid views).
    print("  3D t-SNE ...")
    X_t3 = TSNE(n_components=3, perplexity=args.tsne_perplexity,
                random_state=RANDOM_SEED, n_jobs=-1).fit_transform(X)
    fig2, ax2 = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, (a, b) in zip(ax2, [(0, 1), (0, 2), (1, 2)]):
        sc = ax.scatter(X_t3[:, a], X_t3[:, b], c=y, cmap="plasma", s=8,
                        alpha=0.7, linewidths=0, vmin=0, vmax=args.vmax)
        plt.colorbar(sc, ax=ax, label=f"Band gap (eV, capped {args.vmax})")
        ax.set_title(f"3D t-SNE dim{a+1} vs dim{b+1} - Layer {args.layer}")
        ax.set_xlabel(f"dim {a+1}"); ax.set_ylabel(f"dim {b+1}")
    plt.suptitle(f"3D t-SNE by clean band gap - CrystaLLM Layer {args.layer} "
                 f"(n={len(g):,})", y=1.02)
    plt.tight_layout()
    out3 = out_dir / f"bandgap_tsne3d_layer{args.layer}.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved {out3}")


if __name__ == "__main__":
    main()
