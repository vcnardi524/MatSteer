#!/usr/bin/env python3
"""
  1. Load formulas from the CIF `data_` headers (cifs_v1_prep.pkl.gz)
  2. Load CIF embeddings per layer, intersected with the metadata.parquet symmetry
     label chosen by LABEL_COL (space_group_symbol or point_group)
  3. Exp 1.1: mean pairwise cosine for pairs sharing that label vs pairs not sharing
     it -> ratio per layer (shows where symmetry separability peaks)
  4. Exp 1.2: the same ratio restricted to pairs with the SAME composition, which
     isolates symmetry from the model just knowing the chemistry
  5. Label-permutation null for both, so a ratio can be read against its noise floor
  6. Save the per-layer table to OUT_CSV and the per-layer line graphs to OUT_PNG

No N x N matrix and no subsampling: for unit-norm rows,
    sum_{i<j} x_i . x_j = (||sum_i x_i||^2 - N) / 2
so each pair mean is an exact closed form over group sums at O(N*D) cost. Same-label
pairs come from per-label sums; "different-label" is (all pairs - same-label pairs).
Exp 1.2 applies the identity within each composition group. Rows are unit-normalized,
so mean squared L2 is 2 - 2*cos and reporting cosine alone loses nothing.

Composition key = the NON-REDUCED formula from the `data_` header (e.g.
`data_Sc4Si2P2`) — the composition the model actually sees. augment_cif rewrites the
header to the non-reduced formula, and both bin/tokenize_cifs.py:47 and
scripts/embeddings/extract_cif_embeddings.py:77 strip `#` comment lines, so it is the only
formula in the token stream.

Every number is computed twice: `raw` cosine, and `centered` (global mean removed
before normalizing). Transformer embeddings share a large common mean vector, so raw
cosines all sit near ~1 and ratios compress toward 1.0 whether or not signal exists —
centered is the interpretable one.

Usage:
    python symmetry_separability.py                    # all 16 layers, space group
    python symmetry_separability.py 0 5 10 14          # a subset of layers
    LABEL_COL=point_group python symmetry_separability.py    # point group instead
"""

import gzip
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
from utils import load_labeled_embeddings

# -------------------------------
# Configuration
# -------------------------------
LAYERS = [int(a) for a in sys.argv[1:]] or list(range(16))
DATASET = "v1_all"
METADATA_PATH = "./metadata.parquet"
PKL_PATH = "./CrystaLLM/cifs_v1_prep.pkl.gz"
OUTPUT_DIR = "./analysis"
# Which symmetry label to test: "space_group_symbol" (207 values) or "point_group" (32).
# Set with e.g. LABEL_COL=point_group python symmetry_separability.py
LABEL_COL = os.environ.get("LABEL_COL", "space_group_symbol")
OUT_CSV = os.path.join(OUTPUT_DIR, f"symmetry_separability_{LABEL_COL}.csv")
OUT_PNG = os.path.join(OUTPUT_DIR, f"symmetry_separability_{LABEL_COL}.png")
OUT_PNG_CENTERED = os.path.join(OUTPUT_DIR, f"symmetry_separability_{LABEL_COL}_centered.png")
RANDOM_SEED = 1
DATA_RE = re.compile(r"^data_(\S+)", re.M)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def group_pair_sums(X, codes, n_groups):
    """Per-group (sum of within-group pairwise cosines, number of such pairs).

    Uses sum_{i<j in g} x_i.x_j = (||s_g||^2 - n_g) / 2, where s_g is the sum of the
    group's unit rows. Accumulated in float64: s_g reaches ~1e6 unit vectors and the
    squared norm ~1e12, which float32 cannot hold.
    """
    n = len(codes)
    M = csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(n_groups, n))
    S = M @ X.astype(np.float64, copy=False)
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    sums = (np.einsum("ij,ij->i", S, S) - counts) / 2.0
    return sums, counts * (counts - 1) / 2.0


def plot_results(out, png_path=OUT_PNG, variants=("centered", "raw")):
    """One line graph, layer on x, every ratio and mean cosine on the same axes.

    Style code: colour = which quantity, solid+dots = centered, faded thin = raw,
    dotted = that quantity's permutation null. Pass variants=("centered",) to drop
    the raw curves, which carry no signal and just stack on top of each other at 1.0.

    The Exp 1.1 ratio is the one series that cannot share the axis: after centering,
    mean_cos_diff sits at ~-0.01, so same/diff blows up to -25..-9. It is drawn but
    falls outside YLIM by construction -- read `delta` (same - diff) instead, which is
    on-scale and is the statistic that means something once the data are centered.
    """
    # label_col was added when the script was generalized to point groups; the first
    # space-group CSV predates it, so fall back rather than fail on the older file.
    label_col = out["label_col"].iloc[0] if "label_col" in out else LABEL_COL
    YLIM = (-0.1, 2.0)

    # (column, colour, legend label). Nulls are the dotted twin of their ratio.
    series = [
        ("mean_cos_same_sg",    "C0", "-",  f"1.1 mean cos, same {label_col}"),
        ("mean_cos_diff_sg",    "C0", "--", f"1.1 mean cos, diff {label_col}"),
        ("ratio",               "C1", "-",  "1.1 ratio (off-scale when centered)"),
        ("null_ratio",          "C1", ":",  "1.1 ratio, permuted-label null"),
        ("delta",               "C4", "-",  "1.1 delta (same - diff)"),
        ("cc_mean_cos_same_sg", "C2", "-",  f"1.2 mean cos, same formula + same {label_col}"),
        ("cc_mean_cos_diff_sg", "C2", "--", f"1.2 mean cos, same formula + diff {label_col}"),
        ("cc_ratio",            "C3", "-",  "1.2 ratio"),
        ("cc_null_ratio",       "C3", ":",  "1.2 ratio, permuted-label null"),
        ("cc_delta",            "C5", "-",  "1.2 delta (same - diff)"),
    ]

    fig, ax = plt.subplots(figsize=(13, 8))
    for variant in variants:
        d = out[out["variant"] == variant].sort_values("layer")
        if len(d) == 0:
            continue
        centered = variant == "centered"
        for col, colour, style, name in series:
            ax.plot(d["layer"], d[col], style, color=colour,
                    marker="o" if centered else None, markersize=4,
                    linewidth=1.8 if centered else 1.0,
                    alpha=1.0 if centered else 0.30,
                    label=name if len(variants) == 1 else f"{name} [{variant}]")

    ax.axhline(1.0, color="grey", linewidth=0.8)   # ratio of 1.0 = no separation
    ax.axhline(0.0, color="grey", linewidth=0.8)   # delta of 0 = no separation
    ax.set_xlim(out["layer"].min(), out["layer"].max())
    ax.set_ylim(*YLIM)
    ax.set_xticks(sorted(out["layer"].unique()))
    ax.set_xlabel("transformer block (output of h[layer])")
    ax.set_ylabel("cosine / ratio")
    subtitle = ("centered embeddings" if len(variants) == 1 else
                "bold = centered embeddings, faded = raw (raw is inert: everything sits at ~1.0)")
    ax.set_title(f"Symmetry separability by layer — {label_col}\n{subtitle}")
    ax.grid(alpha=0.25)
    # Below the axes: 20 series need the room, and the top-left corner holds real data.
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {png_path}")


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # -------------------------------
    # Formulas from the CIF data_ headers
    # -------------------------------
    print(f"Loading CIFs from {PKL_PATH} ...")
    with gzip.open(PKL_PATH, "rb") as f:
        cifs = pickle.load(f)
    ids, forms = [], []
    for cid, cif in cifs:
        m = DATA_RE.search(cif)
        if m:
            ids.append(cid)
            forms.append(m.group(1))
    del cifs
    formula_df = pd.DataFrame({"id": ids, "formula": forms})
    print(f"  {len(formula_df):,} CIFs with a data_ header formula")

    results = []
    for layer in LAYERS:
        print(f"\n{'='*60}\nLayer {layer}\n{'='*60}")

        # -------------------------------
        # Load and intersect datasets
        # -------------------------------
        df = load_labeled_embeddings(layer, dataset=DATASET, metadata_path=METADATA_PATH,
                                     label_cols=(LABEL_COL,))
        df = df.merge(formula_df, on="id", how="inner")
        df = df[df[LABEL_COL].notna() & (df[LABEL_COL] != "")]
        df = df.dropna(subset=["formula"]).reset_index(drop=True)

        X = np.vstack(df["embedding"].values)
        label, label_names = pd.factorize(df[LABEL_COL])
        comp = pd.factorize(df["formula"])[0]
        n_label = len(label_names)
        n_total = len(X)
        print(f"Data shape: {X.shape}  ({n_label:,} {LABEL_COL}s, {comp.max()+1:,} formulas)")

        # -------------------------------
        # Composition-controlled subset (Exp 1.2)
        # -------------------------------
        # Compositions seen only once contribute no same-composition pairs at all.
        sub = np.bincount(comp)[comp] >= 2
        comp_s = pd.factorize(comp[sub])[0]
        label_s = label[sub]
        n_comp = comp_s.max() + 1
        cs = pd.factorize(comp_s.astype(np.int64) * n_label + label_s)[0]
        n_cs = cs.max() + 1
        print(f"  composition-controlled subset: {int(sub.sum()):,} structures in "
              f"{n_comp:,} multi-entry composition groups")

        # -------------------------------
        # Permuted labels for the null
        # -------------------------------
        # 1.1 shuffles the label globally. 1.2 must shuffle it *within* each
        # composition group, so the null holds chemistry fixed and only breaks the
        # symmetry association. Both orderings below group rows by composition
        # identically; the second randomizes order inside each block, so assigning
        # one through the other permutes labels within groups.
        label_null = rng.permutation(label)
        base = np.argsort(comp_s, kind="stable")
        shuf = np.lexsort((rng.random(len(comp_s)), comp_s))
        label_s_null = np.empty_like(label_s)
        label_s_null[base] = label_s[shuf]
        cs_null = pd.factorize(comp_s.astype(np.int64) * n_label + label_s_null)[0]
        n_cs_null = cs_null.max() + 1

        for center in (False, True):
            variant = "centered" if center else "raw"

            # -------------------------------
            # Normalize (optional centering + L2)
            # -------------------------------
            Y = X.astype(np.float32, copy=True)
            if center:
                Y -= Y.mean(axis=0, keepdims=True)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
            Y_s = Y[sub]

            # -------------------------------
            # Exp 1.1: same symmetry label vs different label
            # -------------------------------
            tot = Y.astype(np.float64).sum(axis=0)
            all_sum = (tot @ tot - n_total) / 2.0
            all_cnt = n_total * (n_total - 1) / 2.0

            sums, cnts = group_pair_sums(Y, label, n_label)
            same_sum, same_cnt = sums.sum(), cnts.sum()
            mean_same = same_sum / same_cnt
            mean_diff = (all_sum - same_sum) / (all_cnt - same_cnt)
            ratio = mean_same / mean_diff

            # Micro-average is dominated by the largest space groups (Fm-3m,
            # P6_3/mmc, ...); the macro-average weights every space group equally.
            ok = cnts > 0
            macro_same = (sums[ok] / cnts[ok]).mean()

            null_sums, null_cnts = group_pair_sums(Y, label_null, n_label)
            null_same = null_sums.sum() / null_cnts.sum()
            null_diff = (all_sum - null_sums.sum()) / (all_cnt - null_cnts.sum())
            null_ratio = null_same / null_diff

            # -------------------------------
            # Exp 1.2: same composition, same vs different space group
            # -------------------------------
            comp_sums, comp_cnts = group_pair_sums(Y_s, comp_s, n_comp)
            comp_sum, comp_cnt = comp_sums.sum(), comp_cnts.sum()

            cc_sums, cc_cnts = group_pair_sums(Y_s, cs, n_cs)
            cc_same_sum, cc_same_cnt = cc_sums.sum(), cc_cnts.sum()
            cc_mean_same = cc_same_sum / cc_same_cnt
            cc_mean_diff = (comp_sum - cc_same_sum) / (comp_cnt - cc_same_cnt)
            cc_ratio = cc_mean_same / cc_mean_diff

            ccn_sums, ccn_cnts = group_pair_sums(Y_s, cs_null, n_cs_null)
            ccn_same = ccn_sums.sum() / ccn_cnts.sum()
            ccn_diff = (comp_sum - ccn_sums.sum()) / (comp_cnt - ccn_cnts.sum())
            cc_null_ratio = ccn_same / ccn_diff

            results.append({
                "layer": layer, "variant": variant, "label_col": LABEL_COL,
                "n_structures": n_total,
                "same_sg_pairs": same_cnt, "diff_sg_pairs": all_cnt - same_cnt,
                "mean_cos_same_sg": mean_same, "mean_cos_diff_sg": mean_diff,
                "ratio": ratio, "delta": mean_same - mean_diff,
                "macro_mean_cos_same_sg": macro_same, "null_ratio": null_ratio,
                "cc_n_structures": int(sub.sum()),
                "cc_same_sg_pairs": cc_same_cnt, "cc_diff_sg_pairs": comp_cnt - cc_same_cnt,
                "cc_mean_cos_same_sg": cc_mean_same, "cc_mean_cos_diff_sg": cc_mean_diff,
                "cc_ratio": cc_ratio, "cc_delta": cc_mean_same - cc_mean_diff,
                "cc_null_ratio": cc_null_ratio,
            })

            print(f"  [{variant:8s}] "
                  f"1.1 ratio={ratio:.4f} (same={mean_same:.4f} diff={mean_diff:.4f} "
                  f"null={null_ratio:.4f}) | "
                  f"1.2 ratio={cc_ratio:.4f} (same={cc_mean_same:.4f} "
                  f"diff={cc_mean_diff:.4f} null={cc_null_ratio:.4f})", flush=True)
            del Y, Y_s

        del X, df

    # -------------------------------
    # Save and report
    # -------------------------------
    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows to {OUT_CSV}")
    plot_results(out)
    plot_results(out, OUT_PNG_CENTERED, variants=("centered",))

    for variant in ["raw", "centered"]:
        d = out[out["variant"] == variant]
        if len(d) == 0:
            continue
        print(f"\n=== {variant} ===")
        print(d[["layer", "ratio", "null_ratio", "macro_mean_cos_same_sg",
                 "cc_ratio", "cc_null_ratio", "cc_same_sg_pairs"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nDone. All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
