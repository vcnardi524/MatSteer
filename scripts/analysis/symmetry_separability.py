#!/usr/bin/env python3
"""
  1. Load formulas from the CIF `data_` headers (cifs_v1_prep.pkl.gz)
  2. Load CIF embeddings per layer, intersected with metadata.parquet space groups
  3. Exp 1.1: mean pairwise cosine for pairs sharing a space group vs pairs not
     sharing one -> ratio per layer (shows where space-group separability peaks)
  4. Exp 1.2: the same ratio restricted to pairs with the SAME composition, which
     isolates symmetry from the model just knowing the chemistry
  5. Label-permutation null for both, so a ratio can be read against its noise floor
  6. Save the per-layer table to OUT_CSV

No N x N matrix and no subsampling: for unit-norm rows,
    sum_{i<j} x_i . x_j = (||sum_i x_i||^2 - N) / 2
so each pair mean is an exact closed form over group sums at O(N*D) cost. Same-SG
pairs come from per-space-group sums; "different-SG" is (all pairs - same-SG pairs).
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
    python symmetry_separability.py                 # all 16 layers
    python symmetry_separability.py 0 5 10 14
"""

import gzip
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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
        # 1.1 shuffles space group globally. 1.2 must shuffle it *within* each
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
            # Exp 1.1: same space group vs different space group
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
