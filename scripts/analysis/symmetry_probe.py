#!/usr/bin/env python3
"""Exp 1.3: linear probe accuracy for symmetry labels, vs a composition-only baseline.

Exps 1.1/1.2 (symmetry_separability.py) asked whether symmetry shows up as *geometry*
-- are same-label embeddings closer in cosine. This asks the different and more
relevant question for the linear representation hypothesis: is the label linearly
*decodable*. A logistic regression is a linear readout, so accuracy well above the
baselines is evidence the label lives in a linear subspace of the residual stream.

Three numbers per (layer, label, split), and only the gaps between them mean anything:

  majority     - always predict the most common class. The floor. Space groups are
                 badly imbalanced, so this is high and raw accuracy is misleading.
  composition  - logistic regression on element-count features alone, no embeddings.
                 This is "what you get from knowing the chemistry", the control that
                 Exp 1.2 did by restricting pairs.
  embedding    - logistic regression on the layer's residual stream.

  embedding - composition is the quantity of interest.

Two splits, because they answer different questions:

  random      - rows split at random. The same formula can land in both sides, so
                composition can win by memorizing formula -> label and the embedding
                probe can too. Standard, but flattering to both.
  by_formula  - every structure sharing a formula goes wholly to train or to test.
                Nothing can be looked up; the probe must generalize to unseen
                chemistry. This is the honest test of a symmetry representation.

Steps:
  1. Read formulas from the CIF data_ headers (same source as symmetry_separability.py)
  2. Load a layer's embeddings, join metadata labels on id
  3. Drop classes with fewer than MIN_CLASS_COUNT members (can't split or score them)
  4. Subsample to SAMPLE_SIZE -- full 1.6M x 1024 with 750 classes is not tractable
  5. Fit majority / composition / embedding probes under both splits
  6. Save the table to OUT_CSV

Usage:
    python symmetry_probe.py                 # layer 5, all three labels
    python symmetry_probe.py 0 5 10 14       # a subset of layers
    LABEL_COLS=space_group_symbol python symmetry_probe.py    # one label only
    PARTITION=test VARIANT=nosym python symmetry_probe.py    # held-out, symmetry stripped

PARTITION (all|train|val|test) is required; DATASET (v1_all|v1_mp) and VARIANT
(full|nosym) default to v1_all/full. Output lands in
analysis/<DATASET>/<VARIANT>/<PARTITION>/.
"""

import gzip
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
from utils import load_labeled_embeddings, filter_partition, analysis_dir

# -------------------------------
# Configuration
# -------------------------------
LAYERS = [int(a) for a in sys.argv[1:]] or [5]
METADATA_PATH = "./metadata.parquet"
PKL_PATH = "./CrystaLLM/cifs_v1_prep.pkl.gz"
# Which embeddings to read and which slice of CrystaLLM's split to run on. PARTITION has
# no default on purpose: 89.6% of the labelled structures are in the model's own training
# set, so a silent default would quietly measure memorization.
DATASET = os.environ.get("DATASET", "v1_all")
VARIANT = os.environ.get("VARIANT", "full")
PARTITION = os.environ.get("PARTITION")
if PARTITION is None:
    raise SystemExit("Set PARTITION=all|train|val|test, e.g. "
                     "PARTITION=test python symmetry_probe.py")
OUTPUT_DIR = str(analysis_dir(DATASET, VARIANT, PARTITION))
# space_group_symbol (207 classes), point_group (32), wyckoff_letters (750, and
# 461k rows are null so that label loses ~28% of the data).
LABEL_COLS = os.environ.get(
    "LABEL_COLS", "space_group_symbol,point_group,wyckoff_letters").split(",")

# Label(s) in the filename, matching symmetry_separability.py. A fixed name silently
# overwrote a completed point-group run with a later space-group one.
OUT_CSV = os.path.join(OUTPUT_DIR, f"symmetry_probe_{'-'.join(LABEL_COLS)}.csv")
# rows kept per layer; lbfgs on 1.6M x 1024 x 750 classes is not tractable.
# Lower it (SAMPLE_SIZE=5000) for a quick smoke test.
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 150_000))
TEST_FRACTION = 0.25
MIN_CLASS_COUNT = 200    # in the full data, before subsampling
MAX_ITER = int(os.environ.get("MAX_ITER", 200))
RANDOM_SEED = 1
DATA_RE = re.compile(r"^data_(\S+)", re.M)
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")



def composition_matrix(formulas):
    """Element-count features: one column per element, value = atoms of it in the cell.

    'Zn3In3Ga3O12' -> Zn:3, In:3, Ga:3, O:12. Sparse because any one formula touches
    a handful of the ~100 elements. Total atom count is a column sum, so the linear
    model gets cell size for free without it being a separate feature.
    """
    rows, cols, vals = [], [], []
    elements = {}
    for i, formula in enumerate(formulas):
        for symbol, count in ELEMENT_RE.findall(formula):
            j = elements.setdefault(symbol, len(elements))
            rows.append(i)
            cols.append(j)
            vals.append(float(count) if count else 1.0)
    X = csr_matrix((vals, (rows, cols)), shape=(len(formulas), len(elements)))
    return X.toarray().astype(np.float32), list(elements)


def probe(X_train, y_train, X_test, y_test):
    """Fit a multinomial logistic regression and score it. Returns (accuracy, macro F1).

    Macro F1 rides along because the classes are severely imbalanced -- accuracy alone
    can look respectable while the probe only ever names the few biggest space groups.
    """
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=MAX_ITER, n_jobs=-1)
    model.fit(scaler.transform(X_train), y_train)
    pred = model.predict(scaler.transform(X_test))
    return (accuracy_score(y_test, pred),
            f1_score(y_test, pred, average="macro", zero_division=0))


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
        print(f"\n{'='*70}\nLayer {layer}\n{'='*70}")

        # -------------------------------
        # Load and intersect datasets
        # -------------------------------
        df = load_labeled_embeddings(layer, dataset=DATASET, metadata_path=METADATA_PATH,
                                     label_cols=tuple(LABEL_COLS), variant=VARIANT)
        df = filter_partition(df, PARTITION)
        df = df.merge(formula_df, on="id", how="inner")
        df = df.dropna(subset=["formula"]).reset_index(drop=True)

        for label_col in LABEL_COLS:
            print(f"\n--- {label_col} ---")

            # -------------------------------
            # Keep rows with a usable label, drop classes too rare to score
            # -------------------------------
            keep = df[label_col].notna() & (df[label_col] != "")
            sub = df[keep].reset_index(drop=True)
            counts = sub[label_col].value_counts()
            common = counts[counts >= MIN_CLASS_COUNT].index
            sub = sub[sub[label_col].isin(common)].reset_index(drop=True)
            print(f"  {len(sub):,} structures, {len(common):,} classes with "
                  f">={MIN_CLASS_COUNT} members (of {len(counts):,} total)")

            # -------------------------------
            # Subsample to a tractable size
            # -------------------------------
            if len(sub) > SAMPLE_SIZE:
                pick = rng.choice(len(sub), SAMPLE_SIZE, replace=False)
                sub = sub.iloc[pick].reset_index(drop=True)
            print(f"  sampled to {len(sub):,} rows")

            X_emb = np.vstack(sub["embedding"].values).astype(np.float32)
            X_comp, elements = composition_matrix(sub["formula"].tolist())
            y = pd.factorize(sub[label_col])[0]
            formula_code = pd.factorize(sub["formula"])[0]
            print(f"  embeddings {X_emb.shape}, composition {X_comp.shape} "
                  f"({len(elements)} elements), {formula_code.max()+1:,} formulas")

            for split in ("random", "by_formula"):
                # -------------------------------
                # Build the test mask
                # -------------------------------
                if split == "random":
                    order = rng.permutation(len(sub))
                    is_test = np.zeros(len(sub), dtype=bool)
                    is_test[order[:int(TEST_FRACTION * len(sub))]] = True
                else:
                    # Whole formulas go one way or the other, so no formula the probe
                    # trained on reappears at test time.
                    n_formula = formula_code.max() + 1
                    formula_is_test = rng.random(n_formula) < TEST_FRACTION
                    is_test = formula_is_test[formula_code]

                # A class absent from train can never be predicted; scoring it would
                # just charge the probe for labels it was never shown.
                trainable = np.isin(y, np.unique(y[~is_test]))
                is_train = (~is_test) & trainable
                is_eval = is_test & trainable

                y_train, y_test = y[is_train], y[is_eval]
                majority = np.bincount(y_train).argmax()
                majority_acc = float((y_test == majority).mean())

                comp_acc, comp_f1 = probe(X_comp[is_train], y_train,
                                          X_comp[is_eval], y_test)
                emb_acc, emb_f1 = probe(X_emb[is_train], y_train,
                                        X_emb[is_eval], y_test)

                print(f"  [{split:10s}] train {is_train.sum():,} / test {is_eval.sum():,} "
                      f"| majority {majority_acc:.4f} | composition {comp_acc:.4f} "
                      f"| embedding {emb_acc:.4f} | margin {emb_acc - comp_acc:+.4f}")

                results.append(dict(
                    layer=layer, label_col=label_col, split=split,
                    n_train=int(is_train.sum()), n_test=int(is_eval.sum()),
                    n_classes=int(len(np.unique(y_train))),
                    majority_acc=majority_acc,
                    composition_acc=comp_acc, composition_macro_f1=comp_f1,
                    embedding_acc=emb_acc, embedding_macro_f1=emb_f1,
                    margin_over_composition=emb_acc - comp_acc,
                    margin_over_majority=emb_acc - majority_acc,
                ))

        del df

    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
