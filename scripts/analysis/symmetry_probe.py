#!/usr/bin/env python3
"""Exp 1.3: linear probe accuracy for symmetry labels, vs a composition-only baseline.

Exps 1.1/1.2 (symmetry_separability.py) asked whether symmetry shows up as *geometry*
-- are same-label embeddings closer in cosine. This asks the different and more
relevant question for the linear representation hypothesis: is the label linearly
*decodable*. A logistic regression is a linear readout, so accuracy well above the
baselines is evidence the label lives in a linear subspace of the residual stream.

Four numbers per (layer, label, split), and only the gaps between them mean anything:

  majority     - always predict the most common class. The floor. Space groups are
                 badly imbalanced, so this is high and raw accuracy is misleading.
  lookup       - a dictionary: the commonest label among training rows with the same
                 formula, majority class for formulas never seen. Does no generalizing
                 at all, so it is the ceiling on pure memorization of chemistry.
  composition  - the same classifier on element-count features alone, no embeddings.
                 Weaker than lookup on formulas it has seen, stronger on ones it has
                 not. This is "what you get from knowing the chemistry", the control
                 that Exp 1.2 did by restricting pairs.
  embedding    - the same classifier on the layer's residual stream.

  embedding - max(composition, lookup) is the quantity of interest, reported as
  margin_over_best_baseline.

WHERE THE PROBE'S TRAIN AND TEST ROWS COME FROM
-----------------------------------------------
Fixed, not configurable: the probe FITS on CrystaLLM's train split and is SCORED on
CrystaLLM's val split. The two never overlap, so no row is ever both fit and scored,
and every reported number is measured on structures the probe has not seen.

An earlier version cut a single partition 75/25 internally, so PARTITION=val meant
"fit on 75% of val, score on the other 25%". That is a valid experiment but a smaller
one, and it left the 1.46M-row train split unused.

Two splits, because they answer different questions:

  random      - CrystaLLM's val split as it comes. A formula in val may also appear in
                train, so composition can win by memorizing formula -> label and the
                embedding probe can too. Standard, but flattering to both.
  by_formula  - val rows whose formula appears nowhere in the rows actually fit on.
                Nothing can be looked up; the probe must generalize to unseen
                chemistry. This is the honest test of a symmetry representation.

Steps:
  1. Read formulas from the CIF data_ headers (same source as symmetry_separability.py)
  2. Load a layer's embeddings, join metadata labels on id
  3. Assign each row to the train pool, the eval pool, or neither
  4. Subsample the train pool to TRAIN_SAMPLE; the eval pool is used whole
  5. Drop classes with fewer than MIN_CLASS_COUNT rows *in the sampled train set*, so
     the threshold is a real guarantee about what the probe was fit on rather than a
     statement about a pool it only partly saw
  6. Fit majority / lookup / composition / embedding probes once, score both splits
  7. Save one table per label to symmetry_probe_<label>.csv

Usage:
    python symmetry_probe.py                 # layer 5, all three labels
    python symmetry_probe.py 0 5 10 14       # a subset of layers
    LABEL_COLS=space_group_symbol python symmetry_probe.py    # one label only
    VARIANT=nosym python symmetry_probe.py                    # symmetry-stripped CIFs

DATASET (v1_all|v1_mp) and VARIANT (full|nosym) default to v1_all/full. Output lands
in analysis/<DATASET>/<VARIANT>/val/ -- keyed on the eval partition, since that is
what the numbers describe.
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
from utils import load_labeled_embeddings, load_split_index, analysis_dir

# -------------------------------
# Configuration
# -------------------------------
LAYERS = [int(a) for a in sys.argv[1:]] or [5]
METADATA_PATH = "./metadata.parquet"
PKL_PATH = "./CrystaLLM/cifs_v1_prep.pkl.gz"
DATASET = os.environ.get("DATASET", "v1_all")
VARIANT = os.environ.get("VARIANT", "full")
# Fixed by design, not exposed as knobs: fit on train, score on val. Anything else
# risks scoring the probe on rows it was fit on, or on rows the language model itself
# was trained on (89.6% of the labelled structures are in the model's train split).
TRAIN_PARTITION = "train"
EVAL_PARTITION = "val"
OUTPUT_DIR = str(analysis_dir(DATASET, VARIANT, EVAL_PARTITION))
# space_group_symbol (207 classes), point_group (32), wyckoff_letters (750, and
# 461k rows are null so that label loses ~28% of the data).
LABEL_COLS = os.environ.get(
    "LABEL_COLS", "space_group_symbol,point_group,wyckoff_letters").split(",")

# One CSV per label, matching symmetry_separability.py. Keyed on the label rather than
# the whole run so a single job can do every label from one load of each layer's 14.6 GB
# parquet -- splitting labels across jobs re-reads every layer once per job.
def out_csv(label_col):
    return os.path.join(OUTPUT_DIR, f"symmetry_probe_{label_col}.csv")
# Training rows per layer; the full 1.46M x 1024 x 750 classes is not tractable. 400k
# is ~27% of the labelled train split, enough that a class with ~74 members there still
# lands MIN_CLASS_COUNT rows in the fit.
# Lower it (TRAIN_SAMPLE=5000) for a quick smoke test.
TRAIN_SAMPLE = int(os.environ.get("TRAIN_SAMPLE", 400_000))
# Val is scored whole (~162k labelled rows). Evaluation is one predict(), so capping it
# would buy nothing and only add sampling noise to the number being reported.
EVAL_SAMPLE = int(os.environ.get("EVAL_SAMPLE", 0)) or None
# Counted on the SAMPLED train rows, so it is a real guarantee about the fit. At 200
# against the full pool the old run kept only 44 of 205 space groups and 14 of 32 point
# groups -- discarding 79% and 56% of the classes to save 1.6% and 0.6% of the rows.
MIN_CLASS_COUNT = int(os.environ.get("MIN_CLASS_COUNT", 20))
MAX_ITER = int(os.environ.get("MAX_ITER", 200))
RANDOM_SEED = 1
DATA_RE = re.compile(r"^data_(\S+)", re.M)
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")



def assign_pools(df):
    """Tag each row train/eval by CrystaLLM's split and drop everything else.

    One pass over the split index rather than two filter_partition calls, because the
    embedding frame is ~50G a layer and a second copy does not fit.
    """
    split = load_split_index().set_index("id")["split"]
    pool = df["id"].map(split)
    out = df[pool.isin([TRAIN_PARTITION, EVAL_PARTITION])].copy()
    out["pool"] = np.where(pool[pool.isin([TRAIN_PARTITION, EVAL_PARTITION])]
                           == TRAIN_PARTITION, "train", "eval")
    n_tr = int((out["pool"] == "train").sum())
    print(f"  pools: train {n_tr:,} / eval {len(out) - n_tr:,} "
          f"(of {len(df):,} labelled rows)")
    if not n_tr or n_tr == len(out):
        raise SystemExit("One of the pools is empty; check the split index.")
    return out.reset_index(drop=True)


def formula_lookup(formula_code, y, is_train, eval_pool, majority):
    """Memorization baseline: predict the commonest label among training rows sharing
    this formula, falling back to the majority class for formulas never seen.

    This is the ceiling on "how much of the probe's accuracy is just knowing the
    chemistry". Unlike the composition probe it does no generalizing at all -- it is a
    dictionary. Under the by_formula split every eval formula is unseen by
    construction, so this collapses to the majority baseline exactly; that identity is
    a useful check that the split is doing what it claims.
    """
    tbl = pd.DataFrame({"f": formula_code[is_train], "y": y[is_train]})
    # Commonest label per formula; ties broken by label index so runs are reproducible.
    vc = tbl.value_counts(["f", "y"]).reset_index(name="n")
    best = (vc.sort_values(["n", "y"]).drop_duplicates("f", keep="last")
              .set_index("f")["y"])
    pred = pd.Series(formula_code[eval_pool]).map(best)
    return pred.fillna(majority).to_numpy(dtype=int)


def composition_matrix(formulas):
    """Element-count features: one column per element, value = atoms of it in the cell.

    'Zn3In3Ga3O12' -> Zn:3, In:3, Ga:3, O:12. Sparse because any one formula touches
    a handful of the ~100 elements.

    CrystaLLM's preprocessing rewrites the data_ header to the NONREDUCED formula, so
    these counts describe the actual cell, not the reduced unit. Total atom count is
    therefore a row sum, and the linear model gets cell size for free without it being
    a separate feature -- which is part of why this baseline is hard to beat.
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


def probe(X_train, y_train, X_test):
    """Fit a multinomial logistic regression, return its predictions on X_test.

    Both splits are fit on the same train pool and differ only in which eval rows
    count, so the caller fits once here and scores the predictions under each mask.
    """
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(scaler.transform(X_train), y_train)
    return model.predict(scaler.transform(X_test))


def score(y_true, pred):
    """Accuracy and macro F1.

    Macro F1 rides along because the classes are severely imbalanced -- accuracy alone
    can look respectable while the probe only ever names the few biggest space groups.
    """
    return (accuracy_score(y_true, pred),
            f1_score(y_true, pred, average="macro", zero_division=0))


def main():
    # Echo the resolved config. sbatch --export splits on commas, so a multi-label
    # LABEL_COLS passed there silently arrives truncated -- printing it makes a
    # half-configured run obvious in the log instead of looking like a clean success.
    print(f"dataset={DATASET} variant={VARIANT} "
          f"fit={TRAIN_PARTITION} score={EVAL_PARTITION}")
    print(f"labels={LABEL_COLS} layers={LAYERS}")
    print(f"train_sample={TRAIN_SAMPLE:,} eval_sample={EVAL_SAMPLE or 'all'} "
          f"min_class_count={MIN_CLASS_COUNT}")


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

    results = {label: [] for label in LABEL_COLS}
    for layer in LAYERS:
        print(f"\n{'='*70}\nLayer {layer}\n{'='*70}")
        # Re-seeded per layer so every layer is fit on the SAME sampled rows. The
        # composition and lookup baselines do not depend on the layer, so any wobble
        # in them across layers is sampling noise that would leak straight into
        # margin_over_composition -- the one number the experiment is about.
        rng = np.random.default_rng(RANDOM_SEED)

        # -------------------------------
        # Load and intersect datasets
        # -------------------------------
        df = load_labeled_embeddings(layer, dataset=DATASET, metadata_path=METADATA_PATH,
                                     label_cols=tuple(LABEL_COLS), variant=VARIANT)
        df = assign_pools(df)
        df = df.merge(formula_df, on="id", how="inner")
        df = df.dropna(subset=["formula"]).reset_index(drop=True)

        for label_col in LABEL_COLS:
            print(f"\n--- {label_col} ---")

            # -------------------------------
            # Keep rows with a usable label, then subsample each pool
            # -------------------------------
            keep = df[label_col].notna() & (df[label_col] != "")
            sub = df[keep].reset_index(drop=True)
            tr_idx = np.flatnonzero(sub["pool"].to_numpy() == "train")
            ev_idx = np.flatnonzero(sub["pool"].to_numpy() == "eval")
            if len(tr_idx) > TRAIN_SAMPLE:
                tr_idx = rng.choice(tr_idx, TRAIN_SAMPLE, replace=False)
            if EVAL_SAMPLE and len(ev_idx) > EVAL_SAMPLE:
                ev_idx = rng.choice(ev_idx, EVAL_SAMPLE, replace=False)
            sub = sub.iloc[np.concatenate([tr_idx, ev_idx])].reset_index(drop=True)
            in_train_pool = np.arange(len(sub)) < len(tr_idx)
            print(f"  sampled to {in_train_pool.sum():,} train / "
                  f"{(~in_train_pool).sum():,} eval rows")

            # Threshold on the SAMPLED train rows: a class with too few examples there
            # is one the probe cannot learn, whatever the full pool holds.
            counts = sub.loc[in_train_pool, label_col].value_counts()
            common = counts[counts >= MIN_CLASS_COUNT].index
            fit_class = sub[label_col].isin(common).to_numpy()
            print(f"  {len(common):,} classes with >={MIN_CLASS_COUNT} sampled train "
                  f"rows (of {len(counts):,} present); dropping "
                  f"{(~fit_class).sum():,} rows")
            sub = sub[fit_class].reset_index(drop=True)
            in_train_pool = in_train_pool[fit_class]

            X_emb = np.vstack(sub["embedding"].values).astype(np.float32)
            X_comp, elements = composition_matrix(sub["formula"].tolist())
            y = pd.factorize(sub[label_col])[0]
            formula_code = pd.factorize(sub["formula"])[0]
            print(f"  embeddings {X_emb.shape}, composition {X_comp.shape} "
                  f"({len(elements)} elements), {formula_code.max()+1:,} formulas")
            # Formulas the probe actually gets fit on -- what by_formula holds out against.
            train_formulas = np.unique(formula_code[in_train_pool])

            # A class absent from train can never be predicted; scoring it would just
            # charge the probe for labels it was never shown.
            trainable = np.isin(y, np.unique(y[in_train_pool]))
            is_train = in_train_pool & trainable
            eval_pool = (~in_train_pool) & trainable

            # Fit once. Both splits use these same training rows and differ only in
            # which eval rows are allowed to count, so refitting per split would
            # burn a second lbfgs run to reproduce the identical model.
            y_train = y[is_train]
            majority = np.bincount(y_train).argmax()
            print(f"  fitting on {is_train.sum():,} rows, "
                  f"{len(np.unique(y_train)):,} classes ...")
            comp_pred = probe(X_comp[is_train], y_train, X_comp[eval_pool])
            emb_pred = probe(X_emb[is_train], y_train, X_emb[eval_pool])

            lookup_pred = formula_lookup(formula_code, y, is_train, eval_pool, majority)

            # Row i of *_pred corresponds to the i'th True in eval_pool, so each split
            # is scored by masking the predictions rather than predicting again.
            unseen = ~np.isin(formula_code, train_formulas)[eval_pool]
            y_pool = y[eval_pool]

            for split in ("random", "by_formula"):
                sel = np.ones(len(y_pool), dtype=bool) if split == "random" else unseen
                if not sel.any():
                    print(f"  [{split:10s}] no eval rows survive -- skipped")
                    continue
                y_test = y_pool[sel]
                majority_acc = float((y_test == majority).mean())
                look_acc, look_f1 = score(y_test, lookup_pred[sel])
                comp_acc, comp_f1 = score(y_test, comp_pred[sel])
                emb_acc, emb_f1 = score(y_test, emb_pred[sel])

                print(f"  [{split:10s}] train {is_train.sum():,} / test {sel.sum():,} "
                      f"| majority {majority_acc:.4f} | lookup {look_acc:.4f} "
                      f"| composition {comp_acc:.4f} | embedding {emb_acc:.4f} "
                      f"| margin {emb_acc - max(comp_acc, look_acc):+.4f}")

                results[label_col].append(dict(
                    layer=layer, label_col=label_col, split=split,
                    train_partition=TRAIN_PARTITION, eval_partition=EVAL_PARTITION,
                    n_train=int(is_train.sum()), n_test=int(sel.sum()),
                    n_classes=int(len(np.unique(y_train))),
                    majority_acc=majority_acc,
                    lookup_acc=look_acc, lookup_macro_f1=look_f1,
                    composition_acc=comp_acc, composition_macro_f1=comp_f1,
                    embedding_acc=emb_acc, embedding_macro_f1=emb_f1,
                    margin_over_composition=emb_acc - comp_acc,
                    margin_over_lookup=emb_acc - look_acc,
                    margin_over_best_baseline=emb_acc - max(comp_acc, look_acc),
                    margin_over_majority=emb_acc - majority_acc,
                ))

        del df

    for label_col, rows in results.items():
        if not rows:
            print(f"\nNo rows for {label_col} -- nothing written")
            continue
        out = pd.DataFrame(rows)
        out.to_csv(out_csv(label_col), index=False)
        print(f"\nSaved {len(out)} rows to {out_csv(label_col)}")


if __name__ == "__main__":
    main()
