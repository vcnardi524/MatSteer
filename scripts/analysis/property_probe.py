#!/usr/bin/env python3
"""Linear probe R^2 for a CONTINUOUS property across layers, vs a composition baseline.

The regression twin of symmetry_probe.py, which does the same experiment for categorical
symmetry labels. Same question: is the property linearly *decodable* from the residual
stream, beyond what the chemistry already tells you.

Four numbers per (layer, split), and only the gaps between them mean anything:

  mean         - always predict the training mean. R^2 = 0 by construction on the
                 training distribution; on a held-out split it can go slightly negative.
  lookup       - the mean value among training rows with the SAME formula, falling back
                 to the global mean for formulas never seen. Does no generalizing at
                 all, so it is the ceiling on pure memorization of chemistry.
  composition  - ridge on element-count features alone, no embeddings. "What you get
                 from knowing the formula."
  embedding    - the same ridge on the layer's residual stream.

  embedding - max(composition, lookup) is the quantity of interest, reported as
  margin_over_best_baseline.

WHY THE BASELINE MATTERS MORE HERE THAN FOR SYMMETRY
----------------------------------------------------
density_atomic is cell volume over atom count, and composition_matrix() hands the model
element counts whose ROW SUM is the atom count. So the composition baseline gets half of
the target's definition for free. A high embedding R^2 means nothing on its own; only
the margin over composition says the residual stream knows something the formula does not.

Fit on CrystaLLM's train split, scored on val -- never overlapping. Two splits:
  random      - val as it comes. A formula in val may also appear in train, so both the
                lookup baseline and the probe can win by memorizing formula -> value.
  by_formula  - val rows whose formula appears nowhere in the rows actually fit on.
                Nothing can be looked up. The honest test.
Under by_formula the lookup baseline collapses to the mean baseline exactly; that
identity is the check that the split is doing what it claims.

Usage:
    python scripts/analysis/property_probe.py --property density_atomic
    python scripts/analysis/property_probe.py --property density_atomic --layers 0 7 14
    python scripts/analysis/property_probe.py --property efermi \
        --labels metadata_mp.parquet --id-col material_id --dataset v1_mp --no-log
"""
import argparse
import gzip
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import analysis_dir, load_split_index, DATASETS, VARIANTS
from manifold import embedding_files

import pyarrow.parquet as pq

DATA_RE = re.compile(r"^data_(\S+)", re.MULTILINE)
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
PKL_PATH = "CrystaLLM/cifs_v1_prep.pkl.gz"
SEED = 42


def composition_matrix(formulas):
    """Element counts, one column per element. Same construction as symmetry_probe.py.

    The row sum is the total atom count, so a linear model gets cell size for free --
    which is exactly why this baseline is hard to beat for a per-atom property.
    """
    rows, cols, vals, elements = [], [], [], {}
    for i, formula in enumerate(formulas):
        for symbol, count in ELEMENT_RE.findall(formula):
            if not symbol:
                continue
            j = elements.setdefault(symbol, len(elements))
            rows.append(i); cols.append(j); vals.append(float(count) if count else 1.0)
    X = csr_matrix((vals, (rows, cols)), shape=(len(formulas), len(elements)))
    return X.toarray().astype(np.float32), list(elements)


def formula_lookup(formula_code, y, is_train, eval_pool, global_mean):
    """Memorization baseline: the mean value among training rows sharing this formula."""
    tbl = pd.DataFrame({"f": formula_code[is_train], "y": y[is_train]})
    best = tbl.groupby("f")["y"].mean()
    return pd.Series(formula_code[eval_pool]).map(best).fillna(global_mean).to_numpy(float)


def fit_predict(X_train, y_train, X_test, alpha):
    scaler = StandardScaler().fit(X_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(X_train), y_train)
    return model.predict(scaler.transform(X_test))


def score(y_true, pred):
    """R^2 against the TEST set's own variance, plus RMSE in the target's units."""
    ss_res = float(((y_true - pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return (1 - ss_res / ss_tot if ss_tot else float("nan"),
            float(np.sqrt(((y_true - pred) ** 2).mean())))


def stream_layer(layer, dataset, variant, keep_ids, batch_size=50_000):
    """[id, embedding] for the ids we need, streamed -- a layer does not fit in memory."""
    ids, vecs = [], []
    for path in embedding_files(layer, dataset, variant):
        for rb in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                    columns=["id", "embedding"]):
            d = rb.to_pandas()
            d = d[d["id"].isin(keep_ids)]
            if d.empty:
                continue
            ids.extend(d["id"].tolist())
            vecs.append(np.vstack(d["embedding"].to_numpy()).astype(np.float32))
    return pd.DataFrame({"id": ids}), (np.vstack(vecs) if vecs else np.empty((0, 1024), np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="density_atomic")
    ap.add_argument("--labels", default="density_atomic_v1.parquet")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=list(range(16)))
    ap.add_argument("--dataset", default="v1_all", choices=list(DATASETS))
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--train-sample", type=int, default=200_000)
    ap.add_argument("--eval-sample", type=int, default=50_000)
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularisation")
    ap.add_argument("--log", action=argparse.BooleanOptionalAction, default=True,
                    help="Predict log10 of the property. On for density (right-skewed, "
                         "spans 2.5-3734); pass --no-log for a signed property like efermi.")
    ap.add_argument("--min-value", type=float, default=None)
    ap.add_argument("--max-value", type=float, default=None)
    args = ap.parse_args()

    print(f"property={args.property} labels={args.labels} dataset={args.dataset} "
          f"variant={args.variant} log10={args.log}")
    print(f"layers={args.layers}")

    lab = pd.read_parquet(args.labels, columns=[args.id_col, args.property]).dropna()
    if args.id_col != "id":
        lab = lab.rename(columns={args.id_col: "id"})
    lab[args.property] = pd.to_numeric(lab[args.property], errors="coerce")
    lab = lab.dropna(subset=[args.property])
    if args.min_value is not None:
        lab = lab[lab[args.property] >= args.min_value]
    if args.max_value is not None:
        lab = lab[lab[args.property] <= args.max_value]
    if args.log:
        lab = lab[lab[args.property] > 0]
    print(f"  {len(lab):,} labelled structures")

    print(f"Loading formulas from {PKL_PATH} ...")
    with gzip.open(PKL_PATH, "rb") as f:
        cifs = pickle.load(f)
    ids, forms = [], []
    for cid, cif in cifs:
        m = DATA_RE.search(cif)
        if m:
            ids.append(cid); forms.append(m.group(1))
    del cifs
    lab = lab.merge(pd.DataFrame({"id": ids, "formula": forms}), on="id", how="inner")
    print(f"  {len(lab):,} with a data_ header formula")

    split = load_split_index().set_index("id")["split"]
    lab["pool"] = lab["id"].map(split)
    lab = lab[lab["pool"].isin(["train", "val"])].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    tr = np.flatnonzero(lab["pool"].to_numpy() == "train")
    ev = np.flatnonzero(lab["pool"].to_numpy() == "val")
    if len(tr) > args.train_sample:
        tr = rng.choice(tr, args.train_sample, replace=False)
    if args.eval_sample and len(ev) > args.eval_sample:
        ev = rng.choice(ev, args.eval_sample, replace=False)
    lab = lab.iloc[np.concatenate([tr, ev])].reset_index(drop=True)
    is_train = np.arange(len(lab)) < len(tr)
    print(f"  sampled to {is_train.sum():,} train / {(~is_train).sum():,} eval")

    y = lab[args.property].to_numpy(float)
    if args.log:
        y = np.log10(y)
    formula_code = pd.factorize(lab["formula"])[0]
    X_comp, elements = composition_matrix(lab["formula"].tolist())
    train_formulas = np.unique(formula_code[is_train])
    unseen = ~np.isin(formula_code, train_formulas)[~is_train]
    print(f"  composition {X_comp.shape} ({len(elements)} elements), "
          f"{formula_code.max()+1:,} formulas; {unseen.sum():,} eval rows unseen")

    keep = set(lab["id"])
    rows = []
    for layer in args.layers:
        print(f"\n{'='*70}\nLayer {layer}\n{'='*70}")
        got, E = stream_layer(layer, args.dataset, args.variant, keep)
        order = pd.Series(np.arange(len(got)), index=got["id"]).reindex(lab["id"])
        if order.isna().any():
            print(f"  ! {int(order.isna().sum()):,} rows have no embedding -- skipped layer")
            continue
        X_emb = E[order.to_numpy().astype(int)]
        gm = float(y[is_train].mean())
        comp_pred = fit_predict(X_comp[is_train], y[is_train], X_comp[~is_train], args.alpha)
        emb_pred = fit_predict(X_emb[is_train], y[is_train], X_emb[~is_train], args.alpha)
        look_pred = formula_lookup(formula_code, y, is_train, ~is_train, gm)
        y_pool = y[~is_train]

        for sp in ("random", "by_formula"):
            sel = np.ones(len(y_pool), bool) if sp == "random" else unseen
            if not sel.any():
                print(f"  [{sp:10s}] no eval rows -- skipped"); continue
            yt = y_pool[sel]
            mean_r2, mean_rmse = score(yt, np.full(sel.sum(), gm))
            look_r2, look_rmse = score(yt, look_pred[sel])
            comp_r2, comp_rmse = score(yt, comp_pred[sel])
            emb_r2, emb_rmse = score(yt, emb_pred[sel])
            print(f"  [{sp:10s}] test {sel.sum():,} | mean {mean_r2:+.4f} "
                  f"| lookup {look_r2:+.4f} | composition {comp_r2:+.4f} "
                  f"| embedding {emb_r2:+.4f} | margin {emb_r2-max(comp_r2,look_r2):+.4f}")
            rows.append(dict(layer=layer, property=args.property, split=sp,
                             n_train=int(is_train.sum()), n_test=int(sel.sum()),
                             log10=args.log,
                             mean_r2=mean_r2, mean_rmse=mean_rmse,
                             lookup_r2=look_r2, lookup_rmse=look_rmse,
                             composition_r2=comp_r2, composition_rmse=comp_rmse,
                             embedding_r2=emb_r2, embedding_rmse=emb_rmse,
                             margin_over_composition=emb_r2-comp_r2,
                             margin_over_lookup=emb_r2-look_r2,
                             margin_over_best_baseline=emb_r2-max(comp_r2, look_r2)))
        del X_emb, E

    if not rows:
        raise SystemExit("No rows produced.")
    out = analysis_dir(args.dataset, args.variant, "val") / \
        f"property_probe_{args.property.replace('.', '_')}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")
    print(f"\nSaved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
