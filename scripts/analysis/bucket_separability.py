#!/usr/bin/env python3
"""Are the manifold's property buckets separable in embedding space at all?

Manifold steering assumes structures sharing a property bucket cluster together, so a
curve through the bucket means is a meaningful axis. Nothing tested that. It matters
because encode() does not recover the property -- R^2 against the identity line is -0.36
for formation energy and -0.07 for volume per atom, i.e. worse than predicting the mean --
and if the buckets are not separable in the first place, that follows automatically and so
does every steering null.

This is symmetry_separability.py's experiment with property buckets as the label. Buckets
are np.floor(value / width) by default -- the same expression as manifold.bucket_centroids,
so the groups here are
exactly the groups the curve was fitted through.

THREE MEASUREMENTS, each per layer
----------------------------------
  1. same bucket vs different bucket, with a global label-permutation null.
  2. the same restricted to pairs of the SAME COMPOSITION, with the null permuting buckets
     within each composition group. This control matters more here than it did for
     symmetry: formation energy is largely fixed by composition (within-prompt SD 0.076
     against between-prompt 0.602), so without it the ratio mostly measures the model
     knowing the chemistry.
  3. mean cosine as a function of bucket DISTANCE |g - h|. Property buckets are ORDERED,
     unlike space groups, so same-vs-different throws away most of the structure. This is
     the manifold premise stated as a measurement: adjacent buckets should be closest and
     similarity should fall off monotonically with separation. A flat curve means the
     buckets are not laid out along an axis.

NO N x N MATRIX. For unit-norm rows,
    sum_{i<j in g} x_i.x_j = (||S_g||^2 - n_g) / 2
with S_g the group's vector sum, and the cross term between two buckets is just
S_g . S_h / (n_g n_h). Both come from the same group sums, so measurement 3 is free.

Every number is computed twice, `raw` and `centered` (global mean removed before
normalizing). Transformer embeddings share a large common mean, so raw cosines all sit
near ~1 and ratios compress toward 1.0 whether or not signal exists -- centered is the
interpretable one.

Usage:
    python scripts/analysis/bucket_separability.py --property formation_energy_per_atom \
        --labels metadata_mp.parquet --id-col material_id --dataset v1_mp \
        --partition val --width 0.25
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_embeddings, filter_partition, analysis_dir, add_partition_args, DATASET_PKL

# Formulas come from the DATASET's own pickle. Hardcoding v1_all's here silently cut
# a v1_mp run to the 58,650-id overlap between the two corpora -- 48,820 rows after
# the other joins, against the 132,839 actually available.
PKL_PATH = None          # resolved from --dataset via utils.DATASET_PKL
DATA_RE = re.compile(r"^data_(\S+)", re.M)
RANDOM_SEED = 1


def group_sums(X, codes, n_groups):
    """(S, counts) -- per-group vector sums and sizes.

    Everything else is derived from these. float64 throughout: S reaches ~1e6 unit
    vectors and its squared norm ~1e12, which float32 cannot hold.
    """
    n = len(codes)
    M = csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(n_groups, n))
    S = M @ X.astype(np.float64, copy=False)
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    return S, counts


def within_pairs(S, counts):
    """(sum of within-group pairwise cosines, number of such pairs), per group."""
    return (np.einsum("ij,ij->i", S, S) - counts) / 2.0, counts * (counts - 1) / 2.0


def ratio_vs_null(Y, codes, n_groups, codes_null, n_groups_null, n_total=None,
                  parent_sum=None, parent_cnt=None):
    """same-group vs different-group mean cosine, and the same for permuted labels.

    `parent_*` scope the "different" set: for experiment 1 that is all pairs, for
    experiment 2 only pairs sharing a composition.
    """
    if parent_sum is None:
        tot = Y.astype(np.float64).sum(axis=0)
        parent_sum = (tot @ tot - n_total) / 2.0
        parent_cnt = n_total * (n_total - 1) / 2.0
    S, c = group_sums(Y, codes, n_groups)
    s, k = within_pairs(S, c)
    same, same_n = s.sum(), k.sum()
    mean_same = same / same_n
    mean_diff = (parent_sum - same) / (parent_cnt - same_n)
    Sn, cn = group_sums(Y, codes_null, n_groups_null)
    sn, kn = within_pairs(Sn, cn)
    null_same = sn.sum() / kn.sum()
    null_diff = (parent_sum - sn.sum()) / (parent_cnt - kn.sum())
    # DELTA, not the ratio, is the statistic to read for centered embeddings. Centering
    # forces the overall mean cosine to ~0, so mean_diff sits near zero and the ratio
    # explodes or flips sign on noise -- a smoke test produced -132.96. The difference is
    # stable under the same conditions. The ratio is kept because it is the readable one
    # for raw, where mean_diff is safely ~1.
    return dict(mean_same=mean_same, mean_diff=mean_diff,
                ratio=(mean_same / mean_diff) if abs(mean_diff) > 1e-3 else np.nan,
                delta=mean_same - mean_diff,
                null_delta=null_same - null_diff,
                null_ratio=(null_same / null_diff) if abs(null_diff) > 1e-3 else np.nan,
                same_pairs=same_n, diff_pairs=parent_cnt - same_n), S, c


def distance_curve(S, counts, min_count):
    """mean cosine against |bucket_i - bucket_j|, from the group sums alone.

    Between two buckets the mean cosine is S_g . S_h / (n_g n_h); within one it is the
    pairwise mean. Buckets thinner than min_count are dropped -- a 3-structure bucket's
    mean is noise and would dominate a distance bin it happens to land in.
    """
    keep = np.flatnonzero(counts >= min_count)
    if len(keep) < 2:
        return {}
    S, counts = S[keep], counts[keep]
    C = (S @ S.T) / np.outer(counts, counts)             # between-bucket means
    w, wn = within_pairs(S, counts)
    np.fill_diagonal(C, np.where(wn > 0, w / np.maximum(wn, 1), np.nan))
    out = {}
    for a in range(len(keep)):
        for b in range(a, len(keep)):
            d = int(abs(keep[a] - keep[b]))
            out.setdefault(d, []).append(C[a, b])
    return {d: float(np.mean(v)) for d, v in sorted(out.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--id-col", default="id",
                    help="Join key in --labels. v1_mp embeddings are keyed by "
                         "material_id, NOT by metadata_mp's own `id` column.")
    ap.add_argument("--width", type=float, required=True,
                    help="Bucket width, in the property's units. Match the manifold.")
    ap.add_argument("--closed", choices=("left", "right"), default="left",
                    help="Bucket edges; MUST match the manifold being tested, or these "
                         "are not the groups the curve was fitted through. Band gap "
                         "uses right, which isolates the metals at exactly 0.0 in their "
                         "own bucket -- see manifold.bucket_centroids.")
    ap.add_argument("--min-count", type=int, default=30,
                    help="Drop buckets thinner than this from the distance curve")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(16)))
    ap.add_argument("--out-stem", default=None)
    add_partition_args(ap)
    args = ap.parse_args()

    stem = args.out_stem or (f"bucket_separability_{args.property}_w{args.width:g}"
                             + ("_rc" if args.closed == "right" else ""))
    out_dir = str(analysis_dir(args.dataset, args.variant, args.partition,
                                    model=args.model))
    rng = np.random.default_rng(RANDOM_SEED)

    pkl = PKL_PATH or DATASET_PKL[args.dataset]
    print(f"Loading CIFs from {pkl} ...")
    with gzip.open(pkl, "rb") as f:
        cifs = pickle.load(f)
    ids, forms = [], []
    for cid, cif in cifs:
        m = DATA_RE.search(cif)
        if m:
            ids.append(cid); forms.append(m.group(1))
    del cifs
    formula_df = pd.DataFrame({"id": ids, "formula": forms})
    print(f"  {len(formula_df):,} CIFs with a data_ header formula")

    # labels loaded the fit_manifold.py way, NOT via load_labeled_embeddings -- that
    # hardcodes the join on `id`, and metadata_mp's `id` matches no embedding at all.
    lab = pd.read_parquet(args.labels, columns=[args.id_col, args.property]).dropna()
    if args.id_col != "id":
        lab = lab.rename(columns={args.id_col: "id"})
    lab[args.property] = pd.to_numeric(lab[args.property], errors="coerce")
    lab = lab.dropna(subset=[args.property])

    results, curves = [], {}
    for layer in args.layers:
        print(f"\n{'='*60}\nLayer {layer}\n{'='*60}")
        emb = load_embeddings(layer, dataset=args.dataset, variant=args.variant,
                              model=args.model)
        df = emb.merge(lab, on="id", how="inner")
        df = filter_partition(df, args.partition, model=args.model)
        df = df.merge(formula_df, on="id", how="inner").reset_index(drop=True)
        if df.empty:
            print("  no rows after the joins -- check --id-col"); continue

        X = np.vstack(df["embedding"].values)
        # the manifold's own bucketing, manifold.py:50
        v = df[args.property].to_numpy()
        raw_b = (np.floor(v / args.width) if args.closed == "left"
                 else np.ceil(v / args.width) - 1).astype(np.int64)
        bucket, bucket_vals = pd.factorize(raw_b, sort=True)
        n_bucket = len(bucket_vals)
        comp = pd.factorize(df["formula"])[0]
        n_total = len(X)
        sizes = np.bincount(bucket)
        print(f"Data shape: {X.shape}  ({n_bucket:,} buckets of width {args.width:g}, "
              f"{sizes.min():,}-{sizes.max():,} each, {comp.max()+1:,} formulas)")

        sub = np.bincount(comp)[comp] >= 2
        comp_s = pd.factorize(comp[sub])[0]
        bucket_s = bucket[sub]
        n_comp = comp_s.max() + 1
        cs = pd.factorize(comp_s.astype(np.int64) * n_bucket + bucket_s)[0]
        print(f"  composition-controlled subset: {int(sub.sum()):,} structures in "
              f"{n_comp:,} multi-entry composition groups")

        bucket_null = rng.permutation(bucket)
        base = np.argsort(comp_s, kind="stable")
        shuf = np.lexsort((rng.random(len(comp_s)), comp_s))
        bucket_s_null = np.empty_like(bucket_s)
        bucket_s_null[base] = bucket_s[shuf]
        cs_null = pd.factorize(comp_s.astype(np.int64) * n_bucket + bucket_s_null)[0]

        for center in (False, True):
            variant = "centered" if center else "raw"
            Y = X.astype(np.float32, copy=True)
            if center:
                Y -= Y.mean(axis=0, keepdims=True)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
            Y_s = Y[sub]

            e1, S, counts = ratio_vs_null(Y, bucket, n_bucket, bucket_null, n_bucket,
                                          n_total=n_total)
            Sc, cc = group_sums(Y_s, comp_s, n_comp)
            cw, cn = within_pairs(Sc, cc)
            e2, _, _ = ratio_vs_null(Y_s, cs, cs.max() + 1, cs_null, cs_null.max() + 1,
                                     parent_sum=cw.sum(), parent_cnt=cn.sum())
            curve = distance_curve(S, counts, args.min_count)
            curves[(layer, variant)] = curve

            results.append(dict(layer=layer, variant=variant, property=args.property,
                                width=args.width, n_structures=n_total,
                                n_buckets=n_bucket,
                                **{f"e1_{k}": v for k, v in e1.items()},
                                cc_n_structures=int(sub.sum()),
                                **{f"e2_{k}": v for k, v in e2.items()}))
            print(f"  [{variant:8s}] bucket delta={e1['delta']:+.4f} "
                  f"(null {e1['null_delta']:+.4f})  ratio={e1['ratio']:.4f} | "
                  f"same-composition delta={e2['delta']:+.4f} "
                  f"(null {e2['null_delta']:+.4f})")

    res = pd.DataFrame(results)
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False, float_format="%.6g")

    cur = pd.DataFrame([{"layer": l, "variant": v, "distance": d, "mean_cos": c}
                        for (l, v), cv in curves.items() for d, c in cv.items()])
    cur.to_csv(os.path.join(out_dir, f"{stem}_distance.csv"), index=False,
               float_format="%.6g")

    # SAME LAYOUT AS symmetry_separability.plot_results, deliberately: these are the
    # same two experiments with property buckets as the label, and the pair is only
    # comparable if drawn the same way. Colour = quantity, solid+dots = centered,
    # faded thin = raw, dotted = that quantity's permutation null.
    #
    # The Exp 1.1 RATIO cannot share the axis after centering: mean_cos_diff passes
    # through zero (formation energy runs +0.0044, +0.0005, -0.0029 over layers 0-4),
    # so same/diff blows up. It is plotted for parity with the symmetry figure but goes
    # off-scale by construction -- read the delta, which stays on-scale and is the
    # statistic that means something once the data are centered.
    lbl = f"bucket (w={args.width:g})"
    series = [
        ("e1_mean_same",  "C0", "-",  f"1.1 mean cos, same {lbl}"),
        ("e1_mean_diff",  "C0", "--", f"1.1 mean cos, diff {lbl}"),
        ("e1_ratio",      "C1", "-",  "1.1 ratio (off-scale when centered)"),
        ("e1_null_ratio", "C1", ":",  "1.1 ratio, permuted-label null"),
        ("e1_delta",      "C4", "-",  "1.1 delta (same - diff)"),
        ("e2_mean_same",  "C2", "-",  f"1.2 mean cos, same formula + same {lbl}"),
        ("e2_mean_diff",  "C2", "--", f"1.2 mean cos, same formula + diff {lbl}"),
        ("e2_ratio",      "C3", "-",  "1.2 ratio"),
        ("e2_null_ratio", "C3", ":",  "1.2 ratio, permuted-label null"),
        ("e2_delta",      "C5", "-",  "1.2 delta (same - diff)"),
    ]
    fig, ax = plt.subplots(figsize=(13, 8))
    for var in ("centered", "raw"):
        d = res[res.variant == var].sort_values("layer")
        if d.empty:
            continue
        centered = var == "centered"
        for col, colour, style, name in series:
            ax.plot(d["layer"], d[col], style, color=colour,
                    marker="o" if centered else None, markersize=4,
                    linewidth=1.8 if centered else 1.0,
                    alpha=1.0 if centered else 0.30,
                    label=f"{name} [{var}]")
    ax.axhline(1.0, color="grey", linewidth=0.8)    # ratio 1.0 = no separation
    ax.axhline(0.0, color="grey", linewidth=0.8)    # delta 0   = no separation
    ax.set_xlim(res["layer"].min(), res["layer"].max())
    ax.set_ylim(-0.1, 2.0)
    ax.set_xticks(sorted(res["layer"].unique()))
    ax.set_xlabel("transformer block (output of h[layer])")
    ax.set_ylabel("cosine / ratio")
    ax.set_title(f"Bucket separability by layer \u2014 {args.property}, width {args.width:g}"
                 f"{' (right-closed)' if args.closed == 'right' else ''}\n"
                 f"{args.model}  {args.dataset}/{args.variant}/{args.partition}   "
                 f"bold = centered embeddings, faded = raw")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=150,
                facecolor="white")

    # the decay curve
    fig2, ax = plt.subplots(figsize=(9, 5.8))
    cc = cur[cur.variant == "centered"]
    lays = sorted(cc.layer.unique())
    for i, l in enumerate(lays):
        g = cc[cc.layer == l].sort_values("distance")
        ax.plot(g.distance, g.mean_cos, "-", lw=1.8,
                color=plt.get_cmap("viridis")(i / max(len(lays) - 1, 1)), label=f"layer {l}")
    ax.set_xlabel(f"bucket separation  |g - h|   (1 = {args.width:g} in property units)")
    ax.set_ylabel("mean cosine between the two buckets")
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title(f"{args.property}: does similarity fall off with bucket separation?\n"
                 "centered; the manifold premise is a monotone decay from distance 0",
                 fontsize=11, loc="left")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"{stem}_distance.png"), dpi=150,
                 bbox_inches="tight", facecolor="white")
    print(f"\nSaved {out_dir}/{stem}.csv, .png and _distance.*")


if __name__ == "__main__":
    main()
