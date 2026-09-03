#!/usr/bin/env python3
"""
Does steering move the *distribution* of a generated property toward or away from
the ground-truth distribution of the prompts it was given?

One filled curve for the ground truth (the original test structures the prompts came
from), plus one line per steering strength for the generated structures. If steering
works, the lines walk away from the filled curve as |alpha| grows.

HOW A PROMPT BECOMES ONE NUMBER
-------------------------------
Every prompt is generated 3 times. Each prompt contributes ONE point: the mean of its
VALID samples (`validation/*.parquet:is_valid`). Prompts whose samples are all invalid
drop out entirely, so the surviving set shrinks as alpha rises and is biased toward
prompts the model can still generate at that strength -- the per-alpha n in the legend
is the honest record of that, and `--agg all` keeps every valid sample instead.

Averaging 1-3 samples also narrows each curve slightly relative to the true per-sample
spread. That is inherent to the "mean of the valid samples" definition, not a bug.

MEASURING THE SAME THING ON BOTH SIDES
--------------------------------------
A distribution comparison is only meaningful if both sides are measured by the same
instrument, so the ground truth is computed exactly the way the generated values were:

  density_atomic  `s.volume / len(s)` on the postprocessed structure, on BOTH sides
                  (the stored density_atomic_raw column for generated, the same
                  computation applied to cifs_v1_test.pkl.gz for the originals).
                  This column was unusable until 2026-08-19: postprocess silently
                  failed to restore symmetry operators, so pymatgen built only the
                  asymmetric unit while keeping the full _cell_volume, reading 2.673x
                  high at the median (53.48 vs 19.25 A^3/atom). Both are fixed now and
                  the column agrees with the CIF-text reading to a median ratio of
                  1.0000, so text_volume_per_atom is kept only as a cross-check.
  band_gap        MEGNet on the generated CIF. Ground truth is MEGNet on the ORIGINAL
                  test CIF (property_predictions/testset_baseline.parquet), NOT the DFT
                  value -- comparing MEGNet-on-generated against DFT-on-original would
                  fold the predictor's own bias into what looks like a steering effect.
                  `--truth dft` switches to metadata.parquet's dos_electronic.band_gap
                  for reference; it covers the NOMAD ids only.

Values are read from the RAW generated CIF by default (`<base>_raw`), since that is what
the model actually emitted; `--relaxed` reads the M3GNet-relaxed column instead.

All curves are restricted to the ids common to every plotted alpha, so each line
describes the same population of prompts.

ALPHAS AVAILABLE ON DISK (2026-08-17)
-------------------------------------
  band_gap        nosg family: -16, 0, 16, 25, 40, 60   (no 80)
                  sg   family: 11, 16, 25, 40
  density_atomic  0, 40, 80                             (no negative alpha)

OUTPUT
------
analysis/v1_all/test/plots/{prop}_distribution_shift.png

Usage:
    python scripts/plots/plot_steering_distribution_shift.py --property band_gap
    python scripts/plots/plot_steering_distribution_shift.py --property density_atomic
    python scripts/plots/plot_steering_distribution_shift.py --property band_gap \
        --alphas -16 0 16 40 60 --y count
"""
import argparse
import glob
import gzip
import os as _os
import pickle
import re
import sys as _sys

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from utils import analysis_dir, postprocess
from pymatgen.core import Structure

RANDOM_SEED = 42
TEST_PKL = "CrystaLLM/cifs_v1_test.pkl.gz"
NOMAD_METADATA = "metadata.parquet"
DFT_GAP_COL = "dos_electronic.band_gap"   # the clean gap in eV -- see README data conventions

# Colour roles. alpha=0 is the control and is deliberately neutral (the diverging
# midpoint); |alpha| grows light->dark along WARM_RAMP. Validated with the dataviz
# palette checker: worst adjacent CVD dE 18.7, normal-vision dE 23.2.
TRUTH_FILL = "#D9D9D6"
TRUTH_EDGE = "#6E6E68"
COL_ZERO = "#1A1A1A"
COL_NEG = "#2166AC"
WARM_RAMP = ["#E09000", "#C42121", "#5C0A2E"]

PROPS = {
    "band_gap": dict(
        results_dir="bandgap",
        col="predicted_bandgap_ev",
        label="Band gap (eV)",
        scale="linear",
        default_family="nosg",
        measure="predictor",
    ),
    "density_atomic": dict(
        results_dir="density_atomic",
        col="density_atomic",
        label="Volume per atom (A^3/atom)",
        scale="log",
        default_family=None,
        measure="predictor",
    ),
}

# Volume per atom read from the CIF text. Kept for the reference structures and as a
# cross-check: it agrees with the stored density_atomic column to a median ratio of
# 1.0000 (97.1% within 1%), the two differing only by CIF rounding, since _cell_volume
# and the lattice parameters are rounded independently.
#
# The stored column used to read ~2.7x high (median 53.48 vs 19.25 A^3/atom) because
# postprocess silently failed to restore symmetry operators, so pymatgen built only the
# asymmetric unit while keeping the full _cell_volume. Fixed in utils.py 2026-08-19 and
# the column recomputed, so the stored value is now the primary measurement.
VOL_RE = re.compile(r"_cell_volume\s+([-\d.eE]+)")
# Single-element formulas have no space, so pymatgen writes them unquoted
# (`_chemical_formula_sum   Mn4`). Match both forms or elemental structures drop out.
SUM_RE = re.compile(r"_chemical_formula_sum\s+(?:'([^']+)'|(\S+))")
ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def text_volume_per_atom(cif: str) -> float:
    if not isinstance(cif, str):
        return np.nan
    v, f = VOL_RE.search(cif), SUM_RE.search(cif)
    if not (v and f):
        return np.nan
    n = sum(int(c or 1) for el, c in ELEM_RE.findall(f.group(1) or f.group(2)) if el)
    try:
        return float(v.group(1)) / n if n else np.nan
    except ValueError:
        return np.nan


def ramp_colours(n: int) -> list:
    """n colours from WARM_RAMP, spread to the ends so 2 steps use the extremes."""
    if n <= 1:
        return [WARM_RAMP[-1]]
    if n <= len(WARM_RAMP):
        idx = np.linspace(0, len(WARM_RAMP) - 1, n).round().astype(int)
        return [WARM_RAMP[i] for i in idx]
    # More positive alphas than ramp stops: interpolate in RGB. Adjacent separation
    # falls below the validated floor here, which is why the legend is mandatory.
    stops = np.array([matplotlib.colors.to_rgb(c) for c in WARM_RAMP])
    t = np.linspace(0, len(WARM_RAMP) - 1, n)
    return [matplotlib.colors.to_hex(
        stops[int(np.floor(x))] * (1 - (x % 1)) + stops[min(int(np.floor(x)) + 1,
                                                            len(stops) - 1)] * (x % 1))
            for x in t]


def alpha_colour(alpha: float, positives: list) -> str:
    if alpha < 0:
        return COL_NEG
    if alpha == 0:
        return COL_ZERO
    return ramp_colours(len(positives))[positives.index(alpha)]


# How a run's steering strength is read out of its filename. The linear runs carry
# `alpha40.0`; the pca_centroid runs carry `t0.5` and are prefixed `steered_pca_` so the
# two families never collide in one directory. Both methods use 0 to mean "hook adds
# nothing", so the alpha=0 run is also the t=0 control and is listed under both.
STRENGTH_RE = {"linear": re.compile(r"alpha(-?[\d.]+)"),
               "pca": re.compile(r"_t(-?[\d.]+)_")}

# steered_manifold_<split>_d<delta>[_<variant>][_s<scale>]_k<k>_layer<N>.
# project_nomu before project so the longer name wins.
_MANIFOLD_RE = re.compile(
    r"^steered_manifold_[a-z]+_d(-?[\d.]+)"
    r"(?:_(residual|project_nomu|project))?(?:_s([\d.]+))?_k\d+_layer(\d+)")


def kind_of(stem: str) -> str:
    """Which steering method wrote this file, from its prefix.

    Longest prefix first so steered_pcalocal_ is never read as steered_pca_. The
    manifold variants are separate methods because five of them share delta 15 and
    would otherwise collide on one (target, strength) key.
    """
    if stem.startswith("steered_pcalocal_"):
        return "pca_local"
    if stem.startswith("steered_pca_"):
        return "pca_centroid"
    if stem.startswith("steered_manifold_"):
        m = _MANIFOLD_RE.match(stem)
        variant = (m.group(2) if m and m.group(2) else "residual")
        return "manifold" if variant == "residual" else f"manifold_{variant}"
    return "linear"


def sweep_target(stem: str, kind: str):
    """The value identifying one sweep: the property target for pca runs, the arc step
    `delta` for manifold runs (which have no property target when given --delta)."""
    if kind.startswith("manifold"):
        m = _MANIFOLD_RE.match(stem)
        return float(m.group(1)) if m else None
    m = re.search(r"_target([\d.]+)_t[\d.]+_k", stem)
    return float(m.group(1)) if m else None


def sweep_strength(stem: str, kind: str):
    """The knob swept within one sweep. For manifold that is --scale, defaulting to 1.0
    when the stem carries none."""
    if kind.startswith("manifold"):
        m = _MANIFOLD_RE.match(stem)
        return float(m.group(3)) if m and m.group(3) else 1.0
    m = STRENGTH_RE["linear" if kind == "linear" else "pca"].search(stem)
    return float(m.group(1)) if m else None


def discover_sweeps(results_dir: str, method: str) -> list:
    """The distinct (layer, target) sweeps on disk for a method, ascending.

    A run is identified by (layer, target, strength). Any two of those alone collide:
    the same target swept at two layers, or two targets swept over the same t, would
    share a key and silently overwrite each other.
    """
    out = set()
    for f in glob.glob(f"steering_results/{results_dir}/property_predictions/*.parquet"):
        b = _os.path.basename(f)
        kind = kind_of(b)
        if kind != method:
            continue
        lm = re.search(r"_layer(\d+)", b)
        if lm:
            out.add((int(lm.group(1)), sweep_target(b, kind)))
    return sorted(out, key=lambda p: (p[0], p[1] if p[1] is not None else -1))


def discover_targets(results_dir: str, family: str, method: str) -> list:
    """The distinct target values swept for a pca method, ascending."""
    tg = set()
    for f in glob.glob(f"steering_results/{results_dir}/property_predictions/*.parquet"):
        m = re.search(r"_target([\d.]+)_t[\d.]+_k", _os.path.basename(f))
        b = _os.path.basename(f)
        kind = kind_of(b)
        if m and kind == method:
            tg.add(float(m.group(1)))
    return sorted(tg)


def discover_runs(results_dir: str, family: str, method: str = "linear",
                  target: float = None, layer: int = None) -> dict:
    """{strength: stem} for the runs that have BOTH predictions and validation.

    Strength alone identifies a linear run, but a pca run is identified by
    (target, strength) -- two targets swept over the same t would otherwise collide on
    one key and silently overwrite each other. Pass `target` to select one sweep; the
    alpha=0 control has no target and is always included, whatever layer it names.
    """
    runs = {}
    for f in sorted(glob.glob(f"steering_results/{results_dir}/property_predictions/*.parquet")):
        stem = _os.path.basename(f)
        if stem == "testset_baseline.parquet":
            continue
        kind = kind_of(stem)
        is_control = kind == "linear" and re.search(r"alpha-?0(\.0)?_", stem)
        # every method's t=0 is the same no-injection run, so the control is shared
        if kind != method and not (is_control and method != "linear"):
            continue
        strength = sweep_strength(stem, kind)
        if strength is None:
            continue
        if kind != "linear" and not is_control:
            if target is not None:
                tgt = sweep_target(stem, kind)
                if tgt is None or tgt != target:
                    continue
            if layer is not None:
                lm = re.search(r"_layer(\d+)", stem)
                if not lm or int(lm.group(1)) != layer:
                    continue
        is_nosg = stem.endswith("_nosg.parquet")
        if family == "nosg" and not is_nosg:
            continue
        if family == "sg" and is_nosg:
            continue
        if not _os.path.exists(f"steering_results/{results_dir}/validation/{stem}"):
            print(f"  ! {stem}: predictions but no validation -- skipped")
            continue
        runs[strength] = stem
    return runs


def load_alpha(results_dir: str, stem: str, col: str, relaxed: bool,
               agg: str, measure: str) -> pd.DataFrame:
    """[id, value] for one run: valid samples only, one row per prompt if agg='mean'."""
    if measure == "text":
        gen = pd.read_parquet(f"steering_results/{results_dir}/generated_cifs/{stem}")
        cif_col = "cif_relaxed" if relaxed else "cif_steered"
        if relaxed:
            gen = pd.read_parquet(f"steering_results/{results_dir}/relaxed/{stem}")
        value_col = "value"
        pred = gen[["id", "sample"]].copy()
        pred[value_col] = gen[cif_col].map(text_volume_per_atom)
    else:
        value_col = col if relaxed else f"{col}_raw"
        pred = pd.read_parquet(
            f"steering_results/{results_dir}/property_predictions/{stem}")
        if value_col not in pred.columns:
            raise SystemExit(f"{stem}: no column {value_col!r} (have {list(pred.columns)})")
    valid = pd.read_parquet(f"steering_results/{results_dir}/validation/{stem}",
                            columns=["id", "sample", "is_valid"])

    df = pred[["id", "sample", value_col]].merge(valid, on=["id", "sample"], how="left")
    n_all = len(df)
    df = df[df["is_valid"].fillna(False) & df[value_col].notna()]
    df = df.rename(columns={value_col: "value"})[["id", "sample", "value"]]
    frac = len(df) / n_all if n_all else float("nan")
    print(f"    valid+scored samples: {len(df):,}/{n_all:,} ({frac:.1%})  "
          f"prompts surviving: {df['id'].nunique():,}")
    # How many of the 3 samples each surviving prompt kept. Under agg="max" this is a
    # confound, not a footnote: max-of-3 beats max-of-1 by construction, and a 95%-valid
    # control keeps ~3 per prompt while a 10%-valid run keeps ~1. Carried out so the
    # comparison can be read with that in view.
    per_prompt = df.groupby("id").size()
    if agg == "mean":
        df = df.groupby("id", as_index=False)["value"].mean()
    elif agg == "max":
        df = df.groupby("id", as_index=False)["value"].max()
    df.attrs["valid_frac"] = frac
    df.attrs["samples_per_prompt"] = float(per_prompt.mean()) if len(per_prompt) else float("nan")
    return df


def truth_band_gap(source: str, relaxed: bool) -> pd.DataFrame:
    """Ground-truth gap for the original test structures."""
    if source == "dft":
        meta = pd.read_parquet(NOMAD_METADATA, columns=["id", DFT_GAP_COL])
        meta = meta.dropna(subset=[DFT_GAP_COL])
        print(f"  truth=dft: {len(meta):,} ids with {DFT_GAP_COL} (NOMAD only)")
        return meta.rename(columns={DFT_GAP_COL: "value"})[["id", "value"]]

    base = pd.read_parquet(
        "steering_results/bandgap/property_predictions/testset_baseline.parquet")
    col = "predicted_bandgap_ev" if relaxed else "predicted_bandgap_ev_raw"
    # Validity lives in its own file, as for every other run -- the predictions file
    # carries predictions only. Older baselines kept both in one parquet.
    val_path = Path("steering_results/bandgap/validation/testset_baseline.parquet")
    if val_path.exists():
        val = pd.read_parquet(val_path, columns=["id", "sample", "is_valid"])
        base = base.merge(val, on=["id", "sample"], how="left")
        base = base[base["is_valid"].fillna(False)]
    elif "is_valid" in base:
        base = base[base["is_valid"].fillna(False)]
    base = base[base[col].notna()]
    if base.empty:
        raise SystemExit(
            f"testset_baseline.parquet has no values in '{col}'. The baseline has not "
            f"been relaxed yet, so there is no reference for the relaxed gaps -- drop "
            f"--relaxed, or relax the baseline first.")
    print(f"  truth=matched: MEGNet on {len(base):,} original test CIFs ({col})")
    return base.rename(columns={col: "value"})[["id", "value"]]


def parsed_volume_per_atom(cif: str) -> float:
    """s.volume/len(s) after postprocess -- the same computation predictors.py does."""
    try:
        s = Structure.from_str(postprocess(cif, "truth"), fmt="cif")
        return s.volume / len(s)
    except Exception:
        return np.nan


def truth_density_atomic(ids: set) -> pd.DataFrame:
    """volume/natoms on the ORIGINAL test CIFs, computed the same way the generated
    side is (parsed structure, symmetry restored) -- so both curves match."""
    with gzip.open(TEST_PKL, "rb") as f:
        data = pickle.load(f)
    rows = [(cid, parsed_volume_per_atom(text)) for cid, text in data if cid in ids]
    df = pd.DataFrame(rows, columns=["id", "value"]).dropna(subset=["value"])
    print(f"  truth: parsed {len(df):,} original test CIFs "
          f"({len(rows) - len(df)} unparseable)")
    return df


def strength_label(method: str) -> str:
    """What the steering knob is called for this method, for prints and the legend."""
    return "alpha" if method == "linear" else "t"


def kde_curve(v: np.ndarray, grid: np.ndarray, bw: float):
    """Gaussian KDE evaluated on grid, or None when the sample is degenerate."""
    if len(v) < 2 or np.allclose(v, v[0]):
        return None
    try:
        return gaussian_kde(v, bw_method=bw)(grid)
    except np.linalg.LinAlgError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--property", default="band_gap", choices=sorted(PROPS))
    ap.add_argument("--method", choices=["linear", "pca_centroid"], default="linear",
                    help="Which steering family to draw. linear reads alpha runs, "
                         "pca_centroid reads steered_pca_* runs keyed by t (plus the "
                         "alpha=0 control, which is the same no-injection run).")
    ap.add_argument("--no-intersect", action="store_true",
                    help="Draw every curve over its own surviving prompts, and the "
                         "ground truth over all of them, instead of restricting all "
                         "curves to the prompts common to every run. Use it when one "
                         "strength collapses -- the intersection would otherwise shrink "
                         "every other curve to that run's few survivors.")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="Steering strengths to draw (default: every run on disk)")
    ap.add_argument("--family", choices=["nosg", "sg"], default=None,
                    help="band_gap only: prompts without/with a space-group header")
    ap.add_argument("--relaxed", action="store_true",
                    help="Read the M3GNet-relaxed value instead of the raw generated one")
    ap.add_argument("--agg", choices=["mean", "max", "all"], default="mean",
                    help="One point per prompt (mean of its valid samples), or every sample")
    ap.add_argument("--truth", choices=["matched", "dft"], default="matched",
                    help="band_gap only: MEGNet-on-original (matched) or the DFT column")
    ap.add_argument("--x-scale", choices=["auto", "linear", "log"], default="auto")
    ap.add_argument("--y", choices=["density", "count"], default="density",
                    help="Area-1 densities, or densities scaled by each curve's n")
    ap.add_argument("--clip-lo", type=float, default=0.5, help="Lower x percentile")
    ap.add_argument("--clip-hi", type=float, default=99.0, help="Upper x percentile")
    ap.add_argument("--bw", type=float, default=0.25, help="KDE bandwidth (scott factor)")
    args = ap.parse_args()
    np.random.seed(RANDOM_SEED)

    spec = PROPS[args.property]
    family = args.family or spec["default_family"]
    print(f"Property: {args.property}  ({spec['label']})")
    print(f"Value source: {'relaxed' if args.relaxed else 'raw generated CIF'}   "
          f"aggregation: {args.agg}   family: {family}")

    runs = discover_runs(spec["results_dir"], family, args.method)
    if not runs:
        raise SystemExit(f"No runs found for {args.property} "
                         f"(family={family}, method={args.method}).")
    print(f"Runs on disk: strengths {sorted(runs)}")
    if args.alphas is not None:
        missing = [a for a in args.alphas if a not in runs]
        if missing:
            print(f"  ! requested but absent: {missing} -- skipping them")
        alphas = [a for a in args.alphas if a in runs]
    else:
        alphas = sorted(runs)
    if not alphas:
        raise SystemExit("None of the requested alphas exist on disk.")

    print("Loading runs ...")
    per_alpha = {}
    for a in alphas:
        print(f"  {strength_label(args.method)} {a:g}  [{runs[a]}]")
        per_alpha[a] = load_alpha(spec["results_dir"], runs[a], spec["col"],
                                  args.relaxed, args.agg, spec["measure"])

    # A strength that broke the model entirely has nothing to draw. Drop it loudly
    # rather than letting an empty array fall through to the axis-range computation.
    empty = [a for a, d in per_alpha.items() if d.empty]
    for a in empty:
        print(f"  ! {strength_label(args.method)} {a:g}: no valid structures at all "
              f"-- no curve drawn")
        del per_alpha[a]
    alphas = [a for a in alphas if a not in empty]
    if not alphas:
        raise SystemExit("Every run has zero valid structures.")

    # Same prompt population in every curve, truth included -- unless --no-intersect,
    # where each curve keeps its own survivors and truth covers the union. One collapsed
    # strength would otherwise drag every other curve down to its handful of survivors.
    if args.no_intersect:
        common = set.union(*(set(d["id"]) for d in per_alpha.values()))
        print(f"\nNo intersection: each curve keeps its own prompts, truth covers "
              f"all {len(common):,} used")
    else:
        common = set.intersection(*(set(d["id"]) for d in per_alpha.values()))
        print(f"\nPrompts common to all {len(alphas)} alphas: {len(common):,}")
    if len(common) < 10:
        raise SystemExit("Too few shared prompts to compare.")

    if args.property == "band_gap":
        truth = truth_band_gap(args.truth, args.relaxed)
    else:
        truth = truth_density_atomic(common)
    truth = truth[truth["id"].isin(common)]
    print(f"  ground-truth points after restricting to common prompts: {len(truth):,}")
    if truth.empty:
        raise SystemExit("Ground truth does not overlap the generated prompts.")

    series = [("truth", truth["value"].to_numpy(float))]
    for a in alphas:
        d = per_alpha[a]
        series.append((a, d[d["id"].isin(common)]["value"].to_numpy(float)))

    scale = spec["scale"] if args.x_scale == "auto" else args.x_scale
    if scale == "log" and min(v.min() for _, v in series) <= 0:
        print("  ! non-positive values present -- falling back to a linear x axis")
        scale = "linear"
    if scale == "log":
        series = [(k, np.log10(v)) for k, v in series]

    pooled = np.concatenate([v for _, v in series])
    lo, hi = np.percentile(pooled, [args.clip_lo, args.clip_hi])
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 512)

    xlabel = spec["label"] + (" [log10]" if scale == "log" else "")
    positives = sorted(a for a in alphas if a > 0)
    strength_name = strength_label(args.method)

    # Two panels over the SAME curves: linear y shows the shape, log y separates the
    # tails. With a distribution this concentrated the alpha lines superimpose on the
    # linear panel, and "they superimpose" is only a defensible reading if the log
    # panel shows they superimpose out in the tail too.
    fig, axes = plt.subplots(1, 2, figsize=(19, 7.5), sharex=True)
    ylab = "Density (area = 1)" if args.y == "density" else "Density x n (counts)"
    rows = []
    for key, v in series:
        curve = kde_curve(v, grid, args.bw)
        med = float(np.median(v))
        if curve is None:
            print(f"  ! {key}: degenerate sample, no curve drawn")
            continue
        if args.y == "count":
            curve = curve * len(v)

        for ax in axes:
            if key == "truth":
                ax.fill_between(grid, curve, color=TRUTH_FILL, zorder=1,
                                label=f"Ground truth (original test CIFs), n={len(v):,}")
                ax.plot(grid, curve, color=TRUTH_EDGE, lw=1.6, zorder=2)
            else:
                ax.plot(grid, curve, color=alpha_colour(key, positives), lw=2.0, zorder=3,
                        ls="--" if key == 0 else "-",
                        label=f"{strength_name} {key:g}"
                              + ("  (control)" if key == 0 else "")
                              + f", n={len(v):,}")
        vf = per_alpha[key].attrs.get("valid_frac", float("nan")) if key != "truth" else 1.0
        rows.append(dict(series=key, n=len(v), valid_sample_frac=vf, median=med,
                         p25=float(np.percentile(v, 25)),
                         p75=float(np.percentile(v, 75)),
                         p99=float(np.percentile(v, 99))))

    for ax, logy in zip(axes, [False, True]):
        ax.set_xlim(grid[0], grid[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if logy:
            ax.set_yscale("log")
            ax.set_title("log y — tail detail", fontsize=11)
        else:
            ax.set_ylim(bottom=0)
            ax.set_title("linear y — overall shape", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9.5, loc="best")

    src = "relaxed" if args.relaxed else "raw generated"
    fam = f", {family} prompts" if family else ""
    pop = ("each curve over its own surviving prompts, ground truth over all "
           f"{len(common):,}" if args.no_intersect else f"n={len(common):,} shared prompts")
    fig.suptitle(
        f"{args.property} ({args.method}): generated distribution vs ground truth "
        f"across steering strength\n"
        f"layer 14{fam}, {src} CIFs, one point per prompt = mean of its valid samples, "
        f"{pop}", fontsize=12.5, y=1.02)
    fig.tight_layout()

    out_dir = analysis_dir("v1_all", None, "test", subdir="plots")
    tag = "_relaxed" if args.relaxed else ""
    meth = "" if args.method == "linear" else f"_{args.method}"
    out = out_dir / f"{args.property}{meth}_distribution_shift{tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    stats = pd.DataFrame(rows)
    unit = " [log10]" if scale == "log" else ""
    print(f"\n=== summary{unit} ===")
    print(stats.to_string(index=False))
    csv = out_dir / f"{args.property}{meth}_distribution_shift{tag}.csv"
    stats.to_csv(csv, index=False)
    print(f"\nSaved {out}\nSaved {csv}")


if __name__ == "__main__":
    main()
