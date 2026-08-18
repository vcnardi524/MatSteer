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

  density_atomic  `_cell_volume` / atom count from `_chemical_formula_sum`, read from
                  the CIF text on BOTH sides (generated, and the originals in
                  cifs_v1_test.pkl.gz). Both numbers are literal tokens, so this is the
                  property exactly as generated.
                  It deliberately does NOT use the stored `density_atomic_raw` column:
                  that is predictors.py:37 `s.volume / len(s)` over a
                  `Structure.from_str` parse, whose numerator is the full cell but whose
                  denominator is only the atom_site lines pymatgen instantiated (the ops
                  loop holds just 'x, y, z', so nothing expands). It reads 2.673x high
                  at the median here (53.48 vs 19.25 A^3/atom, 11.9% within 1%), by a
                  per-structure factor, so it distorts shape as well as location.
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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from utils import analysis_dir

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
        measure="text",
    ),
}

# Volume per atom, read straight out of the CIF text. Both numbers are literal tokens
# in the file, so this is the property as generated -- no structure parsing involved.
# The stored density_atomic column is NOT usable: property_predictions computes it as
# pymatgen's s.volume/len(s), and pymatgen builds only the sites listed in the
# atom_site loop while keeping the full _cell_volume, so it reads ~2.7x high (median
# 53.48 vs 19.25 A^3/atom here; only 11.9% of rows agree to within 1%).
VOL_RE = re.compile(r"_cell_volume\s+([-\d.eE]+)")
SUM_RE = re.compile(r"_chemical_formula_sum\s+'([^']+)'")
ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def text_volume_per_atom(cif: str) -> float:
    if not isinstance(cif, str):
        return np.nan
    v, f = VOL_RE.search(cif), SUM_RE.search(cif)
    if not (v and f):
        return np.nan
    n = sum(int(c or 1) for el, c in ELEM_RE.findall(f.group(1)) if el)
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


def discover_runs(results_dir: str, family: str) -> dict:
    """{alpha: stem} for the runs that have BOTH predictions and validation."""
    runs = {}
    for f in sorted(glob.glob(f"steering_results/{results_dir}/property_predictions/*.parquet")):
        stem = _os.path.basename(f)
        if stem == "testset_baseline.parquet":
            continue
        m = re.search(r"alpha(-?[\d.]+)", stem)
        if not m:
            continue
        is_nosg = stem.endswith("_nosg.parquet")
        if family == "nosg" and not is_nosg:
            continue
        if family == "sg" and is_nosg:
            continue
        if not _os.path.exists(f"steering_results/{results_dir}/validation/{stem}"):
            print(f"  ! {stem}: predictions but no validation -- skipped")
            continue
        runs[float(m.group(1))] = stem
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
    if agg == "mean":
        df = df.groupby("id", as_index=False)["value"].mean()
    df.attrs["valid_frac"] = frac
    return df


def truth_band_gap(source: str, relaxed: bool) -> pd.DataFrame:
    """Ground-truth gap for the original test structures."""
    if source == "dft":
        meta = pd.read_parquet(NOMAD_METADATA, columns=["id", DFT_GAP_COL])
        meta = meta.dropna(subset=[DFT_GAP_COL])
        print(f"  truth=dft: {len(meta):,} ids with {DFT_GAP_COL} (NOMAD only)")
        return meta.rename(columns={DFT_GAP_COL: "value"})[["id", "value"]]

    path = "steering_results/bandgap/property_predictions/testset_baseline.parquet"
    base = pd.read_parquet(path)
    col = "predicted_bandgap_ev" if relaxed else "predicted_bandgap_ev_raw"
    base = base[base["is_valid"].fillna(False) & base[col].notna()]
    print(f"  truth=matched: MEGNet on {len(base):,} original test CIFs ({col})")
    return base.rename(columns={col: "value"})[["id", "value"]]


def truth_density_atomic(ids: set) -> pd.DataFrame:
    """volume/natoms on the ORIGINAL test CIFs, read from the text the same way the
    generated side is -- so both curves are the same measurement."""
    with gzip.open(TEST_PKL, "rb") as f:
        data = pickle.load(f)
    rows = [(cid, text_volume_per_atom(text)) for cid, text in data if cid in ids]
    df = pd.DataFrame(rows, columns=["id", "value"]).dropna(subset=["value"])
    print(f"  truth: read {len(df):,} original test CIFs from text "
          f"({len(rows) - len(df)} unparseable)")
    return df


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
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="Steering strengths to draw (default: every run on disk)")
    ap.add_argument("--family", choices=["nosg", "sg"], default=None,
                    help="band_gap only: prompts without/with a space-group header")
    ap.add_argument("--relaxed", action="store_true",
                    help="Read the M3GNet-relaxed value instead of the raw generated one")
    ap.add_argument("--agg", choices=["mean", "all"], default="mean",
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

    runs = discover_runs(spec["results_dir"], family)
    if not runs:
        raise SystemExit(f"No runs found for {args.property} (family={family}).")
    print(f"Runs on disk: alphas {sorted(runs)}")
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
        print(f"  alpha {a:g}  [{runs[a]}]")
        per_alpha[a] = load_alpha(spec["results_dir"], runs[a], spec["col"],
                                  args.relaxed, args.agg, spec["measure"])

    # Same prompt population in every curve, truth included.
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
                        label=f"alpha {key:g}" + ("  (control)" if key == 0 else "")
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
    fig.suptitle(
        f"{args.property}: generated distribution vs ground truth across steering strength\n"
        f"layer 14{fam}, {src} CIFs, one point per prompt = mean of its valid samples, "
        f"n={len(common):,} shared prompts", fontsize=12.5, y=1.02)
    fig.tight_layout()

    out_dir = analysis_dir("v1_all", None, "test", subdir="plots")
    tag = "_relaxed" if args.relaxed else ""
    out = out_dir / f"{args.property}_distribution_shift{tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    stats = pd.DataFrame(rows)
    unit = " [log10]" if scale == "log" else ""
    print(f"\n=== summary{unit} ===")
    print(stats.to_string(index=False))
    csv = out_dir / f"{args.property}_distribution_shift{tag}.csv"
    stats.to_csv(csv, index=False)
    print(f"\nSaved {out}\nSaved {csv}")


if __name__ == "__main__":
    main()
