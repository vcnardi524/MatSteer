#!/usr/bin/env python3
"""How far does a steering injection travel through the layers above it?

Steering is applied once, at one layer. These curves show what the blocks above it do with
that perturbation -- amplify, damp, or absorb. Reads the embeddings written by
steer_extract_embeddings.py; nothing here regenerates anything.

PAIRING IS ON PROMPT, NOT ON (id, sample). Sample 1 of a steered run and sample 1 of the
control are unrelated structures -- the model sampled each independently -- so pairing on
(id, sample) would compare two arbitrary draws. Each prompt's embedding is averaged over
its own valid samples first, then the runs are differenced on `id`. That matches how every
other paired comparison in this repo works.

VALID SAMPLES ONLY, by default. Across 106 density arms corr(validity, median density) is
-0.91: degradation moves the property being steered, so a curve computed over broken
generations measures breakage. --include-invalid runs it both ways so the two can be
compared, which is the check that says whether propagation is a property of the injection
or of the damage.

Reported per layer:
  d_abs   ||h_steered - h_control||, mean over prompts
  d_rel   d_abs / ||h_control||     -- the axis injection_magnitude.py uses
  cosine  cosine similarity between the two

Usage:
    python scripts/analysis/injection_propagation.py \
        --runs steered_test_alpha32.0_layer7_nosg \
               steered_manifold_test_d2_residual_s4_k64_layer7_nosg
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import steering_path

CONTROL = "steered_test_alpha0.0_layer0_nosg"
OUT = Path("analysis/v1_all/test")
COLOR = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]


def per_prompt(results_dir, stem, valid_only=True):
    """{layer: DataFrame(id -> embedding)} averaged over each prompt's samples."""
    df = pd.read_parquet(steering_path(results_dir, "embeddings", f"{stem}.parquet"))
    if valid_only:
        df = df[df["is_valid"] == True]
    out = {}
    for layer, g in df.groupby("layer"):
        E = np.vstack(g["embedding"].to_numpy()).astype(np.float64)
        acc = pd.DataFrame(E, index=g["id"].to_numpy()).groupby(level=0).mean()
        out[int(layer)] = acc
    return out


def compare(ctrl, steer):
    rows = []
    for layer in sorted(set(ctrl) & set(steer)):
        a, b = ctrl[layer], steer[layer]
        ids = a.index.intersection(b.index)
        A, B = a.loc[ids].to_numpy(), b.loc[ids].to_numpy()
        diff = np.linalg.norm(B - A, axis=1)
        na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
        cos = (A * B).sum(1) / (na * nb)
        rows.append(dict(layer=layer, n=len(ids), d_abs=diff.mean(),
                         d_rel=(diff / na).mean(), cosine=cos.mean(),
                         ctrl_norm=na.mean(), steer_norm=nb.mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--control", default=CONTROL)
    ap.add_argument("--results-dir", default="density_atomic")
    ap.add_argument("--include-invalid", action="store_true",
                    help="Also compute over ALL samples, not just valid ones, and report "
                         "both. If the curves agree, propagation is not a degradation "
                         "artifact.")
    ap.add_argument("--out-stem", default="injection_propagation")
    args = ap.parse_args()

    frames = []
    for valid_only in ([True, False] if args.include_invalid else [True]):
        tag = "valid" if valid_only else "all"
        ctrl = per_prompt(args.results_dir, args.control, valid_only)
        print(f"control {args.control} [{tag}]: "
              f"{len(next(iter(ctrl.values()))):,} prompts, layers {sorted(ctrl)}")
        for stem in args.runs:
            try:
                s = per_prompt(args.results_dir, stem, valid_only)
            except FileNotFoundError:
                print(f"  ! no embeddings for {stem}"); continue
            f = compare(ctrl, s)
            f.insert(0, "run", stem); f.insert(1, "samples", tag)
            frames.append(f)
            print(f"  {stem} [{tag}]: layer {f.layer.min()} d_rel={f.d_rel.iloc[0]:.4f} "
                  f"-> layer {f.layer.max()} d_rel={f.d_rel.iloc[-1]:.4f}")

    d = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / f"{args.out_stem}.csv", index=False, float_format="%.6g")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, stem in enumerate(args.runs):
        g = d[(d.run == stem) & (d.samples == "valid")].sort_values("layer")
        if g.empty:
            continue
        lab = (stem.replace("steered_manifold_test_", "manifold ")
                   .replace("steered_test_", "linear ")
                   .replace("_k64_layer7_nosg", "").replace("_layer7_nosg", "")
                   .replace("_residual", ""))
        ax1.plot(g.layer, g.d_rel, "-o", color=COLOR[i % len(COLOR)], lw=2, ms=5, label=lab)
        ax2.plot(g.layer, g.cosine, "-o", color=COLOR[i % len(COLOR)], lw=2, ms=5)
    ax1.set_ylabel(r"$\|h_{steered}-h_{control}\|\ /\ \|h_{control}\|$")
    ax2.set_ylabel("cosine similarity to control")
    for ax in (ax1, ax2):
        ax.set_xlabel("layer")
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax1.legend(frameon=False, fontsize=9)
    fig.suptitle("How a layer-7 injection propagates through the layers above it\n"
                 "valid samples only; paired on prompt, averaged over each prompt's samples",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"{args.out_stem}.png", dpi=150, bbox_inches="tight", facecolor="white")

    # Standalone single-panel version of the ratio: this is the headline number, and it
    # reads better without the cosine panel competing for attention.
    fig2, ax = plt.subplots(figsize=(8, 5.5))
    for i, stem in enumerate(args.runs):
        g = d[(d.run == stem) & (d.samples == "valid")].sort_values("layer")
        if g.empty:
            continue
        lab = (stem.replace("steered_manifold_test_", "manifold ")
                   .replace("steered_test_", "linear ")
                   .replace("_k64_layer7_nosg", "").replace("_layer7_nosg", "")
                   .replace("_residual", ""))
        ax.plot(g.layer, g.d_rel, "-o", color=COLOR[i % len(COLOR)], lw=2.2, ms=6, label=lab)
        ax.annotate(lab, (g.layer.iloc[-1], g.d_rel.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, va="center",
                    color=COLOR[i % len(COLOR)])
    inj = int(d.layer.min())
    ax.axvline(inj, color="#999", lw=1, ls="--")
    ax.annotate(f"injected at layer {inj}", (inj, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=8.5, color="#666")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\|h_{\mathrm{steered}}-h_{\mathrm{control}}\|\ /\ \|h_{\mathrm{control}}\|$")
    ax.set_xticks(sorted(d.layer.unique()))
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(x=0.12)
    ax.set_title("A layer-7 injection neither grows nor decays through the layers above it\n"
                 "valid samples only; paired on prompt", fontsize=11, loc="left")
    fig2.tight_layout()
    fig2.savefig(OUT / f"{args.out_stem}_ratio.png", dpi=150, bbox_inches="tight",
                 facecolor="white")
    print(f"\nSaved {OUT/args.out_stem}.csv, .png and _ratio.png")
    with pd.option_context("display.width", 200):
        print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
