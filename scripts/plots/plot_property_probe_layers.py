#!/usr/bin/env python3
"""Probe R^2 against layer, one line per property.

Reads the CSVs that property_probe.py wrote and draws the depth curves side by side.
Scored on `by_formula`, so no test formula appeared in training.

Left panel is the raw R^2. Read it against the layer-0 marker, not against zero. Right
panel subtracts layer 0, leaving what depth adds.

Layer L is the OUTPUT of transformer block L (extract_cif_embeddings.py hooks
model.transformer.h[l]), so layer 0 already has one attention + MLP behind it. There is
no pre-block extraction in this pipeline. The right panel is therefore "what blocks 1-15
add over block 0" -- NOT "what the network adds over the raw text". It does not isolate
how much of the property is simply printed in the CIF.

Usage:
    python scripts/plots/plot_property_probe_layers.py
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, validated for the four-series case: worst adjacent CVD dE 11.0,
# normal-vision 18.4. Every line is also direct-labelled, which is what the
# low contrast-vs-surface of the two lighter hues requires.
SERIES = [
    ("density_atomic",          "analysis/v1_all/full/val/property_probe_density_atomic.csv",
     "volume per atom",     "#D55E00", "o"),
    ("efermi",                  "analysis/v1_mp/full/val/property_probe_efermi.csv",
     "Fermi level",         "#0072B2", "s"),
    ("energy_above_hull",       "analysis/v1_mp/full/val/property_probe_energy_above_hull.csv",
     "energy above hull",   "#009E73", "^"),
    ("dos_electronic.band_gap", "analysis/v1_all/full/val/property_probe_dos_electronic_band_gap.csv",
     "band gap",            "#E69F00", "D"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="by_formula", choices=("by_formula", "random"))
    # top level of analysis/, not under a dataset tree: this figure combines
    # v1_all (density, band gap) with v1_mp (efermi, energy above hull)
    ap.add_argument("--out", default="analysis/property_probe_r2_by_layer.png")
    args = ap.parse_args()

    loaded = []
    for name, path, label, color, marker in SERIES:
        if not Path(path).exists():
            print(f"  ! missing {path} -- skipped")
            continue
        d = pd.read_csv(path)
        d = d[d["split"] == args.split].sort_values("layer")
        loaded.append((label, color, marker, d))
    loaded.sort(key=lambda t: -t[3]["embedding_r2"].max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True)
    rows = []
    for label, color, marker, d in loaded:
        x = d["layer"].to_numpy()
        y = d["embedding_r2"].to_numpy()
        l0 = y[0]
        peak = int(x[y.argmax()])

        axes[0].plot(x, y, color=color, lw=2, marker=marker, ms=5, label=label)
        axes[0].plot(x[0], l0, marker=marker, ms=11, mfc="none", mec=color, mew=2)
        axes[0].annotate(f" {label}", (x[-1], y[-1]), color=color, fontsize=10,
                         va="center", fontweight="bold")

        axes[1].plot(x, y - l0, color=color, lw=2, marker=marker, ms=5)
        axes[1].annotate(f" {label}", (x[-1], y[-1] - l0), color=color, fontsize=10,
                         va="center", fontweight="bold")

        rows.append({"property": label, "composition_r2": d["composition_r2"].iloc[0],
                     "layer0_r2": l0, "peak_layer": peak, "peak_r2": y.max(),
                     "gain_over_layer0": y.max() - l0})

    ax = axes[0]
    ax.set_ylabel("probe $R^2$")
    ax.set_title("probe $R^2$ by layer\nhollow marker = layer 0 (output of the first block)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax = axes[1]
    ax.axhline(0, color="#999", lw=1)
    ax.set_ylabel("$R^2$ minus layer 0")
    ax.set_title("what depth adds beyond the first block\n(layer 0 subtracted)")

    for ax in axes:
        ax.set_xlabel("layer")
        ax.set_xlim(-0.5, 18.5)
        ax.set_xticks(range(0, 16, 2))
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle(f"CrystaLLM linear probes, {args.split} split "
                 "(no test formula seen in training)", fontsize=13)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")

    summary = pd.DataFrame(rows)
    summary.to_csv(str(out).replace(".png", ".csv"), index=False, float_format="%.6g")
    print(summary.to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
