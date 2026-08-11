#!/usr/bin/env python3
"""Render symmetry_probe.csv as a colour-coded table, one row per layer.

The probe writes three accuracies per layer and only the gaps between them mean
anything, which is hard to see in a raw CSV. This lays them side by side with the
margin over the composition baseline shaded, so a layer that actually beats
chemistry stands out from one that does not.

Columns: majority (floor) | composition (chemistry only) | embedding (residual
stream) | margin = embedding - composition.

Reads  analysis/<dataset>/<variant>/<partition>/symmetry_probe_<label>.csv
Writes analysis/<dataset>/<variant>/<partition>/symmetry_probe_table_<label>.png

Usage:
    python scripts/plots/plot_probe_table.py --variant nosym --partition val
    python scripts/plots/plot_probe_table.py --variant nosym --partition all --split random
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import DATASETS, PARTITIONS, VARIANTS, analysis_dir

COLUMNS = [
    ("majority_acc", "majority\n(floor)"),
    ("composition_acc", "composition\n(chemistry only)"),
    ("embedding_acc", "embedding\n(residual stream)"),
    ("margin_over_composition", "margin\n(emb - comp)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v1_all", choices=list(DATASETS))
    ap.add_argument("--variant", default="nosym", choices=list(VARIANTS))
    ap.add_argument("--partition", required=True, choices=list(PARTITIONS))
    ap.add_argument("--label", default="point_group",
                    choices=["point_group", "space_group_symbol", "wyckoff_letters"],
                    help="which symmetry_probe_<label>.csv to render")
    ap.add_argument("--split", default="by_formula", choices=["by_formula", "random"],
                    help="by_formula holds whole formulas out, so nothing can be looked "
                         "up; random is the standard but more flattering split")
    args = ap.parse_args()

    out_dir = analysis_dir(args.dataset, args.variant, args.partition)
    csv = out_dir / f"symmetry_probe_{args.label}.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found -- run symmetry_probe.py with LABEL_COLS={args.label}")
    df = pd.read_csv(csv)
    df = df[df["split"] == args.split].sort_values("layer")
    if df.empty:
        raise SystemExit(f"No rows with split={args.split!r} in {csv}")

    for label in df["label_col"].unique():
        d = df[df["label_col"] == label]
        cells, colours = [], []
        # Shade the margin column only. The three accuracies share a scale but mean
        # different things, so colouring them all would invite reading across the row.
        lo, hi = d["margin_over_composition"].min(), d["margin_over_composition"].max()
        span = (hi - lo) or 1.0
        for _, r in d.iterrows():
            cells.append([f"{r[c]:.4f}" for c, _ in COLUMNS])
            shade = plt.cm.Blues(0.15 + 0.55 * (r["margin_over_composition"] - lo) / span)
            colours.append(["white", "white", "white", shade])

        fig, ax = plt.subplots(figsize=(9, 0.32 * len(d) + 1.6))
        ax.axis("off")
        table = ax.table(cellText=cells,
                         rowLabels=[f"layer {int(l)}" for l in d["layer"]],
                         colLabels=[name for _, name in COLUMNS],
                         cellColours=colours, cellLoc="center", loc="upper center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        n_classes = int(d["n_classes"].iloc[0])
        n_test = int(d["n_test"].iloc[0])
        # State the shading endpoints. The margin barely moves across layers, so a
        # colour scale stretched over that range makes differences well inside the
        # noise look dramatic -- naming the span stops the colour being over-read.
        ax.set_title(
            f"Linear probe — {label}\n"
            f"{args.dataset} / {args.variant} / partition={args.partition} / split={args.split}\n"
            f"{n_classes} classes, ~{n_test:,} test rows per layer   |   "
            f"margin shading spans {lo:.4f}–{hi:.4f} (range {hi - lo:.4f})",
            fontsize=10, pad=12)
        fig.tight_layout()

        png = out_dir / f"symmetry_probe_table_{label}_{args.split}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {png}")


if __name__ == "__main__":
    main()
