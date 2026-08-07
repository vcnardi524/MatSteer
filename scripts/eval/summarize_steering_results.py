#!/usr/bin/env python3
"""
Aggregate validation + novelty stats for every steered run, in one table.

For each config it reads steering_results/validation/. The novelty file
(novelty_<stem>.parquet) is a superset of the plain validation file (same flags
plus is_unique/is_novel), so it is preferred when present; otherwise the plain
<stem>.parquet is used and the uniqueness/novelty columns show as n/a.

Columns:
    total   valid  valid%  sensible%   unique  unique%(of valid)  novel  novel%(of unique)

Optionally (--bandgap) it also summarizes predicted band gaps from the combined
wide file written by combine_bandgap_predictions.py (bandgap_all.parquet). Two
aggregations are emitted — over all samples, and best-of (per-prompt max over
samples) — each as two tables split by whether the prompt carried a space group
(steered runs with the _nosg suffix vs. without). Each table has the baseline on
top, then one row per steered run: configuration, alpha, samples/cif, and
%>0 / mean / std for both unrelaxed and relaxed gaps.

Usage:
    python scripts/eval/summarize_steering_results.py
    python scripts/eval/summarize_steering_results.py --bandgap
    python scripts/eval/summarize_steering_results.py --bandgap steering_results/bandgap_all.parquet
    python scripts/eval/summarize_steering_results.py --bandgap --out steering_results/summary.txt
"""
import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS = "steering_results"


def config_stem(path: Path) -> str:
    """Strip the leading 'novelty_' so both file kinds map to one config name."""
    return re.sub(r"^novelty_", "", path.stem)


def pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "n/a"


def validation_novelty_table(val_dir: Path) -> pd.DataFrame:
    # group files by config, preferring the novelty superset
    by_config: dict[str, Path] = {}
    for f in sorted(val_dir.glob("*.parquet")):
        cfg = config_stem(f)
        if f.stem.startswith("novelty_") or cfg not in by_config:
            by_config[cfg] = f  # novelty_ wins; else first-seen plain file

    # order: space-group runs first, then _nosg; within each, ascending alpha
    def sort_key(item):
        cfg = item[0]
        return (cfg.endswith("_nosg"), run_alpha(cfg) or 0.0)

    rows = []
    for cfg, f in sorted(by_config.items(), key=sort_key):
        df = pd.read_parquet(f)
        n = len(df)
        n_valid = int(df["is_valid"].sum())
        n_sens = int(df["is_sensible"].sum()) if "is_sensible" in df else 0
        has_nov = "is_unique" in df.columns and "is_novel" in df.columns
        n_unique = int(df["is_unique"].sum()) if has_nov else None
        n_novel = int(df["is_novel"].sum()) if has_nov else None

        rows.append({
            "config": cfg,
            "total": n,
            "valid": n_valid,
            "valid%": pct(n_valid, n),
            "sensible%": pct(n_sens, n) if "is_sensible" in df else "n/a",
            "unique": n_unique if has_nov else "n/a",
            "unique%(of valid)": pct(n_unique, n_valid) if has_nov else "n/a",
            "novel": n_novel if has_nov else "n/a",
            "novel%(of unique)": pct(n_novel, n_unique) if has_nov else "n/a",
        })
    return pd.DataFrame(rows)


# two-row header: (group, sub-column). "" group = flat left-hand columns.
BG_COLS = [
    ("", "configuration"), ("", "alpha"), ("", "samples/cif"),
    ("unrelaxed", "%>0"), ("unrelaxed", "mean"), ("unrelaxed", "std"),
    ("relaxed", "%>0"), ("relaxed", "mean"), ("relaxed", "std"),
]


def run_alpha(run: str):
    if run == "baseline":
        return None
    m = re.search(r"alpha(-?[0-9.]+)", run)  # -? so negative alphas (e.g. alpha-16.0) parse
    return float(m.group(1)) if m else None


def _gap_stats(df: pd.DataFrame, col: str | None, agg: str = "all"):
    """(%>0, mean, std) for a bandgap column, or dashes if missing/empty.

    agg="all": over every (id, sample); agg="max": per-prompt best (max over the
    prompt's samples), then stats across prompts.
    """
    if col is None or col not in df.columns:
        return ("—", "—", "—")
    s = pd.to_numeric(df[col], errors="coerce")
    if agg == "max":
        s = s.groupby(df["id"]).max()  # per-prompt max over its samples
    v = s.dropna()
    if v.empty:
        return ("—", "—", "—")
    return (f"{(v > 0).mean():.1%}", f"{v.mean():.3f}", f"{v.std():.3f}")


def _bg_row(df: pd.DataFrame, run: str, cols: dict, agg: str = "all") -> dict:
    raw_col, rel_col = cols.get("raw"), cols.get("relaxed")
    ref = raw_col or rel_col
    samples = int(df.loc[df[ref].notna(), "sample"].nunique()) if ref else 0
    alpha = run_alpha(run)
    unrel, rel = _gap_stats(df, raw_col, agg), _gap_stats(df, rel_col, agg)
    return {
        ("", "configuration"): "baseline" if run == "baseline" else "steered",
        ("", "alpha"): "—" if alpha is None else (int(alpha) if alpha == int(alpha) else alpha),
        ("", "samples/cif"): samples,
        ("unrelaxed", "%>0"): unrel[0], ("unrelaxed", "mean"): unrel[1], ("unrelaxed", "std"): unrel[2],
        ("relaxed", "%>0"): rel[0], ("relaxed", "mean"): rel[1], ("relaxed", "std"): rel[2],
    }


def bandgap_tables(path: Path, agg: str = "all") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two band-gap tables (with-spacegroup, without-spacegroup); baseline on top of each."""
    df = pd.read_parquet(path)

    # discover runs from the wide columns (bandgap_raw_<run> / bandgap_<run>)
    runs: dict[str, dict] = {}
    for col in df.columns:
        if col.startswith("bandgap_raw_"):
            runs.setdefault(col[len("bandgap_raw_"):], {})["raw"] = col
        elif col.startswith("bandgap_"):
            runs.setdefault(col[len("bandgap_"):], {})["relaxed"] = col

    def order(names):
        # baseline first, then steered by ascending alpha
        return sorted(names, key=lambda r: (r != "baseline", run_alpha(r) or 0.0))

    def build(names) -> pd.DataFrame:
        rows = [_bg_row(df, r, runs[r], agg) for r in order(names)]
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.reindex(columns=BG_COLS)
            out.columns = pd.MultiIndex.from_tuples(out.columns)
        return out

    sg = [r for r in runs if r == "baseline" or not r.endswith("_nosg")]
    nosg = [r for r in runs if r == "baseline" or r.endswith("_nosg")]
    return build(sg), build(nosg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS,
                        help="Base results dir; defaults derive <results-dir>/validation "
                             "and <results-dir>/property_all.parquet")
    parser.add_argument("--val-dir", default=None,
                        help="Dir of validation/novelty parquets (default: <results-dir>/validation)")
    parser.add_argument("--bandgap", nargs="?", const="__DEFAULT__", default=None,
                        help="Also summarize band gaps from the combined wide file "
                             "(default: <results-dir>/property_all.parquet)")
    parser.add_argument("--out", default=None,
                        help="Also write the full report to this text file "
                             "(e.g. <results-dir>/summary.txt)")
    args = parser.parse_args()

    val_dir = Path(args.val_dir) if args.val_dir else Path(args.results_dir) / "validation"
    if args.bandgap == "__DEFAULT__":
        args.bandgap = str(Path(args.results_dir) / "property_all.parquet")

    # collect the report as lines so it can go to both stdout and --out
    lines: list[str] = []

    def section(title: str, body: str):
        lines.append(f"\n=== {title} ===")
        lines.append(body)

    vn = validation_novelty_table(val_dir)
    section("Validation + Novelty",
            vn.to_string(index=False) if not vn.empty else "(no validation files found)")

    if args.bandgap:
        bg_path = Path(args.bandgap)
        if not bg_path.exists():
            section(f"Predicted band gap — {bg_path}",
                    "(not found — run combine_bandgap_predictions.py first)")
        else:
            # emit both aggregations: over all samples, and per-prompt best-of
            for agg, note in [("all", "all samples"),
                              ("max", "best-of samples: per-prompt max")]:
                sg, nosg = bandgap_tables(bg_path, agg)
                for title, tbl in [("with space group in prompt", sg),
                                   ("without space group in prompt", nosg)]:
                    section(f"Predicted band gap (eV) — {title} [{note}]",
                            tbl.to_string(index=False) if not tbl.empty else "(no matching runs)")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"Saved report -> {out_path}")


if __name__ == "__main__":
    main()
