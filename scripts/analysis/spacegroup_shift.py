#!/usr/bin/env python3
"""Does steering change the SPACE GROUP the model writes, relative to ground truth?

The nosg prompts carry no space group, so the model chooses one freely. If steering only
rescaled a cell we would expect the symmetry to survive; if it perturbs the structure
qualitatively, the chosen space group should drift away from the prompt's true one.

Compared against GROUND TRUTH, not against the alpha=0 control. The control generates the
same prompt 3 times and can pick a different space group each time, so "different from the
control" would mostly measure the control's own sampling spread. Ground truth is one fixed
symbol per prompt.

The H-M symbol is read straight out of the CIF text. No pymatgen parse is involved, so
the postprocess symmetry-operator trap does not apply here -- postprocess rewrites the
operator list, never the _symmetry_space_group_name_H-M field.

Usage:
    python scripts/analysis/spacegroup_shift.py --runs <stem> [<stem> ...]
"""
import argparse, gzip, pickle, re, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import steering_path

TEST_PKL = "CrystaLLM/cifs_v1_test_sample1000.pkl.gz"
SG_RE = re.compile(r"_symmetry_space_group_name_H-M\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))")


def space_group(cif):
    if not isinstance(cif, str):
        return None
    m = SG_RE.search(cif)
    if not m:
        return None
    return (m.group(1) or m.group(2) or m.group(3)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--results-dir", default="density_atomic")
    ap.add_argument("--out", default="analysis/v1_all/test/spacegroup_shift.csv")
    args = ap.parse_args()

    with gzip.open(TEST_PKL, "rb") as f:
        truth = {i: space_group(c) for i, c in pickle.load(f)}
    truth = {k: v for k, v in truth.items() if v}
    print(f"ground truth: {len(truth):,} prompts with a space group")

    rows = []
    for stem in args.runs:
        gen = pd.read_parquet(steering_path(args.results_dir, "generated_cifs", f"{stem}.parquet"),
                              columns=["id", "sample", "cif_steered"])
        val = pd.read_parquet(steering_path(args.results_dir, "validation", f"{stem}.parquet"),
                              columns=["id", "sample", "is_valid"])
        df = gen.merge(val, on=["id", "sample"], how="left")
        df = df[df["is_valid"] == True].copy()
        df["sg"] = df["cif_steered"].map(space_group)
        df["truth"] = df["id"].map(truth)
        df = df[df["sg"].notna() & df["truth"].notna()]
        df["differs"] = df["sg"] != df["truth"]

        # per SAMPLE, and per PROMPT (a prompt counts as changed if every sample changed)
        per_prompt = df.groupby("id")["differs"].mean()
        rows.append(dict(run=stem, n_samples=len(df), n_prompts=df["id"].nunique(),
                         pct_samples_differ=100 * df["differs"].mean(),
                         pct_prompts_all_differ=100 * (per_prompt == 1).mean(),
                         pct_prompts_any_differ=100 * (per_prompt > 0).mean(),
                         pct_P1=100 * (df["sg"] == "P 1").mean()))
        print(f"  {stem}: {len(df):,} valid samples, {df['id'].nunique():,} prompts")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.4g")
    print()
    print(out.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
