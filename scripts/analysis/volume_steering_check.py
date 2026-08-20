#!/usr/bin/env python3
"""
Did density_atomic steering actually move the cell volume?

Compares, per alpha, the volume of the STEERED generated CIFs against the ORIGINAL
(reference) CIF for the same prompt, plus the alpha=0 baseline generation.

Three volumes are extracted per CIF, and the gap between them is the point:
  written   `_cell_volume`, the number the model literally emits as a token
  lattice   volume recomputed from `_cell_length_{a,b,c}` + `_cell_angle_*`
  per-atom  written / atom count from `_chemical_formula_sum`  (== density_atomic,
            the intensive quantity the steering vector was actually built on)

`_cell_volume` is a literal token in the CIF the model reads and writes, so steering
on density_atomic should be able to move it directly. If `written` moves but `lattice`
does not, the model is emitting a volume token inconsistent with the cell it actually
describes -- steering the text without steering the structure. That distinction is
invisible to any metric computed from the parsed structure alone.

Volumes are extensive, so `written` is only comparable across prompts of identical
composition; the paired per-prompt ratio (steered / original for the SAME id) and the
intensive per-atom volume are the numbers to read.

Usage:
    python scripts/analysis/volume_steering_check.py
    python scripts/analysis/volume_steering_check.py --property density_atomic --valid-only
"""
import argparse
import gzip
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

VOL_RE = re.compile(r"_cell_volume\s+([-\d.eE]+)")
# Single-element formulas have no space, so pymatgen writes them unquoted
# (`_chemical_formula_sum   Mn4`). Match both forms or elemental structures drop out.
SUM_RE = re.compile(r"_chemical_formula_sum\s+(?:'([^']+)'|(\S+))")
LEN_RE = {k: re.compile(rf"_cell_length_{k}\s+([-\d.eE]+)") for k in "abc"}
ANG_RE = {k: re.compile(rf"_cell_angle_{k}\s+([-\d.eE]+)")
          for k in ("alpha", "beta", "gamma")}
ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
TEST_PKL = "CrystaLLM/cifs_v1_test.pkl.gz"


def _f(rx, text):
    m = rx.search(text)
    if not m:
        return np.nan
    try:
        return float(m.group(1))
    except ValueError:
        return np.nan


def lattice_volume(text: str) -> float:
    """Cell volume from the lattice parameters (triclinic formula)."""
    a, b, c = (_f(LEN_RE[k], text) for k in "abc")
    al, be, ga = (np.radians(_f(ANG_RE[k], text)) for k in ("alpha", "beta", "gamma"))
    if not np.isfinite([a, b, c, al, be, ga]).all():
        return np.nan
    ca, cb, cg = np.cos(al), np.cos(be), np.cos(ga)
    disc = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    return a * b * c * np.sqrt(disc) if disc > 0 else np.nan


def natoms(text: str) -> float:
    m = SUM_RE.search(text)
    if not m:
        return np.nan
    n = sum(int(cnt or 1) for el, cnt in ELEM_RE.findall(m.group(1) or m.group(2)) if el)
    return float(n) if n else np.nan


def parse(text: str) -> dict:
    if not isinstance(text, str):
        return dict(written=np.nan, lattice=np.nan, n=np.nan, per_atom=np.nan)
    w, n = _f(VOL_RE, text), natoms(text)
    return dict(written=w, lattice=lattice_volume(text), n=n, per_atom=w / n if n else np.nan)


def parse_frame(texts, ids, samples=None) -> pd.DataFrame:
    rows = [parse(t) for t in texts]
    df = pd.DataFrame(rows)
    df.insert(0, "id", list(ids))
    if samples is not None:
        df.insert(1, "sample", list(samples))
    return df


def load_reference() -> pd.DataFrame:
    """Original (ground-truth) CIFs from the v1 test split, parsed the same way."""
    with gzip.open(TEST_PKL, "rb") as f:
        data = pickle.load(f)
    ids, texts = zip(*data)
    ref = parse_frame(texts, ids)
    return ref.rename(columns={"written": "ref_written", "lattice": "ref_lattice",
                               "n": "ref_n", "per_atom": "ref_per_atom"})


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.percentile(x, [25, 50, 75]) if len(x) else [np.nan] * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="density_atomic")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--valid-only", action="store_true",
                    help="Keep only rows flagged is_valid in validation/")
    args = ap.parse_args()

    root = Path("steering_results") / args.property
    gen_dir = root / "generated_cifs"
    stems = sorted(gen_dir.glob(f"*_layer{args.layer}.parquet"))
    alphas = {}
    for p in stems:
        m = re.search(r"alpha([\d.]+)_layer", p.name)
        if m:
            alphas[float(m.group(1))] = p
    if not alphas:
        raise SystemExit(f"No generated CIFs under {gen_dir}")

    print(f"Reference (original) CIFs from {TEST_PKL}")
    ref = load_reference()
    print(f"  {len(ref):,} reference structures parsed")

    sv = Path("steering_vectors") / args.property / f"layer{args.layer}.parquet"
    if sv.exists():
        s = pd.read_parquet(sv).drop(columns=["steering_vector"], errors="ignore")
        print(f"Steering vector: {s.to_dict('records')[0]}")

    summary = []
    for alpha in sorted(alphas):
        gen = pd.read_parquet(alphas[alpha])
        df = parse_frame(gen["cif_steered"], gen["id"], gen["sample"])

        if args.valid_only:
            vpath = root / "validation" / alphas[alpha].name
            if vpath.exists():
                v = pd.read_parquet(vpath)[["id", "sample", "is_valid"]]
                before = len(df)
                df = df.merge(v, on=["id", "sample"], how="left")
                df = df[df["is_valid"].fillna(False)].drop(columns="is_valid")
                print(f"\nalpha {alpha}: valid-only {len(df):,}/{before:,}")

        df = df.merge(ref, on="id", how="left")
        src = df["id"].str.split("_").str[0].value_counts().to_dict()

        # written-vs-lattice consistency: is the emitted volume token the volume of
        # the cell the model actually describes?
        rel = (df["written"] - df["lattice"]).abs() / df["lattice"]
        consistent = float((rel < 0.01).mean())

        ratio_pa = df["per_atom"] / df["ref_per_atom"]      # paired, intensive
        ratio_w = df["written"] / df["ref_written"]          # paired, extensive
        q = pct(df["per_atom"])
        print(f"\n=== alpha {alpha}  (n={len(df):,}, sources {src}) ===")
        print(f"  parsed _cell_volume:      {df['written'].notna().mean():.1%} of rows")
        print(f"  written vs lattice agree (<1% rel): {consistent:.1%}")
        print(f"  steered volume/atom  q25/med/q75: {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} A^3")
        rq = pct(df["ref_per_atom"])
        print(f"  original volume/atom q25/med/q75: {rq[0]:.2f} / {rq[1]:.2f} / {rq[2]:.2f} A^3")
        pq = pct(ratio_pa)
        print(f"  paired ratio steered/original (per-atom) q25/med/q75: "
              f"{pq[0]:.3f} / {pq[1]:.3f} / {pq[2]:.3f}")
        wq = pct(ratio_w)
        print(f"  paired ratio steered/original (written) q25/med/q75:  "
              f"{wq[0]:.3f} / {wq[1]:.3f} / {wq[2]:.3f}")

        summary.append(dict(
            alpha=alpha, n=len(df),
            median_per_atom=q[1], median_ref_per_atom=rq[1],
            median_ratio_per_atom=pq[1], median_ratio_written=wq[1],
            written_lattice_consistent=consistent,
            mean_per_atom=float(np.nanmean(df["per_atom"])),
            parsed_frac=float(df["written"].notna().mean()),
        ))

    out = pd.DataFrame(summary)
    print("\n=== summary ===")
    print(out.to_string(index=False))
    dest = Path("analysis") / f"volume_steering_check_{args.property}_layer{args.layer}.csv"
    dest.parent.mkdir(exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
