#!/usr/bin/env python3
"""
Add two NOMAD band-gap columns to metadata.parquet:
  electronic.band_gap       <- results.properties.electronic.band_gap[0].value
  dos_electronic.band_gap   <- results.properties.electronic.dos_electronic.band_gap[0].value

Values are stored in eV (NOMAD stores Joules; converted by *6.2415e18).
Then prints max/min/median/mean/q1/q3 over the non-NaN entries of each column.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

J_TO_EV = 6.241509074e18


def as_dict(node):
    """NOMAD sections are sometimes wrapped in a list; coerce list -> first element."""
    if isinstance(node, list):
        return node[0] if node else None
    return node


def first_value(band_gap_list):
    """Return MIN value across NOMAD band_gap channels (spin-up/down), else None.

    A material conducts if either spin channel is metallic, so the minimum gap
    across channels is the physically meaningful one. Also flag multi-element.
    """
    if not isinstance(band_gap_list, list) or not band_gap_list:
        return None, 0
    n = len(band_gap_list)
    vals = [it.get("value") for it in band_gap_list
            if isinstance(it, dict) and it.get("value") is not None]
    if not vals:
        return None, n
    return min(vals), n


def main():
    print("Loading preparsed NOMAD metadata ...")
    pre = pd.read_parquet("preparsed_metadata_nomad.parquet", columns=["id", "results"])
    print(f"  {len(pre):,} rows")

    elec_val, dos_val = {}, {}
    multi_elec = multi_dos = 0
    for id_, r in zip(pre["id"].values, pre["results"].values):
        if r is None:
            continue
        try:
            d = json.loads(r) if isinstance(r, str) else r
        except Exception:
            continue
        elec = as_dict((d.get("properties", {}) or {}).get("electronic")) or {}

        v, n = first_value(elec.get("band_gap"))
        if v is not None:
            elec_val[id_] = v * J_TO_EV
            if n > 1:
                multi_elec += 1

        dos = as_dict(elec.get("dos_electronic")) or {}
        dv, dn = first_value(dos.get("band_gap"))
        if dv is not None:
            dos_val[id_] = dv * J_TO_EV
            if dn > 1:
                multi_dos += 1

    print(f"  electronic.band_gap.value found: {len(elec_val):,}  (multi-element lists: {multi_elec:,})")
    print(f"  dos_electronic.band_gap.value found: {len(dos_val):,}  (multi-element lists: {multi_dos:,})")

    print("\nLoading metadata.parquet ...")
    meta = pd.read_parquet("metadata.parquet")
    print(f"  {len(meta):,} rows, cols={list(meta.columns)}")

    meta["electronic.band_gap"] = meta["id"].map(elec_val)
    meta["dos_electronic.band_gap"] = meta["id"].map(dos_val)

    meta.to_parquet("metadata.parquet", index=False)
    print("  Saved metadata.parquet with 2 new columns (eV).")

    for col in ["electronic.band_gap", "dos_electronic.band_gap"]:
        v = meta[col].dropna().astype(float)
        print(f"\n=== {col}  (eV) ===")
        print(f"  n (non-NaN): {len(v):,}   NaN: {meta[col].isna().sum():,}")
        if len(v):
            print(f"  min    : {v.min():.4f}")
            print(f"  q1     : {v.quantile(0.25):.4f}")
            print(f"  median : {v.median():.4f}")
            print(f"  mean   : {v.mean():.4f}")
            print(f"  q3     : {v.quantile(0.75):.4f}")
            print(f"  max    : {v.max():.4f}")
            print(f"  <0 (impossible): {(v < 0).sum():,}   ==0: {(v == 0).sum():,}   >0: {(v > 0).sum():,}")


if __name__ == "__main__":
    main()
