#!/usr/bin/env python3
"""
Build CrystaLLM/cifs_v1_mp_orig.tar.gz from the MP structures in metadata_mp.parquet.

Each row's `structure` column is a pymatgen as_dict JSON. We convert it to a
pymatgen CIF (same format as cifs_v1_orig.tar.gz: '# generated using pymatgen',
space-group header, symmetry ops) and add it to the tar as '<material_id>.cif'.
The member name minus '.cif' is exactly what tar_to_pickle.py uses as the id, so
naming with `material_id` (e.g. MP_mp-32493) keeps ids aligned with the embeddings.

Downstream (unchanged CrystaLLM tooling):
    python CrystaLLM/bin/tar_to_pickle.py CrystaLLM/cifs_v1_mp_orig.tar.gz \\
        CrystaLLM/cifs_v1_mp.pkl.gz
    python CrystaLLM/bin/preprocess.py CrystaLLM/cifs_v1_mp.pkl.gz \\
        -o CrystaLLM/cifs_v1_mp_prep.pkl.gz

Usage:
    python scripts/data/build_mp_cifs_tar.py
    python scripts/data/build_mp_cifs_tar.py --workers 16 --limit 1000   # quick test
"""
import argparse
import io
import json
import tarfile
import time
from multiprocessing import Pool

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


def make_cif(args):
    """(material_id, structure_json) -> (member_name, cif_text) or (name, None) on failure."""
    material_id, struct_json = args
    name = f"{material_id}.cif"
    if struct_json is None:
        return name, None
    try:
        s = Structure.from_dict(json.loads(struct_json))
        # CifWriter(symprec=0.1) runs its own symmetry analysis and writes the
        # standardized CONVENTIONAL cell regardless of the input cell — so even though
        # MP's SummaryDoc.structure is primitive, the emitted CIF is conventional,
        # matching cifs_v1_orig.tar.gz (verified: mp-10164 -> 16 atoms = orig). No
        # explicit get_conventional_standard_structure() needed.
        try:
            cif = str(CifWriter(s, symprec=0.1))       # symmetrized -> conventional, matches orig
        except Exception:
            cif = str(CifWriter(s))                     # fall back to P1 if spglib fails
        return name, cif
    except Exception:
        return name, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="metadata_mp.parquet")
    ap.add_argument("--out", default="CrystaLLM/cifs_v1_mp_orig.tar.gz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Only first N rows (testing)")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp, columns=["material_id", "structure"])
    if args.limit:
        df = df.head(args.limit)
    rows = list(df.itertuples(index=False, name=None))  # (material_id, structure)
    print(f"Converting {len(rows):,} structures -> {args.out} ...")

    written = skipped = 0
    t0 = time.time()
    with tarfile.open(args.out, "w:gz") as tar, Pool(args.workers) as pool:
        for i, (name, cif) in enumerate(pool.imap(make_cif, rows, chunksize=200), 1):
            if cif is None:
                skipped += 1
                continue
            data = cif.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = int(t0)
            tar.addfile(info, io.BytesIO(data))
            written += 1
            if i % 20000 == 0:
                print(f"  {i:,}/{len(rows):,}  (written {written:,}, skipped {skipped:,}, "
                      f"{i/(time.time()-t0):.0f}/s)", flush=True)

    print(f"\nDone. Wrote {written:,} CIFs, skipped {skipped:,} "
          f"(no structure / conversion error) in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
