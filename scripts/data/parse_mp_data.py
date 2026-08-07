import pandas as pd
import json


def to_material_id(mp_id):
    """Convert a letter-encoded MP id to the canonical numeric one, MP_-prefixed.

    The preparsed ids look like 'MP_mp-bwbt'; the suffix after 'mp-' is a base-26
    encoding (a=0 .. z=25) of the integer MP id, e.g. bwbt -> 32493, ft -> 149,
    n -> 13. We decode it and re-prepend 'MP_' so it matches the embedding ids
    ('MP_mp-32493'). Non-letter suffixes (already numeric) pass through unchanged.
    NB: a handful of 'mvc-' ids keep that prefix here, whereas MP's canonical id
    remaps them to 'mp-' (4/154,879 rows).
    """
    core = mp_id[3:] if mp_id.startswith("MP_") else mp_id
    prefix, sep, suffix = core.partition("-")
    if sep and suffix.isalpha():
        n = 0
        for c in suffix.lower():
            n = n * 26 + (ord(c) - ord("a"))
        core = f"{prefix}-{n}"
    return f"MP_{core}"


df = pd.read_parquet("preparsed_metadata_mp.parquet")

# Convert JSON string -> dictionary
results = df["results"].apply(json.loads)

# Extract all fields you listed
fields = [
    "energy_above_hull",
    "cbm",
    "num_magnetic_sites",
    "property_name",
    "decomposes_to",
    "volume",
    "possible_species",
    "is_metal",
    "efermi",
    "deprecation_reasons",
    "band_gap",
    "composition",
    "bulk_modulus",
    "shear_modulus",
    "equilibrium_reaction_energy_per_atom",
    "formation_energy_per_atom",
    "total_magnetization_normalized_formula_units",
    "composition_reduced",
    "formula_anonymous",
    "e_electronic",
    "task_ids",
    "vbm",
    "nsites",
    "has_reconstructed",
    "weighted_surface_energy_EV_PER_ANG2",
    "has_props",
    "dos_energy_up",
    "dos_energy_down",
    "nelements",
    "uncorrected_energy_per_atom",
    "database_IDs",
    "universal_anisotropy",
    "n",
    "chemsys",
    "is_magnetic",
    "theoretical",
    "density_atomic",
    "surface_anisotropy",
    "weighted_surface_energy",
    "weighted_work_function",
    "e_total",
    "is_gap_direct",
    "is_stable",
    "dos",
    "total_magnetization",
    "num_unique_magnetic_sites",
    "density",
    "e_ionic",
    "shape_factor",
    "ordering",
    "formula_pretty",
    "energy_per_atom",
    "structure",
]

out = pd.DataFrame()

for f in fields:
    # structure is a nested pymatgen as_dict with empty structs (e.g. "properties": {})
    # that pyarrow can't write, so dump it to a JSON string; other fields pass through.
    if f == "structure":
        out[f] = results.apply(lambda x: json.dumps(x["structure"]) if x.get("structure") is not None else None)
    else:
        out[f] = results.apply(lambda x, f=f: x.get(f, None))

# Keep MP id, plus the decoded numeric material_id (MP_-prefixed) for joining
# against the embeddings.
out.insert(0, "id", df["id"])
out.insert(1, "material_id", out["id"].apply(to_material_id))

out.to_parquet("metadata_mp.parquet", index=False)

print(out.shape)
print(out.head())
