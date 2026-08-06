# Current Tasks

Active experiments. See `README.md` for everything completed so far and the
overall pipeline.
## consider probing at the layers to see if they are actually encoding information

## consider different types of steering

## consider different generative models

## Known bugs / to fix

### relax resume trusts stale `relaxed/<stem>.parquet` (data contamination)
`relax_steered_cifs.py` resumes by merging any existing
`steering_results/relaxed/<stem>.parquet` on `(id, sample)` and skipping rows that
already have `cif_relaxed`. It **never checks that the reused `cif_relaxed` was
produced from the current `cif_steered`.** If a stem is regenerated (generation is
stochastic at temperature 1.0, so a re-run yields different structures for the same
prompt), the old relaxed structures no longer correspond to the new `cif_steered`,
but resume keeps them anyway → contaminated relaxed rows feed wrong structures into
`predict_bandgap.py`.

Hit on 2026-07-13: α16 with-SG and α40 with-SG were regenerated to full 30k on
Jul-12 but still had Jun-2 900-row relaxed files; the relax jobs resumed and reused
~850 stale relaxed structures each. Fixed immediately by deleting those two relaxed
files and resubmitting (see below), but the underlying resume logic is still unsafe.

**Fix idea:** guard the resume — only reuse a relaxed row if the source `cif_steered`
matches (e.g. store a hash of `cif_steered` alongside `cif_relaxed`, or refuse to
resume when the relaxed file is older than the generated_cifs file for that stem).

## Backlog / not started
- Predict alpha = 11/25 relaxed files (already relaxed in `steering_results/relaxed/`, no
  prediction columns yet) to fill in the full alpha sweep table.

## Reference / notes

### MP metadata has NO space-group / Wyckoff data — symmetry analyses can't use it
`preparsed_metadata_mp.parquet` and `metadata_mp.parquet` carry no space group,
Wyckoff, point group, or crystal system. For `metadata_mp.parquet` this was checked
against all 5,491 leaf columns (including the nested ones) — zero matches for
symmetry / space_group / wyckoff / point_group / crystal_system.

**Consequence:** anything keyed on symmetry — `symmetry_separability.py`,
`spec_cocluster_analysis.py`'s SG / point-group / sg_wyckoff enrichment — only works
on the NOMAD side (`metadata.parquet`, which does have `space_group_symbol`,
`point_group`, `wyckoff_letters`). Pointing them at `v1_mp` needs the space group
from somewhere else: the prep CIF header (`_symmetry_space_group_name_H-M`, which is
in the token stream the model reads, so it's arguably the better source for BOTH
datasets) or recomputed with `SpacegroupAnalyzer`. Not verified which of those is
cleanest — the CIF-header route was not tested.

Also note the MP id columns differ from the embedding ids: `metadata_mp.parquet` has
`id` = `MP_mp-bwbt` (a short hash-like code) and `material_id` = `MP_mp-32493`, while
`embeddings/v1_mp` ids look like `MP_mp-1217888` — so the join key is
**`material_id`**, not `id`.

### metadata_mp scalar cols are PRIMITIVE-cell, but CIFs/embeddings are CONVENTIONAL
`metadata_mp.parquet`'s `volume`, `nsites` (and any other extensive cell quantity) come
straight from MP's `SummaryDoc`, which describes the **primitive** cell. But the CIFs we
built — and therefore the `v1_mp` embeddings — are **conventional** cells (CifWriter
symprec standardizes; see below). So for centered lattices (I/F/C) the column is a
factor of 2/4 smaller than what the model ingested; only for P lattices do they match.
- Verified: mp-32502 `volume`=529.78 (primitive) but the tokenized CIF has
  `_cell_volume 1059.57` (conventional, 2x); `nsites`=36 vs conventional 72.
- **Implication for steering:** an extensive property like `volume` is a dirty target —
  the label describes a different cell than the embedding. If steering on "size", either
  recompute conventional volume (read `_cell_volume` from the CIF) or use an intensive,
  cell-invariant property (`volume/nsites`, `density`, `density_atomic`). Volume steering
  is shelved for now (2026-07-27).

### MP CIF pipeline: cells are already conventional; ~4% exceed block_size (deferred)
Building CIFs for the MP data (`scripts/build_mp_cifs_tar.py`, from `metadata_mp.parquet`'s
`structure` column → `CrystaLLM/cifs_v1_mp_orig.tar.gz` → `tar_to_pickle` → `preprocess`
→ `tokenize_cifs`):
- MP's `SummaryDoc.structure` is the **primitive** cell, but **`CifWriter(s, symprec=0.1)`
  runs its own symmetry analysis and emits the standardized CONVENTIONAL cell regardless**
  of the input cell. So the written CIFs are conventional and in-distribution — matching
  `cifs_v1_orig.tar.gz` (verified: mp-10164 → 16 atoms = orig; 400/400 atom-count and
  399/400 space-group match vs the orig tar; lattice differs ~33% only due to MP DB
  re-relaxation drift between snapshots). No explicit `get_conventional_standard_structure()`
  needed — that just doubles the spglib work.
- Tokenized length (`tokens_v1_mp`, 154,871 CIFs): min 187 / median 448 / mean 687 /
  p99 4,148 / max 11,745. **6,166 CIFs (3.98%) exceed block_size=2048.**
- **Truncation is architectural, not our choice:** the model uses learned absolute
  positional embeddings (`wpe = nn.Embedding(block_size, n_embd)`, 2048 rows) and
  `_model.py` asserts `past_len + t <= block_size` ("Cannot forward sequence of length
  N, block size is only 2048"). `extract_cif_embeddings.py:86` truncates to block_size to
  avoid that crash. To embed >2048-token CIFs without dropping tokens needs chunk-and-pool;
  a single forward pass beyond 2048 is impossible with this checkpoint.
- **Decision (2026-07-18): 4% tail is marginal, deferred.** Keep the current truncation
  for now; revisit chunk-and-pool or a length filter if MP-embedding quality matters.



### Steering vector definition + class band-gap stats (why negative/high alpha barely move the gap)
Vector (`compute_steering_vector.py`, layer 14, from
`metadata.parquet:dos_electronic.band_gap`):
`steering_vector = normalize(mean(insulators) - mean(metals))`, points metal→insulator.
`+alpha` = toward high gap, `-alpha` = toward metallic. Stored **unit-normalized**
(‖v‖=1), so `alpha` IS the norm of the perturbation added at layer 14; the raw
class-mean-diff norm is 17.49 (the natural metal↔insulator separation), so
alpha≈16 ≈ 1× that separation, alpha 40/60 ≈ 2.3–3.4×.

Class stats (col `dos_electronic.band_gap`, 582,596 labelled):
- **metals** (gap ≤ 0.05): n=**575,593**, mean **0.0034 eV**, median 0.0014
- **insulators** (gap ≥ 1.0): n=**1,905**, mean **2.4142 eV**, median 2.044
- middle (excluded): n=5,098
- overall mean gap 0.0138 eV, median 0.0014; separation ≈ 2.41 eV

Two structural facts that shape every steering result:
1. **302:1 class imbalance** (575k metals vs 1.9k insulators). Direction is fine
   but the insulator pole is estimated from 0.3% of the data — a fragile anchor.
2. **Dataset is ~99% near-zero-gap.** This is the floor effect: a toward-metal
   push (alpha −16) has almost nowhere to go, so alpha −16 ≈ alpha 0 in predicted
   gap is expected, NOT a bug. Also explains why MEGNet `%>0` piles up at ~0 — real
   materials genuinely cluster at zero gap. Steering IS applied correctly (verified:
   alpha 0 → zero vector = true control; hook always registered on `h[14]`,
   KV-cache-aware; validity degrades 95.5%→88.8% across alpha, proving the hook
   fires). The weak band-gap effect is the data prior, not the mechanism.

### Wyckoff info: what each source actually stores
Neither source gives the full `4c`-style Wyckoff label on its own — they carry
opposite halves:

- **CrystaLLM CIF**: has `_atom_site_symmetry_multiplicity` (the `4 / 2 / 2`
  column in the `_atom_site` loop) but NO `_atom_site_Wyckoff_symbol` — the
  **letter is dropped**. Multiplicity only.
- **NOMAD metadata** (`preparsed_metadata_nomad.parquet` →
  `results.material.topology[].symmetry.wyckoff_sets`): has `wyckoff_letter` but
  NO multiplicity field — multiplicity must be derived as `len(indices)`
  (verified constant per (space group, letter): 821 pairs, 0 violations).

So to reconstruct `4c` you must either join the two per site, or run
`SpacegroupAnalyzer(...).get_symmetry_dataset()["wyckoffs"]` (spglib/pymatgen),
which outputs letter + multiplicity together. The `wyckoff_letters` column added
to `metadata.parquet` (2026-07-16) is **letters only, sorted+deduped** (71.8%
coverage); `sg_wyckoff = "<space group> | <letters>"` drives cocluster enrichment.

### How CrystaLLM derives its CIF (not the DB's CIF verbatim)
`CrystaLLM/bin/preprocess.py` → `augment_cif()` takes a raw structural CIF and
rebuilds a standardized, augmented one via `crystallm/_utils.py`:
1. `replace_data_formula_with_nonreduced_formula` — `data_` header → full
   (non-reduced) formula, e.g. `data_Sc4Si2P2`.
2. `semisymmetrize_cif` — collapses the symmetry-op loop to a single
   `1 'x, y, z'` and writes ALL atoms in the cell explicitly (P1-style). This is
   why the header can say `I4/mmm` (#139) while `_symmetry_equiv_pos` lists only
   the identity.
3. `add_atomic_props_block` — injects `_atom_type_electronegativity`,
   `_atom_type_radius`, `_atom_type_ionic_radius` (NOT in the CIF spec; tokenizer
   comment says so). Rationale: model should learn composition→atomic-props first.
4. `round_numbers` — fixed decimal places.

`bin/download.py` only fetches a pre-packaged file from a URL; the raw API→CIF
collection step happened upstream (CrystaLLM authors), before this repo's
preprocessing.
