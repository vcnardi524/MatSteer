# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `README.md` first.** It documents the science, the directory layout, the
`embeddings/` and `analysis/` tree conventions, and the current results. This file
covers how to *run* things and the cross-file conventions that are easy to get wrong.
`tasks.md` tracks what is in progress.

## RULES
All the code should be simple and accurate. It must be human readable. Naming conventions must be intuitive to humans and consitent. When speaking you should be clear and avoid flowery language, simple and accurate communication is ALWAYS the most effective. 

## Running things

There is no build, no test suite, and no package. Every script is a standalone
`python scripts/<stage>/<name>.py` run from the repo root, and real work goes through
SLURM.

Scripts are invoked **by path, not as modules**. They reach `scripts/utils.py` and
`scripts/predictors.py` via `sys.path.insert(0, dirname(dirname(__file__)))`. There
are no `__init__.py` files, so `python -m scripts.analysis.foo` does not work, and
importing one script from another needs `importlib.util.spec_from_file_location`
(see the top of `scripts/analysis/steering_ttest.py`).

### Three virtualenvs, not interchangeable

| venv | Used for | Why separate |
|---|---|---|
| `CrystaLLM/crystallm_venv` | crystallm generation, embeddings, sklearn analysis | torch 2.0.1+cu118 |
| `relax_venv` | M3GNet-PES relaxation | torch 2.4.1+cu121, the only one that runs on the V100 (sm_70) |
| `megnet_venv` | MEGNet band-gap prediction | CPU |
| `llamat_venv` | llamat2_cif embedding extraction | torch 2.4.1+cu121 + transformers |

`crystallm_venv`'s cu130 build **cannot run on the V100**. Anything touching M3GNet
must use `relax_venv`.

### SLURM — use `./run.sh`

```bash
./run.sh <experiment>                    # submit
./run.sh <experiment> --layer 9          # submit, overriding a flag
./run.sh --local <experiment>            # run inline (debugging)
./run.sh --dry-run <experiment>          # print the resolved command and stop
./run.sh --list                          # what experiments exist
```

Settings live in `experiments/<name>.conf` — a sourced bash file naming the script, the
venv, the SLURM resources, and the flags. `run.sh` resolves one, picks the venv, derives
a readable `--job-name`, and submits `slurms/_job.slurm`, which is the single template
and carries no per-experiment resource directives (they are all passed as sbatch CLI
flags instead).

**Overrides are appended to `ARGS` and win**, because argparse takes the last occurrence
of a flag — true for plain store actions, `store_true`, and `BooleanOptionalAction`.
So `./run.sh foo --layer 9` really does run at layer 9.

Every submission appends a line to `experiments/runs.tsv` (tracked): timestamp, job id,
experiment, git SHA, and the fully resolved command. That is the record of what produced
a result, including overrides the config file cannot know about.

**`--export` splits on commas**, which is why `run.sh` exports in the submitting shell
and passes a bare `--export=ALL`. Never put `VAR=value` on an sbatch command line:
`--export=ALL,LABEL_COLS=point_group,space_group_symbol` parses as
`LABEL_COLS=point_group` plus a stray variable, and the job runs one label while looking
successful.

`#SBATCH -V` in the older files is a no-op: in SLURM `-V` means `--version`, and the
"export the environment" meaning is PBS/Torque's `qsub -V`. `_job.slurm` omits it.

Logs go to `logs/<experiment>_<jobid>.out`/`.err`, gitignored. Under `run.sh` the stem
matches the experiment name; the older hand-written slurms each chose their own stem.

Not everything is migrated. The older `slurms/*.slurm` files still work and still take
env vars; `run.sh` is the path for anything new. Scripts with real control flow
(`compute_pca_basis.slurm` runs two scripts with a skip-if-exists guard) or layer loops
(`plot_tsne_pca.slurm`) stay as they are — a loop is not a config.

Some nodes are excluded in the SLURM headers for cause — `node11` advertises 192G but
has been seen with under 5G free, which OOM-killed a job. Keep the exclusions.

## Architecture

### Pipeline stages produce separate stores keyed on `(id, sample)`

The pipeline is `generate → validate → relax → predict`, and each stage writes its own
parquet keyed on `(id, sample)`. Downstream stages **join**, they do not carry data
forward:

- `steering_results/generated_cifs/` — `cif_steered`. The only home for raw CIFs.
  Models that do not emit a CIF add two more columns: `raw_output`, what the model
  actually wrote, and `decode_reason`, why decoding failed or what it had to patch up.
  An empty `cif_steered` means decoding failed, and `raw_output` is kept so the failure
  can be diagnosed without regenerating. crystallm writes neither column — its output
  already is a CIF, so the decode step is the identity.
- `steering_results/relaxed/` — `cif_relaxed`. The only home for relaxed CIFs.
- `steering_results/validation/` — **flags only, no CIF strings**.
- `steering_results/<property>/property_predictions/` — one file per source stem,
  accumulating `<base>_raw` (from the raw CIF) and `<base>` (from the relaxed one).

**alpha=0 controls live in `steering_results/baseline/` ONLY, never under a property.** At
alpha=0 the hook adds exactly zero, so the CIFs, their validity flags and their M3GNet
relaxation are all property-independent — `steered_test_alpha0.0_layer14.parquet` was
previously stored five times over, byte-identical. All four subdirs are shared,
including `property_predictions`: one control file accumulates `density_atomic`,
`band_gap`, `energy_above_hull`… side by side, because `compute_predictions.py` preserves
columns it does not own. `utils.py:steering_path()` looks in the property tree first and
falls back to `baseline/`, so a new property needs nothing regenerated — point
`--results-dir` at `steering_results/baseline` for every stage.

A control is chosen by PROMPT SET, not by `(family, strength)`. Those two are not enough:
`steered_test_alpha0.0_layer0_nosg` (1,000 density prompts) and
`steered_test_clean_alpha0.0_layer14_nosg` (10,286 bandgap ones) are both nosg with
strength 0, and keying on that alone let the wrong one win on sort order — density's nosg
arms silently paired against the bandgap control.
`plot_steering_distribution_shift.py:pick_control()` scores candidates by Jaccard overlap
with the ids the arms actually cover, so an exact prompt set wins over a superset ten
times larger. A baseline is keyed
on the PROMPT SET (sg vs nosg, `--n-samples`), which the filename already carries; it is
NOT keyed on layer, since no injection happens at any layer when alpha is zero.

That last point is verified, not assumed. On 2026-09-07 alpha=0 was generated afresh at
layer 0 and compared against the existing layer-1 controls: `steered_test_alpha0.0_layer0`
and `steered_test_alpha0.0_layer0_nosg` are IDENTICAL to their layer1 counterparts, all
3,000 rows and every column (`DataFrame.equals` true, same file size to the byte). So one
alpha=0 run per prompt set genuinely serves every layer, and the layer number in a
baseline filename is a historical label rather than a property of the run.

So novelty, relaxation, and prediction all read the flags file, join the CIF source on
`(id, sample)`, and process `is_valid == True` rows only. A new stage should follow
that shape rather than widening an existing file.

Files are matched across stores by **stem**, so a filename like
`steered_test_clean_alpha16.0_layer14.parquet` is a key, not a label. Renaming one
breaks the joins.

### Validity means different things for the two models

`is_valid` (CrystaLLM `_metrics.py:146`) is four checks ANDed: formula consistency, atom
site multiplicity, bond length, and space group. Keep the column name so the two models'
sweeps line up, but do not read the two numbers as the same bar.

For crystallm the model states its own `_symmetry_space_group_name_H-M`, so
`is_space_group_consistent` -- stated against `SpacegroupAnalyzer` detected -- is a real
test it can fail. llamat2-cif never writes a space group, so the decoder derives one WITH
`SpacegroupAnalyzer` and writes it; the check then compares that answer against itself and
passes ~100% of the time. **For llamat, validity is effectively a three-check bar.** Say so
next to any number that compares the two.

Two decode-side details that are easy to misread as results:

- `CifWriter(symprec=…)` refines to the CONVENTIONAL cell, so ~3.5% of structures come
  back with twice the atoms in twice the volume. Same crystal, re-expressed; composition
  and every intensive property are untouched. `refine_struct=False` is worse, not better
  (atom count survives 90.5% against 96.5%), because the symmetry operators are only
  valid in the standard setting.
- pymatgen names the data block with `Composition.reduced_formula`, which parenthesises
  grouped units (`data_LiFe(PO3)4`). CrystaLLM's `extract_data_formula` matches
  `data_([A-Za-z0-9]+)` and RAISES on those, so `is_formula_consistent` threw and
  `is_valid` read False for 25% of perfectly good structures. `llamat_prompts.py`
  rewrites the header to the alphanumeric form; the check is unchanged, since it compares
  all three formulas by `.reduced_formula`. Measured effect: is_valid 73.3% -> 98.3%.

### Adding a property

`scripts/predictors.py` holds a `REGISTRY` of `name -> factory -> PropertyPredictor`.
Subclass `PropertyPredictor` (or reuse `GeometricPredictor` for a direct structural
read), set `output_base`, put heavy model loading in `setup()` so the geometric path
stays cheap to import, and register it. `compute_predictions.py --property <name>`
then drives the shared load/validate/checkpoint loop with no further changes.

### `utils.py` owns the directory conventions

`analysis_dir()` and `embeddings_paths()` build the paths described in the README. Use
them rather than composing paths by hand.

Both trees carry a **model** level, because the same corpus read by two models gives two
unrelated sets of vectors and the second would otherwise overwrite the first file for
file. They put it in **different positions, on purpose**:

    embeddings/<dataset>/<model>/<variant>/cif_layer{N}
    analysis/<model>/<dataset>/<variant>/<partition>/

`analysis/` leads with the model because an analysis directory is a per-model
deliverable; `embeddings/` leads with the dataset because a layer's shards are read
together whichever model wrote them. Do not "harmonise" them.

`analysis/corpus/` is **not a model**. Outputs that never load one — corpus property
histograms, Wyckoff/space-group counts, MP hull coverage, metadata property coverage —
go there once rather than being duplicated under every model. `analysis_dir()` and
`analysis_root()` accept `model=CORPUS_DIR`; `add_partition_args` deliberately does not,
since no script can load `corpus` as weights.

`variant=None` does NOT imply corpus, and conflating them would file every steering
result under `corpus/`: the steering tables pass `variant=None` because they come from
generation rather than from reading a CIF variant, and they are entirely model-dependent.
Pass `model` explicitly; it cannot be inferred.

Use `analysis_root(model)` for outputs that span datasets (cross-dataset probe summaries)
or describe the corpus as a whole, rather than writing loose files at `analysis/`.

Two flags that were both `--model` before a second model existed, now split:
`--model` is the registered NAME (`utils.MODELS`: crystallm, llamat2) and decides the
path; `--ckpt-dir` is the filesystem path to the weights. `embeddings_paths()` validates
the name, so a typo raises instead of silently creating a sibling tree that reports zero
rows done and re-extracts everything into it.

A layer index is NOT comparable across models — crystallm has 16 blocks at 1024 dim,
llamat2 has 32 at 4096 — and no steering vector, PCA basis or manifold transfers between
them. `scripts/backends.py` holds the only architecture-specific code, in two backends.
Extraction and steered generation BOTH import them, because both hook the same `blocks`
attribute -- `model.transformer.h` against `model.model.layers` -- one to capture a
hidden state and one to modify it, and two copies of that line would drift.
`extract_cif_embeddings.py` re-exports `load_model` because seven scripts import it from
there. Both models are fed identical CIF text so the comparison is between models rather
than between inputs. Its heavy imports are deliberately inside the loaders, since
crystallm (omegaconf, pinned pymatgen) and llamat2 (transformers) will not share a venv.

`add_partition_args()` makes `--partition` required with no default, deliberately:
89.6% of labelled structures are in CrystaLLM's own training set and only 0.45% in its
test set, so a number computed over `all` cannot separate learning from memorisation.
`filter_partition()` does the filtering, reading the `splits_v1.parquet` cache of
CrystaLLM's three split pickles.

### CrystaLLM is a submodule

Pinned to the `kv-cache` branch. `utils.py` loads `CrystaLLM/crystallm/_utils.py`
directly by file path to reuse `replace_symmetry_operators`, `remove_atom_props_block`
and friends, so the checkout must be populated even for scripts that never generate.

`utils.py:postprocess()` restores symmetry operators on a generated CIF. It must run
before parsing, or structures with a non-`P 1` space group parse wrong. It is a no-op
on already-indented reference CIFs — that difference has bitten this repo before.

## Data conventions

Ground-truth band gap is `metadata.parquet:dos_electronic.band_gap` (eV). **Not**
`energy_lowest_unoccupied - energy_highest_occupied` — those are raw Joules and their
difference is a corrupt LUMO-HOMO gap. Three legacy scripts still use it and are named
in the README; do not copy their approach.

Every steering arm generates from the **same 1,000 test structures**,
`CrystaLLM/cifs_v1_test_sample1000.pkl.gz` (fingerprint `cccf9b87455ac110`). That shared
prompt set is what makes the paired t-test valid — each arm and the alpha=0 control pair
on `id`, so prompt-to-prompt variance differences out. It is gitignored inside the
submodule, so `scripts/data/make_test_sample.py` reproduces it
(`random.Random(42).sample`, verified exact) and `data/test_sample1000_ids.csv` tracks the
ids. Run it with `--verify` before trusting a comparison against older results; it refuses
to overwrite a subset that differs from what it would draw.

**llamat2-cif pairs through `--prompt-csv`, not through its default prompt.** Its
UNCONDITIONAL prompt is one constant 203-token instruction, so there is no per-structure
`id` to pair on and each sample is an independent draw; the ids in those runs
(`draw00000`, …) are draw indices, not materials. `data/llamat_test_sample1000.csv`
fixes that: 1,000 structures drawn from llamat's own test split with
`random.Random(42).sample` (fingerprint `11da7395271991ca`), each turned into a
CONDITIONAL prompt naming its formula, elements and space group. That gives 1,000
distinct prompts, so arms pair on `id` exactly as crystallm's do. Regenerate with
`scripts/data/make_llamat_test_sample.py --verify`.

Two things about that prompt that are easy to get wrong:

- **Composition alone is off-distribution.** Training drew `k = randint(0, 3)` conditions:
  `k == 0` gave the empty dict (fully unconditional) and `k >= 1` gave formula + elements
  plus at least one of `OPTIONAL_CONDITIONS`. Formula + elements ALONE never appeared, so
  there is no clean `nosg` counterpart to crystallm's two prompt sets. Of the three
  optional conditions, `spacegroup.number` is the only one that is not also a steering
  target -- conditioning on `formation_energy_per_atom` or `e_above_hull` while steering
  toward it hands the model the answer. `band_gap` has a phrase in `CONDITION_PHRASES`
  but is NOT in `OPTIONAL_CONDITIONS`, so it was never a training condition and cannot
  leak.
- **The element list comes from the CIF's `_chemical_formula_sum`**, via
  `elements_from_formula_sum`, because that is what training used. test.csv's `elements`
  column is alphabetised and disagrees on 63.7% of structures.

Use `--paired-seed` on top, which gives draw k of every arm the same random stream
(common random numbers). It is OFF by default because turning it on changes the sampling
stream, so existing crystallm results would not reproduce byte-for-byte.

**Steering strength is `--alpha-rel`, not `--alpha`, once more than one model is in play.**
The stored vector is unit-norm, so `alpha` is the ABSOLUTE norm added to the hidden
state -- and hidden states are not the same size. Measured per-token on answer tokens:
crystallm layer 14 has |h| = 165.8, llamat layer 24 has |h| = 22.2. So crystallm's routine
alpha 40 is 24% of its residual stream but **180%** of llamat's, which overwrites rather
than steers: llamat degenerates into repetition and decodes to nothing. `raw_norm` -- the
class-mean difference before normalising -- is ~10% of |h| in BOTH models at every layer
measured (crystallm L14 10.5%, llamat L24 10.9%), so it is the portable unit.
`--alpha-rel 1` means one class separation.

`metadata_mp.parquet` **mixes DFT thermo types**, and this silently corrupts any
comparison. Most rows are GGA/GGA+U, but some are pure `r2SCAN`, whose energies sit on a
different scale and whose `energy_above_hull` is measured against a different hull.
`mp-2912291` (r2SCAN) has a *worse* formation energy than `mp-25977` (GGA_GGA+U) at the
same composition yet a *smaller* stored `energy_above_hull` — impossible within one
scheme. Comparing an r2SCAN row against a GGA/GGA+U hull produces a ~0.3 eV/atom
discrepancy that is not an error. The file stores no `thermo_type` column, so fetch it
live from MP when the scheme matters. `scripts/eval/validate_hull_predictor.py` does this.

`formula_pretty` is **not a unique key** either: 46% of `metadata_mp` rows share a
`(chemsys, formula_pretty)` with another material — 22 distinct materials are called
`LiFe(PO3)4`, with formation energies spanning 0.3 eV/atom. Join on `material_id`.

`utils.py:DEFAULT_LABEL_COLS` asks for `band_gap_ev`, which does not exist in
`metadata.parquet` and is **silently dropped**. `load_labeled_embeddings` therefore
never returns a gap — join it yourself.

`cif_layer<N>` is the **output of transformer block N**, not an input embedding.
`extract_cif_embeddings.py` registers a forward hook on `model.transformer.h[N]`, and
there is no `wte`/`wpe` extraction anywhere in the repo. So layer 0 already has one
attention + MLP behind it. A high layer-0 probe score means "one block suffices", **not**
"no computation needed" — it cannot show a property is read straight off the text. For
that, use a model-free baseline: parse the value out of the CIF and compare. Done for
density on 2026-09-01 — `_cell_volume / sum(_atom_site_symmetry_multiplicity)` reproduces
`density_atomic` exactly (R^2 = 1.000000 on 29,832 CIFs), so its probe scores have a
parse ceiling. The DFT labels (band gap, efermi, energy_above_hull) appear nowhere in a
CIF and have no such ceiling.

Probes must be evaluated on the `by_formula` split, not `random`. Chemical formulas
repeat across the corpus, so about half of val shares a formula with train and can be
answered by memorisation. `by_formula` scores only rows whose formula never appeared in
training. The check that it works: `lookup_acc` collapses to exactly `majority_acc`.
Note the two splits also score **different populations** (majority shifts 0.17 → 0.31),
so the columns are not a like-for-like difficulty comparison.

## What is version-controlled

`.gitignore` excludes all `*.parquet`, `*.pkl*`, `embeddings/`, `steering_vectors/`,
and the results subdirectories — data stays local. But `analysis/` **CSVs and PNGs are
tracked** (~160 files), so a rerun that changes numbers shows up as a diff. Commit the
code and the outputs it produced together.
