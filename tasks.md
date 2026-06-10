# Current Tasks

Active experiments. See `README.md` for everything completed so far and the
overall pipeline.

## 1. Steer all ~10,000 test structures

So far we have only steered **300 prompts** (3 samples each) at alpha = 11/16/25/40.
That sample is biased — the 300 prompts are higher-gap than the test-set average
(and mostly metals in ground truth), which muddies the matched-baseline comparison.

**Goal:** run steered generation over the *full* test split so the steered vs.
baseline comparison is over the same ~10k prompts the baseline already covers
(`validation/testset_baseline.parquet`, 10,266 rows, 1 sample each).

**How:** `slurms/steer_generate_cif.slurm` with `N_PROMPTS=0` (0 = all prompts).
- Pick alpha (likely 16 and 40, the ones already analyzed).
- 3 samples/prompt → ~30k generations per alpha. The script checkpoints every
  100 prompts and resumes from `done_ids`, so it survives the 36 h wall clock
  and can be re-submitted to continue.
- Then relax (`relax_steered_cifs.slurm`) and predict
  (`predict_bandgap.slurm`, raw + relaxed) as usual.

**Open question:** at full scale, does steering produce a measurable shift in the
gap>0 fraction once the prompt set is no longer the biased 300?

## 2. Steer without the space group in the prompt

The current prompt includes the space-group header (`--with-spacegroup`,
`PATTERN_COMP_SG`). That strongly conditions the structure and may be fighting the
steering vector. We want to test steering with a **composition-only** prompt
(`PATTERN_COMP`, drop `--with-spacegroup`).

**How:** remove `--with-spacegroup` from `slurms/steer_generate_cif.slurm`
(script already supports it via the `args.with_spacegroup` branch).

**Caveat — output naming collision:** `steer_generate_cif.py` names the output
`steered_{split}_clean_alpha{alpha}_layer{layer}.parquet` regardless of the
space-group flag. A no-spacegroup run would overwrite / resume into the existing
with-spacegroup parquet. **Before running this, add a `_nosg` suffix (or similar)
to the output path** so the two prompt conditions are kept separate.

**Open question:** does removing the space-group constraint give steering more room
to shift the band gap?

---

## Backlog / not started
- Predict alpha = 11/25 relaxed files (already relaxed in `relaxed/`, no
  prediction columns yet) to fill in the full alpha sweep table.
