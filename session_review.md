# Session review — `geometry_steering`, 2026-08-25 → 08-28

Everything changed on this branch, for review. 18 commits, `fda9a16..b80520a`, all pushed.
102 files, +4,323 / −58.

    b80520a  Steer by sliding along the fitted curve
    0b25bf5  A fitted 1-D manifold, saved, with encode/decode
    f294bee  One runner and named experiment configs
    fb70abc  Bucket width was a confound; its direction of effect is the diagnostic
    a45c4dc  energy_above_hull and efermi: smoothness is a spectrum
    947b1fc  Centroid paths across layers, properties and corpora
    954010c  The property is a curved ridge, thinner than the noise around it
    4c00778  Scope the causal-probe claim to what the probe can see
    7ce73ed  No layer translates the property in a single step
    f6c9956  LayerNorm is not why the steering does nothing
    065738d  README: the two steering methods that came after the linear one
    7dc9338  Session review of the pca_centroid work        <- previous review ended here
    75a5a82  The t-test script writes the results tables, not a person
    4a475eb  One schema for the steering results tables
    f473fa9  A per-prompt local centroid does no better than the global one
    d35be1e  Density targets 22.5 and 30 move the same absolute distance
    9add1ba  Plot the pca_centroid runs, and let a collapsed strength drop out
    ca6d771  Steer inside a PCA subspace toward a target centroid

---

## 1. New code — read these first

| file | what it is |
|---|---|
| **`scripts/manifold.py`** (178) | The `Manifold` class. `encode(z) -> (u, residual)`, `decode(u)`, `project`, `save`/`load`. Also holds `bucket_centroids()`, shared with the plot script. |
| **`scripts/steering/fit_manifold.py`** (202) | Fits a curve through property-bucket centroids and saves it. |
| **`scripts/plots/centroid_pca_plots.py`** (257) | Bucket centroids in PCA space, coloured in property order, with the individual structures behind them. |
| **`scripts/analysis/layer_causal_probe.py`** (259) | Which layer's injection actually moves the digits the model writes. Teacher-forced, no generation. |
| **`scripts/analysis/layernorm_survival.py`** (191) | Does the LayerNorm after the injection erase it. |
| **`scripts/analysis/manifold_distance.py`** (149) | Does steering push activations off the data manifold. |
| **`scripts/data/build_density_atomic_table.py`** (64) | Volume per atom for all 2.29M v1 structures, read off the CIF text. |
| **`scripts/steering/compute_pca_basis.py`**, **`compute_centroid_target.py`** | The PCA subspace and target centroids for pca_centroid steering. |
| **`run.sh`** (139), **`slurms/_job.slurm`** | One entry point; settings in `experiments_configs/*.conf`. |

**Modified:** `scripts/steering/steer_generate_cif.py` (+61, three new methods),
`scripts/analysis/steering_ttest.py` (+274, now the only writer of results CSVs),
`scripts/steering/compute_steering_vector.py` (+62/−24, now streams),
`scripts/utils.py` (CIF-text readers, `RESULT_COLUMNS`, `write_results_table`).

---

## 2. The through-line

Nine steering arms, all null. Then a sequence of checks, each ruling out one explanation:

1. **Reach?** No. Targets 22.5 and 30 need 2.8× different distances and produce the same
   ~0.002 absolute move. Head-to-head p=0.21.
2. **Model degradation?** No. The cleanest nulls are at the best operating points — band
   gap t=0.75 at 85.9% valid, p=0.55.
3. **Centroid choice?** No. `pca_local` aims 10.63 away from the global centroid with
   matched displacement and damage; p=0.65 / p=0.17.
4. **LayerNorm eating the injection?** No. It reaches the logits at 13–23% relative
   magnitude, and under 0.5% of it lies along the direction mean-subtraction annihilates.
5. **Some other layer?** No layer *translates* the property in a single step. The
   integer-digit count — the term carrying orders of magnitude — is exactly zero at all
   16 layers.
6. **Off the data manifold?** No, the opposite. Steering moves *toward* real activations;
   at t=1.0 it is 10× closer than unsteered, and that is where validity collapses.

**What that left**, and it is the finding worth keeping: at t=1.0 the state sits at
Mahalanobis **0** — exactly the class mean, which in 64 dimensions is the emptiest place
in the distribution. Real members sit on a shell 13.43 out. Interpolating toward a
centroid aims at a hole.

And the geometry is real: the density path is ~2× longer than the straight line through
the populated region (27.8° count-weighted turning, replicated on val), with pc2 carrying
it at ρ = −0.95 at every layer.

So: fit the curve, slide along it, keep the off-curve offset. That is `Manifold`.

---

## 3. Results, current

`analysis/v1_all/test/steering_runs.csv`, 51 rows, written only by `steering_ttest.py --all`.
**Read `cohens_d`, not the p-values** — with ~980 paired prompts a 0.5% shift reaches
p=1e-20 while |d| stays at 0.14.

    band_gap        raw      13 steered runs   max |d| = 0.098
    band_gap        relaxed  13 steered runs   max |d| = 0.074
    density_atomic  raw      14 steered runs   max |d| = 0.165

The manifold fit, density layer 14 train, trimmed to ≤ 40 A^3/atom:

    centroid spread about their own mean   6.228
    residual, straight-line fit            4.514   (explains 27.5%)
    residual, fitted curve                 0.227   (explains 96.4%)
    round trip                             9.5e-07
    arc vs property, rank correlation      +1.0000

And what the hook does to position in the distribution, on 4,000 real activations
(a typical member sits at Mahalanobis ≈ √64 = 8):

    unsteered            7.71
    manifold delta=2     7.87
    manifold delta=5     8.35
    manifold delta=10    9.82
    pca_centroid t=1.0   0.00     <- the empty middle; validity there was 12.5%

**Not yet known: whether the manifold method steers the property.** A four-delta sweep
(2/5/10/15, 1,000 prompts × 3) is running. Every check above is about activations; the
nine null arms all passed their activation-level checks too.

---

## 4. Bugs found — several changed reported numbers

**a) Run keys collided silently.** `discover_runs` keyed a run by strength alone, unique
for `alpha` but not for the pca methods where a run is `(layer, target, strength)`. Three
ways: two targets at the same t; one target at two layers (density 30 exists at layer 9
and 14 — layer 9 sorted later and won, so a `--target 30` query returned **d = −0.091
when the layer-14 answer is +0.113**, opposite sign); and sg/nosg sharing a key.

**b) sg runs were paired against a nosg control.** Different prompt sets. `family` is now
part of run identity.

**c) `parse` success is not validity.** I reported density t=1.0 as d=0.677, "3.7× the
best linear run". Only 12.5% of those CIFs were chemically valid and the failures were
malformed grammar. On valid rows, d=0.053. The apparent effect *grows* as the model
degrades.

**d) `lm_head` runs on the last position only** unless `targets` is passed, so an early
LayerNorm figure described one token rather than the sequence. Corrected to 0.166 / 0.191,
and it retracted a claim I had made about the pca subspace having less leverage per unit
norm.

**e) Bucket width was a confound.** Turning angles were compared across properties binned
at different widths. Correcting it produced a better diagnostic than the thing it fixed —
density gets *worse* with wider buckets (real fine structure destroyed), band gap gets
*better* (noise averaged away). It also forced walking back "band gap has no path at all":
at 16 buckets it looks about as good as density does at 24.

**f) A `--local` test overwrote a committed figure.** `runs.tsv` recorded the clobbering
command; `git checkout` restored it.

**Found and recorded, not fixed:** `plot_categorical.slurm`, `plot_tsne_pca.slurm` and
`plot_wyckoff.slurm` never pass `--partition` although the scripts require it — those jobs
die at argparse. `slurm_cocluster.sh` uses `N_CLUSTERS`/`RUN_NAME` with no guard.
`#SBATCH -V` in 9 files is inert (SLURM `-V` is `--version`; the export meaning is
PBS/Torque's `qsub -V`).

---

## 5. Things to check, or disagree with

1. **The steering equation preserves the off-subspace residual** rather than replacing the
   hidden state with its reconstruction. That was one of two readings of the original
   spec; everything on this branch depends on it.
2. **sg vs nosg is inconsistent between properties.** Band gap steers on composition-only
   prompts, density on prompts that dictate the space group, because that is where each
   property's alpha=0 control exists. Each is internally valid; the two are not a
   like-for-like cross-property comparison.
3. **The PCA basis is fitted on mean-pooled whole-CIF embeddings but applied to per-token
   hidden states.** The linear method has the same mismatch. It bites harder for
   `pca_local` and for the manifold, which rely on *position* being meaningful.
4. **`v1_mp` has no split.** 96,229 of 154,879 ids appear nowhere in `splits_v1`, so MP
   runs use `--partition all` and mix training structures with unseen ones. Unverified.
5. **NOMAD is 98.8% metallic.** Every band-gap conclusion rests on a corpus with 6,296
   gapped structures in train and 679 in val. MP has 78,668.
6. **Manifold trimming matters more than expected.** Over the full density range 54.8% of
   the arc length lies above 44 A^3/atom, holding 0.94% of structures. The fitter warns
   and takes `--min-value`/`--max-value`; the fitted curve used for steering is trimmed
   to ≤ 40.
7. **`--delta` is a uniform push**, not a pull toward a target: every token moves the same
   arc distance regardless of where it started. That is deliberate — `pca_centroid` was a
   pull and failed — but it means `--target` is only a convenience for naming a delta.

---

## 6. Next

The sweep now running answers whether sliding along the curve moves the property. Two
outcomes, both informative:

- **Validity holds where `pca_centroid` broke, and d rises** — the on-manifold hypothesis
  is right and this is the first method that works.
- **Validity holds and d does not rise** — leaving the data distribution was a real but
  *separate* problem from controllability, and the causal probe's conclusion stands: these
  activations carry a decodable signal without a controllable one.

Untested after that: steering only the token positions where the property is written
rather than all of them; larger K; activation patching, which tests whether the
representation is causal at all independent of whether *adding* is the right operation.
