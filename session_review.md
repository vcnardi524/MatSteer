# Session review — `geometry_steering`, 2026-08-25 → 08-28

Everything changed on this branch, for review. 20 commits, `fda9a16..HEAD`, all pushed.
~104 files, +4,400 / −60.

Two arcs, in order: **Part A** is the pca_centroid method and its sweeps (commits
`ca6d771..7dc9338`); **Part B** is everything after — the diagnostics that explain why
those came back null, and the manifold built in response.

    b60df3d  Complete the experiments/ -> experiments_configs/ rename
    2d5c888  Session review through the manifold work
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
    7dc9338  Session review of the pca_centroid work
    75a5a82  The t-test script writes the results tables, not a person
    4a475eb  One schema for the steering results tables
    f473fa9  A per-prompt local centroid does no better than the global one
    d35be1e  Density targets 22.5 and 30 move the same absolute distance
    9add1ba  Plot the pca_centroid runs, and let a collapsed strength drop out
    ca6d771  Steer inside a PCA subspace toward a target centroid

---

# PART A — the pca_centroid method

## 1. What was built

### The steering method

Premise under test: the property direction is curved, not a straight line, so
`h + alpha * (mean_high - mean_low)` aims off the manifold. `pca_centroid` instead moves
inside the top-K PCA subspace of the training activations, toward the centroid of a
target class:

    z <- (h - mu) @ W.T                      project into the subspace
    h <- h + t * (centroid_pca - z) @ W      interpolate, map the change back

Only the K principal directions move; the rest of the residual stream passes through, so
it keeps supplying the context the centroid does not specify. There is no negative class
-- the target centroid alone sets the destination. `t=0` is no steering, `t=1` snaps the
subspace coordinates onto the centroid, `t>1` extrapolates past it.

**Design decision to check:** the spec said "unreduced(new_val) is fed as the embedding
for layer l+1", which read literally means `h <- mu + z_new @ W`, discarding the 960
dimensions outside the subspace. I implemented the residual-preserving form above
instead, matching the other half of the spec ("only the principal directions are
affected"). Verified numerically: subspace coordinates land exactly at `(1-t)z + t*c`,
the off-subspace residual is untouched, and the KV-cache tuple passes through.

### New scripts

| file | what it does |
|---|---|
| `scripts/steering/compute_pca_basis.py` | IncrementalPCA over the 2,047,889 layer-L train embeddings, streamed by parquet row group (the full layer is 8.4 GB). Once per layer, shared by every target. |
| `scripts/steering/compute_centroid_target.py` | Class = the N structures nearest `--target`. Writes the centroid in subspace coordinates. `--save-bank` also writes every member's coordinate, for the local variant. `--name` sets the output subdir (so `dos_electronic.band_gap` lands in `bandgap/`). |
| `scripts/data/build_density_atomic_table.py` | Volume per atom for all 2,285,719 v1 structures, read off the CIF text. |
| `slurms/compute_pca_basis.slurm` | Basis + centroid in one job. Reuses an existing basis unless `REFIT=1`. |

### Changed scripts

- **`steer_generate_cif.py`** — gained `--method {linear,pca_centroid,pca_local}`.
  Runs are named `steered_pca_*` / `steered_pcalocal_*` so all three methods coexist in
  one directory. `pca_local` runs a forward pass on the prompt, projects it, and averages
  the `--neighbours` nearest class members into a per-prompt centroid.
- **`utils.py`** — CIF-text scalar readers (`cell_volume_from_text`, `natoms_from_text`,
  `density_atomic_from_text`); `RESULT_COLUMNS` + `write_results_table()`.
- **`plot_steering_distribution_shift.py`** — `--method` to see the new runs at all,
  `--no-intersect` so one collapsed strength does not shrink every other curve,
  and a collapsed run now drops out with a message instead of crashing.
- **`steering_ttest.py`** — `analyse()` is now the single place a run is scored;
  `--all` writes the combined table. See section 4.

### Why density labels come from the CIF text

MP's `density_atomic` column covers only 58,650 of the 2.29M v1 ids -- too few to define
a 100,000-member class over the training set. The same quantity is in every CIF
(`_cell_volume` over the atom count from `_chemical_formula_sum`, both full-cell), so it
is read by regex with no pymatgen parse: 2,285,719 structures in 45 s, 0 unreadable,
median 19.10 A^3/atom. Also produced the histogram at
`analysis/v1_all/all/density_atomic_v1_density_atomic_histogram.png`.

---

## 2. What was run

Artifacts under `steering_vectors/pca_centroid/`: bases `pca_layer{9,14}_k64.parquet`,
centroids for density {22.5, 30} and band gap {0.25, 1.2}, plus one class bank.

| property | method | layer | target | t swept | prompts |
|---|---|---|---|---|---|
| density | pca_centroid | 14 | 30 | 0.25 / 0.5 / 1.0 / 2.0 | 1,000 x 3 |
| density | pca_centroid | 14 | 22.5 | 0.25 / 0.5 / 0.75 / 1.0 | 1,000 x 3 |
| density | pca_centroid | 9 | 30 | 0.25 / 0.5 / 0.75 / 1.0 | 1,000 x 3 |
| density | pca_local | 14 | 30 | 0.25 / 0.5 / 0.75 | 1,000 x 3 |
| band gap | pca_centroid | 14 | 1.2 eV | 0.25 / 0.5 / 0.75 / 1.0 | 1,000 x 3 |
| band gap | pca_centroid | 14 | 0.25 eV | 0.25 / 0.5 / 0.75 / 1.0 | 1,000 x 3 |

Each went through validate -> predict; the band-gap runs were also relaxed (M3GNet) and
re-scored, so they have both raw and relaxed gaps.

**Class sizes differ by property, on purpose.** Density uses 100,000 (target 30 spans
[28.50, 31.50]). Band gap uses **1,000**: the corpus median gap is 0.0014 eV and only
0.3% exceeds 1.0 eV, so "nearest to 1.2" starts scooping the near-zero pile from below
past about 1,000 members -- at 100,000 the "1.2 eV class" has a mean gap of 0.042 eV,
pointing the wrong way entirely.

---

## 3. Results — all null

`analysis/v1_all/test/steering_runs.csv`, 51 rows. **Read `cohens_d`, not `p_wilcoxon`**:
with ~980 paired prompts a 0.5% shift lands at p=1e-20 while |d| stays at 0.14.

    band_gap       raw      13 steered runs   max |d| = 0.098
    band_gap       relaxed  13 steered runs   max |d| = 0.074   min p_holm = 0.195
    density_atomic raw      14 steered runs   max |d| = 0.165

Nothing moved, at any target, layer, or strength, on either method. For scale: density
target 30 needs +0.192 in log10 from the control median and the best run covered 0.9%.

### Three explanations tested and ruled out

**Reach.** Target 22.5 needs +0.068 log10, target 30 needs +0.192 -- 2.8x further. Both
produced the same ~0.002 absolute move, so the displacement does not scale with where
the centroid sits. Head-to-head, p=0.21.

**Model degradation.** The cleanest nulls come at the model's best operating points:
band gap t=0.75 is 85.9% valid with p=0.55; density t=0.5 is 95% valid with d=0.12.

**Centroid choice.** `pca_local` aims at a point a median 10.63 from the global centroid
-- 2.5x the global centroid's entire offset from the corpus mean -- with matched
displacement and matched model damage (96.0/95.2/81.0% valid vs 96.1/95.0/82.5%). Result:
p=0.65 at t=0.25, p=0.17 at t=0.5.

### Relaxation erases the raw band-gap drift

The +0.03 eV nudge in raw CIFs becomes -0.001..-0.013 after M3GNet, every p >= 0.56.
Steering perturbs the written cell in a way MEGNet reads as slightly gapped; relaxation
settles it out. The linear method did the same thing. **Always check the relaxed column.**

### The finding worth keeping

The target class is not a place. Its centroid sits 4.22 from the corpus mean while
members sit a median 13.43 from that centroid -- signal-to-spread 0.31. And the class
matches a single Gaussian of the same covariance (13.43 vs 14.02 median, near-identical
tails), so it is one diffuse cloud, not several lobes with an average falling in the gap.
Structures at 30 A^3/atom are scattered through the same region as everything else.
"Interpolate toward the class" has nothing to aim at -- sufficient to explain every null
without any claim about geometry.

Two genuinely different destinations, matched displacement, matched damage, same
non-result. That points upstream of centroid choice: layer-14 activations may carry a
decodable signal for these properties without a controllable one.

**On the curvature premise specifically: this design could not test it.** The shape check
rules out the one concrete mechanism I could operationalize, and the local-centroid
experiment came back flat. That is a limitation of the experiment, not a refutation.

---

## 4. Bugs found — two of these affected reported numbers

**a) Run keys collided silently.** `discover_runs` keyed a run by strength alone, unique
for `alpha` but not for the pca methods, where a run is `(layer, target, strength)`.
Colliding keys overwrote each other with no error, three ways: two targets at the same t;
**one target at two layers** (density 30 exists at layer 9 and 14 -- layer 9 sorted later
and won, so a `--method pca_centroid` query returned layer 9's numbers labelled layer 14,
d = -0.091 instead of +0.113, wrong sign); and sg/nosg runs sharing a key. Fixed:
discovery filters on target and layer, `discover_sweeps()` enumerates them, method is
decided by filename prefix longest-first so `steered_pcalocal_` is never read as
`steered_pca_`. The tables reported during the session were computed inline and read the
right files, so the conclusions hold -- but the script would have contradicted them.

**b) sg runs were paired against a nosg control.** Different prompt sets. `family` is now
part of run identity, pairing happens within a family, and band gap's sg runs (alpha
11/16/25/40, generated June/July) are listed with population and value and empty paired
statistics rather than dropped or wrongly compared.

**c) `parse` success is not validity.** I reported density t=1.0 as d=0.677, "3.7x the
best linear run", computed over every CIF that parsed. Only 12.5% were chemically valid
and the failures were malformed grammar ("integer modulo by zero" x524). On valid rows
d=0.053. Anything computed from parsed text shows a false signal at any strength that
breaks the CIF grammar, and the apparent effect grows as the model degrades.

---

## 5. Things to check / disagree with

1. **sg vs nosg is inconsistent between properties, and I went along with it.** Band gap
   steers on composition-only prompts, density on prompts that dictate the space group.
   Each pairs against its own control so both are internally valid, but they are not a
   clean cross-property comparison. Both were null so no conclusion changes. Fix is a
   band-gap sg control at alpha=0, which would also un-strand the four June/July sg runs.
2. **The PCA basis is fitted on mean-pooled whole-CIF embeddings but applied to per-token
   hidden states**, and for `pca_local` the prompt lookup pools a header rather than a
   full CIF. The linear method has the same mismatch; it bites harder here because the
   local variant relies on position being meaningful, not just direction.
3. **`--neighbours 256` was not tuned.** Nothing was swept over it.
4. **Band-gap labels are NOMAD-only** -- 521,600 of 2,047,889 train ids. The centroid is
   a class mean over a quarter of the corpus while the basis is fitted over all of it.
5. **I wrote four results CSVs by hand** before consolidating them. They are gone, and
   `steering_ttest.py` writes these files now, but that is why they drifted.

---


---

# PART B — diagnostics, and the manifold

## 7. New code since Part A

| file | what it is |
|---|---|
| **`scripts/manifold.py`** (178) | The `Manifold` class. `encode(z) -> (u, residual)`, `decode(u)`, `project`, `save`/`load`. Also holds `bucket_centroids()`, shared with the plot script so the fitter and the figure cannot drift. |
| **`scripts/steering/fit_manifold.py`** (202) | Buckets a property, projects centroids into the PCA subspace, fits a smoothing spline against **arc length** weighted by `sqrt(count)`, saves a 2048-point polyline. |
| **`scripts/plots/centroid_pca_plots.py`** (257) | Bucket centroids in PCA space, coloured in property order, individual structures behind them at 10% opacity. `--sample`, `--max-per-bucket`, `--min-value/--max-value`, `--id-col`. |
| **`scripts/analysis/layer_causal_probe.py`** (259) | Which layer's injection moves the digits the model writes. Teacher-forced, no generation — 16 layers in minutes where a sweep per layer would be days. |
| **`scripts/analysis/layernorm_survival.py`** (191) | Whether the LayerNorm after the injection erases it. |
| **`scripts/analysis/manifold_distance.py`** (149) | Whether steering pushes activations off the data manifold. |
| **`run.sh`** (139) + **`slurms/_job.slurm`** | One entry point; settings in `experiments_configs/*.conf`; every submission logged to `runs.tsv` with its git SHA. |

**Modified since Part A:** `steer_generate_cif.py` (+61 — `--method manifold` joins linear
/ pca_centroid / pca_local), `steering_ttest.py` (+274 — `analyse()` is the single scorer,
`--all` writes the combined table), `compute_steering_vector.py` (+62/−24 — now streams,
so a vector can be built at every layer; verified against the stored one at cosine
0.99999998), `utils.py` (`RESULT_COLUMNS`, `write_results_table`).

## 8. The through-line

Nine arms, all null. Then each explanation ruled out in turn:

1. **Reach?** No. Targets 22.5 and 30 need 2.8× different distances and produce the same
   ~0.002 absolute move; head-to-head p=0.21.
2. **Degradation?** No. The cleanest nulls are at the best operating points — band gap
   t=0.75 at 85.9% valid, p=0.55.
3. **Centroid choice?** No. `pca_local` aims 10.63 away with matched displacement and
   matched damage; p=0.65 / p=0.17.
4. **LayerNorm eating it?** No. It reaches the logits at 13–23% relative magnitude, and
   under 0.5% lies along the direction mean-subtraction annihilates.
5. **Another layer?** No layer *translates* the property in one step. The integer-digit
   count — the term carrying orders of magnitude — is exactly zero at all 16 layers.
6. **Off the data manifold?** No, the opposite. Steering moves *toward* real activations;
   at t=1.0 it is 10× closer than unsteered, and that is where validity collapses.

**What that left.** At t=1.0 the state sits at Mahalanobis **0** — exactly the class mean,
which in 64 dimensions is the emptiest place in the distribution. Real members sit on a
shell 13.43 out. Interpolating toward a centroid aims at a hole. And the geometry is real:
the density path is ~2× longer than the straight line through the populated region (27.8°
count-weighted turning, replicated on val), pc2 carrying it at ρ = −0.95 at every layer.

So: fit the curve, slide along it, keep the off-curve offset.

## 9. The manifold

    m = Manifold.load(path)
    u, r = m.encode(z)                  # position on the curve, offset from it
    z_new = m.decode(u + delta) + r     # slide along, keep the offset

`encode` returns the residual as a **vector**, not a distance — that is what lets the
steered point stay exactly as far off the curve as it started. Stored as a dense polyline
rather than spline coefficients: scipy fits it, but the hook runs at every token of every
generation step and a scipy call there would sync GPU→CPU thousands of times per CIF.

Fit, density layer 14 train, trimmed to ≤ 40 Å³/atom:

    centroid spread about their own mean   6.228
    residual, straight-line fit            4.514   (explains 27.5%)
    residual, fitted curve                 0.227   (explains 96.4%)
    round trip                             9.5e-07
    arc vs property, rank correlation      +1.0000

Hook verified on 4,000 real activations — delta=0 is a no-op to 2.4e-07, off-subspace
residual untouched to 8.3e-07, displacement equals the curve's own to 1.9e-06, KV-cache
tuple passes through. And position in the distribution (a typical member sits at ≈√64 = 8):

    unsteered            7.71
    manifold delta=2     7.87
    manifold delta=5     8.35
    manifold delta=10    9.82
    pca_centroid t=1.0   0.00     <- the empty middle; validity there was 12.5%

**Not yet known: whether it steers the property.** A four-delta sweep (2/5/10/15, 1,000
prompts × 3) is running. Every check above is about activations; the nine null arms all
passed their activation-level checks too.

## 10. Further results since Part A

`analysis/v1_all/test/steering_runs.csv`, 51 rows, written only by `steering_ttest.py --all`.

    band_gap        raw      13 steered runs   max |d| = 0.098
    band_gap        relaxed  13 steered runs   max |d| = 0.074
    density_atomic  raw      14 steered runs   max |d| = 0.165

**Centroid-path geometry**, count-weighted turning angle (near 90° is a random walk):

    density_atomic  L7/L9/L10/L14   1.70-2.01x tortuosity   26.6-28.4°   pc2 ρ≈-0.95
    efermi          MP L14          7.17x                   56.2°        pc5 -0.91
    energy_above_hull MP L14        25.71x                  81.3°        pc3 +0.91
    band_gap        MP L14          12.74x                  106.7°       pc2 -0.90

Density curvature grows monotonically with depth (1.70× at L7 → 2.01× at L14). Band gap
never traces a path, and it is **not** sample size — MP has 12× NOMAD's gapped population
across 74 buckets and the angle went *up*. A hypothesis that failed: `energy_above_hull`
was chosen because its structural share (55.4%) nearly matches density's (57.1%); it came
back at 81.3° against 27.8°, so "variance surviving a fixed composition" does not predict
whether a representation has a trajectory.

## 11. Bugs found since Part A

**Run keys collided silently.** `discover_runs` keyed on strength alone, unique for
`alpha` but not for the pca methods where a run is `(layer, target, strength)`. Three
ways, including one target at two layers — density 30 exists at L9 and L14, L9 sorted
later and won, so a `--target 30` query returned **d = −0.091 when the layer-14 answer is
+0.113**, opposite sign.

**sg runs were paired against a nosg control** — different prompt sets. `family` is now
part of run identity.

**`lm_head` runs on the last position only** unless `targets` is passed, so an early
LayerNorm figure described one token rather than the sequence. Corrected to 0.166 / 0.191,
retracting a claim about the pca subspace having less leverage per unit norm.

**Bucket width was a confound.** Correcting it produced a better diagnostic than the thing
it fixed: density gets *worse* with wider buckets (real fine structure destroyed), band gap
gets *better* (noise averaged away). It also forced walking back "band gap has no path at
all" — at 16 buckets it looks about as good as density does at 24.

**A `--local` test overwrote a committed figure.** `runs.tsv` recorded the clobbering
command; `git checkout` restored it.

**Found and recorded, not fixed:** `plot_categorical.slurm`, `plot_tsne_pca.slurm` and
`plot_wyckoff.slurm` never pass `--partition` although the scripts require it — those jobs
die at argparse. `slurm_cocluster.sh` uses `N_CLUSTERS`/`RUN_NAME` with no guard.
`#SBATCH -V` in 9 files is inert (SLURM `-V` is `--version`; the export meaning is
PBS/Torque's `qsub -V`).

## 12. Things to check, beyond Part A's list

- **`v1_mp` has no split.** 96,229 of 154,879 ids appear nowhere in `splits_v1`, so MP runs
  use `--partition all` and mix training structures with unseen ones. Unverified.
- **NOMAD is 98.8% metallic.** Every band-gap conclusion rests on 6,296 gapped structures
  in train and 679 in val. MP has 78,668. "Gapped" means ≥ 0.05 eV, never ≠ 0 — 65% of
  NOMAD sits in (0, 0.05), nonzero but physically metallic.
- **Manifold trimming matters more than expected.** Over the full density range 54.8% of
  the arc length lies above 44 Å³/atom, holding 0.94% of structures. The fitted curve used
  for steering is trimmed to ≤ 40.
- **`--delta` is a uniform push, not a pull.** Every token moves the same arc distance
  regardless of where it started. Deliberate — `pca_centroid` was a pull and failed — but
  it means `--target` is only a convenience for naming a delta.

## 13. Next

The running sweep answers whether sliding along the curve moves the property. Both
outcomes are informative:

- **Validity holds where `pca_centroid` broke and d rises** — the on-manifold hypothesis is
  right, and this is the first method that works.
- **Validity holds and d does not rise** — leaving the data distribution was a real but
  *separate* problem from controllability, and the causal probe's conclusion stands: these
  activations carry a decodable signal without a controllable one.

Untested after that: steering only the token positions where the property is written;
larger K; activation patching, which tests whether the representation is causal at all,
independent of whether *adding* is the right operation.
