# Decision record

Why the engine is the way it is. Most of these are **negative results** — things built,
measured, and then not shipped — which are the expensive kind to rediscover.

This is a record, not a specification. Where it disagrees with `AGENTS.md` or the code,
they are right and this is stale. Live constraints that still bind are in `AGENTS.md`
section 8, not here.

## Sampling and variance reduction

**Correlated sampling across beamlets is the default.** Keying each beamlet's histories
on the within-beamlet index rather than the global history index makes corresponding
histories replay the same interaction sequence, differing only in entry position. It
roughly **halves** the renormalized plan-dose error at matched per-beamlet sigma, and is
never worse on raw plan quality. Established by optimizing on one noisy Dij and scoring
on a second, independent ground truth across six statistics levels, in water and through
a heterogeneity (`examples/noise_bias_study.py`). `correlated=False` survives only as a
test instrument.

**Correlated sampling makes plan sums noisier, not only differences quieter.** The
covariance that collapses the variance of a column difference is positive, so it adds to
the variance of a plan sum — and the quadrature of the column variances,
`variance @ w**2`, cannot see it. Measured against the seed-to-seed spread of an
open-field plan at uniform weights (`examples/dij_plan_sigma_demo.py`, 6 MeV, water,
0.4 cm voxels): true variance / quadrature = 1.02 at 2 cm beamlets, 1.07 at 1 cm, 1.4 at
0.5 cm; 1.00 at every pitch for independent columns, where the quadrature also equals a
direct forward run's batch variance. The quadrature is numerically the *same* number in
both modes (marginal column variances are mode-invariant); only its meaning changes. The
library exports the column variances with a warning and leaves the plan-sigma route to
the adapter: an independent Dij, a direct forward run, or batch-resolved scoring.

**Compton splitting ships off** (`PHOTON_SPLIT_N = 1`). Built, unbiased, test-pinned at
N = 2 — and dormant. Efficiency `1/(σ²·time)` came out below 1 on the analytic-water Dij,
then again on a phase-space source at **0.75 / 0.48 / 0.28 for N = 2 / 4 / 8** — worse,
monotonically. Splitting at the first Compton lets copies decorrelate only *after* that
scatter, so variance reduction saturates far below 1/N while cost grows linearly. It also
cannot reach the low-dose tail, which is fed by rare wide-angle scatters that uniform
splitting cannot target. Turning it on needs a new measurement, not an argument.

**Russian roulette on sub-0.5 MeV photons is always on.** The threshold sits just below
the 511 keV annihilation line so annihilation photons stay analog. Unbiased by
construction and test-pinned against a roulette-free run.

**Weight linearity holds only above the roulette weight cap.** Below it, a boost chain
that crosses the cap in one run but not another diverges that history's stream — and
correlated sampling then replays the divergence into every column. Fair per run, pinned
statistically below the cap, bit-pinned above it.

## Electron transport

**Goudsmit-Saunderson is the shipped multiple-scattering model**, with a substep energy
fraction of 0.20. The Gaussian small-angle model remains as the `"gaussian"` instrument —
a model whose approximations differ from the default's is what makes the default's error
visible.

**GS was adopted for accuracy, not speed.** The speed premise was measured and
**falsified**: at the previously shipped substep fraction the angular cap was inert
throughout soft tissue, so larger steps were never available there.

**The multiple-scattering strength is the Class-II first transport moment of the
Moliere-screened Rutherford law**, replacing the
Rossi-Greisen/Highland core width `(14.1/pv)^2 / X_0` on 2026-09-04. Goudsmit-Saunderson
pins `<cos theta> = exp(-s N sigma_tr)` exactly, so the retained strength is
`2(N_A/A)[Z^2 sigma_tr + Z(sigma_tr - sigma_tr,M^hard(E, ECUT))]`: nuclear elastic plus
subthreshold electron scattering. The above-ECUT Moller moment is removed because the
transport loop applies those deflections explicitly. This preserves the soft e-e
component that no other part of the loop supplies while eliminating the Class-II double
count exactly. Highland lacks the energy-growing `ln(1/eta) - 1` factor: the restricted
moment is 1.25x Highland at 1 MeV, 1.68x at 10 MeV and 1.77x at 15 MeV in water. The
hard correction is 0.4-4.2 % of the unrestricted `Z(Z+1)` strength over 0.5-15 MeV,
versus about 13 % removed at all energies by a nuclear-only `Z^2` approximation in water.
Measured on monoenergetic pencil kernels in water against EGSnrc and TOPAS (1 mm voxels,
1 mm axial shell): the integrated build-up at half dmax went from 0.95-0.97 of EGSnrc to
0.99-1.01 and the central-ray excess from +4 / +14 / +19 % (1 / 6 / 15 MeV) to +3 / +2 /
+7 %, inside the TOPAS-vs-EGSnrc spread; beyond dmax nothing moved (CPE). Broad-beam
electron R50 now matches EGS4 (Rogers & Bielajew, Med. Phys. 13, 687 (1986), Table III)
to +0.4-0.9 % at 5-20 MeV and +2.3 % at 3 MeV, where Highland read 8-9 % long — note that
E0 = 2.33 R50 is the AAPM/ETRAN approximation, not the EGS relation, and would mis-set a
gate by 3-8 % at 5-10 MeV. The tabulated
implementation combines EEDL nuclear scattering with the same analytic subthreshold
electron moment (its 15 MeV axial excess fell from +11 % to +8 %; at 6 MeV it now
overscatters slightly, build-up 1.03, because EEDL's nuclear moment is interpolated
across its 0.256-10 MeV shape gap and sits 1.1-1.3x the closed form at 1-6 MeV).
Stated omission: no Mott (spin-relativistic) factor on the
closed-form moment, a few percent at `beta -> 1` in low-Z media.

**Each electron substep is capped at the next voxel face, and so is each hinge chord.**
Without the cap a step carried one medium's stopping power across a boundary into the
neighbour's mass, producing a single-voxel interface spike — **1.95x before, under 1.4x
after** at a pure density step. The cap alone left the hinge's second chord free to cross
the face after the deflection; that residual, dominant at air contrast, was closed on
2026-10-18 (see "The random hinge's second chord" below). What remains is step-size
convergence of the hinge at an interface plus genuine interface dosimetry.

**Positrons stay Møller-approximated** — no Bhabha, annihilation at rest. If upgraded,
both land together as the new default, with a test showing the dosimetric effect.

## Cross-section data

**Berger-Seltzer ICRU-37 stopping is the default even where EEDL data exists.** The EEDL
route is provenance-consistent but sits ~4 % from ESTAR; the analytic form is
ESTAR-exact. Provenance purity lost to accuracy.

**Coherent scattering uses EPDL form factors**, replacing a Thomson-limit angular model.

**Enabling a coherent channel requires refitting pair production.** The analytic pair
channel is calibrated against water totals that *already include* coherent scattering, so
adding a real coherent channel without recalibrating against coherent-free totals
double-counts attenuation. Channels must come from one consistent decomposition of one
library.

**2 %/2 mm on the depth-dose gate is not reachable, and the benchmark is the limiter.**
The tabulated backend barely differs from the analytic one here — the analytic photon
*total* was already NIST-calibrated, so a more accurate channel split moves the shape
little. Decisively: *widening* the phantom toward the infinite-field limit makes the
deep-tail residual **worse**, which is the lateral-integrated pencil beam overestimating
the benchmark's finite-field, 6 mm-tube scoring at depth. The gate stays 5 %/3 mm until a
fully specified benchmark exists. See `AGENTS.md` section 8.

**Cross-section libraries are pinned by SHA-256 and verified fail-closed.** These bytes
*are* the cross-sections; a corrupted cache or a silent upstream revision would surface as
unexplained drift in a dose gate rather than as an error.

## Reference GS startup

**The reference sampler shares the existing persisted GS grid, growing its requested
rectangle on demand.** On 2026-09-05, the existing 21 MB, 5,148-node Warp cache covered
all 494 nodes requested by the collimated Dij ledger test and all 914 requested by the
boundary-truncation test. Loading it solves their repeated construction cost without a
new format or construction-version bump. Cold reference runs grow a dense rectangle;
this can build unrequested interior nodes, but preserves compatibility with the eager
reader. Owned, immutable rows enter the existing node memo so source caches do not
retain obsolete grid allocations. Both grid writers share the existing thread lock.

**Eager reference initialization loses badly on short cold runs.** Interleaved lazy /
eager / lazy / eager measurements on Windows, Python 3.13.14, NumPy 2.4.6, using the
6 MeV analytic-water window and unchanged production build constants:

| Cold workload | Lazy persistence | Eager initialization | Nodes, lazy / eager |
|---|---:|---:|---:|
| One GS sample at 1 MeV, theta-squared 0.003 | 0.587 / 0.613 s | 30.27 / 32.49 s | 4 / 3,562 |
| Collimated Dij, 900 histories | 93.95 / 96.73 s | 32.30 / 31.17 s | 777 / 3,562 |

These initial measurements predate the shared rectangle filler described below.
The lazy rectangle adds some first-run cost relative to the original memory-only
constructor, but avoids a mandatory roughly 31-second startup for even one sample.
This favors lazy persistence (option b); a separate node-store format is unnecessary.

**Sharing column parallelism does not by itself accelerate incremental cold fills.**
The review follow-up factors both builders through `_fill_rectangle`, preserving the
existing threshold of 256 **new** nodes per fill and the 16-worker cap. The reference
sampler requests a whole 2x2 bracket under one lock and saves once. It also returns
immediately when a defensive disk re-read finds coverage. Both writers recover from
save failures and clean up temporary files. Inverse-CDF resolution now participates
in disk and memory identities; existing version-1 grids remain readable only at their
historical 4096-bin resolution. Construction and format are unchanged, so version 1
remains appropriate.

Interleaved before/after/before/after cold measurements against the initial persisted
implementation, with the same production constants and 900-history Dij:

| Cold workload | Before review fixes | After review fixes |
|---|---:|---:|
| One GS sample | 0.615 / 0.599 s | 0.592 / 0.595 s |
| Collimated Dij, 900 histories | 91.81 / 93.30 s | 95.91 / 93.75 s |
| Dij nodes built | 777 | 777 |
| Dij saves / bytes written | 14 / 18,163,256 | 11 / 18,032,692 |

The largest revised extension adds 180 nodes, so **none starts a pool**. These results
do not demonstrate a cold transport speedup and do not recover eager initialization's
roughly 31-second result. A single sample drops from three saves to one; most later
growth events already extend only one axis. Larger extensions now share eager
parallelism, with serial/parallel reconstruction pinned byte-equal after clearing
the node memo. Changing worker-selection policy would need a separate measurement.

Warm review comparisons also ran before/after/before/after in fresh processes against
the same stored grid. Direct 900-history Dij invocations took 1.014/0.982 s before and
1.006/0.997 s after; direct calls to `test_interface_voxel_is_not_spiked` took
61.69/61.12 s before and 60.83/60.92 s after. All made zero node builds and zero saves.
These direct timings exclude pytest initialization and are distinct from the original
pytest timings below.

The review's full fast-suite comparison ran A/B/A: 193.70 s (A), 200.13 s (B),
191.81 s (A), all passing. A runs the 1,292 common tests, excluding the cache-specific
test files; B runs all 1,312 tests, including the twenty cache cases (ten added by
this review). These wall times include pytest startup. Windows reported 2,200 MHz;
they remain laptop measurements with warm transport-grid coverage. Production-dose,
column-sigma and CSC bytes also matched each target's prior result on reference,
Warp CPU and CUDA independently.
The Warp CPU check also extended a four-node reference cache by 3,558 nodes through
the shared parallel filler, then CUDA read that completed grid without rebuilding;
both reproduced their own baseline dose, sigma and CSC bytes.

**Fresh processes reuse the cached nodes without changing dose.** A restores the
original `gs_scaled_deflection_table(exp(i / bins), exp(j / bins), n_u=nodes)` reference
lookup; B uses the persistent lookup. Each measurement starts a fresh interpreter,
with the same existing grid available to Warp in both cases. Times below measure
`pytest.main`, including its initialization. Histories, seeds and tolerances are
unchanged. The named tests ran A/B/A/B; successful full-suite runs ran B/A/B. An earlier
full baseline affected by a competing solve and a pytest-cache shutdown permission
error was excluded. Windows reported nominal clock states of 1,466–2,200 MHz; these
interleaved measurements are not controlled-power hardware baselines.

| Fast workload | Before | After |
|---|---:|---:|
| Full suite | 330.17 s, 1,292 passed | 214.33 s, 1,292 passed; 219.70 s, 1,302 passed |
| `test_weighted_column_ledger_closes` | 76.67 / 91.90 s | 1.606 / 1.451 s |
| `test_interface_voxel_is_not_spiked` | 195.24 / 171.28 s | 67.23 / 63.50 s |

The last full run includes all ten added cache tests. Named-test median paired speedups
are 55.5x and 2.80x, with zero node rebuilds in either cached test. These are standalone
test timings: the boundary test's earlier roughly 84-second *in-suite* profile could
reuse nodes from preceding tests; a standalone cold process builds all 914.

The production 900-history Dij's dose, column sigma and CSC structure were byte-equal
between uncached reference construction, a warm grid, and both cold grid strategies.
Warp CPU and CUDA each reproduced their own prior dose, sigma and CSC bytes when
reading a reference-generated grid; the CPU check extended its partial rectangle
before upload. No comparison requires equality across targets. The fast suite,
strict mypy, ruff, pre-commit and strict documentation build passed. Cutoffs, step
fractions, truncation and node construction are unchanged.

## Scoring and interfaces

**The dose grid is decoupled from the transport grid.** Any origin, spacing or shape,
including subregions, with **no coverage requirement in either direction**: a deposit
inside the CT but outside the dose grid goes to an *unscored* ledger bucket, never a
clamped edge voxel — clamping would corrupt edge dose. Per-voxel mass is rebinned by exact
separable voxel overlap, which is also what *defines* dose-to-medium for a mixed voxel.
Scoring a 2 mm CT on a 3 mm grid is ~0.30x the memory, the largest plan-calculation lever.

**`dose_to_water` is a scoring-output selection, not a physics toggle.** Transport, RNG
streams and the energy books are identical in both modes; only the per-deposit tally
weighting differs. It is refused in KERMA mode, where there is no tracked electron to
evaluate the stopping-power ratio on.

**Sources take engine-frame geometry only.** Gantry and couch rotation stay out of the
core — adapters map frames, for beam sources exactly as for the CT adapter.

**Collimation wrappers consume zero random numbers.** Deterministic Beer-Lambert
transmission weights preserve correlated Dij replay; a sampling implementation would not.

## Rejected

**Track-repeating / macro-MC: rejected on two independent grounds.** Its accuracy failure
modes — air, interfaces, bone, lung, penumbra — are precisely where Monte Carlo on CT
earns its keep, and both known fixes destroy the speed premise. Independently, the Dij is
not transport-bound enough for it to pay.

**Dij denoising: out of scope, permanently.** The optimizer exploits statistical noise; it
exploits denoising bias too, and that bias is spatially correlated and therefore worse.

**Quasi-Monte Carlo source sampling: measured neutral.** Built and unbiased, variance
reduction factor ≈ 1.0 over 48 seeds. Per-voxel Dij dose is transport-dominated counting,
which source-side QMC cannot help.

**A non-radial primary fluence envelope: measured negative.** A fixed-collimator term
C(x, y) on top of the radial fluence is the one extra degree of freedom the virtual-source
literature suggests when diagonals disagree, and the Halcyon beam data does show its
diagonal reading 0.2 to 1.0 per cent above the crossline at equal radius over r = 7-11 cm.
It is not a fluence structure. The differential is flat in measurement-plane coordinates
and grows with depth at fixed fan angle, where anything fixed in the collimator would
scale with divergence and dilute with depth; and the simulation, whose fluence is radial
by construction, already reproduces it (4.0 against 4.3 per cent at the strongest bin). It
is phantom scatter asymmetry near the crossline's field edge, transported for free in both
channels. The model-minus-measurement azimuthal residual is not coherent above its ~1 per
cent noise floor, so there is nothing to fit. Keep the diagnostic: plane-coordinate
flatness plus depth growth reads "scatter"; fan-coordinate constancy would read "fluence".

**Single-GPU micro-optimization is exhausted.** Persistent threads, register and occupancy
tuning, texture-memory reads and step-fraction changes all measured neutral or negative.
The kernel is divergence- and latency-bound. What remains is multi-GPU or algorithmic.

## Magnetic field (MR-linac dose), 2026-10-17

**The field is engine data, not a run option.** `MagneticField` goes to the engine
constructor next to the grid and the cross-sections, is uploaded once per device inside
the kernels' `GridInfo` struct (no kernel signature changed), and is recorded in the
result's provenance. AGENTS.md 2.10: there is no switch; the only way to transport without
a field is to not supply one. A gridded map must cover the transport grid — a zero fringe
inside the grid would be a silent bias, so it is refused rather than clamped.

**Each hinge chord follows the arc to second order: rotate by half the gyration angle,
drift, rotate by the other half**, with the gyroradius at the chord's mid-energy. The
chord of a circular arc lies along its midpoint direction, so this reproduces Bielajew's
(1988) first-order displacement exactly and is closer to the arc at second order (end
point within s theta^2/24), with the direction exact and the component along B conserved
by construction — the property EGSnrc enforces by hand in its renormalisation (Malkov &
Rogers 2016, Appendix A). No separate displacement term is added; it is already in the
chord. Multiple scattering and the field stay decoupled within a substep, as in every
production code.

**The gyration-angle cap is 0.2 rad per substep** (`LORENTZ_STEP_ANGLE_MAX`), the value
Malkov & Rogers validated against the Fano test for their higher-order scheme (stable to
0.4; the historical first-order scheme needed 0.02 at two to three times the cost). It is
gated here by the Fano test in a field and recorded as accuracy-defining. The voxel-face
cap is still taken along the pre-rotation direction, so under a field a chord can end up to
s theta/2 (tens of microns at 1 MeV, 1.5 T) off its straight-line prediction; the next
substep re-reads its voxel. Stated at the cap.

**The field plumbing alone preserves field-free results.** The
kernel branches once per substep on the field mode (the `msc_model_gs` precedent) and the
field-free path executes no new floating-point operation and draws no random number.
Measured interleaved on the RTX 4070 at healthy clocks: a uniform zero field or a zero
trilinear map against no field 0.98 / 1.01 (noise); a 1.5 T field in water 1.05. The
before/after dose bytes of a 6 MeV open field, an electron beam and a 2x2 Dij matched on
the reference engine, Warp CPU and Warp CUDA after every slice. `wp.static`
specialisation was kept as the fallback and was not needed. This measurement predates
the hinge-boundary and inverse-range fixes below, which also change field-free dose.
The standing bit-identity tests compare an explicit zero field with no field within
the current implementation; they do not assert equality with an earlier release.

**Air is the cost.** In air the energy-limited step is metres and the voxel face used to
bound the substep; under a field the gyration cap does, several times per voxel.
Measured on a water / air / water slab, 6 MeV, 10x10 cm: 1.20x wall at 0.35 T and 2.27x
at 1.5 T against the field-free run (Malkov & Rogers report +76 % with half the phantom
air). An exact-helix low-density path (chord substeps without multiple-scattering or
deposit work, the EGSnrc single-scattering-mode precedent) is the lever if this matters;
it needs the Fano-in-field gate and a realistic-geometry measurement, not a guess.

**Outside the transport grid there is no field.** The world beyond the grid stays vacuum,
a charged particle born outside flies straight to the entry face, and one that leaves does
not return. MR-linac setups extend the grid with real air voxels to the phase-space or head
exit plane (the commissioning example's air-gap pattern). The head pre-solve's straight
slide of contaminant electrons and the fringe field in the head are documented
limitations, not loop code.

## Aperture gradients, 2026-10-21

`pyradmc/geometry/aperture_gradient.py`, `examples/aperture_gradient_demo.py`. The
question: can a Monte Carlo gradient of the dose
with respect to a leaf tip or jaw edge, perhaps via Warp's autodiff, beat the bixel-Dij
chain rule that direct aperture optimization uses?

**Do not tape the transport.** Pathwise autodiff of the transport kernels is the wrong
tool on three independent grounds: discrete decisions (interaction channel, voxel index,
rejection loops) have zero derivative almost everywhere, so the gradient is silently
biased rather than noisy; the step loop is a data-dependent `while` loop and secondaries
move through atomic-append queues and reused buffers, none of which a tape can replay;
and the engine compiles with `enable_backward=False` on purpose (3x compile time). Under
attenuation-only collimation it is also unnecessary: an edge coordinate enters a history
only through its primary weight, the transport after the stack never sees it, and the
dose is linear in the weights, so `dD/dx = E[dT/dx K]`. Every gradient estimator is
then an ordinary `Source` with no engine change. For fluence or transmission-mask
parameters the Dij already *is* the Jacobian.

**The tape fits only the geometry, and there it is blind to edges.** Taping the scalar
collimation twins (`backends/warp/aperture_ad.py`) gives each ray's `d(chord)/dx` exactly
(matches the float64 host central difference). That pathwise derivative is the whole of
what autodiff of a fully taped run could return, and it fails twice:

- A straight face hit end-on makes `T` jump, so the derivative is a point mass on the
  edge line, which no sampled ray ever lies on: the tape returns **exactly zero** for a
  jaw under a parallel beam (chi2/ndf 658 against the oracle). This is the boundary-term
  failure known from differentiable rendering (Li et al. 2018, edge sampling).
- At a rounded tip the chord grows like `sqrt(delta)` from the apex, so the per-ray
  derivative goes like `1/sqrt(delta)` and its variance diverges logarithmically (cut
  off only by the float32 ray position). The mean is right: the derivative mass per
  bin across the tip matches the exact estimator to within 1.2 sigma, down to 0.1 um
  from the apex. The estimate is not usable: the top 0.1 % of rays carry 14 % of the sum,
  so a typical run comes out low and its batch sigma does not see what it missed.
  Independent pathwise runs agree with each other (chi2/ndf 1.06 and 1.16) and all
  disagree with the exact estimator (2.4 to 3.4). Efficiency is 1/1000 of it.

**Sampling rays from `|dT/dx|` is exact and cheap.** For a translated edge,
`T(q; x) = T0(q - x)` along the leaf-travel coordinate, so `|dT/dx|` is the total-variation
measure of `T0`, of finite mass `1 - T_min`, sampled by inverting the monotone profile
(`EdgeDerivativeSource`). A jump inverts onto the edge line, a rounded tip onto its
penumbra, and the weight is a constant times the transmission through the rest of the
stack. Against the common-random-number central difference (the oracle) on 1 mm voxels,
2 MeV, real tungsten, electron transport on: chi2/ndf 1.20 (tip, ndf 113) and 1.02 (jaw,
ndf 96), at 76x and 130x the oracle's efficiency. That oracle is already restricted to the
slice of field its weights live on; over the whole field it is about 8x worse again.

**The oracle needed its own care.** At a 0.5 mm finite-difference step the oracle was
biased: chi2/ndf 11.7 (jaw) and 12.4 (tip) against the exact estimator, falling to 1.55
and 1.63 at 0.25 mm and 1.00 and 0.69 at 0.1 mm, about as h^4 (a squared O(h^2) bias).
The integrated sums agreed at every step to within 0.4 %. The per-ray response varies within
a 1 mm voxel column, and a window of +-h averages over it. At low statistics the bias hid
in the noise (chi2/ndf 1.3 at a tenth of the histories).

**A Dij gives correct leaf gradients only with fine bixels *and* the right leaf
model.** The DAO chain rule (`BixelChainRuleSource`, Shepard et al. 2002) spreads the
derivative mass uniformly over the bixel holding the tip, as for a sharp leaf end; the
true mass of a rounded tip lies in its ~6 mm penumbra on the blocked side. The
profile-aware chain rule (`ProfileChainRuleSource`) keeps the same bixel Dij but takes
each bixel's `dw_k/dx` from the transmission drop across it, which converges to the exact
estimator as the pitch shrinks. Measured on the leaf gradient of the dose summed over a
structure whose edge lies a distance from the tip (tip at 5.15 cm, ratio to exact):

| chain rule | at 0 cm | at +0.5 cm | at +1 cm | voxel chi2/ndf vs oracle |
|---|---|---|---|---|
| DAO, 5 mm | 0.98 | 1.86 | 1.04 | 393 |
| DAO, 2.5 mm | 0.74 | 0.57 | 0.90 | 285 |
| DAO, 1 mm | 0.92 | 0.58 | 0.94 | 75 |
| profile, 5 mm | 0.98 | 2.16 | 1.15 | 400 |
| profile, 2.5 mm | 0.79 | 1.07 | 0.94 | 209 |
| profile, 1 mm | 0.97 | 1.00 | 1.03 | 8 |

At 5 mm the error is the bixel resolution and the leaf model does not matter: swept
through one bixel, the worst structure-gradient ratio is 0.24 to 4.8 for DAO and 1.6 to
4.9 for the profile rule. Below the penumbra width the leaf model takes over: the DAO
rule stalls at -43 % with the structure edge 5 mm past the tip however fine the bixels,
while the profile rule converges (within 3-6 % at 1 mm). At a straight jaw face there is
no penumbra and the two rules coincide, both converging with pitch (chi2/ndf 3.2 and 3.5
at 1 mm). An earlier version of this entry read the 5 mm result as a leaf-model
effect; the 1 mm runs show it was resolution.

So a Dij-based direct aperture optimization needs about 1 mm bixels along leaf travel
near the tips plus profile-derived bixel weights to get leaf gradients right in the
penumbra region. The edge-sampled estimator needs neither: its "columns" are the leaf
edges themselves, and every history it spends is on the gradient.

**Feasibility inside an optimizer loop (measured 2026-10-21).** An optimizer needs
not the Jacobian but `g_e = sum_v dL/dD_v dD_v/dx_e`, one number per edge. Measured on
400 edges (20-pair tungsten MLC, 10 step-and-shoot segments with MU weights, C-shaped
PTV around an OAR, 46k voxels of 2.5 x 2.5 x 5 mm, 2 MeV, electrons on, quadratic
objective at a mid-optimization iterate), all edges as beamlets of one `run_dij` and the
columns contracted with `dL/dD` afterwards, six replicas per point against a reference
at 2e5 histories per edge (0.15 % noise):

| histories / edge | error of the gradient vector | cosine to reference |
|---|---|---|
| 300 | 9 % | 0.998 |
| 1 000 | 8 % | 0.999 |
| 3 000 | 3.5 % | 0.9998 |
| 10 000 | 2.4 % | 0.9999 |
| 30 000 | 0.9 % | 1.0000 |

The error falls as 1/sqrt(N); correlated and independent sampling are equivalent (the
correlated mapping gives a slightly better direction at the lowest counts). So a few
hundred to a thousand histories per edge already point the gradient within a few
degrees, which is the stochastic-optimizer regime. A clinical arc of ~15 000 moving
edges then needs roughly 5e6 to 1.5e7 histories per gradient, a second or two of GPU
transport. The relative noise grows as the gradient shrinks near the optimum, so late
iterations will need more histories than early ones.

**The host is the bottleneck, not transport.** An open beam transports 2e6 histories
in 0.04 s on the RTX 4070 (healthy clocks), but weighting rays through the 20-pair MLC in
NumPy costs 5.7 us each: the forward dose of the 10 segments took 100 s, and the first
gradient reference run 470 s. Evaluating the rest-of-stack transmission on the edge's own
leaf pair only (exact under a parallel beam, pinned bit-identical) cut the reference run
to 40 s, of which host ray sampling is still 31 s. Below ~3 000 histories per edge a
fixed ~0.5 s per run dominates, which a profile later showed is not Dij assembly but
the host pre-sampling route seeding and draining the transport queues once per beamlet
(~1.4 ms per edge); scoring the contraction instead of the Dij changed nothing at this
grid size (0.58 s both, matched batch counts).

**A Monte Carlo direct aperture optimization runs at seconds per iteration**
(`examples/aperture_gradient_feasibility.py`, same problem): projected Adam on all 400
tips and the 10 MU weights, each iteration a fresh forward dose (3e5 histories per
segment) and a fresh gradient (1 000 per edge), 1.8 s per iteration on the RTX 4070
with host ray sampling still in the loop. With a `1/sqrt(k)` step the loss fell
monotonically from 1.61 to 0.34 in 25 iterations; at a constant 0.5 mm step it stalled
after ~10 and drifted back up, the stochastic-gradient noise floor. Building the 400
edge sources once per iteration costs 0.1 s, because the mu(E) tables and the inverse
edge profiles do not depend on the leaf positions and are shared (bit-identical to a
fresh build, pinned). The first run of this study also surfaced an engine bug: the Warp
engine chose its built-in generators for `PencilBeamSource`, `ParallelBeamSource` and
`BeamletGridSource` by `isinstance`, so a reweighting subclass silently transported as
the open field (fixed: exact type, as `SpectralBeamSource` already did).

**The fan beam and the GPU route (2026-10-21).** Under a divergent fan an edge move is
not a ray translation, so `FanEdgeDerivativeSource` importance-samples: rays along the
edge from a proposal tabulated in `s = sqrt(|q - q0|)`, each weighted by its exact
`dT/dx` from the chord edge-derivative twins (Hadamard: an endpoint on a surface moving
with velocity `V` moves by `(n.V)/(n.d)`), energies from the beam's own spectrum, the
focused-jaw jump on the edge line. Two fan-specific facts carried weight. The ray tangent
to a rounded tip crosses the mid-plane short of the apex, by `R alpha^2 / 2` (measured
0.095 mm at 2.6 cm off axis against 0.096 predicted), and that sliver holds about a
quarter of the derivative mass: a proposal anchored at the apex was caught as a -27 %
bias (z = -231); the support start is now the tangent in closed form. And near that
tangent the float32 circle test cancels (`|p - c|^2 - R^2 ~ 1e-5` against an ulp of
8e-6) and the kernel dropped exactly the rays carrying the largest weights, a 1-2 % bias
that source-level means over 2e5 rays flagged at 7 sigma (a transport-level chi-squared
over three edges had not); there the ray-circle gap factorizes without cancellation and
the chord is closed-form. Unbiased against quadrature for both tip sides, straight and
focused jaws; agrees with the CRN oracle through transport.

The device route (`backends/warp/aperture.py`) uploads a whole edge set — every
segment's stack in one device table, mu(E) rows per material, the per-edge geometry —
builds the proposals on the device and generates the rays in-kernel; the contraction
(`run_beamlet_contraction`, on both engines) scores `g_j = sum_v a_v D_vj` per column
and batch without a Dij, in a fixed reduction order. A 400-edge fan gradient (6 MV,
electrons on, 10 segments, 1 000 histories per edge) took 0.04 s against 2.67 s on the
host route; rebuilding the 400 edge sources after a leaf move fell from 3.94 s to 0.04 s
with the closed-form support start and device-built tables.

**A Monte Carlo direct aperture optimization on a 6 MV fan runs at 0.6 s per
iteration, fully on the GPU** (`examples/aperture_gradient_fan_dao.py`, RTX 4070 at
healthy clocks): per iteration a device head pre-solve in attenuation mode plus transport
for each of the 10 segments (4e5 histories each, one primary sample shared by the
segments: 0.52 s) and the 400-edge gradient (2 000 histories per edge: 0.09 s). The loss
fell from 1.65 to 0.26 in 40 iterations, about 25 s in all. The forward dose is now the
cost; the gradient no longer is.

Correlated sampling, the shipped Dij default, makes the columns share their
fluctuations, so a chi-squared across columns is inflated (AGENTS.md 8): one fan
comparison failed for exactly that reason before any kernel existed, and an earlier
contraction test had passed only by the luck of its seed. Cross-column comparisons
use the independent mapping.

Open, stated: the parallel-beam sources stay monoenergetic and parallel-only; the fan
estimator needs focused leaf sides, the focal spot at the stack frame's origin and the
aperture axes on the frame's axes (a gantry angle is the adapter's frame rotation, as
for every source). Leaf scatter (the first-Compton pre-solve) moves photons when the
edge moves and is not covered; the attenuation configuration is.

## Exterior air under a uniform field, 2026-10-30

**The problem.** At 1.5 T with 3 cm of air kept around a TG-119 plan, about 70 % of the wall
was field-in-air: the gyration cap binds on every air substep, about 31 per gyration, while
almost nothing physical happens there. The air cannot just be removed: without it the body's
outer shell is 1.8 % low at 1.5 T and 4.2 % low at 0.35 T, because electrons must be able to
curve back.

**What was built, and what is not decided.** `ExteriorAirMap`: the air face-connected to the
grid boundary, with an exact box-to-box safety distance per voxel. Where it is at least
0.5 cm a substep takes exact helix chords limited by that safety and the energy fraction,
because path bounds displacement. Data, not a flag (AGENTS.md 2.10): the map is derived
from the grid, sits next to the grid and the field, and `exterior_air=None` is the current
stepping. The default and the 0.25 cm minimum are recorded below.

**Default-on, by the maintainer, with the shift unexplained (2026-10-31).** The evaluation
below was at a 0.5 cm minimum and recommended waiting until the 0.35 T Fano shift was
explained and the plan itself measured. The maintainer decided otherwise: "keep it on and warn
about the shift, since there's basically no relevant dose change within the relevant regions of
the body", and "I'm okay for a shorter minimum. Performance here is critical." So under a
uniform non-zero field the engines derive the map themselves and warn once per engine, at
construction, so that an optimization loop is not flooded. The shift remains unexplained and
the TG-119 plan's own wall and shell dose remain unmeasured on this branch. Once-at-
construction also means an engine built and never run still warns.

**At the 0.25 cm minimum.** Rerun in full, same statistics. Wall, new over old, 100 M, best of 3,
2520 MHz: 0 T 1.01, 1.01, 0.99 at 0, 3, 6 cm (noise); 0 cm 0.99 and 0.91; 3 cm 0.64 (0.35 T),
0.48 (1.5 T, 6.01 to 2.89 s) and 0.30 across the body (10.0 to 2.98 s); 6 cm 0.45 and 0.38.
Absolute times in that session ran about 20 % above the 0.5 cm table at the same clocks, kernel
unchanged and unexplained, so compare ratios. Realistic: shell -0.039 +- 0.050, -0.045 +- 0.060,
+0.015 +- 0.066, -0.058 +- 0.050 %, body within 0.003 +- 0.018 %, block p 0.10 to 0.57; 6 cm and
noisy CT air the same within noise. Power rose: 4x inflated safety moves the shell -1.13 to
-7.1 % (0.5 cm: -0.46 to -6.7 %), block p at most 1e-210 (0.5 cm: about 1e-6); 1.5x is still
caught only at 0.35 T (-0.70 %). The wrong chord is still invisible in the realistic comparison
(shell at most +0.11 +- 0.07 %).
Fano, map against current: +0.13, +0.02, +0.18, -0.07 % (cavity) and +0.26, -0.03, +0.34, -0.07 %
(long-step voxels), against 1.5x (-1.5 / -1.6 %), 4x (-4.0 / -11.9 %) and the wrong chord (cavity
-7.5 to -24 %, long-step voxels -9 to -44 %). The shift: +0.23 +- 0.04 % and +0.39 +- 0.04 % over
three seeds at 1 MeV, 0.35 T; worst tested +0.57 +- 0.22 and +0.75 +- 0.22 % at 0.5 MeV, 0.35 T with
a 2 cm wall. Not larger than at 0.5 cm, and still not explained. The base Fano gates hold on the
new path at unchanged tolerance when a 2 cm air gap is declared exterior.

The rest of this section is the evaluation at the original 0.5 cm minimum.

**Who is in the air.** Counted on the reference loop (100k histories, 6 MV 10 x 10 cm AP,
3 cm margin, no kernel change): air is only 5 to 11 % of mean substeps. Of those at 1.5 T,
along the body axis, 62 % belong to electrons that never reach the body, 35 % to returning
ones and 4 % to ones that left for good; across the body 31, 17 and 51 %. At 0.35 T across
the body it is 9, 44 and 46 %. So at 1.5 T the cost is mostly *not* the electrons that had
to be kept. The longest single electron is the telling number: 3667 and 8630 substeps fall to
1371 and 2270, while the mean per history falls only from 7.26 to 7.08 and 7.74 to 7.16.
Wall follows the lanes that last, as expected of a width-bound kernel; this was not measured
in the kernel itself, so it is an inference from the reference counters and the wall.

**Wall.** RTX 4070 at 2520 MHz, interleaved, best of 3, one engine alive, 100 M histories,
tabulated body of 3 mm voxels, ratio new over old: 0 T 1.00 at every margin; 0 cm 0.99 and
0.95; 3 cm 0.64 at 0.35 T and 0.65 at 1.5 T along the body (5.89 to 3.80 s), 0.43 at 1.5 T
across it (10.0 to 4.26 s); 6 cm 0.49 and 0.50. At 10 M histories, best of 5, a minimum safety
of 0.25 cm gives 0.63, 0.49, 0.30 and 0.49, 0.40 at the same points, and 1.0 cm gives 0.81,
0.84, 0.67 and 0.61, 0.69. The 0.25 cm map passes the same Fano and realistic comparisons
(shell within 0.06 +- 0.05 %; Fano map against current at most +0.34 +- 0.12 %), without its
own mutant runs.

**Accuracy.** Tabulated, 6 MV, 8 seeds of 40 M per arm, new against old, shell the outer two
voxels: at 3 cm shell -0.039 +- 0.049, -0.046 +- 0.057, +0.005 +- 0.067 and -0.062 +- 0.051 %
(1.5 T along AP, across AP, along LAT, 0.35 T along AP), body within 0.003 +- 0.018 %, block
chi-squared p 0.11 to 0.59. A 6 cm margin and noisy CT air (log-normal, up to 8x density
contrast inside the region) are the same within noise. Power: a 4x inflated safety is caught
everywhere (shell -0.46 to -6.7 %, block p near 1e-6 or below); 1.5x only at 0.35 T
(-0.70 %).

**A wrong chord is invisible in body dose.** The small-step chord put in the long steps moves
the shell by at most +0.07 % at those statistics, so the realistic comparison gives it no
protection. The Fano cavity sees it at once (cavity -7 to -18 %, long-step voxels -10 to
-47 %), the fast twin check at 1.5 T, and the analytic orbit test directly.

**Fano.** Field-free the new cavity is within 0.25 %. Map against current stepping moves it by
+0.07, -0.09, +0.28, -0.14 % (cavity) at 0.5 / 1 MeV and 0.35 / 1.5 T, all inside the 1 % gate.
The +0.28 % is real: over three more seeds at 100 M, 1 MeV and 0.35 T, it is +0.25 +- 0.03 %
(cavity) and +0.47 +- 0.04 % (long-step voxels), and +0.54 / +0.82 % at 0.5 MeV with a 2 cm
wall, always toward unity. Not explained. The current stepping's own -1.3 to -1.4 % on this
cavity is in AGENTS.md section 8.

## Hazards learned the hard way

**Kernel compilation could leak into host modules.** Loading the Warp physics patched
`sys.modules`, and any module first imported inside that window bound the kernel-side
`uniform` permanently — silently breaking the *reference* backend for the rest of the
process. Fixed by evicting modules first imported in the window; regression-pinned in a
subprocess test, because an in-process test cannot reproduce the import ordering.

**Verify GPU clocks before believing any timing on the development laptop.** Its power
state has been observed drifting several-fold, and once ran at 210 MHz under full load —
faking a uniform multi-fold regression across every CUDA benchmark. The tell is CPU rows
getting *faster* while GPU rows get uniformly slower. Only interleaved-median ratios are
trustworthy there.

**Micro-optimizing kernel arithmetic measures zero.** Shared logarithm nodes and float
quantization changed nothing: nvcc already eliminates the redundancy. The bottleneck was
never arithmetic.

**A phase space must be normalized per original primary**, not per stored record. Caught
by a demo whose absolute dose was wrong, not by a unit test; now regression-pinned.

**The random hinge's second chord stops at a voxel face; the nudge across a face is a
free move charged by the next substep in the new voxel.** Found 2026-10-17 by the Fano
test in a field, at *zero* field: a water / air-density / water slab showed gap-to-wall
dose-per-mass of 5.1 and 2.9 for 0.1 and 0.2 cm gaps at 0.5 MeV, and a 2 cm gap at a
hundredfold contrast still +7.9 % (0.5 MeV, ECUT 0.05). Three mechanisms, all in the
boundary handling and all predating the field: (1) the voxel-face cap is taken along the
pre-hinge direction, so after the deflection the second chord could end in the
neighbouring voxel and file its share of the substep's loss — computed with the start
voxel's density — there, with range-out terminations landing there too; (2) the chords
split the full capped length *including* the nudge past the face, so on short final steps
the first chord overshot the face and the second ran inside the neighbour; (3) the nudge
itself was transported path charged at the old voxel's density, so an electron ranging
out in the last micron before an air face was carried into the air with a sliver above
ECUT and terminated there. The fix (2026-10-18, reference loop and Warp twin alike): a
chord that reaches a face stops on it, with its loss scaled to the shortened length and a
Møller event sampled beyond the face not reached (the flight is memoryless and re-sampled
in the next voxel) — PENELOPE's rule for an interface met on the second leg; under a
field the first chord is cut the same way, since it bends off the pre-rotation cap; the
chords split only the path inside the voxel; the particle is carried across by a
geometric nudge that the next substep charges at the new voxel's density (a ``carry``
length), or left just inside the face if it is terminal there; and a face-limited
substep whose second chord ends within a nudge of the face counts as reaching it, else a
rounding residue leaves the particle a zero-length step short of the face for ever (that
infinite loop was found the hard way, by a test hanging 15 hours). Measured on the
embedded Fano phantom at ECUT 0.05 (test instrument), 0.5 MeV, 0.2 cm gap, CUDA and Warp
CPU agreeing: **air contrast 2.9x -> 0.98 +- 0.003; hundredfold +7.9 % -> -0.9 %; tenth
-0.6 % -> +0.9 %**, all for a 0.2 cm gap, while a 2 cm gap is within 0.4 % at every
contrast; the thin gap's residual halves at a substep fraction of 0.03 (-1.1 / -0.9 %)
and is within 1 % at 0.02 (-0.2 / -0.6 % at 0.5 / 1 MeV; -0.8 / -0.4 % under 1.5 T,
which shortens its substeps), so the remaining -2 % at the shipped GS fraction of 0.20
(0.987 at 0.10) is step-size convergence of the hinge at an interface, not bookkeeping,
and the validation tier now gates all three statements (field moves the ratio < 1 %,
field-free residual < 1 % at a tenth and < 3 % to air, converged < 1.5 % at 0.02 with
and without the field). The shipped GS fraction stays 0.20 (AGENTS.md section 8: an
accuracy-defining change needs the realistic-geometry measurement); an earlier draft of
this entry called the baseline fraction 0.05, which is the Gaussian hinge's default,
not the one these runs used. (Building the gate taught a second lesson: a light slab
open to the phantom's lateral faces loses electrons sideways over their metre-scale
range with nothing replacing them — a 17-23 % field-free deficit at a hundredfold
contrast on an 8 cm wide phantom — so the slab is embedded laterally in water.) Dose
effect in the electron return effect's slab (6 MeV, 1.5 T): the air voxels adjacent to
water drop about threefold, the water side within 2 %; field-free water-only results
move at the 1e-4 level (deposited energy 1.7348e6 -> 1.7342e6 MeV on the 64^3 open field).

**A particle that has reached a voxel face is carried across along the face normal, and a
chord cut at a face turns by the cut chord's gyration angle.** Two corrections to the
chord-cut rule above, both found by its first perf A/B (2026-10-18). (1) The nudge across
a face was a micron along the *direction*; at grazing incidence its normal component is
below one float32 ulp of a coordinate near 10 cm, the position does not move, and a bent
chord cut at the face with no in-voxel path left is cut again next substep with
zero-length chords — the cut path has no hinge and no rotation to turn it away — so the
lane sits on the face charging one carry micron per iteration until its energy is gone.
The water / air slab at 1.5 T ran **7.6x slower on the GPU** and not at all slower in the
float64 reference loop (substeps per history 1615 with the cut, 1606 without). The nudge
is now a micron along the normal of the face the chord reached (``voxel_face_axis``,
``nudge_across_face``), always resolvable, the same micron geometrically; it applies to
every crossing, field or not. (2) Both half-rotations of a chord used the full chord's
gyration angle even when the chord was truncated at a face, so every cut over-turned the
direction without the matching path; the second half-turn is now the cut chord's whole
angle minus the first half-turn, so direction and path stay on one arc. Gated by the
Fano-in-field tier and the slab throughput in the perf tier.

**The Warp kernel takes both ends of a substep's continuous loss through the same
inverse-range table.** The reference loop evaluates ``E_end = r^-1(r(E) - rho s)`` on
one piecewise-linear range grid, whose forward and inverse interpolations are exact
inverses of each other. The kernel tabulates range-of-energy and energy-of-range on two
different log grids, and until 2026-10-18 computed ``de = e - E(R(e) - rho s)``: the
round trip ``E(R(e)) - e`` is a fixed error of order 1e-6 MeV per substep, invisible
against a 5 % step but comparable to the tens of micro-eV a gyration-capped substep in
air loses, and because the kernel clamps ``e_end <= e`` every substep whose loss fell
below the error lost *nothing*. Found by the Fano test in a field at air density, where
the gyration cap makes the substeps short: gap-to-wall dose per mass 0.984 at the shipped
cap of 0.2 rad and 0.855 at 0.05 (1 MeV, 1.5 T, 2 cm gap), 1.004 and 1.000 with
``de = E(R(e)) - E(R(e) - rho s)``; field-free the same gap moved 0.9947 -> 0.9995.
One extra table lookup per substep. The lesson generalises: any per-substep quantity
with a fixed absolute error is a bias that scales with the substep *count*, and the
field, by capping the substep in air, multiplies that count a thousandfold relative to
the loss.

**Flipping a validated default re-rolls every fixed-seed statistical test.** Expect around
a 1 % trip rate across the suite, and repair those by fixing the oracle, never by
loosening the tolerance.


## Head pre-solve setup cost, 2026-10-22

`presolve_head_device` was paying a fixed host cost on every call that has nothing to do
with the head physics: a scalar-query loop over the energy grid for every device
material (`_MuTables`, 512 nodes x total and Compton x each material, through
`TabulatedCrossSections.mu_over_rho`), a fresh flatten and 19 small `wp.array` uploads
of the stack, a re-upload of the log-mu tables, and an `astype(np.float32)` copy of
eight primary columns that `sample_batch` already emits as float32. An aperture
optimizer repeats the call hundreds of times with the same materials.

**Decision.** Memoise at the interface boundary, not by vectorising the lookup. The
curves are pure functions of `(source, material, grid)`, so `geometry/mucache.py` keeps
them per source object (weak reference, entry dies with the source, keyed by identity),
returning the very arrays the removed loops built, element for element. Density is not in
the key, so a retuned leaf density reuses the curve and only re-scales and re-logs.
`geometry.head.mu_tables` also memoises the assembled `_MuTables` (bounded at 32 per
source), and the Warp pre-solve keeps the device upload of the tables on that object and
of the stack per stack *object* (identity rather than value keys because `-0.0 == 0.0`; the
object is frozen and `MLC` now coerces its leaf arrays to tuples of floats and the stack its
devices to a tuple, so neither can be mutated in place under the cache; a test pins both
the key and the lock-free weakref eviction, which deadlocked under cyclic GC when it took
the cache lock). The same curve memo serves `_StackTransmission`. Primary columns
go up through `np.ascontiguousarray(..., float32)`, a no-op when they already are.
Vectorised array-valued lookup was rejected: `np.log` and `math.log` may differ in the
last bit, so it could not promise the bit-identity the memo gives for free, and the cold
build is paid once per source.

**Measured** (2026-10-22, RTX 4070 laptop shared with three other agents, so GPU rows
are contended; best-of-N, old and new alternated in separate processes, 3 rounds, all
within 5 % between rounds). 2-tungsten-device stack with a 60-pair MLC and an air-free
attenuation pre-solve:

| quantity | before | after |
|---|---|---|
| `_MuTables` build, tabulated, 3 rows x 512 | 4.6 ms | 0.058 ms warm (cold unchanged) |
| whole pre-solve at 2000 histories (setup-dominated) | 5.1 ms | 1.05 ms |
| pre-solve at 400k histories | 101 ms | 98 ms (primary sampling is 84 ms of it) |
| `BeamFrame` construction | 40 us | 27 us |
| `MLC` (60 pairs) / `BeamLimitingStack` | 10 us / 1 us | unchanged |

A collimator object costs 10 us (`MLC`, 60 pairs) and 1 us (`BeamLimitingStack`) here, already
far under the 0.2 ms target, so their validators were left as they were (a vectorised
rewrite measured no gain and was not exactly equivalent on contrived inputs). Only
`BeamFrame` changed: `np.allclose` written out as the same inequality, identical
decisions. `np.linalg.norm` in `_as_unit` was left alone because replacing it can move the
last bit of every axis.

**Bit identity** was checked on saved outputs of the unchanged code: the three log-mu
tables (analytic water; EPICS tungsten + air) and the seeded pre-solve exit phase space
in attenuation and first-Compton mode on cpu and cuda separately (records sorted, since
the atomic append order is not deterministic) are `np.array_equal` before and after.
Pinned in `tests/unit/test_presolve_setup.py` (against the scalar loops reproduced as
oracles) and `tests/physics/test_warp_presolve_setup.py` (warm call equals cold call).
Not changed: `StackArrays`/`upload_stack` layout, the engine, any kernel, any default.
The memo assumes a `CrossSectionSource` is immutable after construction, as every
shipped source is (stated on the interface; `mucache.clear()` is the escape hatch).
Bit-identity holds for Python-float densities. A numpy `float32` density used to build
float32 tables under NEP 50 scalar rules; densities are now coerced with `float()` where the
pre-solve builds its table key, so such inputs get float64 tables (a deliberate behaviour
change, pinned in `tests/physics/test_warp_presolve_setup.py`).

## Resident Warp device state, 2026-10-22

**The Warp engine keeps its device state between calls; a plan-optimization loop was
paying setup, not transport.** Profiling ten consecutive `run` calls (48x40x24 water,
400k histories each through a mostly closed 6 MV transmission mask, RTX 4070 Laptop)
showed per call about 5 ms re-uploading the cross-section tables — mostly `astype` and
copy of the 12 MB float32 Goudsmit-Saunderson grid, although the host build was already
cached — about 2 ms allocating and zeroing four queues of twelve columns plus RNG slots
and counters, and a `synchronize_device` plus three or four synchronizing readbacks per
batch. Transport kernels were a small fraction of the wall.

Now kept per engine and device: the tables per `(ecut, pcut, e_max, with_gs)` (the GS
grid shared across `pcut`; the four most recently used keys, so an ECUT sweep does not
park a 12 MB grid per cut), the transport grid and field, the last scoring geometry and
up to eight spectra, and a pool of transport workspaces — queues, slots, the counters in
one array, lane streams — shared by `run` and `run_dij`. Two kinds of key:

- **Engine fields** (grid, field, cross-section source) are frozen dataclass fields, so
  a different one is a different engine; in-place mutation of their arrays is
  unsupported, as it already was for the host table cache.
- **Per-call arguments** (the scoring grid, the spectrum) are matched by identity with
  a strong reference held. That is only sound because their arrays are read-only:
  `Spectrum` already was, and `ScoringGrid` / `CylindricalScoringGrid` now copy
  `voxel_mass` and mark it read-only (a review probe showed `sg.voxel_mass *= 2` between
  two calls silently doubling the dose of the second).

**Memory policy: the pool holds what the last call needed, never the largest so far.**
Every call refits its workspaces exactly, growing or shrinking. A steady workload — an
optimizer's loop of equal calls — reuses them untouched; an alternating big/small
workload reallocates on each change, as every call did before. A first version kept the
largest-ever size: it parked 864 MiB behind tiny calls after one 300k run (1760 MiB
after a two-lane run) and let a Dij's dense group maps ride on a big run's queues (peak
1325 -> 2054 MB, 64x64x48 water, 256 beamlets). With the exact refit, peak memory equals
the pre-change engine in every sequence measured (run, dij, run+dij, dij+run, run+run).
What stays held between calls is the tables and voxel maps plus the last call's queues
(about 0.9 GB after a full-width `run` on this card, per lane);
`WarpEngine.release_device_memory()` hands it back. The Dij auto group size counts idle
pooled bytes as free (each is released or reused by the shard), so it is the size a
fresh engine gets — pinned with a simulated device in the test file below.

**Dense dose maps stay per call** — not to spare memory in general (the pool holds the
last call's queues, which are of the same order), but because parking them buys little
and costs a lot: a `run`'s five voxel-sized maps took about 0.1 ms to allocate and zero
on the 48x40x24 grid (profiled, ~25 us each) against ~2 ms for the queues, while holding
them would add 48 bytes per scoring voxel between calls (2.5 GB on a 512x512x200 CT). A
Dij's group maps are sized per call from free memory and stay per call for that reason.

**Books read in one transfer, majorant checked in the same readback.** The int64
counters share one array (one 128-byte line each) and are still cleared and read per
batch (`run`) or per group (Dij), so no int64 sum spans more than before. The kernels'
int32 majorant-violation counter is aliased onto the low half of a zeroed int64 slot of
that array (little-endian on every Warp target; `kernels.py` untouched), so the per-pass
readback raises in the pass that made the violation at no extra transfer. The explicit
`synchronize_device` in front of default-stream readbacks is gone (kept after
concurrent lanes).

Pinned in `tests/integration/test_warp_resident_session.py`: every call on a warmed
engine equals the same call on a fresh one (run routes incl. pre-sampled, dose-to-water,
KERMA, scoring grid, Dij incl. `devices=[...]`, lanes on cuda, a field engine); calls
after an aborted one (a source failing mid-run, a queue overflow) equal a fresh engine's,
and an overflow on a pool fitted to a larger call raises the same error as on a fresh
engine; the pool's size after each call; the bounded table cache; the early majorant
raise. Pre-change golden arrays matched exactly on cpu and cuda:0 after every commit.

Measured, interleaved A/B against the pre-change code in separate processes on a GPU
shared with other jobs (clock checked under load at 2.27 GHz, no power cap); ranges are
the best-of values over three alternations (cuda) or two (cpu), best of 3 loops, 30 tiny
runs, 15 tiny Dij each:

| | cuda before | cuda after | cpu before | cpu after |
|---|---|---|---|---|
| ten collimated runs (400k histories each on cuda, 40k on cpu) | 111-120 ms | 20-22 ms | 395-423 ms | 72 ms |
| tiny `run` (64 histories) | 7.4-8.7 ms | 0.88-0.92 ms | 7.4-7.7 ms | 0.94-0.98 ms |
| tiny `run_dij` (16 beamlets x 16) | 8.4-8.7 ms | 1.62-1.69 ms | 23-24 ms | 15.5-16.6 ms |

Recorded in `tests/perf/test_warp_setup_overhead.py`. What remains of a tiny call is
mostly the drain loop's per-round count readbacks and launch overhead, which this change
left alone.

## Warp drain schedule, 2026-10-22

**The queue drain launches blind rounds and reads the counts back once per block.**
The drain loop used to read every queue count to the host before launching on it — 8-10
synchronizations per drain on a 6 MV-class chunk, which needs 3-5 rounds (primary photons,
their electrons, then bremsstrahlung and delta generations of a few hundred and a few
dozen particles). Rounds are now launched at queue capacity, the transport kernels guard
`tid` against the device-side count, a one-thread kernel retires each consumed queue
(booking overflow and the last busy round on the device), and one status readback per
block of `k` rounds decides whether another block is needed. `k` adapts to the round
count of the previous drain on the device, so a steady workload drains with one readback.
`WarpEngine(drain_rounds=0)` keeps the host-sized schedule, and cpu uses it by default:
there a readback is a memory read and a capacity-wide launch loops over every slot.
**Bit-identical by construction and verified**: a run is a function of the multiset of
particles each launch consumes, and an empty launch consumes nothing; 94 saved pre-change
outputs (every host- and in-kernel-generated run route, Dij routes, run and Dij lanes, a
1.5 T field) matched bit for bit on cpu and cuda. The device phase-space route was checked
in process only — one buffer transported under each schedule — because the device
pre-solve that fills the buffer is itself order-nondeterministic.

**The gain is 3-7 %, not the readback count.** Interleaved in-process A/B, 16 pairs, on
the shared 4070 (clocks checked at 2535 MHz, other jobs at ~60 % utilization): ten
48x40x24 collimated 4e5-history runs 0.934x wall, the 64^3 open field 0.968x. Most of the
time inside the old readbacks was waiting for transport that has to happen anyway, as the
2026-07 replay measurement found. On the collimated optimizer-loop workload the transport
kernels are only ~1.3 of ~15-18 ms per call (Warp kernel timing); the rest is per-run
setup — memsets of freshly allocated queues (~4.1 ms), table and grid uploads (~1.4 ms) —
and the per-batch book readbacks and `synchronize_device` in `run()`. Those, not the
drain, are the next levers for small runs.

**A blind launch's grid is one chunk wide, not the capacity and not one device-full.**
Warp's grid-stride loop walks the empty capacity behind the count guard. A capacity-wide
grid (16x the blocks) cost ~2-3 % against the chunk-wide one; grids of 2-16 blocks per SM
were 1-8 % slower than chunk-wide, because the first round's work then runs serially per
thread — the persistent-threads result again.

**CUDA graph capture of a block: not built, bounded at near zero.** Per drain the host
enqueues a whole block in 0.49 ms (collimated) / 0.67 ms (open) while the block's GPU
work takes 1.5 / 2.3 ms more after that, so the GPU is never starved inside a drain;
replaying a captured graph could save only the first launch's issue latency. Against it:
at the time the queues were reallocated per run (so a capture per run; the resident
workspaces since remove that objection), the round index is baked into the retire
launches, and cpu has no graphs. Revisit only if a workload's per-drain GPU time falls
below its host enqueue time. The drain's status array and pinned readback element live
in the pooled lane workspace, made on first use and kept across refits.

**Review fix (integration):** the sequential `run()` routes now pass the lane's pooled workspace to every chunk transport; before, `_drain_queues` built a throwaway `_Workspace` (fresh status storage) per chunk on that path. A test counts workspace constructions during a warmed sequential run and requires zero.

## Segmented fans in one launch, 2026-10-22

**A plan's segments are the beamlets of one source.** An aperture-optimization loop
evaluated 10 segments of a 6 MV fan per iteration as 10 attenuation-mode device
pre-solves plus 10 `run()` calls: 20 launches with their own buffers, drains and syncs,
GPU utilization 0-35 %. `SegmentedFanSource` (`geometry/segments.py`) makes segment `s`
beamlet `s`: a history samples the `SpectralBeamSource` fan and carries
`exp(-sum_j mu_j(E) L_{s,j})`, the stack transmission at segment `s`'s settings. That is
the pre-solve's `mode="attenuation"` **with `air=None`**, and `CollimatedSource`: no head
scatter, no leaf-scattered photons, no contaminant electrons, and **no air attenuation**
(the pre-solve's optional `AirColumn` has no counterpart — a stated approximation of
this source). One bookkeeping difference: the pre-solve drops primaries below `PCUT`
or of zero weight when it slides them to its exit plane, while here every fan photon
reaches transport, which ends a sub-cutoff photon at once; the routes agree for spectra
above `PCUT` (every shipped MV spectrum; the Ali-Rogers floor is 0.15 MeV). `run_dij`
returns one column per segment. `plan(weights)` is a `Source` whose histories pick
segment `s` with probability `w_s / W` and carry `exp(-tau_s)`, so one `run()` returns
`sum_s (w_s / W) column_s` per history and the caller scales by `total_weight` `W` —
no Dij, no dense per-segment maps, no truncation. The weight is normalized on purpose:
the emitted and deposited books are int64 fixed point (1e-9 MeV quantum, ceiling about
9.2e9 MeV), and a primary weighted by `W` of hundreds of MU wraps them silently after
about 1e7 histories.

**Layout: one stack, per-segment columns.** Segments share the fan, the frame and the
stack structure; they differ in jaw edges and leaf tips only. The device keeps the
pre-solve's `StackArrays` unchanged plus a `SegmentArrays` of edges indexed
`segment * n_devices + device` and tips `segment * n_tips + tip`, so the chord is the
existing `jaw_path_length`/`mlc_path_length` twin at that segment's settings and the
attenuation is the pre-solve's `MuArrays` table built from the same host `_MuTables`. The
constructor takes the settings as arrays over a template stack, so an optimizer builds no
`MLC` per segment (`with_settings`, 0.1 ms for 10 x 20 pairs, also reuses the attenuation
table and keeps this source's settings for anything not given; ten validated `MLC`
stacks cost 1.0 ms); the arrays are stored read-only so the lazily built host stacks
cannot drift from a device upload. Routing is exact-type: `SegmentedFanSource` with an
exact `SpectralBeamSource` fan (and `SegmentedPlanSource` over both) generate in-kernel;
any subclass is host pre-sampled.

**One fan sampler for every generator.** The spectral fan's CDF inversion and ray were
written out in each of three kernels, and this source would have added two more copies
plus a discrete segment selection — sampling code in `backends/`. They now live once in
`pyradmc/geometry/fan.py` as single-source scalar functions (`fan_primary`,
`cdf_lower_bound`, `select_by_cdf`, compiled through `_KERNEL_MODULES`, runnable on the
host and test-pinned there against the fan's own `emit`). The existing generators'
queue output (energy, position, direction, weight, tag, RNG slot, emitted book) was
saved before the refactor and is bit-identical after it on cpu and cuda; the segments
module compiles with the transport kernels' options (`fast_math`), so a segmented
history's energy and direction are bit-identical to `generate_spectral_beamlets`' for
the same `(seed, key)` on one device (test-pinned).

**Statistical evidence, and why the first version of it was empty.** The equivalence
tests as first written compared 1 cm voxels at 2.4k histories per column in KERMA mode.
There, most masked voxels hold about one deposit per batch, so each voxel's z^2 is
bounded near or below 1 while the batched null mean is 1.2-1.4: the statistic sits below
its null *regardless of bias*, and the tests had no power. An independent review showed
it: a column permutation, a missing MLC, a missing jaw, a mean-energy attenuation and
mirrored tips all passed at p = 1.0 (chi-squared/null about 0.5 for every one, the same
as the unbiased control). The tests now score on coarse profile bins through the engines
themselves (1 cm along the leaf travel, 4 cm across, two depth halves: true batch sigma,
tens of events per bin per batch) and mask on the *pooled* dose, not on one side's
upward fluctuations. Calibrated: 60 unbiased seed pairs give KS p 0.68 / 0.29 / 0.50 per
column. Power, reviewer's host-route mutants at 12k histories per column, independent:
column permutation, missing jaw, missing MLC, mirrored tips and bin-low-edge energies
caught; the mean-energy mutant at p = 0.011 (caught at 40k, correlated, together with all
the others). Swapping the two devices' attenuation rows is invisible by construction (both
are the same water). In-kernel mutants — stream key from the group-local index, tag
always 0, weight from the wrong segment, every segment reading segment 0's tips or jaws,
plan weight doubled, plan always segment 0 — are each caught by at least two tests, the
deterministic generator pins among them (queue read back: bit-equality with the spectral
generator, tag `t // per_batch` at `group_start > 0`, weight equal to the host
transmission within the float32 chord tolerance of `test_warp_compiled_stack`, plan
segment from the replayed first uniform) and a correlation pin (identical segments give
bit-identical columns when correlated and different ones when not).

**A fixed-seed re-roll, not a tolerance change.** With the strengthened oracle the
reference comparison at `SEED` lands at p = 0.006 on segment 1. It is a fluctuation:
10 further seed pairs give uniform p-values (KS p 0.18); the Warp in-kernel and host
routes agree at 400k histories per column; and over 12 seed pairs on segment 1's stack
the existing `CollimatedBeamletSource` shows the same ref-vs-Warp low-p rate as this
source (8 % below 0.05 each, one reference seed driving every comparison it enters
low). The test runs at `SEED + 1`. Warp cpu and cuda draw the same counter-based
streams, so their columns differ only by float rounding (about 1e-5 relative): the two
device parametrizations of a test are one statistical trial, not two.

**The device pre-solve route cannot be a fixed-seed oracle on cuda, and a pre-solved
phase space hides variance.** After the integration merge, the comparison against
`presolve_head_device` + `run()` failed about 2 cuda executions in 12. Two causes, neither
a bias. (1) The device pre-solve appends its exit phase space with atomics, so on cuda
the same seed yields the same population in a different order (weight sum identical to
the last digit) and the run, which resamples records by index, a different dose: the
fixed-seed p-value of one segment came out 0.38, 0.0000 and 0.40 on three identical runs
— the seed is not load-bearing there. (2) A run that resamples an `n`-record phase space
with replacement reports a batch sigma blind to that phase space's own sampling error
(the latent variance `geometry.phasespace` already warns about). With records = run
histories (40k), 100 reseeded pairs per segment on cuda gave 10 of 300 p-values below
0.01 and 28 below 0.05 (3 and 15 expected), against 1 and 10 for a Dij-vs-Dij null on
the same oracle; yet the signed masked-dose difference was 0.0 +- 0.1, +0.1 +- 0.15 and
+0.3 +- 0.16 % and no profile bin showed a consistent offset (max per-bin |t| 2.6 / 1.9 /
2.1, null 2.6 / 2.6 / 2.3). The test now pre-solves on the host (`presolve_head`, the
device route's oracle, bit-reproducible: the same p-values on three runs per device)
with 20x the run's histories in records: 5 of 300 below 0.01 and 14 below 0.05, signed
differences +0.06 +- 0.11, -0.06 +- 0.12 and -0.02 +- 0.16 %. The pre-merge branch
(bce9016) gives bit-identical numbers in every one of these sweeps: the merged engine
changed nothing here. The batched oracle's p-values are skewed toward 1 in their bulk
(KS p = 0.003 on the Dij-vs-Dij null), so a KS test against uniform is not the yardstick
for it; the low-tail counts are.

**Measured.** RTX 4070 laptop at 2535 MHz, no throttle reason, the GPU shared with other
jobs, interleaved best of 7, the DAO workload (`tests/perf/test_segmented_fan_throughput.py`):
ten pre-solve + runs 1085 ms; the same sharing one host fan sample (the DAO example's
trick) 469 ms; one `run_dij` 78 ms (**6.0x**); one plan `run` 67 ms (**7.0x**). A single
400k-history `run()` of a device phase space alone costs 39 ms, so the gain is the per-run
fixed cost and the nine extra drains, not the generator.

**Not done.** History allocation in the Dij route stays equal per segment (the `run_dij`
contract); allocation in proportion to MU exists only in the plan view, and is
multinomial, because `emit` never sees the history index. Stratified allocation needs
either a per-column history count in `run_dij` on both engines or an index-aware emit.

**Review fix (integration):** the plan view's host batch (`SegmentedPlanSource.sample_batch`) is built from the segment source's `sample_beamlet_batch`, one batch per distinct segment, not from the bare fan. A segment subclass that reweights its primaries now reaches the Warp host route exactly as it reaches the reference engine through `emit` (reproduced before the fix: the reference halved, Warp did not). Voxel doses are not exactly proportional under reweighting because the photon roulette weight cap changes a few decisions; the test pins the emitted energy exactly and the total dose loosely.

## Vectorized edge sets, 2026-10-22

**An aperture gradient's edges are built from the segments' arrays, not one object per
edge.** The DAO example built 400 `FanEdgeDerivativeSource` objects per iteration
(validation, isolated and retracted stacks and their transmission models) in about
80 ms, then spent another 12 ms in `upload_edge_set` on per-edge records, a scalar
mu-curve loop over 512 energies and 53 small uploads. That is against a 0.1 s
iteration, and an arc's ~15 000 edges would have cost seconds. `SegmentedEdgeSource`
(`geometry/segment_edges.py`) lists every tip and jaw edge of a `SegmentedFanSource`:
segment-major, then device in stack order (or a given selection), then leaf pair,
then side (`neg`, `pos`). It builds the device columns with array arithmetic. The
position-free part comes from the per-edge source's own helper (`_fan_edge_geometry`,
now shared): mid-plane field, edge extent, `L_r / A`, span, `s` grid and isolated
slab, evaluated once per (device, pair, side) slot. It is tiled over the segments and
memoised per (fan, template, selection, segment count), all of which `with_settings`
keeps. The positions come straight from the settings arrays, and `q0` from vectorized
closed forms. On a host route, beamlet `k` builds the equivalent per-edge source on
first use and delegates to it (the oracle). The Warp engine generates the exact type
over an exact `SegmentedFanSource` in-kernel through the existing generator; any
subclass of either takes the host route (test-pinned with a reweighting subclass). The
fan's type is not checked: an edge source reads only the fan's spectrum and geometry,
on both routes.

**One contract, two builders, pinned equal.** `FanEdgeColumns` is the plain-data form
of an edge set (float64 and int32, before the device cast). `fan_edge_columns(records)`
builds it from per-edge records, with the old `upload_edge_set` body's arithmetic
unchanged; `SegmentedEdgeSource.device_columns()` builds it vectorized. The fixture has
focused and straight jaws on both axes; open, nearly closed (1e-6 cm) and closed pairs;
tips on the field limit; and tips whose support start needs the bisection. On it every
column is `np.array_equal` in float64, every uploaded device array (the device-built
proposal tables included) is equal, and `run_beamlet_contraction` (values, sigma,
energy books) is bit-identical, on cpu and on cuda separately, correlated and
independent. Two numerical points carried weight:

- `np.hypot` and `math.hypot` differ by one ulp on 18 % of inputs (measured). The
  vectorized tangent-point slab test therefore re-decides every point within 1e-12
  (relative) of a slab face with the scalar definition. A test that places slab faces
  exactly on the tangent point fails without that guard. `q0` itself uses only `sqrt`
  and arithmetic, both correctly rounded, so it is bit-equal without any guard.
- The bisection fallback is vectorized over the edges that need it, using
  `MLC.path_lengths`' interval primitives in its expression order, so every halving
  takes the per-edge loop's branch.

**Upload.** Every column goes up in one float32 and one int32 transfer, sliced into
the `EdgeSet` fields: a small cuda upload costs ~50 us, a view ~7 us. The mu rows come
from the per-source curve memo (`geometry.mucache`: the same scalar queries, element
for element). A new `s_row` index lets edges share `s` grids, one per slot instead of
one per edge, which saves 30 MB of device memory at 15 000 edges; the per-edge route
keeps `s_row == table`. The proposal tabulation can run its 513 chord evaluations per
edge in a node-parallel kernel before the unchanged per-edge sums. That pays only
while the one-thread-per-edge kernel starves the card (interleaved, same edge set):
400 edges 1.53 -> 0.69 ms, 4 000 edges 1.23 -> 0.83 ms, even at 8 000, and worse at
15 000 (2.29 -> 2.95 ms). So it is taken below 128 edges per SM, on cuda only, and
both forms give the same bits (test-pinned). Before any change, a snapshot was taken
of the per-edge route's uploaded arrays and contractions (256 arrays, cpu and cuda);
it was bit-identical after every step.

**Found and fixed: the in-kernel near-tangent chord was wrong outside its regime.**
`_block_terms` took its closed-form near-tangent chord whenever `s^2 < 2e-3`. That
chord is exact only where `q0` is the tangent ray and the bare disc chord lies inside
the slab. A support start found by bisection is not a tangent ray (it is a ray
clipping a slab corner), and a tangent point close to a slab face has its disc chord
cut by the face. On the fixture (2 cm slab, `R` = 20 cm, off-axis field, 16 000
histories per edge) the host and device routes disagreed on every bisected edge
(+15 to +30 %, z up to 14) and on a closed-form tip whose tangent point is 2.5 mm
inside a face (z = -7.4). Per ray in the near-tangent band, the chord came out 4-170x
too long on the bisected edge, and the chord and rate up to 1.9x and 2x too large
near the face. The per-edge device route had the same defect (the two routes are
bit-identical). The DAO example's geometry (6 cm slab, `R` = 8 cm, tips within
2.7 cm of the axis) was clear of it.

The fix brings the device route back to its host oracle without giving up the
branch's purpose, which is to avoid the float32 cancellation of the twin's circle test
at a genuine tangent:

- **The tangent ray, always.** Every rounded tip carries its closed-form tangent ray
  `q_t` (defined whether or not the tangent point lies in the slab) and
  `t_off = sign (q0 - q_t)`, both from the host in float64. `t_off` is zero when `q0`
  is the tangent ray and is a bisected start's lead otherwise (`FanEdgeRecord.q_tangent`,
  from both builders).
- **The branch condition.** The closed form is taken on a ray's distance from the
  tangent ray, `s_t^2 = s^2 + t_off < 2e-3`, with `q_t` in the gap formula. For a
  tangent start this is the old `s^2 < 2e-3` and the old arithmetic, bit for bit.
- **Inside the slab.** Where the bare disc chord lies inside the slab, checked per ray
  from the tip centre's projection `t_c` on the ray, the closed form is unchanged.
- **Cut by a face.** Where a face cuts the chord, it is clipped there in closed form.
  A clipped end does not move with the edge; an open one moves at `cu +- dhalf/dx`.
  Both faces may cut it, and a chord wholly outside the slab gives zero.
- **The twin only away from the tangent ray.** Bisected starts far from it
  (`t_off >= 2e-3`) and every ray beyond the band take the twin.

A first version gated on a host flag ("`q0` is the tangent ray") and sent every
bisected start to the twin. An independent review found the gap in it (2026-10-23):
a tangent point just *outside* a face makes the bisected corner ray almost tangent,
and the twin cancels exactly as at a tangent start. Integrated derivative mass,
device against host: 0.986 (cpu) / 1.031 (cuda) at 1e-4 cm outside, about 22 000 of
40 000 rays dropped on cpu, and a gradient jump as the flag flipped over ~0.08 mm of
leaf travel. Gating on the distance from `q_t` closes it: across a sweep from 1e-3 cm
inside to 1e-1 cm outside a face, and on every fixture edge, the mass ratio is within
2.5e-4 on both devices.

After the fix all 36 resolved edges agree with the host route: max |z| 2.69, on a
straight-jaw edge the fix does not touch (2.69 before too), and mean z^2 1.49. The
edges that were biased, before -> after (16 000 histories per edge, which cannot
resolve the review finding's 0.2-3 % mass effects; the quadrature test below does):

| edge | z before | z after |
|---|---|---|
| bisected, x = 3.3 | +13.97 | +0.95 |
| bisected, x = 3.0 (closed pair) | -7.92 / +8.74 | -0.69 / -1.42 |
| bisected, x = 3.1 / 3.9 | -8.68 / +9.67 | +0.67 / -2.42 |
| bisected, x = 2.7 (closed pair) | -6.24 / +7.45 | +1.09 / +0.62 |
| near-face tangent, x = 2.5 (closed pair) | -7.38 / +3.00 | -0.62 / -1.03 |

The former xfail now passes, on cpu and cuda. The per-ray tests run on cpu and cuda
against the float64 host chord and rate in the band `s^2` in [1e-4, 2e-3), each ray
classified as the kernel classifies it:

- **Closed-form rays** (whole or clipped): within the repo's float32 chord tolerance of
  2e-3. Measured up to ~1e-3 in rate across the reviewer's geometries (4.9e-4 on the
  near-face fixture edge).
- **Twin rays:** within a bound derived from the twin's arithmetic. Its ray-circle
  discriminant `D = 4 h^2` carries a rounding error of `k eps b^2` (`b ~ 2 t_c`), so
  the rate carries `k eps t_c^2 / (2 h^2)`. A first-order worst-case count of the
  roundings in `b^2 - 4ac` gives `k ~ 6`; the test uses 8. Measured at most 2.7; the
  pre-fix error is at least 840x the bound.

The same model predicts up to 1.9e-2 for the twin on the near-face tangent. That is
the cancellation the closed form avoids.

A deterministic quadrature test integrates `mu exp(-mu l) |dl/dx|` over the proposal
range, kernel against float64 host on the same nodes. It runs on every smooth fixture
edge and the face sweep, on cpu and cuda, and accepts a ratio within 1e-3 of 1:

| gate | worst mass ratio | verdict |
|---|---|---|
| fixed | within 2.5e-4 of 1 | passes |
| tangent-flag gate | 0.9964 (fixture), 0.986-1.031 (sweep) | fails |
| original `s^2`-only gate | up to 1.34 | fails |

Every snapshot stayed bit-identical, so no edge where the closed form held moved.
That covers the 256-array snapshot (proposal tables and contractions, cpu and cuda)
and the DAO example's own geometry (tungsten, 400 edges, proposal tables and
contractions, both sampling modes, cpu and cuda), against both the pre-fix code and
the tangent-flag gate.

The same review found a cache bug, fixed with a test. The layout memo of
`SegmentedEdgeSource` is keyed on fan, template, selection and segment count, but it
also stored the cross-section source. A second segmented fan sharing those but with
other cross sections would have uploaded the first one's mu rows. The cross sections
now come from each source's own segments.

**The reference check and its power.** The reference engine was run against the
in-kernel route in the DAO regime: two segments of a focused jaw over a 6 cm two-pair
MLC, 12 edges, KERMA, independent sampling, 8 batches. The reference ran 1 000
histories per edge (it costs ~1.2 ms per history on this source) and Warp 16 000.
Each compared value is an edge's whole contraction, hundreds of histories' deposits
per batch, so the batched chi-squared sits at its null here rather than below it as
on sparse voxels: over 10 seed pairs, 0 rejections at alpha = 0.01, median p 0.64.
In-kernel mutants on the uploaded columns over the same pairs:

| mutant | caught at alpha = 0.01 |
|---|---|
| one edge's sign flipped | 10/10 (chi2 870-2 400 against a null of 16) |
| one edge +10 % | 0/10 |
| one edge +5 % | 0/10 |
| every edge +2 % | 0/10 |

The reference's per-edge sigma (2-16 %) limits this oracle to gross errors: signs,
mapping, a missing factor. Fine-grained equivalence rests on bit-identity with the
per-edge route, whose own statistics are pinned in `test_fan_edge_inkernel_warp.py`.

**Measured.** RTX 4070 laptop; clocks 1980-2535 MHz under load with no power-cap
reason. The CPU was saturated by another agent's jobs and the GPU shared, so absolute
times drift: the tables give interleaved best-of-9 values, with ranges over three
alternations. One DAO gradient step (tungsten MLC, 10 segments x 20 pairs = 400
edges, 2 000 histories per edge, 2 batches, electrons on), the tree before the change
against the tree after it in alternating processes:

| | build the edges | contraction | step |
|---|---|---|---|
| per-edge objects, before | 82-102 ms | 38-41 ms | 126-151 ms |
| `SegmentedEdgeSource` | 0.37-0.44 ms | 22.7-24.2 ms | 23.4-24.6 ms |

On an idle machine before the change the per-edge step was 51 + 12 (upload) + 18 ms.
Build plus upload of the edge set alone:

| edges | host time | proposal kernels (device) |
|---|---|---|
| 400 | 0.2 ms build, 0.1 ms columns, 1.0-1.7 ms upload | 0.7-0.8 ms |
| 15 000 (125 segments x 60 pairs) | 1.0 ms build, 0.9 ms columns, ~4 ms upload | 2.7 ms |

At 15 000 edges the device holds about 63 MB, nearly all of it the per-edge proposal
tables. The upload's remaining host time is Warp object overhead: ~56 slice views and
~72 struct field assignments, 0.55 ms at 400 edges, the price of the `EdgeSet` field
layout. The DAO example (`tip_gradient` now builds a `SegmentedEdgeSource` over a
`with_settings` iterate) measured 0.03 s per gradient against 0.09 s recorded on
2026-10-21.

**Where a 400-edge contraction's time goes** (Warp activity timing, one call, ~24 ms
wall; 17.4 ms of GPU activity):

| activity | GPU time |
|---|---|
| transport kernels (electron 5.3, photon 1.9) | 7.2 ms |
| contraction reduction (`contract_dij_batch`, 2 batches) | 3.5 ms |
| edge generator, 4 launches | 2.4 ms |
| memsets and fills of the dense group map (400 x 46 080 int64 per batch) | 1.8 ms |
| proposal tables (since reduced to 0.7 ms) | 1.6 ms |
| host-device copies | 0.5 ms |

The remaining ~7 ms is host time: 69 kernel launches per call (Warp's argument
packing of the large structs) and the drain's synchronizing readbacks. Only the
proposals and the upload were inside this change. The next levers belong to the
engine:

- the dense per-group quanta map that the contraction route still fills, zeroes and
  reduces (5.3 ms together). Contracting deposits directly would need a fixed-point
  per-column accumulator to stay deterministic;
- the launch count (69 per call).

Recorded as the next engine lever, deferred: with the gradient at ~24 ms a DAO
iteration should come to ~80 ms, under the 0.1 s target.

**Not done: one upload shared by the forward pass and the gradient.** Letting the edge
kernel read the segmented fan's `StackArrays` + `SegmentArrays` would save the
combined stack's share of one packed transfer (tens of microseconds). It would also
need a second chord path in the generator: the per-edge route allows any stack and
frame per segment, which one template cannot express. Sharing the forward's
attenuation table would not be bit-preserving either. `MuArrays` interpolates
`log(rho mu/rho)` with `inv_dlog` taken from `np.log(geomspace)`, while the edge
kernel interpolates `log(mu/rho)` and multiplies by the density after `exp`, so every
edge weight would move at float32 rounding. The forward `run_dij` also re-uploads its
`SegmentedFanUpload` per call, so there is no resident upload to share yet.

**Open.** A leaf pair with no extent in the field is refused, as the per-edge source
refuses it; an arc whose outer pairs leave the field needs a pair selection (not
built). The tables still hold one proposal row per edge, as the generator requires.

**Single-source sampler, 2026-11-03.** The in-kernel generator implemented the proposal
inversion, the ray geometry, the weight and the near-tangent chord itself, next to the
host's `FanEdgeDerivativeSource._rays`: a sampling routine in `backends/`, against
AGENTS.md 3, and the cancellation-sensitive arithmetic in two places (review of PR #13).
It now lives in `geometry/edge_rays.py`, pure scalar functions that the physics loader
compiles as it compiles `geometry/fan.py` and the collimation twins, and that the host
route calls ray by ray: the crossing (`edge_ray_point`, `edge_proposal_point`), its
place in the frame and field (`edge_ray_crossing`, `edge_in_field`), the twin
dispatch (`edge_chord`, `edge_chord_rate`), the weight (`edge_core`, `edge_weight`),
and the proposal's node density, cell mass and defensive mixture. The energy inversion
is `fan.spectrum_energy`, the edge step's pick `fan.select_by_cdf_at` (host and
kernel), the mu lookup the data layer's `lookup_loglinear_2d`. What the kernel alone
takes is there too, each pinned against the host's float64 value in
`tests/unit/test_edge_rays.py`: the near-tangent closed form (`edge_block_terms`,
`near_tangent_terms`; within 2e-10 of the float64 twin, which needs no such branch),
a focused jaw's slant chord and the rest-of-stack transmission by subtraction, the
proposal's sequential normalization (`edge_proposal_tables`; within 1.2e-14 of the
host's pairwise-renormalized table) and the frame rotation. `backends/warp/aperture.py`
keeps the uploads, the launches, the RNG keying, the loop over a segment's stack devices
and the queue writes. Two operations stay target-specific on purpose: the host divides
the ray by its norm and the kernel multiplies by the reciprocal, because the twin's
float32 circle test amplifies that ulp near a tangent (a dividing kernel moved its
proposal tables by up to 5e-4 and the fixture's contractions by up to 8e-4 relative);
and each route normalizes its own proposal, as before.

Measured before and after on seeded outputs: on the host and reference route every one
of 1 871 arrays (per-edge batches and emits, proposal tables, `FanEdgeColumns` of both
builders, edge-step picks, reference contractions and an edge-step Dij) is
`np.array_equal`. On Warp cpu one operation order changed, the defensive share now
`d (w / s_max)` as on the host: proposal cells and cdf entries moved by one float32
ulp (at most 1.5e-7 relative), 13 of 9 600 arc rays by an ulp in direction and one in
weight, every contraction value, sigma and Dij entry stayed bit-identical, and only
the arc's energy books moved (5e-9 relative). With the old order restored all 105 Warp
arrays were bit-identical, so the move itself changes nothing. CUDA was not
re-measured.

## DAO gradient consistency, 2026-10-22

`tests/dij/test_dao_gradient_consistency.py`, `tests/integration/test_signed_weights.py`,
`examples/aperture_gradient_consistency.py`.
The question: the fan DAO computes its forward dose through `SegmentedFanSource`
(`run_dij`, each primary weighted in-kernel by its segment's stack transmission) and its
leaf gradient through `EdgeGradientBeamletSource` of `FanEdgeDerivativeSource` edges
(in-kernel rays, `run_beamlet_contraction` with the adjoint `a = dL/dD`). The two share
no code for the transmission derivative. Is the second the derivative of the first, at
the demo's settings?

**Answer, as first measured: on the positive side of every pair, yes; on the negative
side, no — by a factor of 0.17 to 0.31, and the cause was transport, not the estimator.
After the transport fix of 2026-10-23 (below), yes on both sides.** Every weight of a
`neg`-side edge is negative (a closing edge lowers the dose), and both engines lose the
continuous-loss deposits of negative-weight electrons: the deposit guard tests the
*weighted* amount, `if amount <= 0.0: return` (`deposit_or_escape` in
`transport/electron.py`, `_deposit_or_escape` in `backends/warp/kernels.py`), and the
bremsstrahlung guard `photon_w > 0.0` drops their photons too. A 2 MeV beam of weight -0.5
in water deposits 0.22 times what its +0.5 twin deposits on the reference engine, Warp cpu
and Warp cuda alike; KERMA mode is exact (`tests/integration/test_signed_weights.py`, both
pinned; the electron case was a strict xfail until the fix). Every earlier transport-level check of an edge
estimator used `+` edges, or KERMA mode (the in-kernel route's tests), and so could not see
it. Fixed on 2026-10-23 with the maintainer's sign-off, reference engine included
(AGENTS.md 2.2; see "The fix" below): the guard decides on the unweighted energy, the
photon guard on `photon_w != 0`, and the roulette cap `w < PHOTON_ROULETTE_WEIGHT_CAP`,
which never stopped a negative weight (unbiased, a variance matter), compares `abs(w)`. A
fourth asymmetry, `_deposit` rounding with `int64(x + 0.5)` (a negative deposit truncated
toward zero, about one 1e-9 MeV quantum on average), was considered and rejected (below). The 400-edge
DAO runs recorded above (loss 1.65 to 0.26) ran with half their gradient components
under-estimated: signs right, magnitudes and therefore the gradient's direction wrong. The
DijAssembler's `row_dose > 0.0` keep rule already drops negative Dij entries (documented:
prefer the contraction).

**Method.** The oracle is the common-random-number central difference of the forward model
in the contracted quantity, `g_FD = a . (D(x + h) - D(x - h)) / 2h`, linear in dose, so
noise-free in `a`. The displaced tips are columns of one `SegmentedFanSource` run with
`run_dij(correlated=True, truncation=0, n_batches=1)`: the Warp segmented generator keys
the stream on the within-column history index only (`key = r`), so every column replays the
same ray and the same transport draws and differs only in its weight. The one
weight-dependent decision is the roulette cap: after `k` soft-photon survivals a weight
`T` has become `2^k T`, and the game is played only while that is below 4, so the two
columns of a ray diverge whenever `T(x - h)` and `T(x + h)` straddle a threshold
`4 * 2^-k` (for `k = 2`, `T = 1` against `T < 1`; for larger `k` any pair of
transmissions across `1/2`, `1/4`, ...). Each column's game is fair, so this costs
variance, not bias. Sigma from 40 independent replicate seeds (a column's batch sigma cannot see the
covariance of the two columns). CRN pays: with `correlated=False` the per-history sigma of
the difference is 10-13x larger at `h = 0.005` cm (variance x100-170). A closed pair, whose
tips may not cross, takes the second-order one-sided stencil `(-3 f0 + 4 f1 - f2) / 2h` in
the opening direction. Each edge's difference runs on the fan **narrowed** to its leaf strip
(+0.5 mm) and blocked side and is scaled by the area ratio: under CRN every other ray gives
bit-identical columns, so this is exact in expectation (the 0.3 cm margin against the
tip's ~0.01 cm reach past the apex, focused sides, uniform area sampling of the fan);
checked against the full fan on two edges (0.5,
1.0, 0.8 and 2.0 sigma apart for the two edges and adjoints, per-history sigma equal within
the noise). The gradient is the shipped in-kernel contraction, independent columns
(`correlated=False`), 40 batches.

Settings: the demo geometry (`examples/aperture_gradient_fan_dao.py`: water 12 x 10 x 12
cm on 2.5 x 2.5 x 5 mm, Varian 6 MV Ali-Rogers fan from 100 cm, tungsten MLC 50-56 cm with
focused sides and 8 cm tips, 20 pairs, EPICS 2023 tabulated data, electrons on, GS MSC,
shipped cutoffs). Plan: the demo's initial plan (`initial_tips`, `default_rng(7)`). Adjoints:
`real`, the demo objective's `dL/dD` at that plan (forward `run_dij` 10 x 2e6 histories,
seed 11, Rx = 1.1 x mean PTV dose, L = 1.589), and `a = 1` (integral dose: opening raises it,
so `+` edges are positive and `-` edges negative). Edges: 4e6 histories each, seed 424242.
Finite differences: 40 replicates x 4e6 histories per column, seeds `100000 (e + 1) + r`,
`h` = 0.002, 0.005, 0.02 cm at the MLC mid-plane, the table at 0.005. RTX 4070 laptop at
2535 MHz under load (throttle 0x1), GPU shared with another agent: about 5 minutes of
transport for the finite differences, under a minute for every edge run. Values are
`a . dD/dx` with `D` in MeV/g per fan history and `x` in cm at the mid-plane.

Before the fix (the shipped estimator on the shipped transport):

| edge | g_FD (real) | g_edge (real) | z | ratio | z, a = 1 | ratio, a = 1 |
|---|---|---|---|---|---|---|
| open-neg (seg 0, pair 9) | -1.3257 ± 0.0028 e-7 | -3.6927 ± 0.0042 e-8 | +338 | 0.279 | +435 | 0.310 |
| open-pos | +9.4135 ± 0.0921 e-10 | +9.4147 ± 0.0946 e-10 | +0.0 | 1.000 | -0.2 | 1.000 |
| near-closed-neg (0.16 cm gap) | -1.3712 ± 0.0039 e-8 | -4.1253 ± 0.0190 e-9 | +223 | 0.301 | +569 | 0.303 |
| near-closed-pos | +1.2662 ± 0.0027 e-8 | +1.2654 ± 0.0028 e-8 | -0.2 | 0.999 | +0.5 | 1.001 |
| closed-pos (seg 0, pair 17) | +2.2193 ± 0.0345 e-9 | +2.1987 ± 0.0113 e-9 | -0.6 | 0.991 | -3.1 | 0.993 |
| closed-outer-neg (seg 1, pair 0) | -2.9603 ± 0.1398 e-10 | -2.1854 ± 0.0510 e-10 | +5.2 | 0.738 | +222 | 0.314 |
| closed-ptv-row-pos (seg 3, pair 6) | +1.2788 ± 0.0098 e-8 | +1.3009 ± 0.0050 e-8 | +2.0 | 1.017 | +0.2 | 1.001 |
| limit-pos (tip at 2.65 cm) | +2.4255 ± 0.0266 e-10 | +2.3860 ± 0.0644 e-10 | -0.6 | 0.984 | +9.2 | 1.013 |
| near-limit-neg (tip at -2.55 cm) | -2.2815 ± 0.0334 e-10 | -2.0876 ± 0.0675 e-10 | +2.6 | 0.915 | +1260 | 0.308 |
| seg5-pos (pair 6) | +1.6419 ± 0.0161 e-9 | +1.6381 ± 0.0131 e-9 | -0.2 | 0.998 | -0.8 | 0.999 |
| seg7-neg (pair 13) | +6.0840 ± 0.0226 e-9 | +1.0729 ± 0.0096 e-9 | -204 | 0.176 | +567 | 0.306 |
| seg9-pos (pair 12) | +9.5822 ± 0.0941 e-10 | +9.4323 ± 0.1205 e-10 | -1.0 | 0.984 | +0.4 | 1.000 |
| seg5-near-closed-pos (pair 3) | +7.7515 ± 0.2093 e-10 | +7.3682 ± 0.1572 e-10 | -1.5 | 0.951 | +0.9 | 1.002 |

After the fix (2026-10-23, `examples/aperture_gradient_consistency.py --study`,
the same seeds and settings; all 13 edges are one contraction there, so the `+` rows are
a fresh, independent sample of the same estimator; the finite differences are the
forward model's and did not move):

| edge | g_FD (real) | g_edge (real) | z | ratio | z, a = 1 | ratio, a = 1 |
|---|---|---|---|---|---|---|
| open-neg | -1.3257 ± 0.0028 e-7 | -1.3241 ± 0.0015 e-7 | +0.5 | 0.999 | +0.2 | 1.000 |
| open-pos | +9.4135 ± 0.0921 e-10 | +9.1981 ± 0.1300 e-10 | -1.4 | 0.977 | -0.7 | 0.999 |
| near-closed-neg | -1.3712 ± 0.0039 e-8 | -1.3692 ± 0.0024 e-8 | +0.4 | 0.999 | +1.5 | 0.998 |
| near-closed-pos | +1.2662 ± 0.0027 e-8 | +1.2737 ± 0.0026 e-8 | +2.0 | 1.006 | +1.2 | 1.002 |
| closed-pos | +2.2193 ± 0.0345 e-9 | +2.2131 ± 0.0110 e-9 | -0.2 | 0.997 | -3.0 | 0.993 |
| closed-outer-neg | -2.9603 ± 0.1398 e-10 | -2.8932 ± 0.0560 e-10 | +0.5 | 0.977 | +0.7 | 0.998 |
| closed-ptv-row-pos | +1.2788 ± 0.0098 e-8 | +1.2919 ± 0.0053 e-8 | +1.2 | 1.010 | -0.1 | 1.000 |
| limit-pos | +2.4255 ± 0.0266 e-10 | +2.3980 ± 0.0620 e-10 | -0.4 | 0.989 | +7.9 * | 1.012 * |
| near-limit-neg | -2.2815 ± 0.0334 e-10 | -2.2475 ± 0.0770 e-10 | +0.4 | 0.985 | -1.2 | 1.001 |
| seg5-pos | +1.6419 ± 0.0161 e-9 | +1.6442 ± 0.0130 e-9 | +0.1 | 1.001 | -0.8 | 0.999 |
| seg7-neg | +6.0840 ± 0.0226 e-9 | +6.0696 ± 0.0190 e-9 | -0.5 | 0.998 | +1.5 | 0.998 |
| seg9-pos | +9.5822 ± 0.0941 e-10 | +9.6364 ± 0.0970 e-10 | +0.4 | 1.006 | -0.5 | 0.999 |
| seg5-near-closed-pos | +7.7515 ± 0.2093 e-10 | +7.3978 ± 0.1800 e-10 | -1.3 | 0.954 | +1.2 | 1.002 |

\* The finite difference's truncation at the field limit (below); a heuristic Richardson
extrapolation from (0.005, 0.02) gives 0.065554 against the edge's 0.065479 ± 0.000095,
z = -0.7. Over the
13 edges chi2 = 10.4/13 (p = 0.66, real; the five `-` edges alone 1.0/5) and, with the
field-limit edge extrapolated, 20.2/13 (p = 0.09, a = 1; `-` alone 6.3/5, p = 0.28). The
`closed-pos` row at a = 1 (-3.0) is the closure property below: the edge returns the
exact derivative at closure, the difference quotient the slope beyond the 0.5-micron
overlap. These match the sign-fold diagnosis below to within the noise, as they must:
the fold was the fix's effect done by hand.

**The fix, 2026-10-23** (maintainer sign-off, conditional on every positive-weight
result staying bit-identical; reference engine and Warp kernels in one commit, AGENTS.md
2.2). Three changes, each in both engines where both had it:

1. The electron deposit guard (`deposit_or_escape` in `transport/electron.py`,
   `_deposit_or_escape` in the Warp kernels, which now takes the particle's weight)
   skips unless the *unweighted* energy is positive, read from the signs of the
   weighted amount and the weight: `if w > 0: skip if amount <= 0; else: skip if
   amount >= 0`. For `w > 0` that is the old test, operation for operation; a
   zero-weight particle (`amount == 0`) is skipped as before.
2. The bremsstrahlung photon is spawned for any nonzero weight (`photon_w != 0`).
3. The photon roulette cap compares `abs(w)` (Compton in both photon loops, the soft
   bremsstrahlung photon in both electron loops): a negative weight used to stay in the
   game for ever; unbiased either way, a variance matter.

**Considered and rejected: symmetric deposit rounding.** The Warp fixed-point deposit
quantizes with `int64(x + 0.5)`, which truncates a negative amount toward zero, so a
negative-weight deposit keeps a bias of about +1 quantum (1e-9 MeV) on average, between
0.5 and 1.5 quanta per deposit (review measurement: mean 1.0005; `_escape` already rounds
with `floor`; the reference engine scores in float64). Correction: the revert's commit
message said "below one quantum"; the mean is one. `floor(x + 0.5)`
removes it and agrees with the cast for every x above -0.5 quanta, but it is not
bit-identical for positive weights: positive-weight electron transport files a few
negative float32 residues of its energy bookkeeping (tens of quanta; the reference
engine's float64 loop files them too, at about 1e-16 MeV), which then round differently.
On the golden arrays below: KERMA routes and the reference engine unchanged; with
electrons, 1 to 85 of 2304 voxels per dose map moved, by at most 4e-9 relative, the
energy books by at most 2e-7 MeV over 2e5 histories. Committed and reverted on the
research branch (neither commit is in this history): the sign-off's bit-identity condition outranks a one-quantum bias (about
5e-8 of the dose here: the Warp mirror residual). Also considered and **not** taken (review): rounding
conditioned on the weight's sign, `-int64(-x + 0.5)` for a negative weight, would keep
positive-weight bit-identity and make Warp's mirror exact, at the cost of threading the
weight into `_deposit` at every call site; the effect it removes is ~5e-8 relative.

Other weight-sign tests found and left alone, because they are not this defect: the
roulette game's `new_w == 0.0` kill tests (sign-symmetric); the head pre-solve's
`weight > 0` / `weight <= 0` tests (`geometry/head.py`, `backends/warp/presolve.py`),
which act on a physical beam's phase space, never on a signed estimator's history (none
is ever pre-solved), and which the record above documents as dropping zero-weight
primaries; source and spectrum validators refusing negative weights by contract; and the
DijAssembler's `row_dose > 0` keep rule (negative Dij entries; documented, use the
contraction).

**Bit-identity.** Seeded outputs of the unchanged code were saved first (31 arrays on the
reference engine, 54 on Warp cpu, 62 on Warp cuda; 227,883 values: dose, sigma and the
four energy books of open-field runs, Dij index/dose/sigma arrays and contractions)
across a 6 MeV pencil with lung- and bone-density slabs (electrons on and KERMA), a 15
MeV pencil (pair production), electron primaries, a uniform 1.5 T field, a host-route
custom source at weight 0.7, the spectral fan (dose to medium and to water), the
lattice Dij, a segmented fan's Dij (correlated and independent), its contraction and
plan view, a `+`-edge gradient contraction, a tabulated water/lung/bone/tungsten 6 MV run
(both scoring modes) and, on cuda, two batch lanes. With the three changes every array
compares exactly equal on all three targets (re-checked after the rounding revert:
0 of 147 arrays differ).

**Tests after the fix.** The strict xfails are gone. `test_signed_weights` pins:

- the in-engine mirror (`-D(-w)` against `D(w)`, two seeds) on all three targets, with
  electrons on and in KERMA mode;
- the mirror on one stream at `|w|` = 0.5 and at `|w|` = 4 = `PHOTON_ROULETTE_WEIGHT_CAP`
  (where every roulette site meets the cap; at 0.5 almost no lineage reaches it): exact
  on the reference engine (2000 histories, so the bremsstrahlung roulette fires), and
  on Warp cpu and cuda up to the rounding, `int64(x + 0.5) + int64(-x + 0.5)` being 0 or
  1 quantum per deposit pair, the summed absolute residual bounded by 1000 deposits per
  history (measured 8.2);
- closed energy books for signed runs (emitted = deposited + escaped + unscored at the
  project's 1e-4 balance tolerance; measured 1e-8) on all three targets;
- a negative-weight broad beam on Warp cpu against the reference engine, scored as a
  depth-dose curve (16 x 16 cm by 5 mm slabs, 16 batches) at **equal history counts**
  (3200 each; see the oracle note below). Calibrated over 30 seed pairs (2 p below 0.05,
  none below 0.01, KS p = 0.19); the old guard in Warp alone gives chi2 894 against a
  null of 23, in the reference engine alone 995 against 23. The first version compared
  3200 reference against 32,000 Warp histories; it was calibrated at its bin occupancy
  (20 seed pairs: 1/20 below 0.05, KS p = 0.12), but equal counts remove the question.
  An earlier version on 2 x 2 cm x 2 mm voxels had no power (the Warp mutant passed with
  mean z^2 1.09: a few deposits per voxel per batch, the trap recorded under "Segmented
  fans").

Mutants (each alone, all other fixes kept): the deposit guard in Warp fails the Warp
mirror tests and all four `-`-side DAO tests, in the reference engine both reference
mirror tests and the ref-vs-Warp check; the Warp bremsstrahlung guard (`photon_w > 0`,
which an independent review found passing every earlier test) fails the Warp one-stream
mirror (4 cases) and the Warp book closure (4 cases); the roulette cap reverted to `w <
CAP` fails, at `|w|` = 4, the reference mirror (Compton site and bremsstrahlung site) and
the Warp mirror (scattered-photon site at both weights, bremsstrahlung site at 4). The
Warp split-path cap site is compile-time dead code while `PHOTON_SPLIT_N = 1`. The
`-`-side DAO gradient tests pass on cpu and cuda.

**An oracle artifact, not an engine difference.** A positive-weight comparison seen on
the way first looked like a Warp/reference disagreement: water `VoxelGrid.uniform_water(
shape=(8, 8, 16), spacing=(0.5, 0.5, 0.5))`, analytic cross sections, a 2 MeV parallel
beam over x, y in [1, 3] cm starting at `z = 0` (the `_SignedBeam` of
`tests/integration/test_signed_weights.py`), electrons on, 8 batches, reference 1600
histories against Warp cpu 16,000: chi2 773 against a null mean of 369 (seed 20260711)
and 487 against 354 (seed 20260722). The review traced it to the oracle: the beam
entering from vacuum at `z0 = -1` fails the same way (770.8 against 368.6); the reference
engine at 1600 against itself at 16,000 fails on 3 of 3 seeds (737, 526, 680 against
~360; reproduced here: 692 against 358); Warp cpu 1600 against 16,000 fails on 6 of 7;
and the reference engine against Warp at **equal** counts (1600 / 1600 and 16,000 /
16,000) passes on both seeds (reproduced). `assert_chi2_consistent_batched` is
miscalibrated with ~200 histories per batch on sparse voxels set against ten times as
many on the other side: the sparse side's batch sigma misrepresents its own voxel
distribution. **Any test comparing unequal history counts on sparse voxels is suspect**;
equalize the counts, or calibrate over seed pairs at the test's own bin occupancy.

The negative-side ratio is not one factor: 0.31 for the integral dose, 0.17 to 0.92 for the
demo adjoint, whose mixed signs weight the lost electron dose differently per edge, so the
defect bends the gradient vector, not just its length.

**The cause, isolated.** A probe-only copy of the edge generator that writes `-w` (every
weight of a `-` edge is negative, so the run is all-positive) and negates the contraction
afterwards — a sign fold, linear in exact arithmetic, never committed — brings every `-`
edge into agreement: z = +0.5, +1.8, +0.1, +1.9, +0.0 (real) and +0.2, +1.9, +0.6, -0.8,
+1.8 (a = 1) for open, near-closed, closed-outer, near-limit and seg7. The copy without the
fold is bit-identical to the shipped kernel on the same seed (both edge sets, both adjoints).
The generator itself is sign-symmetric: mirroring a segment in u maps a `-` edge onto a
`+` edge, and the device mean weights of the pairs agree to 0.02-0.04 % (within 1.3 sigma,
2e7 rays each; the host route mirrors exactly). Over all 13 edges, `-` folded:
chi2 = 15.2/13 (p = 0.30, real) and, with the field-limit edge Richardson-corrected
(below), 19.6/13 (p = 0.10, a = 1).

**Transport-free, every edge** (mean edge weight from the in-kernel generator, 2e7 rays,
against the difference quotient of the mean forward transmission from the forward
kernel's own `segment_transmission`, 4e7 rays, h = 0.005 / 0.01 / 0.02): the open, nearly
closed and other-segment edges agree to 0.0-0.2 % (|z| <= 1.8), `open-neg` and `seg7-neg`
at 0.3-0.4 % (z 2-3, falling with h for seg7); `limit-pos` shows the truncation discussed
below (1.1 / 7 / 30 % at the three steps). **Closed pairs differ by 0.8-1.9 %** (z 3-8):
the edge gives less. Not a defect, the forward model's micron structure: the tangent rays
of two touching rounded tips cross the mid-plane short of the apex by `R alpha^2 / 2`
(2.6e-5 cm near the axis), so the two leaves' penumbrae overlap in a 5e-5 cm band only
while the pair is closed. 13 % of a tip's derivative mass lies within 1e-4 cm of its
tangent (the `1 / sqrt(q - q0)` density), the opposing leaf transmits 0.954 of it there,
and the exact derivative at closure is 1.7 % below the slope seen once the tip has moved
more than about half a micron (`closed-pos`: edge 9.258e-3; the same weights without
`T_rest` 9.432e-3; forward difference quotients 9.40-9.44e-3 on the device, 9.43e-3
± 0.09 on the host in float64). The edge estimator returns the derivative, which is
correct; any finite difference coarser than the overlap returns the secant. The
transport-level closed rows (ratios 0.991-1.017, |z| <= 3.1) are consistent with this
within their resolution. An optimizer stepping by tens of microns sees the secant: at a
closed pair the exact gradient understates the first step by ~1-2 %, a property of the
leaf model, not of the estimator.

**The finite-difference step.** Paired over the replicates, `g_FD(0.02) / g_FD(0.005) - 1`
is within 2 sigma of zero for every edge but one, at a = 1 to 0.1-0.4 % — except the tip at
the field limit, -20.4 ± 0.04 %. There the tangent ray sits `R alpha^2 / 2` = 0.1 mm inside
the field edge, `D(x)` has a nearby `(q_field - q0)^{3/2}` point, and the O(h^2) constant
is ~460 /cm^2: -1.1 % at 0.005, which is the z = +9.2 above. Richardson from (0.005, 0.02),
`(16 g(0.005) - g(0.02)) / 15`, gives 0.065553 ± 0.000041 against the edge's
0.065534 ± 0.000084 (z = -0.2) at a = 1 and z = -1.1 for the real adjoint. The
extrapolation is heuristic: it assumes an `h^2` law fitted from the same two steps, near
a `(q_field - q0)^{3/2}` point that the h = 0.02 stencil straddles. An independent
three-step check (review, 2026-10-23) predicts `g(0.002) / g(0.005) - 1` = +1.15 % from
that law and measures +0.99 ± 0.09 % (1.8 sigma), with the h = 0.002 point itself
contaminated by the float32 jitter below. At h = 0.002 the
difference quotient moves by up to 1.1 % at a = 1 with 3-6 sigma significance on some edges
(closed-pos -1.09 ± 0.19 %, closed-ptv-row-pos -0.73 ± 0.17 %, open-neg +0.25 ± 0.07 %),
with no consistent sign and nothing at 0.02 — not truncation, but float32. The forward
kernel's chord twin (`mlc_path_length`, float32, fast_math) keeps the near-tangent
cancellation that the edge generator's closed form removes (`aperture._NEAR_TANGENT_S2`,
below q - q0 = 2e-3 cm), so `D(x)` jitters on micron scales and a difference quotient
divides that by 2h. Transport-free, on the same rays, the device's difference quotient of
the mean transmission against the host's float64 one: `closed-pos` -1.0 % at h = 0.001 and
0.002 (24 sigma), -0.06 % at 0.005, -0.07 % at 0.02; `open-pos` -0.08, -0.07, +0.05,
+0.02 %. It matters to the oracle, not to the gradient: h = 0.005 is the compromise
(float32 below 0.1 %, truncation negligible), and the field-limit edge needs the
extrapolation. Whether the optimizer's loss should see a forward model with that closed
form too is open.

**Power** (the comparison must fail when it should). Study scale, chi2 over the 8 `+` edges
(and over all 13 with `-` folded), the real adjoint:

| mutant | `+` edges | all, `-` folded |
|---|---|---|
| none (control) | 7.9/8, p = 0.45 | 15.2/13, p = 0.30 |
| sign flipped | p = 0, max abs z 649 | p = 0 |
| wrong segment (same pair, next segment's tips) | p = 0, max abs z 458 | p = 0 |
| wrong pair (neighbouring pair) | p = 0, max abs z 594 | p = 0 |
| support at the apex (the 2026-10-21 sliver bug) | p = 0, max abs z 84 | p = 0 |
| mean energy in every mu | 76/8, p = 3e-13 | 90/13, p = 2e-13 |
| float32 near-tangent chord restored | 27/8, p = 8e-4 | 183/13, p = 3e-32 |
| T_rest dropped | 9.3/8, p = 0.32 (missed) | 16.8/13, p = 0.21 (missed) |
| shipped `-` edges (the transport defect) | — | 2.1e5/13, p = 0 |

Kernel mutants are probe-only copies of `generate_fan_edges` monkeypatched into the engine.
With a = 1 the same mutants are caught by larger margins. The one miss is physics, not
power: in an MLC-only stack the rest of the stack along a tip edge's rays is the opposing
leaf, which those rays miss unless the pair is closed, and there only in the apex sliver —
dropping `T_rest` moves `closed-pos` by 0.2 % and nothing else measurably. With a jaw in the
stack it matters, and the fast test has one (below).

**Variance and cost.** Per transported history, on the full fan at h = 0.005, the finite
difference's variance (two columns per estimate) is 2.5e3 to 3.3e4 times the edge
estimator's (median ~6e3, both adjoints); the edge estimator spends every history on the
derivative, the difference on the 1/20 of the fan in one leaf strip and the few millimetres
of it in the tip's penumbra, and its tangent-ray rays carry weights growing like h^{-1/2}.
GPU cost per history is similar (edge runs ~1.2-2.2e-8 s, difference columns ~2e-8 s with
read-back, shared GPU), so the efficiency ratio is of the same order. That is 30-300x more
than the parallel-beam figure above (76-130x, where the oracle was already restricted to a
slice).

**Fast tier** (`tests/dij/test_dao_gradient_consistency.py`, 8 s on Warp cpu, 2 s on cuda):
a 2 MeV +- 10 % fan, a focused dense-water jaw over a 2-pair dense-water MLC (the jaw's `+`
edge cuts every `+` tip's penumbra, so `T_rest` is far from 1), two segments; edges: an open
pair, a nearly closed pair (0.8 mm), a closed pair (one-sided), the jaw edge; electrons on;
adjoints `a = 1` and a mixed-sign one. Per edge Welch's t with its Welch-Satterthwaite dof,
mapped to a normal score, chi2 over the edges (calibrated at 8-16 replicates/batches, where
the batched helper's normal approximation over 4-5 terms is not). Both sides must pass
(the `-` side was a strict xfail until the fix); the oracle must reject a sign flip, the
wrong pair and the wrong segment (p < 1e-6). Calibration over 20 seed sets x 2 adjoints on
cuda: 3/40 p-values below 0.05 and 1/40 below 0.01, KS p = 0.003 (a bulk skew; the folded
`-` side, KS 0.28); 10 sets on cpu: 2/20 below 0.05, KS 0.19. An independent 12-seed cuda
calibration after the fix (review) finds no edge biased (per-edge mean z within ±0.42;
all z against N(0, 1): KS p = 0.63 and 0.85 for the two adjoints) but per-edge z
variances of 0.26 to 1.7: the bulk skew comes from per-edge sigmas that are
conservative for some edges, not from a bias. Test-scale power (cuda / cpu):
`T_rest` dropped p = 1e-58 / 7e-13; the transport defect p ~ 1e-110 / 1e-25; mean energy,
near-tangent chord and apex support are not caught at this size (a 0.4 MeV wide spectrum,
dense water instead of tungsten; apex p = 4e-3 on cuda only) — the example reproduces the
study for those.

Resolved and open, as of 2026-10-23: (1) the transport sign fix: done (above). (2) The
float32 forward chord at small h: not now — the edge kernel's own closed-form near-tangent
branch is being corrected first for bisection-fallback edges, separately; the forward
kernel keeps the twin. (3) At touching leaves the estimator keeps the exact derivative;
the 1.7 % gap to the secant a finite step sees is a property of the rounded-tip leaf
model, recorded above, not a defect. (4) The fast-tier oracle's mild bulk skew on cuda
(KS p = 0.003): diagnosed as conservative per-edge sigmas, not bias (above); seeds are
deliberately not tuned for it. (5) Open: leaf-scatter (first-Compton) collimation is not
covered by either path, as stated above. (6) Resolved: the apparent ref/Warp
disagreement for a beam starting on the phantom surface is the batched oracle at
unequal history counts (above). (7) Rejected: symmetric deposit rounding, and its
weight-conditioned variant (above). (8) Open: `run_dij` and `DijAssembler` drop
negative-weight Dij entries silently (the `row_dose > 0` keep rule); a warning when a
beamlet column holds negative dose would make that visible. Not now.

## DAO iteration at 0.1 s, 2026-10-23

**One Monte Carlo DAO iteration of the 6 MV fan example (10 segments x 20 leaf pairs, 400
edges, tungsten MLC, electrons on) now takes 68 ms on a laptop RTX 4070**, against 0.6 s on
2026-10-21 and 178 ms after the base engine items merged. Measured on the idle GPU at 2.5
GHz, best of 9 (median 69 ms), 400k histories per segment forward and 2000 per edge:

| part | ms |
|---|---|
| forward: one segmented `run_dij` (correlated columns) | 43 |
| edge contraction over a `SegmentedEdgeSource` | 20 |
| dense columns, plan dose, objective, weight gradient | 2 |
| `with_settings` + edge-source build | 0.4 |

The weight gradient needs no engine call: it is the adjoint's dot product with the forward
pass's Dij columns. The example now runs this route (`examples/aperture_gradient_fan_dao.py`,
100 iterations by default). It is also the first optimization on corrected gradients: before
the negative-weight transport fix (see "DAO gradient consistency"), every neg-side edge
gradient was 3-6x too small. On the corrected gradients the loss falls from 2.15 to 0.24 in
100 iterations (seed 7), with the OAR's dose clearly reduced against the PTV's. The earlier
2026-10-21 run (1.65 -> 0.26 in 40 iterations) used the defective gradients and is
superseded.

Next engine levers, deferred (not needed for the target): the dense per-group map in the
contraction route (5.3 ms) and the launch count per call (69); the forward pass is now the
larger share.

**History budget (measured 2026-10-23).** 100 iterations from the same initial plan, 4
forward x 4 per-edge history levels x 3 MC seeds; each final plan re-scored with 4e6
histories per segment, because the in-loop loss carries the forward noise (its quadratic
terms add the voxel variance, so the demo's printed 0.24 is a true 0.13). Initial true
loss 1.528.

| forward / segment | per edge 500 | 1000 | 2000 | 4000 | ms per iteration |
|---|---|---|---|---|---|
| 50k | 0.1886 | 0.1870 | 0.1851 | 0.1860 | 35-51 |
| 100k | 0.1651 | 0.1652 | 0.1637 | 0.1629 | 43-53 |
| 200k | 0.1448 | 0.1452 | 0.1432 | 0.1424 | 56-61 |
| 400k | 0.1309 | 0.1268 | 0.1260 | 0.1276 | 67-77 |

Seed-to-seed sd 0.0004-0.0041. Three readings:

- The edge-gradient histories do not limit the plan: 500 per edge is as good as 4000 at
  every forward level, and the contraction's cost barely depends on them (it is mostly
  fixed per call).
- The forward histories do, and the gain has not saturated at 400k: the true loss keeps
  falling with forward statistics. The likely mechanism (not isolated) is the one-sided
  OAR and normal-tissue terms: max(0, D + noise - limit) is biased upward by noise, so the
  optimizer converges to the optimum of a noise-inflated objective. A schedule that grows
  the forward histories as the optimizer converges, or a forward estimate averaged over
  iterations, are the natural next experiments.
- About 30 ms per iteration is fixed (two launches, drains, dense Dij columns), so cutting
  histories cannot reach far below that; at the quality end, forward cost is the trade.

The demo keeps 400k forward and 2000 per edge.

**The re-scored loss still carries a noise floor.** The loss is quadratic in the dose,
so any finite-statistics estimate of it is inflated by the voxel variance. Re-scoring
the demo's 500-iteration plan at 2e6 and at 1e7 histories per segment gives 0.092 and
0.077 for the final plan (1.588 and 1.475 for the initial one). Losses re-scored at
equal statistics compare fairly across configurations, as in the table above, but in
absolute terms they are upper bounds. An unbiased estimate needs two independent
forward runs multiplied term by term, (D1 - Rx)(D2 - Rx), which is exact for the PTV
term though not for the one-sided terms; not built.

## VMAT arc demonstrator, 2026-10-23

`examples/vmat_arc_dao.py` optimizes a coplanar arc with one aperture per gantry angle
(default 36 angles at 10 degrees, 10 leaf pairs, 720 edges) on a TG-119-like C-shaped
PTV around a cylindrical OAR, with the same projected Adam as the fan example. It uses
only public API: each angle is its own `SegmentedFanSource` (one segment) whose fan and
beam frame are rotated about the gantry axis, so an iteration is one plan `run` and one
edge contraction per angle.

- **Cost:** 72 launches per iteration, 0.29 s on a laptop RTX 4070 (forward 0.16 s,
  gradient 0.13 s), about 4 ms per launch at 100k forward histories per angle and 500
  per edge. Per-launch overhead, not transport, sets it. One launch for all control
  points (a source holding several fans, each with its own frame; the edge kernel
  already takes per-segment frames) is the engine lever that would remove it; not
  built.
- **Initialization decides the result (DAO is non-convex).** The plain conformal
  projection covers the OAR from every angle around a C, and it is a local minimum for
  the tips: closing a leaf towards the OAR first underdoses the PTV. Gradient descent
  from it only scaled every MU down (re-scored loss 0.50 -> 0.16, PTV and OAR on the
  same DVH). Cutting the OAR's projection out of each row and alternating sides between
  neighbouring angles where the OAR splits a row gives 0.037 -> 0.0067 (2e6 histories
  per angle), with the OAR traded up from ~0.1 to 0.2-0.5 Rx (unpenalized below 0.5)
  for PTV homogeneity.
- One full run showed a sudden 4x slowdown of both forward and gradient from iteration
  97 on; two later runs with a GPU clock logger did not reproduce it (steady 2.0-2.2
  GHz, no throttle flag). Treated as an external clock event, not a code effect.
- Not modelled: leaf-speed and MU-rate constraints between control points, couch and
  collimator rotation, more than one aperture per angle.

## Arc edge sets, 2026-10-23

**The edge-gradient source takes the arc too.** After the arc source landed on develop
("Arcs in one launch", below), `SegmentedEdgeSource` lists the edges of a `SegmentedFanSource` whose
segments carry their own fan and frame. The field-dependent columns (edge extent,
mid-plane field, `L_r / A`, central ray) are evaluated once per distinct (fan, frame)
pair and laid out per segment; each segment's origin and rotation come from its own
frame; the layout cache is keyed on every fan and frame. The in-kernel generator was
unchanged: it already read a per-segment origin and rotation, previously filled by
tiling the template frame.

- **Exact against the per-edge route** on an arc with a turned collimator, a
  non-coplanar beam and per-fan off-axis fields: every host column equal in float64,
  every uploaded device array and the contraction (values, sigma, energy books)
  bit-identical on cpu and cuda, correlated and independent. The single-fan path is
  unchanged (its existing pins pass).
- **Statistical oracle: the host sampler, not the reference engine.** Device against
  the reference engine at 800 histories per edge failed an unmutated run on 2 of 3
  seeds while 0.25 and 0.5 degree frame tilts passed: edge-derivative weights are
  heavy-tailed, and 100 histories per batch give skewed batch means. The test instead
  compares the in-kernel rays with a subclass's Warp host route (each edge's
  `FanEdgeDerivativeSource` sampling, which the reference engine also emits; same
  transport), 8000 histories per edge on each side, cpu, about 1 s. Calibration 3/3
  seed pairs pass on cpu and cuda (chi2/k 0.96-1.76); every frame mutant fails:

  | mutant (in-kernel columns only) | chi2/k, cpu 8k | chi2/k, cuda 40k |
  |---|---|---|
  | none (3 seed pairs) | 0.96-1.76, pass | 1.16-1.69, pass |
  | segment 0's frame everywhere (the pre-port behaviour) | 74.8 | 426 |
  | frames of segments 1 and 2 swapped | 113 | 613 |
  | segment 2 tilted 0.5 deg | 10.6 | 51.8 |
  | segment 2 tilted 0.25 deg | 3.59 | 13.6 |
  | segment 1 tilted 0.5 deg | 21.0 | 62.7 |
  | segment 1 focal spot moved 2 mm in y | 6.76 | 12.2 |

- **The VMAT demonstrator now runs one arc source:** forward one `run_dij` (the angles
  as columns), gradient one edge contraction, so two launches per iteration instead of
  72. At the defaults, GPU at 2535 MHz without throttle: 0.10 s per iteration (forward
  0.08 s, gradient 0.02 s) against 0.29 s; re-scored loss 0.0370 -> 0.0066 (2e6
  histories per angle), as with per-angle sources (0.037 -> 0.0067). The forward is
  now correlated across angles (same uniforms through each angle's fan), which is
  unbiased; the evaluation doses are too.

## Stochastic optimization of the arc: history schedules and ipax, 2026-10-24

The VMAT demonstrator (36 angles, one arc source, 0.06 s + 0.38 s per million forward
histories per angle per iteration on the RTX 4070 laptop) optimized at equal optimizer
wall time, evaluations excluded; every plan scored at one common evaluation seed with
2e6 histories per angle (an upper bound with a noise floor, comparable across rows);
three base seeds each. The study scripts and their data were not committed.

**The step decay, not the history count, held Adam back; with a constant step the
history count should grow.** True loss, mean (sd):

| projected Adam | 15 s | 30 s | 60 s |
|---|---|---|---|
| step 1/sqrt(k), 100k histories (the example) | 0.00719 | 0.00672 | 0.00657 (0.00009) |
| step 1/sqrt(k), 1M | 0.00727 | 0.00662 | 0.00590 |
| step 1/sqrt(k), geometric 100k -> 2M (x1.03 per iteration) | 0.00667 | 0.00642 | 0.00604 |
| step 1/sqrt(k), adaptive norm test (Byrd et al. 2012, theta 0.5) | 0.00731 | 0.00703 | 0.00652 |
| constant step, 100k | 0.00616 | 0.00597 | 0.00581 |
| constant step, 1M | 0.00523 | 0.00404 | 0.00370 |
| constant step, N ~ 100k sqrt(k) (Smith et al. 2018) | 0.00414 | 0.00376 | 0.00347 |
| **constant step, geometric 100k -> 2M** | **0.00398** | **0.00364** | **0.00346 (0.00008)** |

- With the decaying step every history schedule plateaus at 0.0058-0.0066: the step
  shrinks before the noise matters. The norm test grows the sample to its 2M cap
  within ~40 iterations (the forward noise is large against the gradient), so it
  spends its time on few, expensive iterations.
- With a constant step, a constant 100k sample stalls in its noise ball (0.0058,
  no better at 60 s than at 30 s); growing the sample shrinks the ball. Geometric
  growth leads at every time; the spread over seeds is small (sd 1e-5 at 30 s).
- Against the example's default this is -47 % true loss at equal time.

**ipax (interior point, L-BFGS, Breedveld line search and mu) on a fixed sample is
second.** 1M histories per angle: 0.00470 (0.00006) in 31 s; run to convergence
(about 62 iterations) 0.00452 (0.00012) in 44 s; 2M: 0.00461 (0.00017) in 67 s. More
histories do not help, because the limit is not the sample. On ipax's own 1M sample,
Adam's 60 s plan scores 0.00443 against ipax's converged 0.00492, and ipax restarted
from Adam's plan lowers the sample loss to 0.00399 while the true loss rises from
0.00336 to 0.00375. ipax converges to the nearest local minimum of a non-convex
problem; Adam's noisy, momentum-carried steps find a better one; and fitting one
sample costs ~0.0004 of true loss.

**Retrospective approximation with ipax (Pasupathy 2010) is negative as built.**
Stages of 15 iterations at 250k, 500k, 1M, 2M, each on a new fixed sample, warm-started
from the previous stage: 0.00817 (0.00078); x-only warm start 0.00830; initial barrier
1e-3 or 1e-5 on restart 0.0085-0.0091 (one seed 0.0061); longer stages 0.00585 at
151 s. Every restart costs the solver its L-BFGS curvature and recentres the barrier:
the first iterates of a stage jump (true loss 0.0084 -> 0.12 at the 2M switch). A
warm-startable quasi-Newton memory, or switching samples inside one solve, would be
needed; not pursued.

**Leaf-speed limits as linear inequalities work in ipax, given a feasible start and a
small initial barrier.** |tip(a+1) - tip(a)| <= delta per pair and side between
neighbouring control points (no wrap), 700 rows besides the 360 non-crossing rows:

| limit per 10 deg, at the isocentre | none | 2.0 cm | 1.0 cm | 0.5 cm |
|---|---|---|---|---|
| true loss, mean (sd) | 0.00504 (0.00025) | 0.00504 (0.00022) | 0.00521 (0.00004) | 0.00581 (0.00020) |

- Feasible to 0 cm in every run, at the same ~30 s.
- The example's OAR-avoiding start alternates the open side between neighbouring
  angles; no leaf-speed limit allows that. Its nearest feasible point (a QP solved by
  ipax in 0.4-0.7 s, no Monte Carlo) straddles the OAR from every angle (true loss
  4.6 at 2 cm, OAR mean 2 Rx), and the optimizer then stays in that minimum (0.12).
  Started infeasibly, the restoration phase ends in the same place (0.13). A start
  that keeps one side for every angle is as good unconstrained (0.0374: the opposed
  beams cover the other side) and nearly feasible (0.0367 projected even at 0.5 cm);
  the rows above start from it.
- With the default initial barrier (0.1, against an objective scaled to 1 at the
  start) the projected start, on many constraint faces, was pushed inward and the
  line search stalled (0.071, 527 forward runs in 27 iterations). mu_init = 1e-3 fixes
  it; so does projecting onto 0.9 delta (strictly interior), at 1.5x the forward runs.
- Realistic limits are looser: 2.5 cm/s leaves at 6 deg/s gantry allow ~4 cm per 10
  degrees. The tight rows stand in for finer control-point spacing.

**Projected Adam takes the same constraints, and beats ipax at every limit.** The
constraints decouple per leaf pair (72 variables: two tip trajectories along the arc),
so the projection after every Adam step is ten small QPs, solved together by a batched
ADMM in numpy (variables scaled by sqrt(D), OSQP-style residual balancing of rho):
15-40 ms in the Euclidean metric (exact to 1e-8 cm against ipax), 30-130 ms in Adam's
metric D = sqrt(v_hat) (within 1e-3 cm of ipax's solution there; that reference is
itself ill-conditioned), always feasible to 1e-7 cm. Run with the best schedule
(constant step, geometric 100k -> 2M) from the same projected one-sided start, the
projection's ~1-2 s per run counted:

| limit per 10 deg | none | 2.0 cm | 1.0 cm | 0.5 cm | 0.5 cm, Euclidean projection |
|---|---|---|---|---|---|
| projected Adam, 30 s | 0.00390 | 0.00392 | 0.00407 | 0.00465 | 0.00477 |
| projected Adam, 60 s | 0.00359 (0.00012) | 0.00353 (0.00010) | 0.00385 (0.00005) | 0.00437 (0.00007) | 0.00445 (0.00014) |
| ipax, fixed 1M sample, ~30 s | 0.00504 | 0.00504 | 0.00521 | 0.00581 | |

At equal time projected Adam is 20-23 % below ipax, and the constraint costs it about as
much in relative terms (+7 % at 1 cm, +22 % at 0.5 cm by 60 s). Projecting in Adam's
metric is slightly better than Euclidean (0.00437 vs 0.00445), within the spread.

**Optimality gap: the DAO plans are at about twice a fluence-map lower bound.** Beamlets
at every angle: each leaf row x 0.5 cm bixels along the leaf travel (6480 at 36 angles),
each a fan clipped to its bixel with the MLC retracted, so the free non-negative fluence
on them contains every one-aperture-per-angle plan up to the bixel discretization (a tip
inside a bixel), the hardened leakage spectrum and the rounded tips. All beamlets are
segments of one arc source with one fan and frame each: one untruncated, independent
`run_dij` of 6480 x 100k histories takes 9 s (121 M non-zeros). Area-weighted, one
angle's bixels reproduce its open field to 0.999 in the high-dose region and 1.000 in
total. The fluence optimum (L-BFGS-B, convex, 2500 iterations, 7 min on the CPU) on the
same objective and prescription:

| | true loss |
|---|---|
| FMO, in-sample on Dij A / on (A+B)/2 (fits the Dij noise: biased low) | 0.00127 / 0.00129 |
| FMO fluence scored on independent Dij B, C, (B+C)/2, extrapolated in 1/N | 0.00142 |
| DAO, projected Adam unconstrained, 60 s (3 seeds) | 0.00260 (sd 0.00020) |
| DAO, projected Adam, 2 cm per 10 deg, 60 s (3 seeds) | 0.00253 (sd 0.00003) |

The DAO values are extrapolated the same way, from fresh seeds at 2M and 8M histories
per angle: (4 L_8M - L_2M) / 3. The losses quoted so far at 2M per angle (0.0035-0.0037)
carry a noise floor of ~0.001, about 30 % of the number: the loss is quadratic in a noisy
dose. The bound has converged with statistics (0.00127 -> 0.00129 at twice the
histories), so the gap of ~0.0012 (DAO ~1.8-2.0x the bound) is real.

**A second aperture per angle closes about a third of it.** K apertures per angle as K
sub-arcs over the same 36 angles (one arc source of 36 K segments; sub-arc k starts
opening side k of the OAR, MU 1/K each; each sub-arc speed-limited and projected on its
own), projected Adam with the growing sample, 120 s, extrapolated true loss:

| | 1 aperture | 2 apertures | 2 apertures, 2 cm per 10 deg |
|---|---|---|---|
| true loss (sd), 3 seeds | 0.00240 (0.00020) | 0.00199 (0.00002) | 0.00192 (0.00003) |

The gap to the bound (0.0013) shrinks from ~0.0011 to ~0.0007; the leaf-speed limit
costs nothing here either. The rest is unattributed: more apertures, or a better
optimum for the same apertures (the starting point, the optimizer), or both; two
apertures ran fewer iterations (~135 against ~195) in the same time.

## Arcs in one launch, 2026-10-23

**A VMAT arc's control points are the segments of one source.** The arc demonstrator
built one `SegmentedFanSource` per gantry angle and paid one launch per angle per
evaluation: 36 angles, 100k histories each, about 4 ms per launch on the laptop RTX
4070, i.e. launch overhead rather than transport. `SegmentedFanSource` now also takes
**one fan and one beam frame per segment** (`SegmentedFanSource(fans, template, xs,
frames=..., tips_neg=..., ...)`, or `from_stacks(fans, stacks, xs)` taking each
stack's frame). Segment `s` is `fans[s]` through the template's structure placed in
`frames[s]` at its own settings: same physics, same approximations as the single fan
(attenuation mode, no air, no head scatter). `run_dij` returns one column per control
point and `plan(weights)` the MU-mixed arc dose from one `run`, unchanged in contract
(normalized shares, `total_weight`). Chosen over a sibling class because routing,
`with_settings`, the plan view and the device layout are all the same code; a single
fan is the case of one fan repeated.

**Contract per segment, refused by name otherwise.** Each frame's origin on its fan's
focal point (1e-6 cm), the field downstream along its `w`, and — on the per-segment path
only — the fan's aperture axes equal to the frame's `u`/`v` (1e-9 per component): a
rotated focal spot with an unrotated frame is the likeliest arc setup error, and only
the downstream check catches it, and only at large angles. The single-fan path keeps
exactly its old checks (no axis contract), so no existing input is newly refused. All
fans must draw from one spectrum (the same object, or equal edges and CDF): one machine,
one energy, one attenuation table. The structure check is unchanged. `fan` raises for
distinct fans; `fans`, `frames` and `spectrum` are per-segment/shared accessors.
`with_settings` keeps the fans and frames (0.44 ms at 180 segments, revalidation
included; 0.05 ms for a single fan).

**Device layout.** `SegmentArrays` gained the per-segment frame (origin, row-major
rotation, `flatten`'s layout) and fan (`FAN_STRIDE = 14`: focal point, field centre,
u and v axes, widths). The generators read the segment's fan for `fan_primary` and its
frame for the transmission; the stack's own origin/rotation are no longer read there.
A single-fan source uploads its fan and frame once per segment, so there is one kernel
for both. The fan scalars left the kernel signatures. **Bit identity of the single-fan
path:** 345 seeded arrays saved before the change — every primary queue the generators
handed to the drain, Dij (correlated, independent, `group_start > 0`), plan dose and the
energy books of four sources incl. a rotated frame, lanes on cuda, and the host routes
(reference Dij and plan, `sample_beamlet_batch`, plan `sample_batch`) — are
`np.array_equal` after it, on cpu and cuda:0.

**Correlated sampling across fans.** The fan's four uniforms are segment-blind, so a
correlated history `r` replays the same uniforms in every column: the same energy and
in-aperture offset fed to *each segment's own fan* — the same beam's-eye-view ray in
each frame when the apertures are congruent, not the same engine-frame ray. Keys are
unchanged (`r` correlated, `s * n_per + r` independent), independent of the chunk.
Consequence, pinned: on one device a correlated arc column is bit-identical to the Dij
column of that angle's single-fan source (same keys, fixed-point books, the group-size
invariance). The AGENTS.md section 8 caveat on quadrature sigmas of correlated columns
applies to arc columns as it does to beamlets.

**Statistical evidence, with power.** `tests/dij/test_segmented_arc_dij.py`: a plain
beam at gantry 0, a 90-degree gantry with the collimator at 30 degrees, and a
non-coplanar 225-degree beam (a couch-like 25-degree turn about +z) with its own
off-axis, narrower field; v-asymmetric leaf tips in every segment; 2 x 4 x 2 cm bins
(tens of KERMA events per batch), pooled-dose mask, equal histories per side. Arc
columns vs per-angle sources (40k per column), vs the reference engine (12k,
independent mode), the plan view vs the MU-weighted sum of per-angle plan runs (60k
per side, the per-angle runs allotted `N w_s / W`), and vs the reference. Unbiased
p-values at the pinned seeds 0.75-0.94 (per-angle), 0.16-0.98 (reference), 0.71 / 0.40
(plan). Chi-squared/null ratios of 0.66-0.92 on some columns are below 1 — the batched
oracle is conservative there — so power was measured, not assumed: device-upload
mutants applied to multi-segment sources only (scratch probe, cuda), each catch
reported where the mutant touches the column:

| mutant | per-angle columns | reference columns | plan vs per-angle | plan vs reference |
|---|---|---|---|---|
| every segment in segment 0's frame | p = 0 | p = 0 | p = 0 | p = 0 |
| fans of two segments swapped | p = 0 | p = 0 | p = 0 | 9e-288 |
| rotation missing (identity, origins kept) | p = 0 | p = 0 | p = 0 | p = 0 |
| segment 2's frame tilted 0.25 deg about +y | 1.3e-4 | 2.3e-3 | 3.8e-4 | 0.12 |
| segment 1's frame tilted 0.25 deg about +y | 0.010 | 0.83 | 0.60 | 0.32 |

(On the first, coplanar version of the arc — gantry 0/90/225 about +y, every field on
the isocentre — the same probe gave: segment 2 tilted 3 / 1 / 0.5 / 0.25 deg caught at
p = 0 / 0 / 2e-170 / 2e-14 per-angle and 0 / 5e-197 / 2e-31 / 5e-6 reference; segment 0
at 1 / 0.5 deg 0 / 4e-59 per-angle, 5e-43 / 0.046 reference. Sub-degree frame errors
are seen by the per-angle and reference column tests, not reliably by the plan tests.)

**Whole-array mutants are not the whole story: the first version of these tests missed
partial ones.** The mutants above replace a segment's entire frame or fan. An
independent review found that a kernel reading only *some components* from segment
0 — the frame origin's y, the rotation's v row and the fan's v axis
(`segs.origin[1]`, `segs.rotation[3..5]`, `segs.fan[9..11]`) — passed every one of the
43 arc and segment tests while giving device weights off by up to 0.997 on an arc
with collimator angles 0/45/90: every arc in them rotated about +y, so `v` = +y, the
origin's and focal spot's y and every field's centre and size were the same in every
segment. The physics arc now has a collimator-rotated segment and a rotated,
non-coplanar, off-axis, narrower one; the Dij arc the geometry above. Swept
2026-10-23: the review's mutant, its three reads separately, and each of the 26
per-segment components (origin 3, rotation 9, fan 14) replaced individually by
segment 0's — every one fails the deterministic pins (6-10 of 10 in
`tests/physics/test_warp_segment_arc.py`) and the Dij tests (2-9 of 12). With the
orientations but every fan still on the isocentre at one size, the five field-centre
and width mutants passed both files; the per-fan field closed that. The deterministic
pins are the bit-identity pin above and, in `tests/physics/test_warp_segment_arc.py`,
the arc generator against the per-angle generator row for row, the device
transmission per segment against the host, and a 90-degree segment with one leaf bank
closed (a kernel reading another segment's frame sees the slab edge-on and passes
every ray). Host mutants (template frame in `segment_stack`, segment 0's fan in
`emit`) fail 3-4 unit tests each.

**Measured** (`tests/perf/test_segmented_arc_throughput.py`; RTX 4070 laptop at 2535 MHz
under load, no throttle reason, GPU otherwise idle during these runs but shared with the
coordinator's experiments, best of 5 / 3 interleaved, two sessions; MLC of dense water,
electrons on, random apertures per round, re-targeting by `with_settings` included):

| control points | per-angle runs | one `run_dij` | one plan `run` |
|---|---|---|---|
| 36 x 100k | 141-144 ms | 74-78 ms (1.8-1.9x) | 36-37 ms (3.8-4.0x) |
| 180 x 100k | 645-749 ms | 299-331 ms (2.2-2.3x) | 169-171 ms (3.8-4.4x) |

The plan run is now transport-bound (3.6M histories in 36 ms, mostly rays through
closed leaves ending in the soft-photon roulette); the Dij's extra cost is its dense
per-column maps, fold and finalize. The per-angle route's ~4 ms per launch is its
fixed cost. `examples/segmented_arc_demo.py` shows the same on a conformal 36-point arc.

**Not done here.** The edge-gradient route (`SegmentedEdgeSource`, then not on develop;
since ported, see "Arc edge sets" above) needed, to consume per-segment frames: read `segs.origin`/`segs.rotation` at `seg * 3`/`seg * 9`
(or its own per-segment arrays filled from `source.frames`) for the edge-ray chords,
take each edge's ray from `source.fans[seg]` (focal point and aperture through
`segs.fan[seg * FAN_STRIDE ...]`), and drop the fan scalars from its launch the same
way. Per-control-point history allocation still equal in `run_dij` and multinomial by
MU in the plan view (as for one fan). Collimator rotation with a single shared fan is
expressible (per-segment frames, one fan) but the axis contract then forces the frames
to the fan's axes; a rotated collimator needs a fan per segment.

## Aperture importance sampling of segments, 2026-10-29

**Why.** A `SegmentedFanSource` segment samples its fan's whole field `F` uniformly and
carries the stack transmission as its weight. On the TG-119 direct-aperture study the
open aperture was 8-19 % of the fan area (IMRT segments 2-8 %), so 80-92 % of the
forward histories were transported in full at 1-2 % weight, and forward-dose noise
limited the optimization.

**What.** Opt-in, `SegmentedFanSource(..., aperture_sampling=delta, aperture_margin=m)`
(also `from_stacks`; `with_settings` keeps both and rebuilds the tables: +0.2 ms at
10 segments x 20 pairs, +0.8 ms at 180 x 60, vectorized). Segment `s` draws its
in-aperture offset from the defensive mixture `q_s = delta U(F) + (1 - delta) U(A_s)`
(Hesterberg 1995) and multiplies the weight by `p / q_s = 1 / (delta + (1 - delta)
|F| / |A_s| 1_A)`. Columns stay "dose per fan history"; no normalization changes; the
weight is bounded by `1 / delta`; any `A_s` inside `F` is unbiased, so this is variance
reduction, not an approximation. The default `None` is the old four-uniform draw: 164
seeded arrays saved before the change (host `emit` and batches, reference Dij, Warp cpu
and cuda Dij correlated and independent, plan doses, energy books, a rotated frame and
an arc) are `np.array_equal` after it.

**The region.** In the fan's aperture plane, by the per-segment contract (frame origin
on the focal point, aperture axes = frame `u`/`v`, now required on the single-fan path
too when sampling): a mid-plane setting `t` of device `j` maps to the offset
`t c_w / z_j - c_u` (`c` = field centre in frame coordinates). `A_s` = each leaf row's
`[tip_neg - m, tip_pos + m]` over its strip, cut by the jaw openings, clipped to `F`;
outside the bank no MLC constraint; an empty `A_s` falls back to uniform. Bands along
`v` at every mapped strip and jaw edge: a fixed band count per stack structure. Two
refinements of the original design, both derived:

- **Closed pairs keep their margin band.** Touching rounded tips leave a cusp: a ray at
  distance `x` from the contact cuts only `2 sqrt(2 R x)`, so a closed pair transmits
  like an open gap of `1 / (2 mu^2 R)` (0.08 cm at mid-plane for tungsten at 2 MeV,
  `R` = 8 cm), the stripe visible in `segmented_fan_demo.png`. Excluding closed rows
  would put that fluence at weight `1 / delta`; a back-of-envelope second moment for a
  2-row segment among 38 closed rows came out 5x *worse* than uniform.
- **Default margin from the tip geometry**: `R - sqrt(R^2 - (H / 2)^2)` per MLC (0.58 cm
  at `R` = 8, `H` = 6), beyond which a ray parallel to `w` cuts the full leaf. Jaw and
  leaf-side edges are taken exactly (focused: exact for focal rays; unfocused: the band
  they let through is below `exp(-mu H / 2)`).

**One sampler, five uniforms.** `geometry.fan.fan_primary_aperture` (spectrum bin,
within-bin, component by `select_by_cdf` over the per-segment shares with the whole
field first, then `u` and `v` in the chosen rectangle) serves `emit`, the vectorized
batch (`PCG64(seed)` advanced to `5 * offset`, pinned row for row to the scalar
sampler), both plan routes and both Warp generators (a runtime branch on the uploaded
flag; the uniform branch is the old code). Correlated columns still replay the same
uniforms, each through its own proposal, so their rays, and with them their
correlation, differ from the uniform route's. Exact `SpectralBeamSource` fans are
required: the sampler replaces the fan's draw, so a fan subclass is refused rather than
silently bypassed.

**Statistical evidence, with power.** Source level (`tests/unit`): `E_q[factor 1_R] =
|R| / |F|` over a 10 x 9 grid, one chi-squared with the full sample covariance; 280 null
pairs give 1.4 % below 0.01 (KS p 0.16); a 1 % inside-weight error is caught at
p < 1e-8 at 400k draws. Device pins: the queue against the host sampler on the replayed
uniforms; kernel mutants (segment 0's table, the next segment's table, factor dropped)
each fail. Transport (`tests/dij`, dense-water collimators at 3.5 mfp, KERMA, 1 x 4 x 8
cm bins, pooled-dose mask, **equal histories**, 1M per column): aperture vs uniform for
the fan, the arc, the host pre-sampling route and the plan view; reference vs Warp at
12k. Null over 30 reseeded pairs: fan 0-1 / 30 per column below 0.01, arc 0 / 30; mean
chi-squared/null 0.89-1.03 (uniform vs uniform 0.91-0.96, the oracle's own slight
conservatism). The pinned-seed fan ratios of 0.53-0.79 are one shared fluctuation of
correlated columns, so power was measured at those seeds: a 3 % inside-weight mutant
fails at p = 0.0096 / 2e-10 / 3e-8 (fan, three open segments), 1e-3 / 2e-4 (arc),
4e-4 / 4e-23 / 1e-5 (host route), 5e-4 (plan); a 5 % wrong component share at 7e-10 /
3e-12. At 12k the reference comparison sees only gross errors (factor dropped: p = 0);
its `emit` is pinned exactly to the scalar sampler instead. On **real tungsten** (6 MV,
2 x 2 cm segment, focused jaws, electrons on), 40 replicate pairs at 2M, `truncation=0`:
aperture / uniform region sums 1.0004 +- 0.0011 (open field), 0.9996 +- 0.0022
(2-20 %), 1.0019 +- 0.0022 (below 2 %), 1.0003 +- 0.0011 (total).

**Truncation is sampler-dependent in the low-dose tail.** With the default Dij
truncation (1e-3 of the column maximum) the same replicates give a below-2 % ratio of
0.9946 or 1.0130 +- 0.0026 for two definitions of the below-2 % mask (the sign and size
of the shift follow the mask, not the sampler alone): thresholding a noisy per-voxel
estimate is biased, and the bias depends on the per-voxel noise
distribution, which the sampler changes. Not a defect of either sampler, and nothing
was changed; but a uniform-vs-aperture comparison in the tail must use `truncation=0`,
and the AGENTS.md section 2.8 caveat on truncation applies with a sampler-dependent
sign.

**Measured.** RTX 4070 laptop at 2535 MHz under load, no throttle reason, GPU shared but
otherwise idle; interleaved best of 5; 6 MV Ali-Rogers, tungsten (EPICS), 12 x 10 x 12
cm water at 0.25 x 0.25 x 0.5 cm, electrons on, 1M histories per column; sigma = the
region RMS of the batch sigma relative to the column maximum; FOM gain =
`sigma_u^2 t_u / (sigma_a^2 t_a)`. Segments 1 x 1 / 2 x 2 off-axis / 5 x 5 cm in a
10 x 10 cm fan:

| setup | delta | margin (cm) | `|A|/|F|` | FOM gain, open field | FOM gain, 5-20 % |
|---|---|---|---|---|---|
| focused jaws tracking the segment | 0.1 | default 0.58 | 0.04 / 0.08 / 0.31 | 27 / 12 / 3.6 | 24 / 11 / 2.5 |
| | 0.1 | 0.25 | 0.03 / 0.07 / 0.30 | 37 / 16 / 3.8 | 26 / 13 / 2.7 |
| | 0.05 / 0.2 / 0.3 | 0.25 | | 41 / 34 / 30 (1 x 1) | 22 / 26 / 25 (1 x 1) |
| no jaws (closed pairs meet in the field) | 0.1 | default 0.58 | 0.23 / 0.26 / 0.47 | 5.1 / 4.2 / 2.5 | 4.3 / 3.8 / 1.9 |
| | 0.1 | 0.25 | 0.10 / 0.13 / 0.34 | 10.5 / 7.7 / 3.3 | 8.1 / 6.6 / 2.4 |
| | 0.1 | 0.1 | 0.05 / 0.08 / 0.29 | 21 / 12 / 4.0 | 2.6 / 3.4 / 2.4 |

Wall time per history is neutral on the GPU (ratio 0.97-1.04). The gain tracks the
region fraction; `delta` matters little between 0.05 and 0.3. A margin inside the tip
band (0.1 cm) helps the open field but loses the low-dose region, where tip and cusp
leakage then arrive at weight `1 / delta`. Jaws that track the segment matter: without
them every closed pair's band is sampled. `examples/aperture_sampling_demo.py` (2 x 2
cm, 2M, jaws): sigma ratio 0.29 open field, 0.29 at 2-20 %, 0.38 below 2 %; gain x12 /
x12 / x7.

**Not done.** The default margin stays geometric (material-free); 0.25 cm measured
better for tungsten at 6 MV in both regions and both setups, but on one MLC geometry.
No adaptive `delta`. The edge-gradient route (`SegmentedEdgeSource`) does not see the
option.

## First-order dose steps of a segmented fan, 2026-11-01

`SegmentedEdgeStepSource` (`geometry/segment_edge_steps.py`) estimates
`D_s(x + d) - D_s(x) = sum_e d_e dD_s/dx_e`, the derivative taken at the step's midpoint, from
the edge estimators of `SegmentedEdgeSource`: a history picks an edge with probability
`|d_e| / sum |d_e|` and carries the edge's ray times `sum |d_e|`. Histories then follow the
area the edges swept, not the aperture's. An optimizer keeps its segment doses and adds the
step, instead of re-simulating the plan each iteration.

**Two columns per segment.** Opening edges in `2 s`, closing ones in `2 s + 1`, each signed so
that its weights are non-negative, because the Dij's compaction keeps positive voxels only
(AGENTS 2.8 stays untouched). The step is the difference of the two columns.

**Batches.** The kernel stratifies the edge choice over a batch's histories of the column, not
over the column's: stratifying over the column gave each batch a slice of the edges and
inflated the batch sigma 30-fold on the test fixture, with the mean intact. `emit` draws
independently; both are unbiased.

**Validation.** Column contractions against the sum of the edges' own contractions times their
travel (`tests/dij/test_segmented_edge_steps_warp.py`; a 5 % factor error fails the oracle, and
a probe at 3-4M histories per column stays within |z| < 1.4), against the reference engine
through `emit`, and against the Dij's own sum (no voxel lost to compaction).

**On TG-119** (a direct-aperture study outside this repository, RTX 4070, seed 1 of the 4 deg finish of one 8 deg start, plans
scored on fresh 8M / 32M runs of the default engine, so that a biased carried dose cannot
flatter them): 30 iterations re-simulating 2M per segment 5320 at a forward of 2.9 s; carried
from an 8M start, 300k / 100k / 30k per column 5200 / 5222 / 5250 at 1.55 / 0.75 / 0.40 s. So
the noise of the start, frozen for the whole run, costs more than the steps' does: a 2M
start with 100k steps gives 6009. A full re-run half way adds nothing (5401).

**Not done.** Moves of jaws are not in the step (a jaw that tracks the leaves clears them by a
margin, so its effect is attenuation only); no refresh policy.
