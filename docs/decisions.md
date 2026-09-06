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

**Each electron substep is capped at the next voxel face.** Without it a step carried one
medium's stopping power across a boundary into the neighbour's mass, producing a
single-voxel interface spike — **1.95x before, under 1.4x after** at a pure density step.
Residual interface structure is second-order (the hinge can still deflect the short
post-hinge segment across the face) plus genuine interface dosimetry.

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

**Flipping a validated default re-rolls every fixed-seed statistical test.** Expect around
a 1 % trip rate across the suite, and repair those by fixing the oracle, never by
loosening the tolerance.
