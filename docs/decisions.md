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
