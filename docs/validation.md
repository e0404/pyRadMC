# Validation status

Read this before trusting a number. It states what has been gated and against what, and
names the places where the engine is *not* yet validated — those are more useful to you
than the passes.

## What is gated

| Quantity | Reference | Criterion | Measured |
|---|---|---|---|
| Photon mass attenuation, water | NIST XCOM | 1 % | ~0.3 % |
| Electron collision stopping power | ICRU-37 / NIST ESTAR | ESTAR-exact by construction | — |
| CSDA ranges | NIST ESTAR | gated | — |
| Central-axis depth dose, water | EGSnrc (maintainer-supplied) | **5 %/3 mm gamma** | passes |
| Backend vs backend, device vs device | reference oracle | chi-squared detection oracle | passes |
| Dij column truncation | DVH endpoints D2/D50/D98 | < 0.5 % | passes |
| Charged-particle gyration in a magnetic field | analytic helix (Bielajew 1988) | radius within 1.5 %, handedness, straight along B | 0.1 % |
| Backend vs backend, field on (water slabs, electron beam, tabulated spectral thorax at 1.5 T) | reference oracle | chi-squared detection oracle | passes |
| Fano test in a field, density ratios 0.1 / 0.01 / 0.001 (0.2 and 2 cm gaps), 0.5 and 1 MeV, 0.35 and 1.5 T | Fano theorem (Bouchard et al. 2015) | the field moves the gap-to-wall dose per mass by < 1 % | 2026-10-20, GS fraction 0.20: moved by <= 0.25 % at a tenth and a hundredth, <= 0.7 % in a 2 cm air gap; in a 0.2 cm air gap the field shrinks the field-free residual (0.971 / 0.979 -> 0.995), held to that residual's 3 % gate |
| Fano test without a field, same phantoms, shipped GS substep fraction 0.20 | Fano theorem | gap-to-wall within 1 % at a tenth of the wall density, within 3 % down to air density | 2026-10-20, chunk-invariant source: 2 cm gaps 0.997-1.000 at every contrast; 0.2 cm air gap 0.9706 / 0.9785 (0.5 / 1 MeV) |
| Exit-side ERE vs field strength, 10 x 10 cm2 6 MV on 20 x 20 x 8 cm3 water (Raaijmakers et al. 2008, Geant4 + film-verified setup) | literature: excess 40 / 52 / 42 / 25 % of dmax at 0.2 / 0.75 / 1.5 / 3 T, 20 % shift 17.5 / 9.5 / 4 mm, build-up 15 / 12 / 7 mm | excess within 30 % at 0.75-3 T with the 0.75 > 1.5 > 3 T ordering; shift within 4 mm; 1.5 T build-up within 2 mm | excess 0.42 / 0.63 / 0.50 / 0.29 (ratios 0.79 and 0.58 vs 0.81 and 0.60); shift 15 / 7.5 / 2.5 mm; build-up 14.8 / 12.2 / 12.2 mm |
| Fano test, 0.2 cm air gap, substep fraction 0.02, 0 and 1.5 T | Fano theorem | gap-to-wall within 1.5 %, field moves it < 1.5 % | 0.998 / 0.994 at 0 T, 0.992 / 0.996 at 1.5 T (0.5 / 1 MeV, +- 0.3 %) |
| Electron return effect, water / light gap / water | qualitative sign (Raaijmakers 2005) | gain before the gap, loss behind it, lateral push toward the Lorentz force | passes |

Variance reduction is unbiased **by construction** and test-pinned as such against a
roulette-free run; the validation gates pass with it active.

## What is not gated, and why

### The depth-dose criterion is 5 %/3 mm, not 2 %/2 mm

This is the single most important limitation on this page.

2 %/2 mm was measured against the available EGSnrc benchmark and is **not reachable** —
and the limiter is the benchmark, not the cross-section data. The reference file lacks
its EGSnrc transport parameters (ECUT/PCUT, Rayleigh on/off, brems model), its exact
cylinder geometry, and its source normalization.

The evidence that this is a benchmark problem rather than a data problem: the tabulated
backend barely differs from the analytic one on this gate — the analytic photon *total*
was already NIST-calibrated, so a more accurate channel split moves the depth-dose shape
very little — and *widening* the lateral phantom toward the infinite-field limit makes
the deep-tail residual **worse**, which shows the lateral-integrated pencil beam
overestimating the benchmark's finite-field, 6 mm-tube scoring at depth.

A genuine 2 %/2 mm gate needs a fully specified benchmark. Do not loosen the 5 %/3 mm
gate, and do not expect a different data layer to close the gap.

### Correlated columns are statistically dependent

Under the shipped default, per-column Dij sigmas combined in quadrature
(`variance @ weights**2`) give a **lower bound** on a plan's noise, not an estimate of
it. Shared streams make corresponding histories of neighbouring beamlets translates of
each other, so their deposits in one voxel are positively correlated; that covariance
inflates the variance of a sum exactly as it deflates the variance of a difference, and
the quadrature cannot see it. Measured against the seed-to-seed spread of the plan dose
(`examples/dij_plan_sigma_demo.py`, 6 MeV, water, uniform weights), the true plan variance
exceeds the quadrature by a factor of 1.02 at 2 cm beamlets, 1.07 at 1 cm and 1.4 at
0.5 cm. The same quadrature from an *independent* Dij (`correlated=False`) is exact at
every pitch and matches the batch sigma of a direct forward run. `variance_csc()` exports
per-column variance and warns on a correlated Dij; the reference engine pins both
statements in `tests/dij/test_dij_correlated.py`.

This is a correctness trap rather than an inaccuracy: the individual numbers are right,
and the numerically identical quadrature means something different in each mode. A
plan-dose sigma from batch-resolved scoring, which `DijResult` does not carry, would be
exact in either; the choice among the routes belongs to the planning adapter.

### Soft-spectrum columns are not gated against the analytic backend

Below ~1 MeV the analytic cross-section layer is an explicitly few-percent
parameterization (Klein-Nishina without binding, a crude photoelectric power law, no
Rayleigh). Use the tabulated backend where the soft spectrum matters.

### Magnetic-field dose is gated against one published benchmark

The physical gates below use **uniform fields**. Nearest and trilinear map lookups
are tested on host and device, but nonuniform-field dose is not independently
validated. The field is sampled once per electron substep, with no limiter at a
field-map cell boundary or on its gradient. A covering map can therefore still be
under-resolved: a coarse transport voxel can skip a localized nonzero field region
when the substep starts in a zero-field region. Refine the transport grid and check
step-size convergence before using nonuniform maps for dose calculations.

The Lorentz step is pinned against the analytic helix, against the reference engine
with the field on (including the tabulated spectral thorax at 1.5 T) and by the Fano
test in a field down to air density, and since 2026-10-18 against one published
benchmark: the exit-side electron return effect of Raaijmakers et al. (2008, PMB 53,
909) as a function of field strength (`tests/validation/test_ere_field_strength.py`).
On their phantom and field the central-axis dose excess in the last 0.5 mm of water is
0.42 / 0.63 / 0.50 / 0.29 of the field-free dmax at 0.2 / 0.75 / 1.5 / 3 T against
their 40 / 52 / 42 / 25 %, with their field-strength ratios reproduced to a few per
cent, the lateral 20 % shift 15 / 7.5 / 2.5 mm against 17.5 / 9.5 / 4 mm, and the
build-up depth 14.8 / 12.2 mm at 0.75 / 1.5 T against 15 / 12 mm. Two things in that
comparison deserve a note. The effect is as large at low field as at 1.5 T for a broad
beam, with the maximum at 0.75 T, exactly as they report: everything below a few MeV
turns round whatever the radius, and what a weaker field changes is *where* it lands —
one gyration diameter to the Lorentz side, centimetres at 0.2-0.35 T, millimetres at
1.5 T — so the lateral skew of an exit profile is wider at the weaker field, not
narrower. And the 3 T build-up depth (12 mm here, 7 mm in the paper) is the one
quantity outside the gate; 3 T is outside the MR-linac range and it is not gated.
Still missing: Raaijmakers' film phantoms (PMMA-air-PMMA, 0.6 and 1.3 T), the
water-lung-water interface excursions of Ahmad et al. and Li et al. at 1.5 T, and
EGSnrc's EEMF slab example; TOPAS/EGSnrc runs of the example geometry would still be
welcome as an independent-code check.

Two standing caveats. Outside the transport grid there is no field, so an MR-linac
setup must extend the grid with air voxels to the phase-space plane. And at a
**thousandfold** density contrast (water / air) the field-free engine satisfies Fano
only to about 2-3 % at the shipped GS fraction 0.20, and only for a *thin* gap: measured
2026-10-20 with the chunk-invariant source, gap-to-wall dose per mass 0.9706 +- 0.0031
at 0.5 MeV and 0.9785 +- 0.0019 at 1 MeV for a 0.2 cm gap (ECUT 0.05 as the test
instrument), while a 2 cm gap is within 0.4 % at every contrast (the 2026-10-18
numbers quoted in docs/decisions.md, 0.978-0.983, were taken at the same shipped
fraction; an earlier draft labelled it 0.05, which is the *Gaussian* hinge's default).
Shortening the substep removes it: 0.987 at 0.10, within 1 % at 0.03 and 0.02. That
residual is step-size convergence of the
random hinge at an interface (docs/decisions.md records the boundary handling it replaced,
which read 2.9x at air contrast and +8 % at a hundredfold, and the Warp energy-loss round
trip that read 0.855 in air under a field at a tight gyration cap); it sits
in the last millimetre before an air face, where the electron return effect peaks, so
the ERE *magnitude* there carries a few per cent while the sign, the lateral push and the
downstream re-build-up do not. The validation tier gates the field's own contribution at
1 % separately from this field-free residual, and pins the convergence at 0.03.

### Absolute CPU variance-reduction cost is unmeasured

The development machine's power state drifts by several times, so only interleaved-median
*ratios* are trustworthy on it. GPU cost is neutral — a warp retires with its longest
thread. Absolute CPU efficiency figures await stable-power hardware.

## Reproducibility

- Results are bit-reproducible **per device**, for a given seed. RNG streams are pure
  functions of `(seed, history)`, so batching and scheduling cannot change a result.
- Results are **never** bit-identical across CPU and GPU. Atomic float accumulation is
  non-associative. Compare statistically or not at all.
- Dij scheduling — beamlet grouping, chunking, batch merging, multi-device sharding — is
  bit-inert. Over a *mixed* device set (e.g. CUDA plus CPU), which device computes which
  column is timing-dependent, so pass a single device where strict run-to-run
  reproducibility matters.
- Every result carries a `RunProvenance` record naming the version, backend, device,
  seed, cutoffs, scattering model and cross-section source.

## Running the tiers yourself

```bash
pytest                 # fast tiers: unit, physics, integration, dij
pytest -m ""           # every tier, adding validation and perf
pytest -m validation   # the validation tier alone
```

The validation tier needs the EPICS libraries; see [cross-section data](data.md). It also
runs weekly in CI, which is what keeps the pinned library digests honest.
