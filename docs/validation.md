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
| Fano test in a field, density ratios 0.1 / 0.01 / 0.001 (0.2 and 2 cm gaps), 0.5 and 1 MeV, 0.35 and 1.5 T | Fano theorem (Bouchard et al. 2015) | the field moves the gap-to-wall dose per mass by < 1 % | moved by <= 0.4 % at a tenth and a hundredth, <= 0.8 % in a 2 cm air gap; in a 0.2 cm air gap the field shrinks the field-free thin-gap residual (0.98 -> 0.995), held to that residual's 3 % gate |
| Fano test without a field, same phantoms, shipped substep fraction 0.05 | Fano theorem | gap-to-wall within 1 % at a tenth of the wall density, within 3 % down to air density | 2 cm gaps 0.996-1.001 at every contrast; 0.2 cm air gap 0.979 / 0.975 (0.5 / 1 MeV) |
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

### Magnetic-field dose is not yet gated against published benchmarks

The Lorentz step is pinned against the analytic helix, against the reference engine
with the field on (including the tabulated spectral thorax at 1.5 T) and by the Fano
test in a field at lung-like density ratios, and the electron return effect comes out
with the right sign and shape. What is missing is a quantitative gate against an
independent code or measurement: Raaijmakers' film phantoms (PMMA-air-PMMA, 0.6 and
1.3 T), the water-lung-water interface excursions of Ahmad et al. and Li et al. at
1.5 T, and EGSnrc's EEMF slab example. The maintainer is collecting those curves and
TOPAS/EGSnrc runs of the example geometry; until they land, treat magnetic-field dose
as *validated for internal consistency*, not against the literature.

Two standing caveats. Outside the transport grid there is no field, so an MR-linac
setup must extend the grid with air voxels to the phase-space plane. And at a
**thousandfold** density contrast (water / air) the field-free engine satisfies Fano
only to about 2 % at the shipped substep fraction, and only for a *thin* gap: measured
2026-10-18, gap-to-wall dose per mass 0.98 +- 0.003 for a 0.2 cm air-density gap at
0.5 MeV (ECUT 0.05 as the test instrument), -0.9 % at a hundredfold contrast, +0.9 % at a
tenth, while a 2 cm gap is within 0.4 % at every contrast; the thin gap's residual halves
at a substep fraction of 0.03 and is within 1 % at 0.02. That residual is step-size convergence of the
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
