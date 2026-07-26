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

Under the shipped default, per-column Dij sigmas may **not** be combined in quadrature to
form a plan-dose sigma. A valid plan-dose sigma requires per-batch scoring, which
`DijResult` does not carry and therefore does not expose. `variance_csc()` exports
per-column variance and warns.

This is a correctness trap rather than an inaccuracy: the individual numbers are right,
and combining them the obvious way is wrong.

### Soft-spectrum columns are not gated against the analytic backend

Below ~1 MeV the analytic cross-section layer is an explicitly few-percent
parameterization (Klein-Nishina without binding, a crude photoelectric power law, no
Rayleigh). Use the tabulated backend where the soft spectrum matters.

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
