# pyRadMC

Fast photon Monte Carlo dose engine for radiotherapy treatment planning.

> **Status: Phase 4 complete, pre-alpha; Phase 5 underway — the phase-space source,
> the tabulated cross-section backend, multi-material data (ICRP media), a CT image
> adapter, and the decoupled dose grid (arbitrary/subregion scoring grids with
> selectable dose-to-water) are done; the pyRadPlan adapter is the last remaining
> workstream.**
> The engine produces its primary
> product: a **beamlet-resolved dose influence matrix (Dij)** — sparse CSC columns
> with a per-entry statistical uncertainty, computed on CPU and CUDA by tagging
> every history's whole secondary family with its beamlet of origin. Because RNG
> streams are pure functions of (seed, history) and scoring is associative
> fixed-point, *how* beamlets are scheduled (grouping, chunking, batch merging)
> cannot change the matrix by one bit — test-pinned, along with per-column
> statistical equivalence to the reference oracle and a DVH-endpoint
> certification of the default column truncation. Measured end-to-end on a
> laptop RTX 4070 at planning statistics: ~1.1e7 histories/s, i.e. a 100-beamlet
> 6 MeV field at 2–3 % per-beamlet sigma in single-digit seconds. Particles
> carry statistical weights (soft photons play an unbiased Russian roulette; the
> EGSnrc validation gates pass with it active). Correlated sampling across
> beamlets is now the shipped configuration (Phase 4): keying each beamlet's
> histories on the same random streams roughly halves the renormalized
> plan-dose error at a given per-beamlet uncertainty, established by a
> recalculation-and-renormalize study over optimized toy plans in water and
> through a heterogeneity. Compton splitting was built and measured but ships
> off: it does not earn its keep for the analytic-water Dij and cannot reach the
> low-dose tail, so the mechanism is retained (tested) for Phase 5 phase-space
> sources. Do not use for anything clinical, now or later, without independent
> validation.

## Why

Existing photon Monte Carlo engines are, broadly, one of three things: licensed
incompatibly with open-source projects, too slow for treatment-plan optimization, or
closed-source. pyRadMC is an attempt at the fourth thing: permissively licensed, fast
enough for beamlet-resolved planning, and readable.

## What it does

The primary product is a **beamlet-resolved dose influence matrix (Dij)** for photon
IMRT/VMAT planning, computed on CPU and CUDA GPU from a single physics source.

Design commitments:

- **DPM-class physics.** Macroscopic condensed-history electron transport with a random
  hinge; Woodcock delta tracking for photons. Tuned for 1 to 20 MeV in low-Z media. It
  is not a general-purpose Monte Carlo code.
- **Single-source, dual-target.** Physics routines are pure, scalar, allocation-free
  functions compiled to CPU and CUDA through [Warp](https://github.com/NVIDIA/warp)
  (Apache-2.0). A pure-NumPy reference backend serves as the correctness oracle.
- **Planning-grade statistics.** Approximately 2 to 3 percent per-beamlet uncertainty,
  achieved through variance reduction rather than brute force.
- **Data behind an interface.** Analytic cross-sections now, tabulated later, without a
  rewrite.

## What it does not do

- No denoising of the Dij. The optimizer exploits statistical noise; it exploits
  denoising bias too, and the latter is spatially correlated and therefore worse.
- No general geometry. Rectilinear voxel grids only.
- No non-photon primaries.
- No pyRadPlan dependency in the core. An adapter comes later, separately.

## Quick look

![Phase 0: depth-dose curves and dose maps for a broad beam and a pencil beam](examples/phase0_reference_engine.png)

Each phase ships a runnable example (AGENTS.md 7.1). The Phase 0 one transports 2 MeV
photons in water on the reference backend and renders a depth-dose curve and a dose map
for each of two beams: the broad beam (top) pulls away from the primary-only
exponential with depth — scatter buildup — while the pencil beam's narrow axis (bottom)
hugs the same exponential, its scatter visible instead as the halo in the log-scale map:

```bash
pip install -e ".[dev,examples]"
python examples/phase0_reference_engine.py
```

![Phase 1: electron buildup vs the KERMA approximation, and electron-beam depth doses](examples/phase1_electron_transport.png)

Phase 1 added Class II condensed-history electron transport
(`examples/phase1_electron_transport.py`): the left panel contrasts a 6 MeV photon
beam with electrons transported against the same beam in the KERMA approximation —
the buildup region is the difference — and the right panel shows electron-beam depth
doses whose R50 tracks the CSDA range.

![Phase 2: one physics source compiled three ways, and the throughput gap](examples/phase2_warp_backend.png)

Phase 2 shipped the production Warp backend (`examples/phase2_warp_backend.py`): the
same 6 MeV beam computed by the reference interpreter and by the Warp compilation of
the *identical* physics source on CPU and CUDA. The depth-dose curves agree within
their error bands; the bars show why the backend exists.

![Phase 3: one Dij column, the fluence-sum identity, and a wedge plan recombined from the matrix](examples/phase3_dij.png)

Phase 3 shipped the Dij itself (`examples/phase3_dij.py`): one beamlet's dose column
with the truncated tail visible (left), the open field recombined from the columns at
unit weights against an independently simulated open field (middle — columns
partition the field exactly), and the point of the whole exercise (right): a wedge
plan is `Dij @ weights`, no re-simulation, which is the loop a treatment-plan
optimizer runs thousands of times.

The engine API in three lines (swap in `WarpEngine(grid=..., cross_sections=...,
device="cuda:0")` for the production backend — the `run` signatures are identical):

```python
grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
result = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
    source, n_histories=80_000, n_batches=10, seed=20260711
)
# result.dose, result.dose_sigma  (MeV/g per history, batched 1-sigma)
```

## Backends

| Backend | Role | Devices |
|---|---|---|
| `ref` | correctness oracle; never optimized | CPU (NumPy) |
| `warp` | production | CPU, CUDA |

Warp is CUDA-only for GPU. If vendor-neutral GPU becomes a requirement, the physics
layer is framework-agnostic and a Taichi backend is a port, not a rewrite.

## Install

```bash
pip install -e ".[dev]"           # reference backend + tooling
pip install -e ".[warp,dev]"      # production backend (Phase 2+)
pip install -e ".[dev,examples]"  # adds matplotlib for the phase examples
```

## Develop

```bash
make hooks    # once per clone: pre-commit gate (lint, format, types, fast tests)
make test     # fast tiers: unit, physics, integration, dij
make test-all # adds validation (slow) and perf
make lint
make types
```

Read [`AGENTS.md`](AGENTS.md) before contributing. It is the governing document, and it
is written for both human and agentic contributors. Two rules from it are worth
repeating here because they are the ones people break:

1. **The `ref` backend is the oracle.** Never adjust it to match a faster backend.
2. **CPU and GPU results are never bit-identical.** Atomic float accumulation is
   non-associative. Test statistically or not at all.

## Roadmap

| Phase | Content |
|---|---|
| 0 ✅ | Reference photon transport (KERMA approximation), interfaces, test scaffold |
| 1 ✅ | Reference condensed-history electron transport; PDD validation gate¹ |
| 2 ✅ | Warp backend, CPU and CUDA from one source² |
| 3 ✅ | Beamlet tagging, batched Dij assembly, basic variance reduction³ |
| 4 ✅ | Correlated sampling; study of per-beamlet noise vs. optimized-plan bias⁴ |
| 5 🚧 | Tabulated data ✅, phase-space source ✅, multi-material media ✅, CT image adapter ✅, decoupled dose grid + dose-to-water ✅⁵, pyRadPlan adapter (last) |

¹ The nightly validation tier gates against NIST ESTAR ranges and against
maintainer-supplied EGSnrc depth-dose curves (1, 2 and 6 MeV; gamma 5%/3mm). The
Phase 5 tabulated backend (EPDL photons with coherent form factors, EEDL scattering,
ICRU-37 stopping) passes the same gate; the tighter 2%/2mm criterion is limited by the
benchmark's geometry/metadata (finite field, 6 mm scoring tube), not the cross-section
data, so it awaits a fully specified benchmark (see `AGENTS.md` 7.2). See
`examples/phase5_tabulated_demo.py` for the depth-dose comparison.

² Both Warp devices reproduce the reference within the chi-squared detection
oracle; GPU throughput cleared the 1e6 histories/s criterion at ~1.6e7 on a laptop
RTX 4070. The third exit criterion offered "Warp-CPU within 2x of numba-CPU, or the
numba backend is dropped"; the maintainer dropped the (never-built) numba backend at
exit. The plan's Taichi note remains the fallback if Warp-CPU ever becomes the
bottleneck.

³ Exit measured on the same hardware: a 1x1-beamlet Dij column reproduces the
open-field run bit for bit per device; every Dij column is chi-squared-consistent
with the reference under full coupled transport; beamlet grouping, chunking and
batch merging are bit-inert (scheduling cannot change the matrix); the default
column truncation (1e-3 of the column maximum) moves the DVH endpoints D2/D50/D98
by well under 0.5 percent — certified against DVH endpoints, never a matrix norm.
Pipeline throughput at planning statistics: ~1.1e7 histories/s end-to-end (the
perf tier asserts a 2e6 floor; the pipeline carries a fixed host-side cost, so it
is only meaningful to benchmark at realistic history counts). Basic variance
reduction shipped as the statistical-weight infrastructure plus Russian roulette
of sub-0.5-MeV photons (annihilation photons sit just above and stay analog) —
unbiasedness is test-pinned and the validation gates
pass with it active; measured efficiency on the GPU is neutral (kill savings
disappear under warp divergence), and the weight machinery is kept because
correlated sampling (Phase 4) and weighted phase-space sources (Phase 5) are
built on it.

⁴ Correlated sampling across beamlets (keying each beamlet's histories on the
same within-beamlet stream index) ships as the default Dij configuration, with
the independent mapping demoted to a test instrument. The decision rests on a
recalculation-and-renormalize study (`examples/phase4_noise_bias_study.py`,
using the toy fluence optimizer in `pyRadMC.study`): optimizing on a noisy Dij
and scoring on a second independent ground truth, across six statistics levels
to the 2% target sigma in water and a heterogeneity, correlated sampling roughly
halves the renormalized plan-dose error at matched per-beamlet sigma. Compton
splitting at primary sites was implemented on both backends and measured, then
shipped **off** (`PHOTON_SPLIT_N = 1`): its variance-reduction figure of merit
is below 1 on the reference CPU and neutral on the GPU, and it does not reach the
low-dose tail, so the tested mechanism is retained for the Phase 5 phase-space
source rather than run now. Roulette stays always-on. Carried into Phase 5: a
per-batch plan-dose sigma (correlated columns forbid a quadrature one), built
with the adapter that consumes it.

⁵ Dose is accumulated on a **scoring grid decoupled from the transport (CT)
grid** (`pyRadMC.scoring.grid.ScoringGrid`; `scoring_grid=` on both engines'
`run`/`run_dij`): any origin/spacing/shape, including subregions — deposits the
dose grid does not cover go to an *unscored* energy-ledger bucket, never a
clamped edge voxel, and `emitted == deposited + unscored + escaped` stays exact.
Per-voxel mass is rebinned from the CT by exact separable voxel overlap, which
also defines dose-to-medium for mixed voxels. Scoring a 2 mm CT on a 3 mm dose
grid shrinks the Dij and the per-group device buffers by ~0.30x with better
per-voxel statistics; transport physics is invariant (test-pinned to the bit
per device). `scoring_mode="dose_to_water"` additionally weights each deposit
by the restricted collision stopping-power ratio water/medium at the depositing
particle's energy (Siebers et al. 2000) — a scoring-output selection, not a
physics toggle: transport and the energy books are identical in both modes.
See `examples/dose_grid_demo.py`.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
