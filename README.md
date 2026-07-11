# pyRadMC

Fast photon Monte Carlo dose engine for radiotherapy treatment planning.

> **Status: Phase 2 complete, pre-alpha.** The production Warp backend runs the full
> coupled photon-electron physics on CPU and CUDA **from the same physics source**
> the pure-NumPy reference interprets, validated against that reference with the
> chi-squared detection oracle on both devices. Scoring is fixed-point, so one
> device + one seed is bit-reproducible even on CUDA. Measured throughput for a
> 6 MeV broad beam in water with coupled transport: ~1.6e7 histories/s on a laptop
> RTX 4070 (the 1e6/s exit criterion with an order of magnitude to spare) and
> ~4e5/s single-threaded on Warp-CPU. No Dij yet (Phase 3). Do not use for
> anything clinical, now or later, without independent validation.

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
| `numba` | optional CPU cross-check — never built; drop/keep decision pending² | CPU |

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
| 3 | Beamlet tagging, batched Dij assembly, basic variance reduction |
| 4 | Correlated sampling; study of per-beamlet noise vs. optimized-plan bias |
| 5 | Tabulated data, phase-space source, pyRadPlan adapter |

¹ The nightly validation tier gates against NIST ESTAR ranges and against
maintainer-supplied EGSnrc depth-dose curves (1, 2 and 6 MeV; gamma 5%/3mm — the
criterion is limited by the analytic cross-sections, and Phase 5 must pass the same
data at 2%/2mm over the full depth range).

² Both Warp devices reproduce the reference within the chi-squared detection
oracle; GPU throughput cleared the 1e6 histories/s criterion at ~1.6e7 on a laptop
RTX 4070. The third exit criterion ("Warp-CPU within 2x of numba-CPU, or the numba
backend is dropped") is a maintainer decision: no numba backend was ever built, and
building one solely to lose to it would be the tail wagging the dog — the
recommendation on record is to drop it and keep the plan's Taichi note as the
fallback if Warp-CPU ever becomes the bottleneck.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
