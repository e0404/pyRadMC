# pyRadMC

Fast photon Monte Carlo dose engine for radiotherapy treatment planning.

> **Status: Phase 0, pre-alpha.** Interfaces and test scaffold only. No physics yet.
> Nothing here computes dose. Do not use for anything clinical, now or later, without
> independent validation.

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

## Backends

| Backend | Role | Devices |
|---|---|---|
| `ref` | correctness oracle; never optimized | CPU (NumPy) |
| `warp` | production | CPU, CUDA |
| `numba` | optional CPU cross-check | CPU |

Warp is CUDA-only for GPU. If vendor-neutral GPU becomes a requirement, the physics
layer is framework-agnostic and a Taichi backend is a port, not a rewrite.

## Install

```bash
pip install -e ".[dev]"           # reference backend + tooling
pip install -e ".[warp,dev]"      # production backend
```

## Develop

```bash
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
| 0 | Reference photon transport (KERMA approximation), interfaces, test scaffold |
| 1 | Reference condensed-history electron transport; PDD validation gate |
| 2 | Warp backend, CPU and CUDA from one source |
| 3 | Beamlet tagging, batched Dij assembly, basic variance reduction |
| 4 | Correlated sampling; study of per-beamlet noise vs. optimized-plan bias |
| 5 | Tabulated data, phase-space source, pyRadPlan adapter |

## License

Apache-2.0. See [`LICENSE`](LICENSE).
