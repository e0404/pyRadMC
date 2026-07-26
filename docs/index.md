# pyRadMC

Fast photon Monte Carlo dose engine for radiotherapy treatment planning.

!!! warning "Pre-1.0 research software"

    The physics, both backends and the influence-matrix pipeline are complete and
    test-pinned, and the API is stable enough to build on. But this is young software,
    and the validation it has passed is stated precisely on the
    [validation status](validation.md) page rather than claimed broadly.

    **Do not use it for anything clinical, now or later, without independent validation.**

## What it is

The primary product is a **beamlet-resolved dose influence matrix (Dij)** for photon
IMRT/VMAT planning: sparse CSC columns with a per-entry statistical uncertainty, computed
on CPU and CUDA from a single physics source by tagging every history's whole secondary
family with its beamlet of origin.

Because RNG streams are pure functions of `(seed, history)` and scoring is associative
fixed-point, *how* beamlets are scheduled — grouping, chunking, batch merging, which GPU
gets which column — cannot change the matrix by one bit. That is test-pinned, along with
per-column statistical equivalence to the reference oracle and a DVH-endpoint
certification of the default column truncation.

Measured end to end on a laptop RTX 4070 at planning statistics: **~1.1e7 histories/s**,
i.e. a 100-beamlet 6 MeV field at 2–3 % per-beamlet sigma in single-digit seconds.

## Why it exists

Existing photon Monte Carlo engines are, broadly, one of three things: licensed
incompatibly with open-source projects, too slow for treatment-plan optimization, or
closed-source. pyRadMC is an attempt at the fourth thing: permissively licensed, fast
enough for beamlet-resolved planning, and readable.

## Design commitments

- **DPM-class physics.** Macroscopic condensed-history electron transport with a random
  hinge and a Goudsmit-Saunderson multiple-scattering model (the shipped default);
  Woodcock delta tracking for photons. Tuned for 1 to 20 MeV in low-Z media. It is not a
  general-purpose Monte Carlo code.
- **Single-source, dual-target.** Physics routines are pure, scalar, allocation-free
  functions compiled to CPU and CUDA through [Warp](https://github.com/NVIDIA/warp). A
  pure-NumPy reference backend is the correctness oracle and is never optimized.
- **Planning-grade statistics** through variance reduction rather than brute force.
- **Cross-sections behind an interface**, so analytic and tabulated data are the same
  engine and the same API.

## What it deliberately does not do

- No denoising of the Dij. The optimizer exploits statistical noise; it exploits
  denoising bias too, and the latter is spatially correlated and therefore worse.
- No general geometry. Rectilinear voxel grids only.
- No non-photon primaries as a clinical modality.
- No treatment-planning system in the core.

## Two things to know before you trust a number

1. **Under correlated sampling — the default — Dij column sigmas must never be combined
   in quadrature** to form a plan-dose sigma. The columns are statistically dependent.
   See [validation status](validation.md).
2. **CPU and GPU results are never bit-identical.** Atomic float accumulation is
   non-associative. Bit-reproducibility holds *per device*, for a given seed.

Start with [getting started](getting-started.md).
