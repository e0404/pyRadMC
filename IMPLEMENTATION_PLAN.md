# Fast Photon Monte Carlo Engine for Beamlet-Resolved Treatment Planning

## Implementation Plan

**Working title:** `pyRadMC` (placeholder; collision check pending)
**License target:** Apache-2.0 or MIT (permissive, to avoid the exact problem motivating this project)
**Primary deliverable:** beamlet-resolved dose influence matrix (Dij) for photon IMRT/VMAT planning, computed on CPU and NVIDIA GPU from a single physics source.

---

## 0. Design decisions fixed by scoping

| Decision | Choice | Consequence |
|---|---|---|
| Hardware | CPU and CUDA GPU, co-equal from v1 | Warp is the production kernel layer; single-source is mandatory, not aspirational |
| Accuracy | ~2-3% per-beamlet statistical uncertainty (high-dose region, 1σ) | Aggressive variance reduction is in scope; denoising is not (see §6) |
| Integration | Standalone package, pyRadPlan adapter later | No pyRadPlan import in the core; geometry and Dij interfaces are plain NumPy/SciPy |
| Physics data | Analytic first, tabulated later | Cross-section access is an interface from commit one, never a hardcoded formula |
| Development | TDD, agentic (Claude Code) | Test oracles must be statistical and cheap; see §7 |

### Revision to the numba → Warp → Taichi sequence

Given co-equal CPU/GPU, the roles change:

- **NumPy/pure-Python reference (`ref` backend):** slow, obviously correct, single-history-at-a-time. This is the oracle against which everything else is validated. Never optimized. This is the artifact that makes TDD possible at all.
- **Warp (`warp` backend):** the production backend, targeting `cpu` and `cuda` devices from identical kernel source. Ships as v1.
- **numba CPU (`numba` backend):** optional. Retained only as an independent CPU implementation to cross-check Warp's CPU codegen, and as a fallback if Warp's CPU path underperforms. Do not let it become a second physics implementation; it must consume the same physics-function definitions or be deleted.
- **Taichi:** deferred. Introduce only if vendor-neutral GPU (AMD/Intel) becomes a requirement. Warp's `wp.Texture3D` with CPU software fallback (v1.12+) is the deciding feature; Taichi has no equivalent.

**Guard rail:** if at Phase 2 exit the Warp CPU backend is more than ~2x slower than numba CPU on the same physics, revisit. That is the single trigger for reopening this decision.

---

## 1. Physics scheme

The engine is a DPM-class code: macroscopic condensed-history electron transport with aggressive step sizes, tuned for 1-20 MeV in low-Z media. Speed comes from the transport scheme, not the language.

### 1.1 Photons

- **Transport:** Woodcock (delta) tracking against a majorant total cross-section per energy bin. Eliminates per-voxel boundary distance computation; makes the traversal cost independent of voxel size. This is the single most important structural choice for the photon side.
- **Interactions:**
  - Compton: Klein-Nishina, Kahn rejection sampling. Binding effects (incoherent scattering function) via an optional rejection factor; off by default at 6-15 MV.
  - Photoelectric: absorbed locally above PCUT; no fluorescence (justifiable at MV energies; must be documented as a stated approximation).
  - Pair production: above 1.022 MeV, non-negligible in bone at 15 MV. Sample e-/e+ energy split; annihilate positron at rest into two 511 keV photons (ignoring in-flight annihilation).
  - Rayleigh: off by default. Provide the hook; document the omission and quantify it in a validation test.
- **PCUT:** 50 keV default. Below this, deposit locally.

### 1.2 Electrons and positrons

This is where the runtime actually goes. A "fast photon engine" is a fast condensed-history electron engine.

- **Class II condensed history** with an explicit production threshold:
  - Continuous energy loss below threshold via restricted stopping power.
  - Discrete Møller (e-) / Bhabha (e+) delta-ray production above ECUT.
  - Discrete bremsstrahlung above the photon production threshold.
- **Random hinge (DPM scheme):** the step is split at a uniformly-sampled hinge point; energy loss is applied continuously, angular deflection is applied as a single deflection at the hinge. This is what allows the very long steps that make DPM fast.
- **Multiple elastic scattering:** Rutherford-based grouped scattering with hinge deflection sampled from the appropriate angular distribution. Goudsmit-Saunderson is the accurate alternative; start with the DPM approach and treat GS as a later accuracy backend behind the same interface.
- **Energy-loss straggling:** mean loss plus a fluctuation model. Start with mean-loss only, measure the depth-dose penumbra error against the reference, then add straggling if it exceeds tolerance. Do not add complexity before a test demands it.
- **ECUT:** 200 keV kinetic default (DPM's value). Below this, deposit locally along the residual CSDA range or at the point, per a configurable option. This cutoff is *the* accuracy/speed dial; expose it and test its sensitivity.

### 1.3 Materials and geometry

- **Voxelized rectilinear grid.** No CSG, no meshes, in v1.
- **CT → (mass density, material index)** via a configurable HU segmentation table. Cross-sections are stored per material and scaled by density at lookup. Five to seven materials (air, lung, adipose, water/soft tissue, cartilage, bone, cortical bone) is sufficient at MV energies.
- **Dose grid may differ from the CT grid.** Score on a separate, coarser dose grid (typically 2.5-3 mm) while transporting on the CT grid. This decouples transport resolution from Dij memory.
- **Dose to medium** is scored natively. Dose-to-water conversion is a post-processing option via stopping-power ratios; it is not a transport-time decision.

### 1.4 Source model

- **Virtual source:** point or extended (Gaussian) focus, plus a fluence map defined on a bixel grid at the isocenter plane.
- **Primary sampling:** sample a bixel index `b`, then sample position within the bixel, then direction from the focus. Every primary carries `b` as its **beamlet tag**, which it and all its secondaries inherit for the whole cascade.
- **Spectrum:** energy sampled from a tabulated or analytic spectrum; off-axis softening as a radial correction, optional.
- **Phase-space input** (IAEA format) behind the same source interface, deferred to Phase 5.

---

## 2. Beamlet-tagged Dij: the core architectural problem

This is the part that distinguishes the project from a general-purpose MC code, and it must be designed first, not bolted on.

### 2.1 One-pass tagged transport

Simulating each beamlet in a separate run scales linearly in beamlet count and re-pays the geometry and setup cost every time. Instead: one batch of histories, each tagged with its originating bixel, scoring atomically into a `(voxel, beamlet)` structure. Scattered secondaries keep the tag of their originating primary.

### 2.2 Memory: batched dense scoring, then compaction

A dense `(n_voxels × n_beamlets)` buffer is unaffordable (10^6 voxels × 10^3 beamlets × 4 B = 4 TB). A fully sparse atomic hash table is slow and awkward on GPU. The practical structure is:

1. Partition the beamlets into **batches** of size `B` (32-128, tuned to GPU memory).
2. For each batch, allocate a dense `(n_dose_voxels × B)` float32 buffer. At 10^6 dose voxels and B=64, this is 256 MB: fits.
3. Transport all histories for the batch; score with `atomic_add` into `buffer[voxel, b_local]`.
4. **Compact:** threshold each beamlet column at a relative cutoff (default 1e-3 of that column's max, configurable), emit COO triplets, append to a growing store.
5. After all batches, assemble CSC with beamlets as columns (matRad/pyRadPlan convention: Dij is `n_voxels × n_bixels`).

**The truncation threshold is a physics decision with an optimization consequence.** It biases low-dose tails, which is exactly where NTCP objectives and dirty-dose/LET-guided objectives live. Pin it with a test that compares an optimized plan's DVH metrics across thresholds, not just the raw matrix norm.

### 2.3 Scoring on GPU: atomics and divergence

- Dose deposition is a scattered `atomic_add`. Contention is manageable because deposits from a batch spread over many voxels, but a shared-memory staging buffer per block is worth benchmarking.
- **Thread divergence** from per-particle branching is the dominant GPU inefficiency. Mitigations, in order of expected payoff and implementation cost:
  1. Separate photon and electron kernels; maintain persistent particle queues (stacks) between them. This alone removes most divergence.
  2. Sort or partition electron queue entries by material or energy bin before launch.
  3. Persistent-thread kernels with a work queue, so threads that finish a history pull a new one instead of idling.

Do not do (2) and (3) until (1) is profiled.

---

## 3. Variance reduction

The 2-3% per-beamlet target is achievable only with these, not with brute force.

- **Woodcock tracking** (already in §1.1).
- **Russian roulette + splitting** on secondaries by weight, with a configurable survival weight.
- **Range rejection** for electrons: if the residual CSDA range cannot carry the electron out of its current voxel *and* its energy is below a threshold, deposit locally. Cheap, large payoff, small bias; test it.
- **Bremsstrahlung splitting** (uniform or selective) at the source and in high-Z regions.
- **Correlated sampling across beamlets.** Neighbouring bixels traverse nearly the same geometry. Reusing the same random number stream across beamlets within a batch correlates their noise, which cancels partially in the fluence-weighted sum. This is a research-grade lever with real payoff for Dij specifically; schedule it as an experiment (Phase 4), not a v1 requirement.
- **History repetition / STOPS** (simultaneous transport of particle sets): transport a set of particles that share the same interaction sequence but differ in start position. Strong synergy with the beamlet structure.

### What is explicitly out of scope

**Denoising the Dij.** Statistical noise in Dij is not benign for optimization: the optimizer exploits it, producing fluences that are optimal for the noise realization rather than for the true dose. Denoising suppresses variance but introduces spatially correlated bias, and the optimizer exploits that too. If denoising is ever considered, it must be validated by re-forward-computing the optimized fluence with an independent high-statistics MC and comparing DVH endpoints, not by comparing Dij matrices.

**Note on the noise/optimization interaction:** even at 2-3% per beamlet, the optimizer sees a systematically perturbed problem. Plan a dedicated study (Phase 4) sweeping per-beamlet uncertainty and measuring the resulting bias in optimized DVH endpoints. This is publishable in its own right and it is the empirical justification for whatever uncertainty target you ship.

---

## 4. Software architecture

```
src/pyradmc/
  physics/            # target-agnostic, pure, scalar-in-scalar-out
    compton.py        # sampling routines
    photoelectric.py
    pair.py
    brems.py
    moller.py
    msc.py            # multiple scattering, hinge
    eloss.py          # restricted stopping power
  data/
    interface.py      # CrossSectionSource ABC: sigma(E, material) etc.
    analytic.py       # closed-form backend (Phase 0)
    tabulated.py      # table backend + sub-grid product integration
    materials.py      # HU segmentation, material composition
  geometry/
    grid.py           # rectilinear voxel grid, Woodcock majorant
    source.py         # Source ABC; virtual source; (later) phase space
  transport/
    photon.py         # photon step loop
    electron.py       # electron step loop
    queues.py         # particle stacks
  scoring/
    dij.py            # batched dense buffer, threshold, COO -> CSC
    dose.py           # single-fluence dose scoring
  backends/
    ref/              # NumPy reference; the oracle
    warp/             # production: cpu + cuda
    numba/            # optional cross-check
  rng/
    interface.py      # uniform(), normal(), per-history state
  api.py              # compute_dij(...), compute_dose(...)
tests/
  unit/               # sampling distributions, cross-section values
  physics/            # single-interaction, analytic benchmarks
  integration/        # depth dose, profiles, heterogeneous slabs
  dij/                # consistency, truncation, fluence-sum
  perf/               # benchmarks, not correctness
  fixtures/           # synthetic phantoms, small reference datasets
AGENTS.md
CLAUDE.md
```

### The two interfaces that must be right from commit one

**1. `CrossSectionSource`.** Every physics routine takes cross-sections through this. The analytic backend is one implementation; the tabulated backend is another. The tabulated backend must integrate *products* over sub-grid intervals rather than evaluating separately-averaged bin quantities at bin centers; intra-bin covariance between the stopping power and the looked-up quantity is a real numerical hazard, and this requirement should be test-pinned, not left as a convention.

**2. `RNG`.** The CPU and CUDA RNG APIs differ (`numba.cuda.random` xoroshiro128p vs. host generators; Warp has `wp.rand_init`/`wp.randf`). The physics functions must never see this. A thin per-target shim exposing `uniform(state)`, and nothing else, is sufficient.

### Single-source discipline

Every physics routine is a pure function of scalars and passed-in array handles: no allocation, no Python objects, no branching on types. Warp's `@wp.func` compiles these to both targets. The reference backend calls the *same source* through a NumPy-typed shim. If a routine cannot be written this way, it does not belong in `physics/`.

---

## 5. Testing strategy (TDD, agentic)

### 5.1 The fundamental constraint: no bit-reproducibility across targets

Atomic float accumulation is non-associative, and CPU/GPU transcendental intrinsics differ. **CPU and GPU results will never be bit-identical.** Design the test harness around this from day one; discovering it in month four is demoralizing and leads to bad workarounds.

- *Within* a target, with a fixed seed and fixed thread count: bit-reproducible. Enforce it.
- *Across* targets: statistical equivalence only.

### 5.2 Test layers

| Layer | Oracle | Example |
|---|---|---|
| Sampling | Analytic distribution | KS test on Klein-Nishina scattering angles vs. analytic CDF |
| Cross-section | Closed form / NIST values | Total attenuation coefficient in water at 6 MeV |
| Single interaction | Energy-momentum conservation | Compton kinematics; property-based via `hypothesis` |
| Single-particle | Analytic transport | Exponential attenuation of a narrow beam in a homogeneous slab |
| Reference-vs-backend | `ref` backend | Same seed sequence, ~10^4 histories, χ² on scored dose |
| Physical validation | Published data / EGSnrc | PDD and profiles in water, 6 and 15 MV |
| Heterogeneity | Published data | Water/lung/water and water/bone/water slabs |
| Dij consistency | Internal | `Dij @ w` ≈ direct MC dose for fluence `w`, within combined σ |
| Truncation | Internal | DVH endpoints stable across truncation thresholds |

### 5.3 Statistical oracles, concretely

A gamma-index pass rate is a poor unit-test oracle: it is expensive and its failure mode is uninformative. Prefer, in order (this ordering was established by measurement, not assumption; see `tests/unit/test_oracle_power.py` in the scaffold):

1. **χ² over a region of interest**, reported with a p-value. The *detection* oracle. It aggregates evidence across voxels rather than counting tail events, and is measurably more powerful against small systematic bias: at 20k voxels and 2% per-voxel σ it catches a 0.25σ uniform bias that the z-score outlier count misses.
2. **Voxelwise z-score outlier count:** `|d_a - d_b| / sqrt(σ_a² + σ_b²) < k`, with the fraction exceeding `k` tested binomially. The *diagnostic* oracle: it localizes a discrepancy, which χ² cannot. Use it after χ² fails.
3. **Gamma index** only for the physical-validation layer, against published data, where it is the field convention.

Neither oracle resolves uniform bias below roughly 0.1σ_voxel at 20k voxels; the floor scales as 1/sqrt(n_voxels). Enlarge the region of interest, never the tolerance.

Every MC test needs an uncertainty estimate. Score dose *and* dose² per batch so σ is always available; this is not optional infrastructure.

### 5.4 Test speed

TDD dies if the loop is slow. Keep a tiered suite:

- `tests/unit`, `tests/physics`: milliseconds, no transport. Run on every save.
- `tests/integration`: seconds, tiny phantoms (16³), low statistics, loose tolerances. Run on every commit.
- `tests/validation`: minutes, full statistics. Nightly / CI on tag.
- `tests/perf`: benchmarks with recorded baselines; regression alerts, not pass/fail.

Fix seeds and thread counts in the fast tiers so failures are deterministic and reproducible.

---

## 6. Phases and exit criteria

### Phase 0: Reference engine and interfaces (2-3 weeks)
Pure NumPy, single history at a time, analytic cross-sections. Photon transport only: Compton, photoelectric, no electrons (deposit electron energy locally, i.e. a KERMA approximation).

*Exit:* exponential attenuation and broad-beam buildup reproduce analytic expectations; `CrossSectionSource` and `RNG` interfaces are stable; the full test-layer scaffold exists and runs in under 30 seconds.

### Phase 1: Reference electron transport (3-4 weeks)
Add condensed-history electrons with random hinge to the `ref` backend. Still slow, still NumPy.

*Exit:* PDD in water at 6 MV agrees with published EGSnrc data within 2% / 2 mm beyond depth of maximum dose, at high statistics. This is the correctness anchor for the entire project; everything downstream is validated against it.

### Phase 2: Warp backend, dual-target (4-6 weeks)
Port the physics functions to `@wp.func`. Separate photon/electron kernels with queues. Score to a single dose grid (no Dij yet). Run on `cpu` and `cuda` from one source.

*Exit:* both Warp devices reproduce `ref` within the z-score oracle; GPU throughput ≥ 10^6 histories/s on a workstation GPU for a 6 MV field in water; Warp-CPU is within 2x of numba-CPU (or the numba backend is dropped).

### Phase 3: Dij (4-6 weeks)
Beamlet tagging, batched dense scoring, thresholding, CSC assembly. Basic variance reduction: range rejection, Russian roulette, splitting.

*Exit:* `Dij @ w` matches a direct single-fluence MC computation within combined statistical uncertainty; a clinically representative case (~1000 bixels, 3 mm dose grid, 2-3% per-beamlet σ) completes in a target time you set here once Phase 2 throughput is known. **Do not commit to a wall-clock number before Phase 2 measures throughput**; anything stated now is guesswork.

### Phase 4: Variance reduction and the noise/optimization study (4-6 weeks)
Correlated sampling across beamlets; history repetition. In parallel, the empirical study of per-beamlet uncertainty vs. optimized-plan DVH bias.

*Exit:* a defensible, data-backed default for per-beamlet uncertainty; measured speedup from correlated sampling.

### Phase 5: Tabulated data, phase space, pyRadPlan adapter (4-6 weeks)
Swap the analytic backend for tables (with the sub-grid product integration). IAEA phase-space source. A thin adapter exposing Dij in pyRadPlan's expected form.

*Exit:* accuracy improvement over analytic is quantified; adapter round-trips a pyRadPlan case.

---

## 7. Repository setup for agentic TDD

### `AGENTS.md` / `CLAUDE.md` must encode

- **The red→green→refactor loop is mandatory.** No physics code without a failing test first.
- **The `ref` backend is the oracle.** Never "fix" a reference-vs-backend discrepancy by adjusting the reference. If the reference is wrong, that is its own commit with its own analytic justification.
- **Statistical tests need a statistical mindset.** A test that fails once in twenty runs at k=2 is *working correctly*. Seeds are fixed in the fast tiers precisely so this does not become a source of noise; never loosen a tolerance to make a flake go away without first checking the seed is fixed.
- **No cross-target bit-equality assertions.** Ever.
- **Physics functions are pure, scalar, allocation-free.** State this as a hard constraint with an example of a violating function.
- **Every approximation is documented at the point of implementation** with an equation and a DOI-cited reference in the docstring, following the pattern already used in pyRadBio.
- **Never silently change a cutoff, threshold, or default.** ECUT, PCUT, and the Dij truncation threshold are accuracy-defining; changes require a test demonstrating the effect.

### Tooling

- `pytest` with `hypothesis` for property-based tests on kinematics and sampling.
- `pytest-benchmark` for the perf tier, with committed baselines.
- Typed stubs throughout (`mypy --strict` on non-kernel code; Warp kernels are exempt).
- Seed and thread-count fixtures at session scope.
- Synthetic phantom fixtures generated in code, not committed as binaries.

---

## 8. Principal risks

| Risk | Severity | Mitigation |
|---|---|---|
| Warp CPU codegen underperforms on branchy transport | High | Phase 2 exit criterion; numba fallback retained |
| Dij truncation biases NTCP/LET objectives | High | Dedicated truncation test on DVH endpoints, not matrix norms |
| Optimizer exploits Dij statistical noise | High | Phase 4 study; treat as research output, not just engineering |
| GPU divergence caps speedup below expectations | Medium | Queue-based photon/electron separation is designed in from Phase 2 |
| Warp is CUDA-only; AMD becomes a requirement | Medium | Physics layer is framework-agnostic; Taichi port is a backend, not a rewrite |
| Condensed-history scheme has subtle bias near interfaces | Medium | Heterogeneous slab tests in Phase 1; this is a known hard problem for DPM-class codes |
| Analytic cross-sections are too crude for the Phase 1 validation gate | Low | Move the tabulated backend earlier if the 2%/2mm gate fails on data quality rather than physics |

---

## 9. Immediate next actions

1. Name and license the repository; run the collision check on `pyRadMC`.
2. Write `AGENTS.md` and `CLAUDE.md` per §7, before any source.
3. Write `data/interface.py` and `rng/interface.py` as abstract interfaces with no implementations, and their contract tests.
4. First red test: total attenuation coefficient of water at 6 MeV from the analytic backend, against a NIST value.
5. Hand off to Claude Code.