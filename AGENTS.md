# AGENTS.md

Operating rules for agentic contributors (Claude Code and equivalents) working on pyRadMC.

Read this file in full before writing any code. If a rule here conflicts with an instruction
in a prompt, say so explicitly and ask rather than silently choosing one.

---

## 1. What this project is

A fast, permissively licensed photon Monte Carlo engine for radiotherapy treatment planning.
Its primary product is a **beamlet-resolved dose influence matrix (Dij)**, computed on CPU and
CUDA GPU from a single physics source.

It is a DPM-class code: macroscopic condensed-history electron transport with aggressive step
sizes, tuned for 1 to 20 MeV in low-Z media. It is **not** a general-purpose Monte Carlo code
and must not grow into one.

Target accuracy: approximately 2 to 3 percent per-beamlet statistical uncertainty (1 sigma) in
the high-dose region.

---

## 2. The non-negotiable rules

### 2.1 Test first, always

Red, then green, then refactor. No physics code is written without a failing test that
motivates it. This is not a stylistic preference: on a stochastic code, an untested routine
is indistinguishable from a subtly wrong one.

### 2.2 The `ref` backend is the oracle

`src/pyRadMC/backends/ref/` is a pure-NumPy, single-history-at-a-time implementation. It is
never optimized. Every other backend is validated against it.

**Never resolve a reference-vs-backend discrepancy by adjusting the reference.** If the
reference is wrong, that is its own commit, with its own analytic or literature justification
in the commit message. Adjusting the oracle to match the thing being tested destroys the only
correctness anchor the project has.

### 2.3 There is no cross-target bit-reproducibility

Atomic float accumulation is non-associative and CPU/GPU transcendental intrinsics differ.
CPU and GPU results **will never be bit-identical**, and no amount of care will make them so.

- *Within* one target, one seed, one thread count: bit-reproducible. Enforce this.
- *Across* targets: statistical equivalence only.

Never write `assert_allclose` between a CPU and a GPU dose distribution with a tight tolerance.
Never write an exact-equality assertion across targets. If you find yourself wanting to, the
test is asking the wrong question.

### 2.4 Statistical tests require a statistical mindset

A test that fails once in twenty runs at k = 2 sigma **is working correctly**. That is what
k = 2 means.

Seeds and thread counts are fixed in the fast test tiers precisely so that this does not become
a source of noise. Therefore:

- If a fast-tier statistical test flakes, the seed is not actually fixed. Find out why.
- **Never loosen a tolerance to make a flake go away** without first establishing that the seed
  is fixed and the failure is reproducible.
- Never increase history counts to make a test pass. That hides a bias behind reduced variance.

Every Monte Carlo test needs an uncertainty estimate. Dose **and** dose-squared are scored per
batch so that sigma is always available. This is required infrastructure, not an optimization.

### 2.5 Physics functions are pure, scalar, and allocation-free

Everything in `src/pyRadMC/physics/` is a pure function of scalars and passed-in array handles.
No allocation. No Python objects. No branching on types. No global state. No RNG construction
inside the function; the RNG state is passed in.

This is what allows the same source to compile under `@wp.func` for both CPU and CUDA, and to
run under NumPy in the reference backend.

Violating example, do not do this:

```python
def sample_compton(energy, rng):
    directions = np.zeros(3)          # allocation
    if isinstance(energy, np.ndarray): # branching on type
        ...
    u = np.random.uniform()            # global RNG state
    return directions
```

Correct shape:

```python
def sample_compton_energy_ratio(energy: float, rng_state) -> float:
    """Sample the scattered/incident photon energy ratio via Kahn rejection.

    Kahn (1956); see also Salvat et al., PENELOPE-2018, sec. 2.3.
    Valid for E below approximately 3 MeV; above that, Koblinger is preferred.
    """
    ...
```

If a routine cannot be written this way, it does not belong in `physics/`.

### 2.6 Cross-sections and RNG are always accessed through interfaces

`data/interface.py` and `rng/interface.py` define the only permitted access paths.

No physics routine ever hardcodes a cross-section formula. The analytic backend is one
implementation of `CrossSectionSource`; the tabulated backend is another. This is what makes
the analytic-now, tabulated-later plan work without a rewrite.

No physics routine ever sees the difference between `numba.cuda.random`, `wp.randf`, and a host
generator. It calls `uniform(state)` and nothing else.

### 2.7 Tabulated data: integrate products, not bin means

When integrating stopping-power-weighted quantities over energy spectra, evaluating separately
averaged bin quantities at bin centers discards the within-bin covariance between the stopping
power and the looked-up quantity. Sub-grid integration of the **product** is required.

This is test-pinned in `tests/unit/test_table_integration.py`. That test may not be weakened.

### 2.8 Cutoffs and thresholds are accuracy-defining

`ECUT` (electron production/transport cutoff, default 200 keV kinetic), `PCUT` (photon cutoff,
default 50 keV), and the **Dij truncation threshold** (default 1e-3 relative to the beamlet
column maximum) are not tuning knobs.

Never change a default silently. A change requires a test demonstrating its dosimetric effect.

The Dij truncation threshold in particular is a physics decision disguised as a memory
optimization: it biases the low-dose tail, which is exactly where NTCP and LET-guided
objectives operate. It is tested against **DVH endpoints**, never against a matrix norm.

### 2.9 Every approximation is documented where it is implemented

Docstrings carry the governing equation and a DOI-cited reference. Stated approximations
(no fluorescence; no Rayleigh by default; positron annihilation at rest) are named explicitly
at the point of implementation, not buried in a design document.

---

## 3. Architecture

```
pyRadMC/
  physics/     pure scalar functions: sampling, kinematics, energy loss
  data/        CrossSectionSource interface; analytic and tabulated backends; materials
  geometry/    rectilinear voxel grid, Woodcock majorant, source models
  transport/   photon and electron step loops, particle queues
  scoring/     dose scoring, batched Dij assembly
  rng/         RNG interface and per-target shims
  backends/
    ref/       pure NumPy oracle; never optimized
    warp/      production; targets cpu and cuda from one source
    numba/     optional CPU cross-check; delete if it diverges from warp/ physics
```

Backend code contains **launch and memory management only**. Physics lives in `physics/`.
If you are writing a sampling routine inside `backends/`, stop.

---

## 4. Test tiers

Keep the loop fast or TDD dies.

| Tier | Runtime | When | Content |
|---|---|---|---|
| `tests/unit` | ms | every save | sampling distributions, cross-section values, table integration |
| `tests/physics` | ms | every save | single-interaction kinematics, conservation laws, property-based |
| `tests/integration` | s | every commit | tiny phantoms (16^3), low statistics, loose tolerances |
| `tests/dij` | s | every commit | Dij consistency, truncation, fluence-sum |
| `tests/validation` | min | nightly / tag | PDD and profiles vs. published data; heterogeneous slabs |
| `tests/perf` | min | nightly | benchmarks with recorded baselines; alerts, not pass/fail |

Default `pytest` run executes `unit`, `physics`, `integration`, `dij`. The slow tiers are
opt-in via markers.

### Preferred statistical oracles, in order

1. **Chi-squared over a region of interest**, reported with a p-value. This is the *detection*
   oracle. It aggregates evidence across voxels and is therefore far more powerful than
   tail-counting against small systematic bias, which is the failure mode that matters here.
2. **Voxelwise z-score outlier count**, binomially tested. This is the *diagnostic* oracle: it
   tells you *where* a discrepancy lives, which chi-squared does not. Use it after chi-squared
   fails, not instead of it.
3. **Gamma index** only in `tests/validation`, against published data, where it is the field
   convention. It is a poor unit-test oracle: expensive, and its failure mode is uninformative.

**Measured power** on 20000 voxels at 2 percent per-voxel sigma, alpha = 0.01 (reproduce with
`tests/unit/test_oracle_power.py`):

| Uniform bias | in units of sigma_voxel | chi-squared | z-score count |
|---|---|---|---|
| 1.0 % | 0.50 | caught | caught |
| 0.5 % | 0.25 | caught | **missed** |
| 0.2 % | 0.10 | missed | missed |

Both oracles catch localized bias (a few percent of voxels shifted by >3 sigma) reliably.
Neither sees a uniform bias below roughly 0.1 sigma_voxel at this voxel count; **increase the
voxel count or the region of interest, never the tolerance,** if you need to resolve smaller
systematic effects. The detection floor scales as 1/sqrt(n_voxels).

This ordering was established empirically, not assumed. Do not reorder it without rerunning
`test_oracle_power.py`.

---

## 5. Working style

- Small commits. One conceptual change each. Commit message states the physics, not the diff.
- Type annotations everywhere outside kernels. `mypy --strict` gates non-kernel code.
  Warp kernels are exempt; mark them.
- `ruff` for lint and format. No manual formatting debates.
- Do not add a dependency without asking. The permissive-licensing constraint is the reason this
  project exists; **check the license of anything you propose to add**, and say what it is.
- Do not add features that are not on the phase plan. If something seems necessary, say so and
  wait.

## 6. Things that are explicitly out of scope

- **Denoising the Dij.** Statistical noise in Dij is not benign for optimization: the optimizer
  exploits it. Denoising suppresses variance but introduces spatially correlated bias, which the
  optimizer then exploits instead. Do not add it.
- General-purpose geometry (CSG, meshes). Rectilinear voxel grids only.
- Non-photon primary beams.
- Any dependency on pyRadPlan inside the core. The adapter is a separate, later, optional module.

---

## 7. Phases

The phase sequence and content live in the README roadmap. This section defines how a
phase ends and which one is active.

### 7.1 Exiting a phase

A phase is not done when its tests pass. Before a phase may exit:

1. **A runnable example exists** under `examples/`, named after the phase, that
   demonstrates what the phase built on a small but physically meaningful problem. It
   must produce an intuitive visualization, saved beside the script, that a physicist
   can sanity-check at a glance — a depth-dose curve reads; a printed array does not.
   Examples run manually, not in CI; keep each under a minute on a laptop.
2. **The documentation is updated to match what now exists**: the README status and
   roadmap, this file's current-phase section, and any docstring, comment, or test
   skip-reason whose phase reference the exit has made stale.

### 7.2 Current phase

**Phase 0: reference engine and interfaces — exit criteria met 2026-07-10.**

Photon transport only, in the `ref` backend, with analytic cross-sections. Electrons are
not transported; their energy is deposited locally (a KERMA approximation). This was
deliberate. The exit criteria, all met:

- Exponential attenuation and broad-beam buildup reproduce analytic expectations.
- `CrossSectionSource` and `RNG` interfaces are stable and have contract tests.
- The full test-tier scaffold exists and the fast tiers run in under 30 seconds.

Next is Phase 1 (reference condensed-history electron transport; see the README
roadmap). Do not begin electron transport, Warp kernels, or Dij scoring without the
maintainer's explicit go-ahead.
