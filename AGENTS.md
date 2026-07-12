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

No physics routine ever sees the difference between `wp.randf` and a host generator.
It calls `uniform(state)` and nothing else.

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

### 2.10 Physics is data-driven, never flag-driven

What physics runs is decided by the data source and the phase plan — never by a per-run
option. A planning engine's validation certifies **one configuration**; toggleable physics
multiplies the configuration space every backend-equivalence test must cover and invites
silent misconfiguration downstream. Concretely:

- The transport loop is **channel-complete**: it samples every interaction channel the
  ``CrossSectionSource`` reports as nonzero. Enabling a channel (e.g. Rayleigh, Phase 5)
  is a data change in the backend, not a transport flag.
- An approximation is retired by **replacing it as the default**, with a test showing the
  dosimetric effect — never by adding an option next to it. (If positron physics is ever
  upgraded, Bhabha and annihilation in flight go in together, as the new default.)
- The only sanctioned toggles are **test instruments** — options that exist so a test can
  isolate one piece of physics or sampling, like the engine's KERMA mode (``transport_electrons``)
  or ``run_dij``'s ``correlated=False`` independent-sampling mode (the shipped Dij is correlated;
  the independent mapping survives only to isolate column independence for the fluence-sum test) —
  and each must say so in its docstring.

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

**Phase 5: tabulated data, phase space, pyRadPlan adapter — UNDERWAY,
begun 2026-07-12.** The maintainer's explicit go-ahead was given 2026-07-12;
the phase-space source is the workstream in progress (most independent of the
three). Shipped so far, on **both backends**: the IAEA phase-space reader
(``geometry/phasespace.py`` — byte layout confirmed against a reference
implementation and a real file, not memory; validated against a 1.3 GB / 52.4M-
particle Varian TrueBeam file, which drove a streaming mmap reader, an on-disk-
count-wins tolerance for a stale PARTICLES header, and skip-with-count for stray
neutron/proton records) and ``PhaseSpaceSource``, which required the source
contract to carry a per-particle ``kind`` and statistical ``weight`` (``Primary``
gained both, additive; ``run`` weights the emitted-energy book and, for a positron
primary, adds the ``2 m_e c^2`` of annihilation photons its kinetic energy does not
account for, or the ledger cannot close). The **warp port** is done: the engine
host-samples each chunk (``sample_batch``), splits by kind, and seeds the existing
photon/electron queues — which already carry ``kind``/``weight`` and already kind-
separate their kernels — so transport is unchanged; cross-backend chi-squared,
energy-balance and chunk-invariance tests pass on CPU and CUDA. ``sample_batch`` is
**vectorized**: record indices come from one ``PCG64(seed)`` stream advanced to the
chunk offset (chunk-invariant, but a *different* stream from ``emit``'s per-history
spawn, so the backends draw independent records — emitted energy agrees only
statistically, and the exact-match test was relaxed accordingly) and records are
gathered from the mmap through a structured dtype in one fancy-indexed read. This
removed the per-history ``SeedSequence`` construction that dominated the warp path
(measured ~370 ms of 450 ms per 32k chunk); the demo now runs 300k histories in
~0.4 s on a 4070. ``examples/phase5_phasespace_demo.py`` now defaults to Warp (GPU if
present). The Compton-splitting ``N > 1`` turn-on this workstream was meant to
unlock was measured and stays **off** (see the carried item below). Phases 0
through 4 have exited. Phase 4's exit record is reproduced below — it defines
the machinery Phase 5 builds on and carries the warnings that target Phase 5;
earlier records live in the git history of this section.

---

**Phase 4 exit record: correlated sampling and the noise/optimization study —
begun 2026-07-11, exited 2026-07-12.**

What Phase 4 built (IMPLEMENTATION_PLAN.md section Phase 4; README roadmap
row 4):

- **Correlated sampling across beamlets via history repetition, now the
  shipped sampling configuration.** A change of the *stream* mapping only: key
  the physics stream on the within-beamlet index ``r = h % n_per`` instead of
  the global history index ``h``, so corresponding histories in every beamlet
  replay the same interaction sequence and only the entry position differs. The
  scheduling bit-inertness machinery carries over unchanged. ``run_dij``
  defaults to ``correlated=True`` on both engines; ``correlated=False`` is the
  sanctioned **test instrument** (section 2.10) that isolates the column
  independence the fluence-sum identity's quadrature sigma needs — it is not a
  production mode.
- **Statistical landmine that survives the decision:** correlated columns are
  statistically *dependent*. Per-column sigmas stay valid (per-beamlet QA is
  unaffected), but **any sum across columns — a plan dose — cannot combine the
  column sigmas in quadrature.** A valid plan-dose sigma needs per-batch
  scoring (batches are aligned across columns); the current ``DijResult`` does
  not carry per-batch data, so it exposes no plan-dose sigma. Do not
  reintroduce a quadrature plan-dose sigma anywhere.
- **The empirical study behind the decision** lives in
  ``examples/phase4_noise_bias_study.py``: a toy IMRT optimization (scipy, no
  new dependency) over Dij realizations, optimized on one ground truth and
  scored on a second independent one (the clinical recalculation), across six
  statistics levels down to the 2 percent target sigma and through a
  heterogeneity. Correlated sampling roughly halves the renormalized plan-dose
  error at matched per-beamlet sigma and is never worse on raw plan quality;
  its larger across-seed spread is common-mode scale that renormalization
  removes. That is the evidence base for the default above (section 2.10: one
  configuration ships, and this is the one).
- **Compton splitting: mechanism built, shipped OFF.** A primary photon's
  first Compton can split ``PHOTON_SPLIT_N`` ways (each copy weighted 1/N,
  unbiased, energy-conserving per realization), on ref and warp; only the
  primary splits so the population is bounded and the soft-photon roulette culls
  the copies. But ``PHOTON_SPLIT_N`` ships at **1 (off)**: the efficiency
  measurement found it does not earn its keep for the analytic-water Dij (FOM
  ``1/(sigma^2*time)`` < 1 on the reference CPU, ~neutral on the GPU) and,
  decisively, it does **not** help the low-dose tail — the Dij's NTCP/LET region
  — because that tail is fed by rare wide-angle multiple scatters uniform
  primary splitting cannot target (splitting deeper only worsens the FOM;
  measured). The machinery is retained because splitting a phase-space source
  particle (Phase 5) is efficient by construction; the ``N > 1`` path is
  test-pinned with N = 2 as the instrument so it stays validated while dormant.
  The soft-photon roulette (Phase 3) stays always-on regardless.

Exit criteria met: a defensible, data-backed default for the sampling
configuration and per-beamlet sigma (correlated sampling), and a measured
variance-reduction result — including the negative one that keeps Compton
splitting off until Phase 5 justifies it. The runnable phase example
(``examples/phase4_noise_bias_study.py``, with committed figure and CSV) and
the documentation (README, this section) are updated per section 7.1.

Standing items carried forward (Phase 5 targets marked). These outlive the
records they came from; the Phase 0-3 exit records themselves are in git
history.

- **Plan-dose sigma under correlated sampling — Phase 5.** Correlated columns
  are statistically *dependent*: per-column sigmas stay valid (per-beamlet QA
  is unaffected) but a sum across columns — a plan dose — must **not** combine
  the column sigmas in quadrature. A valid plan-dose sigma needs per-batch
  scoring (batches are aligned across columns); ``DijResult`` carries no
  per-batch data, so it exposes none. Build it with the pyRadPlan adapter that
  consumes it; never reintroduce a quadrature plan-dose sigma anywhere.
- **Compton splitting stays OFF — the Phase 5 phase-space FOM was measured and
  is still < 1.** ``PHOTON_SPLIT_N`` ships at 1 (analog); the ``N > 1`` path is
  test-pinned via the N = 2 instrument but dormant. Phase 4 found it does not earn
  its keep for the analytic-water Dij and cannot reach the low-dose tail. Phase 5
  then ran the ``1/(sigma^2*time)`` FOM on a phase-space source, on now-stable-power
  hardware (calibrated at 1.06x spread, not the 4-6x drift that blocked Phase 4):
  the efficiency ratio split/no-split came out **0.75, 0.48, 0.28 at N = 2, 4, 8**
  — worse, monotonically, tight across interleaved repeats. So the value stays 1.
  Why the Phase 4 "efficient by construction" expectation did not hold: the
  retained mechanism splits at the *first Compton*, where copies decorrelate only
  after that scatter, so sigma^2 reduction saturates far below 1/N while cost
  scales ~linearly with N; and emitting a phase-space primary (an index into a
  preloaded array) is as cheap as sampling an analytic beam, so the cost structure
  — and the sub-unity FOM — matches the analytic-beam result. The untested
  alternative, if phase-space variance reduction is ever wanted, is splitting the
  source particle **at emission** (to reuse a finite phase-space file — a
  latent-variance argument the CPU-time FOM does not measure), which is a different
  mechanism from first-Compton splitting. The soft-photon roulette stays always-on
  regardless.
- **VR CPU cost is still unmeasured under controlled conditions.** This laptop's
  power state drifts 4-6x, so only the interleaved-median *ratio* (split cost)
  is trustworthy here; GPU cost is neutral (a warp retires with its longest
  thread). Absolute roulette/split CPU efficiency needs stable-power hardware.
- **Pair-refit warning for whoever enables Rayleigh (Phase 5).** The analytic
  pair channel is calibrated against water totals that *include* coherent
  scattering. Turning on a real coherent channel without recalibrating pair
  against coherent-free totals double-counts attenuation; the tabulated backend
  must take its channels from one consistent decomposition of the same library —
  and must replace the Thomson-limit coherent angular model with form-factor
  sampling in the same change.
- **The benchmark-PDD gamma gate** (validation tier, EGSnrc curves at 5%/3mm)
  is data-limited by the analytic cross-sections. **Phase 5 must pass the same
  file over the full depth range at 2 percent / 2 mm — replace the data layer,
  never loosen this gate.**
- **Positrons stay Moller-approximated** (no Bhabha, annihilation at rest). If
  ever upgraded, Bhabha and annihilation in flight land together as the new
  default per section 2.10, with a test showing the dosimetric effect.

Do not begin Phase 5 without the maintainer's explicit go-ahead.
