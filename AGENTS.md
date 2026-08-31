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

`pyradmc/backends/ref/` is a pure-NumPy, single-history-at-a-time implementation. It is
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

Everything in `pyradmc/physics/` is a pure function of scalars and passed-in array handles.
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

This is test-pinned in `tests/unit/test_interface_contracts.py::test_product_integration_beats_bin_center_evaluation`. That test may not be weakened.

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
  ``CrossSectionSource`` reports as nonzero. Enabling a channel (e.g. Rayleigh)
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
pyradmc/
  physics/     pure scalar functions: sampling, kinematics, energy loss
  data/        CrossSectionSource interface; analytic and tabulated backends; materials
  geometry/    rectilinear voxel grid, Woodcock majorant, Source/BeamletSource interface
  transport/   photon and electron step loops, particle queues
  scoring/     dose scoring, batched Dij assembly
  rng/         RNG interface and per-target shims
  adapters/    optional, out-of-core bridges (CT image; the pyRadPlan adapter, later)
  backends/
    ref/       pure NumPy oracle; never optimized
    warp/      production; targets cpu and cuda from one source
```

Backend code contains **launch and memory management only**. Physics lives in `physics/`.
If you are writing a sampling routine inside `backends/`, stop.

**Sources are user-extensible** (`geometry/source.py`): the sanctioned extension point is
subclassing `Source` (open field) or `BeamletSource` (Dij) and implementing `emit` +
`max_energy`. The reference backend runs it directly; a device backend gets it for free
via the default `sample_batch` host **pre-sampling** route, or in-kernel via an optional
`warp_sampler`/`warp_beamlet_sampler` `@wp.func`. Do **not** re-add closed
`isinstance` dispatch that rejects unknown sources.

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

- Small commits. One conceptual change each. Commit message states the physics, not the
  diff, in the convention of section 9.
- Type annotations everywhere outside kernels. `mypy --strict` gates non-kernel code.
  Warp kernels are exempt; mark them.
- `ruff` for lint and format. No manual formatting debates.
- Library output goes through `logging.getLogger(__name__)`; `warnings.warn` is reserved for
  result caveats and API misuse. No `print` in `pyradmc/` (ruff `T20` enforces this), and no
  handlers/`basicConfig` outside a `__main__` CLI entry — verbosity belongs to the consumer.
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


## 7. Scope, releases, and how work is accepted

Development is no longer organised into numbered phases. What existed as a phase gate is
now two ordinary questions, asked per change:

1. **Is it in scope?** Section 6 says what never is. Beyond that: this is a photon dose
   engine for treatment planning. A change that makes it a general-purpose Monte Carlo
   code, or that adds a modality, needs the maintainer's agreement before it is written,
   not after.
2. **Is it demonstrated?** A change that adds a capability a user would reach for ships
   with a runnable script under `examples/` that exercises it on a small but physically
   meaningful problem and saves an intuitive figure beside itself — a depth-dose curve
   reads; a printed array does not. Examples run manually, not in CI; keep each under a
   minute on a laptop. A change that alters an accuracy-defining default (section 2.8)
   additionally ships with the measurement that justifies it.

Documentation is part of the change, not a follow-up: the README status and capability
list, the docs under `docs/`, and any docstring, comment or test skip-reason the change
has made stale.

Where the gates run: the pre-commit hook (`pre-commit install`, then
`pre-commit run --all-files`) carries the
fast, file-scoped checks — ruff, formatting, `mypy --strict`, file hygiene. It runs **no
tests**, on purpose: a gate slow enough to be resented is a gate that gets bypassed. CI
runs the fast tiers on every push to `main` and `develop` (GitHub Actions and GitLab CI),
and the validation tier on a schedule. Run `pytest` yourself before pushing; do not
rely on the hook to catch a broken test.

Releases are semantic-versioned from `pyradmc.__version__` and cut by the release-candidate
flow of section 9.4; the release workflow refuses a tag that disagrees with `__version__`
or with `CITATION.cff`. Before a release, `pytest -m ""` passes (including the validation
tier), `CHANGELOG.md` records what changed, and anything in section 8 that the release
resolves is struck from it. Publication happens only from GitHub — the GitLab mirror's CI
validates and never deploys, so two pipelines can never race to upload the same version.

Historical decision records — including the measured negative results behind several
current defaults — are in `docs/decisions.md`.

## 8. Known limitations and standing constraints

These outlive the work that produced them. Each is a live constraint, not history.

- **Never combine Dij column sigmas in quadrature.** Under correlated sampling — the
  shipped default — columns are statistically *dependent*. Per-column sigmas stay valid
  (per-beamlet QA is unaffected), but a sum across columns, i.e. a plan dose, must not
  quadrature-combine them. A valid plan-dose sigma needs per-batch scoring, which
  `DijResult` does not carry and therefore does not expose. `variance_csc()` exports
  per-column variance and warns. Do not reintroduce a quadrature plan-dose sigma anywhere.
- **The 2 %/2 mm PDD gate is blocked on the benchmark, not on the data layer.** The
  validation tier gates at 5 %/3 mm against the EGSnrc curves. 2 %/2 mm was measured and is
  not reachable against this file: the limiter is the benchmark's missing geometry and
  transport metadata (see `tests/validation/data/README.md`), demonstrated by the fact that
  widening the phantom toward the infinite-field limit makes the deep-tail residual worse.
  Do not loosen the 5 %/3 mm gate, and do not "fix" this with a different data layer; a
  genuine 2 %/2 mm gate awaits a fully specified benchmark.
- **Compton splitting stays off.** `PHOTON_SPLIT_N` ships at 1 (analog). The `N > 1` path is
  test-pinned via the `N = 2` instrument but dormant. Its efficiency figure of merit was
  measured below 1 on both the analytic-water Dij and a phase-space source (0.75, 0.48,
  0.28 at `N = 2, 4, 8`), and it cannot reach the low-dose tail. Turning it on requires a
  new measurement, not an argument. The soft-photon roulette stays always-on regardless.
- **Enabling a coherent channel requires refitting pair production.** The analytic pair
  channel is calibrated against water totals that *include* coherent scattering. Adding a
  real coherent channel without recalibrating pair against coherent-free totals
  double-counts attenuation. A backend must take its channels from one consistent
  decomposition of the same library.
- **Positrons are Moller-approximated** (no Bhabha, annihilation at rest). If ever upgraded,
  Bhabha and annihilation in flight land together as the new default per section 2.10, with
  a test showing the dosimetric effect.
- **Variance-reduction CPU cost is unmeasured under controlled conditions.** The development
  laptop's power state drifts by several times, so only interleaved-median *ratios* are
  trustworthy on it. GPU cost is neutral (a warp retires with its longest thread). Absolute
  CPU efficiency needs stable-power hardware. More generally: verify clock state before
  believing any timing measured on that machine.
- **A no-op cutoff or step-fraction change is not a no-op.** `ECUT`, `PCUT`, the Dij
  truncation threshold and the substep fraction are accuracy-defining (section 2.8).
  Changing one needs a test demonstrating the dosimetric effect, on realistic spectral and
  heterogeneous geometry — never on an analytic monoenergetic homogeneous phantom, which
  has been measured to be roughly twice as forgiving.

## 9. Git workflow and commit convention

GitHub (`github.com/e0404/pyRadMC`) is the primary remote: pull requests, reviews and
releases happen there. The DKFZ GitLab is a mirror whose CI validates and never deploys.

### 9.1 Branches

| Branch | Role |
|---|---|
| `main` | Releases only. Receives merge commits from `rc/*` branches, each then tagged. |
| `develop` | Integration; the default branch. Every change lands as a squash merge from a task branch. |
| task branches | One per task, cut from `develop`, short-lived, deleted after merging. Descriptive names (`gs-msc-step-rework`, not `fix2`). |
| `rc/X.Y.Z` | Release candidate, cut from `develop` (section 9.4). |

Nothing is committed directly to `main` or `develop`.

### 9.2 Commit convention

Conventional-Commits-style subjects with the classic 50/72 shape:

```
type(scope): imperative description of the change

Body wrapped at 72 columns, separated from the subject by a blank
line. States the physics and the why, not the diff (section 5);
measurements that justify the change belong here.

Co-Authored-By: Someone Else <someone@example.org>
Refs: #123
```

- **Types**: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`,
  `revert`.
- **Scope** is optional; when present it is usually the module: `physics`, `transport`,
  `data`, `geometry`, `scoring`, `rng`, `ref`, `warp`, `dij`, `sources`, `adapters`,
  `examples`, `study`, `api`, `agents`.
- Subject: imperative, no trailing period. The description (the part after
  `type(scope): `) is at most 50 characters; the whole line never exceeds 72.
- Footers — co-authors, PR/issue references — come last, after a blank line, in
  `Key: value` trailer form; they are exempt from the 72-column wrap.
- One conceptual change per commit, as always.

### 9.3 Where the rules bind

On task branches the convention is *encouraged, not enforced* — those commits are
squashed away. It matters where history is permanent:

- **PR titles** — the squash-merge subject on `develop` — are held to it in review.
- So are the version-bump and merge commits that reach `main` and `rc/*`.
- `develop` accepts **squash merges only**; `main` accepts **merge commits from `rc/*`
  only**. Both are protected: PR + green CI required, no force pushes.

There is deliberately no automated message check; if drift ever makes one necessary,
a `commit-msg` hook is the place for it.

### 9.4 Releases

1. Cut `rc/X.Y.Z` from `develop`; open a PR onto `main`.
2. Iterate review and CI on the rc branch. `pytest -m ""` (validation tier included)
   must pass — trigger the validation workflow manually if the schedule has not run.
3. On the rc branch, bump `pyradmc/__init__.py::__version__`, `CITATION.cff` and
   `CHANGELOG.md` (`chore(release): vX.Y.Z`).
4. Merge the PR with a **merge commit** (never squash — `main` keeps the release shape).
5. Tag the merge commit `vX.Y.Z` and push the tag. The release workflow publishes to
   PyPI and GitHub Releases, refusing any tag/version/citation mismatch.
6. Merge `main` back into `develop` and push, so `develop` carries the release commit.
