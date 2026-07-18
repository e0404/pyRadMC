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

- Small commits. One conceptual change each. Commit message states the physics, not the diff.
- Type annotations everywhere outside kernels. `mypy --strict` gates non-kernel code.
  Warp kernels are exempt; mark them.
- `ruff` for lint and format. No manual formatting debates.
- Library output goes through `logging.getLogger(__name__)`; `warnings.warn` is reserved for
  result caveats and API misuse. No `print` in `pyRadMC/` (ruff `T20` enforces this), and no
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
unlock was measured and stays **off** (see the carried item below).

**The tabulated cross-section backend is also done, on both backends.** The
precompiler (``data/tabulated/precompile.py`` ``compile_water``; CLI
``python -m pyRadMC.data.tabulated.build``) assembles a water ``TabulatedData`` from
the EPICS libraries: **EPDL** photon cross sections (validated sub-percent against NIST
XCOM), **EPDL MF=27 coherent form factors** (replacing the Thomson-limit Rayleigh
angular model — sampled by inverting the form-factor cumulative, ported to the Warp
kernel), and **EEDL** elastic scattering power. Electron *stopping* is selectable
(``ElectronStoppingStrategy``): the default ``berger-seltzer`` keeps the analytic
ICRU-37 form (ESTAR-exact); ``eedl`` derives restricted collision (excitation +
ionization spectra integrated below the delta cut) and radiative stopping from EEDL,
which sits ~4% from ESTAR — provenance-consistent, not the accuracy default; Moller and
CSDA stay analytic. Water-only until ``MaterialData`` carries elemental composition (a
deferred task). The backend passes the EGSnrc PDD gate at 5%/3mm
(``test_tabulated_pdd_gamma_against_egsnrc_benchmark``) and is cross-backend consistent;
``examples/phase5_tabulated_demo.py`` is the runnable demonstration.

**Remaining Phase 5 work, reordered by the maintainer 2026-07-13** ("more important
for a standalone MC"), in order:

1. **Material elemental composition — DONE 2026-07-13.** ``MaterialData`` carries
   per-element mass fractions, the ICRU-37 I-value, the exact SBS-1984 Sternheimer
   density-effect coefficients and per-material ESTAR radiative anchors; the ICRP
   reference media (air, lung, adipose, cortical bone — the formulations ESTAR and
   the PDG compilation define, so every number is cross-checkable) joined the
   registry. The water-hardcoded electron machinery moved verbatim into
   ``data/berger_seltzer.py`` as pure functions of a ``MaterialData`` (the analytic
   backend delegates with the water entry — delegation identity test-pinned, the
   oracle did not move), and ``compile_materials`` mixes per-element EPDL/EEDL data
   into a registry-prefix ``TabulatedData``. Two structural changes carried the
   registry growth safely: ``CrossSectionSource.n_materials`` (tables flatten
   exactly the materials a source carries; the analytic source declares water-only
   and rejects anything else) and per-material validation gates — Berger-Seltzer vs
   ESTAR sub-0.4 percent per medium, EPDL mixtures vs NIST XCOM sub-percent for air
   and cortical bone, scattering-power ratios on the Z(Z+1)/A expectation, and a
   ref/warp chi-squared through a heterogeneous slab.
   ``examples/phase5_materials_demo.py`` is the runnable demonstration.
   **Interface substeps — DONE 2026-07-14.** Each electron substep is now capped
   at the next voxel face (``geometry/grid.py`` ``distance_to_voxel_boundary``,
   compiled into the Warp kernel through the physics loader; a shared
   ``BOUNDARY_NUDGE_CM`` biases the boundary-limited step just across the face so
   the half-open convention cannot trap it), so a step's density and material are
   those of the voxel it starts in instead of plowing one medium's stopping power
   across the boundary into the neighbour's mass. The single-voxel interface spike
   is gone on both backends (test-first: ``tests/integration/test_boundary_truncation.py``
   pins that the mass-scaled deposit is smooth across a pure density step — 1.95x
   before, under 1.4x after — and the geometry primitive is unit-pinned). Residual
   interface features are second order (the multiple-scattering hinge can still
   deflect the short post-hinge segment across the face) plus genuine interface
   dosimetry.
2. **CT image adapter — DONE 2026-07-14.** ``pyRadMC/adapters/ct.py`` (out of the
   core, AGENTS.md 6; optional ``pyRadMC[ct]`` extra — SimpleITK, Apache-2.0,
   maintainer-approved). A ``HounsfieldCalibration`` maps HU to per-voxel mass
   density (a Schneider-like piecewise-linear ramp) and to a registry material index
   (HU threshold bins into the ICRP media) — segmentation and density are
   independent, which is exactly the multi-material design (material = composition
   row, per-voxel density scales it). ``grid_from_hu`` is the pure, dependency-free
   path; ``grid_from_image``/``read_ct`` add the SimpleITK reader, handling the ITK
   frame (``(z,y,x)`` mm, origin at the first voxel centre) to engine frame
   (``(x,y,z)`` cm, origin at the lower corner), including a negative-direction-cosine
   axis flip; oblique orientations raise. The default calibration is an illustrative
   cited Schneider curve — a real plan passes the scanner's own. Unit-tested (pure
   calibration + synthetic-image reader, no committed CT data);
   ``examples/phase5_ct_demo.py`` renders MC dose on a synthetic CT slice (dose tracks
   the CT heterogeneities — deeper penetration through low-density lung). The CT
   demo's off-body air pocket was dropped: dose-to-medium in near-vacuum voxels is
   dominated by 1/density variance (speckle at finite statistics), which is inherent,
   not an interface bias.
3. **pyRadPlan adapter — the last remaining Phase 5 workstream. The pyRadMC
   side is complete as of 2026-07-15** (the spectral beam sources and the Dij
   consumer hooks recorded below); what remains is the engine subclass on the
   pyRadPlan side, which imports pyRadMC — pyRadMC stays pyRadPlan-agnostic
   (section 6). The per-batch plan-dose sigma was **resolved by maintainer
   decision 2026-07-14**: the adapter ships per-column variance plus the
   correlated-columns warning instead; per-batch data stays deferred (see the
   carried item below).

**Public source interface — DONE 2026-07-14** (maintainer-directed infrastructure,
precedes the pyRadPlan adapter which will hand the engine beamlet sources). `Source`
and `BeamletSource` ABCs (`geometry/source.py`) are the extension point: a user
implements `emit` + `max_energy` and the reference backend runs it. On Warp both emit
modes have **both routes** — the *simple* host pre-sampling (`sample_batch` /
`sample_beamlet_batch` → `generate_from_upload`, the phase-space path generalized) and
the *advanced* in-kernel wrapping of an optional `warp_sampler`/`warp_beamlet_sampler`
`@wp.func` (`make_generator_kernel` / `make_beamlet_generator_kernel`). The built-in
beam/lattice sources keep their validated in-kernel generators; the engines route by
capability, no longer raising on unknown types. A custom source's cross-backend
chi-squared (open field and Dij, both routes), the Dij group-size invariance, and the
energy ledger are test-pinned; `examples/custom_source_demo.py` (a Gaussian pencil
beam) is the runnable demonstration.

**Spectral beam source — DONE 2026-07-15** (the main pyRadMC-side piece the
pyRadPlan adapter consumes; contract confirmed with the maintainer at session
start). A histogram ``Spectrum`` (``geometry/spectrum.py``: edges + per-bin
content, sampled by CDF inversion, two uniforms) accepts **both** photon-number
and energy-fluence weights via a ``convention`` flag (energy fluence divides by
the bin midpoint — stated narrow-bin approximation); ``ali_rogers_mv`` builds one
from the analytic MV form of Ali and Rogers (2012, doi:10.1088/0031-9155/57/1/31),
function 12/13 with the table 3 mu parameterizations and the table 5 fitted
parameters for the nine benchmark beams (``ALI_ROGERS_BEAMS``), integrating
``psi(E)/E`` per bin on a sub-grid (2.7) and booking the ``C4`` 511 keV line as
``C4/0.511`` photons on the continuum's own scale. The divergent geometry
(``SpectralBeamSource`` open field, ``SpectralBeamletSource`` for the Dij) is a
focal-point fan through per-bixel rectangular apertures, **everything already in
the engine frame** — gantry/couch rotation stays out of the core, the adapter maps
frames exactly like the CT adapter — with uniform aperture sampling giving 1/r^2
for free and the transport loops flying the vacuum from the focal spot to the
grid (both backends already did this). ``emit`` consumes exactly four uniforms
keyed on the history index only, so correlated Dij sampling replays the same
energy and in-aperture offset in every beamlet (test-pinned). On Warp the source
uses the simple route with a **vectorized** ``sample_batch``/
``sample_beamlet_batch`` (the phase-space precedent: one ``PCG64(seed)`` stream
advanced to ``4*history_offset`` — chunk-invariant, beamlet-blind, a *different*
stream from ``emit``, so backends agree statistically, never bit-wise); the
per-history default was a measured demo bottleneck (428 s -> 2.4 s for 4M
histories on the 4070). Cross-backend per-column chi-squared (cpu and cuda) and
the exact polyenergetic ledger are test-pinned;
``examples/spectral_source_demo.py`` (Varian 6 MV fan into water: spectrum, PDD
with d_max ~1.2 cm, profiles tracking the projected field edges) is the runnable
demonstration.

**Beam-limiting devices + treatment-head pre-solve — DONE 2026-07-15**
(maintainer-requested workstream, go-ahead 2026-07-15; the three collimation
configurations were the maintainer's design). What shipped, all test-first and
on both backends where transport is involved:

- **Tungsten in the registry** (``TUNGSTEN = 5``; Ni/Cu atomic weights for later
  alloys). Berger-Seltzer vs ESTAR at Z = 74: collision <= 0.18 %, CSDA <= 1.21 %
  (standard gates, untouched); compiled EPDL vs XCOM sub-percent at 0.5-20 MeV
  with a documented 2 % gate just above the 69.5 keV K edge. The build CLI now
  compiles the whole registry (``--n-materials 1`` = the old water table).
- **``geometry/collimation.py``** — parametric BLD geometry in a beam frame
  (focal-spot origin, engine-frame axes; adapters rotate): straight-edged
  ``JawPair``, rounded-tip ``MLC`` (tip apex on the mid-plane; exact circular
  chords by slab-and-(halfspace-or-disc) inclusion-exclusion; Boyer & Li 1997,
  doi:10.1118/1.597996), ``BeamLimitingStack`` (full-line convention;
  ``from_origin`` for mid-stack starts), ``project_between_planes`` for
  isocenter-plane plan settings. Deferred, stated at the site: tongue-and-groove,
  interleaf gap, focused jaw edges, divergent leaf sides, alloy compositions.
- **The Dij transports and books ``Primary.weight``** (it silently assumed unit
  weight on both engines; the in-kernel ``warp_beamlet_sampler`` contract gained
  a trailing weight). Bit-pinned above the photon roulette weight cap (power-of-
  two scaling is float-exact there); **measured and documented: below the
  absolute cap, exact weight-linearity does not hold** — a boost chain crossing
  4.0 in one of two runs diverges that history's stream, and correlated sampling
  replays it into every column (~20 voxels at the test seed). Fair per run;
  pinned statistically below the cap.
- **Configuration 2/3 sources**: ``CollimatedSource``/``CollimatedBeamletSource``
  (deterministic ``exp(-sum mu_i(E) rho_i t_i)`` weights via one shared log-log
  ``mu_over_rho_total`` table — zero extra uniforms, correlated Dij replay
  survives; narrow-beam total attenuation stated) and ``TransmissionMask*``
  (user 0..1 mask, bilinear in the plateau-exact incremental form) — both on the
  shared ``_RayWeightModel`` wrapper bases; ``GaussianSpotBeam(let)Source``
  (photons born **on** a plane, aimed from a Gaussian focal spot; exactly six
  uniforms, sigma = 0 replays the spectral fan's lines on the same stream).
- **Configuration 1, the head pre-solve** (``geometry/head.py``): one-shot host
  MC, ``mode = auto|first_compton|attenuation``, forced first Compton in the
  stack (vectorized Kahn pinned against the scalar oracle) and in an
  ``AirColumn`` (both Compton arms kept — the contaminant electrons), output an
  ``InMemoryPhaseSpaceSource`` normalized **per original primary** (the N/n
  weight factor; a demo-caught defect, now regression-pinned) with fair
  Russian-roulette population control below ``weight_floor``. Both phase-space
  classes warn once when drawn histories exceed the stored population (batch
  sigmas never include the finite-phase-space latent variance).
  ``examples/collimation_demo.py`` is the runnable demonstration (BEV
  transmission map, wrapper-vs-pre-solve profiles, depth dose).

Carried items from this workstream: **focused jaw edges — the geometry SHIPPED
2026-07-16** (``JawPair(focused=True)``: the edge face pivots through the focal
spot, ``z_mid * q >= edge * w``, so a focal-spot ray sees full-thickness-or-
nothing and the geometric partial-transmission band collapses — verified pure-
geometry, 5.3 mm straight band → 0 mm focused at the validation jaw setting;
default stays straight, so nothing else changed). **Remaining, EPICS-gated:** the
nightly penumbra gate still tests the straight default (measured 15.8 mm vs the
8.9 mm band at 10 cm depth), so re-deriving a focused-edge gate window and
switching the tungsten demo figure to focused both need the EPICS tungsten data +
a warp run and are deferred until that data is on hand (regenerating the demo with
the water fallback would degrade the committed tungsten figure). **Tongue-and-
groove/interleaf leakage**; **W-alloy registry entries** (pure W at alloy density
ships, binders' atomic weights already present); and the **roulette-cap/attenuated-
weight graininess** in deep-leakage regions (absolute cap 4.0 vs ~1e-4 primary
weights; evidence in ``tests/dij/test_weighted_beamlet_ledger.py`` — revisit
only with a measured dosimetric case, per 2.10 as a default replacement).

**Warp port of the pre-solve — DONE 2026-07-16** (maintainer-requested; the carried
item above is retired). ``pyRadMC/backends/warp/presolve.py`` ``presolve_head_device``
runs the head MC as one kernel launch, one thread per primary, on cpu and cuda. The
host ``presolve_head`` stays the ``ref`` oracle (2.2). Structure: the collimation
geometry gained pure scalar ``@wp.func``-compilable twins — ``jaw_path_length`` /
``mlc_path_length`` in ``geometry/collimation.py`` (registered in the Warp physics
loader, pinned equal to the vectorized ``path_lengths`` on the host and float32-checked
on device), so host and device share one geometry definition; the *port deleted no
physics duplication yet but reuses* the existing scalar Compton/direction ``@wp.func``s
directly. A ``BeamLimitingStack.flatten()`` → ``CompiledStack`` → ``StackArrays``
uploads the heterogeneous stack (jaws + MLC, frame, shared leaf arrays) column-parallel;
mu is a device twin of ``_MuTables``. Primaries are still host-sampled
(``source.sample_batch``, cheap) and uploaded; the kernel does the stack + air forced
first Compton and atomic-appends exit particles into an ``ExitBuffer``, normalized per
primary on-device. **Validation (analytic water, EPDL-free, cpu+cuda):** attenuation
mode reproduces the host to float32 exactly (deterministic geometry/mu path — the strong
check); first-Compton totals/counts agree statistically (~0.06 % at 1.2e5 histories,
different RNG stream per 2.3). **The no-copy transport path is wired end to end:** ``return_device_source=True``
returns a ``DevicePhaseSpace`` (population left on the device, only the ``count`` scalar
read back), and ``WarpEngine.run`` transports it directly — a new
``kernels.generate_from_exit_buffer`` samples one record per history from the device
buffer (with replacement, distinct stream from transport), atomic-appends it to the
photon/electron queue as a source primary via the existing ``_push_source``, and books
emitted weight-energy through ``_escape``; ``_transport_chunk_device_buffer`` then drains
like every other route. Validated on cpu+cuda: device-resident vs host-handoff transport
are chi-squared-consistent on a water phantom (independent seeds, same population) and the
no-copy run's energy ledger closes (``tests/integration/test_device_presolve_transport.py``).
The host-handoff ``InMemoryPhaseSpaceSource`` remains the default and carries the same
population (``DevicePhaseSpace.to_phase_space`` materializes it). Ruff: the Warp
loop-variable ``float()``/``int()`` casts need inline ``# noqa: UP018``/``RUF046`` (the
convention already in ``kernels.py``; no per-file blanket ignore).

**Dij consumer hooks — DONE 2026-07-15.** The last pyRadMC-side pieces the
pyRadPlan adapter needs; with these, **pyRadMC's side of the adapter workstream
is complete** — the engine subclass itself is built on the pyRadPlan side
(section 6: pyRadMC stays planning-system-agnostic). ``GY_PER_MEV_PER_G``
(exact by SI definition, elementary charge fixed at 1.602176634e-19 C) converts
the engines' MeV/g-per-history scoring to absolute dose; the ``DijResult`` CSC
exports take ``unit='gy'`` (sigma scales linearly, variance quadratically).
The new ``variance_csc()`` is the export a planning system stores as its
dose-influence variance (pyRadPlan's ``physical_dose_var``): per-entry variances
stay valid within a column, but a correlated-sampling Dij has statistically
*dependent* columns, so the export carries a ``warnings.warn`` result caveat —
cross-column combinations, i.e. a quadrature plan-dose variance, are invalid
(Phase 4 exit record). ``sigma_csc()`` stays silent: per-beamlet QA is its
valid use.

**Decoupled dose grid + dose-to-water — DONE 2026-07-14** (maintainer-directed
infrastructure, precedes the pyRadPlan adapter, which defines its own dose grid
independent of the CT). Dose is *accumulated* on a
``pyRadMC.scoring.grid.ScoringGrid`` (``scoring_grid=`` on both engines'
``run``/``run_dij``; default None scores on the transport grid, byte-identical to
before); transport — Woodcock tracking, stepping, all material lookups — stays on
the transport (CT) grid and is invariant to the scoring grid (the streams never
see it, test-pinned to the bit per device). Any origin/spacing/shape is allowed,
**with no coverage requirement in either direction**: a deposit inside the CT but
outside the dose grid goes to a new *unscored* energy-ledger bucket (never a
clamped edge voxel — clamping would corrupt edge dose), so
``emitted == deposited + unscored + escaped`` closes exactly; an uncovered dose
voxel has zero mass and reports zero dose. Per-voxel mass is rebinned from the CT
by exact separable 1D voxel overlap (non-aligned, non-integer ratios exact),
which also *defines* dose-to-medium for a dose voxel overlaying several CT
voxels. Deposits are position-keyed end to end (``DepositFn`` carries the site,
the kernels a second ``GridInfo``); the Dij columns, ``DijResult.grid_shape``
and the per-group device buffer all live on the scoring voxel count — the
biggest plan-calc memory lever (2 mm CT scored at 3 mm is ~0.30x). Exact
identities are test-pinned on both backends: aligned coarsening equals the
summed-child energy rebin (int64 quanta are associative), subregion voxels are
bit-equal to the covering run's (no clamping), grouping stays bit-inert, and
cross-backend chi-squared holds on shared coarse/subregion grids.
``scoring_mode="dose_to_water"`` (same change) weights each deposit by the
restricted collision stopping-power ratio water/medium at the depositing
particle's cutoff-clamped energy, per substep, from the engine's
``CrossSectionSource`` (in-kernel from the flattened tables on Warp; Siebers et
al. 2000, doi:10.1088/0031-9155/45/4/983). This is a **scoring-OUTPUT
selection, not a physics toggle** (section 2.10): transport and the RNG streams
are identical in both modes and the energy books stay physical — pinned by
water D_w == D_m bit-identity and an exact-constant synthetic-SPR instrument on
both backends. Stated approximations live in ``pyRadMC/scoring/dose_to_water.py``
(sub-cutoff deposits freeze the ratio at ECUT; sub-PCUT photon deposits use the
electron SPR in place of mass-energy-absorption ratios); KERMA mode refuses
dose-to-water (no tracked electron to evaluate the ratio on).
``examples/dose_grid_demo.py`` is the runnable demonstration (2 mm transport vs
3 mm dose grid, subregion ledger, and the bone/lung D_w/D_m profile).

**Progress callback — DONE 2026-07-18** (maintainer-requested; adjacent to the
pyRadPlan adapter, which currently only sees beam-level completion). New
``pyRadMC/progress.py``: a dependency-free ``ProgressEvent``
(``histories_done``/``histories_total``/``elapsed_s``/``rate_hz``) and
``ProgressEmitter``, a thread-safe fan-in a caller passes as
``progress=`` to ``run``/``run_dij`` on both engines. Ticks land at
existing hard-sync points, so no new syncs are introduced: per batch for
``run`` on both backends and for ``ReferenceEngine.run_dij`` (batch-outer,
beamlet-inner), per **beamlet group** for ``WarpEngine.run_dij`` (group-outer,
scheduled off the shared device work queue) — the two Dij backends tick on
different axes that both sum to the same total, so a consumer should read
``histories_done / histories_total`` as the portable signal, not tick count.
The emitter is shared across every ``devices=[...]`` shard thread; its lock
serializes the update-and-dispatch, so the callback is never invoked
concurrently and always sees strictly increasing ``histories_done`` regardless
of which device finishes a group first (``concurrent_batches`` lanes do not
tick individually — only the shard's per-group point does). Every tick also
logs at ``DEBUG`` on ``logging.getLogger("pyRadMC.progress")``, callback or
not, so a caller gets a trace for free without wiring anything (AGENTS.md
section 5). Purely a host-side observation hook — no RNG stream or fold order
changes, so bit-identity and statistical tests are unaffected; pinned in
``tests/unit/test_progress.py`` (the emitter's concurrency guarantee, isolated
from transport) and ``tests/dij/test_dij_progress.py`` (both backends, both
methods, plus the multi-device/concurrent-batches fan-in, GPU-gated). The
pyRadPlan-side bridge into its own ``ProgressReporter``/``StatusReport`` is
adapter work, out of scope here (section 6).

Phases 0 through 4 have exited. Phase 4's exit record is reproduced below — it defines
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
  per-batch data, so it exposes none. **Resolved for the adapter (maintainer,
  2026-07-14): the adapter consumes per-column variance — ``variance_csc()``,
  shipped 2026-07-15 with its ``warnings.warn`` caveat — so the per-batch slice
  is a deferred nicety, not an adapter blocker.** Never reintroduce a quadrature
  plan-dose sigma anywhere.
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
- **The benchmark-PDD gamma gate** (validation tier, EGSnrc curves). The Phase 1
  analytic gate runs at 5%/3mm over truncated ranges. Phase 5 added a tabulated
  gate (`test_tabulated_pdd_gamma_against_egsnrc_benchmark`): the compiled backend
  (EPDL photons + EEDL scattering + ICRU-37 stopping) passes 5%/3mm, confirming the
  full parse→compile→load→transport path reproduces EGSnrc. **The 2%/2mm target
  originally recorded for Phase 5 was measured and is *not* reachable against this
  file, and the limiter is the benchmark's geometry/metadata, not the data layer**
  (2026-07-12): the tabulated backend barely differs from the analytic one on this
  gate — the analytic photon *total* was already NIST-calibrated, so the accurate
  channel split moves the PDD shape little — and *widening* the lateral phantom
  toward the infinite-field limit makes the deep-tail residual *worse*, showing our
  lateral-integrated pencil overestimates the benchmark's finite-field 6 mm-tube
  scoring at depth. Reaching 2%/2mm needs the benchmark's field width and EGSnrc
  transport settings (the missing metadata in `tests/validation/data/README.md`),
  **not** a different data layer. Do not loosen the 5%/3mm gate; a genuine 2%/2mm
  gate awaits a fully specified benchmark. (Form-factor coherent sampling — the
  refinement anticipated here — has since landed; the gate result stands.)
- **Positrons stay Moller-approximated** (no Bhabha, annihilation at rest). If
  ever upgraded, Bhabha and annihilation in flight land together as the new
  default per section 2.10, with a test showing the dosimetric effect.
- **BUG — FIXED 2026-07-18: ``warp_physics()`` leaked the kernel-side ``uniform``
  into ``pyRadMC.geometry.spectrum``.** Constructing a ``WarpEngine`` broke the
  *reference* backend's spectral source for the rest of the process:
  ``physics.py::_load()`` patched ``sys.modules["pyRadMC.rng"]`` and restored only
  the modules named in ``_KERNEL_MODULES``, so any pyRadMC module imported
  transitively *inside* the patch window bound the shim ``uniform`` at module
  level and kept it. The loader now snapshots ``sys.modules`` before the window
  and evicts every pyRadMC module first imported inside it — both the
  ``sys.modules`` entry and the parent-package attribute, since ``from package
  import submodule`` returns the stale attribute otherwise — forcing a clean host
  re-import. Regression-pinned in
  ``tests/integration/test_warp_physics_import_leak.py`` (subprocess checks:
  ``spectrum.uniform is rng.uniform`` after the load, and a reference-side
  ``Spectrum.sample_energy`` after a warp kernels import); the old repro
  ``pytest tests/integration/test_warp_device_reduction.py tests/dij`` passes.

Do not begin Phase 5 without the maintainer's explicit go-ahead.
