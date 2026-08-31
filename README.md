# pyRadMC

Fast photon Monte Carlo dose engine for radiotherapy treatment planning.

> **Status: pre-1.0 research software.** The physics, both backends and the influence-matrix
> pipeline are complete and test-pinned, and the API is stable enough to build on — but it is
> young, and the validation it has passed is documented precisely below rather than claimed
> broadly. **Do not use it for anything clinical, now or later, without independent
> validation.**

The primary product is a **beamlet-resolved dose influence matrix (Dij)** for photon IMRT/VMAT
planning: sparse CSC columns with a per-entry statistical uncertainty, computed on CPU and CUDA
from a single physics source by tagging every history's whole secondary family with its beamlet
of origin.

Because RNG streams are pure functions of `(seed, history)` and scoring is associative
fixed-point, *how* beamlets are scheduled — grouping, chunking, batch merging, which GPU gets
which column — cannot change the matrix by one bit. That is test-pinned, along with per-column
statistical equivalence to the reference oracle and a DVH-endpoint certification of the default
column truncation.

Measured end to end on a laptop RTX 4070 at planning statistics: **~1.1e7 histories/s**, i.e. a
100-beamlet 6 MeV field at 2–3 % per-beamlet sigma in single-digit seconds.

## Why

Existing photon Monte Carlo engines are, broadly, one of three things: licensed incompatibly
with open-source projects, too slow for treatment-plan optimization, or closed-source. pyRadMC
is an attempt at the fourth thing: permissively licensed, fast enough for beamlet-resolved
planning, and readable.

## What it does

- **DPM-class physics.** Macroscopic condensed-history electron transport with a random hinge
  and a Goudsmit-Saunderson multiple-scattering model (the shipped default; a Gaussian model is
  retained as a test instrument); Woodcock delta tracking for photons. Tuned for 1 to 20 MeV in
  low-Z media. It is not a general-purpose Monte Carlo code.
- **Single-source, dual-target.** Physics routines are pure, scalar, allocation-free functions
  compiled to CPU and CUDA through [Warp](https://github.com/NVIDIA/warp) (Apache-2.0). A
  pure-NumPy reference backend serves as the correctness oracle and is never optimized.
- **Planning-grade statistics.** Approximately 2 to 3 percent per-beamlet uncertainty, achieved
  through variance reduction rather than brute force. Correlated sampling across beamlets is the
  shipped default and roughly halves the renormalized plan-dose error at matched per-beamlet
  sigma.
- **Cross-sections behind an interface.** Analytic closed-form parameterizations for fast work;
  tabulated EPDL/EEDL data with coherent form factors and ICRU-37 stopping for accuracy. Same
  engine, same API.
- **Real geometry inputs.** CT images via a Hounsfield calibration, ICRP reference media, jaws
  and a rounded-tip MLC, IAEA phase-space sources, polyenergetic spectra, virtual source
  models driven by a measured radial primary fluence, and a dose grid
  decoupled from the transport grid with optional dose-to-water scoring.
- **Pencil-beam kernels.** Mono-energetic pencil-beam kernels in water, scored in cylindrical
  depth-by-radial-shell bins about the beam axis, on every backend. The cylinder is a *scoring*
  geometry: transport stays on the rectilinear grid, so the physics is unchanged by choosing it.

## What it does not do

- No denoising of the Dij. The optimizer exploits statistical noise; it exploits denoising bias
  too, and the latter is spatially correlated and therefore worse.
- No general geometry. Rectilinear voxel grids only.
- No non-photon primaries as a clinical modality. (Electron and positron primaries exist for
  validation and for phase-space transport.)
- No treatment-planning system in the core. A pyRadPlan adapter lives on the pyRadPlan side.

## Install

```bash
pip install pyRadMC              # reference backend; NumPy and SciPy only
pip install "pyRadMC[warp]"      # production backend, CPU and CUDA
pip install "pyRadMC[ct]"        # CT image reading (SimpleITK)
```

Warp is CUDA-only for GPU. If vendor-neutral GPU becomes a requirement, the physics layer is
framework-agnostic and a Taichi backend is a port, not a rewrite.

## Quick start

```python
from pyRadMC import AnalyticCrossSections, HostRNG, PencilBeamSource, ReferenceEngine, VoxelGrid

grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
source = PencilBeamSource(energy=6.0, position=(8.0, 8.0, -1.0), direction=(0.0, 0.0, 1.0))

result = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
    source, n_histories=80_000, n_batches=10, seed=20260726
)
result.dose          # MeV/g per emitted history, on the scoring grid
result.dose_sigma    # batched 1-sigma
result.provenance    # version, seed, cutoffs, backend, device, cross-sections
```

Swap in `WarpEngine(grid=..., cross_sections=..., device="cuda:0")` for the production backend —
the `run` signatures are identical. On under-filled CUDA forward runs,
`run(..., concurrent_batches=2)` or `4` overlaps independent statistical batches while
preserving same-device bit reproducibility; memory use grows with the lane count.

Everything importable from the top-level `pyRadMC` namespace is public API. Beam-limiting
devices (`pyRadMC.geometry.collimation`), the treatment-head pre-solve (`pyRadMC.geometry.head`),
the CT adapter (`pyRadMC.adapters.ct`) and the table precompiler (`pyRadMC.data.tabulated`) are
documented at subpackage level.

## Examples

Each script under [`examples/`](https://github.com/e0404/pyRadMC/blob/main/examples/) runs a small but physically meaningful problem and
saves a figure beside itself that a physicist can sanity-check at a glance. They need the
`examples` extra (matplotlib) and each runs in well under a minute.

![Depth-dose curves and dose maps for a broad beam and a pencil beam](https://raw.githubusercontent.com/e0404/pyRadMC/main/docs/assets/reference_engine_demo.png)

`reference_engine_demo.py` transports 2 MeV photons in water on the reference backend. The broad
beam (top) pulls away from the primary-only exponential with depth — scatter buildup — while the
pencil beam's narrow axis (bottom) hugs the same exponential, its scatter visible instead as the
halo in the log-scale map.

![Electron buildup vs the KERMA approximation, and electron-beam depth doses](https://raw.githubusercontent.com/e0404/pyRadMC/main/docs/assets/electron_transport_demo.png)

`electron_transport_demo.py` contrasts a 6 MeV photon beam with the same beam under the KERMA
approximation — the buildup region is the difference — and shows electron-beam depth doses whose
R50 tracks the CSDA range.

![One physics source compiled three ways, and the throughput gap](https://raw.githubusercontent.com/e0404/pyRadMC/main/docs/assets/warp_backend_demo.png)

`warp_backend_demo.py` computes the same 6 MeV beam with the reference interpreter and with the
Warp compilation of the *identical* physics source on CPU and CUDA. The depth-dose curves agree
within their error bands; the bars show why the backend exists.

![One Dij column, the fluence-sum identity, and a wedge plan recombined from the matrix](https://raw.githubusercontent.com/e0404/pyRadMC/main/docs/assets/dij_demo.png)

`dij_demo.py` is the point of the whole exercise: one beamlet's dose column with the truncated
tail visible (left); the open field recombined from the columns at unit weights against an
independently simulated open field (middle — columns partition the field exactly); and a wedge
plan as `Dij @ weights`, no re-simulation (right), which is the loop a treatment-plan optimizer
runs thousands of times.

![Central-axis depth doses for a sweep of square fields, and the rectangular output-factor matrix](https://raw.githubusercontent.com/e0404/pyRadMC/main/docs/assets/commissioning_vsm_example.png)

`commissioning_vsm_example.py` runs a full beam-data commissioning session: a water phantom
at a stated SSD, square fields, and the outputs a commissioning report carries — depth doses
with dmax and PDD(10), lateral profiles in both principal directions at five depths, diagonals
of the largest field, field widths and penumbrae, and a rectangular x-by-y output-factor
matrix. Every error bar is the spread of independent replicates, so the nonlinear quantities
carry one too. It is a [jupytext](https://jupytext.readthedocs.io/) percent notebook — open it
as a notebook or run it as a script; the whole configuration is the parameters cell at the top.

The head is **jawless with two stacked, staggered MLC layers** (29 and 28 leaf pairs of 1 cm
projected pitch, offset by half of it), built from published data about the Varian Halcyon and
driven by an unflattened beam. Leaf ends make the inplane edge and leaf *sides* make the
crossplane one, so a symmetric field steps in 1 cm even though each edge lands on a 0.5 cm
lattice. That machine was chosen because it leaves the collimator doing everything, with no
jaw to hide behind. Two results it establishes: field widths reproduce to within 0.1 mm at the
isocentre with **no fitted leaf-position offset**, from the rounded-end tangent geometry plus a
solved radiation-edge correction (the tangent ray grazes the tip, so it marks full transmission
rather than the 50 % edge — worth under 0.02 mm below a 10 x 10 field and 0.84 mm at 28 x 28);
and on an unflattened beam the field edge must be taken at the profile's **inflection point**
rather than at 50 % of the axis, because the classical construction reports a 40 mm penumbra
that describes the cone rather than the collimator.

**The source comes one of two ways.** By default it is the analytic model built in the notebook
— a spectrum, a focal spot, a radial fluence, an extra-focal term and a contaminant-electron
term — and needs no external data; most of that machine's geometry is unpublished, so the
opening cell is an explicit inventory of what is stated, fitted, bounded or invented, and the
fitted off-axis fluence table does *not* ship because it is derived from vendor measurements
(`PRIMARY_FLUENCE_FILE` is the hook for one you fit against your own). Point `PHASESPACE` at an
IAEA pair instead — or set `HALCYON_VSM_PHASESPACE` in the environment and run it unedited —
and the whole source model arrives already sampled in the particles, leaving the notebook to
supply only the collimator. That is the interesting comparison: a virtual source model fitted
through some other code's simplified collimator, pushed through an explicit one. Before
transporting anything that route rebuilds the model's *stated* spectrum and refuses the file if
the records disagree by more than 50 keV in on-axis mean energy, because a phase space that
contradicts its own documentation is a wasted run at best — a check that has already earned its
place. No phase space ships with the repository.

The rest: `tabulated_demo.py` (EPDL/EEDL cross-sections), `materials_demo.py` (bone/lung
heterogeneity), `ct_demo.py` (dose on a CT), `dose_grid_demo.py` (decoupled scoring grid and
dose-to-water), `spectral_source_demo.py` (6 MV polyenergetic fan), `collimation_demo.py`
(staircase MLC field), `virtual_source_demo.py` (measured-primary-fluence source model),
`phasespace_demo.py` (IAEA phase-space source), `custom_source_demo.py`,
and `noise_bias_study.py` (the correlated-sampling decision study).

## Backends

| Backend | Role | Devices |
|---|---|---|
| `ref` | correctness oracle; never optimized | CPU (NumPy) |
| `warp` | production | CPU, CUDA |

Two rules from [`AGENTS.md`](https://github.com/e0404/pyRadMC/blob/main/AGENTS.md) worth repeating, because they are the ones people break:

1. **The `ref` backend is the oracle.** Never adjust it to match a faster backend.
2. **CPU and GPU results are never bit-identical.** Atomic float accumulation is
   non-associative. Test statistically or not at all. Bit-reproducibility holds *per device*,
   for a given seed.

## Validation status

Read this before trusting a number.

- **Photon cross-sections.** The tabulated (EPDL) backend reproduces NIST XCOM total mass
  attenuation for liquid water to ~0.3 %, gated at 1 %.
- **Electron stopping and ranges.** The default Berger-Seltzer ICRU-37 stopping is ESTAR-exact;
  CSDA ranges are gated against NIST ESTAR.
- **Depth dose.** Central-axis depth dose in water is gated against maintainer-supplied EGSnrc
  curves at 1, 2 and 6 MeV with a **5 %/3 mm gamma criterion**. The tighter 2 %/2 mm criterion
  is *not* currently met, and the limit is the benchmark rather than the cross-sections: the
  reference data lacks its EGSnrc transport parameters, exact geometry and source normalization.
  A fully specified benchmark is the open item here.
- **Cross-backend agreement.** Every backend and device pair agrees within a chi-squared
  detection oracle under full coupled transport.
- **Dij integrity.** A 1x1-beamlet Dij column reproduces the open-field run bit for bit per
  device; beamlet grouping, chunking, batch merging and multi-device sharding are bit-inert; the
  default column truncation (1e-3 of the column maximum) moves the DVH endpoints D2/D50/D98 by
  well under 0.5 %, certified against DVH endpoints rather than a matrix norm.
- **Variance reduction is unbiased by construction, and test-pinned as such.** Russian roulette
  of sub-0.5 MeV photons is always on (annihilation photons sit just above it and stay analog);
  the validation gates pass with it active. Compton splitting is implemented and tested but
  ships off: measured, it does not earn its keep and does not reach the low-dose tail.

**A correctness trap worth stating loudly:** under correlated sampling — the default — per-column
uncertainties may **not** be combined in quadrature to obtain a plan-dose sigma. The columns are
statistically dependent. `DijResult.variance_csc()` exports per-column variance and warns about
exactly this. A per-batch plan-dose sigma is a known gap.

Every result carries a `provenance` record — version, backend, device, seed, cutoffs,
multiple-scattering model, substep fraction, and the cross-section citation — so an archived dose
still says what produced it.

## Data

The engine ships no cross-section libraries. The tabulated backend is compiled from the IAEA
EPICS 2023 evaluations, which you fetch yourself:

```bash
python -m pyRadMC.data.tabulated.build --output materials.npz
```

This downloads `EPDL2023.ALL` and `EEDL2023.ALL` (~120 MB) from
[www-nds.iaea.org/epics](https://www-nds.iaea.org/epics/ENDF2023/), caches them under
`~/.cache/pyRadMC/epics`, and compiles a table. **Every library is verified against a pinned
SHA-256 before use** — these bytes *are* the cross-sections, so a corrupted transfer or a silent
upstream revision is refused rather than propagated into dose.

The libraries are the IAEA's; their licence and terms stay with them, and with you. Published
data reproduced in this repository (ICRP/ICRU reference media, Sternheimer coefficients, ESTAR
and XCOM anchors, the Ali & Rogers MV spectrum parameters) is cited at each point of use.

## Develop

```bash
pip install -e ".[warp,dev,examples]"
pre-commit install          # once per clone: lint, format, types, file hygiene
pre-commit run --all-files  # run those gates over the whole tree

pytest                      # fast tiers: unit, physics, integration, dij
pytest -m ""                # every tier, adding validation (slow) and perf
mypy pyRadMC                # types
mkdocs serve                # preview the docs
```

Read [`AGENTS.md`](https://github.com/e0404/pyRadMC/blob/main/AGENTS.md) before contributing. It is the governing document, and it is
written for both human and agentic contributors.

## License

Apache-2.0. See [`LICENSE`](https://github.com/e0404/pyRadMC/blob/main/LICENSE).
