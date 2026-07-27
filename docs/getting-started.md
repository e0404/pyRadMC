# Getting started

## Install

```bash
pip install pyRadMC              # reference backend; NumPy and SciPy only
pip install "pyRadMC[warp]"      # production backend, CPU and CUDA
pip install "pyRadMC[ct]"        # CT image reading (SimpleITK)
```

Warp is CUDA-only for GPU. The core install has no GPU dependency at all: `import
pyRadMC` pulls in no third-party module, so a machine with neither warp nor a GPU can
still use the reference engine.

## A first dose calculation

```python
from pyRadMC import AnalyticCrossSections, HostRNG, PencilBeamSource, ReferenceEngine, VoxelGrid

grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
source = PencilBeamSource(energy=6.0, position=(8.0, 8.0, -1.0), direction=(0.0, 0.0, 1.0))

result = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
    source, n_histories=80_000, n_batches=10, seed=20260726
)
```

`result.dose` is per-voxel dose in MeV/g **per emitted history**; multiply by
`pyRadMC.GY_PER_MEV_PER_G` and your own particles-per-MU scaling for Gy.
`result.dose_sigma` is the batched 1-sigma standard error — batches exist so that
uncertainty is *measured*, not modelled.

### Check the energy books

Every run closes an exact ledger:

```python
assert abs(
    result.energy_emitted
    - (result.energy_deposited + result.energy_unscored + result.energy_escaped)
) < 1e-9 * result.energy_emitted
```

This is bookkeeping, not an estimate. If it does not close, the engine is broken — it is
the cheapest sanity check available, and worth keeping in your own scripts.

### Know what produced your numbers

```python
print(result.provenance.summary())
# pyRadMC 0.1.0 ref/cpu seed=20260726 pcut=0.05 ecut=0.2 msc=gs step=0.2 xs=[analytic ...]
```

A dose array outlives the process that made it. `result.provenance` records the version,
backend, device, seed, cutoffs, multiple-scattering model, resolved substep fraction and
the cross-section citation, so an archived result still says how it was produced.

## Running on the GPU

```python
from pyRadMC import WarpEngine

engine = WarpEngine(grid=grid, cross_sections=xs, device="cuda:0")
result = engine.run(
    source,
    n_histories=8_000_000,
    n_batches=10,
    seed=20260726,
    concurrent_batches=2,
)
```

On CUDA, `concurrent_batches` can overlap independent statistical batches when one
transport stream does not fill the card. Try 1, 2, and 4 on the actual workload;
each lane owns another queue set and dose map, so memory grows approximately linearly.
The batch fold remains ordered, making the lane count bitwise inert on one device.
CPU and the reference engine accept the option but run sequentially. An exact
`TransmissionMaskSource(SpectralBeamSource(...))` composition is generated and
masked on-device, including compaction of exactly-zero mask weights.

The `run` signatures are identical between backends, so swapping the backend is a
one-line change. Results agree statistically across targets, never bit-for-bit: compare
with a chi-squared test, not `assert_allclose`.

## The influence matrix

This is what the engine is for.

```python
from pyRadMC import BeamletGridSource

source = BeamletGridSource(
    energy=6.0, z=-1.0, x_range=(0.0, 10.0), y_range=(0.0, 10.0), n_x=10, n_y=10
)
dij = engine.run_dij(source, n_histories_per_beamlet=200_000, n_batches=10, seed=20260726)

matrix = dij.dose_csc()      # scipy CSC, (n_voxels, n_beamlets)
plan_dose = matrix @ weights # the loop an optimizer runs thousands of times
```

!!! danger "Do not combine column sigmas in quadrature"

    Correlated sampling is the shipped default: beamlet columns share random streams and
    are therefore statistically **dependent**. Per-column sigmas remain valid on their
    own — per-beamlet QA is unaffected — but summing across columns to get a plan-dose
    sigma is invalid. `dij.variance_csc()` exports per-column variance and warns about
    exactly this. A per-batch plan-dose sigma is a known gap.

## Better cross-sections

The analytic parameterization is correct to a few percent and needs no data files. For
accuracy work, compile the tabulated backend from the IAEA EPICS evaluations — see
[cross-section data](data.md).

## Where to go next

- [Validation status](validation.md) — what has actually been gated, and against what.
- [API reference](api.md) — the supported surface.
- The `examples/` directory in the repository: each script solves a small but physically
  meaningful problem and saves a figure a physicist can check at a glance.
