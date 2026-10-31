"""Magnetic-field cost benchmarks. Alerts, not pass/fail — with one floor.

Three questions the field raises about wall time, each recorded via pytest-benchmark
(``pytest -m perf --benchmark-only``) so regressions surface as baseline drift:

1. What does the field-free path pay for the branch the kernel now carries? An
   explicit zero field against no field in water, physics identical. Measured at
   adoption (RTX 4070, interleaved medians): 0.98, i.e. noise; the kernel is
   latency- and divergence-bound and one integer compare per substep is below it.
2. What does a field cost in tissue? 1.5 T in water: 1.05 at adoption — the
   gyration cap binds only below ~1.5 MeV there.
3. What does a field cost in air? A water / air / water slab at 0.35 and 1.5 T:
   1.20 and 2.27 at adoption. In air the gyration cap replaces the voxel face as
   the limiter and multiplies the substep count; this is the known cost, which the
   exterior-air path acts on in air far from the body
   (``test_exterior_air_throughput.py``, docs/decisions.md).

The one hard assertion is the catastrophic floor the open-field benchmark also
carries: the 1.5 T slab must still exceed 1e6 histories per second on CUDA
(adoption headroom ~5x). Ratios, not absolutes, are the trustworthy quantity on the
development laptop (AGENTS.md section 8); the benchmark records both.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.perf, pytest.mark.warp, pytest.mark.gpu]

FLOOR_HISTORIES_PER_S = 1.0e6
N = 400_000


def _water_grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(64, 64, 64), spacing=(0.3, 0.3, 0.3))


def _slab_grid() -> VoxelGrid:
    """Water 0-3 cm, air-density water 3-6 cm, water 6-10 cm (analytic source: water only)."""
    shape = (40, 40, 100)
    density = np.ones(shape)
    density[:, :, 30:60] = 1.205e-3
    return VoxelGrid(
        shape=shape,
        spacing=(0.5, 0.5, 0.1),
        density=density,
        material=np.full(shape, WATER, dtype=np.int32),
    )


def _engine(grid: VoxelGrid, field: MagneticField | None):
    from pyradmc.backends.warp.engine import WarpEngine

    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    # Current stepping on purpose: these baselines record the field's own cost, which is
    # the number the exterior-air path is measured against (test_exterior_air_throughput).
    return WarpEngine(
        grid=grid, cross_sections=xs, device="cuda:0", magnetic_field=field, exterior_air=None
    )


def _bench(benchmark, grid: VoxelGrid, field: MagneticField | None, source) -> float:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    engine = _engine(grid, field)
    engine.run(source, n_histories=10_000, n_batches=1, seed=SEED)  # compile + warm
    benchmark.pedantic(
        lambda: engine.run(source, n_histories=N, n_batches=4, seed=SEED),
        rounds=5,
        iterations=1,
    )
    if benchmark.stats is None:
        return float("nan")
    return N / benchmark.stats["mean"]


WATER_BEAM = ParallelBeamSource(energy=6.0, z=-1.0, x_range=(2.0, 17.0), y_range=(2.0, 17.0))
SLAB_BEAM = ParallelBeamSource(energy=6.0, z=-1.0, x_range=(5.0, 15.0), y_range=(5.0, 15.0))


@pytest.mark.parametrize(
    "field",
    [None, MagneticField.uniform((0.0, 0.0, 0.0)), MagneticField.uniform((1.5, 0.0, 0.0))],
    ids=["no-field", "zero-field", "1.5T"],
)
def test_water_cube(benchmark, field) -> None:
    """Field-free, explicit zero (branch cost) and 1.5 T (tissue cost) in water."""
    _bench(benchmark, _water_grid(), field, WATER_BEAM)


@pytest.mark.parametrize(
    "field",
    [None, MagneticField.uniform((0.35, 0.0, 0.0)), MagneticField.uniform((1.5, 0.0, 0.0))],
    ids=["no-field", "0.35T", "1.5T"],
)
def test_water_air_slab(benchmark, field) -> None:
    """The air cost: the gyration cap binds on every substep in the gap."""
    throughput = _bench(benchmark, _slab_grid(), field, SLAB_BEAM)
    if field is not None and np.isfinite(throughput):
        assert throughput >= FLOOR_HISTORIES_PER_S, (
            f"{throughput:.3e} histories/s in the field is below the catastrophic floor"
        )
