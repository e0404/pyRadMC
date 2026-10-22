"""Warp per-call setup overhead: the plan-optimization loop shape. Alerts, not pass/fail.

A plan optimization calls ``run`` many times on one engine, each with few (or
cheaply collimated) histories, so the fixed cost of a call — uploads, allocations,
synchronizing readbacks — rather than transport sets its wall time. These record
that cost so a regression in the resident device state shows up as baseline drift:

- ``loop``: ten consecutive 6 MV runs through a mostly closed transmission mask on a
  48x40x24 water grid (the measured optimization shape: 400k histories each on
  cuda, where transport kernels are a small fraction of the wall);
- ``tiny run`` / ``tiny Dij``: one call with a handful of histories, i.e. the fixed
  cost alone.

Measured 2026-10-22 on the laptop RTX 4070 (shared with other jobs; interleaved
A/B, best-of ranges), resident state vs the per-call setup it replaced, cuda: loop
111-120 -> 20-22 ms, tiny run 7.4-8.7 -> 0.88-0.92 ms, tiny Dij 8.4-8.7 -> 1.62-1.69
ms; cpu: loop 395-423 -> 72 ms, tiny run 7.4-7.7 -> 0.94-0.98 ms, tiny Dij 23-24 ->
15.5-16.6 ms. The full table is in ``docs/decisions.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.collimation import TransmissionMaskSource
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.perf, pytest.mark.warp]

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return str(request.param)


def _setup(device: str):
    from pyradmc.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(48, 40, 24), spacing=(0.5, 0.5, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    fan = SpectralBeamSource(
        ali_rogers_mv("varian-6mv"), (12.0, 10.0, -100.0), (12.0, 10.0, 0.0), 16.0, 16.0
    )
    mask = np.zeros((32, 32))
    mask[12:20, 13:19] = 1.0  # a 4 x 3 cm opening in a 16 x 16 cm fan
    source = TransmissionMaskSource(
        fan, mask=mask, plane_center=(12.0, 10.0, 0.0), width_u=16.0, width_v=16.0
    )
    lattice = BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(8.0, 16.0), y_range=(6.0, 14.0), n_x=4, n_y=4
    )
    engine = WarpEngine(grid=grid, cross_sections=xs, device=device)
    n = 400_000 if device.startswith("cuda") else 40_000
    engine.run(source, n, 1, SEED)  # compile and first touch of the resident state
    engine.run_dij(lattice, 16, 1, SEED)
    return engine, source, lattice, n


def test_consecutive_collimated_runs(benchmark, device: str) -> None:
    engine, source, _, n = _setup(device)

    def loop() -> None:
        for k in range(10):
            engine.run(source, n, 1, SEED + k)

    benchmark.pedantic(loop, rounds=5, iterations=1)


def test_tiny_run_fixed_cost(benchmark, device: str) -> None:
    engine, source, _, _ = _setup(device)
    benchmark.pedantic(lambda: engine.run(source, 64, 1, SEED), rounds=30, iterations=1)


def test_tiny_dij_fixed_cost(benchmark, device: str) -> None:
    engine, _, lattice, _ = _setup(device)
    benchmark.pedantic(lambda: engine.run_dij(lattice, 16, 1, SEED), rounds=15, iterations=1)
