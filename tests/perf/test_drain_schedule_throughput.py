"""Drain-schedule benchmarks: the optimization-loop workload and an open field.

Alerts, not pass/fail. Each workload runs under the shipped schedule (``drain_rounds``
default: blind rounds with an adaptive block on CUDA) and under the host-sized
schedule (``drain_rounds=0``, one count readback per launch), so a regression in
either — or the gap between them closing — shows up as baseline drift.

The collimated case is what a plan or aperture optimizer calls in a loop: a small
water grid (48x40x24 at 5 mm), a masked 6 MV-like spectral fan, 4e5 histories,
one batch, ten calls per round. Its transport kernels are a small fraction of the
wall (about 1.3 of 15 ms per call, measured 2026-10-22), so host round trips and
per-run setup dominate; this is where the readbacks were visible. The open field
(64^3, 6 MeV, 1e6 histories, ten batches) is the throughput workload of
``test_warp_throughput.py``, here to show the schedule costs nothing when kernels
dominate. Check GPU clocks before believing a regression (docs/decisions.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.collimation import TransmissionMaskSource
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource, SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.perf, pytest.mark.warp, pytest.mark.gpu]

SCHEDULES = [
    pytest.param(None, id="default"),
    pytest.param(0, id="host-sized"),
]


def _engine(shape, spacing, drain_rounds):
    from pyradmc.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=shape, spacing=spacing)
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device="cuda:0", drain_rounds=drain_rounds)


def _collimated_source() -> TransmissionMaskSource:
    spectrum = Spectrum((0.25, 0.5, 1.0, 2.0, 4.0, 6.0), (0.6, 1.0, 1.4, 1.0, 0.4))
    inner = SpectralBeamSource(
        spectrum,
        focal_point=(12.0, 10.0, -90.0),
        center=(12.0, 10.0, 0.0),
        width_u=10.0,
        width_v=10.0,
    )
    mask = np.zeros((20, 20))
    mask[6:14, 4:16] = 1.0
    return TransmissionMaskSource(
        inner, mask=mask, plane_center=(12.0, 10.0, 0.0), width_u=10.0, width_v=10.0
    )


@pytest.fixture(autouse=True)
def _cuda() -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")


@pytest.mark.parametrize("drain_rounds", SCHEDULES)
def test_collimated_optimization_loop(benchmark, drain_rounds) -> None:
    """Ten small collimated runs, the shape of one optimizer iteration's calls."""
    engine = _engine((48, 40, 24), (0.5, 0.5, 0.5), drain_rounds)
    source = _collimated_source()
    engine.run(source, n_histories=400_000, n_batches=1, seed=SEED)  # compile, warm, adapt

    def loop() -> None:
        for _ in range(10):
            engine.run(source, n_histories=400_000, n_batches=1, seed=SEED)

    benchmark.pedantic(loop, rounds=5, iterations=1)


@pytest.mark.parametrize("drain_rounds", SCHEDULES)
def test_open_field(benchmark, drain_rounds) -> None:
    engine = _engine((64, 64, 64), (0.5, 0.5, 0.5), drain_rounds)
    source = ParallelBeamSource(energy=6.0, z=-1.0, x_range=(11.0, 21.0), y_range=(11.0, 21.0))
    engine.run(source, n_histories=10_000, n_batches=1, seed=SEED)  # compile, warm, adapt
    benchmark.pedantic(
        lambda: engine.run(source, n_histories=1_000_000, n_batches=10, seed=SEED),
        rounds=5,
        iterations=1,
    )
