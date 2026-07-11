"""Warp backend throughput benchmarks. Alerts, not pass/fail — with one exception.

The workload is the Phase 2 exit scenario: a 6 MeV parallel field in a 32 cm water
cube with full coupled transport. Benchmarks record wall time via pytest-benchmark
(``make bench``); regressions surface as baseline drift, not failures.

The single hard assertion is the Phase 2 exit criterion itself: at least 1e6
histories per second on a CUDA device. Measured headroom at adoption was ~19x on a
laptop RTX 4070, so tripping this floor means something is catastrophically wrong,
not merely slow. The reference backend is deliberately unbenchmarked (AGENTS.md 2.2).
"""

from __future__ import annotations

import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.perf, pytest.mark.warp]

GPU_EXIT_CRITERION_HISTORIES_PER_S = 1.0e6


def _engine(device: str):
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(64, 64, 64), spacing=(0.5, 0.5, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device=device, chunk_size=262_144)


def _source() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(11.0, 21.0), y_range=(11.0, 21.0))


@pytest.mark.gpu
def test_cuda_throughput_meets_phase2_exit_criterion(benchmark) -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    engine = _engine("cuda:0")
    engine.run(_source(), n_histories=10_000, n_batches=1, seed=SEED)  # compile+warm

    n = 1_000_000
    benchmark.pedantic(
        lambda: engine.run(_source(), n_histories=n, n_batches=10, seed=SEED),
        rounds=3,
        iterations=1,
    )
    throughput = n / benchmark.stats["mean"]
    assert throughput >= GPU_EXIT_CRITERION_HISTORIES_PER_S, (
        f"{throughput:.3e} histories/s is below the Phase 2 exit criterion"
    )


def test_warp_cpu_throughput(benchmark) -> None:
    """CPU baseline for regression tracking; no threshold.

    Also the measurement feeding the plan's numba guard rail (IMPLEMENTATION_PLAN
    section 0): if a numba backend ever exists, Warp-CPU must be within ~2x of it.
    """
    engine = _engine("cpu")
    engine.run(_source(), n_histories=2_000, n_batches=1, seed=SEED)  # compile+warm

    n = 50_000
    benchmark.pedantic(
        lambda: engine.run(_source(), n_histories=n, n_batches=10, seed=SEED),
        rounds=3,
        iterations=1,
    )
