"""Dij pipeline throughput benchmark. Alerts, not pass/fail — with one floor.

The workload is a planning-shaped scenario: a 10x10 beamlet lattice of a
6 MeV field on a 64^3 water phantom, full coupled transport, batched sigma, sparse
assembly at the default truncation. Statistics are set near the planning target
(the phase criterion is 2-3 percent per-beamlet high-dose sigma), because the Dij
pipeline carries a fixed host-side cost per (group, batch, voxel) block — dense
readback plus batch statistics — that only amortizes at realistic history counts.
Benchmarking at toy statistics would measure that fixed cost, not the engine.

Measured at adoption (laptop RTX 4070, 4e5 histories/beamlet -> ~3.9 percent
sigma): ~1.1e7 histories/s end-to-end, ~3.5 s wall. The hard floor is 2e6
histories/s — a 5x margin, so tripping it means something is structurally wrong
with the pipeline, not merely slow.
"""

from __future__ import annotations

import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletGridSource
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.perf, pytest.mark.warp]

DIJ_FLOOR_HISTORIES_PER_S = 2.0e6

N_PER_BEAMLET = 400_000
N_BEAMLETS = 100


def _engine(device: str):
    """Engine at shipped defaults — the chunk size auto-sizes from the device.

    Deliberately not overridden. These benchmarks previously pinned it, which is
    how a fixed default costing 1.82x on open-field throughput survived unnoticed:
    nothing that was timed ever ran at the value users got. A benchmark that does
    not exercise the default cannot defend it.
    """
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(64, 64, 64), spacing=(0.4, 0.4, 0.4))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device=device)


def _lattice() -> BeamletGridSource:
    return BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(2.8, 22.8), y_range=(2.8, 22.8), n_x=10, n_y=10
    )


@pytest.mark.gpu
def test_cuda_dij_throughput_at_planning_statistics(benchmark) -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    engine = _engine("cuda:0")
    engine.run_dij(_lattice(), n_histories_per_beamlet=1_000, n_batches=1, seed=SEED)  # warm

    n = N_BEAMLETS * N_PER_BEAMLET
    benchmark.pedantic(
        lambda: engine.run_dij(
            _lattice(),
            n_histories_per_beamlet=N_PER_BEAMLET,
            n_batches=8,
            seed=SEED,
            beamlet_group_size=50,
        ),
        rounds=3,
        iterations=1,
    )
    if benchmark.stats is None:
        pytest.skip("benchmarking disabled; the throughput floor needs timing data")
    throughput = n / benchmark.stats["mean"]
    assert throughput >= DIJ_FLOOR_HISTORIES_PER_S, (
        f"{throughput:.3e} histories/s is below the Dij pipeline floor"
    )


def test_warp_cpu_dij_throughput(benchmark) -> None:
    """CPU baseline for regression tracking; no threshold, toy statistics.

    On CPU the transport itself is the cost at any statistics, so the fixed-cost
    amortization argument above does not apply and small numbers suffice.
    """
    engine = _engine("cpu")
    engine.run_dij(_lattice(), n_histories_per_beamlet=100, n_batches=1, seed=SEED)  # warm

    benchmark.pedantic(
        lambda: engine.run_dij(_lattice(), n_histories_per_beamlet=500, n_batches=5, seed=SEED),
        rounds=3,
        iterations=1,
    )
