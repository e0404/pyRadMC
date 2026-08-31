"""End-to-end progress callback coverage: both backends, both methods, and the
multi-device / concurrent-batches fan-in paths that make ``ProgressEmitter``'s
thread-safety guarantee load-bearing rather than incidental.

``progress`` is a host-side, post-sync hook — it touches no RNG stream and no
fold order — so these tests pin behavior of the callback only, never dose or
the energy ledger (those stay covered by ``test_dij_ref.py``/``test_dij_warp.py``
and the multi-device/concurrent-batches suites).
"""

from __future__ import annotations

import math

import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, ParallelBeamSource
from pyradmc.progress import ProgressEvent
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

N_PER = 240
N_BATCHES = 4
GROUP = 2

FIELD_X = (2.0, 6.0)
FIELD_Y = (2.0, 6.0)
ENERGY = 6.0
Z0 = -1.0


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _lattice(n_x: int = 3, n_y: int = 2) -> BeamletGridSource:
    return BeamletGridSource(
        energy=ENERGY, z=Z0, x_range=FIELD_X, y_range=FIELD_Y, n_x=n_x, n_y=n_y
    )


def _open_field() -> ParallelBeamSource:
    return ParallelBeamSource(energy=ENERGY, z=Z0, x_range=FIELD_X, y_range=FIELD_Y)


def _assert_well_formed(events: list[ProgressEvent], total: int) -> None:
    """The one contract every ``progress`` consumer gets, regardless of backend."""
    assert events, "progress callback was never invoked"
    done = [e.histories_done for e in events]
    assert done == sorted(done), "histories_done must be non-decreasing"
    assert len(set(done)) == len(done), "histories_done must not repeat"
    assert done[-1] == total
    for e in events:
        assert e.histories_total == total
        assert e.elapsed_s >= 0.0
        assert e.rate_hz >= 0.0


def _ref_engine() -> ReferenceEngine:
    grid = _grid()
    return ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())


def test_ref_run_ticks_once_per_batch() -> None:
    n_histories = N_PER * N_BATCHES
    events: list[ProgressEvent] = []
    _ref_engine().run(
        _open_field(),
        n_histories=n_histories,
        n_batches=N_BATCHES,
        seed=SEED,
        progress=events.append,
    )
    _assert_well_formed(events, n_histories)
    assert len(events) == N_BATCHES


def test_ref_run_dij_ticks_once_per_batch() -> None:
    source = _lattice()
    events: list[ProgressEvent] = []
    _ref_engine().run_dij(
        source,
        n_histories_per_beamlet=N_PER,
        n_batches=N_BATCHES,
        seed=SEED,
        progress=events.append,
    )
    _assert_well_formed(events, source.n_beamlets * N_PER)
    assert len(events) == N_BATCHES


def test_progress_is_opt_in() -> None:
    """No ``progress`` argument must behave exactly as before: no crash, no cost."""
    _ref_engine().run(_open_field(), n_histories=N_PER, n_batches=N_BATCHES, seed=SEED)


wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from pyradmc.backends.warp.engine import WarpEngine  # noqa: E402


def _warp_engine(device: str = "cpu") -> WarpEngine:
    grid = _grid()
    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device, chunk_size=2048)


def test_warp_run_ticks_once_per_batch() -> None:
    n_histories = N_PER * N_BATCHES
    events: list[ProgressEvent] = []
    _warp_engine().run(
        _open_field(),
        n_histories=n_histories,
        n_batches=N_BATCHES,
        seed=SEED,
        progress=events.append,
    )
    _assert_well_formed(events, n_histories)
    assert len(events) == N_BATCHES


def test_warp_run_dij_ticks_once_per_beamlet_group() -> None:
    source = _lattice()
    events: list[ProgressEvent] = []
    _warp_engine().run_dij(
        source,
        n_histories_per_beamlet=N_PER,
        n_batches=N_BATCHES,
        seed=SEED,
        beamlet_group_size=GROUP,
        progress=events.append,
    )
    _assert_well_formed(events, source.n_beamlets * N_PER)
    assert len(events) == math.ceil(source.n_beamlets / GROUP)


@pytest.mark.gpu
def test_warp_run_dij_multi_device_shares_one_emitter() -> None:
    """Two distinct devices means two shard threads ticking the same emitter.

    ``devices=["cpu", "cpu"]`` collapses to one shard (see
    ``test_dij_multi_device.py::test_repeated_devices_collapse_to_one_shard``), so
    exercising the actual cross-thread fan-in needs a second physical device.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    source = _lattice(n_x=4, n_y=2)  # enough groups that both shards do real work
    events: list[ProgressEvent] = []
    _warp_engine("cuda:0").run_dij(
        source,
        n_histories_per_beamlet=N_PER,
        n_batches=N_BATCHES,
        seed=SEED,
        beamlet_group_size=GROUP,
        devices=["cuda:0", "cpu"],
        progress=events.append,
    )
    _assert_well_formed(events, source.n_beamlets * N_PER)


@pytest.mark.gpu
def test_warp_run_dij_concurrent_batches_ticks_once_per_group() -> None:
    """Lanes must not tick individually — only the shard's per-group point does."""
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    source = _lattice()
    events: list[ProgressEvent] = []
    _warp_engine("cuda:0").run_dij(
        source,
        n_histories_per_beamlet=N_PER,
        n_batches=N_BATCHES,
        seed=SEED,
        beamlet_group_size=GROUP,
        concurrent_batches=2,
        progress=events.append,
    )
    _assert_well_formed(events, source.n_beamlets * N_PER)
    assert len(events) == math.ceil(source.n_beamlets / GROUP)
