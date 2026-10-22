"""The Warp drain loop's blind-round schedule is bit-inert and drops nothing.

The photon and electron queues are drained in rounds: a photon launch, then an
electron launch, each consuming one queue and filling the others. The host-sized
schedule (``drain_rounds=0``) reads every queue count back before launching on it —
one device-to-host synchronization per launch. The blind schedule (``drain_rounds=k``)
launches each round at the queue's capacity, the kernels guarding against the
device-side count, and reads the counts back only once per ``k`` rounds; it keeps
issuing blocks until both queues are empty, so the end check, not ``k``, decides
termination. ``None`` adapts ``k`` to the round count the previous drain needed.

Each particle carries its own random-number state and every book and dose tally is
an integer atomic, so a run is a function of the *multiset* of particles each launch
consumes. An empty launch consumes nothing, so the schedule must not move a single
bit: pinned here against the one-round-at-a-time blind schedule, and that against
the host-sized schedule, on a workload (18 MeV electrons with bremsstrahlung) whose
queues stay busy for several rounds. Cross-device equality is *not* asserted
(AGENTS.md 2.3).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, PencilBeamSource
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyradmc.backends.warp.engine import WarpEngine  # noqa: E402
from tests.integration.test_run_concurrent_batches import HostSampledSource  # noqa: E402

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

ENERGY_BALANCE_RTOL = 1.0e-4  # float32 transport + 1e-9 MeV scoring quanta, as elsewhere


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _engine(device: str, drain_rounds: int | None, **kw) -> WarpEngine:
    grid = VoxelGrid.uniform_water(shape=(10, 10, 24), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    kw.setdefault("chunk_size", 256)
    return WarpEngine(grid=grid, cross_sections=xs, device=device, drain_rounds=drain_rounds, **kw)


def _run(engine: WarpEngine, **kw):
    """18 MeV electrons: deltas and bremsstrahlung keep both queues busy for rounds."""
    pencil = PencilBeamSource(energy=18.0, position=(5.0, 5.0, -1.0), direction=(0.0, 0.0, 1.0))
    return engine.run(
        pencil, n_histories=1_024, n_batches=4, seed=SEED, primary_kind="electron", **kw
    )


def _assert_run_bit_identical(got, want) -> None:
    np.testing.assert_array_equal(got.dose, want.dose)
    np.testing.assert_array_equal(got.dose_sigma, want.dose_sigma)
    assert got.energy_emitted == want.energy_emitted
    assert got.energy_deposited == want.energy_deposited
    assert got.energy_escaped == want.energy_escaped
    assert got.energy_unscored == want.energy_unscored


class TestBlindRoundsAreBitInert:
    def test_the_workload_spans_several_blocks(self, device: str) -> None:
        """Guard for the tests below: a drain here needs more rounds than one block.

        Otherwise a block size of 2 would never issue a second block, and the
        multi-block path — read the counts, find them non-empty, go again — would
        pass untested.
        """
        engine = _engine(device, 2)
        _run(engine)
        assert engine._drain.last_rounds[device] > 2

    @pytest.mark.parametrize("rounds", [2, 5, None])
    def test_blocks_match_one_round_at_a_time(self, device: str, rounds: int | None) -> None:
        _assert_run_bit_identical(_run(_engine(device, rounds)), _run(_engine(device, 1)))

    def test_one_round_blocks_match_the_host_sized_schedule(self, device: str) -> None:
        _assert_run_bit_identical(_run(_engine(device, 1)), _run(_engine(device, 0)))

    def test_adaptive_schedule_is_inert_across_repeated_runs(self, device: str) -> None:
        """The adaptive block size changes after the first drain; the result must not."""
        engine = _engine(device, None)
        first, second = _run(engine), _run(engine)
        _assert_run_bit_identical(second, first)
        _assert_run_bit_identical(first, _run(_engine(device, 0)))

    def test_presampled_route(self, device: str) -> None:
        """Host-seeded queues (both kinds at once) drain identically too."""
        results = [
            _engine(device, rounds).run(
                HostSampledSource(), n_histories=2_048, n_batches=2, seed=SEED
            )
            for rounds in (0, 3)
        ]
        _assert_run_bit_identical(results[1], results[0])

    def test_dij(self, device: str) -> None:
        lattice = BeamletGridSource(
            energy=6.0, z=-1.0, x_range=(2.0, 8.0), y_range=(2.0, 8.0), n_x=3, n_y=1
        )
        dijs = [
            _engine(device, rounds).run_dij(
                lattice, n_histories_per_beamlet=300, n_batches=3, seed=SEED, beamlet_group_size=2
            )
            for rounds in (0, 3)
        ]
        for name in ("indptr", "indices", "dose", "sigma"):
            np.testing.assert_array_equal(getattr(dijs[1], name), getattr(dijs[0], name))
        assert dijs[1].energy_deposited == dijs[0].energy_deposited
        assert dijs[1].energy_escaped == dijs[0].energy_escaped

    @pytest.mark.gpu
    def test_concurrent_batch_lanes(self) -> None:
        """Lane streams read their status through the lane's own stream."""
        if not wp.is_cuda_available():
            pytest.skip("no CUDA device")
        _assert_run_bit_identical(
            _run(_engine("cuda:0", 3), concurrent_batches=2),
            _run(_engine("cuda:0", 0)),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("rounds", [1, 3])
def test_one_block_grid_walks_every_slot(monkeypatch, rounds: int) -> None:
    """The grid-stride path: each thread transports many queued particles.

    At the shipped grid a test-sized queue has a thread per slot, so the kernels'
    count guard and Warp's grid-stride loop (``return`` becomes ``continue``) are
    never exercised the way a production queue holding more than a chunk does.
    A one-block grid makes 256 threads walk the whole capacity; the electron
    queue's ~900 delta rays in the second round are several real particles per
    thread. Bits must not move. (cpu has no grid: its launches loop every index.)
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    import pyradmc.backends.warp.engine as engine_module

    consulted = []

    def one_block(device: str, capacity: int, queue_factor: int) -> int:
        consulted.append(capacity)
        return 1

    monkeypatch.setattr(engine_module, "_blind_max_blocks", one_block)
    got = _run(_engine("cuda:0", rounds))
    assert consulted, "the blind schedule did not consult the grid size"
    monkeypatch.undo()
    _assert_run_bit_identical(got, _run(_engine("cuda:0", 0)))


@pytest.mark.gpu
def test_drain_buffers_gain_a_pinned_element_when_a_lane_first_uses_them() -> None:
    """A pooled workspace drained on the default stream, then on a lane stream,
    must get the pinned readback element the lane path copies into, and keep its
    status array across a capacity refit."""
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import _Workspace

    ws = _Workspace("cuda:0")
    ws.fit(64)
    status, host = ws.drain_buffers(None)
    assert host is None
    ws.fit(128)
    lane_status, lane_host = ws.drain_buffers(wp.Stream("cuda:0"))
    assert lane_status is status
    assert lane_host is not None
    assert lane_host.device.is_cpu
    assert lane_host.pinned


@pytest.mark.gpu
def test_pooled_workspace_runs_sequential_then_lanes() -> None:
    """One engine, its pooled workspace drained first without and then with lane
    streams: both match a fresh host-sized engine bit for bit."""
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    engine = _engine("cuda:0", 3)
    sequential = _run(engine)
    lanes = _run(engine, concurrent_batches=2)
    reference = _run(_engine("cuda:0", 0))
    _assert_run_bit_identical(sequential, reference)
    _assert_run_bit_identical(lanes, reference)


class TestNothingIsDropped:
    def test_energy_ledger_closes(self, device: str) -> None:
        """Every particle a blind block leaves queued is drained by a later block."""
        result = _run(_engine(device, 2))
        balance = result.energy_deposited + result.energy_escaped + result.energy_unscored
        assert balance == pytest.approx(result.energy_emitted, rel=ENERGY_BALANCE_RTOL)

    @pytest.mark.parametrize("rounds", [0, 1, 3])
    def test_queue_overflow_still_raises(self, device: str, rounds: int) -> None:
        """A blind launch clamps to capacity; the overflow must still surface.

        One queue slot per history cannot hold the delta rays an 18 MeV electron
        sets in motion (several per primary above the 200 keV cutoff), so the
        electron queue overflows in the first round. The blind kernels never read
        past capacity, and the overflow is recorded on the device and raised at the
        next readback — never silently dropped.
        """
        with pytest.raises(RuntimeError, match="overflow"):
            _run(_engine(device, rounds, queue_factor=1))


def test_negative_drain_rounds_is_rejected() -> None:
    with pytest.raises(ValueError, match="drain_rounds"):
        _engine("cpu", -1)


@pytest.mark.parametrize("presampled", [False, True])
def test_sequential_runs_drain_with_the_pooled_workspace(device, monkeypatch, presampled) -> None:
    """No throwaway workspace: every sequential chunk drains with the lane's pooled one.

    A drain called without a workspace builds a temporary one per chunk (fresh
    status storage each time); the sequential ``run`` routes must hand over the
    pooled workspace exactly as the lane and Dij routes do.
    """
    from pyradmc.backends.warp import engine as engine_module

    engine = _engine(device, None)
    source = HostSampledSource() if presampled else None
    kw = dict(n_histories=1_024, n_batches=4, seed=SEED, transport_electrons=False)
    if presampled:
        engine.run(source, **kw)
    else:
        _run(engine)
    built = []
    original = engine_module._Workspace.__init__

    def counting(self, *args, **kwargs):
        built.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(engine_module._Workspace, "__init__", counting)
    if presampled:
        engine.run(source, **kw)
    else:
        _run(engine)
    assert built == [], f"{len(built)} throwaway workspaces built during a sequential run"
