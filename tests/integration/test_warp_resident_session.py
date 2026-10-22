"""The Warp engine keeps its device state resident between calls.

A plan-optimization loop calls ``run`` many times on one engine with small history
counts, where re-uploading the cross-section tables, the voxel maps, and
reallocating the particle queues on every call was most of the wall time. The
engine now keeps that state on each device across ``run`` and ``run_dij``: tables
per ``(ecut, pcut, e_max, with_gs)`` key (least recently used evicted), the
transport grid, the last scoring geometry, and one pooled transport workspace
(queues, RNG slots, counters) per batch lane, refitted exactly to each call.

Residency is a scheduling change and must be bitwise inert on one device (AGENTS
2.3): every call on a warmed engine is compared exactly against the same call on a
brand-new engine. Nothing here compares across devices.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.collimation import TransmissionMaskSource
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import (
    BeamletGridSource,
    ParallelBeamSource,
    Primary,
    Source,
    SpectralBeamletSource,
    SpectralBeamSource,
)
from pyradmc.geometry.spectrum import ali_rogers_mv
from pyradmc.rng.host import uniform
from pyradmc.scoring.grid import ScoringGrid
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyradmc.backends.warp import engine as engine_module  # noqa: E402
from pyradmc.backends.warp.engine import WarpEngine  # noqa: E402

pytestmark = pytest.mark.warp

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return str(request.param)


class WobbleSource(Source):
    """A user source with no in-kernel sampler: transported by host pre-sampling."""

    @property
    def max_energy(self) -> float:
        return 6.0

    def emit(self, rng_state: object) -> Primary:
        x = 2.0 + 2.0 * uniform(rng_state)
        y = 2.0 + 1.0 * uniform(rng_state)
        return Primary(6.0, x, y, -1.0, 0.0, 0.0, 1.0)


class FailingSource(WobbleSource):
    """Pre-samples its first chunk, then raises: a call aborted mid-transport."""

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, np.ndarray]:
        if history_offset > 0:
            raise RuntimeError("source failed mid-run")
        return dict(super().sample_batch(seed, history_offset, n))


GRID = VoxelGrid.uniform_water(shape=(12, 10, 8), spacing=(0.5, 0.5, 0.5))
XS = AnalyticCrossSections(geometry_densities=GRID.max_density_by_material())
SPECTRUM = ali_rogers_mv("varian-6mv")
PARALLEL = ParallelBeamSource(energy=6.0, z=-1.0, x_range=(2.0, 4.0), y_range=(2.0, 3.0))
SPECTRAL = SpectralBeamSource(SPECTRUM, (3.0, 2.5, -100.0), (3.0, 2.5, 0.0), 2.0, 2.0)
_MASK = np.zeros((4, 4))
_MASK[1:3, 1:3] = 1.0
MASKED = TransmissionMaskSource(
    SPECTRAL, mask=_MASK, plane_center=(3.0, 2.5, 0.0), width_u=2.0, width_v=2.0
)
LATTICE = BeamletGridSource(
    energy=6.0, z=-1.0, x_range=(2.0, 4.0), y_range=(2.0, 3.0), n_x=2, n_y=1
)
LATTICE8 = BeamletGridSource(
    energy=6.0, z=-1.0, x_range=(1.0, 5.0), y_range=(2.0, 3.0), n_x=4, n_y=2
)
FAN = SpectralBeamletSource(
    SPECTRUM, (3.0, 2.5, -100.0), [(2.5, 2.5, 0.0), (3.5, 2.5, 0.0)], 1.0, 1.0
)
COARSE = ScoringGrid.rebin(GRID, shape=(6, 5, 4), spacing=(1.0, 1.0, 1.0), origin=GRID.origin)


def _engine(device: str, **kwargs) -> WarpEngine:
    return WarpEngine(grid=GRID, cross_sections=XS, device=device, **kwargs)


def _assert_run_equal(a, b) -> None:
    np.testing.assert_array_equal(a.dose, b.dose)
    np.testing.assert_array_equal(a.dose_sigma, b.dose_sigma)
    books_a = (a.energy_emitted, a.energy_deposited, a.energy_escaped, a.energy_unscored)
    books_b = (b.energy_emitted, b.energy_deposited, b.energy_escaped, b.energy_unscored)
    assert books_a == books_b


def _assert_dij_equal(a, b) -> None:
    np.testing.assert_array_equal(a.indptr, b.indptr)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.dose, b.dose)
    np.testing.assert_array_equal(a.sigma, b.sigma)
    books_a = (a.energy_emitted, a.energy_deposited, a.energy_escaped, a.energy_unscored)
    books_b = (b.energy_emitted, b.energy_deposited, b.energy_escaped, b.energy_unscored)
    assert books_a == books_b


def _assert_equal(method: str, a, b) -> None:
    if method == "run":
        _assert_run_equal(a, b)
    else:
        _assert_dij_equal(a, b)


def _run(source, n, batches, **kw) -> tuple[str, dict]:
    return "run", dict(source=source, n_histories=n, n_batches=batches, seed=SEED, **kw)


def _dij(source, n, batches, **kw) -> tuple[str, dict]:
    return "run_dij", dict(
        source=source, n_histories_per_beamlet=n, n_batches=batches, seed=SEED, **kw
    )


def _calls(device: str) -> list[tuple[str, dict]]:
    """An interleaving that changes every cache key between neighbouring calls."""
    calls = [
        _run(PARALLEL, 1_200, 4),
        _run(MASKED, 800, 2),
        _dij(LATTICE, 200, 2),
        _run(PARALLEL, 600, 3, ecut=0.5, msc_model="gaussian"),
        _run(SPECTRAL, 900, 3, scoring_grid=COARSE),
        _dij(FAN, 100, 2, beamlet_group_size=1),
        _run(PARALLEL, 64, 1),  # shrinks the pool
        _run(PARALLEL, 1_200, 4),  # regrows it
        _run(SPECTRAL, 600, 2, scoring_mode="dose_to_water"),
        _run(PARALLEL, 600, 2, transport_electrons=False),  # KERMA instrument
        _run(WobbleSource(), 400, 2),  # host pre-sampled route
        _dij(LATTICE, 200, 2, devices=[device]),
        _dij(LATTICE8, 100, 2, scoring_grid=COARSE, scoring_mode="dose_to_water"),
        _run(PARALLEL, 1_200, 4),
    ]
    if device.startswith("cuda"):
        calls += [
            _run(SPECTRAL, 900, 3, concurrent_batches=3),
            _dij(LATTICE, 200, 4, concurrent_batches=2),
            _run(PARALLEL, 1_200, 4),
        ]
    return calls


def test_warm_engine_is_bit_identical_to_fresh_engines(device: str) -> None:
    """Every call on one long-lived engine equals the same call on a new engine."""
    warm = _engine(device)
    for method, kwargs in _calls(device):
        got = getattr(warm, method)(**kwargs)
        want = getattr(_engine(device), method)(**kwargs)
        _assert_equal(method, got, want)


def test_warm_field_engine_is_bit_identical_to_fresh(device: str) -> None:
    field = MagneticField.uniform((0.0, 1.5, 0.0))
    warm = _engine(device, magnetic_field=field)
    for method, kwargs in [_run(PARALLEL, 600, 2), _dij(LATTICE, 100, 2), _run(MASKED, 600, 2)]:
        got = getattr(warm, method)(**kwargs)
        want = getattr(_engine(device, magnetic_field=field), method)(**kwargs)
        _assert_equal(method, got, want)


def test_repeated_seeded_job_is_bit_identical(device: str) -> None:
    """The same seeded run and Dij twice on one engine reproduce exactly."""
    engine = _engine(device)
    run_kwargs = dict(source=MASKED, n_histories=800, n_batches=2, seed=SEED)
    dij_kwargs = dict(source=FAN, n_histories_per_beamlet=100, n_batches=2, seed=SEED)
    first_run, first_dij = engine.run(**run_kwargs), engine.run_dij(**dij_kwargs)
    second_run, second_dij = engine.run(**run_kwargs), engine.run_dij(**dij_kwargs)
    _assert_run_equal(first_run, second_run)
    _assert_dij_equal(first_dij, second_dij)


def test_results_do_not_alias_resident_buffers(device: str) -> None:
    """A returned dose survives the next call (cpu ``numpy()`` is a view)."""
    engine = _engine(device)
    first = engine.run(PARALLEL, n_histories=600, n_batches=2, seed=SEED)
    kept = first.dose.copy(), first.dose_sigma.copy()
    engine.run(PARALLEL, n_histories=600, n_batches=2, seed=SEED + 7)
    np.testing.assert_array_equal(first.dose, kept[0])
    np.testing.assert_array_equal(first.dose_sigma, kept[1])


def test_device_tables_upload_once_per_key(device: str) -> None:
    engine = _engine(device)
    first = engine._upload_tables(6.0, pcut=0.05, ecut=0.2, device=device, with_gs=True)
    again = engine._upload_tables(6.0, pcut=0.05, ecut=0.2, device=device, with_gs=True)
    assert again is first
    other_cut = engine._upload_tables(6.0, pcut=0.05, ecut=0.5, device=device, with_gs=True)
    assert other_cut is not first
    no_gs = engine._upload_tables(6.0, pcut=0.05, ecut=0.2, device=device, with_gs=False)
    assert no_gs is not first
    # The large GS grid is shared by every pcut at one (ecut, e_max).
    other_pcut = engine._upload_tables(6.0, pcut=0.08, ecut=0.2, device=device, with_gs=True)
    assert other_pcut.gs_values.ptr == first.gs_values.ptr


def test_device_table_cache_is_bounded(device: str) -> None:
    """An ECUT sweep must not park one ~12 MB GS grid per cut on the device."""
    engine = _engine(device)
    first = engine._upload_tables(6.0, pcut=0.05, ecut=0.2, device=device, with_gs=True)
    bound = engine_module._TABLES_RESIDENT
    for k in range(bound):
        engine._upload_tables(6.0, pcut=0.05, ecut=0.3 + 0.05 * k, device=device, with_gs=True)
    state = engine._state(device)
    assert len(state.tables) == bound
    assert len(state.table_arrays) <= 2 * bound
    # The oldest key was evicted; asking again re-uploads it (a new struct).
    assert engine._upload_tables(6.0, pcut=0.05, ecut=0.2, device=device, with_gs=True) is not first


def test_grid_uploads_once_per_device(device: str) -> None:
    engine = _engine(device)
    gi, density, material = engine._upload_grid(device)
    again = engine._upload_grid(device)
    assert again[0] is gi and again[1] is density and again[2] is material


def test_run_dij_shares_the_default_scoring_upload_with_run(device: str) -> None:
    """Alternating run and run_dij on the default scoring grid uploads it once."""
    engine = _engine(device)
    engine.run(PARALLEL, n_histories=200, n_batches=2, seed=SEED)
    mass = engine._state(device).scoring[2]
    engine.run_dij(LATTICE, n_histories_per_beamlet=100, n_batches=2, seed=SEED)
    engine.run(PARALLEL, n_histories=200, n_batches=2, seed=SEED)
    assert engine._state(device).scoring[2] is mass


def test_equal_calls_allocate_no_queues_and_a_change_refits(device: str, monkeypatch) -> None:
    allocations = []
    real = engine_module._upload_queue

    def counting(capacity: int, dev: str):
        allocations.append(capacity)
        return real(capacity, dev)

    monkeypatch.setattr(engine_module, "_upload_queue", counting)
    engine = _engine(device)
    engine.run(PARALLEL, n_histories=600, n_batches=2, seed=SEED)
    assert len(allocations) == 4  # one lane: two ping-pong pairs
    engine.run(PARALLEL, n_histories=600, n_batches=2, seed=SEED + 1)
    assert len(allocations) == 4  # a steady workload stays resident
    engine.run(PARALLEL, n_histories=60, n_batches=2, seed=SEED + 1)
    assert allocations[4:] == [60 * engine.queue_factor] * 4  # shrinks to the call
    engine.run(PARALLEL, n_histories=60, n_batches=2, seed=SEED + 2)
    assert len(allocations) == 8


def test_pool_holds_what_the_last_call_needed(device: str) -> None:
    """Never the largest call so far: a tiny call after a big one shrinks the pool."""
    engine = _engine(device, chunk_size=4096)
    engine.run(PARALLEL, n_histories=4096, n_batches=1, seed=SEED)
    assert engine._idle_workspace_bytes(device) == engine_module._queue_bytes(4096, 16, 1)
    engine.run(PARALLEL, n_histories=64, n_batches=1, seed=SEED)
    assert engine._idle_workspace_bytes(device) == engine_module._queue_bytes(64, 16, 1)
    engine.run_dij(LATTICE, n_histories_per_beamlet=200, n_batches=2, seed=SEED)
    assert engine._idle_workspace_bytes(device) == engine_module._queue_bytes(2 * 100, 16, 1)


def test_refit_binds_the_queues_to_the_request(device: str) -> None:
    """The kernels guard writes at ``q.capacity``; it must be this call's request."""
    engine = _engine(device)
    big = engine._acquire_workspaces(device, 1, 1_000)
    engine._release_workspaces(device, big)
    small = engine._acquire_workspaces(device, 1, 100)
    assert small[0] is big[0]
    assert all(q.capacity == 100 and q.kind.shape[0] == 100 for q in small[0].queues)
    assert small[0].slots.shape[0] == 100
    engine._release_workspaces(device, small)


def test_idle_workspace_bytes_match_the_sizing_model(device: str) -> None:
    """The pooled queues are what ``_QUEUE_BYTES_PER_SLOT`` charges per slot.

    The Dij auto sizing counts idle pooled bytes as free (an acquirer reuses or
    releases them before allocating), so the two must use one accounting.
    """
    engine = _engine(device, chunk_size=256)
    engine.run(PARALLEL, n_histories=512, n_batches=2, seed=SEED)
    expected = engine_module._queue_bytes(256, engine.queue_factor, 1)
    assert engine._idle_workspace_bytes(device) == expected


def test_dij_auto_group_size_after_a_run_equals_a_fresh_engine(device: str, monkeypatch) -> None:
    """The pool's idle bytes are budgeted as free, so a warm Dij sizes like a fresh one.

    The device is simulated: a fixed total, minus what the engine's pool parks
    between calls (as the driver would report it). The total is chosen so the dense
    budget binds at three of the eight beamlets.
    """
    chunk, qf, n_voxels = 64, 16, GRID.density.size
    queue_bytes = engine_module._queue_bytes(chunk, qf, 1)
    total = queue_bytes + 2 * (3 * 24 * n_voxels) + 10_000
    current: list[WarpEngine] = []
    monkeypatch.setattr(
        engine_module,
        "_device_free_bytes",
        lambda d: total - current[0]._idle_workspace_bytes(d),
    )
    sizes: list[int] = []
    real_shard = WarpEngine._run_dij_shard

    def spy(self, dev, next_group, **kwargs):
        sizes.append(kwargs["beamlet_group_size"])
        return real_shard(self, dev, next_group, **kwargs)

    monkeypatch.setattr(WarpEngine, "_run_dij_shard", spy)
    dij = dict(source=LATTICE8, n_histories_per_beamlet=100, n_batches=2, seed=SEED)

    fresh = _engine(device, chunk_size=chunk)
    current[:] = [fresh]
    want = fresh.run_dij(**dij)

    warm = _engine(device, chunk_size=chunk)
    current[:] = [warm]
    warm.run(PARALLEL, n_histories=4 * chunk, n_batches=1, seed=SEED)
    assert warm._idle_workspace_bytes(device) > 0
    got = warm.run_dij(**dij)

    assert sizes == [3, 3]
    _assert_dij_equal(got, want)


def test_release_device_memory_drops_state_and_stays_identical(device: str) -> None:
    engine = _engine(device)
    first = engine.run(MASKED, n_histories=800, n_batches=2, seed=SEED)
    engine.release_device_memory()
    assert engine._idle_workspace_bytes(device) == 0
    again = engine.run(MASKED, n_histories=800, n_batches=2, seed=SEED)
    _assert_run_equal(first, again)


def test_majorant_violation_raises_in_the_first_pass(device: str) -> None:
    """The violation count rides the per-batch / per-group book readback.

    The tables declare water at half the density the grid holds, so the majorant
    is too small and the kernels count violations. The raise must come in the pass
    that made the violation — before its progress tick — not at the end of the call.
    """
    from pyradmc.data.materials import WATER

    thin = AnalyticCrossSections(geometry_densities=((WATER, 0.5),))
    engine = WarpEngine(grid=GRID, cross_sections=thin, device=device)
    ticks: list[object] = []
    for _ in range(2):
        with pytest.raises(RuntimeError, match="majorant violated"):
            engine.run(PARALLEL, n_histories=600, n_batches=3, seed=SEED, progress=ticks.append)
        with pytest.raises(RuntimeError, match="majorant violated"):
            engine.run_dij(
                LATTICE,
                n_histories_per_beamlet=200,
                n_batches=2,
                seed=SEED,
                beamlet_group_size=1,
                progress=ticks.append,
            )
    assert ticks == []
    assert engine._idle_workspace_bytes(device) == 0  # aborted calls park nothing


def test_aborted_call_leaves_the_engine_bit_identical(device: str) -> None:
    """After a call dies mid-transport, the next calls equal a fresh engine's."""
    engine = _engine(device, chunk_size=128)
    with pytest.raises(RuntimeError, match="source failed mid-run"):
        engine.run(FailingSource(), n_histories=512, n_batches=2, seed=SEED)
    for method, kwargs in [_run(MASKED, 512, 2), _dij(LATTICE, 100, 2), _run(PARALLEL, 512, 2)]:
        got = getattr(engine, method)(**kwargs)
        want = getattr(_engine(device, chunk_size=128), method)(**kwargs)
        _assert_equal(method, got, want)


def test_queue_overflow_after_a_bigger_call_raises_like_a_fresh_engine(device: str) -> None:
    """Overflow binds at this call's capacity whatever the pool held before.

    ``queue_factor=1`` leaves no room for secondaries: eight 15 MeV electrons
    overflow their queue on the first round. On a pool previously fitted to a much
    larger call the error must be the same, count and capacity included, and the
    engine must stay usable.
    """
    electrons = dict(
        source=ParallelBeamSource(energy=15.0, z=-1.0, x_range=(2.0, 4.0), y_range=(2.0, 3.0)),
        n_histories=8,
        n_batches=1,
        seed=SEED,
        primary_kind="electron",
    )
    with pytest.raises(RuntimeError, match="overflow") as fresh_error:
        _engine(device, queue_factor=1).run(**electrons)

    engine = _engine(device, queue_factor=1)
    engine.run(PARALLEL, n_histories=2_048, n_batches=1, seed=SEED, transport_electrons=False)
    with pytest.raises(RuntimeError, match="overflow") as warm_error:
        engine.run(**electrons)
    assert str(warm_error.value) == str(fresh_error.value)

    after = engine.run(
        PARALLEL, n_histories=1_024, n_batches=2, seed=SEED, transport_electrons=False
    )
    want = _engine(device, queue_factor=1).run(
        PARALLEL, n_histories=1_024, n_batches=2, seed=SEED, transport_electrons=False
    )
    _assert_run_equal(after, want)
