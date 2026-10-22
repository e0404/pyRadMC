"""The Warp engine keeps its device state resident between calls.

A plan-optimization loop calls ``run`` many times on one engine with small history
counts, where re-uploading the cross-section tables, the voxel maps, and
reallocating the particle queues on every call was most of the wall time. The
engine now keeps that state on each device across ``run`` and ``run_dij``: tables
per ``(ecut, pcut, e_max)`` key, the transport grid, the last scoring geometry, and
one pooled transport workspace (queues, RNG slots, counters) per batch lane, grown
on demand.

Residency is a scheduling change and must be bitwise inert on one device (AGENTS
2.3): every call on a warmed engine is compared exactly against the same call on a
brand-new engine. Nothing here compares across devices.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.collimation import TransmissionMaskSource
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import (
    BeamletGridSource,
    ParallelBeamSource,
    SpectralBeamletSource,
    SpectralBeamSource,
)
from pyradmc.geometry.spectrum import ali_rogers_mv
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


def _calls(device: str) -> list[tuple[str, dict]]:
    """An interleaving that changes every cache key between neighbouring calls."""
    calls: list[tuple[str, dict]] = [
        ("run", dict(source=PARALLEL, n_histories=1_200, n_batches=4, seed=SEED)),
        ("run", dict(source=MASKED, n_histories=800, n_batches=2, seed=SEED)),
        ("run_dij", dict(source=LATTICE, n_histories_per_beamlet=200, n_batches=2, seed=SEED)),
        (
            "run",
            dict(
                source=PARALLEL,
                n_histories=600,
                n_batches=3,
                seed=SEED + 1,
                ecut=0.5,
                msc_model="gaussian",
            ),
        ),
        (
            "run",
            dict(source=SPECTRAL, n_histories=900, n_batches=3, seed=SEED, scoring_grid=COARSE),
        ),
        (
            "run_dij",
            dict(
                source=FAN,
                n_histories_per_beamlet=100,
                n_batches=2,
                seed=SEED,
                beamlet_group_size=1,
            ),
        ),
        ("run", dict(source=PARALLEL, n_histories=64, n_batches=1, seed=SEED)),  # shrinks
        ("run", dict(source=PARALLEL, n_histories=1_200, n_batches=4, seed=SEED)),  # regrows
    ]
    if device.startswith("cuda"):
        calls += [
            (
                "run",
                dict(
                    source=SPECTRAL,
                    n_histories=900,
                    n_batches=3,
                    seed=SEED,
                    concurrent_batches=3,
                ),
            ),
            (
                "run_dij",
                dict(
                    source=LATTICE,
                    n_histories_per_beamlet=200,
                    n_batches=4,
                    seed=SEED,
                    concurrent_batches=2,
                ),
            ),
            ("run", dict(source=PARALLEL, n_histories=1_200, n_batches=4, seed=SEED)),
        ]
    return calls


def test_warm_engine_is_bit_identical_to_fresh_engines(device: str) -> None:
    """Every call on one long-lived engine equals the same call on a new engine."""
    warm = _engine(device)
    for method, kwargs in _calls(device):
        got = getattr(warm, method)(**kwargs)
        want = getattr(_engine(device), method)(**kwargs)
        if method == "run":
            _assert_run_equal(got, want)
        else:
            _assert_dij_equal(got, want)


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


def test_grid_uploads_once_per_device(device: str) -> None:
    engine = _engine(device)
    gi, density, material = engine._upload_grid(device)
    again = engine._upload_grid(device)
    assert again[0] is gi and again[1] is density and again[2] is material


def test_second_run_allocates_no_queues(device: str, monkeypatch) -> None:
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
    engine.run(PARALLEL, n_histories=60, n_batches=2, seed=SEED + 1)  # fits: reused
    assert len(allocations) == 4


def test_dij_refits_queues_to_its_own_size(device: str) -> None:
    """A Dij does not ride on a bigger run's queues: its peak stays a fresh engine's.

    The dense group maps are budgeted against free memory, so a shard refits the
    pooled queues to exactly what it needs (here 2 beamlets x 100 histories per
    batch x queue_factor), releasing a larger run's allocation first.
    """
    engine = _engine(device, chunk_size=4096)
    engine.run(PARALLEL, n_histories=4096, n_batches=1, seed=SEED)
    assert engine._idle_workspace_bytes(device) == engine_module._queue_bytes(4096, 16, 1)
    engine.run_dij(LATTICE, n_histories_per_beamlet=200, n_batches=2, seed=SEED)
    assert engine._idle_workspace_bytes(device) == engine_module._queue_bytes(2 * 100, 16, 1)


def test_reused_queues_keep_the_requested_capacity(device: str) -> None:
    """A smaller request on a grown workspace still guards at its own capacity.

    The kernels refuse writes at ``q.capacity`` and the engine raises on a count
    past it; that bound must be the one this call asked for, not the length the
    buffer happens to have from an earlier, larger call, so overflow behaves
    exactly as on a fresh engine.
    """
    engine = _engine(device)
    big = engine._acquire_workspaces(device, 1, 1_000)
    engine._release_workspaces(device, big)
    small = engine._acquire_workspaces(device, 1, 100)
    assert small[0] is big[0]
    assert all(q.capacity == 100 for q in small[0].queues)
    assert all(q.kind.shape[0] >= 100 for q in small[0].queues)
    engine._release_workspaces(device, small)


def test_idle_workspace_bytes_match_the_sizing_model(device: str) -> None:
    """The pooled queues are what ``_QUEUE_BYTES_PER_SLOT`` charges per slot.

    Run-to-run auto sizing counts idle pooled bytes as free (an acquirer reuses or
    releases them before allocating), so the two must use one accounting.
    """
    engine = _engine(device, chunk_size=256)
    engine.run(PARALLEL, n_histories=512, n_batches=2, seed=SEED)
    expected = engine_module._queue_bytes(256, engine.queue_factor, 1)
    assert engine._idle_workspace_bytes(device) == expected


def test_release_device_memory_drops_state_and_stays_identical(device: str) -> None:
    engine = _engine(device)
    first = engine.run(MASKED, n_histories=800, n_batches=2, seed=SEED)
    engine.release_device_memory()
    assert engine._idle_workspace_bytes(device) == 0
    again = engine.run(MASKED, n_histories=800, n_batches=2, seed=SEED)
    _assert_run_equal(first, again)


def test_majorant_violation_still_raises(device: str) -> None:
    """The check now runs once per call (run) or shard (Dij); it must still fire.

    The tables declare water at half the density the grid holds, so the majorant
    is too small and the kernels count violations. An aborted call must also leave
    the engine usable: its workspaces are dropped, not returned to the pool.
    """
    from pyradmc.data.materials import WATER

    thin = AnalyticCrossSections(geometry_densities=((WATER, 0.5),))
    engine = WarpEngine(grid=GRID, cross_sections=thin, device=device)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="majorant violated"):
            engine.run(PARALLEL, n_histories=600, n_batches=3, seed=SEED)
        with pytest.raises(RuntimeError, match="majorant violated"):
            engine.run_dij(LATTICE, n_histories_per_beamlet=200, n_batches=2, seed=SEED)
    assert engine._idle_workspace_bytes(device) == 0
