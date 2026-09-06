"""Fresh-process persistence of the reference sampler's deterministic GS nodes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_grid_cache", {})
    monkeypatch.setattr(gs, "_lazy_grid_cache", {})
    monkeypatch.setattr(gs, "_node_cache", {})


SCRIPT = """
import sys
from pathlib import Path
import numpy as np
import pyradmc.data.goudsmit_saunderson as gs
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.rng.host import HostRNG

gs._GS_GRID_CACHE_DIR = Path(sys.argv[1])
if sys.argv[2] == 'read':
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError('persisted reference nodes must not rebuild')
    gs._gs_raw_cosines = refuse
xs = AnalyticCrossSections()
rng = HostRNG().init_state(42, 0)
values = [xs.sample_gs_cos_theta(1e-12, 1.0, 0, rng) for _ in range(8)]
np.save(sys.argv[3], values)
"""


def test_reference_nodes_survive_a_fresh_process(tmp_path: Path) -> None:
    """A cold reference sample writes the grid; a fresh interpreter only reads."""
    cache = tmp_path / "cache"
    for mode in ("build", "read"):
        subprocess.run(
            [sys.executable, "-c", SCRIPT, str(cache), mode, str(tmp_path / f"{mode}.npy")],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert list(cache.glob("gs-grid-*.npz")), "reference nodes were not persisted"
    assert (tmp_path / "build.npy").read_bytes() == (tmp_path / "read.npy").read_bytes()


def test_reference_reads_the_existing_eager_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original rectangular format is usable without rebuilding or rewriting."""
    import math

    import pyradmc.data.goudsmit_saunderson as gs
    from pyradmc.data.analytic import AnalyticCrossSections
    from pyradmc.rng.host import HostRNG

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    xs = AnalyticCrossSections()
    log_eta = math.log(xs.elastic_screening(1.0, 0))
    grid = gs.build_gs_grid(log_eta, log_eta, 2e-12, cache_dir=tmp_path)
    path = gs._grid_cache_file(tmp_path)
    before = path.read_bytes()
    gs._node_cache.clear()
    gs._grid_cache.clear()

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the reference sampler ignored a stored eager grid")

    monkeypatch.setattr(gs, "_gs_raw_cosines", refuse)
    monkeypatch.setattr(gs, "_save_grid_cache", refuse)
    value = xs.sample_gs_cos_theta(1e-12, 1.0, 0, HostRNG().init_state(42, 0))
    assert np.isfinite(value)
    assert np.isfinite(grid.values).all()
    assert path.read_bytes() == before


def test_lazy_extension_reuses_every_existing_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    first = gs.gs_grid_node(-46, -111).copy()
    original = gs.gs_scaled_deflection_table
    built: list[tuple[float, float]] = []

    def recording(eta: float, theta2: float, **kwargs: int) -> np.ndarray:
        built.append((eta, theta2))
        return original(eta, theta2, **kwargs)

    monkeypatch.setattr(gs, "gs_scaled_deflection_table", recording)
    last = gs.gs_grid_node(-45, -110)
    assert len(built) == 3
    assert not last.flags.writeable
    stored = gs._load_grid_cache(gs._grid_cache_file(tmp_path))
    assert stored is not None
    assert stored.values.shape == (2, 2, 512)
    assert stored.values[0, 0].tobytes() == first.tobytes()


def test_threaded_reference_misses_share_the_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: gs.gs_grid_node(-46, -111), range(8)))
    assert all(not row.flags.writeable for row in rows)
    assert all(row.tobytes() == rows[0].tobytes() for row in rows)
    stored = gs._load_grid_cache(gs._grid_cache_file(tmp_path))
    assert stored is not None
    assert stored.values.shape == (1, 1, 512)
    assert not list(tmp_path.glob("*.partial*"))


def test_reference_extension_preserves_a_later_eager_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    gs.gs_grid_node(-46, -111)
    eager = gs.build_gs_grid(-12.0, -11.5, 2e-12, cache_dir=tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a node covered by the latest disk grid needs no rewrite")

    monkeypatch.setattr(gs, "_save_grid_cache", refuse)
    monkeypatch.setattr(gs, "gs_scaled_deflection_table", refuse)
    gs.gs_grid_node(-45, -111)
    stored = gs._load_grid_cache(gs._grid_cache_file(tmp_path))
    assert stored is not None
    assert stored.ix0 == eager.ix0
    assert stored.values.shape[0] >= eager.values.shape[0]
    assert stored.values.shape[1] >= eager.values.shape[1]
    assert stored.values[: eager.values.shape[0], : eager.values.shape[1]].tobytes() == (
        eager.values.tobytes()
    )


def test_reference_cache_write_failure_keeps_the_node_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)

    def readonly(*args: object, **kwargs: object) -> None:
        raise PermissionError("read-only cache directory")

    monkeypatch.setattr(gs, "_save_grid_cache", readonly)
    first = gs.gs_grid_node(-46, -111)
    second = gs.gs_grid_node(-46, -111)
    assert first.tobytes() == second.tobytes()
    assert not second.flags.writeable


def test_nontrivial_stored_nodes_are_reused_bit_for_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math

    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    values = np.stack(
        [
            [gs.gs_scaled_deflection_table(math.exp(i / 4), math.exp(j / 4)) for j in (-25, -24)]
            for i in (-47, -46)
        ]
    )
    assert np.any(values != 0)
    assert values[0, 0].tobytes() != values[1, 1].tobytes()
    gs._save_grid_cache(gs._grid_cache_file(tmp_path), gs.GSGridTables(values, -47, -25, 512, 4.0))
    gs._node_cache.clear()

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("stored nontrivial nodes must not rebuild")

    monkeypatch.setattr(gs, "_gs_raw_cosines", refuse)
    for i in range(2):
        for j in range(2):
            actual = gs.gs_grid_node(-47 + i, -25 + j)
            assert actual.tobytes() == values[i, j].tobytes()
            assert not actual.flags.writeable


def test_eager_and_reference_builds_share_the_write_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        lazy = pool.submit(gs.gs_grid_node, -45, -110)
        eager = pool.submit(gs.build_gs_grid, -12.0, -11.5, 2e-12, cache_dir=tmp_path)
        row, grid = lazy.result(timeout=10), eager.result(timeout=10)
    stored = gs._load_grid_cache(gs._grid_cache_file(tmp_path))
    assert stored is not None
    assert stored.ix0 <= min(-45, grid.ix0)
    assert stored.ix0 + stored.values.shape[0] > max(-45, grid.ix0 + grid.values.shape[0] - 1)
    assert stored.values[-45 - stored.ix0, -110 - stored.iy0].tobytes() == row.tobytes()
    assert not list(tmp_path.glob("*.partial*"))


def test_reference_rows_do_not_retain_old_grid_rectangles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    first = gs.gs_grid_node(-46, -111)
    gs.gs_grid_node(-45, -110)
    assert first.flags.owndata, "a source row must not pin an obsolete grid allocation"
    assert gs.gs_grid_node(-46, -111) is first, "sources share the process-wide node memo"


def test_sampler_saves_a_whole_bracket_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyradmc.data.goudsmit_saunderson as gs
    from pyradmc.data.analytic import AnalyticCrossSections
    from pyradmc.rng.host import HostRNG

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    saves = []
    original = gs._save_grid_cache

    def record(path: Path, grid: gs.GSGridTables) -> None:
        saves.append(grid.values.shape)
        original(path, grid)

    monkeypatch.setattr(gs, "_save_grid_cache", record)
    xs = AnalyticCrossSections()
    rng = HostRNG().init_state(42, 0)
    xs.sample_gs_cos_theta(1e-12, 1.0, 0, rng)
    xs.sample_gs_cos_theta(1e-12, 1.0, 0, rng)
    assert saves == [(2, 2, 512)]


def test_lazy_fill_uses_column_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    monkeypatch.setattr(gs, "_PARALLEL_BUILD_THRESHOLD", 4)
    monkeypatch.setattr(gs.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(gs, "_BUILD_WORKER_CAP", 2)
    workers = []

    def pool(*, max_workers: int) -> ThreadPoolExecutor:
        workers.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(gs, "ThreadPoolExecutor", pool)
    gs.gs_grid_node(-48, -111)  # small fills remain serial
    gs.gs_grid_node(-46, -109)  # eight new cells cross the threshold
    assert workers == [2]


def test_mu_resolution_is_part_of_both_cache_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect
    import math

    import pyradmc.data.goudsmit_saunderson as gs

    assert inspect.signature(gs.gs_scaled_deflection_table).parameters["n_mu"].default == (
        gs._GS_TABLE_MU_BINS
    )
    before = gs._grid_cache_file(tmp_path)
    monkeypatch.setattr(gs, "_GS_TABLE_MU_BINS", 2048)
    assert gs._grid_cache_file(tmp_path) != before
    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    row = gs.gs_grid_node(-46, -111)
    assert (
        gs._node_cache[(math.exp(-46 / 4), math.exp(-111 / 4), 512, 2048, gs._RAW_SAMPLES)] is row
    )
    assert not any(key[3] == 4096 for key in gs._node_cache)


def test_eager_save_failure_keeps_the_grid_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import pyradmc.data.goudsmit_saunderson as gs

    def readonly(*args: object, **kwargs: object) -> None:
        raise PermissionError("read-only cache directory")

    monkeypatch.setattr(gs, "_save_grid_cache", readonly)
    grid = gs.build_gs_grid(-12.0, -11.5, 2e-12, cache_dir=tmp_path)
    assert np.isfinite(grid.values).all()
    assert gs.build_gs_grid(-12.0, -11.5, 2e-12, cache_dir=tmp_path) is grid
    assert "using memory" in caplog.text


def test_eager_memory_cache_respects_mu_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyradmc.data.goudsmit_saunderson as gs

    first = gs.build_gs_grid(-12.0, -11.5, 2e-12, cache_dir=tmp_path)
    monkeypatch.setattr(gs, "_GS_TABLE_MU_BINS", 2048)
    second = gs.build_gs_grid(-12.0, -11.5, 2e-12, cache_dir=tmp_path)
    assert second is not first
    assert gs._grid_cache_file(tmp_path).is_file()


@pytest.mark.parametrize("failure", ["save", "replace"])
def test_failed_save_removes_partial_and_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import pyradmc.data.goudsmit_saunderson as gs

    grid = gs.GSGridTables(np.zeros((1, 1, 512)), -46, -111, 512, 4.0)
    path = gs._grid_cache_file(tmp_path)
    gs._save_grid_cache(path, grid)
    before = path.read_bytes()
    original = np.savez

    def interrupted(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        raise OSError("disk full")

    def denied(*args: object, **kwargs: object) -> None:
        raise PermissionError("replace denied")

    if failure == "save":
        monkeypatch.setattr(np, "savez", interrupted)
    else:
        monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(OSError):
        gs._save_grid_cache(path, grid)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.partial*"))


def test_legacy_default_mu_cache_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    import pyradmc.data.goudsmit_saunderson as gs

    # Historical key: inverse-CDF resolution was implicitly fixed at 4096.
    key = gs._grid_cache_key()[:-1]
    path = tmp_path / f"gs-grid-{hashlib.sha256(repr(key).encode()).hexdigest()[:16]}.npz"
    np.savez(path, values=np.ones((2, 2, 512)), ix0=-46, iy0=-111, key=repr(key))
    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the compatible legacy grid must be reused")

    monkeypatch.setattr(gs, "_gs_raw_cosines", refuse)
    rows = gs.gs_grid_nodes(-46, -111)
    assert all(np.array_equal(row, np.ones(512)) for row in rows)
    monkeypatch.setattr(gs, "_GS_TABLE_MU_BINS", 2048)
    assert gs._load_grid_cache(gs._grid_cache_file(tmp_path)) is None


def test_parallel_extension_rebuilds_identical_nontrivial_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_RAW_SAMPLES", 512)
    initial = gs._fill_rectangle(-47, -47, -25, -25, None, 512, 4.0, max_workers=1)
    stored = gs.GSGridTables(initial, -47, -25, 512, 4.0)
    gs._node_cache.clear()
    serial = gs._fill_rectangle(-48, -46, -26, -24, stored, 512, 4.0, max_workers=1)
    gs._node_cache.clear()  # actually reconstruct, rather than comparing memo hits
    parallel = gs._fill_rectangle(-48, -46, -26, -24, stored, 512, 4.0, max_workers=3)
    assert np.any(serial != 0)
    assert serial[0, 0].tobytes() != serial[-1, -1].tobytes()
    assert serial.tobytes() == parallel.tobytes()
    assert parallel[1, 1].tobytes() == initial[0, 0].tobytes()
    assert not parallel.flags.writeable


def test_threaded_bracket_misses_share_immutable_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import pyradmc.data.goudsmit_saunderson as gs

    monkeypatch.setattr(gs, "_GS_GRID_CACHE_DIR", tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        brackets = list(pool.map(lambda _: gs.gs_grid_nodes(-46, -111), range(8)))
    for bracket in brackets:
        assert all(row is first for row, first in zip(bracket, brackets[0], strict=True))
        assert all(row.flags.owndata and not row.flags.writeable for row in bracket)
    assert not list(tmp_path.glob("*.partial*"))
