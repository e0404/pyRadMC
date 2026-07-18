"""The engine's host table build is cached per (ecut, pcut, e_max).

``CrossSectionSource.build_tables`` flattens the host data by looping the Python
query API over every grid node — seconds of fixed overhead per call for a tabulated
source. The flattened :class:`~pyRadMC.data.tables.CrossSectionTables` is frozen
after construction, so one build can serve every run, batch lane, and device shard
with the same cutoffs and energy ceiling; only the (cheap) device upload stays per
call. These tests pin that caching contract: same key builds once, a different key
rebuilds, and the cache leaves results bit-identical (the arrays are the same host
objects, so anything else would be a harness bug, not a physics one).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource
from tests.conftest import SEED

pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyRadMC.backends.warp.engine import WarpEngine

pytestmark = pytest.mark.warp


class _CountingCrossSections:
    """Delegating wrapper that counts ``build_tables`` invocations."""

    def __init__(self, inner: AnalyticCrossSections) -> None:
        self._inner = inner
        self.builds = 0

    def build_tables(self, **kwargs):
        self.builds += 1
        return self._inner.build_tables(**kwargs)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _engine() -> tuple[WarpEngine, _CountingCrossSections]:
    grid = VoxelGrid.uniform_water(shape=(8, 8, 8), spacing=(0.5, 0.5, 0.5))
    xs = _CountingCrossSections(
        AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    )
    return WarpEngine(grid=grid, cross_sections=xs, device="cpu"), xs


def test_same_key_builds_host_tables_once() -> None:
    engine, xs = _engine()
    first = engine._upload_tables(6.0, pcut=0.05, ecut=0.2, device="cpu")
    second = engine._upload_tables(6.0, pcut=0.05, ecut=0.2, device="cpu")
    assert xs.builds == 1
    # Device structs are fresh per call (upload is not cached), built from one host table.
    assert first is not second


def test_key_change_rebuilds() -> None:
    engine, xs = _engine()
    engine._upload_tables(6.0, pcut=0.05, ecut=0.2, device="cpu")
    engine._upload_tables(7.0, pcut=0.05, ecut=0.2, device="cpu")  # new e_max
    assert xs.builds == 2
    engine._upload_tables(7.0, pcut=0.05, ecut=0.2, device="cpu")  # cached again
    assert xs.builds == 2


def test_repeated_runs_share_one_build_and_stay_bit_identical() -> None:
    engine, xs = _engine()
    source = ParallelBeamSource(energy=6.0, z=-1.0, x_range=(1.0, 3.0), y_range=(1.0, 3.0))
    first = engine.run(source, n_histories=256, n_batches=2, seed=SEED)
    second = engine.run(source, n_histories=256, n_batches=2, seed=SEED)
    assert xs.builds == 1
    np.testing.assert_array_equal(first.dose, second.dose)
