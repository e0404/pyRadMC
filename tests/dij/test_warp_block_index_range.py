"""A batch's block of a beamlet group must stay inside the kernels' int32 index.

The generators index one batch's block as ``t = t_offset + tid``, an ``int32``, up to
``group * per_batch``. Past ``2**31 - 1`` it wraps and late chunks are silently
attributed to the wrong beamlets: total dose conserved, its distribution wrong
(found on a 143-segment plan at 32M histories per segment). Group size is bit-inert,
so the engine caps the group to fit instead; the cap is exercised here by lowering
the limit, since reaching ``2**31`` histories is not a unit-test budget.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from pyradmc.backends.warp import engine as warp_engine  # noqa: E402
from pyradmc.backends.warp.engine import WarpEngine, _index_safe_group_size  # noqa: E402
from pyradmc.data.analytic import AnalyticCrossSections  # noqa: E402
from pyradmc.geometry.grid import VoxelGrid  # noqa: E402
from pyradmc.geometry.source import BeamletGridSource  # noqa: E402
from pyradmc.progress import ProgressEvent  # noqa: E402
from tests.conftest import SEED  # noqa: E402

N_PER = 240
N_BATCHES = 2  # so a two-beamlet cap (2 * PER_BATCH) still admits N_PER
PER_BATCH = N_PER // N_BATCHES
N_BEAMLETS = 6


def _engine() -> WarpEngine:
    grid = VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device="cpu", chunk_size=2048)


def _lattice() -> BeamletGridSource:
    return BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(2.0, 6.0), y_range=(2.0, 6.0), n_x=3, n_y=2
    )


def _run_dij(group: int | None, events: list[ProgressEvent] | None = None):
    return _engine().run_dij(
        _lattice(),
        n_histories_per_beamlet=N_PER,
        n_batches=N_BATCHES,
        seed=SEED,
        beamlet_group_size=group,
        progress=None if events is None else events.append,
    )


class TestIndexSafeGroupSize:
    def test_the_real_limit_is_int32(self) -> None:
        assert warp_engine._MAX_BLOCK_HISTORIES == 2**31 - 1

    def test_the_case_that_wrapped(self) -> None:
        """143 segments at 16M histories per batch is 2.29e9; at most 134 fit."""
        assert _index_safe_group_size(143, 16_000_000) == 134
        assert 134 * 16_000_000 <= 2**31 - 1 < 135 * 16_000_000

    def test_a_group_that_fits_is_unchanged(self) -> None:
        assert _index_safe_group_size(90, 16_000_000) == 90

    def test_one_beamlet_always_fits_a_legal_batch(self) -> None:
        assert _index_safe_group_size(1024, 2**31 - 1) == 1


class TestRunDijCapsTheGroup:
    @pytest.fixture
    def limit_two_beamlets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(warp_engine, "_MAX_BLOCK_HISTORIES", 2 * PER_BATCH)

    @pytest.mark.parametrize("group", [N_BEAMLETS, None])
    def test_an_oversized_group_is_split(self, limit_two_beamlets, group) -> None:
        events: list[ProgressEvent] = []
        _run_dij(group, events)
        assert len(events) == N_BEAMLETS // 2  # one tick per completed group

    def test_the_split_is_bit_inert(self, monkeypatch: pytest.MonkeyPatch) -> None:
        whole = _run_dij(N_BEAMLETS)
        monkeypatch.setattr(warp_engine, "_MAX_BLOCK_HISTORIES", 2 * PER_BATCH)
        split = _run_dij(N_BEAMLETS)
        np.testing.assert_array_equal(split.indptr, whole.indptr)
        np.testing.assert_array_equal(split.indices, whole.indices)
        np.testing.assert_array_equal(split.dose, whole.dose)

    def test_a_beamlet_past_the_index_range_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The within-beamlet index ``r < n_per`` is int32 too: no group size helps."""
        monkeypatch.setattr(warp_engine, "_MAX_BLOCK_HISTORIES", N_PER - 1)
        with pytest.raises(ValueError, match="histories per beamlet"):
            _run_dij(1)
