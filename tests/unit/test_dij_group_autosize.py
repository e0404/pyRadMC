"""Auto-sizing of the Dij beamlet group from free device memory.

``beamlet_group_size`` is bit-inert (test-pinned in the Dij suite), so its value is
pure scheduling policy: larger groups mean coarser launches and fewer drain
round-trips, bounded by the dense per-group device maps —
``(8 * lanes + 16) * n_voxels`` bytes per beamlet (one int64 quanta map per lane
plus the two float64 running sums). The auto-sizer picks the largest group that
keeps those maps inside a fraction of the *reported free* device memory, clamped
to ``[1, min(n_beamlets, cap)]``, and falls back to the documented default when
free memory is unknown (a cpu device reports none). These tests pin that policy
arithmetic; result-invariance needs no new pin.
"""

from __future__ import annotations

import pytest

pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyradmc.backends.warp.engine import (
    _GROUP_SIZE_CAP,
    _GROUP_SIZE_FALLBACK,
    _auto_group_size,
)

pytestmark = pytest.mark.warp

N_VOXELS_64 = 64**3


def test_unknown_free_memory_falls_back_to_default() -> None:
    assert _auto_group_size(0, N_VOXELS_64, n_beamlets=10_000, lanes=1) == _GROUP_SIZE_FALLBACK


def test_fallback_is_clamped_to_the_beamlet_count() -> None:
    assert _auto_group_size(0, N_VOXELS_64, n_beamlets=8, lanes=1) == 8


def test_budget_arithmetic_single_lane() -> None:
    # 24 bytes per (beamlet, voxel); half of 1 GB budgeted -> 79 groups on a 64^3 map.
    free = 1_000_000_000
    expected = (free // 2) // (24 * N_VOXELS_64)
    assert _auto_group_size(free, N_VOXELS_64, n_beamlets=10_000, lanes=1) == expected


def test_extra_lanes_shrink_the_group() -> None:
    free = 1_000_000_000
    one = _auto_group_size(free, N_VOXELS_64, n_beamlets=10_000, lanes=1)
    two = _auto_group_size(free, N_VOXELS_64, n_beamlets=10_000, lanes=2)
    assert two == (free // 2) // (32 * N_VOXELS_64) < one


def test_huge_memory_is_capped() -> None:
    assert _auto_group_size(10**15, N_VOXELS_64, n_beamlets=10**6, lanes=1) == _GROUP_SIZE_CAP


def test_tiny_memory_floors_at_one_group() -> None:
    assert _auto_group_size(1, N_VOXELS_64, n_beamlets=10_000, lanes=1) == 1
