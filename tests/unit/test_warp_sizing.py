"""Auto-sizing policy for the Warp backend's launch and memory knobs.

Every knob these helpers decide is **bit-inert** — chunk size, beamlet group size
and the truncation layout change only how work is cut up, never a scored value
(pinned in the Dij and device-reduction suites). So these tests are about the
*policy* holding together: that the footprint arithmetic matches what the engine
actually allocates, that the budget cannot be over-committed, and that the
degenerate ends of each range stay sane.
"""

from __future__ import annotations

import pytest

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from warp.types import type_size_in_bytes  # noqa: E402

from pyradmc.backends.warp import kernels  # noqa: E402
from pyradmc.backends.warp.engine import (  # noqa: E402
    _CHUNK_SIZE_CAP,
    _CHUNK_SIZE_FALLBACK,
    _GROUP_SIZE_CAP,
    _QUEUE_BYTES_PER_SLOT,
    _auto_chunk_size,
    _auto_group_size,
    _queue_bytes,
    _upload_queue,
)


class TestQueueFootprint:
    def test_slot_size_matches_an_actual_allocation(self) -> None:
        """The sizing constant must track ``Queue``; adding a column must trip this.

        Four queues per lane plus the lane's uint32 RNG-slot array, which is what
        ``_run_dij_shard`` and ``_transport_run`` allocate per chunk history.
        """
        n = 1024
        q = _upload_queue(n, "cpu")
        per_queue = sum(
            v.size * type_size_in_bytes(v.dtype)
            for name in dir(q)
            if not name.startswith("_")
            and isinstance(v := getattr(q, name, None), wp.array)
            and v.size == n  # the scalar ``count`` is not per-slot
        )
        assert per_queue % n == 0
        expected = 4 * (per_queue // n) + type_size_in_bytes(wp.uint32)
        assert expected == _QUEUE_BYTES_PER_SLOT

    def test_queue_bytes_scales_with_chunk_factor_and_lanes(self) -> None:
        base = _queue_bytes(1000, 16, 1)
        assert base == 1000 * 16 * _QUEUE_BYTES_PER_SLOT
        assert _queue_bytes(2000, 16, 1) == 2 * base
        assert _queue_bytes(1000, 32, 1) == 2 * base
        assert _queue_bytes(1000, 16, 3) == 3 * base


class TestAutoGroupSize:
    """The dense maps are budgeted *after* the queues are charged."""

    def test_charging_the_queues_never_grows_the_group(self) -> None:
        free, n_voxels, n_beamlets = 8 * 2**30, 1_000_000, 500
        unaware = _auto_group_size(free, n_voxels, n_beamlets, 1, 0)
        aware = _auto_group_size(free, n_voxels, n_beamlets, 1, _queue_bytes(262_144, 16, 1))
        assert aware <= unaware
        assert aware >= 1

    def test_the_dense_maps_fit_what_is_left_after_the_queues(self) -> None:
        """The point of the change: dense + queues must fit reported-free memory."""
        free, n_voxels, n_beamlets, lanes = 6 * 2**30, 1_000_000, 500, 1
        queue = _queue_bytes(262_144, 16, lanes)
        group = _auto_group_size(free, n_voxels, n_beamlets, lanes, queue)
        dense = group * (8 * lanes + 16) * n_voxels
        assert dense + queue <= free

    def test_a_queue_footprint_larger_than_memory_still_yields_a_usable_group(self) -> None:
        group = _auto_group_size(2**30, 1_000_000, 500, 1, 8 * 2**30)
        assert group == 1  # clamped, not zero or negative

    def test_unknown_free_memory_falls_back(self) -> None:
        assert _auto_group_size(0, 1_000_000, 500, 1, 0) == min(128, 500)
        assert _auto_group_size(0, 1_000_000, 7, 1, 0) == 7

    def test_result_is_capped(self) -> None:
        huge = _auto_group_size(1 << 50, 1, 10_000, 1, 0)
        assert huge == _GROUP_SIZE_CAP


class TestAutoChunkSize:
    def test_cpu_keeps_the_historic_default(self) -> None:
        assert _auto_chunk_size("cpu", 16) == _CHUNK_SIZE_FALLBACK

    @pytest.mark.gpu
    def test_cuda_sizes_within_the_cap_and_affords_its_queues(self) -> None:
        if not wp.is_cuda_available():
            pytest.skip("no CUDA device")
        chunk = _auto_chunk_size("cuda:0", 16)
        assert 1 <= chunk <= _CHUNK_SIZE_CAP
        free = int(wp.get_device("cuda:0").free_memory)
        assert _queue_bytes(chunk, 16, 1) <= free, "auto chunk must fit reported-free memory"

    @pytest.mark.gpu
    def test_a_larger_queue_factor_never_enlarges_the_chunk(self) -> None:
        if not wp.is_cuda_available():
            pytest.skip("no CUDA device")
        assert _auto_chunk_size("cuda:0", 64) <= _auto_chunk_size("cuda:0", 16)


class TestTruncationLayout:
    """The launch width adapts to the group and the device; the tiling never changes."""

    @pytest.mark.parametrize("n_voxels", [1, 2, 7, 63, 64, 255, 1001, 4096, 1_000_000])
    @pytest.mark.parametrize("n_columns", [1, 22, 154, 1024])
    def test_tiling_covers_every_voxel_exactly_once(self, n_voxels: int, n_columns: int) -> None:
        n_chunks, chunk = kernels.truncation_chunk_layout(n_voxels, n_columns, 36)
        assert n_chunks >= 1 and chunk >= 1
        assert n_chunks * chunk >= n_voxels, "tail uncovered"
        assert (n_chunks - 1) * chunk < n_voxels, "wholly empty trailing chunk"

    def test_a_wider_group_needs_fewer_chunks_per_column(self) -> None:
        """The launch is n_columns * n_chunks, so the two trade off."""
        narrow, _ = kernels.truncation_chunk_layout(1_000_000, 32, 36)
        wide, _ = kernels.truncation_chunk_layout(1_000_000, 1024, 36)
        assert wide < narrow

    def test_a_wider_device_asks_for_more_threads(self) -> None:
        small, _ = kernels.truncation_chunk_layout(1_000_000, 154, 36)
        big, _ = kernels.truncation_chunk_layout(1_000_000, 154, 144)
        assert big > small

    def test_per_thread_work_has_a_floor(self) -> None:
        """A small grid must not degenerate into near-empty threads."""
        _, chunk = kernels.truncation_chunk_layout(256, 1024, 144)
        assert chunk >= 64

    def test_defaults_reproduce_the_single_column_reference(self) -> None:
        assert kernels.truncation_chunk_layout(1_000_000) == kernels.truncation_chunk_layout(
            1_000_000, 1, 0
        )
