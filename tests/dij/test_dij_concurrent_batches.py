"""Concurrent batch lanes within one device (``run_dij(concurrent_batches=...)``).

Batch streaming made the batch axis sequential in time; ``concurrent_batches=L``
transports up to ``L`` batches of the current group concurrently, each lane on its
own CUDA stream with its own queues and quanta map. The one ordering that carries
numerical meaning — the float64 left fold of the batch sums — is serialized in batch
order regardless of which lane finishes first, so the contract these tests pin is:

**any ``concurrent_batches`` produces the bitwise-identical Dij on the same device.**

That makes the knob a pure scheduling dial, like ``chunk_size`` and
``beamlet_group_size`` (AGENTS.md 2.3): safe to tune per card without revalidating
physics. The lanes' contributions to the shared energy counters are int64 atomics,
order-independent by integer associativity, so the books are bit-equal too.

A cpu device has no streams; ``concurrent_batches`` is documented to fall back to
the sequential path there, pinned below.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from pyradmc.backends.warp.engine import WarpEngine  # noqa: E402
from pyradmc.data.analytic import AnalyticCrossSections  # noqa: E402
from pyradmc.geometry.grid import VoxelGrid  # noqa: E402
from pyradmc.geometry.source import BeamletGridSource  # noqa: E402
from tests.conftest import SEED  # noqa: E402
from tests.dij.test_custom_beamlet_source_warp import StripBeamletSource  # noqa: E402

N_PER = 300
N_BATCHES = 6
GROUP = 2


def _engine(device: str) -> WarpEngine:
    grid = VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device=device, chunk_size=2048)


def _lattice() -> BeamletGridSource:
    return BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(2.0, 6.0), y_range=(2.0, 6.0), n_x=3, n_y=1
    )


def _run(device: str, source, lanes: int, **kw):
    return _engine(device).run_dij(
        source,
        n_histories_per_beamlet=N_PER,
        n_batches=N_BATCHES,
        seed=SEED,
        beamlet_group_size=GROUP,
        concurrent_batches=lanes,
        **kw,
    )


def _assert_dij_bit_identical(got, want) -> None:
    np.testing.assert_array_equal(got.indptr, want.indptr)
    np.testing.assert_array_equal(got.indices, want.indices)
    np.testing.assert_array_equal(got.dose, want.dose)
    np.testing.assert_array_equal(got.sigma, want.sigma)
    assert got.energy_deposited == want.energy_deposited
    assert got.energy_escaped == want.energy_escaped
    assert got.energy_unscored == want.energy_unscored
    assert got.energy_emitted == want.energy_emitted


@pytest.mark.gpu
@pytest.mark.parametrize("lanes", [2, 3])
def test_lattice_lanes_bit_identical_to_sequential(lanes: int) -> None:
    """The core contract, on the in-kernel lattice route.

    Lanes overlap transport but the fold is forced into batch order, so the Dij —
    pattern, values, sigmas, and books — must not move by a single bit.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    _assert_dij_bit_identical(_run("cuda:0", _lattice(), lanes), _run("cuda:0", _lattice(), 1))


@pytest.mark.gpu
def test_presampled_lanes_bit_identical_to_sequential() -> None:
    """The same contract on the host pre-sampling route.

    This route uploads primaries from lane threads, so it additionally pins that
    per-chunk host-to-device transfers are correctly ordered against lane-stream
    launches — the failure mode would be transported garbage, not a small error.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    source = StripBeamletSource(3)
    _assert_dij_bit_identical(
        _run("cuda:0", source, 2, transport_electrons=False),
        _run("cuda:0", source, 1, transport_electrons=False),
    )


def test_cpu_falls_back_to_the_sequential_path() -> None:
    """A cpu device has no streams: lanes collapse to 1, bit-identically."""
    _assert_dij_bit_identical(_run("cpu", _lattice(), 4), _run("cpu", _lattice(), 1))


@pytest.mark.gpu
def test_lanes_combine_with_device_sharding() -> None:
    """Both options at once: every sharded column still bit-matches a whole run.

    Lanes are within-device and sharding is across devices, so their composition
    must preserve the per-column contract of ``test_dij_multi_device`` — each
    column bit-identical to the whole-run column of whichever device computed it.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    source = _lattice()
    sharded = _run("cuda:0", source, 2, devices=["cuda:0", "cpu"])
    whole = [_run(d, source, 1) for d in ("cuda:0", "cpu")]
    for j in range(source.n_beamlets):
        lo, hi = int(sharded.indptr[j]), int(sharded.indptr[j + 1])
        col = (sharded.indices[lo:hi], sharded.dose[lo:hi], sharded.sigma[lo:hi])
        matched = False
        for w in whole:
            wlo, whi = int(w.indptr[j]), int(w.indptr[j + 1])
            if (
                np.array_equal(col[0], w.indices[wlo:whi])
                and np.array_equal(col[1], w.dose[wlo:whi])
                and np.array_equal(col[2], w.sigma[wlo:whi])
            ):
                matched = True
                break
        assert matched, f"column {j} bit-matches neither device's whole run"


def test_nonpositive_lane_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="concurrent_batches"):
        _run("cpu", _lattice(), 0)
