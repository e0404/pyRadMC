"""The first-order dose step on the Warp engine: in-kernel against the host and the edges."""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.geometry.segment_edge_steps import SegmentedEdgeStepSource
from pyradmc.geometry.segment_edges import SegmentedEdgeSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.dij.test_segmented_edge_source_warp import (
    STAT_KWARGS,
    XS_DEMO,
    _demo_grid,
    _demo_segments,
    _demo_weights,
)

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyradmc.backends.warp import aperture  # noqa: E402  (needs warp)

pytestmark = pytest.mark.warp

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


def _engine(device: str):
    from pyradmc.backends.warp.engine import WarpEngine

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return WarpEngine(grid=_demo_grid(), cross_sections=XS_DEMO, device=device)


def _source() -> tuple[SegmentedEdgeSource, np.ndarray, SegmentedEdgeStepSource]:
    edges = SegmentedEdgeSource(_demo_segments())
    steps = (
        0.02
        * (1.0 + np.arange(edges.n_beamlets) % 3)
        * (-1.0) ** (np.arange(edges.n_beamlets) // 2)
    )
    return edges, steps, SegmentedEdgeStepSource(edges, steps)


@pytest.mark.parametrize("device", DEVICES)
def test_columns_sum_the_edges_contractions_times_their_travel(device: str) -> None:
    """The kernel's edge choice and factor against the edges' own in-kernel contractions.

    Four compared values: 32 batches, whose sigmas are close to exact, against the null's
    normal approximation. A probe at 4M histories per column finds no bias (|z| < 1.4).
    """
    edges, steps, source = _source()
    engine = _engine(device)
    kwargs = {**STAT_KWARGS, "n_batches": 32}
    per_edge = engine.run_beamlet_contraction(
        edges, _demo_weights(), n_histories_per_beamlet=64_000, seed=SEED, **kwargs
    )
    summed = engine.run_beamlet_contraction(
        source, _demo_weights(), n_histories_per_beamlet=192_000, seed=SEED + 1, **kwargs
    )
    columns = edges.device_columns()
    expected = np.zeros(source.n_beamlets)
    variance = np.zeros(source.n_beamlets)
    for e, step in enumerate(steps):
        k = 2 * int(columns.segment[e]) + int(columns.sign[e] * step < 0.0)
        expected[k] += abs(step) * columns.sign[e] * per_edge.values[e]
        variance[k] += (step * per_edge.sigma[e]) ** 2
    assert np.all(np.abs(summed.values) > 3.0 * summed.sigma)  # every column resolved
    sigma = np.sqrt(variance)
    assert_chi2_consistent_batched(expected, sigma, 32, summed.values, summed.sigma, 32)
    with pytest.raises(AssertionError):  # the oracle sees a 5 % error in a column's factor
        assert_chi2_consistent_batched(1.05 * expected, sigma, 32, summed.values, summed.sigma, 32)


def test_reference_engine_agrees_statistically() -> None:
    """The host route (the per-edge oracle through ``emit``) against the in-kernel one."""
    _, _, source = _source()
    device = _engine("cpu").run_beamlet_contraction(
        source, _demo_weights(), n_histories_per_beamlet=48_000, seed=SEED, **STAT_KWARGS
    )
    reference = ReferenceEngine(grid=_demo_grid(), cross_sections=XS_DEMO, rng=HostRNG())
    ref = reference.run_beamlet_contraction(
        source, _demo_weights(), n_histories_per_beamlet=3_000, seed=SEED, **STAT_KWARGS
    )
    assert_chi2_consistent_batched(ref.values, ref.sigma, 8, device.values, device.sigma, 8)


def test_the_in_kernel_route_is_taken() -> None:
    _, _, source = _source()
    assert aperture.is_device_edge_set(source)


@pytest.mark.parametrize("device", DEVICES)
def test_the_dij_keeps_every_deposit_of_a_column(device: str) -> None:
    """Non-negative columns: nothing is lost to the Dij's positive-voxel compaction."""
    _, _, source = _source()
    engine = _engine(device)
    kwargs = dict(
        n_histories_per_beamlet=8_000,
        n_batches=8,
        seed=SEED,
        correlated=False,
        transport_electrons=False,
    )
    dij = engine.run_dij(source, truncation=0.0, **kwargs)
    summed = np.array(
        [dij.dose[dij.indptr[k] : dij.indptr[k + 1]].sum() for k in range(dij.n_beamlets)]
    )
    contraction = engine.run_beamlet_contraction(source, np.ones(_demo_grid().shape), **kwargs)
    assert summed == pytest.approx(contraction.values, rel=1e-9)
