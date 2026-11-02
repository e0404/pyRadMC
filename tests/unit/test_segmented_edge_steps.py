"""The first-order dose step of a segmented fan: which edge a history takes, and its weight."""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.geometry.segment_edge_steps import SegmentedEdgeStepSource
from pyradmc.geometry.segment_edges import SegmentedEdgeSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED
from tests.unit.test_segmented_edge_source import _segments


def _steps(edges: SegmentedEdgeSource) -> np.ndarray:
    """Every other edge moved, by a different amount each, both ways."""
    steps = np.zeros(edges.n_beamlets)
    moved = np.arange(len(steps[::2]))
    steps[::2] = 0.01 * (1.0 + moved) * (-1.0) ** moved
    return steps


def test_each_segment_has_an_opening_and_a_closing_column() -> None:
    edges = SegmentedEdgeSource(_segments())
    steps = _steps(edges)
    source = SegmentedEdgeStepSource(edges, steps)
    assert source.n_beamlets == 2 * edges.segments.n_segments
    assert source.max_energy == edges.max_energy
    columns = edges.device_columns()
    first, edge = source.selection.first, source.selection.edge
    for column in range(source.n_beamlets):
        segment, closing = divmod(column, 2)
        moved = np.flatnonzero((columns.segment == segment) & (steps != 0.0))
        expected = moved[(columns.sign[moved] * steps[moved] < 0.0) == bool(closing)]
        if expected.size:
            assert set(edge[first[column] : first[column + 1]]) == set(expected)


def test_an_edge_is_chosen_by_its_travel_and_its_ray_carries_the_columns_total() -> None:
    edges = SegmentedEdgeSource(_segments())
    steps = _steps(edges)
    source = SegmentedEdgeStepSource(edges, steps)
    selection, sign = source.selection, edges.device_columns().sign
    for column in range(source.n_beamlets):
        lo, hi = selection.first[column], selection.first[column + 1]
        picks = selection.edge[lo:hi]
        travel = np.abs(steps[picks])
        if travel.sum() == 0.0:
            assert selection.factor[lo:hi] == pytest.approx(0.0)
            continue
        assert selection.cdf[hi - 1] == 1.0
        assert np.diff(selection.cdf[lo:hi], prepend=0.0) == pytest.approx(travel / travel.sum())
        assert selection.factor[lo:hi] == pytest.approx(sign[picks] * travel.sum())


def test_a_column_nothing_moved_in_emits_zero_weight() -> None:
    edges = SegmentedEdgeSource(_segments())
    source = SegmentedEdgeStepSource(edges, np.zeros(edges.n_beamlets))
    rng = HostRNG()
    for column in range(source.n_beamlets):
        assert source.selection.first[column + 1] == source.selection.first[column] + 1
        assert source.emit(column, rng.init_state(SEED, column)).weight == 0.0


def test_emitted_weights_are_non_negative_and_average_to_the_edges_derivatives() -> None:
    """Non-negative: the Dij keeps positive voxels only, so each column must be one-signed."""
    edges = SegmentedEdgeSource(_segments())
    steps = _steps(edges)
    source = SegmentedEdgeStepSource(edges, steps)
    columns = edges.device_columns()
    rng = HostRNG()
    n = 4_000
    for column in range(source.n_beamlets):
        weights = np.array([source.emit(column, rng.init_state(SEED, h)).weight for h in range(n)])
        assert np.all(weights >= 0.0)
        segment, closing = divmod(column, 2)
        moved = np.flatnonzero(
            (columns.segment == segment)
            & (steps != 0.0)
            & ((columns.sign * steps < 0.0) == bool(closing))
        )
        expected, variance = 0.0, weights.var() / n
        for e in moved:
            batch = edges.sample_beamlet_batch(SEED + 1, 0, 40_000, int(e))["weight"]
            expected += abs(steps[e]) * abs(float(np.mean(batch, dtype=np.float64)))
            variance += (steps[e] * float(np.std(batch, dtype=np.float64))) ** 2 / 40_000
        assert abs(weights.mean() - expected) <= 4.0 * np.sqrt(variance) + 1e-12


@pytest.mark.parametrize("n", [1, 7])
def test_a_step_per_edge_is_required(n: int) -> None:
    edges = SegmentedEdgeSource(_segments())
    with pytest.raises(ValueError, match="one step per edge"):
        SegmentedEdgeStepSource(edges, np.zeros(n))


def test_a_step_must_be_finite() -> None:
    edges = SegmentedEdgeSource(_segments())
    steps = np.zeros(edges.n_beamlets)
    steps[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SegmentedEdgeStepSource(edges, steps)
