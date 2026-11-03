"""The first-order dose change of a segmented fan under a small move of its edges.

An aperture optimizer moves every edge a little per iteration and a full forward run
re-simulates the whole aperture to see it. To first order, with the derivative taken at
the step's midpoint so that the error is of third order in the step,

    D_s(x + d) - D_s(x) = sum_e d_e dD_s/dx_e,

and each ``dD_s/dx_e`` has an unbiased estimator already, the edge's
:class:`~pyradmc.geometry.aperture_gradient.FanEdgeDerivativeSource`. This source draws
that sum: a history of a segment picks one edge with probability ``|d_e| / sum |d_e|``
and carries the edge's ray with its weight times ``sum |d_e|``. The histories needed
scale with the area the edges swept, not with the aperture's.

**Two beamlets per segment**: ``2 s`` lists the edges that open (``sign d_e > 0``, the
transmission rises) and ``2 s + 1`` those that close, each with the sign that makes its
weights non-negative, so the Dij's compaction, which keeps positive voxels, loses nothing.
The dose change of segment ``s`` is column ``2 s`` minus column ``2 s + 1``. A column
nothing moved in lists one edge at factor zero.

Physics, estimator and stated approximations are the edge source's. The Warp engine
generates this exact type in-kernel from :attr:`SegmentedEdgeStepSource.selection`,
picking the edge as :meth:`SegmentedEdgeStepSource.emit` does
(:func:`~pyradmc.geometry.fan.select_by_cdf_at`) from a position stratified over the
batch's histories of the column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyradmc.geometry.fan import select_by_cdf_at
from pyradmc.geometry.segment_edges import SegmentedEdgeSource
from pyradmc.geometry.source import BeamletSource, Primary
from pyradmc.rng import RNGState, uniform

__all__ = ["EdgeStepSelection", "SegmentedEdgeStepSource"]


@dataclass(frozen=True)
class EdgeStepSelection:
    """Per column, the edges a history can pick: entries ``first[c]`` to ``first[c + 1]``."""

    first: NDArray[np.int32]
    edge: NDArray[np.int32]
    cdf: NDArray[np.float64]
    factor: NDArray[np.float64]


def _selection(
    segment: NDArray[np.int32],
    sign: NDArray[np.float64],
    steps: NDArray[np.float64],
    n_columns: int,
) -> EdgeStepSelection:
    """Group the moved edges by column, each with its cumulative share and weight factor."""
    column = 2 * segment + (sign * steps < 0.0)
    moved = np.flatnonzero(steps != 0.0)
    moved = moved[np.argsort(column[moved], kind="stable")]
    counts = np.bincount(column[moved], minlength=n_columns)
    first = np.zeros(n_columns + 1, dtype=np.int32)
    first[1:] = np.cumsum(np.maximum(counts, 1))
    edge = np.zeros(first[-1], dtype=np.int32)
    cdf = np.ones(first[-1], dtype=np.float64)
    factor = np.zeros(first[-1], dtype=np.float64)
    start = 0
    for c, count in enumerate(counts):
        lo = first[c]
        if count == 0:  # one entry at factor zero: no branch in a kernel
            edge[lo] = np.flatnonzero(segment == c // 2)[0]
            continue
        picks = moved[start : start + count]
        start += count
        travel = np.abs(steps[picks])
        total = travel.sum()
        edge[lo : lo + count] = picks
        cdf[lo : lo + count] = np.cumsum(travel) / total
        cdf[lo + count - 1] = 1.0
        factor[lo : lo + count] = sign[picks] * total
    return EdgeStepSelection(first, edge, cdf, factor)


class SegmentedEdgeStepSource(BeamletSource):
    """The edges' moves of a segmented fan as an opening and a closing beamlet per segment.

    Parameters
    ----------
    edges
        The edges, on the settings where the derivative is taken (the step's midpoint).
    steps
        Every edge's move (mid-plane cm), in ``edges``' beamlet order.
    """

    def __init__(self, edges: SegmentedEdgeSource, steps: NDArray[np.float64]) -> None:
        """Group the moved edges into their columns."""
        steps = np.asarray(steps, dtype=np.float64)
        if steps.shape != (edges.n_beamlets,):
            raise ValueError(f"one step per edge: {edges.n_beamlets}, got {steps.shape}")
        if not np.all(np.isfinite(steps)):
            raise ValueError("steps must be finite")
        self._edges = edges
        self._n_columns = 2 * edges.segments.n_segments
        columns = edges.device_columns()
        self._selection = _selection(columns.segment, columns.sign, steps, self._n_columns)

    @property
    def edges(self) -> SegmentedEdgeSource:
        """The edges the steps move."""
        return self._edges

    @property
    def selection(self) -> EdgeStepSelection:
        """What a history of each column picks from."""
        return self._selection

    @property
    def max_energy(self) -> float:
        """The fan spectrum's top edge."""
        return self._edges.max_energy

    @property
    def n_beamlets(self) -> int:
        """Two per segment: opening, then closing."""
        return self._n_columns

    def emit(self, beamlet: int, rng_state: RNGState) -> Primary:
        """Emit one ray: one uniform for the edge, then the edge source's four."""
        if not 0 <= beamlet < self._n_columns:
            raise IndexError(f"beamlet {beamlet} outside {self._n_columns} columns")
        selection = self._selection
        lo, hi = int(selection.first[beamlet]), int(selection.first[beamlet + 1])
        # The in-kernel generator's pick: the first entry whose share exceeds the draw.
        pick = lo + select_by_cdf_at(selection.cdf, lo, hi - lo, uniform(rng_state))
        ray = self._edges.edge(int(selection.edge[pick])).emit(rng_state)
        return Primary(ray.energy, ray.x, ray.y, ray.z, ray.ux, ray.uy, ray.uz,
                       weight=ray.weight * float(selection.factor[pick]))  # fmt: skip
