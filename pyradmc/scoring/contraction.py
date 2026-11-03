"""Beamlet contraction: ``g_j = sum_v a_v D_vj`` scored without assembling the Dij.

An optimizer consumes the Dij only through products with a voxel vector — the
dose ``D w`` and, for gradients, ``D^T a`` with ``a = dL/dD``. When the columns are
recomputed every iteration (aperture gradients: each column is one leaf edge's
``dD/dx``, which moves with the leaves), assembling, truncating and storing them is
the dominant cost and buys nothing. :class:`ContractionScorer` accumulates the
contracted scalar per beamlet and batch directly, so each component carries a
batch-spread sigma like every other tally (AGENTS.md 2.4), and memory is one row of
voxel weights plus three numbers per beamlet.

No truncation applies: the contraction sums every deposit, so the low-dose-tail
bias the Dij truncation threshold introduces (AGENTS.md 2.8) does not arise, and
the columns may be signed (a closing leaf edge has a negative derivative).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyradmc.backends.results import RunProvenance
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.scoring.grid import ScoringGrid

__all__ = ["ContractionResult", "ContractionScorer"]

_F64 = NDArray[np.float64]


@dataclass(frozen=True)
class ContractionResult:
    """``values[j] = sum_v a_v D_vj`` per beamlet, with its batch standard error.

    ``D_vj`` is beamlet ``j``'s dose per emitted history of that beamlet (MeV/g),
    exactly as in :class:`~pyradmc.scoring.dij.DijResult`, so ``values`` is what
    contracting the untruncated Dij columns with the same weights gives. The energy
    books are those of the equivalent ``run_dij``.
    """

    values: _F64
    sigma: _F64
    n_histories_per_beamlet: int
    n_batches: int
    correlated: bool
    energy_emitted: float
    energy_deposited: float
    energy_escaped: float
    energy_unscored: float
    provenance: RunProvenance | None = None

    @property
    def n_beamlets(self) -> int:
        """Number of contracted columns."""
        return int(self.values.size)


class ContractionScorer:
    """Accumulates ``sum_v a_v E_v / m_v`` per beamlet and batch.

    The drop-in for :class:`~pyradmc.scoring.dij.BatchedBeamletScorer` in an
    engine's beamlet loop: the same ``deposit_at`` / ``end_batch`` protocol, but
    each deposit is folded into its beamlet's scalar at once (``a_v`` times the
    voxel dose increment), so no ``(n_beamlets, n_voxels)`` block exists.
    """

    def __init__(
        self,
        grid: VoxelGrid | ScoringGrid,
        n_batches: int,
        n_beamlets: int,
        voxel_weights: NDArray[np.floating],
    ) -> None:
        if n_batches < 1:
            raise ValueError(f"need at least one batch, got {n_batches}")
        if n_beamlets < 1:
            raise ValueError(f"need at least one beamlet, got {n_beamlets}")
        scoring = grid if isinstance(grid, ScoringGrid) else ScoringGrid.for_grid(grid)
        weights = np.asarray(voxel_weights, dtype=np.float64)
        if weights.shape != scoring.shape:
            raise ValueError(
                f"voxel weight shape {weights.shape} != scoring grid shape {scoring.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("voxel weights must be finite")
        mass = scoring.voxel_mass.reshape(-1)
        # Weight per unit deposited energy: a_v / m_v, zero on uncovered voxels.
        self._per_energy = np.divide(
            weights.reshape(-1), mass, out=np.zeros(mass.size), where=mass > 0.0
        )
        self._scoring = scoring
        self._ny, self._nz = scoring.shape[1], scoring.shape[2]
        self._n_batches = n_batches
        self._closed = 0
        self._current = np.zeros(n_beamlets)
        self._sum = np.zeros(n_beamlets)
        self._sum_sq = np.zeros(n_beamlets)
        self._energy_deposited = 0.0
        self._energy_unscored = 0.0

    def deposit_at(
        self,
        beamlet: int,
        x: float,
        y: float,
        z: float,
        energy: float,
        scored: float | None = None,
    ) -> None:
        """Fold one deposit (MeV) at a position (cm) into its beamlet's scalar."""
        if self._scoring.contains(x, y, z):
            ix, iy, iz = self._scoring.voxel_index(x, y, z)
            flat = (ix * self._ny + iy) * self._nz + iz
            self._current[beamlet] += self._per_energy[flat] * (
                energy if scored is None else scored
            )
            self._energy_deposited += energy
        else:
            self._energy_unscored += energy

    def end_batch(self, n_histories: int) -> None:
        """Close the current batch of ``n_histories`` emitted histories per beamlet."""
        if self._closed >= self._n_batches:
            raise RuntimeError("all batches already closed")
        if n_histories < 1:
            raise ValueError(f"empty batch ({n_histories} histories per beamlet)")
        batch = self._current / n_histories
        self._sum += batch
        self._sum_sq += batch**2
        self._current[:] = 0.0
        self._closed += 1

    def finalize(self) -> tuple[_F64, _F64, float, float]:
        """Return ``(values, sigma, energy_deposited, energy_unscored)``."""
        if self._closed != self._n_batches:
            raise RuntimeError(f"finalize after {self._closed}/{self._n_batches} batch closures")
        n = self._n_batches
        mean = self._sum / n
        if n == 1:
            sigma = np.zeros_like(mean)
        else:
            # The clamped estimator of BatchedDoseScorer / BatchedBeamletScorer.
            variance = np.maximum(0.0, self._sum_sq / n - mean**2) * n / (n - 1)
            sigma = np.sqrt(variance / n)
        return mean, sigma, self._energy_deposited, self._energy_unscored
