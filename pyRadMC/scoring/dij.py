"""Batched Dij scoring and sparse assembly.

The Dij pipeline has two host-side stages, shared by every backend:

:class:`BatchedBeamletScorer`
    The per-column generalization of :class:`pyRadMC.scoring.dose.BatchedDoseScorer`:
    dense ``(n_beamlets, n_voxels)`` energy accumulation, closed per batch, so each
    Dij column carries a batch-spread sigma (AGENTS.md 2.4). Engines score one
    *group* of beamlets at a time through it — the group size bounds the dense
    memory and is statistically and bit-wise inert (test-pinned).

:class:`DijAssembler`
    Collects the finalized dense blocks into one sparse CSC-layout Dij, applying
    the truncation threshold. Truncation is relative to each column's own maximum
    (AGENTS.md 2.8): it is a physics decision disguised as a memory optimization,
    biases the low-dose tail, and its default lives in :data:`pyRadMC.DIJ_TRUNCATION_RELATIVE`.
    Its dosimetric effect is tested against DVH endpoints in the dij tier, never
    against a matrix norm.

Doses are per emitted history *of that beamlet*, in MeV/g, so a column scaled by a
fluence weight is that beamlet's dose contribution; absolute calibration stays out
of the scorer, as in the dose module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.scoring.grid import ScoringGrid

if TYPE_CHECKING:
    from scipy.sparse import csc_array

__all__ = ["BatchedBeamletScorer", "BeamletDoseBlock", "DijAssembler", "DijResult"]


@dataclass(frozen=True)
class BeamletDoseBlock:
    """Finalized dense dose block for a contiguous group of beamlets.

    Attributes
    ----------
    dose
        ``(n_beamlets, n_voxels)`` mean dose per emitted history of each beamlet,
        MeV/g, scoring-grid voxels flattened C-order.
    sigma
        Matching 1-sigma standard error from the batch spread; zero where no batch
        deposited and, degenerately, for a single batch.
    energy_deposited
        Total energy deposited by the group into the scoring grid over all
        batches, MeV. Exact bookkeeping for conservation checks, not an estimate.
    energy_unscored
        Total energy deposited inside the transport grid but outside the scoring
        grid, MeV. Exactly zero when the scoring grid covers the transport grid.
    """

    dose: np.ndarray
    sigma: np.ndarray
    energy_deposited: float
    energy_unscored: float


class BatchedBeamletScorer:
    """Accumulates energy deposits into per-batch, per-beamlet dose rows.

    Usage mirrors :class:`~pyRadMC.scoring.dose.BatchedDoseScorer`: ``deposit`` (or
    ``deposit_block``) any number of times, ``end_batch`` after each batch with the
    per-beamlet history count of that batch, ``finalize`` once. The beamlet index
    here is *local to the group* being scored; the assembler restores global
    column positions.
    """

    def __init__(self, grid: VoxelGrid | ScoringGrid, n_batches: int, n_beamlets: int) -> None:
        if n_batches < 1:
            raise ValueError(f"need at least one batch, got {n_batches}")
        if n_beamlets < 1:
            raise ValueError(f"need at least one beamlet, got {n_beamlets}")
        scoring = grid if isinstance(grid, ScoringGrid) else ScoringGrid.for_grid(grid)
        self._n_batches = n_batches
        self._n_beamlets = n_beamlets
        self._closed = 0
        self._scoring = scoring
        self._grid_shape = scoring.shape
        self._ny = scoring.shape[1]
        self._nz = scoring.shape[2]
        n_voxels = scoring.n_voxels
        # Row of per-voxel masses, flattened C-order like the deposits.
        self._voxel_mass = scoring.voxel_mass.reshape(n_voxels)  # g
        self._covered = self._voxel_mass > 0.0
        self._current_energy = np.zeros((n_beamlets, n_voxels), dtype=np.float64)
        self._batch_sum = np.zeros((n_beamlets, n_voxels), dtype=np.float64)
        self._batch_sum_sq = np.zeros((n_beamlets, n_voxels), dtype=np.float64)
        self._energy_deposited = 0.0
        self._energy_unscored = 0.0

    def deposit(
        self, beamlet: int, ix: int, iy: int, iz: int, energy: float, scored: float | None = None
    ) -> None:
        """Add an energy deposit, in MeV, to a voxel of one beamlet's current batch.

        ``energy`` is the physical energy the ledger books; ``scored`` (default:
        the same) is what the dose tally accumulates.
        """
        flat = (ix * self._ny + iy) * self._nz + iz
        self._current_energy[beamlet, flat] += energy if scored is None else scored
        self._energy_deposited += energy

    def deposit_at(
        self,
        beamlet: int,
        x: float,
        y: float,
        z: float,
        energy: float,
        scored: float | None = None,
    ) -> None:
        """Add a deposit at a position (cm) inside the transport grid.

        Routes to the containing scoring voxel of the beamlet's current batch, or
        to the unscored bucket when the position lies outside the scoring grid
        (mirror of :meth:`pyRadMC.scoring.dose.BatchedDoseScorer.deposit_at`); the
        unscored ledger always books the physical energy.
        """
        if self._scoring.contains(x, y, z):
            ix, iy, iz = self._scoring.voxel_index(x, y, z)
            self.deposit(beamlet, ix, iy, iz, energy, scored)
        else:
            self._energy_unscored += energy

    def add_unscored(self, energy: float) -> None:
        """Book energy, in MeV, that a kernel backend tallied as unscored."""
        self._energy_unscored += energy

    def deposit_block(self, energy: np.ndarray) -> None:
        """Add a whole ``(n_beamlets, n_voxels)`` energy block to the current batch.

        The bulk entry point for kernel backends; statistically identical to the
        equivalent sequence of :meth:`deposit` calls.
        """
        if energy.shape != self._current_energy.shape:
            raise ValueError(f"energy block shape {energy.shape} != {self._current_energy.shape}")
        self._current_energy += energy
        self._energy_deposited += float(energy.sum())

    def end_batch(self, n_histories: int) -> None:
        """Close the current batch of ``n_histories`` emitted histories per beamlet."""
        if self._closed >= self._n_batches:
            raise RuntimeError("all batches already closed")
        if n_histories < 1:
            raise ValueError(f"empty batch ({n_histories} histories per beamlet)")
        # Uncovered scoring voxels (zero mass) have zero dose by definition; the
        # masked divide leaves them at exactly 0 instead of 0/0 = NaN.
        batch_dose = np.divide(
            self._current_energy,
            self._voxel_mass[np.newaxis, :] * n_histories,
            out=np.zeros_like(self._current_energy),
            where=self._covered[np.newaxis, :],
        )
        self._batch_sum += batch_dose
        self._batch_sum_sq += batch_dose**2
        self._current_energy[:] = 0.0
        self._closed += 1

    def finalize(self) -> BeamletDoseBlock:
        """Combine the batch means into per-column dose and standard error."""
        if self._closed != self._n_batches:
            raise RuntimeError(f"finalize after {self._closed}/{self._n_batches} batch closures")
        n = self._n_batches
        mean = self._batch_sum / n
        if n == 1:
            sigma = np.zeros_like(mean)
        else:
            # Same clamped estimator as BatchedDoseScorer.finalize.
            variance = np.maximum(0.0, self._batch_sum_sq / n - mean**2) * n / (n - 1)
            sigma = np.sqrt(variance / n)
        return BeamletDoseBlock(
            dose=mean,
            sigma=sigma,
            energy_deposited=self._energy_deposited,
            energy_unscored=self._energy_unscored,
        )


@dataclass(frozen=True)
class DijResult:
    """Sparse beamlet-resolved dose influence matrix, CSC layout by column.

    Column ``j`` holds beamlet ``j``'s dose per emitted history (MeV/g) in the
    voxels that survived truncation; ``sigma`` is the matching per-entry standard
    error. Voxel indices are flat C-order over ``grid_shape``.

    ``correlated`` records the stream mapping the Dij was computed under
    (Phase 4). When True, columns share random streams and are statistically
    *dependent*: each per-entry ``sigma`` stays valid on its own, but sigmas
    must never be combined across columns in quadrature — cross-column
    covariance is not carried here.
    """

    grid_shape: tuple[int, int, int]
    n_beamlets: int
    n_histories_per_beamlet: int
    n_batches: int
    truncation: float
    indptr: np.ndarray
    indices: np.ndarray
    dose: np.ndarray
    sigma: np.ndarray
    energy_emitted: float
    energy_deposited: float
    energy_escaped: float
    energy_unscored: float = 0.0
    correlated: bool = False
    scoring_mode: str = "dose_to_medium"

    @property
    def n_voxels(self) -> int:
        """Number of voxels (rows of the matrix)."""
        return int(np.prod(self.grid_shape))

    def column_dense(self, beamlet: int) -> np.ndarray:
        """One beamlet's truncated dose column as a dense ``grid_shape`` array."""
        return self._scatter(self.dose, beamlet)

    def sigma_dense(self, beamlet: int) -> np.ndarray:
        """One beamlet's per-entry sigma as a dense ``grid_shape`` array."""
        return self._scatter(self.sigma, beamlet)

    def _scatter(self, values: np.ndarray, beamlet: int) -> np.ndarray:
        if not 0 <= beamlet < self.n_beamlets:
            raise IndexError(f"beamlet {beamlet} outside 0..{self.n_beamlets - 1}")
        lo, hi = int(self.indptr[beamlet]), int(self.indptr[beamlet + 1])
        out = np.zeros(self.n_voxels, dtype=np.float64)
        out[self.indices[lo:hi]] = values[lo:hi]
        return out.reshape(self.grid_shape)

    def dose_for_weights(self, weights: np.ndarray) -> np.ndarray:
        """Dense dose grid for a fluence-weight vector: ``sum_j w_j * column_j``."""
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (self.n_beamlets,):
            raise ValueError(f"weights shape {w.shape} != ({self.n_beamlets},)")
        out = np.zeros(self.n_voxels, dtype=np.float64)
        for j in range(self.n_beamlets):
            lo, hi = int(self.indptr[j]), int(self.indptr[j + 1])
            np.add.at(out, self.indices[lo:hi], w[j] * self.dose[lo:hi])
        return out.reshape(self.grid_shape)

    def dose_csc(self) -> csc_array:
        """Export the dose matrix as a ``scipy.sparse.csc_array``, (n_voxels, n_beamlets)."""
        from scipy.sparse import csc_array

        return csc_array(
            (self.dose, self.indices, self.indptr), shape=(self.n_voxels, self.n_beamlets)
        )

    def sigma_csc(self) -> csc_array:
        """Export the per-entry sigma as a ``scipy.sparse.csc_array``, aligned with the dose."""
        from scipy.sparse import csc_array

        return csc_array(
            (self.sigma, self.indices, self.indptr), shape=(self.n_voxels, self.n_beamlets)
        )


@dataclass
class DijAssembler:
    """Assembles finalized dense blocks into one sparse :class:`DijResult`.

    Blocks arrive in ascending, gap-free beamlet order (the engines iterate groups
    in order); ``finalize`` checks complete coverage. Truncation keeps an entry
    when ``dose >= truncation * column_max`` and the dose is nonzero.
    """

    grid_shape: tuple[int, int, int]
    n_beamlets: int
    n_histories_per_beamlet: int
    n_batches: int
    truncation: float
    correlated: bool = False
    scoring_mode: str = "dose_to_medium"
    _next_beamlet: int = 0
    _indices: list[np.ndarray] = field(default_factory=list)
    _dose: list[np.ndarray] = field(default_factory=list)
    _sigma: list[np.ndarray] = field(default_factory=list)
    _counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the truncation threshold."""
        if self.truncation < 0.0:
            raise ValueError(f"negative truncation {self.truncation}")

    def add_block(self, start: int, dose: np.ndarray, sigma: np.ndarray) -> None:
        """Truncate and append a dense block whose first beamlet is column ``start``."""
        if start != self._next_beamlet:
            raise RuntimeError(f"block starts at {start}, expected {self._next_beamlet}")
        if dose.shape != sigma.shape or dose.ndim != 2:
            raise ValueError("dose and sigma must be matching (n_beamlets, n_voxels) blocks")
        for row_dose, row_sigma in zip(dose, sigma, strict=True):
            col_max = float(row_dose.max(initial=0.0))
            keep = row_dose >= self.truncation * col_max
            keep &= row_dose > 0.0
            idx = np.flatnonzero(keep)
            self._indices.append(idx.astype(np.int64))
            self._dose.append(row_dose[idx])
            self._sigma.append(row_sigma[idx])
            self._counts.append(int(idx.size))
        self._next_beamlet += dose.shape[0]

    def finalize(
        self,
        energy_emitted: float,
        energy_deposited: float,
        energy_escaped: float,
        energy_unscored: float = 0.0,
    ) -> DijResult:
        """Concatenate the columns into CSC arrays and close the books.

        The energy tallies are the *engine's* to close, not the assembler's: a
        kernel backend keeps them in exact integer quanta so that the books, like
        the matrix, are invariant to how beamlets were grouped into blocks.
        """
        if self._next_beamlet != self.n_beamlets:
            raise RuntimeError(f"assembled {self._next_beamlet}/{self.n_beamlets} beamlet columns")
        indptr = np.zeros(self.n_beamlets + 1, dtype=np.int64)
        np.cumsum(self._counts, out=indptr[1:])
        empty = np.zeros(0, dtype=np.float64)
        return DijResult(
            grid_shape=self.grid_shape,
            n_beamlets=self.n_beamlets,
            n_histories_per_beamlet=self.n_histories_per_beamlet,
            n_batches=self.n_batches,
            truncation=self.truncation,
            indptr=indptr,
            indices=(
                np.concatenate(self._indices) if self._indices else np.zeros(0, dtype=np.int64)
            ),
            dose=np.concatenate(self._dose) if self._dose else empty,
            sigma=np.concatenate(self._sigma) if self._sigma else empty,
            energy_emitted=energy_emitted,
            energy_deposited=energy_deposited,
            energy_escaped=energy_escaped,
            energy_unscored=energy_unscored,
            correlated=self.correlated,
            scoring_mode=self.scoring_mode,
        )
