"""Batched dose scoring.

Dose and dose-squared are accumulated per batch so that a per-voxel sigma is always
available (AGENTS.md section 2.4). This is required infrastructure: a Monte Carlo
estimate without an uncertainty is not an estimate.

Doses are per emitted history, in MeV/g. Absolute-dose conversion (Gy per MU or per
particle) is a calibration concern that does not belong in the scorer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyRadMC.geometry.grid import VoxelGrid

__all__ = ["BatchedDoseScorer", "DoseResult"]


@dataclass(frozen=True)
class DoseResult:
    """Finalized batched dose estimate.

    Attributes
    ----------
    dose
        Per-voxel mean dose over batches, MeV/g per emitted history.
    dose_sigma
        Per-voxel 1-sigma standard error of that mean, from the spread of the batch
        means. Zero in voxels no batch touched — and, degenerately, for a single
        batch, where no spread estimate exists.
    energy_deposited
        Total energy deposited over all batches, MeV. Exact bookkeeping, not an
        estimate; used for energy-conservation checks.
    n_batches
        Number of batches combined.
    """

    dose: np.ndarray
    dose_sigma: np.ndarray
    energy_deposited: float
    n_batches: int


class BatchedDoseScorer:
    """Accumulates energy deposits into per-batch dose grids.

    Usage: ``deposit`` any number of times, ``end_batch`` after each batch,
    ``finalize`` once, exactly ``n_batches`` batches later.
    """

    def __init__(self, grid: VoxelGrid, n_batches: int) -> None:
        if n_batches < 1:
            raise ValueError(f"need at least one batch, got {n_batches}")
        self._n_batches = n_batches
        self._closed = 0
        self._voxel_mass = grid.density * grid.voxel_volume  # g
        self._current_energy = np.zeros(grid.shape, dtype=np.float64)
        self._batch_sum = np.zeros(grid.shape, dtype=np.float64)
        self._batch_sum_sq = np.zeros(grid.shape, dtype=np.float64)
        self._energy_deposited = 0.0

    def deposit(self, ix: int, iy: int, iz: int, energy: float) -> None:
        """Add an energy deposit, in MeV, to a voxel of the current batch."""
        self._current_energy[ix, iy, iz] += energy
        self._energy_deposited += energy

    def deposit_grid(self, energy: np.ndarray) -> None:
        """Add a whole per-voxel energy grid, in MeV, to the current batch.

        The bulk entry point for kernel backends, which score device-side and hand
        back one array per batch; statistically identical to an equivalent sequence
        of :meth:`deposit` calls.
        """
        if energy.shape != self._current_energy.shape:
            raise ValueError(
                f"energy grid shape {energy.shape} != dose grid {self._current_energy.shape}"
            )
        self._current_energy += energy
        self._energy_deposited += float(energy.sum())

    def end_batch(self, n_histories: int) -> None:
        """Close the current batch of ``n_histories`` emitted histories."""
        if self._closed >= self._n_batches:
            raise RuntimeError("all batches already closed")
        if n_histories < 1:
            raise ValueError(f"empty batch ({n_histories} histories)")
        batch_dose = self._current_energy / (self._voxel_mass * n_histories)
        self._batch_sum += batch_dose
        self._batch_sum_sq += batch_dose**2
        self._current_energy[:] = 0.0
        self._closed += 1

    def finalize(self) -> DoseResult:
        """Combine the batch means into the dose estimate and its standard error."""
        if self._closed != self._n_batches:
            raise RuntimeError(f"finalize after {self._closed}/{self._n_batches} batch closures")
        n = self._n_batches
        mean = self._batch_sum / n
        if n == 1:
            sigma = np.zeros_like(mean)
        else:
            # Sample variance of the batch means; clamped against the cancellation
            # of sum_sq/n - mean^2 going epsilon-negative in float64.
            variance = np.maximum(0.0, self._batch_sum_sq / n - mean**2) * n / (n - 1)
            sigma = np.sqrt(variance / n)
        return DoseResult(
            dose=mean,
            dose_sigma=sigma,
            energy_deposited=self._energy_deposited,
            n_batches=n,
        )
