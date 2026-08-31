"""Batched dose scoring.

Dose and dose-squared are accumulated per batch so that a per-voxel sigma is always
available (AGENTS.md section 2.4). This is required infrastructure: a Monte Carlo
estimate without an uncertainty is not an estimate.

Scoring happens on a *scoring geometry*: a :class:`~pyradmc.scoring.grid.ScoringGrid`
(rectilinear, by default the transport grid itself, but any coarser, offset or
subregion dose grid) or a
:class:`~pyradmc.scoring.cylinder.CylindricalScoringGrid` (depth by radial shell,
for pencil-beam kernels). The scorer is written against the two queries both answer
— ``flat_index`` and a ``voxel_mass`` map — and knows nothing else about either.

Position-based deposits (:meth:`BatchedDoseScorer.deposit_at`) that fall inside the
transport grid but outside the scoring geometry are booked to the **unscored** energy
bucket — never clamped into an edge bin, which would corrupt edge dose — so the
engine ledger ``emitted == deposited + unscored + escaped`` stays exact.

Doses are per emitted history, in MeV/g. Absolute-dose conversion (Gy per MU or per
particle) is a calibration concern that does not belong in the scorer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyradmc.geometry.grid import VoxelGrid
from pyradmc.scoring.cylinder import CylindricalScoringGrid
from pyradmc.scoring.grid import ScoringGrid

__all__ = ["BatchedDoseScorer", "DoseResult", "ScoringGeometry"]

ScoringGeometry = ScoringGrid | CylindricalScoringGrid
"""A geometry dose may be accumulated on: rectilinear voxels, or cylindrical shells.

The engines accept either through their ``scoring_grid`` argument, and every
consumer — the batched scorer here, the Warp ``_scoring_info`` upload — is written
against the pair of queries both provide (``flat_index`` for routing, ``voxel_mass``
for normalization) rather than against one geometry's fields. Transport is invariant
to the choice in both cases: the RNG streams never see it.
"""


@dataclass(frozen=True)
class DoseResult:
    """Finalized batched dose estimate.

    Attributes
    ----------
    dose
        Per-voxel mean dose over batches, MeV/g per emitted history, on the
        scoring grid. Zero in scoring voxels the transport grid does not cover
        (their mass is zero and no energy can arrive there).
    dose_sigma
        Per-voxel 1-sigma standard error of that mean, from the spread of the batch
        means. Zero in voxels no batch touched — and, degenerately, for a single
        batch, where no spread estimate exists.
    energy_deposited
        Total energy deposited *into the scoring grid* over all batches, MeV.
        Exact bookkeeping, not an estimate; used for energy-conservation checks.
    energy_unscored
        Total energy deposited inside the transport grid but outside the scoring
        geometry, MeV. Exactly zero when the scoring grid covers the transport grid
        (in particular for the default transport-grid scoring); for a cylindrical
        geometry it carries everything the shells do not span, which is ordinarily
        nonzero and is what keeps the engine's energy ledger exact.
    n_batches
        Number of batches combined.
    """

    dose: np.ndarray
    dose_sigma: np.ndarray
    energy_deposited: float
    energy_unscored: float
    n_batches: int


class BatchedDoseScorer:
    """Accumulates energy deposits into per-batch dose grids.

    Usage: ``deposit``/``deposit_at`` any number of times, ``end_batch`` after each
    batch, ``finalize`` once, exactly ``n_batches`` batches later.

    Accepts any :data:`ScoringGeometry` or, as a convenience for the
    score-on-the-transport-grid case, the transport
    :class:`~pyradmc.geometry.grid.VoxelGrid` itself.
    """

    def __init__(self, grid: VoxelGrid | ScoringGeometry, n_batches: int) -> None:
        if n_batches < 1:
            raise ValueError(f"need at least one batch, got {n_batches}")
        scoring = grid if not isinstance(grid, VoxelGrid) else ScoringGrid.for_grid(grid)
        self._n_batches = n_batches
        self._closed = 0
        self._scoring = scoring
        self._voxel_mass = scoring.voxel_mass  # g
        self._covered = scoring.voxel_mass > 0.0
        self._current_energy = np.zeros(scoring.shape, dtype=np.float64)
        # Flat view of the same buffer: position-routed deposits address it by the
        # scoring geometry's own C-order index, which is the one query both the
        # rectilinear and the cylindrical geometry answer. Same memory, same
        # arithmetic — the indexed path below is unchanged.
        self._flat_energy = self._current_energy.reshape(-1)
        self._batch_sum = np.zeros(scoring.shape, dtype=np.float64)
        self._batch_sum_sq = np.zeros(scoring.shape, dtype=np.float64)
        self._energy_deposited = 0.0
        self._energy_unscored = 0.0

    def deposit(
        self, ix: int, iy: int, iz: int, energy: float, scored: float | None = None
    ) -> None:
        """Add an energy deposit, in MeV, to a scoring voxel of the current batch.

        The rectilinear entry point, kept for callers holding voxel indices;
        position-routed deposits go through :meth:`deposit_at`, which serves every
        scoring geometry.

        ``energy`` is the physical energy the ledger books; ``scored`` (default:
        the same) is what the dose tally accumulates — the dose-to-water weighted
        amount when that mode is on.
        """
        self._current_energy[ix, iy, iz] += energy if scored is None else scored
        self._energy_deposited += energy

    def deposit_at(
        self, x: float, y: float, z: float, energy: float, scored: float | None = None
    ) -> None:
        """Add a deposit at a position (cm) inside the transport grid.

        Routes to the containing scoring bin, or to the unscored bucket when the
        position lies outside the scoring geometry — the unscored ledger always
        books the *physical* energy. This is the transport loops'
        :data:`~pyradmc.transport.particles.DepositFn`.

        The routing goes through the geometry's own ``flat_index``, which is the
        single query a rectilinear grid and a cylindrical shell binning both
        answer, so neither the scorer nor the transport loop above it knows which
        one it is scoring on.
        """
        index = self._scoring.flat_index(x, y, z)
        if index >= 0:
            self._flat_energy[index] += energy if scored is None else scored
            self._energy_deposited += energy
        else:
            self._energy_unscored += energy

    def add_unscored(self, energy: float) -> None:
        """Book energy, in MeV, that a kernel backend tallied as unscored."""
        self._energy_unscored += energy

    def deposit_grid(self, energy: np.ndarray, booked: float | None = None) -> None:
        """Add a whole per-voxel energy grid, in MeV, to the current batch.

        The bulk entry point for kernel backends, which score device-side and hand
        back one array per batch; statistically identical to an equivalent sequence
        of :meth:`deposit` calls. ``booked`` overrides the physical energy entered
        into the ledger (a dose-to-water backend hands the water-weighted grid for
        the tally but the physical total for the books); default: the grid's sum.
        """
        if energy.shape != self._current_energy.shape:
            raise ValueError(
                f"energy grid shape {energy.shape} != dose grid {self._current_energy.shape}"
            )
        self._current_energy += energy
        self._energy_deposited += float(energy.sum()) if booked is None else booked

    def end_batch(self, n_histories: int) -> None:
        """Close the current batch of ``n_histories`` emitted histories."""
        if self._closed >= self._n_batches:
            raise RuntimeError("all batches already closed")
        if n_histories < 1:
            raise ValueError(f"empty batch ({n_histories} histories)")
        # Uncovered voxels (zero mass) have zero dose by definition; the masked
        # divide leaves them at exactly 0 instead of 0/0 = NaN.
        batch_dose = np.divide(
            self._current_energy,
            self._voxel_mass * n_histories,
            out=np.zeros_like(self._current_energy),
            where=self._covered,
        )
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
            energy_unscored=self._energy_unscored,
            n_batches=n,
        )
