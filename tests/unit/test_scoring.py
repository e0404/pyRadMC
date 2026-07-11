"""Batched dose scoring: the mean/sigma bookkeeping, checked against hand arithmetic.

Sigma from batch statistics is required infrastructure (AGENTS.md section 2.4); if the
formulas here drift, every statistical oracle in the suite silently loses calibration.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_grid():
    from pyRadMC.geometry.grid import VoxelGrid

    return VoxelGrid.uniform_water(shape=(2, 1, 1), spacing=(2.0, 1.0, 1.0))


def test_mean_and_sigma_match_hand_calculation() -> None:
    """Three batches, one voxel fed known energies: mean and sigma are textbook."""
    from pyRadMC.scoring.dose import BatchedDoseScorer

    grid = _make_grid()
    scorer = BatchedDoseScorer(grid, n_batches=3)

    # Voxel (0,0,0): mass = 1.0 g/cm^3 * 2 cm^3 = 2 g. 10 histories per batch.
    for batch_energy in (4.0, 6.0, 8.0):
        scorer.deposit(0, 0, 0, batch_energy)
        scorer.end_batch(n_histories=10)
    result = scorer.finalize()

    # Per-history dose per batch: (4, 6, 8) MeV / 2 g / 10 = (0.2, 0.3, 0.4) MeV/g.
    assert result.dose[0, 0, 0] == pytest.approx(0.3)
    # Standard error of the mean of the three batch values: std(ddof=1)/sqrt(3).
    expected_sigma = np.std([0.2, 0.3, 0.4], ddof=1) / np.sqrt(3.0)
    assert result.dose_sigma[0, 0, 0] == pytest.approx(expected_sigma)

    # Untouched voxel: zero dose, zero spread.
    assert result.dose[1, 0, 0] == 0.0
    assert result.dose_sigma[1, 0, 0] == 0.0

    assert result.energy_deposited == pytest.approx(18.0)


def test_deposits_within_a_batch_accumulate() -> None:
    from pyRadMC.scoring.dose import BatchedDoseScorer

    scorer = BatchedDoseScorer(_make_grid(), n_batches=1)
    scorer.deposit(1, 0, 0, 1.5)
    scorer.deposit(1, 0, 0, 2.5)
    scorer.end_batch(n_histories=4)
    result = scorer.finalize()
    assert result.dose[1, 0, 0] == pytest.approx(4.0 / 2.0 / 4.0)


def test_finalize_requires_all_batches_closed() -> None:
    from pyRadMC.scoring.dose import BatchedDoseScorer

    scorer = BatchedDoseScorer(_make_grid(), n_batches=2)
    scorer.deposit(0, 0, 0, 1.0)
    scorer.end_batch(n_histories=1)
    with pytest.raises(RuntimeError, match="batch"):
        scorer.finalize()


def test_deposit_grid_equals_equivalent_scalar_deposits() -> None:
    """The kernel backends' bulk entry point is exactly a batch of deposit calls."""
    from pyRadMC.scoring.dose import BatchedDoseScorer

    grid = _make_grid()
    energy = np.zeros(grid.shape)
    energy[0, 0, 0] = 1.25
    energy[1, 0, 0] = 0.75

    bulk = BatchedDoseScorer(grid, n_batches=1)
    bulk.deposit_grid(energy)
    bulk.end_batch(n_histories=5)

    scalar = BatchedDoseScorer(grid, n_batches=1)
    scalar.deposit(0, 0, 0, 1.25)
    scalar.deposit(1, 0, 0, 0.75)
    scalar.end_batch(n_histories=5)

    a, b = bulk.finalize(), scalar.finalize()
    np.testing.assert_array_equal(a.dose, b.dose)
    assert a.energy_deposited == pytest.approx(b.energy_deposited)


def test_deposit_grid_rejects_shape_mismatch() -> None:
    from pyRadMC.scoring.dose import BatchedDoseScorer

    scorer = BatchedDoseScorer(_make_grid(), n_batches=1)
    with pytest.raises(ValueError, match="shape"):
        scorer.deposit_grid(np.zeros((1, 2, 3)))
