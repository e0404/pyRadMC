"""Dij scoring: batched per-beamlet accumulation and sparse assembly (Phase 3).

The scorer is the per-column generalization of BatchedDoseScorer: dose *and*
dose-squared per batch per beamlet, so every Dij column carries a sigma
(AGENTS.md 2.4). The assembler turns dense column blocks into the sparse Dij,
applying the truncation threshold of AGENTS.md 2.8 — whose dosimetric effect is
tested in the dij tier against DVH endpoints, not here and never via a norm.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC import DIJ_TRUNCATION_RELATIVE
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.scoring.dij import BatchedBeamletScorer, DijAssembler


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))


N_VOX = 8


class TestBatchedBeamletScorer:
    def test_deposits_land_in_their_beamlet_column(self) -> None:
        scorer = BatchedBeamletScorer(_grid(), n_batches=1, n_beamlets=2)
        scorer.deposit(0, 0, 0, 0, 3.0)
        scorer.deposit(1, 1, 1, 1, 5.0)
        scorer.end_batch(n_histories=1)
        block = scorer.finalize()
        # Water at 1 g/cm^3, 1 cm^3 voxels: dose in MeV/g equals energy in MeV.
        assert block.dose[0, 0] == pytest.approx(3.0)
        assert block.dose[1, N_VOX - 1] == pytest.approx(5.0)
        assert np.count_nonzero(block.dose) == 2

    def test_flat_voxel_index_is_c_order(self) -> None:
        """(ix, iy, iz) flattens C-order, matching the engines' edep layout."""
        scorer = BatchedBeamletScorer(_grid(), n_batches=1, n_beamlets=1)
        scorer.deposit(0, 1, 0, 1, 1.0)  # flat = (1*2 + 0)*2 + 1 = 5
        scorer.end_batch(n_histories=1)
        assert scorer.finalize().dose[0, 5] == pytest.approx(1.0)

    def test_dose_is_per_emitted_history_of_that_beamlet(self) -> None:
        scorer = BatchedBeamletScorer(_grid(), n_batches=1, n_beamlets=1)
        scorer.deposit(0, 0, 0, 0, 6.0)
        scorer.end_batch(n_histories=4)
        assert scorer.finalize().dose[0, 0] == pytest.approx(1.5)

    def test_sigma_from_batch_spread(self) -> None:
        """Two batches a, b: sigma of the mean is |a - b| / 2, per column."""
        scorer = BatchedBeamletScorer(_grid(), n_batches=2, n_beamlets=2)
        scorer.deposit(0, 0, 0, 0, 1.0)
        scorer.end_batch(n_histories=1)
        scorer.deposit(0, 0, 0, 0, 3.0)
        scorer.end_batch(n_histories=1)
        block = scorer.finalize()
        assert block.dose[0, 0] == pytest.approx(2.0)
        assert block.sigma[0, 0] == pytest.approx(1.0)
        assert block.sigma[1].max() == 0.0  # untouched column: no spread

    def test_deposit_block_matches_pointwise_deposits(self) -> None:
        a = BatchedBeamletScorer(_grid(), n_batches=1, n_beamlets=2)
        b = BatchedBeamletScorer(_grid(), n_batches=1, n_beamlets=2)
        energies = np.zeros((2, N_VOX))
        a.deposit(0, 0, 1, 0, 2.5)
        energies[0, 2] = 2.5
        a.deposit(1, 1, 1, 1, 0.5)
        energies[1, 7] = 0.5
        b.deposit_block(energies)
        for s in (a, b):
            s.end_batch(n_histories=2)
        ra, rb = a.finalize(), b.finalize()
        np.testing.assert_array_equal(ra.dose, rb.dose)
        assert ra.energy_deposited == pytest.approx(rb.energy_deposited)

    def test_energy_deposited_is_exact_bookkeeping(self) -> None:
        scorer = BatchedBeamletScorer(_grid(), n_batches=1, n_beamlets=2)
        scorer.deposit(0, 0, 0, 0, 1.25)
        scorer.deposit(1, 0, 0, 0, 0.75)
        scorer.end_batch(n_histories=1)
        assert scorer.finalize().energy_deposited == pytest.approx(2.0)

    def test_batch_protocol_is_enforced(self) -> None:
        scorer = BatchedBeamletScorer(_grid(), n_batches=2, n_beamlets=1)
        scorer.end_batch(n_histories=1)
        with pytest.raises(RuntimeError):
            scorer.finalize()
        scorer.end_batch(n_histories=1)
        with pytest.raises(RuntimeError):
            scorer.end_batch(n_histories=1)


class TestDijAssembler:
    def _assemble(self, dose: np.ndarray, sigma: np.ndarray, truncation: float):
        n_beamlets = dose.shape[0]
        asm = DijAssembler(
            grid_shape=(2, 2, 2),
            n_beamlets=n_beamlets,
            n_histories_per_beamlet=10,
            n_batches=2,
            truncation=truncation,
        )
        asm.add_block(0, dose, sigma)
        return asm.finalize(
            energy_emitted=1.0, energy_deposited=float(dose.sum()), energy_escaped=0.0
        )

    def test_truncation_drops_below_relative_column_max(self) -> None:
        dose = np.zeros((1, N_VOX))
        dose[0, :4] = [10.0, 10.0e-3, 9.0e-3, 0.0]  # threshold at 1e-3 * 10 = 1e-2
        dij = self._assemble(dose, np.zeros_like(dose), DIJ_TRUNCATION_RELATIVE)
        kept = dij.column_dense(0).ravel()
        assert kept[0] == pytest.approx(10.0)
        assert kept[1] == pytest.approx(10.0e-3)  # exactly at threshold: kept
        assert kept[2] == 0.0  # below threshold: truncated
        assert dij.indices.size == 2

    def test_truncation_is_per_column(self) -> None:
        dose = np.zeros((2, N_VOX))
        dose[0, 0] = 100.0
        dose[1, 0] = 1.0
        dose[1, 1] = 5.0e-3  # above 1e-3 of *its own* column max, kept
        dij = self._assemble(dose, np.zeros_like(dose), DIJ_TRUNCATION_RELATIVE)
        assert dij.column_dense(1).ravel()[1] == pytest.approx(5.0e-3)

    def test_zero_truncation_keeps_every_nonzero_entry(self) -> None:
        dose = np.zeros((1, N_VOX))
        dose[0, :3] = [1.0, 1.0e-9, 1.0e-12]
        dij = self._assemble(dose, np.zeros_like(dose), truncation=0.0)
        assert dij.indices.size == 3

    def test_sigma_stays_aligned_with_kept_entries(self) -> None:
        dose = np.zeros((1, N_VOX))
        sigma = np.zeros((1, N_VOX))
        dose[0, :3] = [1.0, 0.5, 1.0e-5]
        sigma[0, :3] = [0.1, 0.05, 0.9]
        dij = self._assemble(dose, sigma, DIJ_TRUNCATION_RELATIVE)
        np.testing.assert_allclose(dij.sigma_dense(0).ravel()[:2], [0.1, 0.05])
        assert dij.sigma_dense(0).ravel()[2] == 0.0

    def test_multi_block_assembly_and_weighted_dose(self) -> None:
        asm = DijAssembler(
            grid_shape=(2, 2, 2),
            n_beamlets=3,
            n_histories_per_beamlet=10,
            n_batches=2,
            truncation=0.0,
        )
        dose = np.arange(3 * N_VOX, dtype=float).reshape(3, N_VOX) + 1.0
        asm.add_block(0, dose[:2], np.zeros((2, N_VOX)))
        asm.add_block(2, dose[2:], np.zeros((1, N_VOX)))
        dij = asm.finalize(energy_emitted=1.0, energy_deposited=0.0, energy_escaped=0.0)
        weights = np.array([1.0, 0.0, 2.0])
        expected = (weights[:, None] * dose).sum(axis=0).reshape(2, 2, 2)
        np.testing.assert_allclose(dij.dose_for_weights(weights), expected)

    def test_blocks_must_cover_all_beamlets_exactly_once(self) -> None:
        asm = DijAssembler(
            grid_shape=(2, 2, 2),
            n_beamlets=2,
            n_histories_per_beamlet=10,
            n_batches=2,
            truncation=0.0,
        )
        asm.add_block(0, np.zeros((1, N_VOX)), np.zeros((1, N_VOX)))
        with pytest.raises(RuntimeError):
            asm.finalize(energy_emitted=1.0, energy_deposited=0.0, energy_escaped=0.0)

    def test_csc_export_matches_dense_columns(self) -> None:
        rng = np.random.default_rng(3)
        dose = rng.random((3, N_VOX))
        dose[dose < 0.3] = 0.0
        dij = self._assemble_multi(dose)
        csc = dij.dose_csc()
        assert csc.shape == (N_VOX, 3)
        for j in range(3):
            np.testing.assert_allclose(
                np.asarray(csc[:, [j]].todense()).ravel(), dij.column_dense(j).ravel()
            )

    def _assemble_multi(self, dose: np.ndarray):
        asm = DijAssembler(
            grid_shape=(2, 2, 2),
            n_beamlets=dose.shape[0],
            n_histories_per_beamlet=10,
            n_batches=2,
            truncation=0.0,
        )
        asm.add_block(0, dose, np.zeros_like(dose))
        return asm.finalize(energy_emitted=1.0, energy_deposited=0.0, energy_escaped=0.0)
