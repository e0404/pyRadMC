"""Dij on the reference engine: the correctness anchor for beamlet scoring.

Two properties make the design testable to the bit on one target:
streams are pure functions of (seed, history_index), and beamlet assignment is a
deterministic function of the history index. A 1x1 lattice therefore transports
*exactly* the histories of the open-field run, and its single Dij column must
equal that run's dose bit for bit — normalization, batching and sigma included.
Everything statistical (the fluence-sum identity against an open field) uses the
batched chi-squared oracle per AGENTS.md 4.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc import DIJ_TRUNCATION_RELATIVE
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

FIELD_X = (2.0, 14.0)
FIELD_Y = (2.0, 14.0)
ENERGY = 6.0
Z0 = -1.0


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(2.0, 2.0, 0.5))


def _engine() -> ReferenceEngine:
    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())


def _lattice(n_x: int, n_y: int) -> BeamletGridSource:
    return BeamletGridSource(
        energy=ENERGY, z=Z0, x_range=FIELD_X, y_range=FIELD_Y, n_x=n_x, n_y=n_y
    )


def _open_field() -> ParallelBeamSource:
    return ParallelBeamSource(energy=ENERGY, z=Z0, x_range=FIELD_X, y_range=FIELD_Y)


class TestSingleBeamletIsTheOpenField:
    """1x1 lattice, untruncated: the Dij degenerates to the dose run, bitwise."""

    def test_column_equals_open_field_dose_bitwise(self) -> None:
        engine = _engine()
        dij = engine.run_dij(
            _lattice(1, 1),
            n_histories_per_beamlet=1_200,
            n_batches=6,
            seed=SEED,
            truncation=0.0,
        )
        run = engine.run(_open_field(), n_histories=1_200, n_batches=6, seed=SEED)
        np.testing.assert_array_equal(dij.column_dense(0), run.dose)
        np.testing.assert_array_equal(dij.sigma_dense(0), run.dose_sigma)
        assert dij.energy_deposited == run.energy_deposited
        assert dij.energy_escaped == run.energy_escaped
        assert dij.energy_emitted == run.energy_emitted


class TestFluenceSum:
    def test_uniform_weights_reproduce_the_open_field_statistically(self) -> None:
        """Mean over untruncated columns is an open-field dose estimate.

        Different stratification means different histories, so the comparison is
        statistical (chi-squared, batched sigmas on both sides) — never bitwise.

        ``correlated=False`` is load-bearing here, not incidental: the sigma of
        the column mean below combines the per-column sigmas in quadrature,
        which is valid only when the columns are statistically independent. The
        shipped default is correlated, whose columns share streams; a
        quadrature sigma would then understate the variance and the chi-squared
        would false-fail. The independent mapping is the sanctioned test
        instrument for exactly this (AGENTS.md 2.10). The identity under test —
        that columns partition the open field — holds in expectation regardless
        of the mapping.
        """
        engine = _engine()
        n_x, n_y = 2, 2
        dij = engine.run_dij(
            _lattice(n_x, n_y),
            n_histories_per_beamlet=800,
            n_batches=8,
            seed=SEED,
            transport_electrons=False,
            truncation=0.0,
            correlated=False,
        )
        run = engine.run(
            _open_field(),
            n_histories=3_200,
            n_batches=8,
            seed=SEED + 1,  # independent histories for a fair statistical comparison
            transport_electrons=False,
        )
        n = dij.n_beamlets
        dose_sum = dij.dose_for_weights(np.full(n, 1.0 / n))
        # Columns are independent, so the variance of the mean is the mean of
        # variances over n; sigma maps combine in quadrature.
        var_sum = np.zeros(dij.grid_shape)
        for j in range(n):
            var_sum += dij.sigma_dense(j) ** 2
        sigma_sum = np.sqrt(var_sum) / n
        mask = run.dose > 0.1 * run.dose.max()
        mask &= (sigma_sum > 0.0) & (run.dose_sigma > 0.0)
        assert np.count_nonzero(mask) > 200, "mask too small to detect anything"
        assert_chi2_consistent_batched(
            dose_sum,
            sigma_sum,
            dij.n_batches,
            run.dose,
            run.dose_sigma,
            run.n_batches,
            mask=mask,
        )


class TestColumnPhysics:
    def test_columns_are_localized_under_their_beamlet(self) -> None:
        """Each beamlet's dose maximum lies inside its own x-y footprint (KERMA)."""
        engine = _engine()
        src = _lattice(2, 2)
        dij = engine.run_dij(
            src,
            n_histories_per_beamlet=400,
            n_batches=2,
            seed=SEED,
            transport_electrons=False,
        )
        grid = _grid()
        for j in range(src.n_beamlets):
            col = dij.column_dense(j)
            ix, iy, _ = np.unravel_index(np.argmax(col), col.shape)
            x = grid.origin[0] + (ix + 0.5) * grid.spacing[0]
            y = grid.origin[1] + (iy + 0.5) * grid.spacing[1]
            x_lo, x_hi, y_lo, y_hi = src.beamlet_bounds(j)
            # Half a voxel of slack: the maximum voxel's center can sit just
            # outside a beamlet edge that cuts through the voxel.
            assert x_lo - 1.0 <= x <= x_hi + 1.0
            assert y_lo - 1.0 <= y <= y_hi + 1.0

    def test_high_dose_entries_carry_sigma(self) -> None:
        """Dose-squared scoring reaches every column (AGENTS.md 2.4)."""
        dij = _engine().run_dij(
            _lattice(2, 2),
            n_histories_per_beamlet=400,
            n_batches=4,
            seed=SEED,
            transport_electrons=False,
        )
        for j in range(4):
            col = dij.column_dense(j)
            sig = dij.sigma_dense(j)
            high = col > 0.5 * col.max()
            assert np.all(sig[high] > 0.0)


class TestBookkeeping:
    def test_energy_conservation_is_exact(self) -> None:
        """float64 reference: emitted = deposited + escaped to arithmetic precision.

        Truncation must not touch the energy books - they are exact bookkeeping
        from the scorer, closed before sparsification.
        """
        dij = _engine().run_dij(
            _lattice(2, 1),
            n_histories_per_beamlet=300,
            n_batches=2,
            seed=SEED,
        )
        assert dij.energy_deposited + dij.energy_escaped == pytest.approx(
            dij.energy_emitted, rel=1.0e-12
        )
        assert dij.energy_emitted == pytest.approx(600 * ENERGY, rel=1.0e-12)

    def test_default_truncation_sparsifies_but_keeps_the_high_dose_region(self) -> None:
        # Runs under the shipped correlated default: truncation logic is
        # mapping-independent, so it is certified where it lives. The scatter
        # tail the non-vacuity guard needs is realization-dependent, though,
        # and the correlated mapping's compact low-statistics realization has
        # none at 400 histories/beamlet; 800 populates it reliably.
        engine = _engine()
        kwargs = dict(
            source=_lattice(2, 2),
            n_histories_per_beamlet=800,
            n_batches=2,
            seed=SEED,
            transport_electrons=False,
        )
        full = engine.run_dij(truncation=0.0, **kwargs)
        cut = engine.run_dij(**kwargs)  # default DIJ_TRUNCATION_RELATIVE
        assert cut.truncation == DIJ_TRUNCATION_RELATIVE
        assert cut.indices.size < full.indices.size
        for j in range(4):
            col_full = full.column_dense(j)
            col_cut = cut.column_dense(j)
            high = col_full >= 0.01 * col_full.max()
            np.testing.assert_array_equal(col_cut[high], col_full[high])

    def test_histories_must_divide_into_batches(self) -> None:
        with pytest.raises(ValueError):
            _engine().run_dij(_lattice(1, 1), n_histories_per_beamlet=5, n_batches=2, seed=SEED)
