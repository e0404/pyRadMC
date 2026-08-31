"""Dose-to-water scoring through both engines: exact identities per backend.

``scoring_mode`` is a scoring-OUTPUT selection, not a physics toggle (AGENTS.md
2.10): transport — streams, interaction sequences, stepping — is identical in
both modes, only the tally weighting differs. That makes the tests exact:

- medium == water: SPR is the ratio of identical numbers, so D_w is
  bit-identical to D_m;
- a synthetic medium whose restricted stopping is water's / 1.1 at every energy
  (proportional rows under log-linear interpolation) makes D_w == 1.1 * D_m to
  accumulation rounding;
- the energy ledger books *physical* energy in both modes, so
  ``energy_deposited`` is identical across modes and the three-bucket balance
  closes;
- KERMA mode has no tracked electron to evaluate the SPR on (it would need
  mass-energy-absorption ratios instead), so dose-to-water there is rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED
from tests.unit.test_dose_to_water_spr import DELTA_CUT, _two_material_tables

SPR = 1.1


def _water_grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(2.0, 2.0, 0.5))


def _medium_grid() -> VoxelGrid:
    """Uniform phantom of the synthetic 'medium' (tabulated material row 1)."""
    shape = (8, 8, 16)
    return VoxelGrid(
        shape=shape,
        spacing=(2.0, 2.0, 0.5),
        density=np.ones(shape),
        material=np.full(shape, 1, dtype=np.int32),
    )


def _source() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(2.0, 14.0), y_range=(2.0, 14.0))


def _ref_engine(grid: VoxelGrid, xs) -> ReferenceEngine:
    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())


def _medium_xs(grid: VoxelGrid) -> TabulatedCrossSections:
    return TabulatedCrossSections(
        _two_material_tables(spr=SPR), geometry_densities=grid.max_density_by_material()
    )


def _run(engine, mode: str, n: int = 400):
    return engine.run(
        _source(), n_histories=n, n_batches=4, seed=SEED, ecut=DELTA_CUT, scoring_mode=mode
    )


class TestReference:
    def test_water_dose_to_water_is_bit_identical_to_dose_to_medium(self) -> None:
        grid = _water_grid()
        engine = _ref_engine(grid, AnalyticCrossSections(grid.max_density_by_material()))
        dm = _run(engine, "dose_to_medium")
        dw = _run(engine, "dose_to_water")
        np.testing.assert_array_equal(dw.dose, dm.dose)
        np.testing.assert_array_equal(dw.dose_sigma, dm.dose_sigma)
        assert dw.energy_deposited == dm.energy_deposited
        assert dw.scoring_mode == "dose_to_water"
        assert dm.scoring_mode == "dose_to_medium"

    def test_uniform_medium_scales_by_the_constant_spr(self) -> None:
        grid = _medium_grid()
        engine = _ref_engine(grid, _medium_xs(grid))
        dm = _run(engine, "dose_to_medium")
        dw = _run(engine, "dose_to_water")

        touched = dm.dose > 0.0
        assert np.count_nonzero(touched) > 100
        np.testing.assert_allclose(dw.dose[touched], SPR * dm.dose[touched], rtol=1e-9)

        # The ledger is physical in both modes: transport is identical, so the
        # books are identical, and the three buckets close exactly.
        assert dw.energy_deposited == dm.energy_deposited
        assert dw.energy_escaped == dm.energy_escaped
        assert dw.energy_emitted == dm.energy_emitted
        assert dw.energy_deposited + dw.energy_unscored + dw.energy_escaped == pytest.approx(
            dw.energy_emitted, rel=1e-12
        )

    def test_dij_column_scales_by_the_constant_spr(self) -> None:
        grid = _medium_grid()
        engine = _ref_engine(grid, _medium_xs(grid))
        lattice = BeamletGridSource(
            energy=6.0, z=-1.0, x_range=(2.0, 14.0), y_range=(2.0, 14.0), n_x=1, n_y=1
        )
        kwargs = dict(
            n_histories_per_beamlet=400,
            n_batches=4,
            seed=SEED,
            ecut=DELTA_CUT,
            truncation=0.0,
        )
        dm = engine.run_dij(lattice, scoring_mode="dose_to_medium", **kwargs)
        dw = engine.run_dij(lattice, scoring_mode="dose_to_water", **kwargs)
        a, b = dm.column_dense(0), dw.column_dense(0)
        touched = a > 0.0
        np.testing.assert_allclose(b[touched], SPR * a[touched], rtol=1e-9)
        assert dw.energy_deposited == dm.energy_deposited
        assert dw.scoring_mode == "dose_to_water"

    def test_kerma_mode_rejects_dose_to_water(self) -> None:
        grid = _water_grid()
        engine = _ref_engine(grid, AnalyticCrossSections(grid.max_density_by_material()))
        with pytest.raises(ValueError, match="dose_to_water"):
            engine.run(
                _source(),
                n_histories=4,
                n_batches=1,
                seed=SEED,
                transport_electrons=False,
                scoring_mode="dose_to_water",
            )

    def test_unknown_scoring_mode_is_rejected(self) -> None:
        grid = _water_grid()
        engine = _ref_engine(grid, AnalyticCrossSections(grid.max_density_by_material()))
        with pytest.raises(ValueError, match="scoring_mode"):
            engine.run(_source(), n_histories=4, n_batches=1, seed=SEED, scoring_mode="dose")
