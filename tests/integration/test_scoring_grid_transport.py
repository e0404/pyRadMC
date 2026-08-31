"""Decoupled scoring grid through the reference engine: exact identities.

Transport is unchanged by the scoring grid — the RNG streams never see it — so
every property here is exact, never statistical:

- the default (``scoring_grid=None``) is bit-identical to scoring on the
  transport grid explicitly;
- an aligned integer-ratio coarsening receives exactly the summed child energy,
  so its dose is the mass-weighted mean of the fine dose;
- a subregion grid books the missing deposits in the unscored ledger bucket and
  its edge voxels match the covering grid's same voxels (no clamping);
- the three-bucket ledger emitted == deposited + unscored + escaped closes to
  float64 accumulation precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, ParallelBeamSource
from pyradmc.rng.host import HostRNG
from pyradmc.scoring.grid import ScoringGrid
from tests.conftest import SEED

ENERGY = 6.0
Z0 = -1.0


def _grid() -> VoxelGrid:
    """Heterogeneous water phantom: mass rebinning must see nonuniform density."""
    shape = (8, 8, 16)
    rng = np.random.default_rng(20260714)
    density = rng.uniform(0.5, 1.5, size=shape)
    from pyradmc.data.materials import WATER

    return VoxelGrid(
        shape=shape,
        spacing=(2.0, 2.0, 0.5),
        density=density,
        material=np.full(shape, WATER, dtype=np.int32),
    )


def _engine() -> ReferenceEngine:
    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())


def _source() -> ParallelBeamSource:
    return ParallelBeamSource(energy=ENERGY, z=Z0, x_range=(2.0, 14.0), y_range=(2.0, 14.0))


def _run(engine: ReferenceEngine, scoring_grid: ScoringGrid | None, n: int = 400):
    return engine.run(_source(), n_histories=n, n_batches=4, seed=SEED, scoring_grid=scoring_grid)


def test_default_is_bit_identical_to_explicit_transport_grid_scoring() -> None:
    engine = _engine()
    a = _run(engine, None)
    b = _run(engine, ScoringGrid.for_grid(engine.grid))
    np.testing.assert_array_equal(a.dose, b.dose)
    np.testing.assert_array_equal(a.dose_sigma, b.dose_sigma)
    assert a.energy_deposited == b.energy_deposited
    assert a.energy_escaped == b.energy_escaped
    assert a.energy_unscored == 0.0
    assert b.energy_unscored == 0.0


def test_aligned_coarsening_is_the_exact_energy_rebin_of_the_fine_run() -> None:
    """2x2x2 coarse scoring: same deposits, same batches — the coarse dose is the
    mass-weighted mean of the fine dose, exactly (transport untouched)."""
    engine = _engine()
    fine = _run(engine, None)
    coarse_sg = ScoringGrid.rebin(
        engine.grid, shape=(4, 4, 8), spacing=(4.0, 4.0, 1.0), origin=engine.grid.origin
    )
    coarse = _run(engine, coarse_sg)

    # Transport does not see the scoring grid: identical streams, identical escape.
    assert coarse.energy_escaped == fine.energy_escaped
    assert coarse.energy_emitted == fine.energy_emitted
    assert coarse.energy_unscored == 0.0
    assert coarse.energy_deposited == pytest.approx(fine.energy_deposited, rel=1e-12)

    fine_mass = engine.grid.density * engine.grid.voxel_volume
    fine_energy = fine.dose * fine_mass  # per-history energy per fine voxel
    child = fine_energy.reshape(4, 2, 4, 2, 8, 2).sum(axis=(1, 3, 5))
    child_mass = fine_mass.reshape(4, 2, 4, 2, 8, 2).sum(axis=(1, 3, 5))
    np.testing.assert_allclose(coarse.dose, child / child_mass, rtol=1e-12)

    # The energy ledger closes on three buckets, exactly.
    assert coarse.energy_deposited + coarse.energy_unscored + coarse.energy_escaped == (
        pytest.approx(coarse.energy_emitted, rel=1e-12)
    )


def test_subregion_scoring_grid_books_unscored_and_never_clamps() -> None:
    """A dose grid covering only the entrance half of the phantom: energy landing
    deeper is unscored (not clamped into the last slab), the ledger closes, and
    the covered voxels are bit-identical to the covering run's same voxels."""
    engine = _engine()
    full = _run(engine, None)
    # Integer-exact subregion: first 8 of 16 z-slabs (z in [0, 4)), full x-y.
    sub_sg = ScoringGrid.rebin(
        engine.grid, shape=(8, 8, 8), spacing=(2.0, 2.0, 0.5), origin=(0.0, 0.0, 0.0)
    )
    sub = _run(engine, sub_sg)

    assert sub.energy_unscored > 0.0
    assert sub.energy_deposited + sub.energy_unscored + sub.energy_escaped == pytest.approx(
        sub.energy_emitted, rel=1e-12
    )
    assert sub.energy_escaped == full.energy_escaped
    assert sub.energy_deposited + sub.energy_unscored == pytest.approx(
        full.energy_deposited, rel=1e-12
    )
    # No clamping: the subregion's voxels — its z-edge slab included — carry
    # exactly the covering run's dose in the same region.
    np.testing.assert_allclose(sub.dose, full.dose[:, :, :8], rtol=1e-12)
    np.testing.assert_allclose(sub.dose_sigma, full.dose_sigma[:, :, :8], rtol=1e-12)


def test_dij_on_a_coarse_subregion_scoring_grid() -> None:
    """run_dij: the Dij columns live on the scoring grid (its shape, its voxel
    count), the unscored bucket closes the ledger, and a 1x1 lattice keeps the
    open-field anchor bit for bit on the same scoring grid."""
    engine = _engine()
    sg = ScoringGrid.rebin(
        engine.grid, shape=(4, 4, 4), spacing=(4.0, 4.0, 1.0), origin=(0.0, 0.0, 0.0)
    )
    lattice = BeamletGridSource(
        energy=ENERGY, z=Z0, x_range=(2.0, 14.0), y_range=(2.0, 14.0), n_x=1, n_y=1
    )
    dij = engine.run_dij(
        lattice,
        n_histories_per_beamlet=400,
        n_batches=4,
        seed=SEED,
        truncation=0.0,
        scoring_grid=sg,
    )
    run = engine.run(_source(), n_histories=400, n_batches=4, seed=SEED, scoring_grid=sg)

    assert dij.grid_shape == sg.shape
    assert dij.n_voxels == sg.n_voxels
    np.testing.assert_array_equal(dij.column_dense(0), run.dose)
    np.testing.assert_array_equal(dij.sigma_dense(0), run.dose_sigma)
    assert dij.energy_unscored == run.energy_unscored
    assert dij.energy_unscored > 0.0  # the 4-slab grid covers only z in [0, 4)
    assert dij.energy_deposited + dij.energy_unscored + dij.energy_escaped == pytest.approx(
        dij.energy_emitted, rel=1e-12
    )
