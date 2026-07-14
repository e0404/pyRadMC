"""Scoring (dose) grid decoupled from the transport grid: geometry + mass rebin.

The dose grid is where energy is *accumulated*; transport stays on the fine CT grid.
Its per-voxel mass is the overlap-weighted sum of transport-voxel masses (separable
1D overlaps per axis), which is what defines dose-to-medium for a dose voxel that
overlays several CT voxels. Getting this mass map wrong scales dose voxel by voxel,
so it is pinned here against hand arithmetic and exact conservation identities.
"""

from __future__ import annotations

import numpy as np
import pytest


def _heterogeneous_grid(shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    from pyRadMC.geometry.grid import VoxelGrid

    rng = np.random.default_rng(20260714)
    density = rng.uniform(0.2, 2.5, size=shape)
    return VoxelGrid(
        shape=shape,
        spacing=spacing,
        origin=origin,
        density=density,
        material=np.zeros(shape, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Construction and geometry
# ---------------------------------------------------------------------------


def test_for_grid_is_the_transport_grid_byte_identical() -> None:
    """Default scoring grid: same geometry, mass exactly density * voxel_volume.

    The engines' scoring_grid=None path goes through this constructor, and the
    project promise is byte-identical behaviour to the pre-dose-grid engine — so
    the mass map must be the very same product the scorer used before, not a
    rebin that agrees only to rounding.
    """
    from pyRadMC.scoring.grid import ScoringGrid

    grid = _heterogeneous_grid()
    sg = ScoringGrid.for_grid(grid)

    assert sg.shape == grid.shape
    assert sg.spacing == grid.spacing
    assert sg.origin == grid.origin
    np.testing.assert_array_equal(sg.voxel_mass, grid.density * grid.voxel_volume)


def test_scoring_grid_validates_geometry() -> None:
    from pyRadMC.scoring.grid import ScoringGrid

    grid = _heterogeneous_grid()
    with pytest.raises(ValueError, match="shape"):
        ScoringGrid.rebin(grid, shape=(0, 1, 1), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="spacing"):
        ScoringGrid.rebin(grid, shape=(1, 1, 1), spacing=(1.0, -1.0, 1.0), origin=(0.0, 0.0, 0.0))


def test_half_open_convention_matches_transport_grid() -> None:
    """contains/voxel_index delegate to the shared geometry primitives."""
    from pyRadMC.scoring.grid import ScoringGrid

    grid = _heterogeneous_grid()
    sg = ScoringGrid.for_grid(grid)

    # Lower faces inside, upper faces outside — same as VoxelGrid (test-pinned there).
    assert sg.contains(0.0, 0.0, 0.0)
    assert not sg.contains(4.0, 2.0, 2.0)
    assert sg.voxel_index(0.0, 0.0, 0.0) == (0, 0, 0)
    assert sg.voxel_index(1.0, 2.5, 3.999) == (1, 2, 3)


# ---------------------------------------------------------------------------
# Mass map by voxel-overlap rebinning
# ---------------------------------------------------------------------------


def test_rebin_identity_geometry_matches_direct_product() -> None:
    """Rebinning onto the transport geometry itself reproduces density * volume."""
    from pyRadMC.scoring.grid import ScoringGrid

    grid = _heterogeneous_grid()
    sg = ScoringGrid.rebin(grid, shape=grid.shape, spacing=grid.spacing, origin=grid.origin)
    np.testing.assert_allclose(sg.voxel_mass, grid.density * grid.voxel_volume, rtol=1e-12)


def test_rebin_aligned_coarsening_sums_child_masses() -> None:
    """A 2x aligned coarsening: each dose voxel's mass is the sum of its 8 children."""
    from pyRadMC.scoring.grid import ScoringGrid

    grid = _heterogeneous_grid(shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))
    sg = ScoringGrid.rebin(grid, shape=(2, 2, 2), spacing=(2.0, 2.0, 2.0), origin=grid.origin)

    fine_mass = grid.density * grid.voxel_volume
    expected = fine_mass.reshape(2, 2, 2, 2, 2, 2).sum(axis=(1, 3, 5))
    np.testing.assert_allclose(sg.voxel_mass, expected, rtol=1e-12)


def test_rebin_non_aligned_covering_grid_conserves_total_mass() -> None:
    """A non-aligned, non-integer-ratio dose grid strictly covering the CT keeps
    the total mass: every CT voxel is fully overlapped by exactly one partition
    of dose voxels, so the overlap fractions sum to one per CT voxel."""
    from pyRadMC.scoring.grid import ScoringGrid

    grid = _heterogeneous_grid(shape=(5, 4, 3), spacing=(0.9, 1.1, 1.3), origin=(0.2, -0.3, 0.5))
    # Covers [-0.5, 5.42] x [-1.0, 4.92] x [0.0, 5.92] in x/y/z: a superset of the CT.
    sg = ScoringGrid.rebin(
        grid, shape=(8, 8, 8), spacing=(0.74, 0.74, 0.74), origin=(-0.5, -1.0, 0.0)
    )
    total_ct_mass = float(np.sum(grid.density)) * grid.voxel_volume
    assert float(sg.voxel_mass.sum()) == pytest.approx(total_ct_mass, rel=1e-12)


def test_rebin_subregion_mass_by_hand() -> None:
    """One offset dose voxel over a 1D density ramp: mass matches hand arithmetic."""
    from pyRadMC.geometry.grid import VoxelGrid
    from pyRadMC.scoring.grid import ScoringGrid

    density = np.array([1.0, 2.0, 4.0, 8.0]).reshape(4, 1, 1)
    grid = VoxelGrid(
        shape=(4, 1, 1),
        spacing=(1.0, 1.0, 1.0),
        density=density,
        material=np.zeros((4, 1, 1), dtype=np.int32),
    )
    # Dose voxel spans x in [0.25, 1.75]: 0.75 cm of rho=1 and 0.75 cm of rho=2.
    sg = ScoringGrid.rebin(grid, shape=(1, 1, 1), spacing=(1.5, 1.0, 1.0), origin=(0.25, 0.0, 0.0))
    assert float(sg.voxel_mass[0, 0, 0]) == pytest.approx(0.75 * 1.0 + 0.75 * 2.0, rel=1e-12)


def test_rebin_dose_voxel_outside_ct_has_zero_mass() -> None:
    """No CT-coverage requirement: an uncovered dose voxel simply has zero mass."""
    from pyRadMC.scoring.grid import ScoringGrid

    grid = _heterogeneous_grid(shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
    sg = ScoringGrid.rebin(grid, shape=(2, 1, 1), spacing=(2.0, 2.0, 2.0), origin=(0.0, 0.0, 0.0))
    # Second dose voxel spans x in [2, 4]: entirely outside the CT's [0, 2].
    assert float(sg.voxel_mass[1, 0, 0]) == 0.0
    assert float(sg.voxel_mass[0, 0, 0]) > 0.0
