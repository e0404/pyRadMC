"""Heterogeneous-material transport plumbing .

A two-material synthetic table (water row plus an 'air' row at half the water
attenuation — a test instrument, not physics) through a slab phantom: the voxel
material index must reach the per-material table row on both backends. Real
multi-material physics is validated in ``tests/validation`` with the compiled EPICS
data; this fast-tier test pins the *plumbing* — grid material array to source row to
kernel lookup — with statistics loose enough for the integration tier.

Both slab voxels and water voxels carry unit density, isolating the material-index
path from the density-scaling path (which tests already pin).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.unit.test_tabulated_source import _two_material_tables

SHAPE = (8, 8, 16)
SPACING = (2.0, 2.0, 1.0)
SLAB = slice(5, 9)  # z-voxels holding material 1


def _slab_grid() -> VoxelGrid:
    material = np.full(SHAPE, 0, dtype=np.int32)
    material[:, :, SLAB] = 1
    return VoxelGrid(
        shape=SHAPE,
        spacing=SPACING,
        density=np.ones(SHAPE, dtype=np.float64),
        material=material,
    )


def _source_for(grid: VoxelGrid) -> TabulatedCrossSections:
    return TabulatedCrossSections(
        _two_material_tables(), geometry_densities=grid.max_density_by_material()
    )


def test_material_row_reaches_the_reference_transport() -> None:
    """KERMA in the weak slab sits far below the neighbouring water voxels.

    The slab's every channel is half the water value, so if the material index
    reaches the lookup, interaction density (and KERMA) drops inside it by roughly
    that factor; if the index were ignored (all rows read as water), the profile
    would be smooth through the slab. The 0.75 threshold sits many sigma from both.
    """
    grid = _slab_grid()
    engine = ReferenceEngine(grid=grid, cross_sections=_source_for(grid), rng=HostRNG())
    result = engine.run(
        ParallelBeamSource(energy=2.0, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0)),
        n_histories=4_000,
        n_batches=8,
        seed=SEED,
        transport_electrons=False,
    )
    pdd = result.dose.sum(axis=(0, 1))
    in_slab = pdd[SLAB].mean()
    upstream = pdd[3:5].mean()
    assert in_slab < 0.75 * upstream, f"slab KERMA {in_slab:.3e} vs upstream {upstream:.3e}"


@pytest.mark.warp
def test_heterogeneous_dose_agrees_across_backends() -> None:
    """Ref and Warp agree through the material slab (chi-squared, high-dose mask).

    Pins the whole multi-material kernel path: per-material table upload, the
    voxel-material lookup inside the kernel, and the Woodcock majorant over both
    materials.
    """
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _slab_grid()
    source = ParallelBeamSource(energy=2.0, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0))

    ref = ReferenceEngine(grid=grid, cross_sections=_source_for(grid), rng=HostRNG()).run(
        source, n_histories=6_000, n_batches=12, seed=SEED, transport_electrons=False
    )
    warp = WarpEngine(grid=grid, cross_sections=_source_for(grid), device="cpu").run(
        source, n_histories=24_000, n_batches=12, seed=SEED, transport_electrons=False
    )
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        warp.dose,
        warp.dose_sigma,
        warp.n_batches,
        mask=mask,
    )
