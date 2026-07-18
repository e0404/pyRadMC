"""The device material map is uint8; a source beyond its range must refuse.

The Warp backend uploads the per-voxel material index as ``uint8`` (the registry
holds a handful of materials; 4x narrower random loads in the transport kernels'
hottest path). ``astype`` would *silently truncate* an index above 255 into a
different valid-looking material, so the engine must raise instead the moment a
source declares more materials than the dtype can address. Values in range are
untouched integers, so transport is bit-identical to the int32 upload — pinned
by the existing multi-material and cross-backend suites, not re-pinned here.
"""

from __future__ import annotations

import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletGridSource, ParallelBeamSource
from tests.conftest import SEED

pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyRadMC.backends.warp.engine import WarpEngine

pytestmark = pytest.mark.warp


class _ManyMaterialCrossSections(AnalyticCrossSections):
    """Declares more materials than uint8 can address; queries stay water-only."""

    @property
    def n_materials(self) -> int:
        return 300


def _engine() -> WarpEngine:
    grid = VoxelGrid.uniform_water(shape=(4, 4, 4), spacing=(0.5, 0.5, 0.5))
    xs = _ManyMaterialCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device="cpu")


def test_run_refuses_more_materials_than_uint8() -> None:
    source = ParallelBeamSource(energy=6.0, z=-1.0, x_range=(0.5, 1.5), y_range=(0.5, 1.5))
    with pytest.raises(ValueError, match="256 materials"):
        _engine().run(source, n_histories=8, n_batches=1, seed=SEED)


def test_run_dij_refuses_more_materials_than_uint8() -> None:
    source = BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(0.5, 1.5), y_range=(0.5, 1.5), n_x=1, n_y=1
    )
    with pytest.raises(ValueError, match="256 materials"):
        _engine().run_dij(source, n_histories_per_beamlet=8, n_batches=1, seed=SEED)
