"""Shared phantom, source and weights of the Warp beamlet-contraction suites.

``tests/dij/test_beamlet_contraction_warp.py`` and
``tests/dij/test_contraction_deposit_fold_warp.py`` pin the same contraction from two
sides, so every number in them depends on the same 10 x 10 x 8 water phantom, the same
3 x 3 beamlet grid and the same signed weights. They live here once so that changing one
changes both.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource

__all__ = ["engine", "phantom_grid", "phantom_xs", "source", "weights"]


def phantom_grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(10, 10, 8), spacing=(1.0, 1.0, 1.0))


def phantom_xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def source() -> BeamletGridSource:
    return BeamletGridSource(2.0, -1.0, (2.0, 8.0), (2.0, 8.0), 3, 3)


def weights(shape: tuple[int, int, int]) -> np.ndarray:
    """Signed, so the contraction cancels in part, as an objective gradient does."""
    x, y, z = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return np.sin(0.7 * x) + 0.3 * y - 0.1 * z


def engine(device: str, grid: VoxelGrid | None = None, xs=None):
    """A ``WarpEngine`` on ``device`` over ``grid`` (default: the phantom).

    Skips when ``device`` is CUDA and none is available. ``xs`` defaults to the analytic
    cross sections of ``grid``.
    """
    wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp import engine as warp_engine

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    grid = grid or phantom_grid()
    xs = xs or phantom_xs(grid)
    return warp_engine.WarpEngine(grid=grid, cross_sections=xs, device=device)
