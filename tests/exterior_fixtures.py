"""Shared pieces of the exterior-air gates: a body in air, and its comparison statistics.

The phantom is what the approximation has to survive and what AGENTS.md section 8 asks
for: spectral, heterogeneous, tabulated. An elliptical tissue cylinder with two lungs and
an air margin in x and y, its long axis along z and running the full grid length (a CT
volume is cut through the patient, so no air at the z ends), 6 MV from a focal spot.

The statistics are replicate-level on purpose. Air and low-dose voxels are sparse, batched
chi-squared has no power there, and a ratio well below 1 is a red flag rather than a pass;
so doses are rebinned into blocks and each arm's uncertainty is the spread over seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from pyradmc.data.materials import AIR, LUNG, WATER
from pyradmc.geometry.grid import VoxelGrid

__all__ = ["BODY_AXES_CM", "BodyMasks", "block_sum", "body_grid", "body_masks"]

BODY_AXES_CM = (15.0, 10.0)
LUNG_DENSITY = 0.26
AIR_DENSITY = 1.205e-3
BODY_LENGTH_CM = 18.0
SHELL_VOXELS = 2


def body_grid(margin_cm: float, spacing: float = 0.3) -> VoxelGrid:
    """Elliptical tissue cylinder, two lungs, ``margin_cm`` of air around it in x and y."""
    a, b = BODY_AXES_CM
    nx = round((2.0 * (a + margin_cm)) / spacing)
    ny = round((2.0 * (b + margin_cm)) / spacing)
    nz = round(BODY_LENGTH_CM / spacing)
    x = (np.arange(nx) + 0.5) * spacing - 0.5 * nx * spacing
    y = (np.arange(ny) + 0.5) * spacing - 0.5 * ny * spacing
    xg, yg = np.meshgrid(x, y, indexing="ij")
    density2d = np.full((nx, ny), AIR_DENSITY)
    material2d = np.full((nx, ny), AIR, dtype=np.int32)
    body = (xg / a) ** 2 + (yg / b) ** 2 <= 1.0
    density2d[body] = 1.0
    material2d[body] = WATER
    for side in (-1.0, 1.0):
        lung = ((xg - side * 7.0) / 4.5) ** 2 + ((yg + 1.0) / 6.0) ** 2 <= 1.0
        density2d[lung] = LUNG_DENSITY
        material2d[lung] = LUNG
    shape = (nx, ny, nz)
    return VoxelGrid(
        shape=shape,
        spacing=(spacing, spacing, spacing),
        density=np.broadcast_to(density2d[:, :, None], shape).copy(),
        material=np.broadcast_to(material2d[:, :, None], shape).copy(),
    )


@dataclass(frozen=True)
class BodyMasks:
    """Voxel masks of the body: all of it, and its outer shell in the x-y plane."""

    body: npt.NDArray[np.bool_]
    shell: npt.NDArray[np.bool_]


def body_masks(grid: VoxelGrid) -> BodyMasks:
    """Body voxels and the outer ``SHELL_VOXELS`` layers next to the air.

    Eroded in the x-y plane only: the z ends are cut faces of the volume, not a surface
    that faces air.
    """
    body = np.asarray(grid.material) != AIR
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[:, 1, 1] = True
    structure[1, :, 1] = True
    inner = ndimage.binary_erosion(body, structure, iterations=SHELL_VOXELS, border_value=1)
    return BodyMasks(body=body, shell=body & ~inner)


def block_sum(values: npt.NDArray[np.float64], block: int) -> npt.NDArray[np.float64]:
    """Sum over cubes of ``block`` voxels a side, dropping a partial layer at the far end."""
    nx, ny, nz = values.shape
    trimmed = values[: nx - nx % block, : ny - ny % block, : nz - nz % block]
    mx, my, mz = (n // block for n in trimmed.shape)
    return trimmed.reshape(mx, block, my, block, mz, block).sum(axis=(1, 3, 5))
