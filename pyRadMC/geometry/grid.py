"""Rectilinear voxel grid.

The only geometry this engine will ever have (AGENTS.md section 6): axis-aligned
voxels, per-voxel mass density and material index. Woodcock (delta) tracking removes
the need for surface crossing logic; the transport loop only ever asks "which voxel am
I in and what is there" (see :mod:`pyRadMC.transport.photon`).

Lengths are in cm, densities in g/cm^3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pyRadMC.data.materials import MATERIALS, WATER

__all__ = ["VoxelGrid"]


@dataclass(frozen=True)
class VoxelGrid:
    """Axis-aligned voxel grid with per-voxel density and material.

    Voxels are half-open boxes: a position on an internal boundary belongs to the
    voxel on the upper side, and positions on the upper outer faces are outside. This
    convention is test-pinned; changing it moves dose by one voxel at every boundary.

    Attributes
    ----------
    shape
        Number of voxels along (x, y, z).
    spacing
        Voxel edge lengths in cm.
    origin
        Position of the lower corner of voxel (0, 0, 0), in cm.
    density
        Mass density per voxel in g/cm^3, shape ``shape``.
    material
        Material index per voxel (see :mod:`pyRadMC.data.materials`), shape ``shape``.
    """

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    density: np.ndarray = field(default=None)  # type: ignore[assignment]
    material: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Validate array shapes, dtypes, and physical ranges."""
        if any(n < 1 for n in self.shape):
            raise ValueError(f"empty grid shape {self.shape}")
        if any(s <= 0.0 for s in self.spacing):
            raise ValueError(f"non-positive spacing {self.spacing}")
        if self.density is None or self.material is None:
            raise ValueError("density and material arrays are required")
        if self.density.shape != self.shape:
            raise ValueError(f"density shape {self.density.shape} != grid shape {self.shape}")
        if self.material.shape != self.shape:
            raise ValueError(f"material shape {self.material.shape} != grid shape {self.shape}")
        if np.any(self.density <= 0.0):
            raise ValueError("non-positive voxel density; vacuum is not supported in Phase 0")
        if np.any((self.material < 0) | (self.material >= len(MATERIALS))):
            raise ValueError("material index outside the registry")

    @classmethod
    def uniform_water(
        cls,
        shape: tuple[int, int, int],
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> VoxelGrid:
        """A homogeneous unit-density water grid, the Phase 0 workhorse phantom."""
        return cls(
            shape=shape,
            spacing=spacing,
            origin=origin,
            density=np.ones(shape, dtype=np.float64),
            material=np.full(shape, WATER, dtype=np.int32),
        )

    @property
    def voxel_volume(self) -> float:
        """Volume of one voxel in cm^3."""
        return self.spacing[0] * self.spacing[1] * self.spacing[2]

    def contains(self, x: float, y: float, z: float) -> bool:
        """Whether the position lies inside the grid (upper faces excluded)."""
        return (
            self.origin[0] <= x < self.origin[0] + self.shape[0] * self.spacing[0]
            and self.origin[1] <= y < self.origin[1] + self.shape[1] * self.spacing[1]
            and self.origin[2] <= z < self.origin[2] + self.shape[2] * self.spacing[2]
        )

    def voxel_index(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Voxel containing the position; the caller guarantees ``contains``.

        ``floor`` (not ``int()``, which truncates toward zero) keeps the half-open
        convention correct for positions below the origin during the containment
        check race — and is what the Warp kernels will compile to.
        """
        return (
            int(math.floor((x - self.origin[0]) / self.spacing[0])),
            int(math.floor((y - self.origin[1]) / self.spacing[1])),
            int(math.floor((z - self.origin[2]) / self.spacing[2])),
        )

    def max_density_by_material(self) -> tuple[tuple[int, float], ...]:
        """The ``(material, max density)`` pairs present, for the Woodcock majorant.

        Feeding anything less than the true per-material maximum into
        :meth:`pyRadMC.data.interface.CrossSectionSource.majorant` silently biases
        the transport; this method exists so callers never compute it by hand.
        """
        pairs = []
        for material in np.unique(self.material):
            rho_max = float(self.density[self.material == material].max())
            pairs.append((int(material), rho_max))
        return tuple(pairs)
