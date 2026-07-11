"""Rectilinear voxel grid.

The only geometry this engine will ever have (AGENTS.md section 6): axis-aligned
voxels, per-voxel mass density and material index. Woodcock (delta) tracking removes
the need for surface crossing logic; the transport loop only ever asks "which voxel am
I in and what is there" (see :mod:`pyRadMC.transport.photon`).

The point queries live in pure scalar module-level functions (AGENTS.md section 2.5)
that the :class:`VoxelGrid` methods delegate to and the Warp kernels compile directly,
so the half-open voxel convention and the slab-clipping edge cases have exactly one
definition. A convention drift between host and kernel geometry would move dose by one
voxel at every boundary — the single source is the guard.

Lengths are in cm, densities in g/cm^3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pyRadMC.data.materials import MATERIALS, WATER

__all__ = ["VoxelGrid", "point_axis_index", "point_inside", "slab_entry_distance"]


def point_inside(
    x: float,
    y: float,
    z: float,
    x_lo: float,
    y_lo: float,
    z_lo: float,
    x_hi: float,
    y_hi: float,
    z_hi: float,
) -> bool:
    """Half-open box containment: lower faces inside, upper faces outside.

    The convention is test-pinned (see :class:`VoxelGrid`); every containment
    decision on every target goes through this one comparison chain.
    """
    if x < x_lo or x >= x_hi:
        return False
    if y < y_lo or y >= y_hi:
        return False
    return not (z < z_lo or z >= z_hi)


def point_axis_index(position: float, origin: float, spacing: float) -> int:
    """Voxel index along one axis for a position the caller guarantees inside.

    ``floor`` (not ``int()``, which truncates toward zero) keeps the half-open
    convention correct for positions below the origin during the containment
    check race — identical semantics on the host and under Warp, where
    ``int(math.floor(...))`` compiles to a floor-then-cast.
    """
    # Host math.floor already returns int, but the Warp compile of this same
    # source gets a float32 floor; the cast is what makes both return an index.
    return int(math.floor((position - origin) / spacing))  # noqa: RUF046


def slab_entry_distance(
    x: float,
    y: float,
    z: float,
    ux: float,
    uy: float,
    uz: float,
    x_lo: float,
    y_lo: float,
    z_lo: float,
    x_hi: float,
    y_hi: float,
    z_hi: float,
) -> float:
    """Distance along (ux, uy, uz) from an *outside* point to the box; inf if missed.

    Standard axis-aligned slab clipping, unrolled per axis so the same source
    compiles under Warp (AGENTS.md 2.5). The accumulation order (x, then y, then z)
    is fixed; the :class:`VoxelGrid` method delegates here, so host and kernel
    cannot disagree about grazing rays.
    """
    t_near = 0.0
    t_far = math.inf

    if ux == 0.0:
        if x < x_lo or x >= x_hi:
            return math.inf
    else:
        t1 = (x_lo - x) / ux
        t2 = (x_hi - x) / ux
        if t1 > t2:
            t_swap = t1
            t1 = t2
            t2 = t_swap
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

    if uy == 0.0:
        if y < y_lo or y >= y_hi:
            return math.inf
    else:
        t1 = (y_lo - y) / uy
        t2 = (y_hi - y) / uy
        if t1 > t2:
            t_swap = t1
            t1 = t2
            t2 = t_swap
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

    if uz == 0.0:
        if z < z_lo or z >= z_hi:
            return math.inf
    else:
        t1 = (z_lo - z) / uz
        t2 = (z_hi - z) / uz
        if t1 > t2:
            t_swap = t1
            t1 = t2
            t2 = t_swap
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

    if t_near > t_far or t_far <= 0.0:
        return math.inf
    return t_near


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
    density: np.ndarray
    material: np.ndarray
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate array shapes, dtypes, and physical ranges."""
        if any(n < 1 for n in self.shape):
            raise ValueError(f"empty grid shape {self.shape}")
        if any(s <= 0.0 for s in self.spacing):
            raise ValueError(f"non-positive spacing {self.spacing}")
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
        """Build a homogeneous unit-density water grid, the Phase 0 workhorse phantom."""
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

    @property
    def upper_corner(self) -> tuple[float, float, float]:
        """Position of the upper outer corner, in cm (outside, half-open)."""
        return (
            self.origin[0] + self.shape[0] * self.spacing[0],
            self.origin[1] + self.shape[1] * self.spacing[1],
            self.origin[2] + self.shape[2] * self.spacing[2],
        )

    def contains(self, x: float, y: float, z: float) -> bool:
        """Whether the position lies inside the grid (upper faces excluded)."""
        hi = self.upper_corner
        return point_inside(x, y, z, *self.origin, *hi)

    def voxel_index(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Voxel containing the position; the caller guarantees ``contains``."""
        return (
            point_axis_index(x, self.origin[0], self.spacing[0]),
            point_axis_index(y, self.origin[1], self.spacing[1]),
            point_axis_index(z, self.origin[2], self.spacing[2]),
        )

    def distance_to_entry(
        self, x: float, y: float, z: float, ux: float, uy: float, uz: float
    ) -> float:
        """Distance along (ux, uy, uz) to the grid surface; 0 inside; inf if missed.

        The region outside the grid is vacuum, so a particle born outside flies this
        distance for free before Woodcock tracking starts. The grid is convex: a
        straight flight that leaves it never re-enters, so this is only ever needed
        once per particle. Clipping delegates to :func:`slab_entry_distance`.
        """
        if self.contains(x, y, z):
            return 0.0
        return slab_entry_distance(x, y, z, ux, uy, uz, *self.origin, *self.upper_corner)

    def max_density_by_material(self) -> tuple[tuple[int, float], ...]:
        """Collect the ``(material, max density)`` pairs, for the Woodcock majorant.

        Feeding anything less than the true per-material maximum into
        :meth:`pyRadMC.data.interface.CrossSectionSource.majorant` silently biases
        the transport; this method exists so callers never compute it by hand.
        """
        pairs = []
        for material in np.unique(self.material):
            rho_max = float(self.density[self.material == material].max())
            pairs.append((int(material), rho_max))
        return tuple(pairs)
