"""Scoring (dose) grid, decoupled from the transport grid.

Transport — Woodcock tracking, stepping, density and material lookups — always runs
on the fine transport (CT) :class:`~pyRadMC.geometry.grid.VoxelGrid`; only dose
*accumulation* happens on the grid defined here. Scoring directly onto a coarser
grid shrinks the Dij and the per-group device buffers in proportion to the voxel
count and improves per-voxel statistics; scoring fine and resampling afterwards
would pay the fine-grid memory and gain nothing statistically.

A scoring grid may have any origin, spacing and shape — including a subregion of
the transport grid. There is deliberately **no coverage requirement in either
direction**: a deposit inside the transport grid but outside the scoring grid goes
to an *unscored* energy ledger bucket (never clamped into an edge voxel, which
would corrupt edge dose), and a scoring voxel not covered by the transport grid
simply has zero mass and can receive no energy.

The per-voxel mass is rebinned from the transport grid by exact voxel overlap:
``mass_J = sum_i rho_i * V(overlap of transport voxel i with scoring voxel J)``,
computed as a separable product of 1D overlap lengths per axis, so non-aligned and
non-integer-ratio grids are exact. This mass also *defines* dose-to-medium for a
scoring voxel overlaying several transport voxels: total energy over total mass.

Geometry queries delegate to the same scalar primitives as the transport grid
(:func:`~pyRadMC.geometry.grid.point_inside`,
:func:`~pyRadMC.geometry.grid.point_axis_index`), so the half-open voxel
convention has exactly one definition on every target.

Lengths are in cm, masses in g.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyRadMC.geometry.grid import VoxelGrid, point_axis_index, point_inside

__all__ = ["ScoringGrid"]


def _axis_overlap_lengths(
    scoring_origin: float,
    scoring_spacing: float,
    scoring_n: int,
    transport_origin: float,
    transport_spacing: float,
    transport_n: int,
) -> np.ndarray:
    """Overlap length in cm between every scoring and transport voxel pair, 1D.

    Returns a ``(scoring_n, transport_n)`` matrix; entry ``(a, i)`` is the length of
    the intersection of scoring interval ``a`` with transport interval ``i``. The 3D
    overlap volume is the product of the three per-axis lengths (the intervals are
    axis-aligned boxes), which is what makes the rebin separable and exact.
    """
    s_lo = scoring_origin + scoring_spacing * np.arange(scoring_n, dtype=np.float64)
    s_hi = s_lo + scoring_spacing
    t_lo = transport_origin + transport_spacing * np.arange(transport_n, dtype=np.float64)
    t_hi = t_lo + transport_spacing
    overlap = np.minimum(s_hi[:, None], t_hi[None, :]) - np.maximum(s_lo[:, None], t_lo[None, :])
    result: np.ndarray = np.maximum(overlap, 0.0)
    return result


@dataclass(frozen=True)
class ScoringGrid:
    """Axis-aligned dose-scoring grid with per-voxel mass from the transport grid.

    Construct through :meth:`for_grid` (score on the transport grid itself — the
    engines' default, byte-identical to scoring without a separate dose grid) or
    :meth:`rebin` (arbitrary geometry, mass by exact voxel overlap). The mass map
    is only meaningful for the transport grid it was built from; hand the engine
    a scoring grid built from the same :class:`~pyRadMC.geometry.grid.VoxelGrid`
    it transports on.

    Voxels follow the transport grid's half-open convention (lower faces inside,
    upper faces outside), delegated to the shared geometry primitives.

    Attributes
    ----------
    shape
        Number of scoring voxels along (x, y, z).
    spacing
        Scoring voxel edge lengths in cm.
    origin
        Position of the lower corner of scoring voxel (0, 0, 0), in cm.
    voxel_mass
        Mass per scoring voxel in g, shape ``shape``. Zero where the transport
        grid does not cover the scoring voxel (dose is reported as zero there —
        no energy can arrive without transport-grid coverage).
    """

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    voxel_mass: np.ndarray

    def __post_init__(self) -> None:
        """Validate geometry and mass-map consistency."""
        if any(n < 1 for n in self.shape):
            raise ValueError(f"empty scoring grid shape {self.shape}")
        if any(s <= 0.0 for s in self.spacing):
            raise ValueError(f"non-positive scoring spacing {self.spacing}")
        if self.voxel_mass.shape != self.shape:
            raise ValueError(
                f"voxel_mass shape {self.voxel_mass.shape} != scoring grid shape {self.shape}"
            )
        if np.any(self.voxel_mass < 0.0):
            raise ValueError("negative scoring voxel mass")

    @classmethod
    def for_grid(cls, grid: VoxelGrid) -> ScoringGrid:
        """Score on the transport grid itself: same geometry, mass = density * volume.

        This is the engines' ``scoring_grid=None`` default. The mass is the direct
        per-voxel product — the identical arithmetic the scorer used before scoring
        grids existed — not an overlap rebin, so the default path stays
        byte-identical, not merely equal to rounding.
        """
        return cls(
            shape=grid.shape,
            spacing=grid.spacing,
            origin=grid.origin,
            voxel_mass=grid.density * grid.voxel_volume,
        )

    @classmethod
    def rebin(
        cls,
        grid: VoxelGrid,
        shape: tuple[int, int, int],
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float],
    ) -> ScoringGrid:
        """Build a scoring grid of arbitrary geometry, mass by exact voxel overlap.

        ``mass_J = sum_i rho_i * prod_axis overlap_1d`` over transport voxels ``i``:
        exact for non-aligned and non-integer-ratio grids, and it degrades gracefully
        at the edges — a partially covered scoring voxel carries the mass of the
        covered part only, which is also the only part deposits can arrive in.
        """
        if any(n < 1 for n in shape):
            raise ValueError(f"empty scoring grid shape {shape}")
        if any(s <= 0.0 for s in spacing):
            raise ValueError(f"non-positive scoring spacing {spacing}")
        wx = _axis_overlap_lengths(
            origin[0], spacing[0], shape[0], grid.origin[0], grid.spacing[0], grid.shape[0]
        )
        wy = _axis_overlap_lengths(
            origin[1], spacing[1], shape[1], grid.origin[1], grid.spacing[1], grid.shape[1]
        )
        wz = _axis_overlap_lengths(
            origin[2], spacing[2], shape[2], grid.origin[2], grid.spacing[2], grid.shape[2]
        )
        # Separable contraction: mass_abc = sum_ijk rho_ijk * wx_ai * wy_bj * wz_ck.
        partial_x = np.tensordot(wx, grid.density, axes=(1, 0))  # (a, j, k)
        partial_xy = np.einsum("bj,ajk->abk", wy, partial_x)  # (a, b, k)
        mass = np.einsum("ck,abk->abc", wz, partial_xy)  # (a, b, c)
        return cls(shape=shape, spacing=spacing, origin=origin, voxel_mass=mass)

    @property
    def n_voxels(self) -> int:
        """Number of scoring voxels."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def upper_corner(self) -> tuple[float, float, float]:
        """Position of the upper outer corner, in cm (outside, half-open)."""
        return (
            self.origin[0] + self.shape[0] * self.spacing[0],
            self.origin[1] + self.shape[1] * self.spacing[1],
            self.origin[2] + self.shape[2] * self.spacing[2],
        )

    def contains(self, x: float, y: float, z: float) -> bool:
        """Whether the position lies inside the scoring grid (upper faces excluded)."""
        hi = self.upper_corner
        return point_inside(x, y, z, *self.origin, *hi)

    def voxel_index(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Scoring voxel containing the position; the caller guarantees ``contains``."""
        return (
            point_axis_index(x, self.origin[0], self.spacing[0], self.shape[0]),
            point_axis_index(y, self.origin[1], self.spacing[1], self.shape[1]),
            point_axis_index(z, self.origin[2], self.spacing[2], self.shape[2]),
        )
