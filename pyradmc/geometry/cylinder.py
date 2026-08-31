"""Cylindrical scoring bins: depth along z, radial shells about a beam axis.

This is a **scoring** geometry, not a transport geometry. Transport stays on the
rectilinear :class:`~pyradmc.geometry.grid.VoxelGrid` — Woodcock tracking, density
and material lookups are unchanged, and nothing here is ever consulted while a
particle is being moved (AGENTS.md 6: rectilinear voxel grids only). What these
functions do is decide *where a deposit is filed*, exactly as
:func:`~pyradmc.geometry.grid.point_axis_index` does for the rectilinear scoring
grid.

The geometry is the one a mono-energetic pencil-beam kernel is defined on (Mackie
et al., Med. Phys. 12(2):188-196, 1985, doi:10.1118/1.595774; Ahnesjo, Med. Phys.
16(4):577-592, 1989, doi:10.1118/1.596360): a narrow beam along +z entering a
homogeneous medium, with dose binned by depth and by radius about the beam axis.
Radial shells rather than lateral voxels because the distribution is
axially symmetric, so a shell averages over the whole azimuth and reaches a
useful per-bin sigma for a fraction of the histories a Cartesian grid needs.

As with the voxel grid, the point queries live in pure scalar module-level
functions (AGENTS.md 2.5) that the host container
(:class:`~pyradmc.scoring.cylinder.CylindricalScoringGrid`) delegates to and the
Warp kernels compile directly, so the half-open convention has exactly one
definition on every target. A convention drift between host and kernel would move
the steep near-axis gradient by a whole shell.

Shell edges are supplied as **squared** radii throughout, so no square root is
taken per deposit and the comparisons are exact in the same arithmetic the caller
built the edges in.

Lengths are in cm.
"""

from __future__ import annotations

from pyradmc.data.handles import Table1D

__all__ = [
    "cylinder_contains",
    "cylinder_radius_squared",
    "edge_bin_index",
]


def cylinder_radius_squared(x: float, y: float, axis_x: float, axis_y: float) -> float:
    """Squared perpendicular distance from the cylinder axis, in cm^2.

    The axis runs parallel to z through ``(axis_x, axis_y)``. Squared because every
    consumer compares against squared shell edges; taking the root per deposit
    would cost a transcendental and buy nothing.
    """
    dx = x - axis_x
    dy = y - axis_y
    return dx * dx + dy * dy


def cylinder_contains(
    x: float,
    y: float,
    z: float,
    axis_x: float,
    axis_y: float,
    z_lo: float,
    z_hi: float,
    r_inner_squared: float,
    r_outer_squared: float,
) -> bool:
    """Half-open containment: ``z_lo <= z < z_hi`` and ``r_in <= r < r_out``.

    Lower faces and the inner surface are inside; the exit plane and the outer
    surface are outside, matching :func:`~pyradmc.geometry.grid.point_inside` so
    that a deposit landing exactly on a boundary is claimed by exactly one bin and
    never by two or none.
    """
    if z < z_lo or z >= z_hi:
        return False
    r_squared = cylinder_radius_squared(x, y, axis_x, axis_y)
    return not (r_squared < r_inner_squared or r_squared >= r_outer_squared)


def edge_bin_index(value: float, edges: Table1D, n_bins: int) -> int:
    """Bin containing ``value``; the caller guarantees containment.

    ``edges`` holds ``n_bins + 1`` strictly increasing boundaries; the returned index
    ``k`` satisfies ``edges[k] <= value < edges[k + 1]``, i.e. the same
    lower-inclusive convention the voxel grid uses on every axis.

    Serves **both** cylindrical axes: the radial one searches squared radii against
    squared edges (no root per deposit), the depth one searches z directly. One
    primitive because the two axes differ only in what they measure.

    Binary search rather than an analytic formula because the edges are arbitrary
    by design: a pencil kernel wants geometric radial shells to resolve the
    near-axis gradient and fine depth bins only through the build-up region.
    Hard-coding a spacing law would make bin choice an accuracy-defining default
    instead of the caller's. At the bin counts this geometry is used with (tens to
    low hundreds), the search is a handful of comparisons on a resident array.

    The result is clamped to ``[0, n_bins - 1]``, mirroring
    :func:`~pyradmc.geometry.grid.point_axis_index`: a radius within a rounding ULP
    of the last edge can pass containment in float64 and land on ``n_bins`` in
    the float32 kernel, which would index out of bounds. Only such ULP-edge points
    are affected; the convention is unchanged for every interior point.

    The bounds are declared ``int(...)`` rather than as bare literals because Warp
    treats a literal-initialized name as a compile-time constant and refuses to
    mutate it inside a dynamic loop; the explicit constructor is what makes them
    dynamic variables on that target. On the host it is the same integer.
    """
    lo = int(0)  # noqa: UP018, RUF046  (see the docstring: Warp needs a dynamic variable)
    hi = n_bins - 1
    while lo < hi:
        # +1 biases the midpoint upward so the loop cannot stall at lo == hi - 1.
        mid = (lo + hi + 1) >> 1
        if value >= edges[mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo
