"""Cylindrical (depth x radial shell) scoring geometry: bins, masses, routing.

A pencil-beam kernel is read straight off this geometry, so every property here
scales the answer directly and silently:

- a shell mass wrong by the annulus formula scales that shell's dose;
- a radial bin edge off by one moves the steep near-axis gradient a whole bin;
- a containment convention that disagrees with the transport grid double-counts or
  drops the deposits sitting on a boundary.

The scalar primitives are pinned separately from the host container because the Warp
kernels compile *those functions*, not the container (see
:mod:`pyRadMC.backends.warp.physics`); a convention that lived only in the dataclass
would not be the one running on the device.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyRadMC.data.materials import WATER
from pyRadMC.geometry.cylinder import (
    cylinder_contains,
    cylinder_radius_squared,
    edge_bin_index,
)
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.scoring.cylinder import (
    CylindricalScoringGrid,
    common_bin_divisor,
    geometric_edges,
    graded_edges,
    uniform_edges,
)

EDGES = np.array([0.0, 0.5, 1.0, 2.0, 4.0])


def _cylinder(**kwargs) -> CylindricalScoringGrid:
    """Unit-density water cylinder: 4 depth bins of 1 cm, the EDGES shells."""
    defaults = {
        "density": 1.0,
        "axis": (0.0, 0.0),
        "depth_edges": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "radial_edges": EDGES,
    }
    return CylindricalScoringGrid.uniform(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# Scalar primitives (these are what the kernels compile)
# ---------------------------------------------------------------------------


def test_radius_squared_is_measured_from_the_axis() -> None:
    assert cylinder_radius_squared(3.0, 4.0, 0.0, 0.0) == pytest.approx(25.0)
    assert cylinder_radius_squared(3.0, 4.0, 3.0, 4.0) == pytest.approx(0.0)
    assert cylinder_radius_squared(1.0, 1.0, -1.0, -1.0) == pytest.approx(8.0)


def test_bin_index_is_half_open_upward() -> None:
    """``r_in <= r < r_out``: the same lower-inclusive convention as the voxel grid.

    A point exactly on a shell boundary belongs to the *outer* shell, so two
    adjacent shells can never both claim it and none can drop it.
    """
    edges_sq = EDGES**2
    n = len(EDGES) - 1
    assert edge_bin_index(0.0, edges_sq, n) == 0
    assert edge_bin_index(0.49**2, edges_sq, n) == 0
    assert edge_bin_index(0.5**2, edges_sq, n) == 1  # on the edge -> outward
    assert edge_bin_index(0.99**2, edges_sq, n) == 1
    assert edge_bin_index(1.0**2, edges_sq, n) == 2
    assert edge_bin_index(3.999**2, edges_sq, n) == 3


def test_bin_index_clamps_like_point_axis_index() -> None:
    """A radius within a rounding ULP of the outer edge stays in the last shell.

    Mirrors :func:`~pyRadMC.geometry.grid.point_axis_index`: containment is the
    caller's decision, and the index lookup must never return an out-of-range bin
    for a point that has already passed it.
    """
    edges_sq = EDGES**2
    n = len(EDGES) - 1
    assert edge_bin_index(1.0e30, edges_sq, n) == n - 1
    assert edge_bin_index(-1.0, edges_sq, n) == 0


def test_bin_index_agrees_with_searchsorted_everywhere() -> None:
    """The in-kernel binary search is NumPy searchsorted over the clamped range.

    Property test over irregular edges: the hand-rolled search exists only because
    a kernel cannot call NumPy, so it must agree bin for bin with the obvious host
    implementation, not merely on the cases someone thought to enumerate.
    """
    rng = np.random.default_rng(20260714)
    edges = np.sort(rng.uniform(0.0, 6.0, size=12))
    edges[0] = 0.0
    edges_sq = edges**2
    n = len(edges) - 1
    for r in rng.uniform(0.0, edges[-1], size=500):
        expected = int(np.clip(np.searchsorted(edges_sq, r * r, side="right") - 1, 0, n - 1))
        assert edge_bin_index(r * r, edges_sq, n) == expected


def test_contains_is_half_open_in_depth_and_radius() -> None:
    args = (0.0, 0.0, 0.0, 4.0, 0.0, 16.0)  # axis (0,0), z in [0,4), r^2 in [0,16)
    assert cylinder_contains(0.0, 0.0, 0.0, *args)  # lower depth face is inside
    assert not cylinder_contains(0.0, 0.0, 4.0, *args)  # upper depth face is outside
    assert not cylinder_contains(0.0, 0.0, -1.0e-9, *args)
    assert cylinder_contains(3.999, 0.0, 2.0, *args)
    assert not cylinder_contains(4.0, 0.0, 2.0, *args)  # outer surface is outside


# ---------------------------------------------------------------------------
# Shell mass: the annulus formula, exactly
# ---------------------------------------------------------------------------


def test_shell_mass_is_the_annulus_volume_times_density() -> None:
    """``m = rho * pi * (r_out^2 - r_in^2) * dz``, per shell, identical per depth."""
    cyl = _cylinder(density=1.25)
    expected = 1.25 * math.pi * (EDGES[1:] ** 2 - EDGES[:-1] ** 2) * 1.0
    assert cyl.voxel_mass.shape == (4, len(EDGES) - 1)
    for iz in range(4):
        np.testing.assert_allclose(cyl.voxel_mass[iz], expected, rtol=1e-14)


def test_shell_masses_sum_to_the_full_cylinder_mass() -> None:
    """Conservation: no volume is lost or double-counted between the shells."""
    cyl = _cylinder(density=0.9)
    total = 0.9 * math.pi * EDGES[-1] ** 2 * (4 * 1.0)
    assert float(cyl.voxel_mass.sum()) == pytest.approx(total, rel=1e-14)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_radial_edges_must_be_increasing_and_non_negative() -> None:
    with pytest.raises(ValueError, match="increasing"):
        _cylinder(radial_edges=np.array([0.0, 1.0, 0.5]))
    with pytest.raises(ValueError, match="negative"):
        _cylinder(radial_edges=np.array([-1.0, 1.0]))
    with pytest.raises(ValueError, match="at least one shell"):
        _cylinder(radial_edges=np.array([1.0]))


def test_depth_binning_is_validated() -> None:
    with pytest.raises(ValueError, match="at least one depth bin"):
        _cylinder(depth_edges=np.array([1.0]))
    with pytest.raises(ValueError, match="increasing"):
        _cylinder(depth_edges=np.array([0.0, 2.0, 1.0]))


# ---------------------------------------------------------------------------
# Position routing
# ---------------------------------------------------------------------------


def test_flat_index_is_c_order_depth_major() -> None:
    """``flat = iz * n_shells + ir`` — the order ``voxel_mass.reshape(-1)`` implies.

    The device scores into a flat buffer sized ``n_voxels`` and the host reshapes it
    to ``shape``; if this index disagreed with C order the kernel would come back
    transposed on the GPU and correct on the reference.
    """
    cyl = _cylinder()
    n_shells = len(EDGES) - 1
    for iz, ir, x, y, z in [(0, 0, 0.0, 0.0, 0.5), (2, 3, 3.0, 0.0, 2.5), (3, 1, 0.0, 0.75, 3.5)]:
        assert cyl.voxel_index(x, y, z) == (iz, ir)
        assert cyl.flat_index(x, y, z) == iz * n_shells + ir


def test_flat_index_reports_minus_one_outside() -> None:
    """Outside is signalled in band, so the kernel needs no second containment call."""
    cyl = _cylinder()
    assert cyl.flat_index(0.0, 0.0, -0.1) == -1  # before the entrance plane
    assert cyl.flat_index(0.0, 0.0, 4.0) == -1  # past the exit plane
    assert cyl.flat_index(4.0, 0.0, 2.0) == -1  # outside the outer shell


def test_off_axis_cylinder_bins_about_its_own_axis() -> None:
    cyl = _cylinder(axis=(8.0, 8.0))
    assert cyl.voxel_index(8.0, 8.0, 0.5) == (0, 0)
    assert cyl.voxel_index(8.0 + 1.5, 8.0, 0.5) == (0, 2)
    assert cyl.flat_index(0.0, 0.0, 0.5) == -1


def test_contains_matches_flat_index() -> None:
    """Two entry points, one convention: ``contains`` iff ``flat_index >= 0``."""
    cyl = _cylinder(axis=(1.0, -1.0))
    rng = np.random.default_rng(20260715)
    for x, y, z in rng.uniform(-6.0, 6.0, size=(400, 3)):
        assert cyl.contains(x, y, z) == (cyl.flat_index(x, y, z) >= 0)


# ---------------------------------------------------------------------------
# Building from a transport grid
# ---------------------------------------------------------------------------


def test_for_grid_takes_density_from_the_phantom_and_centres_on_it() -> None:
    grid = VoxelGrid.uniform_water(shape=(16, 16, 8), spacing=(1.0, 1.0, 1.0))
    edges = uniform_edges(4.0, 4)
    cyl = CylindricalScoringGrid.for_grid(grid, radial_edges=edges)
    assert cyl.axis == (8.0, 8.0)  # the transport grid's lateral centre
    assert (cyl.depth_origin, cyl.depth_upper, cyl.n_depth) == (0.0, 8.0, 8)
    expected = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    np.testing.assert_allclose(cyl.voxel_mass[0], expected, rtol=1e-14)


def test_for_grid_refuses_a_cylinder_the_phantom_does_not_contain() -> None:
    """Claiming mass outside the phantom divides real energy by fictitious grams."""
    grid = VoxelGrid.uniform_water(shape=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="outside the transport grid"):
        CylindricalScoringGrid.for_grid(grid, radial_edges=uniform_edges(6.0, 3))


def test_for_grid_refuses_a_heterogeneous_medium() -> None:
    """The analytic annulus mass is exact only in a uniform medium; say so, loudly.

    An overlap rebin cannot be exact here — an annulus is not separable over
    Cartesian axes — so rather than ship a silent approximation the constructor
    refuses what it cannot compute and points at the rectilinear scoring grid.
    """
    density = np.ones((8, 8, 8))
    density[4, 4, 4] = 1.6
    grid = VoxelGrid(
        shape=(8, 8, 8),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        density=density,
        material=np.full((8, 8, 8), WATER, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="uniform"):
        CylindricalScoringGrid.for_grid(grid, radial_edges=uniform_edges(3.0, 3))


def test_for_grid_ignores_heterogeneity_outside_the_cylinder() -> None:
    """Only the medium the shells actually overlay has to be uniform."""
    density = np.ones((16, 16, 8))
    density[0, 0, :] = 3.0  # a far corner, well outside a 2 cm cylinder
    grid = VoxelGrid(
        shape=(16, 16, 8),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        density=density,
        material=np.full((16, 16, 8), WATER, dtype=np.int32),
    )
    cyl = CylindricalScoringGrid.for_grid(grid, radial_edges=uniform_edges(2.0, 2))
    assert float(cyl.voxel_mass.sum()) == pytest.approx(math.pi * 4.0 * 8.0, rel=1e-14)


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------


def test_uniform_edges_span_zero_to_r_max() -> None:
    e = uniform_edges(5.0, 10)
    assert e.shape == (11,)
    assert e[0] == 0.0
    assert e[-1] == pytest.approx(5.0)
    np.testing.assert_allclose(np.diff(e), 0.5)


def test_geometric_edges_resolve_the_core_and_reach_r_max() -> None:
    """A pencil kernel's gradient is near the axis; geometric edges put bins there.

    The innermost bin is the disc ``[0, r_min)`` — geometric spacing cannot start
    at zero — and the rest share one ratio out to ``r_max``.
    """
    e = geometric_edges(r_max=10.0, n_shells=6, r_min=0.05)
    assert e.shape == (7,)
    assert e[0] == 0.0
    assert e[1] == pytest.approx(0.05)
    assert e[-1] == pytest.approx(10.0)
    ratios = e[2:] / e[1:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)


def test_edge_helpers_validate_their_arguments() -> None:
    with pytest.raises(ValueError, match="at least one shell"):
        uniform_edges(5.0, 0)
    with pytest.raises(ValueError, match="positive"):
        uniform_edges(0.0, 4)
    with pytest.raises(ValueError, match="at least two shells"):
        geometric_edges(r_max=10.0, n_shells=1, r_min=0.05)
    with pytest.raises(ValueError, match="r_min"):
        geometric_edges(r_max=10.0, n_shells=4, r_min=20.0)


# ---------------------------------------------------------------------------
# Graded depth binning
# ---------------------------------------------------------------------------
#
# A kernel database needs fine depth bins through the build-up region and coarse
# ones in the tail, so the depth axis takes arbitrary edges exactly as the radial
# axis does. Both go through the same bin-lookup primitive.


DEPTH_SEGMENTS = [(0.0, 0.5, 0.010), (0.5, 2.0, 0.025), (2.0, 6.0, 0.25), (6.0, 32.0, 1.0)]


def test_graded_edges_builds_the_segments_it_is_given() -> None:
    edges = graded_edges(DEPTH_SEGMENTS)
    assert edges.size == 153  # 50 + 60 + 16 + 26 bins
    assert edges[0] == 0.0
    assert edges[-1] == pytest.approx(32.0)
    np.testing.assert_allclose(np.diff(edges)[:50], 0.010)
    np.testing.assert_allclose(np.diff(edges)[50:110], 0.025)
    np.testing.assert_allclose(np.diff(edges)[110:126], 0.25)
    np.testing.assert_allclose(np.diff(edges)[126:], 1.0)


def test_graded_edges_refuses_a_segment_that_is_not_a_whole_number_of_steps() -> None:
    """Silently rounding the last step would misplace every downstream edge."""
    with pytest.raises(ValueError, match="whole number of steps"):
        graded_edges([(0.0, 1.0, 0.3)])


def test_graded_edges_refuses_discontiguous_or_backwards_segments() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        graded_edges([(0.0, 1.0, 0.5), (2.0, 3.0, 0.5)])
    with pytest.raises(ValueError, match="positive"):
        graded_edges([(0.0, 1.0, -0.5)])


def test_graded_depth_mass_is_the_annulus_volume_of_each_bin() -> None:
    """Bin mass tracks its own thickness, not a single global spacing."""
    depth = np.array([0.0, 0.1, 0.5, 2.5])
    cyl = CylindricalScoringGrid.uniform(
        density=1.0, axis=(0.0, 0.0), depth_edges=depth, radial_edges=EDGES
    )
    area = math.pi * (EDGES[1:] ** 2 - EDGES[:-1] ** 2)
    for iz, thickness in enumerate((0.1, 0.4, 2.0)):
        np.testing.assert_allclose(cyl.voxel_mass[iz], area * thickness, rtol=1e-14)
    assert cyl.n_depth == 3
    np.testing.assert_allclose(cyl.depth_thickness, [0.1, 0.4, 2.0])


def test_graded_depth_routes_positions_into_the_right_bin() -> None:
    depth = np.array([0.0, 0.1, 0.5, 2.5])
    cyl = CylindricalScoringGrid.uniform(
        density=1.0, axis=(0.0, 0.0), depth_edges=depth, radial_edges=EDGES
    )
    assert cyl.voxel_index(0.0, 0.0, 0.05)[0] == 0
    assert cyl.voxel_index(0.0, 0.0, 0.1)[0] == 1  # on the edge -> deeper bin
    assert cyl.voxel_index(0.0, 0.0, 0.49)[0] == 1
    assert cyl.voxel_index(0.0, 0.0, 0.5)[0] == 2
    assert cyl.flat_index(0.0, 0.0, 2.5) == -1  # exit plane is outside
    assert cyl.flat_index(0.0, 0.0, -1e-9) == -1


def test_uniform_depth_still_reachable_through_the_edge_helper() -> None:
    """The common case is one call away; the geometry has one representation."""
    cyl = CylindricalScoringGrid.uniform(
        density=1.0,
        axis=(0.0, 0.0),
        depth_edges=uniform_edges(4.0, 4),
        radial_edges=EDGES,
    )
    assert cyl.n_depth == 4
    np.testing.assert_allclose(cyl.depth_thickness, 1.0)
    np.testing.assert_allclose(cyl.depth_centers, [0.5, 1.5, 2.5, 3.5])


# ---------------------------------------------------------------------------
# Suggested deposit resolution
# ---------------------------------------------------------------------------
#
# Sub-substep deposits sit at a fixed spacing from the step start, and step starts
# are pinned to transport voxel faces, so the point set is locked to the voxel
# lattice. A spacing that does not divide the scoring bin width beats against it,
# and the fixed phase makes that beat a standing ripple instead of noise. The
# suggested value must therefore be commensurate, not merely small.


def test_common_bin_divisor_finds_the_exact_divisor() -> None:
    widths = np.array([0.010] * 3 + [0.025] * 2 + [0.25, 1.0])
    g = common_bin_divisor(widths, floor=0.010 / 16)
    assert g == pytest.approx(0.005, rel=1e-9)
    for w in (0.010, 0.025, 0.25, 1.0):
        assert (w / g) == pytest.approx(round(w / g), rel=1e-9)


def test_common_bin_divisor_of_one_width_is_that_width() -> None:
    g = common_bin_divisor(np.array([0.02, 0.02, 0.02]), floor=0.001)
    assert g == pytest.approx(0.02, rel=1e-9)


def test_common_bin_divisor_falls_back_to_the_floor_when_incommensurable() -> None:
    """Arbitrary widths drive the divisor to zero; the floor bounds the cost."""
    rng = np.random.default_rng(20260716)
    widths = rng.uniform(0.01, 0.9, size=6)
    floor = 0.001
    assert common_bin_divisor(widths, floor) >= floor


def test_suggested_resolution_divides_every_depth_bin() -> None:
    """The property that makes the suggestion safe to pass unexamined."""
    cyl = CylindricalScoringGrid.uniform(
        density=1.0,
        axis=(0.0, 0.0),
        depth_edges=graded_edges(DEPTH_SEGMENTS),
        radial_edges=EDGES,
    )
    r = cyl.deposit_resolution_cm
    assert r > 0.0
    for w in np.unique(np.round(cyl.depth_thickness, 12)):
        assert (w / r) == pytest.approx(round(w / r), rel=1e-9), f"{w} not divisible by {r}"
    # Fine enough to put at least two deposits in the narrowest bin.
    assert r <= 0.5 * cyl.depth_thickness.min() * (1.0 + 1e-9)
