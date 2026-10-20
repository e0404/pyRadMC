"""Static magnetic field data model and its single-source lookup.

``MagneticField`` is world data the engines will carry next to the voxel grid: a
rectilinear map of B in tesla, sampled at cell centres, with the transport grid's
lower-corner half-open convention. ``field_at`` is the pure scalar lookup that both
the reference loop and the Warp kernels will call per substep. What is pinned here:

* construction validation and the one canonical form per representation
  (``uniform`` is a ``(1, 1, 1)`` nearest map; a zero field keeps its mode);
* ``on_grid`` copies the transport grid's geometry bit-exactly, so a per-voxel map
  reads the same cell the transport loop is in;
* ``covers`` at the half-open edge, in both directions;
* a digest that moves with any byte and a provenance line naming the mode, the
  shape and ``|B|max``;
* the lookup: uniform never touches the array; nearest reproduces the transport
  grid's own ``voxel_index``; trilinear is exact for an affine field inside the
  map, equal to the sample at every centre, flat beyond the last centre, and
  indifferent to a degenerate (``n == 1``) axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SEED


def _random_grid(rng: np.random.Generator, shape=(4, 5, 6)):
    from pyradmc.geometry.grid import VoxelGrid

    spacing = tuple(float(s) for s in rng.uniform(0.2, 0.7, size=3))
    origin = tuple(float(o) for o in rng.uniform(-3.0, 3.0, size=3))
    return VoxelGrid.uniform_water(shape=shape, spacing=spacing, origin=origin)


def _random_values(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=(*shape, 3))


def _centres(field) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centre coordinates along each axis of a field's sample grid."""
    return tuple(
        field.origin[a] + (np.arange(field.shape[a]) + 0.5) * field.spacing[a] for a in range(3)
    )


def _field_args(field):
    """The scalar argument list ``field_at`` takes after ``values``, mirroring ``sample``."""
    b0 = field.values[0, 0, 0]
    return (
        field.mode,
        float(b0[0]),
        float(b0[1]),
        float(b0[2]),
        *field.origin,
        *field.spacing,
        *field.shape,
    )


# ---------------------------------------------------------------------------
# Construction, validation and canonical forms
# ---------------------------------------------------------------------------


def test_mode_constants_are_distinct_small_ints() -> None:
    from pyradmc.geometry.field import FIELD_NEAREST, FIELD_NONE, FIELD_TRILINEAR, FIELD_UNIFORM

    assert (FIELD_NONE, FIELD_UNIFORM, FIELD_NEAREST, FIELD_TRILINEAR) == (0, 1, 2, 3)


def test_construction_validates_values_geometry_and_interpolation() -> None:
    from pyradmc.geometry.field import MagneticField

    good = np.zeros((2, 3, 4, 3))
    with pytest.raises(ValueError, match="values"):
        MagneticField(values=np.zeros((2, 3, 4)), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="values"):
        MagneticField(values=np.zeros((2, 3, 4, 2)), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="empty"):
        MagneticField(values=np.zeros((2, 0, 4, 3)), spacing=(1.0, 1.0, 1.0))
    bad = good.copy()
    bad[1, 2, 3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        MagneticField(values=bad, spacing=(1.0, 1.0, 1.0))
    bad[1, 2, 3, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        MagneticField(values=bad, spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="spacing"):
        MagneticField(values=good, spacing=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="spacing"):
        MagneticField(values=good, spacing=(1.0, -0.5, 1.0))
    with pytest.raises(ValueError, match="spacing"):
        MagneticField(values=good, spacing=(1.0, float("nan"), 1.0))
    with pytest.raises(ValueError, match="origin"):
        MagneticField(values=good, spacing=(1.0, 1.0, 1.0), origin=(0.0, float("inf"), 0.0))
    with pytest.raises(ValueError, match="interpolation"):
        MagneticField(values=good, spacing=(1.0, 1.0, 1.0), interpolation="cubic")
    # Both names construct.
    MagneticField(values=good, spacing=(1.0, 1.0, 1.0), interpolation="trilinear")
    MagneticField(values=good, spacing=(1.0, 1.0, 1.0), interpolation="nearest")


def test_values_are_stored_as_contiguous_float64() -> None:
    """One canonical byte layout: the digest and the device upload depend on it."""
    from pyradmc.geometry.field import MagneticField

    ints = np.ones((2, 2, 2, 3), dtype=np.int32)
    field = MagneticField(values=ints, spacing=(1.0, 1.0, 1.0))
    assert field.values.dtype == np.float64
    assert field.values.flags.c_contiguous
    fortran = np.asfortranarray(np.arange(2 * 2 * 2 * 3, dtype=np.float64).reshape(2, 2, 2, 3))
    field = MagneticField(values=fortran, spacing=(1.0, 1.0, 1.0))
    assert field.values.flags.c_contiguous
    np.testing.assert_array_equal(field.values, fortran)


@pytest.mark.parametrize("constructor", ["direct", "uniform", "on_grid"])
def test_field_owns_a_read_only_copy_of_input_values(constructor: str) -> None:
    from pyradmc.geometry.field import MagneticField
    from pyradmc.geometry.grid import VoxelGrid

    values = np.full((2, 2, 2, 3), 0.5)
    if constructor == "uniform":
        values = values[0, 0, 0]
        field = MagneticField.uniform(values)
    elif constructor == "on_grid":
        grid = VoxelGrid.uniform_water(shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
        field = MagneticField.on_grid(grid, values)
    else:
        field = MagneticField(values=values, spacing=(1.0, 1.0, 1.0))
    digest = field.digest
    values[...] = np.nan
    assert field.sample(0.5, 0.5, 0.5) == (0.5, 0.5, 0.5)
    assert field.digest == digest
    with pytest.raises(ValueError, match="read-only"):
        field.values[0, 0, 0, 0] = 1.5


@pytest.mark.parametrize("axis", range(3))
def test_spacing_must_be_finite(axis: int) -> None:
    from pyradmc.geometry.field import MagneticField

    spacing = [1.0, 1.0, 1.0]
    spacing[axis] = np.inf
    with pytest.raises(ValueError, match="spacing"):
        MagneticField(values=np.zeros((2, 2, 2, 3)), spacing=tuple(spacing))


def test_uniform_is_one_nearest_cell_with_infinite_extent() -> None:
    from pyradmc.geometry.field import FIELD_UNIFORM, MagneticField
    from pyradmc.geometry.grid import VoxelGrid

    field = MagneticField.uniform((0.0, 0.0, 1.5))
    assert field.shape == (1, 1, 1)
    assert field.values.shape == (1, 1, 1, 3)
    assert field.interpolation == "nearest"
    assert field.is_uniform
    assert field.mode == FIELD_UNIFORM
    assert field.max_magnitude == 1.5
    np.testing.assert_array_equal(field.values[0, 0, 0], [0.0, 0.0, 1.5])

    far = VoxelGrid.uniform_water(shape=(3, 3, 3), spacing=(10.0, 10.0, 10.0), origin=(1e3,) * 3)
    assert field.covers(far)
    assert field.sample(-1e6, 2e5, 3e4) == (0.0, 0.0, 1.5)


def test_single_cell_map_must_be_nearest() -> None:
    """A (1, 1, 1) map is the uniform representation; it has exactly one canonical form."""
    from pyradmc.geometry.field import MagneticField

    with pytest.raises(ValueError, match="uniform"):
        MagneticField(values=np.zeros((1, 1, 1, 3)), spacing=(1.0, 1.0, 1.0))
    # ... which `uniform` produces, and which direct construction with "nearest" equals.
    direct = MagneticField(
        values=np.array([[[[0.0, 0.0, 1.5]]]]), spacing=(1.0, 1.0, 1.0), interpolation="nearest"
    )
    assert direct.digest == MagneticField.uniform((0.0, 0.0, 1.5)).digest


def test_on_grid_copies_the_transport_geometry_bit_exactly() -> None:
    from pyradmc.geometry.field import FIELD_NEAREST, MagneticField

    rng = np.random.default_rng(SEED)
    grid = _random_grid(rng)
    values = _random_values(rng, grid.shape)
    field = MagneticField.on_grid(grid, values)

    assert field.shape == grid.shape
    assert field.spacing == grid.spacing
    assert field.origin == grid.origin
    assert field.upper_corner == grid.upper_corner
    assert field.interpolation == "nearest"
    assert field.mode == FIELD_NEAREST
    assert not field.is_uniform
    assert field.covers(grid)
    np.testing.assert_array_equal(field.values, values)

    with pytest.raises(ValueError, match="shape"):
        MagneticField.on_grid(grid, values[:-1])


def test_zero_field_keeps_its_mode() -> None:
    """Plan decision 10: an all-zero field is data, not the absence of a field.

    Collapsing it to ``FIELD_NONE`` would leave the kernel's own zero-vector guard
    unexercised by the B=0 bit-identity test; the field-free path is ``None``.
    """
    from pyradmc.geometry.field import (
        FIELD_NEAREST,
        FIELD_NONE,
        FIELD_TRILINEAR,
        FIELD_UNIFORM,
        MagneticField,
    )
    from pyradmc.geometry.grid import VoxelGrid

    grid = VoxelGrid.uniform_water(shape=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
    assert MagneticField.uniform((0.0, 0.0, 0.0)).mode == FIELD_UNIFORM
    assert MagneticField.on_grid(grid, np.zeros((2, 2, 2, 3))).mode == FIELD_NEAREST
    zero_map = MagneticField(values=np.zeros((2, 2, 2, 3)), spacing=(1.0, 1.0, 1.0))
    assert zero_map.mode == FIELD_TRILINEAR
    assert FIELD_NONE not in {
        MagneticField.uniform((0.0, 0.0, 0.0)).mode,
        zero_map.mode,
    }


def test_max_magnitude_is_the_largest_vector_norm() -> None:
    from pyradmc.geometry.field import MagneticField

    values = np.zeros((2, 2, 2, 3))
    values[1, 0, 1] = (0.3, -0.4, 1.2)  # |B| = 1.3
    values[0, 1, 0] = (1.0, 0.0, 0.0)
    field = MagneticField(values=values, spacing=(1.0, 1.0, 1.0))
    assert field.max_magnitude == pytest.approx(1.3, rel=1e-15)


# ---------------------------------------------------------------------------
# Coverage at the half-open edge
# ---------------------------------------------------------------------------


def test_covers_at_the_half_open_edge_both_ways() -> None:
    from pyradmc.geometry.field import MagneticField
    from pyradmc.geometry.grid import VoxelGrid

    grid = VoxelGrid.uniform_water(
        shape=(4, 3, 2), spacing=(0.5, 0.25, 1.0), origin=(-1.0, 2.0, 0.0)
    )
    values = np.zeros((8, 6, 4, 3))

    def field(origin, spacing):
        return MagneticField(values=values, spacing=spacing, origin=origin)

    # Same box, finer sampling: the upper corners coincide exactly.
    exact = field(grid.origin, (0.25, 0.125, 0.5))
    assert exact.upper_corner == grid.upper_corner
    assert exact.covers(grid)
    # Larger box on every side.
    assert field((-2.0, 1.0, -1.0), (0.5, 0.5, 1.0)).covers(grid)
    # Upper corner one step short on one axis (grid's half-open upper face is
    # outside the grid, but the voxels just below it are not in the field).
    assert not field(grid.origin, (0.25, 0.125, 0.5 - 1e-9)).covers(grid)
    # Origin a hair above the grid origin on one axis.
    assert not field((-1.0 + 1e-9, 2.0, 0.0), (0.25, 0.125, 0.5)).covers(grid)
    # Origin a hair below is fine — provided the box still reaches the grid's upper
    # corner (a shifted origin alone also shifts the upper corner down).
    assert field((-1.0 - 1e-9, 2.0, 0.0), (0.25 + 1e-9, 0.125, 0.5)).covers(grid)
    assert not field((-1.0 - 1e-9, 2.0, 0.0), (0.25, 0.125, 0.5)).covers(grid)
    # Nearest and trilinear use the same box rule.
    assert MagneticField(
        values=values, spacing=(0.25, 0.125, 0.5), origin=grid.origin, interpolation="nearest"
    ).covers(grid)


# ---------------------------------------------------------------------------
# Digest and provenance
# ---------------------------------------------------------------------------


def test_digest_changes_with_any_byte() -> None:
    from pyradmc.geometry.field import MagneticField

    rng = np.random.default_rng(SEED)
    values = _random_values(rng, (3, 2, 4))
    base = MagneticField(values=values, spacing=(0.5, 0.5, 0.5), origin=(0.0, 0.0, 0.0))
    assert len(base.digest) == 64
    assert int(base.digest, 16) >= 0
    same = MagneticField(values=values.copy(), spacing=(0.5, 0.5, 0.5), origin=(0.0, 0.0, 0.0))
    assert same.digest == base.digest

    flipped = values.copy()
    flipped[2, 1, 3, 1] = np.nextafter(flipped[2, 1, 3, 1], np.inf)
    assert MagneticField(values=flipped, spacing=(0.5, 0.5, 0.5)).digest != base.digest
    assert MagneticField(values=values, spacing=(0.5, 0.5, 0.5 + 1e-12)).digest != base.digest
    assert MagneticField(
        values=values, spacing=(0.5, 0.5, 0.5), origin=(0.0, 1e-12, 0.0)
    ).digest != (base.digest)
    assert (
        MagneticField(values=values, spacing=(0.5, 0.5, 0.5), interpolation="nearest").digest
        != base.digest
    )
    # Same bytes under a different shape is a different field.
    assert MagneticField(values=values.reshape(2, 3, 4, 3), spacing=(0.5, 0.5, 0.5)).digest != (
        base.digest
    )


def test_provenance_names_mode_shape_and_max_magnitude() -> None:
    from pyradmc.geometry.field import MagneticField

    uniform = MagneticField.uniform((0.0, 0.0, 1.5)).provenance
    assert "uniform" in uniform
    assert "1.5" in uniform
    assert "T" in uniform

    values = np.zeros((4, 5, 6, 3))
    values[3, 4, 5] = (0.0, 0.35, 0.0)
    tri = MagneticField(values=values, spacing=(1.0, 1.0, 1.0))
    assert "trilinear" in tri.provenance
    assert "4x5x6" in tri.provenance
    assert "|B|max" in tri.provenance
    assert "0.35" in tri.provenance
    assert tri.digest[:8] in tri.provenance
    near = MagneticField(values=values, spacing=(1.0, 1.0, 1.0), interpolation="nearest")
    assert "nearest" in near.provenance


def test_uniform_provenance_preserves_the_field_components() -> None:
    from pyradmc.geometry.field import MagneticField

    b = 1.5
    changed = np.nextafter(b, np.inf)
    assert (
        MagneticField.uniform((b, 0.0, 0.0)).provenance
        != MagneticField.uniform((changed, 0.0, 0.0)).provenance
    )


# ---------------------------------------------------------------------------
# field_at on the host
# ---------------------------------------------------------------------------


def test_field_at_uniform_returns_the_constants_without_reading_values() -> None:
    from pyradmc.geometry.field import FIELD_UNIFORM, field_at

    poison = np.full((1, 1, 1, 3), np.nan)
    got = field_at(
        poison, FIELD_UNIFORM, 0.1, -0.2, 1.5, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 1, 5.0, 6.0, 7.0
    )
    assert got == (0.1, -0.2, 1.5)


def test_field_at_nearest_reproduces_voxel_index_on_a_random_grid() -> None:
    from pyradmc.geometry.field import MagneticField, field_at

    rng = np.random.default_rng(SEED)
    grid = _random_grid(rng)
    values = _random_values(rng, grid.shape)
    field = MagneticField.on_grid(grid, values)
    args = _field_args(field)

    lo = np.asarray(grid.origin)
    hi = np.asarray(grid.upper_corner)
    points = lo + rng.uniform(0.0, 1.0, size=(200, 3)) * (hi - lo)
    for x, y, z in points:
        assert grid.contains(x, y, z)
        want = values[grid.voxel_index(x, y, z)]
        got = field_at(field.values, *args, x, y, z)
        assert got == (want[0], want[1], want[2])

    # The lower faces belong to their cell; the upper outer face is clamped into
    # the last cell (ULP guard, point_axis_index semantics).
    assert field_at(field.values, *args, *grid.origin) == tuple(values[0, 0, 0])
    assert field_at(field.values, *args, *grid.upper_corner) == tuple(values[-1, -1, -1])


def test_field_at_trilinear_is_exact_for_an_affine_field_inside_the_map() -> None:
    from pyradmc.geometry.field import FIELD_TRILINEAR, MagneticField, field_at

    rng = np.random.default_rng(SEED)
    grid = _random_grid(rng, shape=(5, 4, 6))
    cx, cy, cz = (
        grid.origin[a] + (np.arange(grid.shape[a]) + 0.5) * grid.spacing[a] for a in range(3)
    )
    amat = rng.normal(size=(3, 3))
    offset = rng.normal(size=3)
    xx, yy, zz = np.meshgrid(cx, cy, cz, indexing="ij")
    pos = np.stack([xx, yy, zz], axis=-1)
    values = pos @ amat.T + offset
    field = MagneticField(values=values, spacing=grid.spacing, origin=grid.origin)
    assert field.mode == FIELD_TRILINEAR
    args = _field_args(field)

    # Strictly inside the convex hull of the centres.
    lo = np.array([cx[0], cy[0], cz[0]])
    hi = np.array([cx[-1], cy[-1], cz[-1]])
    points = lo + rng.uniform(0.0, 1.0, size=(300, 3)) * (hi - lo)
    for p in points:
        want = amat @ p + offset
        got = field_at(field.values, *args, *p)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_field_at_trilinear_equals_the_sample_at_every_cell_centre() -> None:
    from pyradmc.geometry.field import MagneticField, field_at

    rng = np.random.default_rng(SEED)
    grid = _random_grid(rng, shape=(3, 4, 5))
    values = _random_values(rng, grid.shape)
    field = MagneticField(values=values, spacing=grid.spacing, origin=grid.origin)
    args = _field_args(field)
    cx, cy, cz = _centres(field)
    for i, x in enumerate(cx):
        for j, y in enumerate(cy):
            for k, z in enumerate(cz):
                got = field_at(field.values, *args, x, y, z)
                np.testing.assert_allclose(got, values[i, j, k], rtol=1e-12, atol=0.0)


def test_field_at_trilinear_clamps_flat_beyond_the_last_centre() -> None:
    from pyradmc.geometry.field import MagneticField, field_at

    rng = np.random.default_rng(SEED)
    values = _random_values(rng, (4, 3, 5))
    field = MagneticField(values=values, spacing=(0.5, 0.25, 1.0), origin=(-1.0, 2.0, 0.0))
    args = _field_args(field)
    cx, cy, cz = _centres(field)
    y0, z0 = cy[1], cz[2]

    # Beyond the outermost centre both bracketing loads are the edge cell. Between
    # the last centre and one spacing past it the weight is still fractional and
    # ``a*(1-w) + a*w`` equals ``a`` only to rounding; from one spacing on (and
    # anywhere below the first centre) the weight clamps to exactly 1 (0) and the
    # result is the edge sample bit for bit.
    edge_hi = tuple(values[-1, 1, 2])
    for x in (cx[-1] + 1e-9, cx[-1] + 0.1, field.upper_corner[0]):
        got = field_at(field.values, *args, x, y0, z0)
        np.testing.assert_allclose(got, edge_hi, rtol=1e-12, atol=0.0)
    for x in (cx[-1] + field.spacing[0], field.upper_corner[0] + 7.0):
        assert field_at(field.values, *args, x, y0, z0) == edge_hi
    edge_lo = tuple(values[0, 1, 2])
    for x in (cx[0] - 1e-9, cx[0] - 0.1, field.origin[0], field.origin[0] - 7.0):
        assert field_at(field.values, *args, x, y0, z0) == edge_lo
    # Same on the other axes, at a corner well outside the map.
    far = (field.upper_corner[0] + 1.0, field.origin[1] - 1.0, field.upper_corner[2] + 1.0)
    assert field_at(field.values, *args, *far) == tuple(values[-1, 0, -1])


def test_field_at_trilinear_handles_degenerate_axes() -> None:
    """A map that is one cell thick in x and z interpolates along y only."""
    from pyradmc.geometry.field import MagneticField, field_at

    rng = np.random.default_rng(SEED)
    values = _random_values(rng, (1, 6, 1))
    field = MagneticField(values=values, spacing=(2.0, 0.5, 3.0), origin=(0.0, -1.0, 0.0))
    args = _field_args(field)
    cy = _centres(field)[1]
    ys = np.linspace(-3.0, 4.0, 41)  # covers below, inside and beyond the centres
    for x in (-5.0, 0.0, 1.0, 1.9999, 2.0, 10.0):
        for z in (-1.0, 0.0, 1.5, 3.0, 9.0):
            for y in ys:
                got = field_at(field.values, *args, x, y, z)
                want = [np.interp(y, cy, values[0, :, 0, c]) for c in range(3)]
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_sample_delegates_to_field_at() -> None:
    from pyradmc.geometry.field import MagneticField, field_at

    rng = np.random.default_rng(SEED)
    grid = _random_grid(rng)
    values = _random_values(rng, grid.shape)
    fields = [
        MagneticField.uniform((0.2, -0.3, 1.4)),
        MagneticField.on_grid(grid, values),
        MagneticField(values=values, spacing=grid.spacing, origin=grid.origin),
    ]
    lo = np.asarray(grid.origin) - 0.5
    hi = np.asarray(grid.upper_corner) + 0.5
    points = lo + rng.uniform(0.0, 1.0, size=(50, 3)) * (hi - lo)
    for field in fields:
        args = _field_args(field)
        for x, y, z in points:
            assert field.sample(x, y, z) == field_at(field.values, *args, x, y, z)
