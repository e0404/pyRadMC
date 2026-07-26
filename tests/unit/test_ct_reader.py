"""SimpleITK CT image -> VoxelGrid.

The image-reading half: SimpleITK indexes (z, y, x) and works in mm with a physical
origin at the first voxel's centre, while the engine grid is (x, y, z) in cm with the
origin at voxel (0,0,0)'s lower corner. These tests pin that translation on synthetic
in-memory images (no committed CT data), plus a temp-file round trip for ``read_ct``.
Needs the optional ``pyRadMC[ct]`` extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK", reason="CT adapter needs the pyRadMC[ct] extra")

pytestmark = pytest.mark.ct

from pyRadMC.adapters.ct import grid_from_image, read_ct  # noqa: E402
from pyRadMC.data.materials import AIR, CORTICAL_BONE, WATER  # noqa: E402


def _image_from_hu(
    hu_xyz: np.ndarray,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> sitk.Image:
    """Build a SimpleITK image from an (x, y, z) HU array (transposing to its z,y,x)."""
    image = sitk.GetImageFromArray(np.transpose(hu_xyz, (2, 1, 0)))
    image.SetSpacing(spacing_mm)
    image.SetOrigin(origin_mm)
    image.SetDirection(direction)
    return image


def test_orientation_round_trips_x_y_z() -> None:
    """A distinctive HU voxel lands at the same (x, y, z) index in the grid.

    Guards the z,y,x <-> x,y,z transpose: a scrambled transpose would put the marker in
    the wrong voxel while still producing a valid grid.
    """
    hu = np.zeros((2, 3, 4), dtype=np.float64)  # water everywhere
    hu[1, 2, 3] = 800.0  # a bone voxel at a corner
    hu[0, 0, 0] = -1000.0  # an air voxel at the opposite corner
    grid = grid_from_image(_image_from_hu(hu))
    assert grid.shape == (2, 3, 4)
    assert grid.material[1, 2, 3] == CORTICAL_BONE
    assert grid.material[0, 0, 0] == AIR
    assert grid.material[1, 1, 1] == WATER


def test_spacing_is_converted_mm_to_cm() -> None:
    hu = np.zeros((2, 2, 2))
    grid = grid_from_image(_image_from_hu(hu, spacing_mm=(3.0, 1.5, 5.0)))
    assert grid.spacing == pytest.approx((0.3, 0.15, 0.5))


def test_origin_is_voxel_corner_in_cm() -> None:
    """ITK origin is the first voxel centre in mm; the grid wants the lower corner in cm."""
    hu = np.zeros((2, 2, 2))
    grid = grid_from_image(
        _image_from_hu(hu, spacing_mm=(4.0, 4.0, 4.0), origin_mm=(10.0, 20.0, 30.0))
    )
    # centre 10 mm - half a 4 mm voxel = 8 mm = 0.8 cm.
    assert grid.origin == pytest.approx((0.8, 1.8, 2.8))


def test_negative_direction_cosine_flips_the_axis() -> None:
    """An axis whose physical coordinate decreases with index is flipped to increasing.

    The grid is stored in increasing-coordinate order; a -1 direction cosine (an LPS/RAS
    sign flip, common in clinical CT) reverses that axis, so the array is flipped and
    the origin recomputed to the true minimum-coordinate corner.
    """
    hu = np.zeros((3, 2, 2))
    hu[0, 0, 0] = 800.0  # bone at index 0 along x
    # x decreases with index: origin (centre of first voxel) is the MAX-x end.
    flipped_x = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    image = _image_from_hu(
        hu, spacing_mm=(2.0, 2.0, 2.0), origin_mm=(20.0, 0.0, 0.0), direction=flipped_x
    )
    grid = grid_from_image(image)
    # After flipping, the bone voxel is now at the far (max-index) end of x.
    assert grid.material[2, 0, 0] == CORTICAL_BONE
    assert grid.material[0, 0, 0] == WATER
    # Min-x centre = 20 - 2*(3-1) = 16 mm; lower corner = 16 - 1 = 15 mm = 1.5 cm.
    assert grid.origin[0] == pytest.approx(1.5)


def test_oblique_orientation_is_rejected() -> None:
    hu = np.zeros((2, 2, 2))
    oblique = (0.985, 0.174, 0.0, -0.174, 0.985, 0.0, 0.0, 0.0, 1.0)  # ~10 deg in-plane
    with pytest.raises(ValueError, match="oblique"):
        grid_from_image(_image_from_hu(hu, direction=oblique))


def test_non_3d_image_is_rejected() -> None:
    image_2d = sitk.GetImageFromArray(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="3D"):
        grid_from_image(image_2d)


def test_read_ct_round_trips_through_a_file(tmp_path: Path) -> None:
    """read_ct on a written image matches grid_from_image on the in-memory one."""
    hu = np.zeros((3, 4, 5))
    hu[2, 3, 4] = 800.0
    hu[0, 0, 0] = -1000.0
    image = _image_from_hu(hu, spacing_mm=(2.0, 2.0, 3.0), origin_mm=(5.0, 5.0, 5.0))
    path = tmp_path / "phantom.nrrd"
    sitk.WriteImage(image, str(path))

    from_file = read_ct(str(path))
    in_memory = grid_from_image(image)
    assert from_file.shape == in_memory.shape
    assert from_file.spacing == pytest.approx(in_memory.spacing)
    assert from_file.origin == pytest.approx(in_memory.origin)
    np.testing.assert_array_equal(from_file.material, in_memory.material)
    np.testing.assert_allclose(from_file.density, in_memory.density)
