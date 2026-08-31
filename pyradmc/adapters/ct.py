"""CT image -> :class:`~pyradmc.geometry.grid.VoxelGrid` adapter.

Turns a patient CT into the engine's geometry: a Hounsfield-unit calibration maps each
voxel's CT number to a mass density (a piecewise-linear ramp) and to a registry
material index (HU threshold bins into the ICRP media the materials task added), and
:func:`grid_from_hu` assembles the grid. :func:`read_ct` reads an image file through
SimpleITK; everything above it is pure NumPy and needs no optional dependency, so the
calibration is testable and reusable on its own.

Segmentation and density are deliberately independent, which is the whole point of the
multi-material design: the material index selects the elemental composition (the
compiled cross-section row) and the per-voxel density scales it. A voxel at -500 HU is
*lung tissue* (composition) at ~0.5 g/cm^3 (density), not a fixed reference-density
material.

The default calibration is illustrative, in Schneider's spirit (Schneider, Bortfeld &
Schlegel, Phys. Med. Biol. 45, 459 (2000), doi:10.1088/0031-9155/45/2/314; bilinear
density ramp after Schneider, Pedroni & Lomax, Phys. Med. Biol. 41, 111 (1996),
doi:10.1088/0031-9155/41/1/009). CT calibration is scanner-specific: a real plan passes
its own :class:`HounsfieldCalibration`, and these defaults are a runnable starting
point, not a clinical curve.

This adapter is out of the core (AGENTS.md 6); ``read_ct`` needs the optional
``pyradmc[ct]`` extra (SimpleITK, Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pyradmc.data.materials import ADIPOSE, AIR, CORTICAL_BONE, LUNG, MATERIALS, WATER
from pyradmc.geometry.grid import VoxelGrid

if TYPE_CHECKING:
    import SimpleITK as sitk

__all__ = [
    "DEFAULT_CALIBRATION",
    "HounsfieldCalibration",
    "grid_from_hu",
    "grid_from_image",
    "read_ct",
]

_MM_PER_CM = 10.0


# Default HU -> mass density ramp (g/cm^3), linearly interpolated and clamped at the
# ends. Air and water are the fixed physical anchors; the bone segment reaches liquid
# water's cortical-bone reference density (~1.85 g/cm^3) near 1400 HU and continues to
# dense mineral above. Nodes ascending in HU.
_DEFAULT_DENSITY_HU: tuple[float, ...] = (-1000.0, 0.0, 1000.0, 3000.0)
_DEFAULT_DENSITY_VALUES: tuple[float, ...] = (0.00121, 1.000, 1.600, 2.800)

# Default HU -> material bins. n thresholds cut the HU axis into n+1 bins; bin i (HU in
# [thresholds[i-1], thresholds[i])) takes material_indices[i]. Illustrative Schneider-
# style cut points into the ICRP registry media.
_DEFAULT_MATERIAL_THRESHOLDS: tuple[float, ...] = (-950.0, -120.0, -20.0, 125.0)
_DEFAULT_MATERIAL_INDICES: tuple[int, ...] = (AIR, LUNG, ADIPOSE, WATER, CORTICAL_BONE)


@dataclass(frozen=True)
class HounsfieldCalibration:
    """A CT-number calibration: HU -> mass density and HU -> registry material.

    Attributes
    ----------
    density_hu, density_values
        The density ramp: HU control points (strictly ascending) and the mass density
        in g/cm^3 at each. Interpolated linearly, clamped flat beyond the ends. All
        densities must be positive — the grid rejects non-positive density.
    material_thresholds
        Ascending HU cut points. ``n`` thresholds define ``n + 1`` bins.
    material_indices
        The registry material index for each bin, low-HU to high-HU; length must be
        ``len(material_thresholds) + 1``. Each must be a valid index into
        :data:`pyradmc.data.materials.MATERIALS`.
    """

    density_hu: tuple[float, ...]
    density_values: tuple[float, ...]
    material_thresholds: tuple[float, ...]
    material_indices: tuple[int, ...]
    _density_hu_arr: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _density_val_arr: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _threshold_arr: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _index_arr: npt.NDArray[np.int32] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the ramp and the bins, then cache their array forms."""
        hu = np.asarray(self.density_hu, dtype=np.float64)
        values = np.asarray(self.density_values, dtype=np.float64)
        if hu.shape != values.shape or hu.size < 2:
            raise ValueError("density_hu and density_values must be equal-length (>=2)")
        if not np.all(np.diff(hu) > 0.0):
            raise ValueError("density_hu must be strictly ascending")
        if not np.all(values > 0.0):
            raise ValueError("density_values must all be positive (g/cm^3)")

        thresholds = np.asarray(self.material_thresholds, dtype=np.float64)
        indices = np.asarray(self.material_indices, dtype=np.int32)
        if thresholds.size and not np.all(np.diff(thresholds) > 0.0):
            raise ValueError("material_thresholds must be strictly ascending")
        if indices.size != thresholds.size + 1:
            raise ValueError(
                f"{thresholds.size} thresholds define {thresholds.size + 1} bins, "
                f"but got {indices.size} material_indices"
            )
        if np.any((indices < 0) | (indices >= len(MATERIALS))):
            raise ValueError("material_indices must all be valid registry indices")

        object.__setattr__(self, "_density_hu_arr", hu)
        object.__setattr__(self, "_density_val_arr", values)
        object.__setattr__(self, "_threshold_arr", thresholds)
        object.__setattr__(self, "_index_arr", indices)

    def density(self, hu: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Mass density in g/cm^3 for the given HU (scalar or array), clamped at the ends."""
        return np.interp(
            np.asarray(hu, dtype=np.float64), self._density_hu_arr, self._density_val_arr
        )

    def material(self, hu: npt.ArrayLike) -> npt.NDArray[np.int32]:
        """Registry material index for the given HU (scalar or array).

        ``np.digitize`` places each HU in its bin (left-closed, ``x < threshold``);
        the bin index gathers the material index.
        """
        bins = np.digitize(np.asarray(hu, dtype=np.float64), self._threshold_arr)
        return self._index_arr[bins]


DEFAULT_CALIBRATION = HounsfieldCalibration(
    density_hu=_DEFAULT_DENSITY_HU,
    density_values=_DEFAULT_DENSITY_VALUES,
    material_thresholds=_DEFAULT_MATERIAL_THRESHOLDS,
    material_indices=_DEFAULT_MATERIAL_INDICES,
)
"""A runnable, scanner-independent default calibration; see the module docstring."""


def grid_from_hu(
    hu: npt.NDArray[np.float64],
    spacing: tuple[float, float, float],
    *,
    calibration: HounsfieldCalibration = DEFAULT_CALIBRATION,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> VoxelGrid:
    """Assemble a :class:`VoxelGrid` from a HU volume and a voxel ``spacing`` (cm).

    Pure NumPy: ``hu`` is a ``(nx, ny, nz)`` array of CT numbers already in the engine's
    axis order. The calibration maps it to per-voxel density and material; the grid
    validates the result (positive density, in-registry material).
    """
    hu = np.asarray(hu, dtype=np.float64)
    nx, ny, nz = hu.shape
    return VoxelGrid(
        shape=(nx, ny, nz),
        spacing=spacing,
        origin=origin,
        density=calibration.density(hu),
        material=calibration.material(hu).astype(np.int32),
    )


def grid_from_image(
    image: sitk.Image,
    *,
    calibration: HounsfieldCalibration = DEFAULT_CALIBRATION,
) -> VoxelGrid:
    """Convert a 3D SimpleITK CT image into a :class:`VoxelGrid`.

    Handles the frame mismatch between ITK and the engine: SimpleITK indexes ``(z, y, x)``
    and works in mm with the physical origin at voxel (0,0,0)'s *centre*, while the grid
    is ``(x, y, z)`` in cm with the origin at that voxel's lower *corner*. The image's
    intensities are taken to be Hounsfield units already (true for NIfTI/NRRD and for a
    rescale-applied DICOM series).

    Only axis-aligned CTs are supported: the direction cosine matrix must be diagonal.
    A negative diagonal entry (an LPS/RAS sign flip, common clinically) means that axis's
    physical coordinate decreases with index; the array is flipped and the origin
    recomputed so the grid is stored in increasing-coordinate order. An oblique
    (off-diagonal) orientation raises — resample to an axis-aligned grid first.
    """
    import SimpleITK as sitk

    if image.GetDimension() != 3:
        raise ValueError(f"expected a 3D CT image, got {image.GetDimension()}D")

    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    if not np.allclose(direction - np.diag(np.diagonal(direction)), 0.0, atol=1.0e-6):
        raise ValueError(
            "oblique CT orientation is not supported; resample to an axis-aligned grid first"
        )

    # SimpleITK array is (z, y, x); transpose to the engine's (x, y, z). Spacing, origin
    # and the direction diagonal are all in image (x, y, z) order, so they line up.
    hu = np.transpose(sitk.GetArrayFromImage(image).astype(np.float64), (2, 1, 0))
    spacing_mm = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin_mm = np.asarray(image.GetOrigin(), dtype=np.float64)
    signs = np.sign(np.diagonal(direction))

    first_center_mm = origin_mm.copy()
    for axis in range(3):
        if signs[axis] < 0.0:
            hu = np.flip(hu, axis=axis)
            # Physical coordinate fell with index, so the minimum-coordinate voxel was
            # the last one; after the flip it is index 0.
            first_center_mm[axis] = origin_mm[axis] - spacing_mm[axis] * (hu.shape[axis] - 1)

    lower_corner_mm = first_center_mm - 0.5 * spacing_mm
    ox, oy, oz = (lower_corner_mm / _MM_PER_CM).tolist()
    sx, sy, sz = (spacing_mm / _MM_PER_CM).tolist()
    return grid_from_hu(
        np.ascontiguousarray(hu),
        spacing=(sx, sy, sz),
        calibration=calibration,
        origin=(ox, oy, oz),
    )


def read_ct(
    path: str,
    *,
    calibration: HounsfieldCalibration = DEFAULT_CALIBRATION,
) -> VoxelGrid:
    """Read a CT image file into a :class:`VoxelGrid` (needs the ``pyradmc[ct]`` extra).

    Accepts anything SimpleITK reads as a single volume — NIfTI, NRRD, MetaImage, a
    single multi-frame DICOM. For a DICOM *series* (a directory of slices), read it with
    ``SimpleITK.ImageSeriesReader`` yourself and pass the image to :func:`grid_from_image`,
    so slice ordering and the rescale slope/intercept are handled explicitly. See
    :func:`grid_from_image` for the frame and orientation handling.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("read_ct needs the optional CT extra: pip install 'pyradmc[ct]'") from exc

    return grid_from_image(sitk.ReadImage(path), calibration=calibration)
