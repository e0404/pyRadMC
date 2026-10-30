"""Exterior air under a uniform magnetic field: the region and its safety distance.

World data for the exact-helix step of :func:`pyradmc.transport.electron.electron_steps`
and its Warp twin. In air the energy-limited step is metres and almost nothing physical
happens, yet under a field the gyration cap
(:data:`~pyradmc.transport.electron.LORENTZ_STEP_ANGLE_MAX`) makes a particle take about
31 substeps per gyration. The air cannot be removed, because electrons have to be able
to curve back into the body, but far from the body the curve need not be chopped up.

The exterior is the set of voxels below :data:`EXTERIOR_AIR_DENSITY_MAX` that are
face-connected to the grid boundary, so lungs and sealed cavities keep the current
stepping exactly. Each exterior voxel carries a **safety distance**: the smallest distance
from *any point of that voxel* to any non-exterior voxel or to the outside of the grid.
Path length bounds displacement, so a step of path ``s <= safety`` keeps the whole arc,
the hinge and the chord midpoints in the exterior, wherever in the voxel it starts.

The promise (:func:`safety_distance`) is exact rather than an estimate: the box-to-box
distance from a voxel to its nearest blocked voxel is computed, never a centre-to-centre
figure, and it is test-pinned against brute force and against random balls. Where the
value would be below ``min_safety_cm`` the map stores zero, which is the kernels' signal
to run the current stepping unchanged: behaviour at the body surface and at the grid
boundary is exactly that of a run without a map.

**This is data, not a flag (AGENTS.md 2.10).** The engines take an
:class:`ExteriorAirMap` next to the grid and the field; a run without one is the current
stepping, and a map is inert unless the field is uniform and non-zero
(:func:`resolve_exterior_safety`). Until the maintainer rules on the default, the map's
absence is what lets the two paths be compared on one grid. ``safety`` is a plain array
on purpose: a test can hand in an inflated map as a mutant without patching a kernel.

Host-side only: NumPy and SciPy, one source for both engines. The kernels read the array.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from pyradmc.geometry.field import FIELD_UNIFORM, MagneticField
from pyradmc.geometry.grid import VoxelGrid

__all__ = [
    "EXTERIOR_AIR_DENSITY_MAX",
    "EXTERIOR_MIN_SAFETY_CM",
    "SAFETY_GUARD_CM",
    "ExteriorAirMap",
    "exterior_region",
    "resolve_exterior_safety",
    "safety_distance",
]

EXTERIOR_AIR_DENSITY_MAX: float = 0.01
"""Upper density bound of an exterior voxel, in g/cm^3 (exclusive). Accuracy-defining.

About eight times dry air (1.205e-3), a decade below the lightest tissue this engine meets
(inflated lung near 0.1) and below immobilisation foam (near 0.03). The value decides what
may be stepped through without the gyration cap, so it errs low. A noisy CT air voxel above
it is simply not exterior: it blocks, and the air around it falls back to current stepping.
That failure costs time, never accuracy. Within the bound a long step charges its loss at
the start voxel's density and material (:func:`pyradmc.transport.electron.electron_steps`),
so the bound also limits how far that can be from the truth. Changing it needs the gates of
AGENTS.md section 8."""

EXTERIOR_MIN_SAFETY_CM: float = 0.5
"""Smallest safety at which a long step is taken, in cm. Accuracy-defining.

Below it the voxel stores zero and the current stepping runs unchanged, which is what
keeps the electron return at the body surface on the validated scheme: on a 3 mm grid the
two voxel layers next to the body (and next to the grid boundary) stay on the current
stepping. A smaller value buys more of the near-surface air, where the returning electrons
are, and moves the new path closer to the surface; that is a maintainer decision with the
gates of AGENTS.md section 8, not a tuning knob."""

SAFETY_GUARD_CM: float = 1.0e-3
"""Subtracted from every stored safety, in cm: the float32 margin.

Ten nudges (:data:`~pyradmc.transport.electron.BOUNDARY_NUDGE_CM`), hundreds of float32
ulps at the coordinates of a patient grid. It absorbs the rounding of the uploaded value,
of the position and of the path, so the geometric promise survives single precision."""


def exterior_region(
    grid: VoxelGrid, density_max: float = EXTERIOR_AIR_DENSITY_MAX
) -> npt.NDArray[np.bool_]:
    """Voxels below ``density_max`` that are face-connected to the grid boundary.

    Six-connectivity is the conservative reading: a corner touch is not a path. A channel
    that does reach the boundary (a trachea cut by the top slice) does make what it opens
    into exterior; that air is then stepped like any other, the safety distance keeps the
    steps inside it, and a narrow channel is below the minimum safety anyway.
    """
    light = np.asarray(grid.density) < density_max
    labels, _ = ndimage.label(light)
    faces = np.concatenate(
        [
            labels[0, :, :].ravel(),
            labels[-1, :, :].ravel(),
            labels[:, 0, :].ravel(),
            labels[:, -1, :].ravel(),
            labels[:, :, 0].ravel(),
            labels[:, :, -1].ravel(),
        ]
    )
    touching = np.unique(faces)
    touching = touching[touching != 0]
    return np.isin(labels, touching)


def safety_distance(
    region: npt.NDArray[np.bool_], spacing: tuple[float, float, float]
) -> npt.NDArray[np.float64]:
    r"""Exact distance from each region voxel's box to the nearest other box or the outside.

    Between voxels ``p`` and ``q`` the smallest distance over all their points is the gap

    .. math::

        g(p, q) = \sqrt{\sum_a \left(\max(|p_a - q_a| - 1, 0)\, h_a\right)^2}.

    Dilating the blocked set by the 3x3x3 box turns that into a plain centre distance: any
    blocked ``q`` has a dilated voxel ``q'`` one step nearer ``p`` per axis, at centre
    distance exactly ``g(p, q)``, and no dilated voxel is nearer than the gap of the blocked
    voxel that produced it. So a Euclidean distance transform of the complement of the
    dilation is :math:`\min_q g(p, q)`, with no triangle-inequality slack. The outside of
    the grid is a blocked layer of ghost voxels, whose gap from a voxel is that voxel's
    distance to the face. Region voxels touching anything blocked get zero.

    Parameters
    ----------
    region
        Boolean voxel mask; everything else blocks.
    spacing
        Voxel edge lengths in cm; anisotropy is honoured per axis.
    """
    blocked = np.pad(~np.asarray(region, dtype=bool), 1, constant_values=True)
    dilated = ndimage.binary_dilation(blocked, structure=np.ones((3, 3, 3), dtype=bool))
    distance = ndimage.distance_transform_edt(~dilated, sampling=spacing)
    return np.asarray(distance[1:-1, 1:-1, 1:-1], dtype=np.float64)


def _grid_digest(grid: VoxelGrid) -> str:
    """SHA-256 over what the map was derived from: geometry, density and material."""
    h = hashlib.sha256()
    h.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    h.update(np.asarray(grid.spacing, dtype=np.float64).tobytes())
    h.update(np.asarray(grid.origin, dtype=np.float64).tobytes())
    h.update(np.asarray(grid.density, dtype=np.float64).tobytes())
    h.update(np.asarray(grid.material, dtype=np.int64).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ExteriorAirMap:
    """Where a charged particle may take a long exact-helix step, and how long.

    Build with :meth:`from_grid`, or :meth:`from_region` for a caller's own region. A
    named approximation with the gates of AGENTS.md section 8: within a long step the
    deposition is not spatially resolved, and the loss is charged at the start voxel's
    density and material; see :func:`pyradmc.transport.electron.electron_steps`.

    Attributes
    ----------
    safety
        Per voxel, in cm: the longest path a step starting anywhere in the voxel may
        take, or zero for the current stepping. Stored read-only.
    grid_digest
        Digest of the grid it was derived from; the engines refuse a map of another grid.
    density_max
        The region's density bound in g/cm^3, or ``None`` for a caller-supplied region.
    min_safety_cm
        The minimum safety it was built with.
    """

    safety: npt.NDArray[np.float64]
    grid_digest: str
    density_max: float | None = EXTERIOR_AIR_DENSITY_MAX
    min_safety_cm: float = EXTERIOR_MIN_SAFETY_CM

    def __post_init__(self) -> None:
        """Store the safety array as a read-only, C-contiguous float64 copy."""
        safety = np.array(self.safety, dtype=np.float64, order="C", copy=True)
        if safety.ndim != 3:
            raise ValueError(f"safety shape {safety.shape} must be (nx, ny, nz)")
        safety.setflags(write=False)
        object.__setattr__(self, "safety", safety)

    @classmethod
    def from_region(
        cls,
        grid: VoxelGrid,
        region: npt.NDArray[np.bool_],
        min_safety_cm: float = EXTERIOR_MIN_SAFETY_CM,
        *,
        density_max: float | None = None,
    ) -> ExteriorAirMap:
        """Bound a given region of ``grid``. Everything outside the region blocks."""
        if min_safety_cm < 0.0:
            raise ValueError(f"min_safety_cm must be non-negative, got {min_safety_cm}")
        region = np.asarray(region, dtype=bool)
        if region.shape != tuple(grid.shape):
            raise ValueError(f"region shape {region.shape} != grid shape {grid.shape}")
        # The minimum is compared with the geometric distance, so a voxel exactly at it
        # qualifies on any grid; the guard then only shortens the step it stores.
        raw = safety_distance(region, grid.spacing)
        safety = np.where(
            (raw >= min_safety_cm) & (raw > SAFETY_GUARD_CM), raw - SAFETY_GUARD_CM, 0.0
        )
        return cls(
            safety=safety,
            grid_digest=_grid_digest(grid),
            density_max=density_max,
            min_safety_cm=min_safety_cm,
        )

    @classmethod
    def from_grid(
        cls,
        grid: VoxelGrid,
        density_max: float = EXTERIOR_AIR_DENSITY_MAX,
        min_safety_cm: float = EXTERIOR_MIN_SAFETY_CM,
    ) -> ExteriorAirMap:
        """Derive the exterior of ``grid`` (:func:`exterior_region`) and bound it."""
        return cls.from_region(
            grid, exterior_region(grid, density_max), min_safety_cm, density_max=density_max
        )

    def matches(self, grid: VoxelGrid) -> bool:
        """Whether this map was derived from a grid with these very numbers."""
        return self.grid_digest == _grid_digest(grid)

    @property
    def digest(self) -> str:
        """SHA-256 over the safety bytes and the grid digest; a tampered map differs."""
        h = hashlib.sha256()
        h.update(self.safety.tobytes())
        h.update(self.grid_digest.encode("ascii"))
        return h.hexdigest()

    @property
    def provenance(self) -> str:
        """One line naming the threshold, the minimum safety and the digest."""
        return (
            f"exterior air rho<{self.density_max} min_safety={self.min_safety_cm} cm "
            f"sha256={self.digest[:12]}"
        )


def resolve_exterior_safety(
    air: ExteriorAirMap | None, field: MagneticField | None
) -> npt.NDArray[np.float64] | None:
    """Return the safety array a run actually uses, or ``None`` for current stepping.

    Both engines ask here, so they cannot disagree. A map is inert with no field or a zero
    field, where there is no gyration cap to lift and the field-free path must stay byte
    identical, and with a non-uniform map, where a long step would skip the field's
    variation (no field-gradient limiter exists; see
    :class:`~pyradmc.geometry.field.MagneticField`).
    """
    if air is None or field is None:
        return None
    if field.mode != FIELD_UNIFORM or field.max_magnitude <= 0.0:
        return None
    return air.safety
