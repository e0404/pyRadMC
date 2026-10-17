"""Static external magnetic field on a rectilinear sample grid.

World data for transport in a static magnetic flux density ``B(x, y, z)`` (MR-linac
geometries). The field is **data, not a flag** (AGENTS.md 2.10): the only way to
transport without one is not to supply one, and an all-zero field is still a field
(it keeps its mode, so the zero-vector guard of the step loop stays exercised by the
B = 0 bit-identity test; see :data:`FIELD_NONE`).

One data model, three representations, told apart by a derived integer ``mode``
that crosses into kernels unchanged (:class:`~pyradmc.data.interface.PhotonProcess`
style):

* :data:`FIELD_UNIFORM` — a single sample, shape ``(1, 1, 1)``, infinite extent,
  zero array loads per lookup (:meth:`MagneticField.uniform`);
* :data:`FIELD_NEAREST` — one vector per cell, one load; the "vector per transport
  voxel" case built by :meth:`MagneticField.on_grid`, exact per voxel-capped substep;
* :data:`FIELD_TRILINEAR` — an arbitrary rectilinear map, eight loads, interpolated
  between cell centres.

Samples sit at **cell centres** ``origin + (i + 0.5) * spacing`` of half-open cells
with the :class:`~pyradmc.geometry.grid.VoxelGrid` lower-corner convention, so an
``on_grid`` field is literally the transport grid's numbers. Components are in the
engine frame, in tesla; positions in cm. A gridded field must *cover* the transport
grid (:meth:`MagneticField.covers`; the engines refuse one that does not), so the
flat clamp at the sample-grid edge only ever guards float round-off, exactly as the
table lookups do (:mod:`pyradmc.data.tables`). Outside the transport grid the world
stays vacuum and field-free.

The lookup :func:`field_at` is written in the pure-scalar style (AGENTS.md 2.5) so
the Warp physics loader compiles this very source to ``@wp.func``. The sample array
is a :data:`~pyradmc.data.handles.Vector3Field` handle: ``(nx, ny, nz, 3)`` NumPy on
the host, ``wp.array3d(dtype=wp.vec3)`` in kernels. The one idiom that is valid on
both targets is the :data:`~pyradmc.data.handles.Table3D` partial-index trick:
``v = values[i, j, k]`` (a length-3 view on the host, a ``vec3`` copy in Warp)
followed by ``float(v[0])`` per component — verified to compile and agree on
``cpu`` and ``cuda:0``. Never index ``values[i, j, k, c]`` with four subscripts:
that is a fourth array dimension on the host and a compile error in Warp.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from pyradmc.data.handles import Vector3Field
from pyradmc.geometry.grid import VoxelGrid, point_axis_index

__all__ = [
    "FIELD_NEAREST",
    "FIELD_NONE",
    "FIELD_TRILINEAR",
    "FIELD_UNIFORM",
    "MagneticField",
    "field_at",
]

FIELD_NONE, FIELD_UNIFORM, FIELD_NEAREST, FIELD_TRILINEAR = 0, 1, 2, 3
"""Field modes. ``FIELD_NONE`` means *no field object was given* and is the
engines' default; it is never the mode of a :class:`MagneticField` (an all-zero
field keeps the mode of its representation) and is never passed to
:func:`field_at`, whose callers branch on it first so the field-free step loop
executes no new floating-point operation."""

_INTERPOLATIONS = ("trilinear", "nearest")
_MODE_NAMES = {FIELD_UNIFORM: "uniform", FIELD_NEAREST: "nearest", FIELD_TRILINEAR: "trilinear"}


@dataclass(frozen=True)
class MagneticField:
    """Static magnetic flux density ``B(x, y, z)`` on a rectilinear sample grid, tesla.

    Samples sit at *cell centres* ``origin + (i + 0.5) * spacing`` of half-open
    cells with the :class:`~pyradmc.geometry.grid.VoxelGrid` lower-corner
    convention, so :meth:`on_grid` is literally the transport grid's numbers.
    Components are in the engine frame (the physical frame of whatever produced
    the grid; a gantry rotation never touches the field).

    ``values`` is stored as a C-contiguous ``float64`` copy of whatever was passed,
    so the digest and the device upload have one canonical byte layout.

    Attributes
    ----------
    values
        Field components per sample, shape ``(nx, ny, nz, 3)``, tesla, finite.
    spacing
        Sample-cell edge lengths in cm, ``> 0``.
    origin
        Lower corner of sample cell ``(0, 0, 0)``, in cm.
    interpolation
        ``"trilinear"`` (between cell centres, flat beyond the outermost centres)
        or ``"nearest"`` (constant per cell). A data-layout choice, not physics: the
        map is what it is; this says how it is read between its samples. A single-
        cell map must be ``"nearest"`` — that is the uniform representation, built
        by :meth:`uniform`, and it has exactly one canonical form.
    """

    values: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    interpolation: str = "trilinear"

    def __post_init__(self) -> None:
        """Validate the sample array, the geometry and the interpolation name."""
        values = np.ascontiguousarray(self.values, dtype=np.float64)
        if values.ndim != 4 or values.shape[3] != 3:
            raise ValueError(f"values shape {values.shape} must be (nx, ny, nz, 3)")
        if any(n < 1 for n in values.shape[:3]):
            raise ValueError(f"empty field shape {values.shape[:3]}")
        if not np.all(np.isfinite(values)):
            raise ValueError("non-finite field values")
        if len(self.spacing) != 3 or not all(s > 0.0 for s in self.spacing):
            raise ValueError(f"non-positive spacing {self.spacing}")
        if len(self.origin) != 3 or not all(math.isfinite(o) for o in self.origin):
            raise ValueError(f"non-finite origin {self.origin}")
        if self.interpolation not in _INTERPOLATIONS:
            raise ValueError(
                f"interpolation {self.interpolation!r} is not one of {_INTERPOLATIONS}"
            )
        if values.shape[:3] == (1, 1, 1) and self.interpolation != "nearest":
            raise ValueError(
                "a single-cell map is the uniform field; build it with "
                "MagneticField.uniform (interpolation is 'nearest')"
            )
        object.__setattr__(self, "values", values)

    # -- constructors -------------------------------------------------------

    @classmethod
    def uniform(cls, b: tuple[float, float, float]) -> MagneticField:
        """Build a spatially constant field ``b = (bx, by, bz)`` in tesla, infinite extent.

        One sample, shape ``(1, 1, 1)``, ``"nearest"``; the lookup returns the
        constants without reading the array (:data:`FIELD_UNIFORM`).
        """
        values = np.asarray(b, dtype=np.float64).reshape(1, 1, 1, 3)
        return cls(values=values, spacing=(1.0, 1.0, 1.0), interpolation="nearest")

    @classmethod
    def on_grid(cls, grid: VoxelGrid, values: np.ndarray) -> MagneticField:
        """One vector per transport voxel, read as constant within the voxel.

        Copies the grid's origin and spacing bit-exactly, so the lookup indexes the
        very cell :meth:`~pyradmc.geometry.grid.VoxelGrid.voxel_index` reports.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (*grid.shape, 3):
            raise ValueError(f"values shape {values.shape} != grid shape {grid.shape} + (3,)")
        return cls(values=values, spacing=grid.spacing, origin=grid.origin, interpolation="nearest")

    # -- geometry -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of sample cells along (x, y, z)."""
        nx, ny, nz = self.values.shape[:3]
        return (int(nx), int(ny), int(nz))

    @property
    def upper_corner(self) -> tuple[float, float, float]:
        """Upper outer corner of the sample grid, in cm (outside, half-open).

        The same arithmetic as :attr:`VoxelGrid.upper_corner`, so an ``on_grid``
        field's corner equals the transport grid's exactly.
        """
        return (
            self.origin[0] + self.shape[0] * self.spacing[0],
            self.origin[1] + self.shape[1] * self.spacing[1],
            self.origin[2] + self.shape[2] * self.spacing[2],
        )

    @property
    def is_uniform(self) -> bool:
        """Whether this is the single-sample (spatially constant) representation."""
        return self.shape == (1, 1, 1)

    @property
    def max_magnitude(self) -> float:
        """Largest ``|B|`` over the samples, tesla (provenance; step-cap sizing)."""
        return float(np.sqrt(np.einsum("ijkc,ijkc->ijk", self.values, self.values)).max())

    @property
    def mode(self) -> int:
        """Kernel-side representation: :data:`FIELD_UNIFORM`, ``_NEAREST`` or ``_TRILINEAR``."""
        if self.is_uniform:
            return FIELD_UNIFORM
        return FIELD_NEAREST if self.interpolation == "nearest" else FIELD_TRILINEAR

    def covers(self, grid: VoxelGrid) -> bool:
        """Whether the transport grid's bounding box lies within the sample grid.

        A uniform field covers everything. A gridded field covers the grid when,
        on every axis, ``origin <= grid.origin`` and ``grid.upper_corner <=
        upper_corner``: every transport position (``< grid.upper_corner``, half-open)
        is then inside the half-open sample box, and the edge clamp of
        :func:`field_at` is reached only by round-off.
        """
        if self.is_uniform:
            return True
        lo, hi = self.origin, self.upper_corner
        g_lo, g_hi = grid.origin, grid.upper_corner
        return all(lo[a] <= g_lo[a] and g_hi[a] <= hi[a] for a in range(3))

    # -- lookup -------------------------------------------------------------

    def sample(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """``B`` in tesla at an engine-frame position, via :func:`field_at`."""
        b0 = self.values[0, 0, 0]
        return field_at(
            self.values,
            self.mode,
            float(b0[0]),
            float(b0[1]),
            float(b0[2]),
            self.origin[0],
            self.origin[1],
            self.origin[2],
            self.spacing[0],
            self.spacing[1],
            self.spacing[2],
            self.shape[0],
            self.shape[1],
            self.shape[2],
            x,
            y,
            z,
        )

    # -- identity -----------------------------------------------------------

    @property
    def digest(self) -> str:
        """SHA-256 over the sample bytes, shape, spacing, origin and interpolation."""
        h = hashlib.sha256()
        h.update(np.asarray(self.shape, dtype=np.int64).tobytes())
        h.update(np.asarray(self.spacing, dtype=np.float64).tobytes())
        h.update(np.asarray(self.origin, dtype=np.float64).tobytes())
        h.update(self.interpolation.encode("ascii"))
        h.update(self.values.tobytes())
        return h.hexdigest()

    @property
    def provenance(self) -> str:
        """One line naming the representation, the shape, ``|B|max`` and the digest.

        Carried in :class:`~pyradmc.backends.results.RunProvenance` so a result
        states the field it was computed in.
        """
        if self.is_uniform:
            bx, by, bz = (float(c) for c in self.values[0, 0, 0])
            return f"uniform B=({bx:g}, {by:g}, {bz:g}) T"
        nx, ny, nz = self.shape
        return (
            f"{_MODE_NAMES[self.mode]} {nx}x{ny}x{nz} |B|max={self.max_magnitude:.4g} T "
            f"sha256={self.digest[:12]}"
        )


# -- kernel-side lookup (single-source; compiled to @wp.func for Warp) ----------


def _lower_cell(t: float, n: int) -> int:
    """Lower bracketing cell of a centre coordinate ``t`` (in cells), clamped to ``[0, n-1]``.

    The same floor-then-cast as :func:`~pyradmc.geometry.grid.point_axis_index`
    (host ``math.floor`` returns int; the Warp compile gets a float32 floor, and
    the cast is what makes both an index).
    """
    i = int(math.floor(t))  # noqa: RUF046
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return i


def _lerp(a: float, b: float, w: float) -> float:
    """Return ``a * (1 - w) + b * w``, the one accumulation form used by every lerp here."""
    return a * (1.0 - w) + b * w


def field_at(
    values: Vector3Field,
    mode: int,
    b0x: float,
    b0y: float,
    b0z: float,
    ox: float,
    oy: float,
    oz: float,
    sx: float,
    sy: float,
    sz: float,
    nx: int,
    ny: int,
    nz: int,
    x: float,
    y: float,
    z: float,
) -> tuple[float, float, float]:
    """``B`` (tesla) at an engine-frame position; the caller guarantees ``mode != FIELD_NONE``.

    ``mode`` selects the representation (:data:`FIELD_UNIFORM` returns
    ``(b0x, b0y, b0z)`` without touching ``values``; :data:`FIELD_NEAREST` reads
    the cell containing the position with the transport grid's own index
    arithmetic; :data:`FIELD_TRILINEAR` interpolates between the eight nearest
    cell centres). ``FIELD_NONE`` is never passed: the step loops branch on it
    before calling, so the field-free path executes no new floating-point
    operation.

    Clamps flat at the sample-grid edge — beyond the outermost centres in
    trilinear mode, outside the box in nearest mode. The engines refuse a map
    that does not cover the transport grid (:meth:`MagneticField.covers`), so
    only round-off reaches the clamp, as with the energy tables
    (:func:`~pyradmc.data.tables.lookup_loglinear_1d`). A degenerate axis
    (``n == 1``) reads the same cell twice, so the map is constant along it.

    The accumulation order is fixed (z, then y, then x, per component) so the
    float64 host and the float32 kernel differ only by precision.

    Parameters
    ----------
    values
        Sample handle, ``(nx, ny, nz)`` of 3-vectors (see module docstring).
    mode
        :data:`FIELD_UNIFORM`, :data:`FIELD_NEAREST` or :data:`FIELD_TRILINEAR`.
    b0x, b0y, b0z
        The constant returned in uniform mode (the single sample's components).
    ox, oy, oz, sx, sy, sz, nx, ny, nz
        Sample-grid origin (cm), spacing (cm) and cell counts.
    x, y, z
        Query position, cm, engine frame.
    """
    if mode == FIELD_UNIFORM:
        return b0x, b0y, b0z
    if mode == FIELD_NEAREST:
        ix = point_axis_index(x, ox, sx, nx)
        iy = point_axis_index(y, oy, sy, ny)
        iz = point_axis_index(z, oz, sz, nz)
        v = values[ix, iy, iz]
        return float(v[0]), float(v[1]), float(v[2])

    # Trilinear between cell centres: centre coordinates in cells, lower cell,
    # upper cell (the same cell on a degenerate or clamped axis), weights in [0, 1].
    tx = (x - ox) / sx - 0.5
    ty = (y - oy) / sy - 0.5
    tz = (z - oz) / sz - 0.5
    ix = _lower_cell(tx, nx)
    iy = _lower_cell(ty, ny)
    iz = _lower_cell(tz, nz)
    ix1 = min(ix + 1, nx - 1)
    iy1 = min(iy + 1, ny - 1)
    iz1 = min(iz + 1, nz - 1)
    wx = min(max(tx - float(ix), 0.0), 1.0)
    wy = min(max(ty - float(iy), 0.0), 1.0)
    wz = min(max(tz - float(iz), 0.0), 1.0)

    v000 = values[ix, iy, iz]
    v001 = values[ix, iy, iz1]
    v010 = values[ix, iy1, iz]
    v011 = values[ix, iy1, iz1]
    v100 = values[ix1, iy, iz]
    v101 = values[ix1, iy, iz1]
    v110 = values[ix1, iy1, iz]
    v111 = values[ix1, iy1, iz1]

    bx = _lerp(
        _lerp(
            _lerp(float(v000[0]), float(v001[0]), wz), _lerp(float(v010[0]), float(v011[0]), wz), wy
        ),
        _lerp(
            _lerp(float(v100[0]), float(v101[0]), wz), _lerp(float(v110[0]), float(v111[0]), wz), wy
        ),
        wx,
    )
    by = _lerp(
        _lerp(
            _lerp(float(v000[1]), float(v001[1]), wz), _lerp(float(v010[1]), float(v011[1]), wz), wy
        ),
        _lerp(
            _lerp(float(v100[1]), float(v101[1]), wz), _lerp(float(v110[1]), float(v111[1]), wz), wy
        ),
        wx,
    )
    bz = _lerp(
        _lerp(
            _lerp(float(v000[2]), float(v001[2]), wz), _lerp(float(v010[2]), float(v011[2]), wz), wy
        ),
        _lerp(
            _lerp(float(v100[2]), float(v101[2]), wz), _lerp(float(v110[2]), float(v111[2]), wz), wy
        ),
        wx,
    )
    return bx, by, bz
