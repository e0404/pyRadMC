"""Beam-limiting devices: jaws and MLC as parametric ray-attenuation geometry.

The devices live in a local **beam frame**: origin at the focal spot, ``w`` the beam
axis increasing downstream, ``u`` the leaf-travel direction, ``v`` the leaf-width
direction (a right-handed orthonormal triad). Everything the devices store — slab
extents ``z_top < z_bottom`` along ``w``, jaw edges, leaf tip positions — is a
**physical coordinate at the device mid-plane** ``z_mid = (z_top + z_bottom) / 2``,
in cm. A treatment plan states field settings projected to the isocenter plane;
:func:`project_between_planes` converts them (divergent scaling through the focal
point). Like the divergent fan sources, the frame is given directly in the engine
frame ((x, y, z) cm, lower-corner origin): gantry/couch rotation stays out of the
core — an adapter rotates the frame axes, exactly as it rotates a source.

The geometry primitive is :meth:`~JawPair.path_lengths` — the length of the
chord(s) a straight ray cuts through the device material — from which callers build
deterministic attenuation (``exp(-mu * t)``) or a pre-solve interaction depth. Path
lengths use the **full-line convention**: intersections are computed along the
entire line, with no ``t >= 0`` clamp, because the caller guarantees the physical
trajectory traversed the devices. That holds from both directions this module is
used in — a focal-spot source emits *upstream of* the devices, the planar exit
source emits *downstream of* them. (:meth:`BeamLimitingStack.path_lengths` offers
``from_origin=True`` for the pre-solve's scattered photons, which start mid-stack.)

Stated v1 approximations (each also noted where it bites):

- Jaw edges are straight (unfocused); a focused-edge option is deferred.
- MLC leaves have rounded tips (exact circular chords) but **no tongue-and-groove
  step and no interleaf gap**: adjacent leaves tile the ``v`` axis exactly, so
  interleaf leakage is absent by construction. Divergent (focused) leaf *sides*
  are likewise deferred; the ``v`` strips are parallel-sided.
- Banks are laterally unbounded: jaws extend to infinity away from their edge and
  across the field, the outer MLC leaf edges end the bank. The orthogonal device
  provides the physical lateral bound, as it does in a real head.
- Devices are homogeneous: one registry material and one density each. Real
  heavy-alloy leaves are modelled as pure tungsten at the alloy density (see
  :data:`pyRadMC.data.materials.TUNGSTEN`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MLC",
    "BeamFrame",
    "BeamLimitingStack",
    "JawPair",
    "project_between_planes",
]

_F64 = NDArray[np.float64]

_ORTHONORMAL_TOLERANCE = 1.0e-9


def _as_unit(vector: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be a nonzero vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


@dataclass(frozen=True)
class BeamFrame:
    """The local beam frame: origin at the focal spot, orthonormal (u, v, w) axes.

    ``w`` points downstream (source to patient), ``u`` is the leaf-travel
    direction, ``v`` the leaf-width direction. Axes are given in the engine frame
    and normalized on construction; they must form a right-handed orthonormal
    triad (``u x v = w``), because a mirrored frame would silently swap the
    ``+u``/``-u`` leaf banks.
    """

    origin: tuple[float, float, float]
    u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    w_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        """Normalize the axes and validate the right-handed orthonormal triad."""
        u = _as_unit(self.u_axis, "u_axis")
        v = _as_unit(self.v_axis, "v_axis")
        w = _as_unit(self.w_axis, "w_axis")
        object.__setattr__(self, "u_axis", u)
        object.__setattr__(self, "v_axis", v)
        object.__setattr__(self, "w_axis", w)
        axes = np.array([u, v, w])
        if not np.allclose(axes @ axes.T, np.eye(3), atol=_ORTHONORMAL_TOLERANCE):
            raise ValueError("beam frame axes must be orthonormal")
        if float(np.dot(np.cross(axes[0], axes[1]), axes[2])) < 1.0 - 1.0e-9:
            raise ValueError("beam frame axes must be right-handed (u x v = w)")

    def to_local(self, points: _F64, directions: _F64) -> tuple[_F64, _F64]:
        """Map engine-frame points and directions into the beam frame (vectorized).

        Points are translated and rotated; directions are rotated only. Both
        arguments are ``(n, 3)`` arrays; two ``(n, 3)`` arrays are returned.
        """
        rotation = np.array([self.u_axis, self.v_axis, self.w_axis]).T
        p_local = (np.asarray(points, dtype=np.float64) - np.asarray(self.origin)) @ rotation
        d_local = np.asarray(directions, dtype=np.float64) @ rotation
        return p_local, d_local


def project_between_planes(coordinate: float | _F64, z_from: float, z_to: float) -> float | _F64:
    """Scale an in-plane coordinate between two planes through the focal point.

    Divergent (projective) scaling: a point at transverse coordinate ``c`` on the
    plane ``w = z_from`` lies on the focal-point ray that crosses ``w = z_to`` at
    ``c * z_to / z_from``. This converts plan settings stated at the isocenter
    plane (e.g. leaf tips at SAD) to physical device-mid-plane coordinates.
    """
    if z_from == 0.0:
        raise ValueError("the source plane may not contain the focal point (z_from = 0)")
    scale = z_to / z_from
    if isinstance(coordinate, np.ndarray):
        return coordinate * scale
    return float(coordinate) * scale


# --- ray/interval primitives ------------------------------------------------------
#
# Intervals along the ray parameter t (cm, since directions are unit vectors) are
# (lo, hi) array pairs; empty intervals are encoded (+inf, -inf) so that interval
# intersection stays a plain elementwise max/min with no branching.


def _slab_interval(p: _F64, d: _F64, lo: float, hi: float) -> tuple[_F64, _F64]:
    """Give the t-interval where ``lo <= p + t*d <= hi`` (full-line convention)."""
    safe = np.where(d == 0.0, 1.0, d)
    t_a = (lo - p) / safe
    t_b = (hi - p) / safe
    t_lo = np.minimum(t_a, t_b)
    t_hi = np.maximum(t_a, t_b)
    inside = (p >= lo) & (p <= hi)
    t_lo = np.where(d == 0.0, np.where(inside, -np.inf, np.inf), t_lo)
    t_hi = np.where(d == 0.0, np.where(inside, np.inf, -np.inf), t_hi)
    return t_lo, t_hi


def _halfspace_interval(p: _F64, d: _F64, edge: float, *, above: bool) -> tuple[_F64, _F64]:
    """Give the t-interval where ``p + t*d >= edge`` (``<=`` for ``above=False``)."""
    safe = np.where(d == 0.0, 1.0, d)
    t_edge = (edge - p) / safe
    # For d > 0 the ray enters {q >= edge} at t_edge; for d < 0 it leaves there.
    opens_up = (d > 0.0) == above
    t_lo = np.where(opens_up, t_edge, -np.inf)
    t_hi = np.where(opens_up, np.inf, t_edge)
    inside = (p >= edge) if above else (p <= edge)
    t_lo = np.where(d == 0.0, np.where(inside, -np.inf, np.inf), t_lo)
    t_hi = np.where(d == 0.0, np.where(inside, np.inf, -np.inf), t_hi)
    return t_lo, t_hi


def _intersect(a: tuple[_F64, _F64], b: tuple[_F64, _F64]) -> tuple[_F64, _F64]:
    return np.maximum(a[0], b[0]), np.minimum(a[1], b[1])


def _disc_interval(
    p_q: _F64, d_q: _F64, p_w: _F64, d_w: _F64, center_q: float, center_w: float, radius: float
) -> tuple[_F64, _F64]:
    """Give the t-interval where the ray is inside a disc in the (q, w) plane.

    The disc is the cross-section of the leaf-tip cylinder (axis parallel to the
    remaining coordinate). Solving ``|r(t) - c|^2 <= R^2`` gives the quadratic
    ``a t^2 + b t + c0 <= 0``; a tangent ray (zero discriminant) has measure-zero
    overlap and counts as a miss.
    """
    off_q = p_q - center_q
    off_w = p_w - center_w
    a = d_q * d_q + d_w * d_w
    b = 2.0 * (d_q * off_q + d_w * off_w)
    c0 = off_q * off_q + off_w * off_w - radius * radius
    discriminant = b * b - 4.0 * a * c0
    safe_a = np.where(a == 0.0, 1.0, a)
    half_width = np.sqrt(np.maximum(discriminant, 0.0))
    hit = (a > 0.0) & (discriminant > 0.0)
    # a == 0: the ray runs parallel to the cylinder axis — fully inside or outside.
    along_axis_inside = (a == 0.0) & (c0 <= 0.0)
    miss_lo = np.where(along_axis_inside, -np.inf, np.inf)
    miss_hi = np.where(along_axis_inside, np.inf, -np.inf)
    t_lo = np.where(hit, (-b - half_width) / (2.0 * safe_a), miss_lo)
    t_hi = np.where(hit, (-b + half_width) / (2.0 * safe_a), miss_hi)
    return t_lo, t_hi


def _length(interval: tuple[_F64, _F64]) -> _F64:
    lo, hi = interval
    # Empty intervals are (+inf, -inf): hi - lo = -inf, clamped to zero. A genuine
    # (-inf, +inf) or half-infinite hit has infinite length (transmission zero).
    return np.maximum(0.0, hi - lo)


@dataclass(frozen=True)
class JawPair:
    """One pair of straight-edged jaw blocks limiting one transverse coordinate.

    The pair is a slab ``z_top <= w <= z_bottom`` (local frame, cm downstream of
    the focal spot) holding two laterally unbounded blocks: the ``-axis`` block
    occupies ``q <= edge_neg`` and the ``+axis`` block ``q >= edge_pos``, where
    ``q`` is the local ``u`` or ``v`` coordinate selected by ``axis``. Edges are
    physical coordinates at the device mid-plane; because the edge faces are
    parallel to ``w`` (straight, unfocused jaws — the v1 approximation stated in
    the module docstring), the mid-plane edge is also the edge at every depth.

    ``edge_neg <= edge_pos`` (the blocks may touch — a closed pair — but never
    interpenetrate). ``density`` is in g/cm^3 and ``material`` is a registry index
    (:data:`pyRadMC.data.materials.TUNGSTEN` for a real jaw).
    """

    axis: str
    z_top: float
    z_bottom: float
    edge_neg: float
    edge_pos: float
    material: int
    density: float

    def __post_init__(self) -> None:
        """Validate the axis choice, slab extents, edge ordering, and material data."""
        if self.axis not in ("u", "v"):
            raise ValueError(f"axis must be 'u' or 'v', got {self.axis!r}")
        if self.z_top <= 0.0:
            raise ValueError(f"z_top must be downstream of the focal spot, got {self.z_top}")
        if self.z_bottom <= self.z_top:
            raise ValueError(f"z_bottom must exceed z_top, got {self.z_bottom} <= {self.z_top}")
        if self.edge_neg > self.edge_pos:
            raise ValueError(f"edges cross: edge_neg {self.edge_neg} > edge_pos {self.edge_pos}")
        if self.material < 0:
            raise ValueError(f"material index must be non-negative, got {self.material}")
        if self.density <= 0.0:
            raise ValueError(f"density must be positive, got {self.density}")

    def path_lengths(self, p_local: _F64, d_local: _F64, *, from_origin: bool = False) -> _F64:
        """Material chord length of each ray through the pair (full line, cm).

        ``p_local``/``d_local`` are ``(n, 3)`` beam-frame origins and unit
        directions; returns ``(n,)`` lengths. A ray parallel to the slab inside a
        block returns ``inf`` (its transmission is zero). ``from_origin=True``
        restricts to the forward half-line ``t >= 0`` (a mid-stack start).
        """
        p = np.asarray(p_local, dtype=np.float64)
        d = np.asarray(d_local, dtype=np.float64)
        q_index = 0 if self.axis == "u" else 1
        slab = _slab_interval(p[:, 2], d[:, 2], self.z_top, self.z_bottom)
        if from_origin:
            slab = np.maximum(slab[0], 0.0), slab[1]
        block_pos = _halfspace_interval(p[:, q_index], d[:, q_index], self.edge_pos, above=True)
        block_neg = _halfspace_interval(p[:, q_index], d[:, q_index], self.edge_neg, above=False)
        return _length(_intersect(slab, block_pos)) + _length(_intersect(slab, block_neg))


@dataclass(frozen=True)
class MLC:
    """A multi-leaf collimator: paired leaf banks with rounded tips.

    Leaves travel along the local ``u`` axis and tile the ``v`` axis in strips
    bounded by ``leaf_edges_v`` (``n_pairs + 1`` strictly increasing mid-plane
    coordinates, cm). Pair ``i`` occupies the strip ``[leaf_edges_v[i],
    leaf_edges_v[i+1])``; its ``-u`` leaf tip sits at ``tips_neg[i]`` and its
    ``+u`` leaf tip at ``tips_pos[i]`` — the tip coordinate is the **apex** of the
    rounded end, the most-protruding physical point, which lies on the device
    mid-plane. Opposing tips may touch (a closed pair) but never cross.

    **Rounded tips, exactly.** The leaf end is a cylinder of radius ``tip_radius``
    whose axis is parallel to ``v`` on the mid-plane: in the ``(u, w)`` plane the
    ``+u`` leaf cross-section is::

        slab(z_top <= w <= z_bottom)  intersected with
        ( {u >= tip + R}  union  disc((tip + R, z_mid), R) )

    (mirrored for the ``-u`` bank), so a ray through the tip region cuts the exact
    circular chord and the radiation-field edge softens the way a real rounded end
    does. ``tip_radius >= (z_bottom - z_top) / 2`` keeps the arc spanning the full
    leaf height (real MLCs have tip radii of ~8-16 cm against ~6-7 cm leaf height;
    see Boyer & Li, Med. Phys. 24, 757 (1997), doi:10.1118/1.597996, for the
    rounded-end field-offset analysis this geometry reproduces). The tip
    coordinate here is the *physical* apex: any light-field/radiation-field
    calibration offset a vendor states is the caller's to apply before
    construction (:func:`project_between_planes` converts isocenter-plane plans).

    Deferred, stated: no tongue-and-groove step, no interleaf gap (adjacent
    strips tile ``v`` exactly — interleaf leakage is absent by construction), and
    parallel (unfocused) leaf sides. The bank ends at the outer strip edges; the
    orthogonal jaws provide the lateral bound beyond them.
    """

    z_top: float
    z_bottom: float
    leaf_edges_v: tuple[float, ...]
    tips_neg: tuple[float, ...]
    tips_pos: tuple[float, ...]
    tip_radius: float
    material: int
    density: float

    def __post_init__(self) -> None:
        """Validate slab extents, tip-arc coverage, leaf tiling, and material data."""
        if self.z_top <= 0.0:
            raise ValueError(f"z_top must be downstream of the focal spot, got {self.z_top}")
        if self.z_bottom <= self.z_top:
            raise ValueError(f"z_bottom must exceed z_top, got {self.z_bottom} <= {self.z_top}")
        height = self.z_bottom - self.z_top
        if self.tip_radius < height / 2.0:
            raise ValueError(
                f"tip_radius {self.tip_radius} must cover the half-height {height / 2.0} "
                "or the arc does not span the leaf face"
            )
        if len(self.leaf_edges_v) < 2:
            raise ValueError("leaf_edges_v needs at least two edges (one pair)")
        edges = np.asarray(self.leaf_edges_v, dtype=np.float64)
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("leaf_edges_v must be strictly increasing")
        n_pairs = len(self.leaf_edges_v) - 1
        if len(self.tips_neg) != n_pairs:
            raise ValueError(f"tips_neg has {len(self.tips_neg)} entries for {n_pairs} pairs")
        if len(self.tips_pos) != n_pairs:
            raise ValueError(f"tips_pos has {len(self.tips_pos)} entries for {n_pairs} pairs")
        for i, (neg, pos) in enumerate(zip(self.tips_neg, self.tips_pos, strict=True)):
            if neg > pos:
                raise ValueError(f"pair {i}: tip positions cross ({neg} > {pos})")
        if self.material < 0:
            raise ValueError(f"material index must be non-negative, got {self.material}")
        if self.density <= 0.0:
            raise ValueError(f"density must be positive, got {self.density}")

    @property
    def n_pairs(self) -> int:
        """Number of leaf pairs (one fewer than the strip edges)."""
        return len(self.leaf_edges_v) - 1

    def path_lengths(self, p_local: _F64, d_local: _F64, *, from_origin: bool = False) -> _F64:
        """Material chord length of each ray through both banks (full line, cm).

        Per leaf, with ``S`` the slab-and-strip intersection, ``H`` the flat-body
        half-space and ``D`` the tip disc, the chord is ``len(S & H) + len(S & D)
        - len(S & H & D)`` (inclusion-exclusion over the union), so steep rays
        that leave the disc and re-enter the flat body are still exact.
        ``from_origin=True`` restricts to the forward half-line ``t >= 0``.
        """
        p = np.asarray(p_local, dtype=np.float64)
        d = np.asarray(d_local, dtype=np.float64)
        p_u, p_v, p_w = p[:, 0], p[:, 1], p[:, 2]
        d_u, d_v, d_w = d[:, 0], d[:, 1], d[:, 2]
        z_mid = (self.z_top + self.z_bottom) / 2.0
        slab = _slab_interval(p_w, d_w, self.z_top, self.z_bottom)
        if from_origin:
            slab = np.maximum(slab[0], 0.0), slab[1]
        total = np.zeros(p.shape[0], dtype=np.float64)
        for i in range(self.n_pairs):
            strip = _slab_interval(p_v, d_v, self.leaf_edges_v[i], self.leaf_edges_v[i + 1])
            in_strip = _intersect(slab, strip)
            for tip, sign in ((self.tips_pos[i], 1.0), (self.tips_neg[i], -1.0)):
                center_u = tip + sign * self.tip_radius
                body = _halfspace_interval(p_u, d_u, center_u, above=sign > 0.0)
                disc = _disc_interval(p_u, d_u, p_w, d_w, center_u, z_mid, self.tip_radius)
                total += (
                    _length(_intersect(in_strip, body))
                    + _length(_intersect(in_strip, disc))
                    - _length(_intersect(_intersect(in_strip, body), disc))
                )
        return total


@dataclass(frozen=True)
class BeamLimitingStack:
    """An ordered stack of beam-limiting devices sharing one beam frame.

    Devices are ordered ascending in ``z_top`` and must not overlap along ``w``
    (they are physically stacked; the pre-solve also relies on tracing them
    sequentially). :meth:`path_lengths` takes **engine-frame** rays, maps them
    into the beam frame once, and returns one chord-length column per device —
    devices never interact geometrically, so the stack is exactly the
    concatenation of its parts. Rotating the frame and the rays together leaves
    every length unchanged (the adapter contract: gantry/couch rotation happens
    outside the core, here as much as for the sources).
    """

    frame: BeamFrame
    devices: tuple[JawPair | MLC, ...]

    def __post_init__(self) -> None:
        """Validate that the devices are stacked: ascending and disjoint along w."""
        if not self.devices:
            raise ValueError("a stack needs at least one device")
        tops = [device.z_top for device in self.devices]
        if tops != sorted(tops):
            raise ValueError("devices must be ordered ascending in z_top")
        for upstream, downstream in zip(self.devices, self.devices[1:], strict=False):
            if downstream.z_top < upstream.z_bottom:
                raise ValueError(
                    f"devices overlap in z: [{upstream.z_top}, {upstream.z_bottom}] and "
                    f"[{downstream.z_top}, {downstream.z_bottom}]"
                )

    @property
    def materials(self) -> tuple[int, ...]:
        """Registry material index per device, in stack order."""
        return tuple(device.material for device in self.devices)

    @property
    def densities(self) -> tuple[float, ...]:
        """Density in g/cm^3 per device, in stack order."""
        return tuple(device.density for device in self.devices)

    @property
    def exit_z(self) -> float:
        """The downstream face of the last device (local w, cm)."""
        return self.devices[-1].z_bottom

    def path_lengths(self, origins: _F64, directions: _F64, *, from_origin: bool = False) -> _F64:
        """Per-device material chord lengths for engine-frame rays.

        ``origins``/``directions`` are ``(n, 3)`` engine-frame arrays (directions
        need not be re-normalized here; the frame rotation preserves norms, and
        unit directions are the caller's contract as everywhere in the engine).
        Returns ``(n, n_devices)``, column order = ``devices`` order. With
        ``from_origin=True`` each device sees only the forward half-line — the
        pre-solve's scattered photons start mid-stack.
        """
        p_local, d_local = self.frame.to_local(
            np.asarray(origins, dtype=np.float64), np.asarray(directions, dtype=np.float64)
        )
        columns = [
            device.path_lengths(p_local, d_local, from_origin=from_origin)
            for device in self.devices
        ]
        return np.column_stack(columns)
