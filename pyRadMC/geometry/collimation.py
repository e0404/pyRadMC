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

- Jaw edges are straight (unfocused) **by default**; :class:`JawPair` also offers
  ``focused=True``, whose edge face pivots through the focal spot so a focal-spot
  ray sees full-thickness-or-nothing (the geometric partial-transmission band
  collapses). MLC leaf ends are already focused via their rounded tips.
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

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyRadMC.data.handles import Table1D
from pyRadMC.data.interface import CrossSectionSource
from pyRadMC.geometry.source import BeamletSource, Primary, Source
from pyRadMC.rng import RNGState

__all__ = [
    "MLC",
    "BeamFrame",
    "BeamLimitingStack",
    "CollimatedBeamletSource",
    "CollimatedSource",
    "CompiledStack",
    "JawPair",
    "TransmissionMaskBeamletSource",
    "TransmissionMaskSource",
    "jaw_path_length",
    "mlc_path_length",
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


def _focused_halfspace_interval(a: _F64, b: _F64, *, above: bool) -> tuple[_F64, _F64]:
    """Give the t-interval where ``a + t*b >= 0`` (``<=`` for ``above=False``).

    The focused-edge analogue of :func:`_halfspace_interval`: the boundary is a
    plane through the focal spot (the frame origin), so the half-space condition
    is still affine in t but with a general slope ``b`` instead of a lone
    direction cosine. ``a``/``b`` are the constant and t-coefficient of the signed
    distance to that plane, formed by the caller from the mid-plane edge.
    """
    safe = np.where(b == 0.0, 1.0, b)
    t_root = -a / safe
    # For b > 0 the ray enters {a + t*b >= 0} at t_root; for b < 0 it leaves there.
    opens_up = (b > 0.0) == above
    t_lo = np.where(opens_up, t_root, -np.inf)
    t_hi = np.where(opens_up, np.inf, t_root)
    inside = (a >= 0.0) if above else (a <= 0.0)
    t_lo = np.where(b == 0.0, np.where(inside, -np.inf, np.inf), t_lo)
    t_hi = np.where(b == 0.0, np.where(inside, np.inf, -np.inf), t_hi)
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


# --- kernel-side scalar twins ------------------------------------------------------
#
# The device pre-solve traces one ray per thread, so it cannot use the vectorized
# primitives above. These pure, scalar, allocation-free functions are the
# single-ray twins of :meth:`JawPair.path_lengths` / :meth:`MLC.path_lengths`: the
# Warp physics loader compiles them to ``@wp.func`` (like the ``geometry.grid`` point
# queries), and they run under NumPy on the host. They take the ray already in the
# **local beam frame** (the kernel applies :meth:`BeamFrame.to_local` once) and the
# same mid-plane device parameters the dataclasses hold. Empty t-intervals are
# encoded ``(+inf, -inf)`` and infinite (parallel-trapped) chords stay ``inf``,
# exactly as in the vectorized path; ``tests/unit/test_collimation.py`` pins the twin
# equal to the vectorized answer on random rays, so the two definitions cannot drift.
# They keep ``math`` (not NumPy) and unrolled control flow so one source compiles for
# ref, cpu and cuda (AGENTS.md 2.5).


def _clip_len(lo: float, hi: float) -> float:
    """Length of the t-interval ``[lo, hi]``; zero if empty (also for ``inf<=inf``)."""
    if hi <= lo:
        return 0.0
    return hi - lo


def _affine_interval_length(s_lo: float, s_hi: float, a: float, b: float, above: bool) -> float:
    """Length of ``[s_lo, s_hi]`` intersected with the half-space ``a + t*b >= 0``.

    ``above=False`` uses ``a + t*b <= 0``. This is the scalar form of a slab
    intersected with :func:`_halfspace_interval` (straight edge: ``a = q0 - edge``,
    ``b = d_q``) or :func:`_focused_halfspace_interval` (focused edge: ``a``/``b``
    the affine coefficients of the pivoting plane).
    """
    if b == 0.0:
        inside = a >= 0.0 if above else a <= 0.0
        if not inside:
            return 0.0
        return _clip_len(s_lo, s_hi)
    t_root = -a / b
    opens_up = b > 0.0 if above else b < 0.0
    if opens_up:
        return _clip_len(max(s_lo, t_root), s_hi)
    return _clip_len(s_lo, min(s_hi, t_root))


def jaw_path_length(
    p_q: float,
    d_q: float,
    p_w: float,
    d_w: float,
    z_top: float,
    z_bottom: float,
    edge_neg: float,
    edge_pos: float,
    focused: int,
    from_origin: int,
) -> float:
    """Scalar chord length of one local-frame ray through a jaw pair (cm).

    ``p_q``/``d_q`` are the block-limited coordinate (local ``u`` or ``v``, the
    caller selects by ``axis``); ``p_w``/``d_w`` the beam-axis coordinate. Twin of
    :meth:`JawPair.path_lengths`; ``focused`` and ``from_origin`` are 0/1 flags.
    """
    if d_w == 0.0:
        if p_w < z_top or p_w > z_bottom:
            return 0.0
        s_lo = -math.inf
        s_hi = math.inf
    else:
        t_a = (z_top - p_w) / d_w
        t_b = (z_bottom - p_w) / d_w
        s_lo = min(t_a, t_b)
        s_hi = max(t_a, t_b)
    if from_origin != 0:
        s_lo = max(s_lo, 0.0)
    if focused != 0:
        z_mid = 0.5 * (z_top + z_bottom)
        a_pos = z_mid * p_q - edge_pos * p_w
        b_pos = z_mid * d_q - edge_pos * d_w
        a_neg = z_mid * p_q - edge_neg * p_w
        b_neg = z_mid * d_q - edge_neg * d_w
    else:
        a_pos = p_q - edge_pos
        b_pos = d_q
        a_neg = p_q - edge_neg
        b_neg = d_q
    return _affine_interval_length(s_lo, s_hi, a_pos, b_pos, True) + _affine_interval_length(
        s_lo, s_hi, a_neg, b_neg, False
    )


def _mlc_tip_length(
    is_lo: float,
    is_hi: float,
    p_u: float,
    d_u: float,
    p_w: float,
    d_w: float,
    center_u: float,
    z_mid: float,
    radius: float,
    above: bool,
) -> float:
    """Inclusion-exclusion chord of one rounded leaf end within a strip interval.

    ``[is_lo, is_hi]`` is the ray's t-interval inside the slab-and-strip. The leaf
    end is the flat body half-space (``u >= center_u`` for ``above``) unioned with
    the tip disc of ``radius`` centred at ``(center_u, z_mid)`` in the ``(u, w)``
    plane, so the chord is ``len(body) + len(disc) - len(body & disc)`` — the scalar
    form of :meth:`MLC.path_lengths`'s per-tip term.
    """
    # Flat body half-space bounds along t.
    if d_u == 0.0:
        inside = p_u >= center_u if above else p_u <= center_u
        if inside:
            bo_lo = -math.inf
            bo_hi = math.inf
        else:
            bo_lo = math.inf
            bo_hi = -math.inf
    else:
        t_edge = (center_u - p_u) / d_u
        opens_up = d_u > 0.0 if above else d_u < 0.0
        if opens_up:
            bo_lo = t_edge
            bo_hi = math.inf
        else:
            bo_lo = -math.inf
            bo_hi = t_edge
    # Tip-disc bounds: |r(t) - c|^2 <= R^2 in the (u, w) plane.
    off_u = p_u - center_u
    off_w = p_w - z_mid
    a_q = d_u * d_u + d_w * d_w
    b_q = 2.0 * (d_u * off_u + d_w * off_w)
    c_q = off_u * off_u + off_w * off_w - radius * radius
    if a_q > 0.0:
        discriminant = b_q * b_q - 4.0 * a_q * c_q
        if discriminant > 0.0:
            half_width = math.sqrt(discriminant)
            di_lo = (-b_q - half_width) / (2.0 * a_q)
            di_hi = (-b_q + half_width) / (2.0 * a_q)
        else:
            di_lo = math.inf
            di_hi = -math.inf
    else:
        # a_q == 0: the ray runs parallel to the cylinder axis — fully in or out.
        if c_q <= 0.0:
            di_lo = -math.inf
            di_hi = math.inf
        else:
            di_lo = math.inf
            di_hi = -math.inf
    body_len = _clip_len(max(is_lo, bo_lo), min(is_hi, bo_hi))
    disc_len = _clip_len(max(is_lo, di_lo), min(is_hi, di_hi))
    both_len = _clip_len(max(is_lo, max(bo_lo, di_lo)), min(is_hi, min(bo_hi, di_hi)))
    return body_len + disc_len - both_len


def mlc_path_length(
    p_u: float,
    p_v: float,
    p_w: float,
    d_u: float,
    d_v: float,
    d_w: float,
    z_top: float,
    z_bottom: float,
    tip_radius: float,
    n_pairs: int,
    edges_off: int,
    tips_off: int,
    leaf_edges_v: Table1D,
    tips_neg: Table1D,
    tips_pos: Table1D,
    from_origin: int,
) -> float:
    """Scalar chord length of one local-frame ray through both MLC banks (cm).

    Twin of :meth:`MLC.path_lengths`. Leaf data is addressed through flat array
    handles so several MLCs can share the device buffers: pair ``i`` reads
    ``leaf_edges_v[edges_off+i : edges_off+i+2]`` and ``tips_{neg,pos}[tips_off+i]``.
    """
    if d_w == 0.0:
        if p_w < z_top or p_w > z_bottom:
            return 0.0
        s_lo = -math.inf
        s_hi = math.inf
    else:
        t_a = (z_top - p_w) / d_w
        t_b = (z_bottom - p_w) / d_w
        s_lo = min(t_a, t_b)
        s_hi = max(t_a, t_b)
    if from_origin != 0:
        s_lo = max(s_lo, 0.0)
    z_mid = 0.5 * (z_top + z_bottom)
    total = float(0.0)  # noqa: UP018 -- Warp needs a float() cast to declare a loop-mutated var
    for i in range(n_pairs):
        edge_lo = leaf_edges_v[edges_off + i]
        edge_hi = leaf_edges_v[edges_off + i + 1]
        if d_v == 0.0:
            if p_v < edge_lo or p_v > edge_hi:
                st_lo = math.inf
                st_hi = -math.inf
            else:
                st_lo = -math.inf
                st_hi = math.inf
        else:
            v_a = (edge_lo - p_v) / d_v
            v_b = (edge_hi - p_v) / d_v
            st_lo = min(v_a, v_b)
            st_hi = max(v_a, v_b)
        is_lo = max(s_lo, st_lo)
        is_hi = min(s_hi, st_hi)
        tip_pos = tips_pos[tips_off + i]
        tip_neg = tips_neg[tips_off + i]
        total += _mlc_tip_length(
            is_lo, is_hi, p_u, d_u, p_w, d_w, tip_pos + tip_radius, z_mid, tip_radius, True
        )
        total += _mlc_tip_length(
            is_lo, is_hi, p_u, d_u, p_w, d_w, tip_neg - tip_radius, z_mid, tip_radius, False
        )
    return total


@dataclass(frozen=True)
class JawPair:
    """One pair of jaw blocks limiting one transverse coordinate.

    The pair is a slab ``z_top <= w <= z_bottom`` (local frame, cm downstream of
    the focal spot) holding two laterally unbounded blocks: the ``-axis`` block
    occupies ``q <= edge_neg`` and the ``+axis`` block ``q >= edge_pos`` **at the
    device mid-plane** ``z_mid``, where ``q`` is the local ``u`` or ``v``
    coordinate selected by ``axis``. Edges are always physical coordinates at the
    mid-plane; ``focused`` selects how the edge face runs through the slab:

    - ``focused=False`` (default): straight, unfocused edges — the face is
      parallel to ``w``, so the mid-plane edge is the edge at every depth. A ray
      near the edge crosses partial thickness over a wide lateral band (the v1
      approximation stated in the module docstring).
    - ``focused=True``: the edge face is the plane containing the focal spot (the
      frame origin) and the mid-plane edge line, so the ``+axis`` block is
      ``z_mid * q >= edge_pos * w`` (``<=`` with ``edge_neg`` for ``-axis``). Every
      ray *from the focal spot* then sees either the full slant thickness or
      nothing — the geometric partial-transmission band collapses to the focal
      spot size — while off-focal rays (e.g. from a planar source) still cut the
      exact chord of the wedge-shaped block. This is the physically faithful
      geometry of a divergent-machine jaw; the default stays straight so existing
      configurations are unchanged.

    ``edge_neg <= edge_pos`` (the blocks may touch — a closed pair — but never
    interpenetrate; the two focused planes meet only at the focal spot, so the
    ordering holds at every depth). ``density`` is in g/cm^3 and ``material`` is a
    registry index (:data:`pyRadMC.data.materials.TUNGSTEN` for a real jaw).
    """

    axis: str
    z_top: float
    z_bottom: float
    edge_neg: float
    edge_pos: float
    material: int
    density: float
    focused: bool = False

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

    def _edge_intervals(
        self, q: _F64, d_q: _F64, w: _F64, d_w: _F64
    ) -> tuple[tuple[_F64, _F64], tuple[_F64, _F64]]:
        """Give the (+block, -block) half-space t-intervals for this edge style."""
        if not self.focused:
            return (
                _halfspace_interval(q, d_q, self.edge_pos, above=True),
                _halfspace_interval(q, d_q, self.edge_neg, above=False),
            )
        # Focused edge: the face pivots through the focal spot (frame origin), so
        # the block half-space z_mid * q >= edge * w is affine in t with the
        # constant a = z_mid * q0 - edge * w0 and slope b = z_mid * d_q - edge * d_w.
        z_mid = (self.z_top + self.z_bottom) / 2.0
        a_pos = z_mid * q - self.edge_pos * w
        b_pos = z_mid * d_q - self.edge_pos * d_w
        a_neg = z_mid * q - self.edge_neg * w
        b_neg = z_mid * d_q - self.edge_neg * d_w
        return (
            _focused_halfspace_interval(a_pos, b_pos, above=True),
            _focused_halfspace_interval(a_neg, b_neg, above=False),
        )

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
        block_pos, block_neg = self._edge_intervals(p[:, q_index], d[:, q_index], p[:, 2], d[:, 2])
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

    def flatten(self) -> CompiledStack:
        """Flatten the stack into parallel arrays for the device pre-solve.

        The heterogeneous ``devices`` tuple becomes the column-parallel
        :class:`CompiledStack`: one row per device (jaws and MLCs share the same
        rows, each reading only the fields its ``kind`` uses), plus the frame and
        the concatenated MLC leaf arrays addressed by per-device offsets. This is
        the host, Warp-free half of putting the geometry on the device — a backend
        uploads these arrays and the kernel indexes them, exactly as
        :func:`pyRadMC.data.tables.build_cross_section_tables` does for the
        cross-sections. The vectorized :meth:`path_lengths` stays the host path.
        """
        n = len(self.devices)
        rot = np.array([self.frame.u_axis, self.frame.v_axis, self.frame.w_axis], dtype=np.float64)
        kind = np.zeros(n, dtype=np.int32)
        z_top = np.zeros(n, dtype=np.float64)
        z_bottom = np.zeros(n, dtype=np.float64)
        jaw_axis = np.zeros(n, dtype=np.int32)
        jaw_edge_neg = np.zeros(n, dtype=np.float64)
        jaw_edge_pos = np.zeros(n, dtype=np.float64)
        jaw_focused = np.zeros(n, dtype=np.int32)
        mlc_tip_radius = np.zeros(n, dtype=np.float64)
        mlc_n_pairs = np.zeros(n, dtype=np.int32)
        mlc_edges_off = np.zeros(n, dtype=np.int32)
        mlc_tips_off = np.zeros(n, dtype=np.int32)
        leaf_edges: list[float] = []
        tips_neg: list[float] = []
        tips_pos: list[float] = []
        for j, device in enumerate(self.devices):
            z_top[j] = device.z_top
            z_bottom[j] = device.z_bottom
            if isinstance(device, JawPair):
                kind[j] = 0
                jaw_axis[j] = 0 if device.axis == "u" else 1
                jaw_edge_neg[j] = device.edge_neg
                jaw_edge_pos[j] = device.edge_pos
                jaw_focused[j] = 1 if device.focused else 0
            else:
                kind[j] = 1
                mlc_tip_radius[j] = device.tip_radius
                mlc_n_pairs[j] = device.n_pairs
                mlc_edges_off[j] = len(leaf_edges)
                mlc_tips_off[j] = len(tips_pos)
                leaf_edges.extend(device.leaf_edges_v)
                tips_neg.extend(device.tips_neg)
                tips_pos.extend(device.tips_pos)
        # Never expose a zero-length leaf array: a backend may reject an empty
        # device buffer, and the offsets addressing them are unused when no MLC is
        # present (every ``mlc_n_pairs`` is then zero, so the arrays are not read).
        return CompiledStack(
            n_devices=n,
            origin=np.asarray(self.frame.origin, dtype=np.float64),
            rotation=rot.reshape(-1),
            kind=kind,
            z_top=z_top,
            z_bottom=z_bottom,
            material=np.asarray(self.materials, dtype=np.int32),
            density=np.asarray(self.densities, dtype=np.float64),
            jaw_axis=jaw_axis,
            jaw_edge_neg=jaw_edge_neg,
            jaw_edge_pos=jaw_edge_pos,
            jaw_focused=jaw_focused,
            mlc_tip_radius=mlc_tip_radius,
            mlc_n_pairs=mlc_n_pairs,
            mlc_edges_off=mlc_edges_off,
            mlc_tips_off=mlc_tips_off,
            leaf_edges_v=np.asarray(leaf_edges or [0.0], dtype=np.float64),
            tips_neg=np.asarray(tips_neg or [0.0], dtype=np.float64),
            tips_pos=np.asarray(tips_pos or [0.0], dtype=np.float64),
        )


@dataclass(frozen=True)
class CompiledStack:
    """A :class:`BeamLimitingStack` flattened into device-uploadable arrays.

    Column-parallel over ``n_devices``: ``kind[j]`` is 0 for a jaw pair (reading
    ``jaw_*[j]``) and 1 for an MLC (reading ``mlc_*[j]`` and the shared leaf arrays
    at ``mlc_edges_off[j]`` / ``mlc_tips_off[j]``). ``rotation`` is the row-major
    ``(u, v, w)`` frame basis; a kernel maps an engine-frame ray to the local frame
    by ``p_local_i = rotation[i] . (p - origin)``. All arrays are host NumPy; the
    backend casts on upload (float32 for Warp). Produced by
    :meth:`BeamLimitingStack.flatten`.
    """

    n_devices: int
    origin: _F64
    rotation: _F64
    kind: NDArray[np.int32]
    z_top: _F64
    z_bottom: _F64
    material: NDArray[np.int32]
    density: _F64
    jaw_axis: NDArray[np.int32]
    jaw_edge_neg: _F64
    jaw_edge_pos: _F64
    jaw_focused: NDArray[np.int32]
    mlc_tip_radius: _F64
    mlc_n_pairs: NDArray[np.int32]
    mlc_edges_off: NDArray[np.int32]
    mlc_tips_off: NDArray[np.int32]
    leaf_edges_v: _F64
    tips_neg: _F64
    tips_pos: _F64


class _RayWeightModel(ABC):
    """A deterministic per-ray weight factor in [0, 1] (transmission, mask value).

    The scalar ``emit`` path (:meth:`factor_for`) is implemented on top of the
    vectorized :meth:`factors`, so both engine routes evaluate one arithmetic and
    can never acquire a route-relative weight bias.
    """

    @abstractmethod
    def factors(self, energies: _F64, origins: _F64, directions: _F64) -> _F64:
        """Per-ray weight factors in [0, 1], vectorized over ``(n,)``/``(n, 3)``."""

    def factor_for(self, primary: Primary) -> float:
        """Evaluate the scalar ``emit`` path — same table, same arithmetic."""
        factors = self.factors(
            np.array([primary.energy]),
            np.array([[primary.x, primary.y, primary.z]]),
            np.array([[primary.ux, primary.uy, primary.uz]]),
        )
        return float(factors[0])


class _StackTransmission(_RayWeightModel):
    """Narrow-beam transmission through a stack: ``exp(-sum_i (mu/rho)_i(E) rho_i t_i)``.

    One per-device log-log table of ``mu_over_rho_total`` is built at construction
    through the :class:`~pyRadMC.data.interface.CrossSectionSource` interface
    (AGENTS.md 2.6 — never a hardcoded cross-section), and **both** the scalar
    ``emit`` path and the vectorized batch path evaluate this same table, so the
    two engine routes can never acquire a route-relative weight bias.

    Stated approximations (Beer-Lambert with the **total** attenuation
    coefficient, coherent included): narrow-beam geometry — photons scattered in
    the devices are removed, none are transported onward (slightly conservative,
    since coherent and small-angle incoherent scatter is forward-peaked; the head
    pre-solve is the higher-fidelity path that keeps the first Compton photon) —
    and no electron contamination from the device surfaces. Energies below
    ``e_min`` clamp to the table edge (log-log interpolation; the W K-edge at
    69.5 keV sits below any MV spectrum's useful range).
    """

    def __init__(
        self,
        stack: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        e_max: float,
        e_min: float,
        n_table: int,
    ) -> None:
        if max(stack.materials) >= cross_sections.n_materials:
            raise ValueError(
                f"stack material index {max(stack.materials)} is beyond this "
                f"cross-section source ({cross_sections.n_materials} materials); "
                "device materials like tungsten need the tabulated backend"
            )
        if not 0.0 < e_min < e_max:
            raise ValueError(f"need 0 < e_min < e_max, got {e_min}, {e_max}")
        self._stack = stack
        energies = np.geomspace(e_min, e_max, n_table)
        self._log_energies = np.log(energies)
        self._log_mu = np.stack(
            [
                np.log([cross_sections.mu_over_rho_total(float(e), material) for e in energies])
                for material in stack.materials
            ]
        )
        self._densities = np.asarray(stack.densities, dtype=np.float64)

    def factors(self, energies: _F64, origins: _F64, directions: _F64) -> _F64:
        """Per-ray transmission factors in [0, 1], vectorized."""
        thickness = self._stack.path_lengths(origins, directions)  # (n, n_devices)
        log_e = np.log(np.asarray(energies, dtype=np.float64))
        mu = np.exp(
            np.stack([np.interp(log_e, self._log_energies, row) for row in self._log_mu])
        )  # (n_devices, n)
        # A ray trapped parallel to a slab has infinite thickness: 0 * inf would be
        # NaN for a zero-mu device, but mu is strictly positive here, so tau is inf
        # and the factor cleanly zero.
        tau = (mu.T * self._densities * thickness).sum(axis=1)
        return np.asarray(np.exp(-tau), dtype=np.float64)


class _MaskTransmission(_RayWeightModel):
    """A user 0..1 transmission mask on a plane rectangle (pure fluence shaping).

    ``mask[i, j]`` covers pixel ``i`` along ``u_axis`` and ``j`` along ``v_axis``;
    pixel centres tile the ``width_u x width_v`` rectangle about ``plane_center``.
    Lookups interpolate bilinearly between pixel centres — written in the
    incremental form ``v00 + fu*(v10-v00) + ...`` so a plateau of equal pixels
    (an all-open mask in particular) evaluates to the pixel value *exactly* —
    clamped in the half-pixel border band. A ray crossing the plane outside the
    rectangle, or running parallel to it, carries weight zero. Energy plays no
    role: this is configuration 2 of the BLD workstream (forward fields and
    sequenced shapes for aperture optimization), not an attenuation model.

    The plane is intersected along the ray's full line (the stack convention), so
    the wrapped source may emit on either side of it.
    """

    def __init__(
        self,
        mask: NDArray[np.float64],
        plane_center: tuple[float, float, float],
        width_u: float,
        width_v: float,
        u_axis: tuple[float, float, float],
        v_axis: tuple[float, float, float],
    ) -> None:
        values = np.asarray(mask, dtype=np.float64)
        if values.ndim != 2 or values.size == 0:
            raise ValueError(f"the mask must be a non-empty 2D array, got shape {values.shape}")
        if not np.all(np.isfinite(values)) or values.min() < 0.0 or values.max() > 1.0:
            raise ValueError("mask values must be finite and within 0..1")
        if width_u <= 0.0 or width_v <= 0.0:
            raise ValueError(f"widths must be positive, got {width_u}, {width_v}")
        u = np.asarray(_as_unit(u_axis, "u_axis"))
        v = np.asarray(_as_unit(v_axis, "v_axis"))
        if abs(float(np.dot(u, v))) > 1.0e-9:
            raise ValueError("u_axis and v_axis must be orthogonal")
        self._mask = values
        self._center = np.asarray(plane_center, dtype=np.float64)
        self._widths = (float(width_u), float(width_v))
        self._u_axis = u
        self._v_axis = v
        self._normal = np.cross(u, v)

    def _interpolate(self, coordinate: _F64, axis: int) -> tuple[NDArray[np.int64], _F64]:
        """Clamped integer pixel index and fractional offset along one mask axis."""
        n_pixels = self._mask.shape[axis]
        width = self._widths[axis]
        pixel = np.clip(
            (coordinate + width / 2.0) / (width / n_pixels) - 0.5, 0.0, float(n_pixels - 1)
        )
        low = np.minimum(pixel.astype(np.int64), max(n_pixels - 2, 0))
        return low, pixel - low

    def factors(self, energies: _F64, origins: _F64, directions: _F64) -> _F64:
        """Per-ray mask values in [0, 1]; ``energies`` is unused (fluence-only)."""
        del energies
        p = np.asarray(origins, dtype=np.float64)
        d = np.asarray(directions, dtype=np.float64)
        denominator = d @ self._normal
        safe = np.where(denominator == 0.0, 1.0, denominator)
        t = ((self._center - p) @ self._normal) / safe
        offset = p + t[:, None] * d - self._center
        u = offset @ self._u_axis
        v = offset @ self._v_axis
        inside = (
            (denominator != 0.0)
            & (np.abs(u) <= self._widths[0] / 2.0)
            & (np.abs(v) <= self._widths[1] / 2.0)
        )
        i0, fu = self._interpolate(u, 0)
        j0, fv = self._interpolate(v, 1)
        i1 = np.minimum(i0 + 1, self._mask.shape[0] - 1)
        j1 = np.minimum(j0 + 1, self._mask.shape[1] - 1)
        v00 = self._mask[i0, j0]
        v10 = self._mask[i1, j0]
        v01 = self._mask[i0, j1]
        v11 = self._mask[i1, j1]
        # Incremental bilinear form: equal neighbours make every difference term
        # exactly zero, so plateaus (and the all-ones mask) are bit-inert.
        value = v00 + fu * (v10 - v00) + fv * (v01 - v00) + fu * fv * (v11 - v10 - v01 + v00)
        return np.asarray(np.where(inside, value, 0.0), dtype=np.float64)


def _attenuated_batch(
    columns: dict[str, NDArray[np.generic]], model: _RayWeightModel
) -> dict[str, NDArray[np.generic]]:
    """Multiply a pre-sampling batch's weight column by the model's ray factors."""
    origins = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
    directions = np.stack([columns["ux"], columns["uy"], columns["uz"]], axis=1).astype(np.float64)
    factors = model.factors(columns["energy"].astype(np.float64), origins, directions)
    out = dict(columns)
    out["weight"] = (columns["weight"].astype(np.float64) * factors).astype(np.float32)
    return out


class _FactorWrappedSource(Source):
    """Shared delegation for open-field wrappers: inner source, per-ray factor.

    The wrapper consumes **zero extra uniforms**, so the inner source's RNG
    contract (draw count, correlated replay, chunk-invariant vectorized batches)
    survives wrapping unchanged. A factor of zero leaves the history in flight at
    weight zero — cheap, and it keeps the draw count fixed.
    """

    def __init__(self, inner: Source, model: _RayWeightModel) -> None:
        self._inner = inner
        self._model = model

    @property
    def max_energy(self) -> float:
        """The inner source's maximum energy (a weight factor never raises it)."""
        return self._inner.max_energy

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit the inner primary with its weight rescaled; no draws consumed here."""
        primary = self._inner.emit(rng_state)
        return primary._replace(weight=primary.weight * self._model.factor_for(primary))

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, np.ndarray]:
        """Return the inner batch with the weight column rescaled (vectorized)."""
        return _attenuated_batch(self._inner.sample_batch(seed, history_offset, n), self._model)


class _FactorWrappedBeamletSource(BeamletSource):
    """Shared delegation for beamlet wrappers: inner source, per-ray factor.

    The factor is deterministic in ``(energy, ray)``, so under correlated
    sampling every beamlet still replays the same energy and aperture offset —
    only the beamlet's ray, and hence its factor, differs. Per-column sigmas
    stay valid; the Phase 4 rule (never combine column sigmas in quadrature) is
    untouched.
    """

    def __init__(self, inner: BeamletSource, model: _RayWeightModel) -> None:
        self._inner = inner
        self._model = model

    @property
    def max_energy(self) -> float:
        """The inner source's maximum energy (a weight factor never raises it)."""
        return self._inner.max_energy

    @property
    def n_beamlets(self) -> int:
        """The inner source's beamlet count."""
        return self._inner.n_beamlets

    def emit(self, beamlet: int, rng_state: RNGState) -> Primary:
        """Emit the inner beamlet primary with its weight rescaled."""
        primary = self._inner.emit(beamlet, rng_state)
        return primary._replace(weight=primary.weight * self._model.factor_for(primary))

    def sample_beamlet_batch(
        self, seed: int, history_offset: int, n: int, beamlet: int
    ) -> dict[str, np.ndarray]:
        """Return the inner beamlet batch with the weight column rescaled (vectorized)."""
        return _attenuated_batch(
            self._inner.sample_beamlet_batch(seed, history_offset, n, beamlet), self._model
        )


class CollimatedSource(_FactorWrappedSource):
    """A source seen through a beam-limiting stack: deterministic weight attenuation.

    Delegates emission to ``inner`` and multiplies each primary's statistical
    weight by the stack transmission along its ray. Physics and approximations:
    :class:`_StackTransmission`. This is the Dij-compatible baseline configuration
    ("attenuation"); the head pre-solve (``geometry/head.py``) is the
    higher-fidelity forward path.

    The full-line stack convention means the inner source may emit upstream of the
    devices (a focal-spot fan) or downstream of them (the planar exit source) —
    either way the emitted ray's line crosses the stack and is attenuated once.
    """

    def __init__(
        self,
        inner: Source,
        stack: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        *,
        e_min: float = 0.010,
        n_table: int = 512,
    ) -> None:
        """Build the shared transmission table up to the inner source's max energy."""
        super().__init__(
            inner, _StackTransmission(stack, cross_sections, inner.max_energy, e_min, n_table)
        )


class CollimatedBeamletSource(_FactorWrappedBeamletSource):
    """A beamlet source seen through a beam-limiting stack (Dij configuration).

    See :class:`CollimatedSource`; the beamlet-dependent transmission along each
    bixel's rays is precisely the collimation signal in the Dij columns.
    """

    def __init__(
        self,
        inner: BeamletSource,
        stack: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        *,
        e_min: float = 0.010,
        n_table: int = 512,
    ) -> None:
        """Build the shared transmission table up to the inner source's max energy."""
        super().__init__(
            inner, _StackTransmission(stack, cross_sections, inner.max_energy, e_min, n_table)
        )


class TransmissionMaskSource(_FactorWrappedSource):
    """A source shaped by a user 0..1 transmission mask on a plane rectangle.

    Configuration 2 of the BLD workstream: forward fields and sequenced shapes
    for aperture optimization, applied as a deterministic per-ray weight with no
    cross-sections involved. Mask conventions, bilinear lookup, and the
    outside-rectangle/parallel-ray zero: :class:`_MaskTransmission`.
    """

    def __init__(
        self,
        inner: Source,
        *,
        mask: NDArray[np.float64],
        plane_center: tuple[float, float, float],
        width_u: float,
        width_v: float,
        u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        """Wrap ``inner`` with a mask plane given in the engine frame (cm)."""
        super().__init__(
            inner, _MaskTransmission(mask, plane_center, width_u, width_v, u_axis, v_axis)
        )


class TransmissionMaskBeamletSource(_FactorWrappedBeamletSource):
    """A beamlet source shaped by a user 0..1 transmission mask (Dij configuration).

    See :class:`TransmissionMaskSource`.
    """

    def __init__(
        self,
        inner: BeamletSource,
        *,
        mask: NDArray[np.float64],
        plane_center: tuple[float, float, float],
        width_u: float,
        width_v: float,
        u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        """Wrap ``inner`` with a mask plane given in the engine frame (cm)."""
        super().__init__(
            inner, _MaskTransmission(mask, plane_center, width_u, width_v, u_axis, v_axis)
        )
