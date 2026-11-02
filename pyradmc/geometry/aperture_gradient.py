"""Aperture gradients: derivatives of the dose with respect to a leaf tip or jaw edge.

**Prototype** (branch ``aperture-gradient-prototype``): parallel beams, unfocused
parameter devices, monoenergetic sources. It exists to measure whether a Monte
Carlo leaf-position gradient differs from the bixel-Dij chain rule that direct
aperture optimization uses, before anything is built on it.

**The identity.** Under attenuation-only collimation (:class:`CollimatedSource`) an
edge coordinate ``x`` — one MLC leaf tip, one jaw edge — enters a history only
through its primary's weight ``T(ray; x)``, the narrow-beam stack transmission; the
transport that follows never sees ``x``. The forward dose is linear in the weights,
so with ``K(ray)`` the dose response per unit primary weight,

    D(x)     = E_phi[ T(ray; x) K(ray) ]
    dD/dx    = E_phi[ dT/dx (ray; x) K(ray) ],

where ``phi`` is the source's ray density. Every estimator below is therefore an
ordinary :class:`~pyradmc.geometry.source.Source` — one engine run, one dose map,
with batch sigma — that differs only in how it samples rays for ``dT/dx`` and
weights them. No engine, kernel or transport change is involved, and the gradient
tally can never become a sampling routine inside a backend.

**Why sampling, not differentiation.** ``dT/dx`` is a measure, not always a
function: a straight edge face makes ``T`` jump, and the derivative there is a
point mass on the edge line. Pathwise differentiation of the simulation (autodiff
of a taped run, :class:`PathwiseDerivativeSource`) sees only the smooth part — zero
everywhere for a straight jaw — the boundary-term failure known from
differentiable rendering (Li et al., ACM TOG 37, 222 (2018),
doi:10.1145/3272127.3275109). At a rounded tip the smooth part is complete but
behaves like ``1/sqrt(delta)`` at the apex, so the pathwise estimator's variance
diverges logarithmically. :class:`EdgeDerivativeSource` removes both problems by
**sampling rays from** ``|dT/dx|`` itself: for a translated edge
``T(q; x) = T0(q - x)`` along the leaf-travel coordinate ``q``, so ``|dT/dx|`` is the
total-variation measure of ``T0``, of finite mass ``1 - T_min``, sampled exactly by
inverting the monotone ``T0``. A jump inverts to the edge line; a smooth tip to its
penumbra. The weight is then a constant times the transmission through the rest
of the stack.

Stated prototype approximations, each refused at construction where it bites:

- **Parallel beam along the frame ``w`` axis, identity frame axes.** Translating
  the edge is then exactly a translation of the rays. A divergent fan breaks the
  equality (an off-axis ray crosses the slab obliquely); a fan version needs
  ``dT/dx`` per ray direction, not one profile.
- **Unfocused parameter device** (straight jaw faces, parallel-sided leaves): a
  focused face rotates about the focal spot when its edge moves.
- **Monoenergetic beam**: one transmission profile; a spectrum needs one per energy.
- The inverse profile is tabulated (``n_table`` nodes, linear in the cumulative
  fraction); a jump is represented exactly, a smooth tip to interpolation accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyradmc.data.interface import CrossSectionSource
from pyradmc.geometry.collimation import (
    MLC,
    BeamFrame,
    BeamLimitingStack,
    CompiledStack,
    JawPair,
    _disc_interval,
    _focused_halfspace_interval,
    _halfspace_interval,
    _intersect,
    _length,
    _slab_interval,
    _StackTransmission,
    jaw_edge_derivative,
    mlc_tip_edge_derivative,
)
from pyradmc.geometry.source import (
    BeamletSource,
    ParallelBeamSource,
    Primary,
    Source,
    SpectralBeamSource,
)
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng import RNGState, uniform

__all__ = [
    "BixelChainRuleSource",
    "EdgeDerivativeSource",
    "EdgeGradientBeamletSource",
    "EdgeParameter",
    "FanEdgeColumns",
    "FanEdgeDerivativeSource",
    "FanEdgeRecord",
    "PathwiseDerivativeSource",
    "ProfileChainRuleSource",
    "RayFiniteDifferenceSource",
    "clear_caches",
    "edge_value",
    "fan_edge_columns",
    "fan_support_start",
    "isolated_block",
    "isolated_parameter",
    "tip_support_bisection",
    "tip_support_starts",
    "with_edge",
]

_F64 = NDArray[np.float64]

# Where a block is moved to take it out of the field (cm, mid-plane): far beyond any
# grid, yet small enough that path lengths stay well-conditioned in float64.
_RETRACT_CM = 1.0e4
_BISECTION_STEPS = 64
_AXIS_TOLERANCE = 1.0e-12
# The transmission tables match CollimatedSource's defaults, so a gradient source and
# the forward source it differentiates evaluate the same mu(E).
_E_MIN = 0.010
_N_MU_TABLE = 512


@dataclass(frozen=True)
class EdgeParameter:
    """One edge coordinate of a stack: an MLC leaf tip or a jaw edge.

    ``device`` indexes :attr:`BeamLimitingStack.devices`. ``side`` is ``"pos"`` (the
    ``+axis`` block: a jaw's ``edge_pos``, a pair's ``tips_pos``) or ``"neg"``.
    ``pair`` is the MLC leaf pair and must be ``None`` for a jaw. The coordinate is
    the device mid-plane position stored on the device (cm).
    """

    device: int
    side: str
    pair: int | None = None


def _resolve(stack: BeamLimitingStack, parameter: EdgeParameter) -> JawPair | MLC:
    """Return the parameter's device, refusing indices and sides that do not exist."""
    if not 0 <= parameter.device < len(stack.devices):
        raise ValueError(f"device {parameter.device} not in a {len(stack.devices)}-device stack")
    if parameter.side not in ("pos", "neg"):
        raise ValueError(f"side must be 'pos' or 'neg', got {parameter.side!r}")
    device = stack.devices[parameter.device]
    if isinstance(device, MLC):
        if parameter.pair is None or not 0 <= parameter.pair < device.n_pairs:
            raise ValueError(f"an MLC edge needs a pair in [0, {device.n_pairs})")
    elif parameter.pair is not None:
        raise ValueError("a jaw edge takes no leaf pair")
    return device


def edge_value(stack: BeamLimitingStack, parameter: EdgeParameter) -> float:
    """Return the parameter's current mid-plane coordinate (cm)."""
    device = _resolve(stack, parameter)
    if isinstance(device, MLC):
        assert parameter.pair is not None
        tips = device.tips_pos if parameter.side == "pos" else device.tips_neg
        return float(tips[parameter.pair])
    return device.edge_pos if parameter.side == "pos" else device.edge_neg


def _set(values: tuple[float, ...], index: int, value: float) -> tuple[float, ...]:
    return (*values[:index], float(value), *values[index + 1 :])


def with_edge(
    stack: BeamLimitingStack, parameter: EdgeParameter, value: float
) -> BeamLimitingStack:
    """Return a copy of ``stack`` with the parameter's coordinate set to ``value`` (cm).

    The device re-validates, so a move that crosses the opposing block is refused.
    """
    device = _resolve(stack, parameter)
    moved: JawPair | MLC
    if isinstance(device, MLC):
        assert parameter.pair is not None
        i = parameter.pair
        if parameter.side == "pos":
            moved = replace(device, tips_pos=_set(device.tips_pos, i, value))
        else:
            moved = replace(device, tips_neg=_set(device.tips_neg, i, value))
    elif parameter.side == "pos":
        moved = replace(device, edge_pos=float(value))
    else:
        moved = replace(device, edge_neg=float(value))
    devices = list(stack.devices)
    devices[parameter.device] = moved
    return replace(stack, devices=tuple(devices))


# Tables that never depend on an edge's position, shared across sources: an optimizer
# re-targets hundreds of edges per iteration. Keyed on the cross-section source's id;
# the entry holds the source itself, so a recycled id cannot alias a dead one.
_TABLES: dict[tuple[Any, ...], tuple[CrossSectionSource, _StackTransmission]] = {}
_PROFILES: dict[tuple[Any, ...], tuple[CrossSectionSource, _F64]] = {}


def clear_caches() -> None:
    """Drop the shared mu(E) tables and inverse edge profiles (tests, memory)."""
    _TABLES.clear()
    _PROFILES.clear()


def _transmission_for(
    stack: BeamLimitingStack, cross_sections: CrossSectionSource, e_max: float
) -> _StackTransmission:
    """Build a stack transmission whose mu(E) rows come from the shared table cache."""
    key = (id(cross_sections), stack.materials, stack.densities, e_max)
    entry = _TABLES.get(key)
    if entry is None or entry[0] is not cross_sections:
        entry = (
            cross_sections,
            _StackTransmission(stack, cross_sections, e_max, _E_MIN, _N_MU_TABLE),
        )
        _TABLES[key] = entry
    return entry[1].for_stack(stack)


def _block_shape(device: JawPair | MLC) -> tuple[Any, ...]:
    """Everything the edge profile depends on under a parallel beam, but the position."""
    if isinstance(device, MLC):
        return (
            "mlc",
            device.z_top,
            device.z_bottom,
            device.tip_radius,
            device.material,
            device.density,
        )
    return ("jaw", device.z_top, device.z_bottom, device.material, device.density)


def _retracted(stack: BeamLimitingStack, parameter: EdgeParameter) -> BeamLimitingStack:
    """Return the stack with the parameter's block moved out of the field."""
    far = _RETRACT_CM if parameter.side == "pos" else -_RETRACT_CM
    return with_edge(stack, parameter, far)


def _strip_restricted(stack: BeamLimitingStack, parameter: EdgeParameter) -> BeamLimitingStack:
    """Return ``stack`` with the parameter's MLC cut down to the parameter's leaf pair.

    Only for the edge-sampled sources: their rays are parallel to ``w`` inside the
    pair's strip, and a parallel-sided leaf of another pair has an empty slab
    interval along such a ray, so every dropped pair contributes an exact ``+0.0``
    to the chord sum (pinned bit-identical in the unit tier). The host cost of the
    rest-of-stack transmission then no longer scales with the number of pairs.
    """
    device = stack.devices[parameter.device]
    if not isinstance(device, MLC):
        return stack
    assert parameter.pair is not None
    i = parameter.pair
    single = replace(
        device,
        leaf_edges_v=(device.leaf_edges_v[i], device.leaf_edges_v[i + 1]),
        tips_neg=(device.tips_neg[i],),
        tips_pos=(device.tips_pos[i],),
    )
    devices = list(stack.devices)
    devices[parameter.device] = single
    return replace(stack, devices=tuple(devices))


def isolated_block(stack: BeamLimitingStack, parameter: EdgeParameter) -> BeamLimitingStack:
    """Return a one-device stack holding only the parameter's block (or leaf).

    The opposing block is retracted and, for an MLC, only the parameter's pair
    kept, so the block's chord is exactly the part of the stack's chords that
    depends on the parameter. :func:`isolated_parameter` addresses it.
    """
    device = _resolve(stack, parameter)
    x = edge_value(stack, parameter)
    pos = parameter.side == "pos"
    alone: JawPair | MLC
    if isinstance(device, MLC):
        assert parameter.pair is not None
        i = parameter.pair
        alone = replace(
            device,
            leaf_edges_v=(device.leaf_edges_v[i], device.leaf_edges_v[i + 1]),
            tips_neg=(-_RETRACT_CM,) if pos else (x,),
            tips_pos=(x,) if pos else (_RETRACT_CM,),
        )
    else:
        alone = replace(
            device,
            edge_neg=-_RETRACT_CM if pos else x,
            edge_pos=x if pos else _RETRACT_CM,
        )
    return replace(stack, devices=(alone,))


def isolated_parameter(parameter: EdgeParameter) -> EdgeParameter:
    """Return the parameter's address inside :func:`isolated_block`'s stack."""
    return replace(parameter, device=0, pair=None if parameter.pair is None else 0)


def _require_parallel_geometry(
    beam: Source, stack: BeamLimitingStack, parameter: EdgeParameter
) -> JawPair | MLC:
    """Refuse configurations where edge translation is not ray translation."""
    if not isinstance(beam, ParallelBeamSource):
        raise ValueError("the prototype supports ParallelBeamSource beams only")
    frame = stack.frame
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    axes = (frame.u_axis, frame.v_axis, frame.w_axis)
    if not np.allclose(axes, identity, atol=_AXIS_TOLERANCE):
        raise ValueError("the prototype needs identity beam-frame axes (w along +z)")
    device = _resolve(stack, parameter)
    if isinstance(device, JawPair) and device.focused:
        raise ValueError("a focused jaw face rotates when its edge moves; use focused=False")
    if isinstance(device, MLC) and device.focused_sides:
        raise ValueError(
            "focused leaf sides make T depend on v in the strip; use focused_sides=False"
        )
    return device


class _EdgeSampledSource(Source):
    """Shared body of the estimators that place their rays along the edge.

    Rays are vertical, at a coordinate ``r`` along the edge drawn uniformly over the
    edge's extent inside the field (the leaf strip for an MLC tip) and a coordinate
    ``q`` across it drawn by :meth:`_draw_q`. The weight is the constant
    :attr:`weight_scale` times the transmission through the stack with the
    parameter's block retracted, and zero for a ray outside the beam's field. Two
    uniforms per history, always.
    """

    def __init__(
        self,
        beam: ParallelBeamSource,
        stack: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        parameter: EdgeParameter,
    ) -> None:
        device = _require_parallel_geometry(beam, stack, parameter)
        self._beam = beam
        self._x = edge_value(stack, parameter)
        self._sign = 1.0 if parameter.side == "pos" else -1.0
        self._q_axis = 0 if isinstance(device, MLC) or device.axis == "u" else 1
        self._origin = (stack.frame.origin[0], stack.frame.origin[1])
        local_x = (beam.x_range[0] - self._origin[0], beam.x_range[1] - self._origin[0])
        local_y = (beam.y_range[0] - self._origin[1], beam.y_range[1] - self._origin[1])
        r_field = local_y if self._q_axis == 0 else local_x
        if isinstance(device, MLC):
            assert parameter.pair is not None
            strip = (device.leaf_edges_v[parameter.pair], device.leaf_edges_v[parameter.pair + 1])
            self._r_range = (max(strip[0], r_field[0]), min(strip[1], r_field[1]))
            span = device.tip_radius
        else:
            self._r_range = r_field
            span = 0.0
        r_length = self._r_range[1] - self._r_range[0]
        if r_length <= 0.0:
            raise ValueError("the parameter's edge lies outside the beam's field")
        # Beyond this distance into the block every vertical ray sees the full slab.
        self._span = span + (device.z_bottom - device.z_top) + 1.0
        # The isolated block is evaluated with its edge at q = 0 (a canonical,
        # position-free profile), so a source rebuilt after the edge moved reuses
        # the cached profile and is bit-identical to one built from scratch.
        canonical = with_edge(isolated_block(stack, parameter), isolated_parameter(parameter), 0.0)
        self._isolated = _transmission_for(canonical, cross_sections, beam.max_energy)
        self._other = _transmission_for(
            _strip_restricted(_retracted(stack, parameter), parameter),
            cross_sections,
            beam.max_energy,
        )
        self._profile_key = (
            _block_shape(device),
            parameter.side,
            beam.energy,
            id(cross_sections),
        )
        self._t_min = float(self._isolated_transmission(np.array([self._span]))[0])
        area = (beam.x_range[1] - beam.x_range[0]) * (beam.y_range[1] - beam.y_range[0])
        self._weight_scale = self._sign * (1.0 - self._t_min) * r_length / area

    @property
    def max_energy(self) -> float:
        """The beam energy."""
        return self._beam.max_energy

    @property
    def weight_scale(self) -> float:
        """``sign * (1 - T_min) * L_edge / A``: the derivative measure per history."""
        return self._weight_scale

    @property
    def t_min(self) -> float:
        """Transmission of a vertical ray through the full thickness of the block."""
        return self._t_min

    def _engine_points(self, q: _F64, r: _F64) -> _F64:
        """Engine-frame points on the beam's emission plane for local ``(q, r)``."""
        u, v = (q, r) if self._q_axis == 0 else (r, q)
        z = np.full_like(u, self._beam.z)
        return np.stack([self._origin[0] + u, self._origin[1] + v, z], axis=1)

    def _isolated_transmission(self, depth: _F64) -> _F64:
        """Isolated-block transmission at ``depth`` cm into the block from its edge."""
        q = self._sign * depth  # canonical frame: the edge at q = 0
        r = np.full_like(q, 0.5 * (self._r_range[0] + self._r_range[1]))
        points = self._engine_points(q, r)
        directions = np.tile([0.0, 0.0, 1.0], (q.size, 1))
        return self._isolated.factors(np.full(q.size, self._beam.energy), points, directions)

    def _q_for(self, fractions: _F64) -> _F64:
        """Map uniform fractions to across-edge coordinates (vectorized)."""
        raise NotImplementedError

    def _rays(self, u_r: _F64, u_q: _F64) -> tuple[_F64, _F64]:
        """Engine-frame points and weights for the uniforms ``(u_r, u_q)``.

        The single arithmetic behind both :meth:`emit` and :meth:`sample_batch`, so
        the two engine routes cannot acquire a route-relative weight difference.
        """
        r = self._r_range[0] + (self._r_range[1] - self._r_range[0]) * u_r
        points = self._engine_points(self._q_for(u_q), r)
        directions = np.tile([0.0, 0.0, 1.0], (r.size, 1))
        other = self._other.factors(np.full(r.size, self._beam.energy), points, directions)
        inside = (
            (points[:, 0] >= self._beam.x_range[0])
            & (points[:, 0] <= self._beam.x_range[1])
            & (points[:, 1] >= self._beam.y_range[0])
            & (points[:, 1] <= self._beam.y_range[1])
        )
        return points, np.where(inside, self._weight_scale * other, 0.0)

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one vertical ray along the edge; consumes exactly two uniforms."""
        u_r = uniform(rng_state)
        u_q = uniform(rng_state)
        points, weights = self._rays(np.array([u_r]), np.array([u_q]))
        x, y, z = points[0]
        return Primary(self._beam.energy, x, y, z, 0.0, 0.0, 1.0, weight=float(weights[0]))

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, NDArray[Any]]:
        """Vectorized pre-sampling of ``n`` rays through the shared :meth:`_rays` arithmetic.

        The phase-space / divergent-fan precedent (``_DivergentFan.sample_through_batch``):
        the two per-history uniforms come from one ``PCG64(seed)`` stream advanced to
        ``2 * history_offset``, so history ``h`` always consumes draws ``2h, 2h + 1``
        (chunk-invariant). That is a different stream from :meth:`emit`'s per-history
        spawn: the routes draw independent rays, compared statistically, never bit-wise.
        """
        bitgen = np.random.PCG64(seed)
        bitgen.advance(2 * history_offset)
        draws = np.random.Generator(bitgen).random((n, 2))
        points, weights = self._rays(draws[:, 0], draws[:, 1])
        f32 = np.float32
        return {
            "particle_type": np.ones(n, dtype=np.int32),
            "energy": np.full(n, self._beam.energy, dtype=f32),
            "x": points[:, 0].astype(f32),
            "y": points[:, 1].astype(f32),
            "z": points[:, 2].astype(f32),
            "ux": np.zeros(n, dtype=f32),
            "uy": np.zeros(n, dtype=f32),
            "uz": np.ones(n, dtype=f32),
            "weight": weights.astype(f32),
        }


class EdgeDerivativeSource(_EdgeSampledSource):
    """The exact estimator: rays sampled from ``|dT/dx|`` across the edge.

    With ``t`` uniform, the ray sits at the depth into the block where the isolated
    block transmits ``T_min + t (1 - T_min)`` — the inverse of the monotone
    edge profile, so the rays follow the total-variation measure of ``T0``. A
    straight face (a jump) maps every ``t`` onto the edge line: that is the
    boundary term pathwise differentiation omits. The profile is inverted once at
    construction by bisection on the exact chord geometry and tabulated on
    ``n_table`` nodes in ``t``. Module docstring: the identity and the stated
    prototype restrictions.
    """

    def __init__(
        self,
        beam: ParallelBeamSource,
        stack: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        parameter: EdgeParameter,
        *,
        n_table: int = 4097,
    ) -> None:
        """Tabulate the inverse edge profile of ``parameter`` for this beam."""
        super().__init__(beam, stack, cross_sections, parameter)
        if n_table < 2:
            raise ValueError(f"n_table must be at least 2, got {n_table}")
        self._fractions = np.linspace(0.0, 1.0, n_table)
        key = (*self._profile_key, n_table)
        cached = _PROFILES.get(key)
        if cached is not None and cached[0] is cross_sections:
            self._depths = cached[1]
            return
        targets = self._t_min + self._fractions * (1.0 - self._t_min)
        # Smallest depth whose transmission is <= target: T0 decreases with depth,
        # so this is the inverse at a smooth point and the jump location at a jump.
        lo = np.zeros(n_table)
        hi = np.full(n_table, self._span)
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            above = self._isolated_transmission(mid) > targets
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        self._depths = hi
        _PROFILES[key] = (cross_sections, hi)

    def edge_coordinate(self, fraction: float) -> float:
        """Map a uniform ``fraction`` to its across-edge coordinate ``q`` (cm)."""
        return float(self._q_for(np.array([fraction]))[0])

    def _q_for(self, fractions: _F64) -> _F64:
        depths = np.interp(fractions, self._fractions, self._depths)
        return np.asarray(self._x + self._sign * depths, dtype=np.float64)


class BixelChainRuleSource(_EdgeSampledSource):
    """What direct aperture optimization computes from a bixel Dij.

    DAO (Shepard et al., Med. Phys. 29, 1007 (2002), doi:10.1118/1.1477415) maps a
    leaf tip to bixel weights by the open fraction of the bixel it sits in,
    ``w_k = f_k + (1 - f_k) T_min``, so ``dw_k/dx = (1 - T_min) / pitch`` and the
    gradient is that bixel's Dij column times it. As a source: the same derivative
    measure as :class:`EdgeDerivativeSource`, spread uniformly over the bixel
    instead of over the tip's true penumbra. ``bixel_origin`` (local ``q``, cm) and
    ``bixel_pitch`` define the bixel lattice; an edge exactly on a bixel boundary
    belongs to the bixel above it.
    """

    def __init__(
        self,
        beam: ParallelBeamSource,
        stack: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        parameter: EdgeParameter,
        *,
        bixel_pitch: float,
        bixel_origin: float = 0.0,
    ) -> None:
        """Locate the bixel holding the edge on the given lattice."""
        super().__init__(beam, stack, cross_sections, parameter)
        if bixel_pitch <= 0.0:
            raise ValueError(f"bixel_pitch must be positive, got {bixel_pitch}")
        index = math.floor((self._x - bixel_origin) / bixel_pitch)
        self._bixel = (bixel_origin + index * bixel_pitch, bixel_origin + (index + 1) * bixel_pitch)

    @property
    def bixel(self) -> tuple[float, float]:
        """The ``[lo, hi)`` across-edge extent of the bixel holding the edge (cm)."""
        return self._bixel

    def _q_for(self, fractions: _F64) -> _F64:
        return self._bixel[0] + (self._bixel[1] - self._bixel[0]) * fractions


class ProfileChainRuleSource(_EdgeSampledSource):
    """The bixel chain rule with bixel weights taken from the edge's own profile.

    DAO's open-fraction model treats the leaf end as sharp. Here bixel ``k`` carries
    its mean transmission instead, ``w_k(x) = (1/pitch) int_k T(q; x) dq``, so

        dw_k/dx = sign * m_k / pitch,    m_k = int_k |dT/dx| dq
                                             = the drop of T0 across bixel k,

    and the gradient ``sum_k Dij_k dw_k/dx`` needs nothing beyond the bixel Dij
    DAO already uses — only a different ``dw/dx``. As a source: bixel ``k`` is drawn
    with probability ``m_k / (1 - T_min)`` and the ray placed uniformly inside it
    (one uniform: the bixel from the cumulative masses, the in-bixel position from
    the residual). The masses come from the exact chord geometry at the bixel
    boundaries, not from the inverse table. As ``bixel_pitch -> 0`` this is
    :class:`EdgeDerivativeSource`. The lattice convention is that of
    :class:`BixelChainRuleSource`.
    """

    def __init__(
        self,
        beam: ParallelBeamSource,
        stack: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        parameter: EdgeParameter,
        *,
        bixel_pitch: float,
        bixel_origin: float = 0.0,
    ) -> None:
        """Integrate the edge profile over each bixel of the lattice it crosses."""
        super().__init__(beam, stack, cross_sections, parameter)
        if bixel_pitch <= 0.0:
            raise ValueError(f"bixel_pitch must be positive, got {bixel_pitch}")
        relative = (self._x - bixel_origin) / bixel_pitch
        n_bixels = math.ceil(self._span / bixel_pitch) + 1
        steps = np.arange(n_bixels)
        if self._sign > 0.0:
            index = math.floor(relative) + steps
            far_edge = bixel_origin + (index + 1) * bixel_pitch
        else:
            index = math.ceil(relative) - 1 - steps
            far_edge = bixel_origin + index * bixel_pitch
        depths = np.minimum(np.abs(far_edge - self._x), self._span)
        # Transmission just outside the block is 1 (a straight face's own plane
        # counts as inside, so start from the limit rather than from depth 0).
        profile = np.concatenate([[1.0], self._isolated_transmission(depths)])
        masses = np.maximum(-np.diff(profile), 0.0)
        keep = masses > 0.0
        self._lows = bixel_origin + index[keep] * bixel_pitch
        self._cdf = np.cumsum(masses[keep]) / masses[keep].sum()
        self._pitch = bixel_pitch

    @property
    def bixel_masses(self) -> tuple[_F64, _F64]:
        """``(bixel lower edges, normalized masses m_k / (1 - T_min))`` with m_k > 0."""
        masses = np.diff(np.concatenate([[0.0], self._cdf]))
        return np.asarray(self._lows, dtype=np.float64), np.asarray(masses, dtype=np.float64)

    def _q_for(self, fractions: _F64) -> _F64:
        k = np.minimum(np.searchsorted(self._cdf, fractions, side="right"), self._cdf.size - 1)
        lower = np.where(k > 0, self._cdf[np.maximum(k - 1, 0)], 0.0)
        mass = self._cdf[k] - lower
        residual = np.clip((fractions - lower) / mass, 0.0, 1.0)
        return np.asarray(self._lows[k] + self._pitch * residual, dtype=np.float64)


class EdgeGradientBeamletSource(BeamletSource):
    """Many edge-gradient sources as the beamlets of one run.

    Beamlet ``j`` emits exactly edge source ``j``'s rays, so a beamlet run of this
    source computes every edge's ``dD/dx`` at once: as Dij columns (``run_dij``)
    or, what an optimizer needs, already contracted with ``dL/dD``
    (``run_beamlet_contraction``). Weights stay **signed** — a closing edge has a
    negative derivative — which a contraction sums as they are; ``run_dij``'s
    truncation is relative to a positive column maximum and is no place for
    negative columns, so prefer the contraction (or carry the signs outside).
    """

    def __init__(self, edges: Sequence[Source]) -> None:
        """Wrap the per-edge sources in beamlet order."""
        if not edges:
            raise ValueError("an edge set needs at least one edge")
        self._edges = tuple(edges)

    @property
    def edges(self) -> tuple[Source, ...]:
        """The per-edge sources, in beamlet order."""
        return self._edges

    @property
    def max_energy(self) -> float:
        """The highest energy any edge emits."""
        return max(edge.max_energy for edge in self._edges)

    @property
    def n_beamlets(self) -> int:
        """One beamlet per edge."""
        return len(self._edges)

    def emit(self, beamlet: int, rng_state: RNGState) -> Primary:
        """Emit one ray of edge ``beamlet``."""
        return self._edges[beamlet].emit(rng_state)

    def sample_beamlet_batch(
        self, seed: int, history_offset: int, n: int, beamlet: int
    ) -> dict[str, NDArray[Any]]:
        """Return edge ``beamlet``'s vectorized batch."""
        return self._edges[beamlet].sample_batch(seed, history_offset, n)


class RayFiniteDifferenceSource(Source):
    """The common-random-number central difference, as one reweighted run.

    Emits the forward beam's rays with weight ``(T(x + h) - T(x - h)) / 2h``: the
    two finite-difference runs share every history, so their difference is one run
    with a combined weight, and its batch sigma is the difference's sigma. Unbiased
    for the finite difference at step ``h`` (not for the derivative: ``O(h^2)``),
    for any beam. It spends most histories on rays the edge never touches — the
    reason it is the oracle here and not the estimator. As ``h -> 0`` its mean stays
    the finite difference (a jump is still caught) while its variance grows as
    ``1/h``; it does not become the pathwise derivative.
    """

    def __init__(
        self,
        beam: Source,
        stack: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        parameter: EdgeParameter,
        *,
        step: float,
    ) -> None:
        """Build the transmission tables of the two displaced stacks."""
        if step <= 0.0:
            raise ValueError(f"step must be positive, got {step}")
        x = edge_value(stack, parameter)
        self._beam = beam
        self._step = step
        self._plus = _StackTransmission(
            with_edge(stack, parameter, x + step),
            cross_sections,
            beam.max_energy,
            _E_MIN,
            _N_MU_TABLE,
        )
        self._minus = _StackTransmission(
            with_edge(stack, parameter, x - step),
            cross_sections,
            beam.max_energy,
            _E_MIN,
            _N_MU_TABLE,
        )

    @property
    def max_energy(self) -> float:
        """The wrapped beam's maximum energy."""
        return self._beam.max_energy

    def _factors(self, energies: _F64, origins: _F64, directions: _F64) -> _F64:
        """Give the finite-difference weight factor per ray (one arithmetic, both routes)."""
        plus = self._plus.factors(energies, origins, directions)
        minus = self._minus.factors(energies, origins, directions)
        return np.asarray((plus - minus) / (2.0 * self._step), dtype=np.float64)

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit the beam's primary with the finite-difference weight; no extra draws."""
        primary = self._beam.emit(rng_state)
        factor = self._factors(
            np.array([primary.energy]),
            np.array([[primary.x, primary.y, primary.z]]),
            np.array([[primary.ux, primary.uy, primary.uz]]),
        )
        return primary._replace(weight=primary.weight * float(factor[0]))

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, NDArray[Any]]:
        """Return the beam's batch with the weight column reweighted (vectorized)."""
        columns = self._beam.sample_batch(seed, history_offset, n)
        origins = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
        directions = np.stack([columns["ux"], columns["uy"], columns["uz"]], axis=1).astype(
            np.float64
        )
        factors = self._factors(columns["energy"].astype(np.float64), origins, directions)
        out = dict(columns)
        out["weight"] = (columns["weight"].astype(np.float64) * factors).astype(np.float32)
        return out


class PathwiseDerivativeSource(Source):
    """What autodiff of the whole simulation returns: ``dT/dx`` at each fixed ray.

    Emits the forward beam's rays with weight

        w_beam * dT/dx = w_beam * T(ray) * (-(mu/rho) rho) * d(chord_block)/dx,

    the chord derivative of the parameter's block taken by Warp's tape
    (:func:`pyradmc.backends.warp.aperture_ad.edge_chord_gradients`). Because the
    transport after the stack is independent of ``x``, this is exactly the
    gradient a fully taped run would produce, if the transport kernels could be
    taped at all. Valid for any beam, divergent included: the derivative is per
    ray, at that ray's direction.

    It is unbiased only where ``T`` is continuous in ``x`` along every ray — a
    rounded tip, or a straight face crossed obliquely — and there its variance
    diverges (logarithmically) at a tip's apex, where the chord grows like
    ``sqrt(delta)``. Where the beam hits a straight face end-on, ``T`` jumps, the
    tape returns zero and the whole derivative (the boundary term) is lost. Needs
    the ``warp`` extra; the energy uses ``cross_sections`` directly, which agrees
    with the stack tables at a monoenergetic beam's energy (a table node).
    """

    def __init__(
        self,
        beam: Source,
        stack: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        parameter: EdgeParameter,
        *,
        device: str = "cpu",
    ) -> None:
        """Bind the forward transmission and the parameter's isolated block."""
        from pyradmc.backends.warp.aperture_ad import edge_chord_gradients

        self._chord_gradients = edge_chord_gradients
        self._beam = beam
        self._frame = stack.frame
        self._block = isolated_block(stack, parameter).devices[0]
        self._side = parameter.side
        self._device = device
        self._cross_sections = cross_sections
        self._forward = _StackTransmission(
            stack, cross_sections, beam.max_energy, _E_MIN, _N_MU_TABLE
        )

    @property
    def max_energy(self) -> float:
        """The wrapped beam's maximum energy."""
        return self._beam.max_energy

    def _factors(self, energies: _F64, origins: _F64, directions: _F64) -> _F64:
        """``dT/dx`` per ray: one arithmetic for both engine routes."""
        p_local, d_local = self._frame.to_local(origins, directions)
        _, d_chord = self._chord_gradients(
            self._block, self._side, p_local, d_local, device=self._device
        )
        unique, inverse = np.unique(energies, return_inverse=True)
        mu = (
            np.array(
                [
                    self._cross_sections.mu_over_rho_total(float(e), self._block.material)
                    for e in unique
                ]
            )[inverse]
            * self._block.density
        )
        transmission = self._forward.factors(energies, origins, directions)
        return np.asarray(-mu * transmission * d_chord, dtype=np.float64)

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit the beam's primary with the pathwise weight; no extra draws."""
        primary = self._beam.emit(rng_state)
        factor = self._factors(
            np.array([primary.energy]),
            np.array([[primary.x, primary.y, primary.z]]),
            np.array([[primary.ux, primary.uy, primary.uz]]),
        )
        return primary._replace(weight=primary.weight * float(factor[0]))

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, NDArray[Any]]:
        """Return the beam's batch reweighted by one taped launch over all its rays."""
        columns = self._beam.sample_batch(seed, history_offset, n)
        origins = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
        directions = np.stack([columns["ux"], columns["uy"], columns["uz"]], axis=1).astype(
            np.float64
        )
        factors = self._factors(columns["energy"].astype(np.float64), origins, directions)
        out = dict(columns)
        out["weight"] = (columns["weight"].astype(np.float64) * factors).astype(np.float32)
        return out


def _tip_tangent_ray(x: float, sign: float, z: float, radius: float) -> tuple[float, float]:
    """Mid-plane crossing ``q_t`` and depth ``w`` of a rounded tip's tangent focal ray.

    The tip's circle (radius ``R``, centre ``c = x + sign R`` on the mid-plane ``z``)
    seen from the focal spot: the root of ``z^2 (c - q)^2 = R^2 (q^2 + z^2)`` on the
    open side, and the depth of the tangent point (the foot of the perpendicular from
    the centre). Defined whether or not that point lies inside the slab.
    """
    c = x + sign * radius
    root = math.sqrt(z * z + c * c - radius * radius)
    q_t = (c * z * z - sign * z * radius * root) / (z * z - radius * radius)
    norm = math.hypot(q_t, z)
    foot = (c * q_t + z * z) / norm
    return q_t, foot * z / norm


def _tip_support_start(
    x: float, sign: float, z: float, radius: float, z_top: float, z_bottom: float
) -> float | None:
    """Evaluate the rounded-tip branch of :func:`_support_start` on plain floats.

    The single scalar definition: :func:`_support_start` calls it, and the
    vectorized :func:`tip_support_starts` defers to it wherever its own slab
    decision could differ in the last bit (``np.hypot`` and ``math.hypot`` are
    not bit-identical) or its arithmetic is not finite.
    """
    q_t, w_tangent = _tip_tangent_ray(x, sign, z, radius)
    if not z_top <= w_tangent <= z_bottom:
        return None
    return q_t


def _support_start(block: JawPair | MLC, x: float, sign: float, z_mid: float) -> float | None:
    """Mid-plane coordinate of the first focal ray with a chord through ``block``.

    A rounded tip (circle of radius ``R`` centred at ``c = x + sign R`` on the
    mid-plane, in the ``(u, w)`` projection where the fan geometry is
    ``v``-invariant): the tangent from the focal spot, the root of
    ``z^2 (c - q)^2 = R^2 (q^2 + z^2)`` on the open side,

        q0 = (c z^2 - sign z R sqrt(z^2 + c^2 - R^2)) / (z^2 - R^2),

    valid while the tangent point lies inside the slab (``None`` otherwise, for a
    bisection fallback). A straight face at ``e``: the ray through its far corner,
    ``q0 = e z_mid / w_far`` with ``w_far`` the slab face the diverging ray reaches
    the face at first. A focused face is crossed end-on: ``q0 = x``.
    """
    z = z_mid
    if isinstance(block, MLC):
        return _tip_support_start(x, sign, z, block.tip_radius, block.z_top, block.z_bottom)
    if block.focused:
        return x
    w_far = block.z_bottom if sign * x >= 0.0 else block.z_top
    return x * z / w_far


# A vectorized slab decision closer than this (relative) to a slab face is re-decided
# by the scalar definition: np.hypot and math.hypot differ by an ulp on ~18 % of
# inputs (measured), which moves the tangent point by a few ulps, far below this.
_SLAB_DECISION_MARGIN = 1.0e-12


def tip_support_starts(
    x: _F64,
    sign: _F64,
    z: _F64,
    radius: _F64,
    z_top: _F64,
    z_bottom: _F64,
) -> tuple[_F64, NDArray[np.bool_]]:
    """Vectorized :func:`_tip_support_start`: ``(q0, closed_form)`` per rounded tip.

    The same float64 operations in the same order (``np.sqrt`` is correctly
    rounded, as ``math.sqrt`` is), so ``q0`` is bit-identical wherever the closed
    form applies. Only the tangent-point slab test goes through ``np.hypot``; an
    element whose decision falls within ``_SLAB_DECISION_MARGIN`` of a slab face, or
    whose arithmetic is not finite, is re-decided (or made to raise) by the scalar
    definition itself. The first array is the tangent ray's crossing ``q_t`` for
    every tip (:func:`_tip_tangent_ray`'s, bit for bit); ``closed_form`` is False
    where the tangent point leaves the slab, and there the support start is not
    ``q_t`` but needs :func:`tip_support_bisection`.
    """
    with np.errstate(all="ignore"):
        c = x + sign * radius
        root = np.sqrt(z * z + c * c - radius * radius)
        q0 = (c * z * z - sign * z * radius * root) / (z * z - radius * radius)
        norm = np.hypot(q0, z)
        foot = (c * q0 + z * z) / norm
        w_tangent = foot * z / norm
        closed = (z_top <= w_tangent) & (w_tangent <= z_bottom)
        margin = _SLAB_DECISION_MARGIN * np.maximum(np.abs(z_top), np.abs(z_bottom))
        doubtful = (
            ~np.isfinite(w_tangent)
            | (np.abs(w_tangent - z_top) <= margin)
            | (np.abs(w_tangent - z_bottom) <= margin)
        )
    for k in np.flatnonzero(doubtful):
        start = _tip_support_start(
            float(x[k]), float(sign[k]), float(z[k]), float(radius[k]),
            float(z_top[k]), float(z_bottom[k]),
        )  # fmt: skip
        closed[k] = start is not None
        if start is not None:
            q0[k] = start
    return q0, closed


def _isolated_tip_chords(
    local: _F64,
    z_top: _F64,
    z_bottom: _F64,
    radius: _F64,
    edge_lo: _F64,
    edge_hi: _F64,
    tip_pos: _F64,
    tip_neg: _F64,
) -> _F64:
    """:meth:`MLC.path_lengths` of one focused-side leaf pair per ray, parameters per ray.

    Ray ``k`` starts at the focal spot along ``local[k]`` and meets its own
    one-pair block (strip ``[edge_lo, edge_hi]``, tips ``tip_pos``/``tip_neg``).
    The interval primitives and the expressions are :meth:`MLC.path_lengths`'s, in
    its order (the ``+u`` leaf first), so each chord is that method's float64 value
    for the same block.
    """
    p = np.zeros_like(local)
    p_u, p_v, p_w = p[:, 0], p[:, 1], p[:, 2]
    d_u, d_v, d_w = local[:, 0], local[:, 1], local[:, 2]
    z_mid = (z_top + z_bottom) / 2.0
    slab = _slab_interval(p_w, d_w, z_top, z_bottom)
    total = np.zeros(local.shape[0], dtype=np.float64)
    strip = _intersect(
        _focused_halfspace_interval(
            z_mid * p_v - edge_lo * p_w, z_mid * d_v - edge_lo * d_w, above=True
        ),
        _focused_halfspace_interval(
            z_mid * p_v - edge_hi * p_w, z_mid * d_v - edge_hi * d_w, above=False
        ),
    )
    in_strip = _intersect(slab, strip)
    for tip, sign in ((tip_pos, 1.0), (tip_neg, -1.0)):
        center_u = tip + sign * radius
        body = _halfspace_interval(p_u, d_u, center_u, above=sign > 0.0)
        disc = _disc_interval(p_u, d_u, p_w, d_w, center_u, z_mid, radius)
        total += (
            _length(_intersect(in_strip, body))
            + _length(_intersect(in_strip, disc))
            - _length(_intersect(_intersect(in_strip, body), disc))
        )
    return total


def tip_support_bisection(
    x: _F64,
    sign: _F64,
    span: _F64,
    r_c: _F64,
    z_top: _F64,
    z_bottom: _F64,
    radius: _F64,
    edge_lo: _F64,
    edge_hi: _F64,
) -> _F64:
    """Vectorized bisection fallback of :class:`FanEdgeDerivativeSource` for rounded tips.

    Per tip: the first mid-plane crossing ``q``, coming from outside the block, whose
    focal ray through ``(q, r_c)`` has a chord in the isolated block (the opposing
    leaf retracted to ``-+_RETRACT_CM``), by ``_BISECTION_STEPS`` halvings of
    ``[x - sign, x + sign span]``. The midpoints, the ray directions (``(q, r_c,
    z_mid)`` normalized as ``_directions`` does) and the chord test are the
    per-edge loop's, element by element, so every halving takes the same branch.
    """
    pos = sign > 0.0
    tip_pos = np.where(pos, x, _RETRACT_CM)
    tip_neg = np.where(pos, -_RETRACT_CM, x)
    z_mid = 0.5 * (z_top + z_bottom)
    lo = x - sign * 1.0
    hi = x + sign * span
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        local = np.stack([mid, r_c, z_mid], axis=1)
        local /= np.linalg.norm(local, axis=1, keepdims=True)
        chord = _isolated_tip_chords(
            local, z_top, z_bottom, radius, edge_lo, edge_hi, tip_pos, tip_neg
        )
        inside = chord > 0.0
        hi = np.where(inside, mid, hi)
        lo = np.where(inside, lo, mid)
    return np.asarray(hi, dtype=np.float64)


def fan_support_start(record: FanEdgeRecord) -> float:
    """Return the support start ``q0`` of a fan edge (closed form, see :func:`_support_start`)."""
    q0 = _support_start(record.block, record.x, record.sign, record.z_mid)
    if q0 is None:
        raise ValueError("the tangent point lies outside the slab; no closed-form start")
    return q0


def _require_fan_frame(beam: Source, frame: BeamFrame, defensive: float) -> None:
    """Refuse a beam, frame or defensive share the fan edge estimator cannot use."""
    if not isinstance(beam, SpectralBeamSource):
        raise ValueError("the fan estimator needs a SpectralBeamSource beam")
    if not np.allclose(frame.origin, beam.focal_point, atol=1.0e-9):
        raise ValueError("the beam's focal point must be the stack frame's origin")
    if not (
        np.allclose(beam.u_axis, frame.u_axis, atol=_AXIS_TOLERANCE)
        and np.allclose(beam.v_axis, frame.v_axis, atol=_AXIS_TOLERANCE)
    ):
        raise ValueError("the beam's aperture axes must be the frame's u and v axes")
    if not 0.0 <= defensive < 1.0:
        raise ValueError(f"the defensive share must be in [0, 1), got {defensive}")


def _require_fan_device(device: JawPair | MLC) -> None:
    """Refuse a parameter device the fan edge estimator cannot differentiate."""
    if isinstance(device, MLC) and not device.focused_sides:
        raise ValueError(
            "under a fan the MLC needs focused_sides=True (a focal ray stays in its strip)"
        )


@dataclass(frozen=True)
class _FanEdgeGeometry:
    """What a fan edge's sampling geometry takes from its device, but its position."""

    z_mid: float
    q_axis: int
    field: tuple[tuple[float, float], tuple[float, float]]
    r_range: tuple[float, float]
    scale: float
    span: float
    r_c: float


def _fan_edge_geometry(
    beam: SpectralBeamSource,
    origin: _F64,
    axes: _F64,
    device: JawPair | MLC,
    pair: int | None,
) -> _FanEdgeGeometry:
    """Mid-plane field, edge extent and measure scale of one edge of ``device``.

    The one definition behind :class:`FanEdgeDerivativeSource` and the vectorized
    :class:`~pyradmc.geometry.segment_edges.SegmentedEdgeSource`: the beam's
    aperture projected to the device mid-plane, the edge's extent ``r_range`` in it
    (the leaf strip for a tip), ``L_r / A`` and the proposal span. None of it
    depends on the edge's position.
    """
    z_mid = 0.5 * (device.z_top + device.z_bottom)
    q_axis = 0 if isinstance(device, MLC) or device.axis == "u" else 1
    centre = (np.asarray(beam.center, dtype=np.float64) - origin) @ axes.T
    if centre[2] <= 0.0:
        raise ValueError("the aperture must lie downstream of the focal spot")
    scale = z_mid / centre[2]
    half = (0.5 * beam.width_u * scale, 0.5 * beam.width_v * scale)
    field = (
        (centre[0] * scale - half[0], centre[0] * scale + half[0]),
        (centre[1] * scale - half[1], centre[1] * scale + half[1]),
    )
    r_field = field[1 - q_axis]
    if isinstance(device, MLC):
        assert pair is not None
        strip = (device.leaf_edges_v[pair], device.leaf_edges_v[pair + 1])
        r_range = (max(strip[0], r_field[0]), min(strip[1], r_field[1]))
    else:
        r_range = r_field
    r_length = r_range[1] - r_range[0]
    if r_length <= 0.0:
        raise ValueError("the parameter's edge lies outside the beam's field")
    tip = device.tip_radius if isinstance(device, MLC) else 0.0
    return _FanEdgeGeometry(
        z_mid=z_mid,
        q_axis=q_axis,
        field=field,
        r_range=r_range,
        scale=r_length / (4.0 * half[0] * half[1]),
        span=tip + (device.z_bottom - device.z_top) + 1.0,
        r_c=0.5 * (r_range[0] + r_range[1]),
    )


@dataclass(frozen=True)
class FanEdgeRecord:
    """One fan edge as plain data: the contract with an in-kernel generator.

    ``kind`` is 0 for a rounded MLC tip, 1 for a straight jaw face (both smooth: a
    proposal sampled as ``q = q0 + sign s^2``, ``s`` in ``[0, sqrt(span)]``) and 2 for a
    focused jaw (a jump: every ray on ``q = x``). ``block`` is the isolated block
    (:func:`isolated_block`), ``stack``/``device`` the full stack and the parameter's
    device in it: the rest-of-stack transmission is the stack's optical depth minus
    the block's. The proposal itself is built by the consumer — the derivative
    density along the edge's central ray ``r_c`` at ``e_ref`` on ``n_table`` nodes, with
    a ``defensive`` uniform share — so a device can tabulate it in parallel; any
    consistent proposal is unbiased. ``q_tangent`` is a rounded tip's tangent focal
    ray (``_tip_tangent_ray``; ``None`` for a jaw): ``q0`` itself when the tangent
    point lies in the slab, short of a bisected ``q0`` otherwise. A generator may use
    a tangent-ray closed form near it. The remaining fields are those of
    :class:`FanEdgeDerivativeSource`, whose ``_rays`` the generator mirrors.
    """

    stack: BeamLimitingStack
    device: int
    block: JawPair | MLC
    kind: int
    sign: float
    x: float
    q0: float
    z_mid: float
    q_axis: int
    r_range: tuple[float, float]
    field: tuple[tuple[float, float], tuple[float, float]]
    scale: float
    span: float
    r_c: float
    e_ref: float
    n_table: int
    defensive: float
    spectrum: Spectrum
    cross_sections: CrossSectionSource
    e_max: float
    q_tangent: float | None = None


_I32 = NDArray[np.int32]


@dataclass(frozen=True)
class FanEdgeColumns:
    """A whole fan edge set as host columns: what an in-kernel generator uploads.

    One entry per edge, in beamlet order, at float64 (int32 for indices); a device
    backend casts on upload (float32 for Warp). The fields are those of
    :class:`FanEdgeRecord` flattened: ``kind``/``sign``/``x``/``q0``/``z_mid``/
    ``q_axis``/``scale``/``r_c``/``e_ref``/``defensive`` as named, ``r_lo``/``r_hi``
    the edge's extent along itself, ``u_lo``..``v_hi`` the field at its mid-plane,
    ``b_*`` its isolated block (slab, tip radius, the strip edges of a tip or the
    retracted-and-moving edges of a jaw, a jaw's focusing); ``q_t`` is a rounded
    tip's tangent ray (``q0`` for a jaw) and ``t_off = sign (q0 - q_t)`` (float64,
    zero when ``q0`` is the tangent ray, the bisected start's lead otherwise), so
    ``s^2 + t_off`` is a ray's distance from the tangent ray. ``segment`` indexes the
    distinct stacks, whose devices are concatenated in ``stack`` (rows
    ``seg_lo[s]:seg_hi[s]``, frames in ``seg_origin``/``seg_rot``), and ``device`` is
    the edge's row there. ``table`` is the edge's proposal row (``-1`` for a jump
    edge, which has none; the cumulative proposal on it is the consumer's to build)
    and ``s_row`` the row of ``s`` holding its ``s`` grid, ``n_table`` nodes. Edges
    whose grids are equal may share a row: the per-edge route gives every smooth
    edge its own (``s_row == table``), the segmented route one per edge slot.

    Built from per-edge records by :func:`fan_edge_columns` and, without per-edge
    objects, by :meth:`~pyradmc.geometry.segment_edges.SegmentedEdgeSource.device_columns`
    (pinned equal, column by column).
    """

    kind: _I32
    segment: _I32
    device: _I32
    q_axis: _I32
    table: _I32
    s_row: _I32
    sign: _F64
    x: _F64
    q0: _F64
    z_mid: _F64
    r_lo: _F64
    r_hi: _F64
    u_lo: _F64
    u_hi: _F64
    v_lo: _F64
    v_hi: _F64
    scale: _F64
    r_c: _F64
    e_ref: _F64
    defensive: _F64
    b_z_top: _F64
    b_z_bottom: _F64
    b_radius: _F64
    b_lo: _F64
    b_hi: _F64
    b_focused: _I32
    q_t: _F64
    t_off: _F64
    n_table: int
    s: _F64
    seg_origin: _F64
    seg_rot: _F64
    seg_lo: _I32
    seg_hi: _I32
    stack: CompiledStack
    spectrum: Spectrum
    cross_sections: CrossSectionSource
    e_max: float

    @property
    def n_edges(self) -> int:
        """Number of edges (beamlets)."""
        return int(self.kind.shape[0])


_PER_DEVICE_STACK_FIELDS = (
    "kind", "z_top", "z_bottom", "material", "density", "jaw_axis", "jaw_edge_neg",
    "jaw_edge_pos", "jaw_focused", "mlc_tip_radius", "mlc_focused_sides", "mlc_n_pairs",
)  # fmt: skip


def _concatenate_stacks(stacks: Sequence[CompiledStack]) -> tuple[CompiledStack, list[int]]:
    """One CompiledStack holding every segment's devices; per-segment first device.

    The MLC leaf arrays are offset-addressed already, so shifting each segment's
    offsets keeps every per-device chord dispatcher valid on the combined table.
    """
    firsts = []
    total = 0
    edges_shift = 0
    tips_shift = 0
    parts: dict[str, list[NDArray[Any]]] = {}
    for c in stacks:
        firsts.append(total)
        total += c.n_devices
        for name in _PER_DEVICE_STACK_FIELDS:
            parts.setdefault(name, []).append(np.asarray(getattr(c, name)))
        parts.setdefault("mlc_edges_off", []).append(np.asarray(c.mlc_edges_off) + edges_shift)
        parts.setdefault("mlc_tips_off", []).append(np.asarray(c.mlc_tips_off) + tips_shift)
        parts.setdefault("leaf_edges_v", []).append(np.asarray(c.leaf_edges_v))
        parts.setdefault("tips_neg", []).append(np.asarray(c.tips_neg))
        parts.setdefault("tips_pos", []).append(np.asarray(c.tips_pos))
        edges_shift += len(c.leaf_edges_v)
        tips_shift += len(c.tips_pos)
    combined = CompiledStack(
        n_devices=total,
        origin=stacks[0].origin,
        rotation=stacks[0].rotation,
        **{name: np.concatenate(values) for name, values in parts.items()},
    )
    return combined, firsts


def fan_edge_columns(records: Sequence[FanEdgeRecord]) -> FanEdgeColumns:
    """Flatten per-edge records into the columns of one edge set.

    All edges must share one spectrum and one cross-section source and ``e_max`` (one
    beam quality per run) and the smooth edges one ``n_table``; segments are told
    apart by their stack *objects*, numbered in order of first appearance.
    """
    if not records:
        raise ValueError("an edge set needs at least one edge")
    first = records[0]
    for rec in records:
        if not np.array_equal(rec.spectrum.edges, first.spectrum.edges) or not np.array_equal(
            rec.spectrum.cdf, first.spectrum.cdf
        ):
            raise ValueError("an in-kernel edge set needs one shared spectrum")
        if rec.cross_sections is not first.cross_sections or rec.e_max != first.e_max:
            raise ValueError("an in-kernel edge set needs one cross-section source and e_max")

    stack_ids: dict[int, int] = {}
    compiled: list[CompiledStack] = []
    for rec in records:
        if id(rec.stack) not in stack_ids:
            stack_ids[id(rec.stack)] = len(compiled)
            compiled.append(rec.stack.flatten())
    combined, firsts = _concatenate_stacks(compiled)

    n_edges = len(records)
    smooth = [k for k, rec in enumerate(records) if rec.kind != 2]
    n_table = records[smooth[0]].n_table if smooth else 2
    if any(records[k].n_table != n_table for k in smooth):
        raise ValueError("every smooth edge of a set needs the same n_table")
    table_index = np.full(n_edges, -1, dtype=np.int32)
    table_index[smooth] = np.arange(len(smooth), dtype=np.int32)
    s_rows = [np.linspace(0.0, math.sqrt(records[k].span), n_table) for k in smooth]

    def column(get: Any) -> _F64:
        return np.array([get(rec) for rec in records], dtype=np.float64)

    def ints(values: Any) -> _I32:
        return np.asarray(values, dtype=np.int32)

    def block_lo(rec: FanEdgeRecord) -> float:
        b = rec.block
        return b.leaf_edges_v[0] if isinstance(b, MLC) else b.edge_neg

    def block_hi(rec: FanEdgeRecord) -> float:
        b = rec.block
        return b.leaf_edges_v[1] if isinstance(b, MLC) else b.edge_pos

    seg_of = [stack_ids[id(rec.stack)] for rec in records]
    return FanEdgeColumns(
        kind=ints([rec.kind for rec in records]),
        segment=ints(seg_of),
        device=ints([firsts[seg] + rec.device for seg, rec in zip(seg_of, records, strict=True)]),
        q_axis=ints([rec.q_axis for rec in records]),
        table=table_index,
        s_row=table_index.copy(),
        sign=column(lambda r: r.sign),
        x=column(lambda r: r.x),
        q0=column(lambda r: r.q0),
        z_mid=column(lambda r: r.z_mid),
        r_lo=column(lambda r: r.r_range[0]),
        r_hi=column(lambda r: r.r_range[1]),
        u_lo=column(lambda r: r.field[0][0]),
        u_hi=column(lambda r: r.field[0][1]),
        v_lo=column(lambda r: r.field[1][0]),
        v_hi=column(lambda r: r.field[1][1]),
        scale=column(lambda r: r.scale),
        r_c=column(lambda r: r.r_c),
        e_ref=column(lambda r: r.e_ref),
        defensive=column(lambda r: r.defensive),
        b_z_top=column(lambda r: r.block.z_top),
        b_z_bottom=column(lambda r: r.block.z_bottom),
        b_radius=column(lambda r: r.block.tip_radius if isinstance(r.block, MLC) else 0.0),
        b_lo=column(block_lo),
        b_hi=column(block_hi),
        b_focused=ints(
            [0 if isinstance(rec.block, MLC) else int(rec.block.focused) for rec in records]
        ),
        q_t=column(lambda r: r.q0 if r.q_tangent is None else r.q_tangent),
        t_off=column(lambda r: 0.0 if r.q_tangent is None else r.sign * (r.q0 - r.q_tangent)),
        n_table=int(n_table),
        s=np.concatenate(s_rows) if s_rows else np.zeros(1),
        seg_origin=np.concatenate([c.origin for c in compiled]),
        seg_rot=np.concatenate([c.rotation for c in compiled]),
        seg_lo=ints(firsts),
        seg_hi=ints([f + c.n_devices for f, c in zip(firsts, compiled, strict=True)]),
        stack=combined,
        spectrum=first.spectrum,
        cross_sections=first.cross_sections,
        e_max=first.e_max,
    )


class FanEdgeDerivativeSource(Source):
    """The exact edge gradient under a divergent fan, by importance sampling.

    A fan ray is fixed by its crossing ``(q, r)`` of the parameter device's
    mid-plane (``q`` across the edge, ``r`` along it). Moving the edge is no longer
    a translation of these rays, so there is no single profile to invert; instead

        dD/dx = E_proposal[ (L_r / A) * dT/dx(ray, E) * T_rest(ray, E) / p(q) ],

    with ``r`` uniform over the edge's extent in the field (length ``L_r``), ``E``
    drawn from the beam's own spectrum, ``q`` from a tabulated proposal density
    ``p``, ``A`` the beam's aperture area projected to the mid-plane, and ``dT/dx``
    exact per ray: ``-mu(E) T_block dl/dx``, with the chord's edge derivative from
    the scalar twins (:func:`~pyradmc.geometry.collimation.mlc_tip_edge_derivative`,
    :func:`~pyradmc.geometry.collimation.jaw_edge_derivative`). Any proposal that
    covers the support is unbiased. This one is the derivative density itself along
    the edge's central ray at the spectrum's mean energy, so the weight ratio stays
    near one, plus a defensive uniform share. It is tabulated in
    ``s = sqrt(|q - q0|)`` from the support start ``q0``: a rounded tip's derivative
    grows like ``1/sqrt(q - q0)`` from the ray tangent to the tip, and in ``s`` that
    singularity is a finite density, matched by the proposal, so the ratio stays
    bounded. ``q0`` is found by bisection on the chord: under the fan the tangent ray
    crosses the mid-plane short of the apex (by about ``R alpha^2 / 2``, 0.3 mm 5 cm
    off axis), so the parallel profile's support would miss a sliver of mass.

    A **focused jaw** seen from its focal spot is end-on: every ray sees the full
    slant thickness or nothing, ``T`` jumps at ``q = x``, and the derivative is the
    boundary term ``sign (1 - T_full) delta(q - x)``. It is sampled on the edge
    line, exactly as :class:`EdgeDerivativeSource` samples a straight face under a
    parallel beam.

    Geometry contract (refused otherwise): the beam is a :class:`SpectralBeamSource`
    whose focal point is the stack frame's origin and whose aperture axes are the
    frame's ``u`` and ``v`` axes (aperture plane normal to ``w``); an MLC parameter
    has focused sides, so a focal ray stays inside its strip through the slab (which
    also makes the strip-restricted rest of stack exact). Four uniforms per history:
    two for the energy, one along the edge, one across it. The host batch route
    evaluates the chord derivative with a per-ray Python loop: correct, not fast.
    """

    def __init__(
        self,
        beam: SpectralBeamSource,
        stack: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        parameter: EdgeParameter,
        *,
        n_table: int = 513,
        defensive: float = 0.05,
    ) -> None:
        """Validate the fan geometry and tabulate the proposal along the edge."""
        frame = stack.frame
        _require_fan_frame(beam, frame, defensive)
        device = _resolve(stack, parameter)
        _require_fan_device(device)
        self._beam = beam
        self._stack = stack
        self._device_index = parameter.device
        self._cross_sections = cross_sections
        self._axes = np.array([frame.u_axis, frame.v_axis, frame.w_axis])
        self._origin = np.asarray(frame.origin, dtype=np.float64)
        self._x = edge_value(stack, parameter)
        self._sign = 1.0 if parameter.side == "pos" else -1.0
        geometry = _fan_edge_geometry(beam, self._origin, self._axes, device, parameter.pair)
        self._z_mid = geometry.z_mid
        self._q_axis = geometry.q_axis
        self._field = geometry.field
        self._r_range = geometry.r_range
        self._scale = geometry.scale

        isolated = isolated_block(stack, parameter)
        self._block = isolated.devices[0]
        self._block_t = _transmission_for(isolated, cross_sections, beam.max_energy)
        rest = _retracted(stack, parameter)
        if isinstance(device, MLC):
            rest = _strip_restricted(rest, parameter)
        self._other = _transmission_for(rest, cross_sections, beam.max_energy)
        self._jump = isinstance(device, JawPair) and device.focused
        self._span = geometry.span
        self._r_c = geometry.r_c
        self._n_table = n_table
        self._defensive = defensive
        self._q0 = self._x
        self._tables: tuple[_F64, _F64, _F64] | None = None
        self._q_tangent: float | None = None
        if self._jump:
            return
        q0 = _support_start(self._block, self._x, self._sign, self._z_mid)
        if isinstance(self._block, MLC):
            self._q_tangent = _tip_tangent_ray(
                self._x, self._sign, self._z_mid, self._block.tip_radius
            )[0]
        if q0 is None:
            # Support start by bisection on the chord: the first crossing, coming
            # from outside the block, with a chord.
            lo, hi = self._x - self._sign * 1.0, self._x + self._sign * self._span
            for _ in range(_BISECTION_STEPS):
                mid = 0.5 * (lo + hi)
                if self._block_chords(np.array([mid]), np.array([self._r_c]))[0] > 0.0:
                    hi = mid
                else:
                    lo = mid
            q0 = hi
        self._q0 = q0

    def _proposal(self) -> tuple[_F64, _F64, _F64]:
        """Return the host proposal tables ``(s, cdf, cell_density)``, built on first use.

        Only the host routes need them (an in-kernel generator tabulates its own
        from :meth:`device_record`), so construction stays cheap for an optimizer
        that rebuilds every edge each iteration.
        """
        if self._tables is None:
            n = self._n_table
            s_max = math.sqrt(self._span)
            s = np.linspace(0.0, s_max, n)
            q = self._q0 + self._sign * s**2
            e_ref = np.full(n, self._beam.spectrum.mean_energy)
            density = np.abs(self._derivative(q, np.full(n, self._r_c), e_ref))
            f = 2.0 * s * density
            mass = 0.5 * (f[:-1] + f[1:]) * np.diff(s)
            uniform_share = np.diff(s) / s_max
            if mass.sum() > 0.0:
                mass = (1.0 - self._defensive) * mass / mass.sum() + self._defensive * uniform_share
            else:
                mass = uniform_share
            self._tables = (s, np.cumsum(mass) / mass.sum(), mass / mass.sum() / np.diff(s))
        return self._tables

    @property
    def max_energy(self) -> float:
        """The beam spectrum's top energy."""
        return self._beam.max_energy

    def device_record(self) -> FanEdgeRecord:
        """Everything an in-kernel generator needs for this edge, as plain data."""
        block = self._block
        if self._jump:
            kind = 2
        elif isinstance(block, MLC):
            kind = 0
        else:
            kind = 1
        return FanEdgeRecord(
            stack=self._stack,
            device=self._device_index,
            block=block,
            kind=kind,
            sign=self._sign,
            x=self._x,
            q0=self._q0,
            z_mid=self._z_mid,
            q_axis=self._q_axis,
            r_range=self._r_range,
            field=self._field,
            scale=self._scale,
            span=self._span,
            r_c=self._r_c,
            e_ref=self._beam.spectrum.mean_energy,
            n_table=self._n_table,
            defensive=self._defensive,
            spectrum=self._beam.spectrum,
            cross_sections=self._cross_sections,
            e_max=self._beam.max_energy,
            q_tangent=self._q_tangent,
        )

    def _directions(self, q: _F64, r: _F64) -> tuple[_F64, _F64]:
        """Local and engine-frame unit directions of the focal rays through (q, r)."""
        u, v = (q, r) if self._q_axis == 0 else (r, q)
        local = np.stack([u, v, np.full_like(u, self._z_mid)], axis=1)
        local /= np.linalg.norm(local, axis=1, keepdims=True)
        return local, local @ self._axes

    def _block_chords(self, q: _F64, r: _F64) -> _F64:
        """Chord of each focal ray through the parameter's isolated block (cm)."""
        local, _ = self._directions(q, r)
        return np.asarray(self._block.path_lengths(np.zeros_like(local), local), dtype=np.float64)

    def _chord_rates(self, local: _F64) -> _F64:
        """Evaluate the block chord's pathwise edge derivative per local ray (scalar twins)."""
        block = self._block
        pos = 1 if self._sign > 0.0 else 0
        out = np.empty(len(local))
        for k, (du, dv, dw) in enumerate(local):
            if isinstance(block, MLC):
                out[k] = mlc_tip_edge_derivative(
                    0.0, 0.0, 0.0, du, dv, dw, block.z_top, block.z_bottom, block.tip_radius,
                    block.leaf_edges_v[0], block.leaf_edges_v[1], self._x, 1, pos,
                )  # fmt: skip
            else:
                d_q = du if block.axis == "u" else dv
                out[k] = jaw_edge_derivative(
                    0.0, d_q, 0.0, dw, block.z_top, block.z_bottom, block.edge_neg,
                    block.edge_pos, 1 if block.focused else 0, pos,
                )  # fmt: skip
        return out

    def _derivative(self, q: _F64, r: _F64, energies: _F64) -> _F64:
        """``dT_block/dx`` per focal ray: ``-mu(E) T_block dl/dx``."""
        local, engine = self._directions(q, r)
        origins = np.tile(self._origin, (len(q), 1))
        mu = self._block_t.linear_attenuation(energies)[:, 0]
        transmission = self._block_t.factors(energies, origins, engine)
        return np.asarray(-mu * transmission * self._chord_rates(local), dtype=np.float64)

    def _rays(self, energies: _F64, u_r: _F64, u_q: _F64) -> tuple[_F64, _F64]:
        """Engine-frame directions and weights for one batch of draws.

        The single arithmetic behind :meth:`emit` and :meth:`sample_batch`.
        """
        n = len(energies)
        r = self._r_range[0] + (self._r_range[1] - self._r_range[0]) * u_r
        origins = np.tile(self._origin, (n, 1))
        if self._jump:
            q = np.full(n, self._x)
            _, engine = self._directions(q, r)
            # Just inside the block, where the ray sees the full slant thickness.
            _, inside = self._directions(q + self._sign * 1.0e-7, r)
            full = self._block_t.factors(energies, origins, inside)
            core = self._sign * (1.0 - full)
        else:
            s_grid, cdf, cell_density = self._proposal()
            k = np.minimum(np.searchsorted(cdf, u_q, side="right"), cdf.size - 1)
            lower = np.where(k > 0, cdf[np.maximum(k - 1, 0)], 0.0)
            residual = np.clip((u_q - lower) / (cdf[k] - lower), 0.0, 1.0)
            s = s_grid[k] + (s_grid[k + 1] - s_grid[k]) * residual
            q = self._q0 + self._sign * s**2
            _, engine = self._directions(q, r)
            # Proposal density in q: the cell's density in s over |dq/ds| = 2 s.
            density = cell_density[k] / np.maximum(2.0 * s, 1.0e-300)
            core = np.where(s > 0.0, self._derivative(q, r, energies) / density, 0.0)
        local_u, local_v = (q, r) if self._q_axis == 0 else (r, q)
        inside_field = (
            (local_u >= self._field[0][0])
            & (local_u <= self._field[0][1])
            & (local_v >= self._field[1][0])
            & (local_v <= self._field[1][1])
        )
        other = self._other.factors(energies, origins, engine)
        weights = np.where(inside_field, self._scale * core * other, 0.0)
        return engine, np.asarray(weights, dtype=np.float64)

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one focal ray along the edge; consumes exactly four uniforms."""
        energy = self._beam.spectrum.sample_energy(rng_state)
        u_r = uniform(rng_state)
        u_q = uniform(rng_state)
        directions, weights = self._rays(np.array([energy]), np.array([u_r]), np.array([u_q]))
        d = directions[0]
        x0, y0, z0 = (float(c) for c in self._origin)
        return Primary(energy, x0, y0, z0, d[0], d[1], d[2], weight=float(weights[0]))

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, NDArray[Any]]:
        """Pre-sample on the chunk-invariant ``PCG64(seed)`` stream, four draws per history."""
        bitgen = np.random.PCG64(seed)
        bitgen.advance(4 * history_offset)
        draws = np.random.Generator(bitgen).random((n, 4))
        energies = self._beam.spectrum.sample_energies(draws[:, 0], draws[:, 1])
        directions, weights = self._rays(energies, draws[:, 2], draws[:, 3])
        f32 = np.float32
        return {
            "particle_type": np.ones(n, dtype=np.int32),
            "energy": energies.astype(f32),
            "x": np.full(n, self._origin[0], dtype=f32),
            "y": np.full(n, self._origin[1], dtype=f32),
            "z": np.full(n, self._origin[2], dtype=f32),
            "ux": directions[:, 0].astype(f32),
            "uy": directions[:, 1].astype(f32),
            "uz": directions[:, 2].astype(f32),
            "weight": weights.astype(f32),
        }
