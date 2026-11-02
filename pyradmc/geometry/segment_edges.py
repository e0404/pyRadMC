"""Every leaf and jaw edge of a segmented fan as the beamlets of one gradient source.

An aperture optimizer needs ``dL/dx`` for every movable edge of every segment, once
per iteration. :class:`~pyradmc.geometry.aperture_gradient.EdgeGradientBeamletSource`
takes one :class:`~pyradmc.geometry.aperture_gradient.FanEdgeDerivativeSource` per
edge, and building those objects (validation, isolated and retracted stacks, their
transmission models) cost 0.13 ms each: 51 ms for the 400 edges of a 10-segment,
20-pair plan, and seconds for an arc's ~15 000. :class:`SegmentedEdgeSource` lists
the same edges over a :class:`~pyradmc.geometry.segments.SegmentedFanSource` —
segments share the stack structure and differ in their settings arrays and, for the
control points of an arc, in their fan and beam frame — and builds the plain-data
columns an in-kernel generator reads
(:class:`~pyradmc.geometry.aperture_gradient.FanEdgeColumns`) with array arithmetic:

- everything that does not depend on an edge's position (mid-plane field, the
  edge's extent, ``L_r / A``, the proposal span and ``s`` grid, the isolated block's
  slab) is evaluated per device and leaf pair by the per-edge source's own helper,
  once per distinct (fan, frame) pair, and laid out over the segments;
- the positions come straight from the settings arrays, and the support start
  ``q0`` from the vectorized closed forms
  (:func:`~pyradmc.geometry.aperture_gradient.tip_support_starts`, with the
  bisection fallback :func:`~pyradmc.geometry.aperture_gradient.tip_support_bisection`
  where a rounded tip's tangent point leaves the slab);
- the segments' stacks are concatenated by tiling the template's flattened layout
  and inserting each segment's settings and frame, as the per-edge route
  concatenates ``flatten()`` of each segment's stack.

The columns are pinned **equal**, element for element in float64, to
:func:`~pyradmc.geometry.aperture_gradient.fan_edge_columns` of the per-edge records
built on :meth:`~pyradmc.geometry.segments.SegmentedFanSource.segment_stack`, so the
Warp engine's in-kernel route returns bit-identical contractions either way. Physics,
estimator and stated approximations are :class:`FanEdgeDerivativeSource`'s.

**Beamlet order**: segment-major; inside a segment the selected devices in the order
given (stack order by default); inside an MLC leaf pair by pair; inside a pair or a
jaw the ``"neg"`` edge then the ``"pos"`` edge. :attr:`SegmentedEdgeSource.parameters`
lists ``(segment, EdgeParameter)`` per beamlet and :attr:`SegmentedEdgeSource.values`
the edge coordinates.

**Host routes** (the reference engine, the Warp host fallback for a subclass): beamlet
``k`` delegates to the equivalent :class:`FanEdgeDerivativeSource`, built on first
use and kept. That per-edge source is the oracle, so its speed does not matter here.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyradmc.geometry.aperture_gradient import (
    _RETRACT_CM,
    EdgeParameter,
    FanEdgeColumns,
    FanEdgeDerivativeSource,
    _fan_edge_geometry,
    _require_fan_device,
    _require_fan_frame,
    tip_support_bisection,
    tip_support_starts,
)
from pyradmc.geometry.collimation import MLC, CompiledStack
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import BeamletSource, Primary
from pyradmc.rng import RNGState

__all__ = ["SegmentedEdgeSource"]

_F64 = NDArray[np.float64]
_I32 = NDArray[np.int32]

_DEFAULT_N_TABLE = 513
_DEFAULT_DEFENSIVE = 0.05


@dataclass(frozen=True)
class _Slots:
    """The edges of one segment, position-free: one entry per (device, pair, side)."""

    device: _I32  # template device index
    pair: NDArray[np.int64]  # leaf pair, -1 for a jaw
    column: NDArray[np.int64]  # column of the settings matrix holding the position
    kind: _I32
    sign: _F64
    q_axis: _I32
    z_mid: _F64
    r_lo: _F64
    r_hi: _F64
    u_lo: _F64
    u_hi: _F64
    v_lo: _F64
    v_hi: _F64
    scale: _F64
    span: _F64
    r_c: _F64
    z_top: _F64
    z_bottom: _F64
    radius: _F64
    strip_lo: _F64
    strip_hi: _F64
    focused: _I32
    s_rows: _F64  # (n_slots, n_table): the s grid of each slot (unused for a jump)


def _slot_table(
    segments: SegmentedFanSource,
    devices: tuple[int, ...],
    n_table: int,
    segment: int,
    compiled: CompiledStack,
    prefix: str,
) -> _Slots:
    """Evaluate the position-free part of every edge of one segment, per edge slot.

    ``_fan_edge_geometry`` is the per-edge source's own helper, so every value here
    is the one :class:`FanEdgeDerivativeSource` computes for that edge on segment
    ``segment``'s fan and frame. ``compiled`` is the template's ``flatten()``.
    """
    fan = segments.fans[segment]
    template = segments.template
    frame = segments.frames[segment]
    origin = np.asarray(frame.origin, dtype=np.float64)
    axes = np.array([frame.u_axis, frame.v_axis, frame.w_axis])
    n_tips = int(segments.tips_neg.shape[1])
    n_devices = len(template.devices)
    rows: list[tuple[Any, ...]] = []
    for j in devices:
        device = template.devices[j]
        _require_fan_device(device)
        pairs: Sequence[int | None] = range(device.n_pairs) if isinstance(device, MLC) else (None,)
        for pair in pairs:
            try:
                geometry = _fan_edge_geometry(fan, origin, axes, device, pair)
            except ValueError as error:
                where = f"device {j}" if pair is None else f"device {j}, pair {pair}"
                raise ValueError(f"{prefix}{where}: {error}") from None
            for side, sign in (("neg", -1.0), ("pos", 1.0)):
                if isinstance(device, MLC):
                    assert pair is not None
                    tip = int(compiled.mlc_tips_off[j]) + pair
                    column = tip if side == "neg" else n_tips + tip
                    kind = 0
                    strip = (device.leaf_edges_v[pair], device.leaf_edges_v[pair + 1])
                    radius = device.tip_radius
                    focused = 0
                else:
                    column = 2 * n_tips + (j if side == "neg" else n_devices + j)
                    kind = 2 if device.focused else 1
                    strip = (0.0, 0.0)
                    radius = 0.0
                    focused = int(device.focused)
                rows.append(
                    (
                        j, -1 if pair is None else pair, column, kind, sign,
                        geometry.q_axis, geometry.z_mid, geometry.r_range[0],
                        geometry.r_range[1], geometry.field[0][0], geometry.field[0][1],
                        geometry.field[1][0], geometry.field[1][1], geometry.scale,
                        geometry.span, geometry.r_c, device.z_top, device.z_bottom, radius,
                        strip[0], strip[1], focused,
                    )
                )  # fmt: skip
    cols = list(zip(*rows, strict=True))

    def f64(i: int) -> _F64:
        return np.array(cols[i], dtype=np.float64)

    def i32(i: int) -> _I32:
        return np.array(cols[i], dtype=np.int32)

    span = f64(14)
    s_rows = np.array([np.linspace(0.0, math.sqrt(float(sp)), n_table) for sp in cols[14]])
    return _Slots(
        device=i32(0),
        pair=np.array(cols[1], dtype=np.int64),
        column=np.array(cols[2], dtype=np.int64),
        kind=i32(3),
        sign=f64(4),
        q_axis=i32(5),
        z_mid=f64(6),
        r_lo=f64(7),
        r_hi=f64(8),
        u_lo=f64(9),
        u_hi=f64(10),
        v_lo=f64(11),
        v_hi=f64(12),
        scale=f64(13),
        span=span,
        r_c=f64(15),
        z_top=f64(16),
        z_bottom=f64(17),
        radius=f64(18),
        strip_lo=f64(19),
        strip_hi=f64(20),
        focused=i32(21),
        s_rows=s_rows,
    )


@dataclass(frozen=True)
class _Layout:
    """Everything of an edge set but the positions, tiled over the segments.

    The slot table plus every column and stack field the settings do not touch, at
    full length and read-only: shared by every edge set of this layout.
    """

    slots: _Slots
    static: dict[str, Any]  # FanEdgeColumns fields
    stack: dict[str, Any]  # CompiledStack fields
    straight: NDArray[np.int64]  # edges on a straight jaw face
    tips: NDArray[np.int64]  # rounded-tip edges
    pos: NDArray[np.bool_]
    span: _F64
    strip_lo: _F64
    strip_hi: _F64


def _layout(segments: SegmentedFanSource, devices: tuple[int, ...], n_table: int) -> _Layout:
    """Lay the segments' slot tables and the template's flattened stack end to end.

    One slot table per distinct (fan, frame) pair, so a shared fan costs one; the
    field-dependent columns follow each segment's own. The stack follows
    ``_concatenate_stacks`` over each segment's ``flatten()``: the template's
    per-device rows tiled, the leaf-array offsets shifted by one segment's
    leaf-array lengths, each segment's frame from its own; the settings are filled
    in per edge set.
    """
    template = segments.template.flatten()
    fans, frames = segments.fans, segments.frames
    distinct: dict[tuple[int, int], int] = {}
    which = [
        distinct.setdefault((id(f), id(b)), len(distinct))
        for f, b in zip(fans, frames, strict=True)
    ]
    first = {w: s for s, w in reversed(list(enumerate(which)))}
    tables = [
        _slot_table(
            segments, devices, n_table, first[w], template,
            f"segment {first[w]}, " if len(distinct) > 1 else "",
        )
        for w in range(len(distinct))
    ]  # fmt: skip
    slots = tables[0]  # what is not field-dependent: the same in every table
    n_segments = segments.n_segments
    n_slots = int(slots.kind.shape[0])
    n_devices = len(segments.template.devices)

    def tile(values: NDArray[Any]) -> NDArray[Any]:
        return np.tile(np.asarray(values), n_segments)

    def per_segment(name: str) -> NDArray[Any]:
        return np.concatenate([getattr(tables[w], name) for w in which])

    kind = per_segment("kind")
    segment = np.repeat(np.arange(n_segments, dtype=np.int32), n_slots)
    smooth = kind != 2
    table = np.full(kind.shape[0], -1, dtype=np.int32)
    table[smooth] = np.arange(int(smooth.sum()), dtype=np.int32)
    # One s grid per smooth edge slot, shared by that slot's edges in every
    # segment (the grid depends on the slot's span only).
    slot_smooth = slots.kind != 2
    slot_row = np.where(slot_smooth, np.cumsum(slot_smooth) - 1, -1).astype(np.int32)
    if np.any(slot_smooth):
        n_table_set = n_table
        s = slots.s_rows[slot_smooth].reshape(-1)
    else:
        n_table_set = 2
        s = np.zeros(1)
    seg_lo = (np.arange(n_segments, dtype=np.int32) * np.int32(n_devices)).astype(np.int32)
    spectrum = segments.spectrum
    # Each segment's frame as its stack's ``flatten()`` gives it.
    origins = [np.asarray(f.origin, dtype=np.float64) for f in frames]
    rotations = [
        np.array([f.u_axis, f.v_axis, f.w_axis], dtype=np.float64).reshape(-1) for f in frames
    ]
    static: dict[str, Any] = dict(
        kind=kind,
        segment=segment,
        device=(segment * np.int32(n_devices) + tile(slots.device)).astype(np.int32),
        q_axis=per_segment("q_axis"),
        table=table,
        s_row=tile(slot_row),
        sign=per_segment("sign"),
        z_mid=per_segment("z_mid"),
        r_lo=per_segment("r_lo"),
        r_hi=per_segment("r_hi"),
        u_lo=per_segment("u_lo"),
        u_hi=per_segment("u_hi"),
        v_lo=per_segment("v_lo"),
        v_hi=per_segment("v_hi"),
        scale=per_segment("scale"),
        r_c=per_segment("r_c"),
        e_ref=np.full(kind.shape[0], spectrum.mean_energy, dtype=np.float64),
        b_z_top=per_segment("z_top"),
        b_z_bottom=per_segment("z_bottom"),
        b_radius=per_segment("radius"),
        b_focused=per_segment("focused"),
        n_table=n_table_set,
        s=np.asarray(s, dtype=np.float64),
        seg_origin=np.concatenate(origins),
        seg_rot=np.concatenate(rotations),
        seg_lo=seg_lo,
        seg_hi=(seg_lo + np.int32(n_devices)).astype(np.int32),
        spectrum=spectrum,
        e_max=segments.max_energy,
    )
    step = np.arange(n_segments, dtype=np.int32)[:, None]

    def shifted(offsets: _I32, length: int) -> _I32:
        return np.asarray((offsets[None, :] + step * np.int32(length)).reshape(-1))

    stack: dict[str, Any] = dict(
        n_devices=n_segments * template.n_devices,
        origin=origins[0],
        rotation=rotations[0],
        kind=tile(template.kind),
        z_top=tile(template.z_top),
        z_bottom=tile(template.z_bottom),
        material=tile(template.material),
        density=tile(template.density),
        jaw_axis=tile(template.jaw_axis),
        jaw_focused=tile(template.jaw_focused),
        mlc_tip_radius=tile(template.mlc_tip_radius),
        mlc_focused_sides=tile(template.mlc_focused_sides),
        mlc_n_pairs=tile(template.mlc_n_pairs),
        mlc_edges_off=shifted(template.mlc_edges_off, len(template.leaf_edges_v)),
        mlc_tips_off=shifted(template.mlc_tips_off, len(template.tips_pos)),
        leaf_edges_v=tile(template.leaf_edges_v),
    )
    if segments.tips_neg.shape[1] == 0:  # no MLC: flatten() exposes one unread 0.0
        stack["tips_neg"] = tile(template.tips_neg)
        stack["tips_pos"] = tile(template.tips_pos)
    for value in (*static.values(), *stack.values()):
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return _Layout(
        slots=slots,
        static=static,
        stack=stack,
        straight=np.flatnonzero(kind == 1),
        tips=np.flatnonzero(kind == 0),
        pos=static["sign"] > 0.0,
        span=per_segment("span"),
        strip_lo=per_segment("strip_lo"),
        strip_hi=per_segment("strip_hi"),
    )


# The layout depends only on the fans, the frames, the template, the selection and
# the segment count, which ``with_settings`` keeps between an optimizer's iterates:
# memoised by identity, holding the objects so a recycled id cannot alias a dead one.
_LAYOUT_CACHE: dict[tuple[Any, ...], tuple[tuple[Any, ...], _Layout]] = {}
_LAYOUT_CACHE_SIZE = 16
_layout_lock = threading.Lock()


def _layout_for(segments: SegmentedFanSource, devices: tuple[int, ...], n_table: int) -> _Layout:
    held = (segments.template, *segments.fans, *segments.frames)
    key = (tuple(id(obj) for obj in held), devices, n_table)
    with _layout_lock:
        entry = _LAYOUT_CACHE.get(key)
    if entry is not None and all(a is b for a, b in zip(entry[0], held, strict=True)):
        return entry[1]
    layout = _layout(segments, devices, n_table)
    with _layout_lock:
        if len(_LAYOUT_CACHE) >= _LAYOUT_CACHE_SIZE:
            del _LAYOUT_CACHE[next(iter(_LAYOUT_CACHE))]
        _LAYOUT_CACHE[key] = (held, layout)
    return layout


class SegmentedEdgeSource(BeamletSource):
    """The leaf-tip and jaw edges of a segmented fan as beamlets, one per edge.

    Beamlet ``k`` is the edge-derivative estimator of edge ``k``
    (:class:`~pyradmc.geometry.aperture_gradient.FanEdgeDerivativeSource`; order and
    construction: the module docstring), so ``run_beamlet_contraction`` with
    ``dL/dD`` returns ``dL/dx`` for every edge per fan history through its segment
    (scale by the segment's weight for a plan). Weights are signed: prefer the
    contraction to ``run_dij``.

    Parameters
    ----------
    segments
        The segmented fan; every segment's settings are read from it, so build the
        next iterate's edges from ``segments.with_settings(...)``.
    devices
        Indices of the template devices whose edges are listed, in beamlet order
        (default: every device, in stack order).
    n_table, defensive
        The per-edge proposal's table size and defensive uniform share, as in
        :class:`FanEdgeDerivativeSource` (whose defaults these are).

    Refused, as the per-edge source refuses them (naming the segment on an arc): a
    beam frame that does not sit on its fan's focal point with that fan's aperture
    axes, parallel-sided leaves, a leaf pair with no extent in a segment's field, a
    defensive share outside ``[0, 1)``. The
    Warp engine generates this exact type (over an exact ``SegmentedFanSource``)
    in-kernel from :meth:`device_columns`; a subclass keeps the host route.
    """

    def __init__(
        self,
        segments: SegmentedFanSource,
        *,
        devices: Sequence[int] | None = None,
        n_table: int = _DEFAULT_N_TABLE,
        defensive: float = _DEFAULT_DEFENSIVE,
    ) -> None:
        """List the edges and evaluate their position-free geometry."""
        if not isinstance(segments, SegmentedFanSource):
            raise ValueError("a segmented edge source needs a SegmentedFanSource")
        template = segments.template
        fans, frames = segments.fans, segments.frames
        # Each distinct (fan, frame) pair once, at its first segment.
        pairs = {
            (id(f), id(b)): s
            for s, (f, b) in reversed(list(enumerate(zip(fans, frames, strict=True))))
        }
        for s in sorted(pairs.values()):
            try:
                _require_fan_frame(fans[s], frames[s], defensive)
            except ValueError as error:
                if len(pairs) == 1:
                    raise
                raise ValueError(f"segment {s}: {error}") from None
        n_devices = len(template.devices)
        selected = tuple(range(n_devices)) if devices is None else tuple(int(j) for j in devices)
        if not selected:
            raise ValueError("an edge set needs at least one device")
        for j in selected:
            if not 0 <= j < n_devices:
                raise ValueError(f"device {j} not in a {n_devices}-device stack")
        if len(set(selected)) != len(selected):
            raise ValueError(f"device listed twice: {selected}")
        self._segments = segments
        self._devices = selected
        self._n_table = int(n_table)
        self._defensive = float(defensive)
        self._layout = _layout_for(segments, selected, self._n_table)
        self._slots = self._layout.slots
        settings = np.concatenate(
            [segments.tips_neg, segments.tips_pos, segments.jaw_neg, segments.jaw_pos], axis=1
        )
        values = np.asarray(settings[:, self._slots.column].reshape(-1), dtype=np.float64)
        values.setflags(write=False)
        self._values = values
        self._edges: dict[int, FanEdgeDerivativeSource] = {}
        self._edges_lock = threading.Lock()

    # --- the BeamletSource contract ----------------------------------------------------

    @property
    def max_energy(self) -> float:
        """The fan spectrum's top edge."""
        return self._segments.max_energy

    @property
    def n_beamlets(self) -> int:
        """One beamlet per edge."""
        return int(self._values.shape[0])

    def emit(self, beamlet: int, rng_state: RNGState) -> Primary:
        """Emit one ray of edge ``beamlet`` (the per-edge source's four uniforms)."""
        return self.edge(beamlet).emit(rng_state)

    def sample_beamlet_batch(
        self, seed: int, history_offset: int, n: int, beamlet: int
    ) -> dict[str, NDArray[Any]]:
        """Return edge ``beamlet``'s vectorized batch (the per-edge source's)."""
        return self.edge(beamlet).sample_batch(seed, history_offset, n)

    # --- what the edges are ------------------------------------------------------------

    @property
    def segments(self) -> SegmentedFanSource:
        """The segmented fan whose edges these are."""
        return self._segments

    @property
    def devices(self) -> tuple[int, ...]:
        """The template devices whose edges are listed, in beamlet order."""
        return self._devices

    @property
    def values(self) -> _F64:
        """The edge coordinates (mid-plane cm), one per beamlet, read-only."""
        return self._values

    @cached_property
    def parameters(self) -> tuple[tuple[int, EdgeParameter], ...]:
        """``(segment, EdgeParameter)`` of every beamlet, in beamlet order."""
        slots = self._slots
        per_segment = [
            EdgeParameter(
                device=int(j),
                side="pos" if sign > 0.0 else "neg",
                pair=None if pair < 0 else int(pair),
            )
            for j, pair, sign in zip(slots.device, slots.pair, slots.sign, strict=True)
        ]
        return tuple(
            (s, parameter) for s in range(self._segments.n_segments) for parameter in per_segment
        )

    def edge(self, beamlet: int) -> FanEdgeDerivativeSource:
        """Return the per-edge source of ``beamlet`` (built on first use, then kept)."""
        if not 0 <= beamlet < self.n_beamlets:
            raise IndexError(f"beamlet {beamlet} outside an edge set of {self.n_beamlets}")
        with self._edges_lock:
            edge = self._edges.get(beamlet)
        if edge is None:
            segment, parameter = self.parameters[beamlet]
            segments = self._segments
            edge = FanEdgeDerivativeSource(
                segments.fans[segment],
                segments.segment_stack(segment),
                segments.cross_sections,
                parameter,
                n_table=self._n_table,
                defensive=self._defensive,
            )
            with self._edges_lock:
                edge = self._edges.setdefault(beamlet, edge)
        return edge

    # --- what a device backend uploads -------------------------------------------------

    def device_columns(self) -> FanEdgeColumns:
        """Build the edge set's host columns, vectorized (module docstring).

        Equal, column for column, to
        :func:`~pyradmc.geometry.aperture_gradient.fan_edge_columns` over the
        per-edge records of the same edges. Columns the settings do not touch are
        shared, read-only, between the edge sets of one layout.
        """
        layout = self._layout
        static = layout.static
        segments = self._segments
        x = np.array(self._values, dtype=np.float64)
        sign = static["sign"]
        z_mid = static["z_mid"]
        z_top = static["b_z_top"]
        z_bottom = static["b_z_bottom"]
        radius = static["b_radius"]

        # Support starts: a focused jaw is crossed end-on at x; a straight face from
        # its far corner; a rounded tip from the tangent ray, else by bisection.
        q0 = x.copy()
        tangent_q = None
        k = layout.straight
        if k.size:
            w_far = np.where(sign[k] * x[k] >= 0.0, z_bottom[k], z_top[k])
            q0[k] = x[k] * z_mid[k] / w_far
        k = layout.tips
        if k.size:
            start, closed = tip_support_starts(
                x[k], sign[k], z_mid[k], radius[k], z_top[k], z_bottom[k]
            )
            q0[k] = start
            tangent_q = start.copy()
            b = k[~closed]
            if b.size:
                q0[b] = tip_support_bisection(
                    x[b], sign[b], layout.span[b], static["r_c"][b], z_top[b], z_bottom[b],
                    radius[b], layout.strip_lo[b], layout.strip_hi[b],
                )  # fmt: skip

        # The tangent ray of every rounded tip (q0 itself unless the start was bisected).
        q_t = q0.copy()
        if tangent_q is not None:
            q_t[layout.tips] = tangent_q

        # The isolated block: a tip's strip; a jaw's moving edge and the retracted one.
        tip = static["kind"] == 0
        b_lo = np.where(tip, layout.strip_lo, np.where(layout.pos, -_RETRACT_CM, x))
        b_hi = np.where(tip, layout.strip_hi, np.where(layout.pos, x, _RETRACT_CM))

        stack = dict(layout.stack)
        stack["jaw_edge_neg"] = np.asarray(segments.jaw_neg.reshape(-1), dtype=np.float64)
        stack["jaw_edge_pos"] = np.asarray(segments.jaw_pos.reshape(-1), dtype=np.float64)
        if "tips_neg" not in stack:
            stack["tips_neg"] = np.asarray(segments.tips_neg.reshape(-1), dtype=np.float64)
            stack["tips_pos"] = np.asarray(segments.tips_pos.reshape(-1), dtype=np.float64)
        return FanEdgeColumns(
            **static,
            x=x,
            q0=q0,
            defensive=np.full(x.shape[0], self._defensive, dtype=np.float64),
            b_lo=np.asarray(b_lo, dtype=np.float64),
            b_hi=np.asarray(b_hi, dtype=np.float64),
            q_t=q_t,
            t_off=sign * (q0 - q_t),
            cross_sections=segments.cross_sections,
            stack=CompiledStack(**stack),
        )
