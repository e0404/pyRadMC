"""Device layout and in-kernel generation for the segmented fan source.

The Warp half of :class:`~pyradmc.geometry.segments.SegmentedFanSource`.

One uploaded :class:`~pyradmc.backends.warp.presolve.StackArrays` carries the shared
structure of every segment (device kinds, slabs, materials, leaf strips, focusing)
and a :class:`SegmentArrays` carries the per-segment columns — jaw edges indexed
``segment * n_devices + device``, leaf tips ``segment * n_tips + tip``, the beam
frame (origin at ``segment * 3``, row-major rotation at ``segment * 9``) and the fan
(``segment * FAN_STRIDE``: focal point, field centre, ``u``/``v`` aperture axes,
widths) — so segment ``s``'s chord through device ``j`` is the device pre-solve's own
``jaw_path_length``/``mlc_path_length`` twin evaluated at that segment's settings, in
that segment's frame, on a ray of that segment's fan. A single-fan source uploads its
one fan and frame once per segment: one layout and one kernel serve one fan and an
arc of them alike. (The stack's own ``origin``/``rotation`` are uploaded but not read
here.) The attenuation table is the pre-solve's
:class:`~pyradmc.backends.warp.presolve.MuArrays` (uploaded from the source's own
host table), so the in-kernel weight is the device pre-solve's ``mode="attenuation"``
weight without an air column, segment by segment.

The generators draw their primaries through the shared single-source samplers of
:mod:`pyradmc.geometry.fan` (``fan_primary``, ``select_by_cdf``), the same functions
the spectral generators in :mod:`~pyradmc.backends.warp.kernels` call, and this
module compiles with the same options as that one (``fast_math``, no adjoints), so a
segmented history's energy and direction are bit-identical to the spectral fan's for
the same ``(seed, key)`` on one device (test-pinned). The transmission weight is
pinned against the host at float32 precision.

This module is Warp kernel code: exempt from ``mypy --strict`` (AGENTS.md 5) and
importable only when the ``warp`` optional dependency is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from pyradmc.backends.warp.kernels import Queue, _escape
from pyradmc.backends.warp.physics import warp_physics
from pyradmc.backends.warp.presolve import (
    MuArrays,
    StackArrays,
    _mu_lookup,
    _upload_mu,
    upload_stack,
)
from pyradmc.rng.warp_shim import WarpRNGState, init_slot, uniform
from pyradmc.transport.particles import PHOTON

__all__ = [
    "FAN_STRIDE",
    "SegmentArrays",
    "SegmentedFanUpload",
    "generate_segmented_fan",
    "generate_segmented_plan",
    "segment_transmission",
    "segment_transmission_kernel",
    "upload_segmented_fan",
]

wp.set_module_options({"enable_backward": False, "fast_math": True})

_p = warp_physics()
jaw_path_length = _p.jaw_path_length
mlc_path_length = _p.mlc_path_length
fan_primary = _p.fan_primary
select_by_cdf = _p.select_by_cdf

# Floats per segment in SegmentArrays.fan: focal point (3), field centre (3),
# u axis (3), v axis (3), width_u, width_v — fan_primary's geometry arguments.
FAN_STRIDE = 14


@wp.struct
class SegmentArrays:
    """Per-segment settings, frames and fans over one shared stack layout.

    ``jaw_neg``/``jaw_pos`` hold ``n_segments * n_devices`` edges (entries of non-jaw
    devices unused); ``tips_neg``/``tips_pos`` hold ``n_segments * n_tips`` leaf tips
    in the flattened-stack order, so segment ``s`` reads the MLC at device ``j`` from
    ``s * n_tips + mlc_tips_off[j]``. ``origin`` (3 per segment) and ``rotation`` (9
    per segment, the row-major ``(u, v, w)`` basis of
    :class:`~pyradmc.geometry.collimation.CompiledStack`) are segment ``s``'s beam
    frame; ``fan`` holds :data:`FAN_STRIDE` floats per segment, the geometry
    arguments of ``fan_primary`` for segment ``s``'s fan.
    """

    n_segments: int
    n_devices: int
    n_tips: int
    jaw_neg: wp.array(dtype=float)
    jaw_pos: wp.array(dtype=float)
    tips_neg: wp.array(dtype=float)
    tips_pos: wp.array(dtype=float)
    origin: wp.array(dtype=float)
    rotation: wp.array(dtype=float)
    fan: wp.array(dtype=float)


@dataclass
class SegmentedFanUpload:
    """Everything the segmented-fan generator reads, resident on one device."""

    stack: StackArrays
    segments: SegmentArrays
    mu: MuArrays
    edges: wp.array
    cdf: wp.array


def upload_segmented_fan(source, device: str) -> SegmentedFanUpload:
    """Upload a :class:`~pyradmc.geometry.segments.SegmentedFanSource` to ``device``.

    The stack layout comes from the template (``flatten``), the settings from the
    source's per-segment arrays, each segment's frame and fan from its
    :attr:`~pyradmc.geometry.segments.SegmentedFanSource.frames` and
    :attr:`~pyradmc.geometry.segments.SegmentedFanSource.fans` (the frame in
    ``flatten``'s layout), the attenuation table from the source's own host table —
    one definition of each, shared with the host route.
    """

    def farr(values) -> wp.array:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        if flat.size == 0:  # a backend may reject an empty buffer; never read
            flat = np.zeros(1, dtype=np.float32)
        return wp.array(flat, dtype=float, device=device)

    segments = SegmentArrays()
    segments.n_segments = int(source.n_segments)
    segments.n_devices = int(source.jaw_neg.shape[1])
    segments.n_tips = int(source.tips_neg.shape[1])
    segments.jaw_neg = farr(source.jaw_neg)
    segments.jaw_pos = farr(source.jaw_pos)
    segments.tips_neg = farr(source.tips_neg)
    segments.tips_pos = farr(source.tips_pos)
    frames = source.frames
    segments.origin = farr(np.array([frame.origin for frame in frames], dtype=np.float64))
    segments.rotation = farr(
        np.array([(frame.u_axis, frame.v_axis, frame.w_axis) for frame in frames], np.float64)
    )
    segments.fan = farr(
        np.array(
            [
                (*f.focal_point, *f.center, *f.u_axis, *f.v_axis, f.width_u, f.width_v)
                for f in source.fans
            ],
            dtype=np.float64,
        )
    )
    spectrum = source.spectrum
    return SegmentedFanUpload(
        stack=upload_stack(source.template.flatten(), device),
        segments=segments,
        mu=_upload_mu(source.attenuation_tables, device),
        edges=wp.array(spectrum.edges.astype(np.float32), dtype=float, device=device),
        cdf=wp.array(spectrum.cdf.astype(np.float32), dtype=float, device=device),
    )


@wp.func
def _segment_path_length(
    stack: StackArrays,
    segs: SegmentArrays,
    seg: int,
    j: int,
    p_u: float,
    p_v: float,
    p_w: float,
    d_u: float,
    d_v: float,
    d_w: float,
) -> float:
    """Chord of one local-frame ray through device ``j`` at segment ``seg``'s settings.

    The per-segment twin of the pre-solve's ``device_path_length``: the structure
    from ``stack``, the jaw edges and the leaf-tip offset from ``segs``.
    """
    if stack.kind[j] == 0:
        e_neg = segs.jaw_neg[seg * segs.n_devices + j]
        e_pos = segs.jaw_pos[seg * segs.n_devices + j]
        q = p_u
        d_q = d_u
        if stack.jaw_axis[j] != 0:
            q = p_v
            d_q = d_v
        return jaw_path_length(
            q,
            d_q,
            p_w,
            d_w,
            stack.z_top[j],
            stack.z_bottom[j],
            e_neg,
            e_pos,
            stack.jaw_focused[j],
            0,
        )
    return mlc_path_length(
        p_u,
        p_v,
        p_w,
        d_u,
        d_v,
        d_w,
        stack.z_top[j],
        stack.z_bottom[j],
        stack.mlc_tip_radius[j],
        stack.mlc_n_pairs[j],
        stack.mlc_edges_off[j],
        seg * segs.n_tips + stack.mlc_tips_off[j],
        stack.leaf_edges_v,
        segs.tips_neg,
        segs.tips_pos,
        stack.mlc_focused_sides[j],
        0,
    )


@wp.func
def segment_transmission(
    stack: StackArrays,
    segs: SegmentArrays,
    mu: MuArrays,
    seg: int,
    energy: float,
    x: float,
    y: float,
    z: float,
    ux: float,
    uy: float,
    uz: float,
) -> float:
    """``exp(-sum_j mu_j(E) L_{seg,j})`` for one engine-frame ray (full line).

    The device pre-solve's attenuation-mode arithmetic (frame map, per-device
    chord, ``_mu_lookup``), in segment ``seg``'s frame with the chords at its
    settings.
    """
    o = seg * 3
    k = seg * 9
    rx = x - segs.origin[o + 0]
    ry = y - segs.origin[o + 1]
    rz = z - segs.origin[o + 2]
    r0 = segs.rotation[k + 0]
    r1 = segs.rotation[k + 1]
    r2 = segs.rotation[k + 2]
    r3 = segs.rotation[k + 3]
    r4 = segs.rotation[k + 4]
    r5 = segs.rotation[k + 5]
    r6 = segs.rotation[k + 6]
    r7 = segs.rotation[k + 7]
    r8 = segs.rotation[k + 8]
    p_u = r0 * rx + r1 * ry + r2 * rz
    p_v = r3 * rx + r4 * ry + r5 * rz
    p_w = r6 * rx + r7 * ry + r8 * rz
    d_u = r0 * ux + r1 * uy + r2 * uz
    d_v = r3 * ux + r4 * uy + r5 * uz
    d_w = r6 * ux + r7 * uy + r8 * uz
    tau = float(0.0)  # noqa: UP018 - Warp needs float() to declare a loop-mutated var
    for j in range(stack.n_devices):
        thickness = _segment_path_length(stack, segs, seg, j, p_u, p_v, p_w, d_u, d_v, d_w)
        mu_j = _mu_lookup(mu.log_total, j, mu.log_e_min, mu.inv_dlog, mu.n_points, energy)
        tau += mu_j * thickness
    return wp.exp(-tau)


@wp.kernel
def segment_transmission_kernel(
    stack: StackArrays,
    segs: SegmentArrays,
    mu: MuArrays,
    energy: wp.array(dtype=float),
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
    z: wp.array(dtype=float),
    ux: wp.array(dtype=float),
    uy: wp.array(dtype=float),
    uz: wp.array(dtype=float),
    out: wp.array2d(dtype=float),
) -> None:
    """Probe: ``out[ray, seg]`` = the transmission the generator weights by."""
    tid, seg = wp.tid()
    out[tid, seg] = segment_transmission(
        stack, segs, mu, seg, energy[tid], x[tid], y[tid], z[tid], ux[tid], uy[tid], uz[tid]
    )


@wp.func
def _queue_segment_primary(
    tid: int,
    state: WarpRNGState,
    seg: int,
    tag: int,
    edges: wp.array(dtype=float),
    cdf: wp.array(dtype=float),
    n_bins: int,
    stack: StackArrays,
    segs: SegmentArrays,
    mu: MuArrays,
    slots: wp.array(dtype=wp.uint32),
    emitted: wp.array(dtype=wp.int64),
    q: Queue,
):
    """Sample segment ``seg``'s fan photon, weight it by its transmission, queue it.

    ``fan_primary`` through the segment's fan (four uniforms, host order), from its
    focal point, at its stack transmission; tagged ``tag``, ``weight * energy``
    booked into ``emitted``.
    """
    b = seg * FAN_STRIDE
    fx = segs.fan[b + 0]
    fy = segs.fan[b + 1]
    fz = segs.fan[b + 2]
    energy, dx, dy, dz = fan_primary(
        state,
        edges,
        cdf,
        n_bins,
        fx,
        fy,
        fz,
        segs.fan[b + 3],
        segs.fan[b + 4],
        segs.fan[b + 5],
        segs.fan[b + 6],
        segs.fan[b + 7],
        segs.fan[b + 8],
        segs.fan[b + 9],
        segs.fan[b + 10],
        segs.fan[b + 11],
        segs.fan[b + 12],
        segs.fan[b + 13],
    )
    weight = segment_transmission(stack, segs, mu, seg, energy, fx, fy, fz, dx, dy, dz)
    q.kind[tid] = PHOTON
    q.beamlet[tid] = tag
    q.primary[tid] = 1  # source photon: the only one that may split
    q.energy[tid] = energy
    q.weight[tid] = weight
    q.x[tid] = fx
    q.y[tid] = fy
    q.z[tid] = fz
    q.ux[tid] = dx
    q.uy[tid] = dy
    q.uz[tid] = dz
    q.rng[tid] = slots[tid]
    _escape(emitted, weight * energy)


@wp.kernel
def generate_segmented_fan(
    seed: int,
    group_start: int,
    n_per: int,
    per_batch: int,
    batch: int,
    t_offset: int,
    correlated: int,
    edges: wp.array(dtype=float),
    cdf: wp.array(dtype=float),
    n_bins: int,
    stack: StackArrays,
    segs: SegmentArrays,
    mu: MuArrays,
    slots: wp.array(dtype=wp.uint32),
    emitted: wp.array(dtype=wp.int64),
    q: Queue,
):
    """Segment primaries for one chunk of one batch of a segment group.

    The Dij block mapping of ``kernels.generate_spectral_beamlets`` — group-local
    segment ``t // per_batch``, within-segment index ``r = batch * per_batch + i``,
    stream key ``r`` (correlated) or ``segment * n_per + r`` — and the fan photon of
    ``fan_primary`` through the segment's own fan. The weight is the segment's stack
    transmission (in its own frame) along the sampled ray at the sampled energy;
    ``weight * energy`` is booked into ``emitted``. The key never depends on the
    chunk, so the stream of a history is chunk-invariant.
    """
    tid = wp.tid()
    t = t_offset + tid
    local = t // per_batch
    i = t - local * per_batch
    r = batch * per_batch + i
    seg = group_start + local
    key = seg * n_per + r
    if correlated != 0:
        key = r
    slots[tid] = init_slot(seed, key)
    state = WarpRNGState()
    state.slots = slots
    state.idx = tid
    _queue_segment_primary(
        tid, state, seg, local, edges, cdf, n_bins, stack, segs, mu, slots, emitted, q
    )


@wp.kernel
def generate_segmented_plan(
    seed: int,
    history_offset: int,
    selection_cdf: wp.array(dtype=float),
    n_segments: int,
    edges: wp.array(dtype=float),
    cdf: wp.array(dtype=float),
    n_bins: int,
    stack: StackArrays,
    segs: SegmentArrays,
    mu: MuArrays,
    slots: wp.array(dtype=wp.uint32),
    emitted: wp.array(dtype=wp.int64),
    q: Queue,
):
    """Plan-dose primaries: a segment drawn by its share, then its weighted fan photon.

    The in-kernel twin of :meth:`~pyradmc.geometry.segments.SegmentedPlanSource.emit`:
    one uniform selects the segment (``select_by_cdf`` over the normalized shares),
    ``fan_primary``'s four uniforms follow in the host order through that segment's
    fan, and the weight is that segment's stack transmission (no ``total_weight``
    factor: the books stay at the scale of one fan history). One open-field history
    per thread at a fixed slot (``beamlet = 0``); ``weight * energy`` is booked into
    ``emitted``.
    """
    tid = wp.tid()
    slots[tid] = init_slot(seed, history_offset + tid)
    state = WarpRNGState()
    state.slots = slots
    state.idx = tid
    seg = select_by_cdf(selection_cdf, n_segments, uniform(state))
    _queue_segment_primary(
        tid, state, seg, 0, edges, cdf, n_bins, stack, segs, mu, slots, emitted, q
    )
