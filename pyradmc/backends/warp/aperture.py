"""In-kernel fan edge-gradient rays: the device route of the fan edge sets.

The host pre-sampling route seeds and drains the transport queues once per beamlet;
for an edge set (one beamlet per leaf edge, hundreds to thousands of them) that
per-edge overhead dominated the gradient's wall clock (~1.4 ms per edge on the
RTX 4070, measured 2026-10-21). This module uploads a whole edge set once — an
``EdgeGradientBeamletSource`` of
:class:`~pyradmc.geometry.aperture_gradient.FanEdgeDerivativeSource` edges, or a
:class:`~pyradmc.geometry.segment_edges.SegmentedEdgeSource`, both reduced to the
same host :class:`~pyradmc.geometry.aperture_gradient.FanEdgeColumns` (pinned equal)
and packed into one float32 and one int32 transfer — and generates their rays
in-kernel. The arithmetic is not here: the generators call the single-source scalar
functions of :mod:`pyradmc.geometry.edge_rays` (and
:func:`~pyradmc.geometry.fan.spectrum_energy`), compiled by the physics loader, which
the host route of :class:`~pyradmc.geometry.aperture_gradient.FanEdgeDerivativeSource`
runs too:

- energy by spectrum CDF inversion (the spectral beamlet route's draws), then the
  position along the edge, then the proposal draw ``q = q0 + sign s^2`` from the
  edge's tabulated CDF (a jump edge sits on ``q = x``): ``edge_ray_point``;
- the focal ray through ``(q, r)`` on the parameter device's mid-plane and its field
  test: ``edge_ray_direction``, ``edge_in_field``;
- the weight ``scale * (-mu T_block dl/dx) / p(q) * T_rest`` from the compiled chord
  twins and their edge derivatives (``edge_block_terms``, with its near-tangent
  closed form), or ``scale * sign (1 - T_full) * T_rest`` on a jump, where
  ``T_rest = exp(-(tau_stack - mu l_block))``: the parameter block's optical depth
  enters the stack's additively, so no retracted stack is needed (``edge_core``,
  ``edge_weight``, ``rest_transmission``).

What stays here is the device side: the uploads, the proposal and generator
launches, the thread-to-history mapping and RNG keying, the loop over a segment's
stack devices (the pre-solve's chord dispatcher) and the queue writes.

Every segment's stack (one per control point, each with its own frame) is
concatenated into one device table, addressed per segment by a device range and
per edge by its segment; the MLC leaf arrays are offset-addressed already, so the
pre-solve's chord dispatcher works on the combined table unchanged. mu(E) rows use
the host transmission tables' log grid (``geomspace(0.010, e_max, 512)``,
clamped log-linear interpolation), so host and device attenuate alike.

This module is Warp kernel plumbing: exempt from ``mypy --strict`` (AGENTS.md 5).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from pyradmc.backends.warp.kernels import PHOTON, Queue, _escape
from pyradmc.backends.warp.physics import warp_physics
from pyradmc.backends.warp.presolve import StackArrays, device_path_length
from pyradmc.geometry.aperture_gradient import (
    EdgeGradientBeamletSource,
    FanEdgeColumns,
    FanEdgeDerivativeSource,
    fan_edge_columns,
)
from pyradmc.geometry.mucache import mu_curve
from pyradmc.geometry.segment_edge_steps import SegmentedEdgeStepSource
from pyradmc.geometry.segment_edges import SegmentedEdgeSource
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.rng.warp_shim import WarpRNGState, init_slot, uniform

__all__ = [
    "EdgeSet",
    "edge_columns",
    "generate_edge_steps",
    "generate_fan_edges",
    "is_device_edge_set",
    "upload_edge_columns",
    "upload_edge_set",
]

wp.set_module_options({"enable_backward": False})

_p = warp_physics()
edge_block_terms = _p.edge_block_terms
edge_core = _p.edge_core
edge_in_field = _p.edge_in_field
edge_node_density = _p.edge_node_density
edge_proposal_tables = _p.edge_proposal_tables
edge_q = _p.edge_q
edge_ray_direction = _p.edge_ray_direction
edge_ray_point = _p.edge_ray_point
edge_weight = _p.edge_weight
focused_slab_chord = _p.focused_slab_chord
frame_direction = _p.frame_direction
lookup_loglinear_2d = _p.lookup_loglinear_2d
proposal_cell_mass = _p.proposal_cell_mass
rest_transmission = _p.rest_transmission
select_by_cdf_at = _p.select_by_cdf_at
spectrum_energy = _p.spectrum_energy
stratified_position = _p.stratified_position

# The host transmission tables' grid (collimation._StackTransmission via the
# aperture-gradient sources): same nodes, same interpolation.
_E_MIN = 0.010
_N_MU = 512
# Below this distance from a rounded tip's tangent ray (s_t^2 = |q - q_t|, cm) the
# chord takes the closed-form near-tangent path (edge_rays.edge_block_terms: the disc
# chord, clipped where a slab face cuts it). Above it the twin answers: just past 2e-3
# in the DAO regime its float32 error is 3-7e-3 relative in the rate and up to
# 2.5e-3 cm in the chord, unbiased (derivative-mass ratio 0.9998, measured
# 2026-10-23). A float32 choice, so it lives with the float32 target.
_NEAR_TANGENT_S2 = wp.constant(2.0e-3)
# The floor of the proposal's 2 s in float32 (edge_rays.edge_proposal_point).
_DENSITY_FLOOR = wp.constant(1.0e-30)


@wp.struct
class EdgeSet:
    """A set of fan edges, their segments' stacks and mu tables, on one device."""

    n_table: int
    kind: wp.array(dtype=wp.int32)
    segment: wp.array(dtype=wp.int32)
    device: wp.array(dtype=wp.int32)
    q_axis: wp.array(dtype=wp.int32)
    table: wp.array(dtype=wp.int32)
    s_row: wp.array(dtype=wp.int32)
    sign: wp.array(dtype=float)
    x: wp.array(dtype=float)
    q0: wp.array(dtype=float)
    z_mid: wp.array(dtype=float)
    r_lo: wp.array(dtype=float)
    r_hi: wp.array(dtype=float)
    u_lo: wp.array(dtype=float)
    u_hi: wp.array(dtype=float)
    v_lo: wp.array(dtype=float)
    v_hi: wp.array(dtype=float)
    scale: wp.array(dtype=float)
    r_c: wp.array(dtype=float)
    e_ref: wp.array(dtype=float)
    defensive: wp.array(dtype=float)
    b_z_top: wp.array(dtype=float)
    b_z_bottom: wp.array(dtype=float)
    b_radius: wp.array(dtype=float)
    b_lo: wp.array(dtype=float)
    b_hi: wp.array(dtype=float)
    b_focused: wp.array(dtype=wp.int32)
    q_t: wp.array(dtype=float)
    t_off: wp.array(dtype=float)
    s: wp.array(dtype=float)
    cdf: wp.array(dtype=float)
    cell: wp.array(dtype=float)
    seg_origin: wp.array(dtype=float)
    seg_rot: wp.array(dtype=float)
    seg_lo: wp.array(dtype=wp.int32)
    seg_hi: wp.array(dtype=wp.int32)
    stack: StackArrays
    log_e_min: float
    inv_dlog: float
    n_mu: int
    log_mu: wp.array2d(dtype=float)
    # An edge step's choice per column (SegmentedEdgeStepSource.selection); unset otherwise.
    step_first: wp.array(dtype=wp.int32)
    step_edge: wp.array(dtype=wp.int32)
    step_cdf: wp.array(dtype=float)
    step_factor: wp.array(dtype=float)


def is_device_edge_set(source) -> bool:
    """Report whether ``source`` is an edge set this module generates in-kernel.

    Exact types only (an ``emit``/``sample_beamlet_batch`` override must not be
    bypassed): an ``EdgeGradientBeamletSource`` of ``FanEdgeDerivativeSource`` edges,
    or a ``SegmentedEdgeSource`` over an exact ``SegmentedFanSource``. The fan's own
    type is not checked on either: an edge source reads only the fan's spectrum and
    geometry, never its ``emit``/``sample_batch``, on the host route as here.
    """
    if type(source) is SegmentedEdgeStepSource:
        return is_device_edge_set(source.edges)
    if type(source) is SegmentedEdgeSource:
        return type(source.segments) is SegmentedFanSource
    return type(source) is EdgeGradientBeamletSource and all(
        type(edge) is FanEdgeDerivativeSource for edge in source.edges
    )


def edge_columns(source) -> FanEdgeColumns:
    """Build the host columns of an in-kernel edge set (:func:`is_device_edge_set` sources).

    A segmented edge source builds them vectorized; a list of per-edge sources goes
    through their records. The two are pinned column-equal for the same edges.
    """
    if type(source) is SegmentedEdgeStepSource:
        return source.edges.device_columns()
    if type(source) is SegmentedEdgeSource:
        return source.device_columns()
    return fan_edge_columns([edge.device_record() for edge in source.edges])


def upload_edge_set(source, device: str) -> tuple[EdgeSet, tuple]:
    """Upload a fan edge set; return it and the shared spectrum's ``(edges, cdf)``."""
    uploaded = upload_edge_columns(edge_columns(source), device)
    if type(source) is SegmentedEdgeStepSource:
        es, selection = uploaded[0], source.selection
        es.step_first = wp.array(selection.first, dtype=wp.int32, device=device)
        es.step_edge = wp.array(selection.edge, dtype=wp.int32, device=device)
        es.step_cdf = wp.array(selection.cdf, dtype=float, device=device)
        es.step_factor = wp.array(selection.factor, dtype=float, device=device)
    return uploaded


def _log_mu_rows(columns: FanEdgeColumns) -> np.ndarray:
    """``log(mu/rho)`` on the host transmission grid, one row per combined device.

    Through the per-source curve memo: element for element the scalar
    ``mu_over_rho_total(float(e), material)`` queries on ``geomspace(_E_MIN, e_max,
    _N_MU)`` that this upload used to repeat on every call.
    """
    rows = {
        int(material): np.log(
            mu_curve(columns.cross_sections, int(material), _E_MIN, columns.e_max, _N_MU, None)[1]
        )
        for material in np.unique(columns.stack.material)
    }
    return np.array([rows[int(material)] for material in columns.stack.material])


_EDGE_INTS = (
    "kind", "segment", "device", "q_axis", "table", "s_row", "b_focused", "seg_lo", "seg_hi",
)  # fmt: skip
_EDGE_FLOATS = (
    "sign", "x", "q0", "z_mid", "r_lo", "r_hi", "u_lo", "u_hi", "v_lo", "v_hi", "scale", "r_c",
    "e_ref", "defensive", "b_z_top", "b_z_bottom", "b_radius", "b_lo", "b_hi", "q_t", "t_off", "s",
    "seg_origin", "seg_rot",
)  # fmt: skip
_STACK_INTS = (
    "kind", "material", "jaw_axis", "jaw_focused", "mlc_focused_sides", "mlc_n_pairs",
    "mlc_edges_off", "mlc_tips_off",
)  # fmt: skip
_STACK_FLOATS = (
    "origin", "rotation", "z_top", "z_bottom", "density", "jaw_edge_neg", "jaw_edge_pos",
    "mlc_tip_radius", "leaf_edges_v", "tips_neg", "tips_pos",
)  # fmt: skip


def _packed(parts: list[np.ndarray], dtype, wp_dtype, device: str) -> list[wp.array]:
    """Upload ``parts`` in one transfer; return one device view per part.

    Each part is cast exactly as a separate ``wp.array(np.asarray(part, dtype))``
    would cast it; the views keep the shared buffer alive (Warp's slice back-ref).
    One copy instead of one per column: a small upload costs ~50 us on the RTX
    4070 laptop, a view ~7 us.
    """
    flat = [np.asarray(part, dtype=dtype).reshape(-1) for part in parts]
    if any(f.size == 0 for f in flat):
        raise ValueError("an edge-set column is empty")
    buffer = wp.array(np.concatenate(flat), dtype=wp_dtype, device=device)
    views = []
    start = 0
    for f in flat:
        views.append(buffer[start : start + f.size])
        start += f.size
    return views


def upload_edge_columns(columns: FanEdgeColumns, device: str) -> tuple[EdgeSet, tuple]:
    """Upload host edge columns to ``device`` and tabulate the proposals there.

    Every float column goes up as float32, every index as int32 (the stack's as
    :func:`~pyradmc.backends.warp.presolve.upload_stack` casts them), packed into one
    buffer per type; the cumulative proposal on each smooth edge's ``s`` grid is
    built by ``build_proposals``.
    """
    smooth = np.flatnonzero(columns.table >= 0).astype(np.int32)
    n_table = columns.n_table
    stack = columns.stack
    log_mu = _log_mu_rows(columns)

    ints = _packed(
        [getattr(columns, name) for name in _EDGE_INTS]
        + [getattr(stack, name) for name in _STACK_INTS]
        + ([smooth] if len(smooth) else []),
        np.int32,
        wp.int32,
        device,
    )
    floats = _packed(
        [getattr(columns, name) for name in _EDGE_FLOATS]
        + [getattr(stack, name) for name in _STACK_FLOATS]
        + [log_mu, columns.spectrum.edges, columns.spectrum.cdf],
        np.float32,
        float,
        device,
    )

    es = EdgeSet()
    es.n_table = int(n_table)
    for name, view in zip(_EDGE_INTS, ints, strict=False):
        setattr(es, name, view)
    for name, view in zip(_EDGE_FLOATS, floats, strict=False):
        setattr(es, name, view)
    es.stack = StackArrays()
    es.stack.n_devices = stack.n_devices
    for name, view in zip(_STACK_INTS, ints[len(_EDGE_INTS) :], strict=False):
        setattr(es.stack, name, view)
    for name, view in zip(_STACK_FLOATS, floats[len(_EDGE_FLOATS) :], strict=False):
        setattr(es.stack, name, view)
    cells = max(1, len(smooth) * (n_table - 1))
    tables = wp.zeros(2 * cells, dtype=float, device=device)
    es.cdf = tables[:cells]
    es.cell = tables[cells:]
    es.log_e_min = float(math.log(_E_MIN))
    es.inv_dlog = float((_N_MU - 1) / (math.log(columns.e_max) - math.log(_E_MIN)))
    es.n_mu = _N_MU
    tail = len(_EDGE_FLOATS) + len(_STACK_FLOATS)
    es.log_mu = floats[tail].reshape(log_mu.shape)
    if len(smooth):
        _tabulate_proposals(es, ints[-1], len(smooth), n_table, device)
    return es, (floats[tail + 1], floats[tail + 2])


# Below this many smooth edges per SM the one-thread-per-edge proposal kernel leaves
# the GPU mostly idle and the node densities go first, one thread per node. Measured
# on the RTX 4070 laptop (46 SMs), interleaved, same edge set: 400 edges 1.53 ->
# 0.69 ms, 4 000 edges 1.23 -> 0.83 ms, 8 000 even, 15 000 edges 2.29 -> 2.95 ms
# (the split then only adds the density array's traffic). Bit-identical either way.
_SPLIT_EDGES_PER_SM = 128


def _tabulate_proposals(es, smooth: wp.array, n_smooth: int, n_table: int, device: str) -> None:
    """Launch the proposal tabulation, node densities split out when that is faster."""
    d = wp.get_device(device)
    if d.is_cuda and n_smooth < _SPLIT_EDGES_PER_SM * int(getattr(d, "sm_count", 0) or 0):
        densities = wp.empty(n_smooth * n_table, dtype=float, device=device)
        wp.launch(
            proposal_densities, dim=(n_table, n_smooth), inputs=[es, smooth, densities],
            device=device,
        )  # fmt: skip
        wp.launch(build_proposals, dim=n_smooth, inputs=[es, smooth, densities, 1], device=device)
    else:
        unused = wp.empty(1, dtype=float, device=device)
        wp.launch(build_proposals, dim=n_smooth, inputs=[es, smooth, unused, 0], device=device)


@wp.func
def _mu(es: EdgeSet, row: int, energy: float) -> float:
    """Macroscopic mu (1/cm) of global device ``row``: the host np.interp in log E.

    ``log(mu/rho)`` interpolated linearly in log E on the row (the data layer's
    clamped lookup), then times the device's density.
    """
    log_mu = lookup_loglinear_2d(es.log_mu, row, es.log_e_min, es.inv_dlog, es.n_mu, energy)
    return wp.exp(log_mu) * es.stack.density[row]


@wp.func
def _block_terms(es: EdgeSet, j: int, qq: float, s: float, du: float, dv: float, dw: float):
    """Return ``(l_block, rate, l_rest)`` of edge ``j``'s block along one focal ray.

    :func:`~pyradmc.geometry.edge_rays.edge_block_terms` on edge ``j``'s columns, with
    the float32 near-tangent threshold ``_NEAR_TANGENT_S2``: the twins' chord and
    rate, or within ``s^2 + t_off < 2e-3`` of a rounded tip's tangent ray the
    cancellation-free closed form (whole, or clipped where a slab face cuts it).
    ``t_off`` and ``q_t`` come from the host in float64.
    """
    l_block, rate, l_rest = edge_block_terms(
        es.kind[j], es.sign[j], es.x[j], es.z_mid[j], es.b_z_top[j], es.b_z_bottom[j],
        es.b_radius[j], es.b_lo[j], es.b_hi[j], es.b_focused[j], es.q_axis[j], es.q_t[j],
        es.t_off[j], _NEAR_TANGENT_S2, qq, s, du, dv, dw,
    )  # fmt: skip
    return l_block, rate, l_rest


@wp.func
def _node_density(es: EdgeSet, j: int, sk: float, mu_block: float) -> float:
    """``f = 2 s |dT/dx|`` of edge ``j`` at proposal node ``s = sk``.

    Along the edge's central ray ``r_c`` at the reference energy (``mu_block``), at
    ``q = q0 + sign s^2``: :func:`~pyradmc.geometry.edge_rays.edge_node_density`.
    """
    qq = edge_q(es.q0[j], es.sign[j], sk)
    _lu, _lv, du, dv, dw = edge_ray_direction(qq, es.r_c[j], es.q_axis[j], es.z_mid[j])
    l_block, rate, _l_rest = _block_terms(es, j, qq, sk, du, dv, dw)
    return edge_node_density(sk, mu_block, wp.exp(-mu_block * l_block), rate)


@wp.kernel
def proposal_densities(es: EdgeSet, smooth: wp.array(dtype=wp.int32), f: wp.array(dtype=float)):
    """Evaluate the proposal's node densities, one thread per (node, smooth edge).

    ``f[k * n_smooth + t] = _node_density`` of smooth edge ``t`` at node ``k``:
    node-major, so these writes and :func:`build_proposals`' per-edge reads are
    both coalesced across a warp.
    """
    k, t = wp.tid()
    j = smooth[t]
    mu_block = _mu(es, es.device[j], es.e_ref[j])
    f[k * smooth.shape[0] + t] = _node_density(es, j, es.s[es.s_row[j] * es.n_table + k], mu_block)


@wp.kernel
def build_proposals(
    es: EdgeSet,
    smooth: wp.array(dtype=wp.int32),
    f: wp.array(dtype=float),
    precomputed: int,
):
    """Tabulate one smooth edge's proposal on the device, one thread per edge.

    The node densities (``_node_density``: read from ``f`` when ``precomputed``,
    else evaluated here) turned into trapezoid cell masses
    (:func:`~pyradmc.geometry.edge_rays.proposal_cell_mass`), then into the cumulative
    ``cdf`` and the per-cell density ``cell`` by
    :func:`~pyradmc.geometry.edge_rays.edge_proposal_tables` (defensive share,
    sequential normalization). A fixed summation order; both density routes give
    the same bits (test-pinned). Any consistent proposal is unbiased, and the
    generator reads exactly these tables.
    """
    t = wp.tid()
    j = smooth[t]
    n = es.n_table
    soff = es.s_row[j] * n
    coff = es.table[j] * (n - 1)
    mu_block = _mu(es, es.device[j], es.e_ref[j])
    # The derivative mass per cell, and its total.
    total = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    f_prev = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    for k in range(n):
        sk = es.s[soff + k]
        # Warp compiles the conditional as two branches and a select: lazy, not both.
        f_k = f[k * smooth.shape[0] + t] if precomputed != 0 else _node_density(es, j, sk, mu_block)
        if k > 0:
            mass = proposal_cell_mass(f_prev, f_k, es.s[soff + k - 1], sk)
            es.cell[coff + k - 1] = mass
            total += mass
        f_prev = f_k
    edge_proposal_tables(es.cell, es.cdf, coff, es.s, soff, n - 1, total, es.defensive[j])


@wp.func
def _queue_edge_ray(
    es: EdgeSet,
    j: int,
    factor: float,
    energy: float,
    u_r: float,
    u_q: float,
    local: int,
    tid: int,
    slots: wp.array(dtype=wp.uint32),
    emitted: wp.array(dtype=wp.int64),
    q: Queue,
):
    """Queue edge ``j``'s ray at ``(u_r, u_q)``, its weight x ``factor``.

    The host ``FanEdgeDerivativeSource._rays`` through the same
    :mod:`~pyradmc.geometry.edge_rays` functions, on the edge's uploaded columns;
    only the transmissions differ in form (here the stack's optical depth with the
    block's chord subtracted, ``rest_transmission``; there a retracted stack).
    """
    kind = es.kind[j]
    sign = es.sign[j]
    seg = es.segment[j]
    cells = es.n_table - 1
    s, qq, rr, dens = edge_ray_point(
        kind, es.x[j], es.q0[j], sign, es.r_lo[j], es.r_hi[j], es.cdf, es.cell, es.s,
        es.table[j] * cells, es.s_row[j] * es.n_table, cells, _DENSITY_FLOOR, u_r, u_q,
    )  # fmt: skip
    lu, lv, du, dv, dw = edge_ray_direction(qq, rr, es.q_axis[j], es.z_mid[j])
    inside = edge_in_field(lu, lv, es.u_lo[j], es.u_hi[j], es.v_lo[j], es.v_hi[j])

    row = es.device[j]
    mu_block = _mu(es, row, energy)
    l_block, rate, l_rest = _block_terms(es, j, qq, s, du, dv, dw)
    tau = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    for dev in range(es.seg_lo[seg], es.seg_hi[seg]):
        tau += _mu(es, dev, energy) * device_path_length(
            es.stack, dev, 0.0, 0.0, 0.0, du, dv, dw, 0
        )
    t_rest = rest_transmission(tau, mu_block, l_rest)
    # On a focused face (a jump) every ray sees the full slant thickness of the slab.
    l_att = l_block
    if kind == 2:
        l_att = focused_slab_chord(es.b_z_top[j], es.b_z_bottom[j], dw)
    core = edge_core(kind, sign, s, mu_block, wp.exp(-mu_block * l_att), rate, dens)
    weight = edge_weight(inside, factor, es.scale[j], core, t_rest)

    ex, ey, ez = frame_direction(es.seg_rot, seg * 9, du, dv, dw)
    q.kind[tid] = PHOTON
    q.beamlet[tid] = local
    q.primary[tid] = 1
    q.energy[tid] = energy
    q.weight[tid] = weight
    q.x[tid] = es.seg_origin[seg * 3 + 0]
    q.y[tid] = es.seg_origin[seg * 3 + 1]
    q.z[tid] = es.seg_origin[seg * 3 + 2]
    q.ux[tid] = ex
    q.uy[tid] = ey
    q.uz[tid] = ez
    q.rng[tid] = slots[tid]
    _escape(emitted, weight * energy)


@wp.kernel
def generate_fan_edges(
    seed: int,
    group_start: int,
    n_per: int,
    per_batch: int,
    batch: int,
    t_offset: int,
    correlated: int,
    sp_edges: wp.array(dtype=float),
    sp_cdf: wp.array(dtype=float),
    n_bins: int,
    es: EdgeSet,
    slots: wp.array(dtype=wp.uint32),
    emitted: wp.array(dtype=wp.int64),
    q: Queue,
):
    """Edge-gradient primaries for one batch's block of a beamlet group.

    The spectral beamlet route's thread-to-history mapping and RNG keying, then
    the host ``FanEdgeDerivativeSource._rays`` arithmetic, through the same scalar
    functions (:mod:`pyradmc.geometry.edge_rays`), in float32. Four uniforms
    per primary, in the host order: two for the energy, one along the edge, one
    across it. Emitted ``weight * energy`` is booked signed (closing edges are
    negative), like the wrapped-sampler route.
    """
    tid = wp.tid()
    t = t_offset + tid
    local = t // per_batch
    i = t - local * per_batch
    r = batch * per_batch + i
    h = (group_start + local) * n_per + r
    key = h
    if correlated != 0:
        key = r
    slots[tid] = init_slot(seed, key)
    state = WarpRNGState()
    state.slots = slots
    state.idx = tid
    u_bin = uniform(state)
    u_within = uniform(state)
    u_r = uniform(state)
    u_q = uniform(state)
    energy = spectrum_energy(sp_edges, sp_cdf, n_bins, u_bin, u_within)

    j = group_start + local
    _queue_edge_ray(es, j, 1.0, energy, u_r, u_q, local, tid, slots, emitted, q)


@wp.kernel
def generate_edge_steps(
    seed: int,
    group_start: int,
    n_per: int,
    per_batch: int,
    batch: int,
    t_offset: int,
    correlated: int,
    sp_edges: wp.array(dtype=float),
    sp_cdf: wp.array(dtype=float),
    n_bins: int,
    es: EdgeSet,
    slots: wp.array(dtype=wp.uint32),
    emitted: wp.array(dtype=wp.int64),
    q: Queue,
):
    """Edge-step primaries (:class:`SegmentedEdgeStepSource`) for one batch of a group.

    :func:`generate_fan_edges` with the edge drawn per history, one uniform before its
    four, stratified over the batch's histories of the column: each batch holds every edge,
    so its mean stays an estimate of the column's, and each edge's share follows its travel
    to within one history, which the host's independent draw has only on average.
    """
    tid = wp.tid()
    t = t_offset + tid
    local = t // per_batch
    i = t - local * per_batch
    r = batch * per_batch + i
    column = group_start + local
    h = column * n_per + r
    key = h
    if correlated != 0:
        key = r
    slots[tid] = init_slot(seed, key)
    state = WarpRNGState()
    state.slots = slots
    state.idx = tid
    position = stratified_position(i, per_batch, uniform(state))
    u_bin = uniform(state)
    u_within = uniform(state)
    u_r = uniform(state)
    u_q = uniform(state)
    energy = spectrum_energy(sp_edges, sp_cdf, n_bins, u_bin, u_within)
    # The first entry whose cumulative share exceeds the position (the host's pick,
    # select_by_cdf_at); the last holds 1.
    first = es.step_first[column]
    pick = first + select_by_cdf_at(es.step_cdf, first, es.step_first[column + 1] - first, position)
    _queue_edge_ray(
        es, es.step_edge[pick], es.step_factor[pick], energy, u_r, u_q, local, tid, slots, emitted,
        q,
    )  # fmt: skip
