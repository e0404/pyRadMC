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
in-kernel, mirroring the host ``_rays`` arithmetic in float32:

- energy by spectrum CDF inversion (the spectral beamlet route's draws), then the
  position along the edge, then the proposal draw ``q = q0 + sign s^2`` from the
  edge's tabulated CDF (a jump edge sits on ``q = x``);
- the focal ray through ``(q, r)`` on the parameter device's mid-plane;
- the weight ``scale * (-mu T_block dl/dx) / p(q) * T_rest`` from the compiled chord
  twins and their edge derivatives, or ``scale * sign (1 - T_full) * T_rest`` on a
  jump, where ``T_rest = exp(-(tau_stack - mu l_block))``: the parameter block's
  optical depth enters the stack's additively, so no retracted stack is needed.

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
jaw_path_length = _p.jaw_path_length
jaw_edge_derivative = _p.jaw_edge_derivative
mlc_leaf_path_length = _p.mlc_leaf_path_length
mlc_tip_edge_derivative = _p.mlc_tip_edge_derivative

# The host transmission tables' grid (collimation._StackTransmission via the
# aperture-gradient sources): same nodes, same interpolation.
_E_MIN = 0.010
_N_MU = 512
# Below this distance from a rounded tip's tangent ray (s_t^2 = |q - q_t|, cm) the
# chord takes the closed-form near-tangent path (_block_terms: the disc chord, clipped
# where a slab face cuts it). Above it the twin answers: just past 2e-3 in the DAO
# regime its float32 error is 3-7e-3 relative in the rate and up to 2.5e-3 cm in the
# chord, unbiased (derivative-mass ratio 0.9998, measured 2026-10-23).
_NEAR_TANGENT_S2 = wp.constant(2.0e-3)


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
    """Macroscopic mu (1/cm) of global device ``row``: the host np.interp in log E."""
    t = (wp.log(energy) - es.log_e_min) * es.inv_dlog
    t = wp.min(wp.max(t, 0.0), float(es.n_mu - 1))
    i = wp.min(int(t), es.n_mu - 2)
    f = t - float(i)
    return wp.exp(es.log_mu[row, i] * (1.0 - f) + es.log_mu[row, i + 1] * f) * es.stack.density[row]


@wp.func
def _block_terms(es: EdgeSet, j: int, qq: float, s: float, du: float, dv: float, dw: float):
    """Return ``(l_block, rate, l_rest)`` of edge ``j``'s block along one focal ray.

    ``l_block``/``rate`` are the chord and its edge derivative that the weight uses;
    ``l_rest`` is the twin's chord, which the stack's optical depth also contains,
    so ``tau_stack - mu l_rest`` is the rest of the stack exactly. Near a rounded
    tip's tangent ray ``q_t`` the twin's float32 circle test cancels
    (``|p - c|^2 - R^2 ~ 1e-5`` against an ulp of 8e-6 at ``R = 8`` cm) and loses
    the rays carrying the largest weights. There the gap between ray and circle
    factorizes without cancellation, in the ray's distance from the tangent ray
    ``s_t^2 = |q - q_t| = s^2 + t_off`` (``t_off = sign (q0 - q_t)``, float64 on the
    host: zero when ``q0`` is the tangent ray, the lead of a bisected start whose
    tangent point lies outside the slab),

        g = s_t^2 (z + sign R (q + q_t) / (A + A_t)) / A,   A = sqrt(q^2 + z^2),

    and the chord is the disc chord, so chord and rate are closed-form: exact, not an
    expansion. The branch is taken for ``s_t^2 < _NEAR_TANGENT_S2`` on every rounded
    tip, whether its support start is the tangent ray or a bisected one: a tangent
    point just outside a face makes the corner ray nearly tangent, and the twin would
    cancel there exactly as at a tangent start (0.4-3 % of the derivative mass,
    measured, 2026-10-23 review). Then:

    - the bare disc chord lies inside the slab, checked per ray: both chord ends
      ``(t_c -+ half) cw`` within ``[z_top, z_bottom]``, with ``t_c`` the projection
      of the tip centre on the ray in the ``(u, w)`` plane. Where a face cuts it, the
      chord is clipped there in closed form too: a clipped end does not move with
      the edge, an open one moves at ``dt_c/dx +- dhalf/dx`` (``dt_c/dx = cu``), so a
      genuine tangent near a face keeps the cancellation-free form. The bare chord
      and its rate were up to 1.9x and 2x too large there (measured).

    Until 2026-10-22 the branch was taken on ``s^2`` alone with ``q0`` as the tangent
    ray, which biased bisected starts and face-cut chords by 15-30 % (z up to 14
    against the host route; decisions.md, "Vectorized edge sets"). Where ``q0`` is
    the tangent ray and the chord lies in the slab, the closed form is unchanged, bit
    for bit (``t_off`` is zero and ``q_t`` is ``q0``).
    """
    kind = es.kind[j]
    sign = es.sign[j]
    zm = es.z_mid[j]
    above = int(0)  # noqa: UP018, RUF046 -- Warp needs the constructor to declare a mutated var
    if sign > 0.0:
        above = 1
    l_block = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    rate = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    l_rest = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if kind == 0:
        l_block = mlc_leaf_path_length(
            0.0, 0.0, 0.0, du, dv, dw, es.b_z_top[j], es.b_z_bottom[j], es.b_radius[j],
            es.b_lo[j], es.b_hi[j], es.x[j], 1, above,
        )  # fmt: skip
        # The stack's optical depth below holds the twin's chord of this very leaf:
        # subtracting the same float32 value keeps T_rest exact either way.
        l_rest = l_block
        closed = int(0)  # noqa: UP018, RUF046 -- Warp needs the constructor to declare a mutated var
        s_t2 = s * s + es.t_off[j]  # the ray's distance from the tangent ray, squared s
        if s_t2 < _NEAR_TANGENT_S2:
            # The closed-form near-tangent chord (see the docstring).
            radius = es.b_radius[j]
            q_t = es.q_t[j]
            a_q = wp.sqrt(qq * qq + zm * zm)
            a_0 = wp.sqrt(q_t * q_t + zm * zm)
            gap = s_t2 * (zm + sign * radius * (qq + q_t) / (a_q + a_0)) / a_q
            gap = wp.max(gap, 0.0)
            half = wp.sqrt(gap * (2.0 * radius - gap))
            to_3d = 1.0 / wp.sqrt(du * du + dw * dw)
            # Both ends of the bare disc chord inside the slab, along the (u, w) ray.
            cw = dw * to_3d
            t_c = (es.x[j] + sign * radius) * du * to_3d + zm * cw
            closed = 1
            if (t_c - half) * cw >= es.b_z_top[j] and (t_c + half) * cw <= es.b_z_bottom[j]:
                l_block = 2.0 * half * to_3d
                rate = 0.0
                if half > 0.0:
                    rate = to_3d * 2.0 * (radius - gap) / half * (-sign * zm / a_q)
            else:
                # A slab face cuts the chord: clip it there; a clipped end stays put
                # when the edge moves, an open one moves at dt_c/dx +- dhalf/dx.
                t_top = es.b_z_top[j] / cw
                t_bottom = es.b_z_bottom[j] / cw
                t_lo = wp.max(t_c - half, t_top)
                t_hi = wp.min(t_c + half, t_bottom)
                l_block = wp.max(t_hi - t_lo, 0.0) * to_3d
                rate = 0.0
                if half > 0.0 and t_hi > t_lo:
                    d_half = (radius - gap) / half * (-sign * zm / a_q)
                    d_c = du * to_3d
                    if t_c + half <= t_bottom:
                        rate += d_c + d_half
                    if t_c - half >= t_top:
                        rate -= d_c - d_half
                    rate = rate * to_3d
        if closed == 0:
            rate = mlc_tip_edge_derivative(
                0.0, 0.0, 0.0, du, dv, dw, es.b_z_top[j], es.b_z_bottom[j], es.b_radius[j],
                es.b_lo[j], es.b_hi[j], es.x[j], 1, above,
            )  # fmt: skip
    else:
        d_q = du
        if es.q_axis[j] == 1:
            d_q = dv
        l_block = jaw_path_length(
            0.0, d_q, 0.0, dw, es.b_z_top[j], es.b_z_bottom[j], es.b_lo[j], es.b_hi[j],
            es.b_focused[j], 0,
        )  # fmt: skip
        l_rest = l_block
        rate = jaw_edge_derivative(
            0.0, d_q, 0.0, dw, es.b_z_top[j], es.b_z_bottom[j], es.b_lo[j], es.b_hi[j],
            es.b_focused[j], above,
        )  # fmt: skip

    return l_block, rate, l_rest


@wp.func
def _node_density(es: EdgeSet, j: int, sk: float, mu_block: float) -> float:
    """``f = 2 s |dT/dx|`` of edge ``j`` at proposal node ``s = sk``.

    Along the edge's central ray ``r_c`` at the reference energy (``mu_block``), at
    ``q = q0 + sign s^2``: the derivative density in ``s``, finite at a tip's
    tangent ray, zero at ``s = 0``.
    """
    zm = es.z_mid[j]
    rr = es.r_c[j]
    qq = es.q0[j] + es.sign[j] * sk * sk
    lu = qq
    lv = rr
    if es.q_axis[j] == 1:
        lu = rr
        lv = qq
    inv = 1.0 / wp.sqrt(lu * lu + lv * lv + zm * zm)
    l_block, rate, _l_rest = _block_terms(es, j, qq, sk, lu * inv, lv * inv, zm * inv)
    f_k = 2.0 * sk * wp.abs(mu_block * wp.exp(-mu_block * l_block) * rate)
    if sk == 0.0:
        f_k = 0.0
    return f_k


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
    """Tabulate one smooth edge's proposal: the host ``_proposal``, on the device.

    The node densities (``_node_density``: read from ``f`` when ``precomputed``,
    else evaluated here) turned into cell masses ``(f_k + f_k+1) ds / 2``, mixed
    with the defensive uniform share and normalized into the cumulative ``cdf`` and
    the per-cell density ``cell``. One thread per edge, sequential over its cells (a
    fixed summation order; both density routes give the same bits, test-pinned);
    any consistent proposal is unbiased, and the generator reads exactly these
    tables.
    """
    t = wp.tid()
    j = smooth[t]
    n = es.n_table
    soff = es.s_row[j] * n
    coff = es.table[j] * (n - 1)
    s_max = es.s[soff + n - 1]
    mu_block = _mu(es, es.device[j], es.e_ref[j])
    # First pass: the derivative mass per cell, and its total.
    total = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    f_prev = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    for k in range(n):
        sk = es.s[soff + k]
        # Warp compiles the conditional as two branches and a select: lazy, not both.
        f_k = f[k * smooth.shape[0] + t] if precomputed != 0 else _node_density(es, j, sk, mu_block)
        if k > 0:
            mass = 0.5 * (f_prev + f_k) * (sk - es.s[soff + k - 1])
            es.cell[coff + k - 1] = mass
            total += mass
        f_prev = f_k
    # Second pass: mix with the defensive share, normalize, accumulate.
    share = es.defensive[j]
    running = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    for c in range(n - 1):
        width = es.s[soff + c + 1] - es.s[soff + c]
        mass = width / s_max
        if total > 0.0:
            mass = (1.0 - share) * es.cell[coff + c] / total + share * width / s_max
        running += mass
        es.cdf[coff + c] = running
        es.cell[coff + c] = mass / width
    # The cumulative sum ends at 1 up to rounding; pin the last cell exactly.
    es.cdf[coff + n - 2] = 1.0


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
    """Queue edge ``j``'s ray at ``(u_r, u_q)``: the host ``_rays``, its weight x ``factor``."""
    kind = es.kind[j]
    sign = es.sign[j]
    seg = es.segment[j]
    rr = es.r_lo[j] + (es.r_hi[j] - es.r_lo[j]) * u_r
    qq = es.x[j]
    dens = float(1.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    s = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if kind != 2:
        cells = es.n_table - 1
        coff = es.table[j] * cells
        # The first cell whose cumulative mass exceeds u_q (searchsorted 'right').
        lo_c = int(0)  # noqa: UP018, RUF046 -- Warp needs the constructor to declare a mutated var
        hi_c = cells - 1
        while lo_c < hi_c:
            mid = (lo_c + hi_c) // 2
            if es.cdf[coff + mid] > u_q:
                hi_c = mid
            else:
                lo_c = mid + 1
        lower = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
        if lo_c > 0:
            lower = es.cdf[coff + lo_c - 1]
        width = es.cdf[coff + lo_c] - lower
        resid = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
        if width > 0.0:
            resid = wp.min(wp.max((u_q - lower) / width, 0.0), 1.0)
        soff = es.s_row[j] * es.n_table
        s = es.s[soff + lo_c] + (es.s[soff + lo_c + 1] - es.s[soff + lo_c]) * resid
        qq = es.q0[j] + sign * s * s
        dens = es.cell[coff + lo_c] / wp.max(2.0 * s, 1.0e-30)

    lu = qq
    lv = rr
    if es.q_axis[j] == 1:
        lu = rr
        lv = qq
    zm = es.z_mid[j]
    inv = 1.0 / wp.sqrt(lu * lu + lv * lv + zm * zm)
    du = lu * inv
    dv = lv * inv
    dw = zm * inv
    inside = lu >= es.u_lo[j] and lu <= es.u_hi[j] and lv >= es.v_lo[j] and lv <= es.v_hi[j]

    row = es.device[j]
    mu_block = _mu(es, row, energy)
    l_block, rate, l_rest = _block_terms(es, j, qq, s, du, dv, dw)
    tau = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    for dev in range(es.seg_lo[seg], es.seg_hi[seg]):
        tau += _mu(es, dev, energy) * device_path_length(
            es.stack, dev, 0.0, 0.0, 0.0, du, dv, dw, 0
        )
    t_rest = wp.exp(-(tau - mu_block * l_rest))

    core = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if kind == 2:
        # On a focused face every ray sees the full slant thickness of the slab.
        l_full = (es.b_z_bottom[j] - es.b_z_top[j]) / wp.abs(dw)
        core = sign * (1.0 - wp.exp(-mu_block * l_full))
    elif s > 0.0:
        core = -mu_block * wp.exp(-mu_block * l_block) * rate / dens
    weight = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if inside:
        weight = factor * es.scale[j] * core * t_rest

    ro = seg * 9
    ex = du * es.seg_rot[ro + 0] + dv * es.seg_rot[ro + 3] + dw * es.seg_rot[ro + 6]
    ey = du * es.seg_rot[ro + 1] + dv * es.seg_rot[ro + 4] + dw * es.seg_rot[ro + 7]
    ez = du * es.seg_rot[ro + 2] + dv * es.seg_rot[ro + 5] + dw * es.seg_rot[ro + 8]
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
    the host ``FanEdgeDerivativeSource._rays`` arithmetic in float32. Four uniforms
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
    kb = wp.min(wp.lower_bound(sp_cdf, u_bin), n_bins - 1)
    e_lo = sp_edges[kb]
    energy = e_lo + (sp_edges[kb + 1] - e_lo) * u_within

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
    position = (float(i) + uniform(state)) / float(per_batch)
    u_bin = uniform(state)
    u_within = uniform(state)
    u_r = uniform(state)
    u_q = uniform(state)
    kb = wp.min(wp.lower_bound(sp_cdf, u_bin), n_bins - 1)
    e_lo = sp_edges[kb]
    energy = e_lo + (sp_edges[kb + 1] - e_lo) * u_within
    # The first entry whose cumulative share exceeds the position; the last holds 1.
    lo = es.step_first[column]
    hi = es.step_first[column + 1] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if es.step_cdf[mid] > position:
            hi = mid
        else:
            lo = mid + 1
    _queue_edge_ray(
        es, es.step_edge[lo], es.step_factor[lo], energy, u_r, u_q, local, tid, slots, emitted, q
    )
