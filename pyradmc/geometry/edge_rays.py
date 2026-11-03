"""Single-source scalar arithmetic of the fan edge-gradient rays.

Pure, scalar, allocation-free functions of scalars and array handles (AGENTS.md 2.5):
they run under NumPy on the host, where
:class:`~pyradmc.geometry.aperture_gradient.FanEdgeDerivativeSource` calls them ray by
ray, and the Warp physics loader compiles them to ``@wp.func`` for the in-kernel edge
generators (:mod:`pyradmc.backends.warp.aperture`), which keep only the uploads, the
launches, the RNG keying, the optical-depth loop over a segment's stack devices and
the queue writes. Estimator, notation and stated
approximations are :class:`~pyradmc.geometry.aperture_gradient.FanEdgeDerivativeSource`'s:
a ray is fixed by its crossing ``(q, r)`` of the parameter device's mid-plane, ``q``
across the edge and ``r`` along it, and its weight is

    scale * dT/dx / p(q) * T_rest,   dT/dx = -mu T_block dl/dx   (smooth edge),
    scale * sign (1 - T_full) * T_rest                           (focused jaw: a jump).

Shared by both routes, so neither can drift from the other:

- :func:`edge_ray_point` draws the crossing: ``r`` uniform along the edge and, on a
  smooth edge, ``q = q0 + sign s^2`` with ``s`` from the tabulated proposal
  (:func:`edge_proposal_point`, the inversion of a cumulative table); a jump sits on
  the edge line.
- :func:`edge_ray_crossing` places the crossing in the frame, with the distance that
  normalizes the focal ray through it, and :func:`edge_in_field` is its field test.
- :func:`edge_chord_rate` dispatches the chord's edge derivative to the collimation
  twins (:func:`~pyradmc.geometry.collimation.mlc_tip_edge_derivative`,
  :func:`~pyradmc.geometry.collimation.jaw_edge_derivative`).
- :func:`edge_core` and :func:`edge_weight` turn transmission, rate and proposal
  density into the weight; :func:`edge_node_density`, :func:`proposal_cell_mass` and
  :func:`defensive_mass` build the proposal from the derivative density.

Taken by the kernel only, because in float32 a quantity the host evaluates directly
must be re-expressed (each pinned against the host's float64 value in
``tests/unit/test_edge_rays.py``):

- :func:`edge_block_terms` and :func:`near_tangent_terms`: the block's chord and rate,
  with a cancellation-free closed form near a rounded tip's tangent ray, where the
  twin's float32 circle test cancels. The host's float64 twin does not need it.
- :func:`focused_slab_chord`, :func:`rest_transmission`: a jump's slant chord and the
  rest of the stack's transmission by subtraction from the whole stack's optical
  depth, where the host evaluates a retracted stack.
- :func:`edge_proposal_tables`: the proposal normalized by one sequential sum and
  pinned to 1 at its end, where the host renormalizes by the pairwise total. Any
  consistent proposal is unbiased and each route reads exactly the table it built.
- :func:`edge_ray_direction`: the unit direction by the norm's reciprocal, where the
  host divides by the norm; :func:`frame_direction`: the rotation to the engine
  frame, a matrix product on the host.
- :func:`stratified_position`: an edge step's stratified draw over a batch's histories,
  where the host's :meth:`~pyradmc.geometry.segment_edge_steps.SegmentedEdgeStepSource.emit`
  draws independently.

The float arguments are float64 on the host and float32 in a kernel; constants that
depend on that (the proposal-density floor, the near-tangent threshold) are passed
in. Integer loop variables are declared with ``int(0)`` and floats with
``float(0.0)`` because Warp needs the cast to declare a variable mutated in a branch
or loop.
"""

from __future__ import annotations

import math

from pyradmc.data.handles import Table1D
from pyradmc.geometry.collimation import (
    jaw_edge_derivative,
    jaw_path_length,
    mlc_leaf_path_length,
    mlc_tip_edge_derivative,
)
from pyradmc.geometry.fan import select_by_cdf_at

__all__ = [
    "defensive_mass",
    "edge_block_terms",
    "edge_chord",
    "edge_chord_rate",
    "edge_core",
    "edge_in_field",
    "edge_node_density",
    "edge_proposal_point",
    "edge_proposal_tables",
    "edge_q",
    "edge_ray_crossing",
    "edge_ray_direction",
    "edge_ray_point",
    "edge_weight",
    "focused_slab_chord",
    "frame_direction",
    "near_tangent_terms",
    "proposal_cell_mass",
    "rest_transmission",
    "stratified_position",
]


# -- the ray: crossing, direction, field ---------------------------------------------


def edge_q(q0: float, sign: float, s: float) -> float:
    """Mid-plane crossing ``q = q0 + sign s^2`` of proposal coordinate ``s``.

    ``sign`` is exactly +-1, so ``(sign s) s`` and ``sign (s s)`` are the same float.
    """
    return q0 + sign * s * s


def edge_proposal_point(
    cdf: Table1D,
    cell: Table1D,
    s_grid: Table1D,
    cell_offset: int,
    s_offset: int,
    n_cells: int,
    u: float,
    q0: float,
    sign: float,
    density_floor: float,
) -> tuple[float, float, float]:
    """Invert a tabulated proposal at the uniform ``u``: ``(s, q, density in q)``.

    The ``n_cells`` cumulative masses ``cdf`` and cell densities ``cell`` (in ``s``)
    start at ``cell_offset``, the ``n_cells + 1`` nodes of ``s`` at ``s_offset``. The
    cell is the first whose cumulative mass exceeds ``u`` (``searchsorted``
    ``side="right"``, so a zero-mass cell is never drawn), clamped to the last;
    ``s`` is linear in the residual within it, clamped to the cell. The density in
    ``q`` is the cell's density in ``s`` over ``|dq/ds| = 2 s``, floored at
    ``density_floor`` (1e-300 in float64, 1e-30 in float32): only ``s = 0`` reaches
    the floor, and :func:`edge_core` gives that ray no weight.
    """
    k = select_by_cdf_at(cdf, cell_offset, n_cells, u)
    lower = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if k > 0:
        lower = cdf[cell_offset + k - 1]
    width = cdf[cell_offset + k] - lower
    residual = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if width > 0.0:
        residual = min(max((u - lower) / width, 0.0), 1.0)
    s_lo = s_grid[s_offset + k]
    s = s_lo + (s_grid[s_offset + k + 1] - s_lo) * residual
    density = cell[cell_offset + k] / max(2.0 * s, density_floor)
    return s, edge_q(q0, sign, s), density


def edge_ray_point(
    kind: int,
    x: float,
    q0: float,
    sign: float,
    r_lo: float,
    r_hi: float,
    cdf: Table1D,
    cell: Table1D,
    s_grid: Table1D,
    cell_offset: int,
    s_offset: int,
    n_cells: int,
    density_floor: float,
    u_r: float,
    u_q: float,
) -> tuple[float, float, float, float]:
    """Draw an edge ray's mid-plane crossing from two uniforms: ``(s, q, r, density)``.

    ``r`` is uniform over the edge's extent ``[r_lo, r_hi]`` (``u_r``). A smooth edge
    (``kind`` 0, a rounded tip, or 1, a straight jaw face) draws ``s`` and ``q`` from
    its proposal (``u_q``, :func:`edge_proposal_point`); a jump (``kind`` 2, a focused
    jaw) puts every ray on the edge line ``q = x``, at ``s = 0`` and unit density, and
    reads no table.
    """
    r = r_lo + (r_hi - r_lo) * u_r
    s = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    q = float(x)
    density = float(1.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if kind != 2:
        s, q, density = edge_proposal_point(
            cdf, cell, s_grid, cell_offset, s_offset, n_cells, u_q, q0, sign, density_floor
        )
    return s, q, r, density


def edge_ray_crossing(q: float, r: float, q_axis: int, z_mid: float) -> tuple[float, float, float]:
    """Return the mid-plane crossing ``(q, r)`` in the frame and its distance: ``(lu, lv, norm)``.

    ``(lu, lv)`` is the crossing in the frame's ``(u, v)`` (``q`` is ``v`` when
    ``q_axis`` is 1) and ``norm = |(lu, lv, z_mid)|``, so the focal ray's local unit
    direction is ``(lu, lv, z_mid) / norm``. The float64 host divides by ``norm`` (as
    ``np.linalg.norm`` normalization does, bit for bit); the float32 kernel multiplies
    by its reciprocal (:func:`edge_ray_direction`).
    """
    lu = q
    lv = r
    if q_axis == 1:
        lu = r
        lv = q
    return lu, lv, math.sqrt(lu * lu + lv * lv + z_mid * z_mid)


def edge_ray_direction(
    q: float, r: float, q_axis: int, z_mid: float
) -> tuple[float, float, float, float, float]:
    """Return the kernel's focal ray through crossing ``(q, r)``: ``(lu, lv, du, dv, dw)``.

    :func:`edge_ray_crossing`, normalized by one reciprocal and three products, the
    in-kernel generator's form since it was written. The host divides instead; the
    two differ by an ulp per component, and they are kept apart because the twin's
    float32 circle test amplifies that ulp near a tip's tangent ray: dividing in the
    kernel moved its proposal tables by up to 5e-4 and the fixture's edge
    contractions by up to 8e-4 relative (Warp cpu, measured 2026-11-03), where the
    rest of the shared arithmetic changed them at float32 rounding only.
    """
    lu, lv, norm = edge_ray_crossing(q, r, q_axis, z_mid)
    inv = 1.0 / norm
    return lu, lv, lu * inv, lv * inv, z_mid * inv


def edge_in_field(lu: float, lv: float, u_lo: float, u_hi: float, v_lo: float, v_hi: float) -> int:
    """Return 1 if the crossing ``(lu, lv)`` lies in the closed mid-plane field, else 0."""
    inside = int(0)  # noqa: UP018, RUF046 -- Warp needs the constructor to declare a mutated var
    if lu >= u_lo and lu <= u_hi and lv >= v_lo and lv <= v_hi:
        inside = 1
    return inside


def frame_direction(
    rotation: Table1D, offset: int, du: float, dv: float, dw: float
) -> tuple[float, float, float]:
    """Engine-frame direction of the local ``(du, dv, dw)``: ``d @ R``.

    ``rotation[offset:offset + 9]`` holds the frame's ``u``, ``v`` and ``w`` axes row by
    row (``CompiledStack.rotation``). The host applies the same matrix as one array
    product.
    """
    o = offset
    ex = du * rotation[o + 0] + dv * rotation[o + 3] + dw * rotation[o + 6]
    ey = du * rotation[o + 1] + dv * rotation[o + 4] + dw * rotation[o + 7]
    ez = du * rotation[o + 2] + dv * rotation[o + 5] + dw * rotation[o + 8]
    return ex, ey, ez


# -- the parameter block: chord and edge rate ------------------------------------------


def edge_chord(
    kind: int,
    sign: float,
    x: float,
    z_top: float,
    z_bottom: float,
    radius: float,
    lo: float,
    hi: float,
    focused: int,
    q_axis: int,
    du: float,
    dv: float,
    dw: float,
) -> float:
    """Chord (cm) of the focal ray ``(du, dv, dw)`` through the isolated block (twins).

    ``kind`` 0 is a rounded tip, one leaf of strip ``[lo, hi]`` at ``x`` with radius
    ``radius`` and focused sides (:func:`~pyradmc.geometry.collimation.mlc_leaf_path_length`);
    otherwise a jaw with edges ``lo``/``hi`` along local axis ``q_axis``
    (:func:`~pyradmc.geometry.collimation.jaw_path_length`). The block's side is
    ``sign`` (+1: the ``+axis`` block).
    """
    above = int(0)  # noqa: UP018, RUF046 -- Warp needs the constructor to declare a mutated var
    if sign > 0.0:
        above = 1
    chord = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if kind == 0:
        chord = mlc_leaf_path_length(
            0.0, 0.0, 0.0, du, dv, dw, z_top, z_bottom, radius, lo, hi, x, 1, above
        )
    else:
        d_q = du
        if q_axis == 1:
            d_q = dv
        chord = jaw_path_length(0.0, d_q, 0.0, dw, z_top, z_bottom, lo, hi, focused, 0)
    return chord


def edge_chord_rate(
    kind: int,
    sign: float,
    x: float,
    z_top: float,
    z_bottom: float,
    radius: float,
    lo: float,
    hi: float,
    focused: int,
    q_axis: int,
    du: float,
    dv: float,
    dw: float,
) -> float:
    """Return the chord's edge derivative ``dl/dx`` (twins; see :func:`edge_chord`).

    :func:`~pyradmc.geometry.collimation.mlc_tip_edge_derivative` for a tip,
    :func:`~pyradmc.geometry.collimation.jaw_edge_derivative` for a jaw, both
    differentiating the moving side ``sign``.
    """
    above = int(0)  # noqa: UP018, RUF046 -- Warp needs the constructor to declare a mutated var
    if sign > 0.0:
        above = 1
    rate = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if kind == 0:
        rate = mlc_tip_edge_derivative(
            0.0, 0.0, 0.0, du, dv, dw, z_top, z_bottom, radius, lo, hi, x, 1, above
        )
    else:
        d_q = du
        if q_axis == 1:
            d_q = dv
        rate = jaw_edge_derivative(0.0, d_q, 0.0, dw, z_top, z_bottom, lo, hi, focused, above)
    return rate


def near_tangent_terms(
    sign: float,
    x: float,
    z_mid: float,
    z_top: float,
    z_bottom: float,
    radius: float,
    q_t: float,
    s_t2: float,
    qq: float,
    du: float,
    dw: float,
) -> tuple[float, float]:
    """Chord and edge rate of a rounded tip near its tangent ray, cancellation-free.

    Near the tangent ray ``q_t`` the twin's circle test cancels in float32
    (``|p - c|^2 - R^2 ~ 1e-5`` against an ulp of 8e-6 at ``R = 8`` cm) and loses the
    rays carrying the largest weights. There the gap between ray and circle
    factorizes without cancellation in the ray's distance from the tangent ray,
    ``s_t2 = s_t^2 = |q - q_t|``,

        g = s_t^2 (z + sign R (q + q_t) / (A + A_t)) / A,   A = sqrt(q^2 + z^2),

    and the chord is the disc chord ``2 sqrt(g (2R - g))`` with its closed-form rate:
    exact, not an expansion. Where that chord lies inside the slab, checked per ray
    from the projection ``t_c`` of the tip centre on the ray in the ``(u, w)`` plane,
    it is returned whole; where a slab face cuts it, it is clipped there in closed
    form too: a clipped end does not move with the edge, an open one at
    ``dt_c/dx -+ dhalf/dx`` (``dt_c/dx = cu``), so a genuine tangent near a face keeps
    the cancellation-free form (the bare chord and its rate were up to 1.9x and 2x too
    large there, measured 2026-10-22). Lengths are 3-D (cm).
    """
    a_q = math.sqrt(qq * qq + z_mid * z_mid)
    a_0 = math.sqrt(q_t * q_t + z_mid * z_mid)
    gap = s_t2 * (z_mid + sign * radius * (qq + q_t) / (a_q + a_0)) / a_q
    gap = max(gap, 0.0)
    half = math.sqrt(gap * (2.0 * radius - gap))
    to_3d = 1.0 / math.sqrt(du * du + dw * dw)
    # Both ends of the bare disc chord inside the slab, along the (u, w) ray.
    cw = dw * to_3d
    t_c = (x + sign * radius) * du * to_3d + z_mid * cw
    chord = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    rate = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if (t_c - half) * cw >= z_top and (t_c + half) * cw <= z_bottom:
        chord = 2.0 * half * to_3d
        if half > 0.0:
            rate = to_3d * 2.0 * (radius - gap) / half * (-sign * z_mid / a_q)
    else:
        # A slab face cuts the chord: clip it there; a clipped end stays put when the
        # edge moves, an open one moves at dt_c/dx +- dhalf/dx.
        t_top = z_top / cw
        t_bottom = z_bottom / cw
        t_lo = max(t_c - half, t_top)
        t_hi = min(t_c + half, t_bottom)
        chord = max(t_hi - t_lo, 0.0) * to_3d
        if half > 0.0 and t_hi > t_lo:
            d_half = (radius - gap) / half * (-sign * z_mid / a_q)
            d_c = du * to_3d
            if t_c + half <= t_bottom:
                rate += d_c + d_half
            if t_c - half >= t_top:
                rate -= d_c - d_half
            rate = rate * to_3d
    return chord, rate


def edge_block_terms(
    kind: int,
    sign: float,
    x: float,
    z_mid: float,
    z_top: float,
    z_bottom: float,
    radius: float,
    lo: float,
    hi: float,
    focused: int,
    q_axis: int,
    q_t: float,
    t_off: float,
    near_tangent_s2: float,
    qq: float,
    s: float,
    du: float,
    dv: float,
    dw: float,
) -> tuple[float, float, float]:
    """Return ``(l_block, rate, l_rest)`` of an edge's block along one focal ray.

    ``l_block`` and ``rate`` are the chord and edge rate the weight uses; ``l_rest``
    is the twin's chord (:func:`edge_chord`), the one the stack's optical depth
    contains, so ``tau_stack - mu l_rest`` is the rest of the stack exactly
    (:func:`rest_transmission`). On a rounded tip (``kind`` 0) a ray within
    ``s_t^2 = s^2 + t_off < near_tangent_s2`` of the tangent ray ``q_t``
    (``t_off = sign (q0 - q_t)``: zero when the support start ``q0`` is the tangent
    ray, a bisected start's lead otherwise) takes :func:`near_tangent_terms`, whether
    its support start is the tangent ray or a bisected one: a tangent point just
    outside a face makes the corner ray nearly tangent, and the twin cancels there as
    at a tangent start (0.4-3 % of the derivative mass, measured in the 2026-10-23
    review). Everywhere else, and with ``near_tangent_s2 <= 0``, the twins
    answer (:func:`edge_chord`, :func:`edge_chord_rate`). The float32 kernel passes
    2e-3; in float64 the twin does not cancel and needs no such branch.

    Until 2026-10-22 the branch was taken on ``s^2`` alone with ``q0`` as the tangent
    ray, which biased bisected starts and face-cut chords by 15-30 % (z up to 14
    against the host route; docs/decisions.md, "Vectorized edge sets"). Where ``q0``
    is the tangent ray and the chord lies in the slab, the closed form is that
    original one, bit for bit (``t_off`` is zero and ``q_t`` is ``q0``).
    """
    l_rest = edge_chord(kind, sign, x, z_top, z_bottom, radius, lo, hi, focused, q_axis, du, dv, dw)
    l_block = float(l_rest)
    rate = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    closed = int(0)  # noqa: UP018, RUF046 -- Warp needs the constructor to declare a mutated var
    if kind == 0:
        s_t2 = s * s + t_off
        if s_t2 < near_tangent_s2:
            l_block, rate = near_tangent_terms(
                sign, x, z_mid, z_top, z_bottom, radius, q_t, s_t2, qq, du, dw
            )
            closed = 1
    if closed == 0:
        rate = edge_chord_rate(
            kind, sign, x, z_top, z_bottom, radius, lo, hi, focused, q_axis, du, dv, dw
        )
    return l_block, rate, l_rest


def focused_slab_chord(z_top: float, z_bottom: float, dw: float) -> float:
    """Slant chord of a focal ray through a focused jaw's slab: ``(z_bottom - z_top) / |dw|``.

    On a focused face every focal ray inside the block sees this full thickness.
    """
    return (z_bottom - z_top) / abs(dw)


def rest_transmission(tau: float, mu: float, l_rest: float) -> float:
    """Transmission of the stack without the parameter block, ``exp(-(tau - mu l_rest))``.

    ``tau`` is the whole stack's optical depth along the ray, in which the block's
    chord ``l_rest`` enters additively at attenuation ``mu``: no retracted stack is
    needed.
    """
    return math.exp(-(tau - mu * l_rest))


def stratified_position(index: int, count: int, u: float) -> float:
    """Return ``(index + u) / count``: the ``index``-th of ``count`` strata of ``[0, 1)``.

    An edge step's history ``index`` of a batch's ``count`` draws its edge at this
    position (``u`` uniform), so every batch holds every edge and each edge's share of
    the batch follows its travel to within one history. One position per stratum: the
    batch mean is unbiased.
    """
    return (float(index) + u) / float(count)


# -- the weight ------------------------------------------------------------------------


def edge_core(
    kind: int, sign: float, s: float, mu: float, t_block: float, rate: float, density: float
) -> float:
    """Return the weight's derivative factor ``dT/dx / p(q)``, or a jump's mass.

    A smooth edge: ``-mu T_block rate / density`` (``T_block = exp(-mu l_block)``),
    zero at ``s = 0`` (the support start, which carries no derivative mass). A jump
    (``kind`` 2): ``sign (1 - T_full)``, with ``t_block`` the full slab's
    transmission.
    """
    core = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if kind == 2:
        core = sign * (1.0 - t_block)
    elif s > 0.0:
        core = -mu * t_block * rate / density
    return core


def edge_weight(inside: int, factor: float, scale: float, core: float, t_rest: float) -> float:
    """Return a ray's weight ``factor * scale * core * T_rest``, zero outside the field.

    ``scale`` is ``L_r / A``; ``factor`` multiplies it (1 for an edge's own column, the
    signed travel total for an edge step).
    """
    weight = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if inside != 0:
        weight = factor * scale * core * t_rest
    return weight


# -- the proposal ----------------------------------------------------------------------


def edge_node_density(s: float, mu: float, t_block: float, rate: float) -> float:
    """Proposal density in ``s`` at a node: ``f = 2 s |dT/dx|``, ``dT/dx = -mu T rate``.

    The derivative density along the edge's central ray at the reference energy, in
    ``s``: finite at a tip's tangent ray, zero at ``s = 0``.
    """
    f = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    if s != 0.0:
        f = 2.0 * s * abs(-mu * t_block * rate)
    return f


def proposal_cell_mass(f_lo: float, f_hi: float, s_lo: float, s_hi: float) -> float:
    """Trapezoid mass ``(f_lo + f_hi) (s_hi - s_lo) / 2`` of one proposal cell."""
    return 0.5 * (f_lo + f_hi) * (s_hi - s_lo)


def defensive_mass(
    mass: float, total: float, defensive: float, width: float, s_max: float
) -> float:
    """Mix a cell's normalized mass with the defensive uniform share.

    ``(1 - defensive) mass / total + defensive width / s_max``: any consistent mixture
    covers the support, and the uniform share bounds the weight ratio (Hesterberg,
    Technometrics 37, 185 (1995), doi:10.1080/00401706.1995.10484303).
    """
    return (1.0 - defensive) * mass / total + defensive * (width / s_max)


def edge_proposal_tables(
    cell: Table1D,
    cdf: Table1D,
    cell_offset: int,
    s_grid: Table1D,
    s_offset: int,
    n_cells: int,
    total: float,
    defensive: float,
) -> None:
    """Turn cell masses into a proposal table in place: the device construction.

    On entry ``cell[cell_offset:]`` holds the ``n_cells`` cell masses
    (:func:`proposal_cell_mass`) and ``total`` their sum. Each is mixed with the
    defensive share (:func:`defensive_mass`; all uniform when ``total`` is zero),
    accumulated into ``cdf`` in one sequential pass and replaced by its density in
    ``s``; the last cumulative entry is pinned to 1. The host's proposal
    renormalizes by the pairwise total instead, so the two agree to rounding (pinned);
    each route reads only its own table.
    """
    s_max = s_grid[s_offset + n_cells]
    running = float(0.0)  # noqa: UP018 -- Warp needs the constructor to declare a mutated var
    for c in range(n_cells):
        width = s_grid[s_offset + c + 1] - s_grid[s_offset + c]
        mass = width / s_max
        if total > 0.0:
            mass = defensive_mass(cell[cell_offset + c], total, defensive, width, s_max)
        running += mass
        cdf[cell_offset + c] = running
        cell[cell_offset + c] = mass / width
    # The cumulative sum ends at 1 up to rounding; pin the last cell exactly.
    cdf[cell_offset + n_cells - 1] = 1.0
