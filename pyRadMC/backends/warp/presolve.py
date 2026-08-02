"""Device treatment-head pre-solve: the head MC as Warp kernels (cpu and cuda).

The host :func:`pyRadMC.geometry.head.presolve_head` is the ``ref`` oracle; this
module is its device twin. One thread per primary traces the ray through the
beam-limiting stack (and an optional air column), forces the first Compton
interaction, and appends the surviving exit-plane particles into shared output
buffers with an atomic counter — the same append pattern the transport queues use.
The geometry is the ``@wp.func`` collimation twins
(:func:`pyRadMC.geometry.collimation.jaw_path_length` /
:func:`~pyRadMC.geometry.collimation.mlc_path_length`), so host and device share one
definition; the Compton sampling and direction updates are the same single-source
physics the transport kernels call. Cross-target agreement is statistical only
(AGENTS.md 2.3): the device stream is a different generator from the host's, so the
exit phase spaces match in distribution and in the energy ledger, never bit-wise.

This module is Warp kernel code: exempt from ``mypy --strict`` (AGENTS.md 5) and
importable only when the ``warp`` optional dependency is installed.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from pyRadMC import ECUT_MEV, PCUT_MEV
from pyRadMC.backends.warp.physics import warp_physics
from pyRadMC.geometry.collimation import BeamLimitingStack, CompiledStack
from pyRadMC.geometry.head import AirColumn, _MuTables, resolve_presolve
from pyRadMC.geometry.phasespace import IAEA_ELECTRON, IAEA_PHOTON, InMemoryPhaseSpaceSource
from pyRadMC.geometry.source import Source
from pyRadMC.rng.warp_shim import WarpRNGState, init_slot, uniform

__all__ = [
    "DevicePhaseSpace",
    "ExitBuffer",
    "MuArrays",
    "StackArrays",
    "device_path_length",
    "presolve_head_device",
    "stack_path_lengths_kernel",
    "upload_stack",
]

# No differentiation here either: skip adjoint kernel compilation (see kernels.py).
wp.set_module_options({"enable_backward": False})

_p = warp_physics()
jaw_path_length = _p.jaw_path_length
mlc_path_length = _p.mlc_path_length
sample_compton_energy_ratio = _p.sample_compton_energy_ratio
compton_cos_theta = _p.compton_cos_theta
compton_electron_cos_theta = _p.compton_electron_cos_theta
rotate_direction = _p.rotate_direction

_TWO_PI = 2.0 * math.pi
_PI = math.pi


@wp.struct
class StackArrays:
    """A :class:`~pyRadMC.geometry.collimation.CompiledStack` uploaded to a device.

    Column-parallel over ``n_devices`` with the shared MLC leaf arrays; the field
    names mirror ``CompiledStack`` one-for-one. ``origin``/``rotation`` carry the
    beam frame so a kernel maps engine-frame rays to the local frame in place.
    """

    n_devices: int
    origin: wp.array(dtype=float)
    rotation: wp.array(dtype=float)
    kind: wp.array(dtype=wp.int32)
    z_top: wp.array(dtype=float)
    z_bottom: wp.array(dtype=float)
    material: wp.array(dtype=wp.int32)
    density: wp.array(dtype=float)
    jaw_axis: wp.array(dtype=wp.int32)
    jaw_edge_neg: wp.array(dtype=float)
    jaw_edge_pos: wp.array(dtype=float)
    jaw_focused: wp.array(dtype=wp.int32)
    mlc_tip_radius: wp.array(dtype=float)
    mlc_focused_sides: wp.array(dtype=wp.int32)
    mlc_n_pairs: wp.array(dtype=wp.int32)
    mlc_edges_off: wp.array(dtype=wp.int32)
    mlc_tips_off: wp.array(dtype=wp.int32)
    leaf_edges_v: wp.array(dtype=float)
    tips_neg: wp.array(dtype=float)
    tips_pos: wp.array(dtype=float)


def upload_stack(compiled: CompiledStack, device: str) -> StackArrays:
    """Copy a flattened stack to ``device`` as float32/int32 arrays."""
    stack = StackArrays()
    stack.n_devices = compiled.n_devices

    def farr(values: np.ndarray) -> wp.array:
        return wp.array(np.asarray(values, dtype=np.float32), dtype=float, device=device)

    def iarr(values: np.ndarray) -> wp.array:
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    stack.origin = farr(compiled.origin)
    stack.rotation = farr(compiled.rotation)
    stack.kind = iarr(compiled.kind)
    stack.z_top = farr(compiled.z_top)
    stack.z_bottom = farr(compiled.z_bottom)
    stack.material = iarr(compiled.material)
    stack.density = farr(compiled.density)
    stack.jaw_axis = iarr(compiled.jaw_axis)
    stack.jaw_edge_neg = farr(compiled.jaw_edge_neg)
    stack.jaw_edge_pos = farr(compiled.jaw_edge_pos)
    stack.jaw_focused = iarr(compiled.jaw_focused)
    stack.mlc_tip_radius = farr(compiled.mlc_tip_radius)
    stack.mlc_focused_sides = iarr(compiled.mlc_focused_sides)
    stack.mlc_n_pairs = iarr(compiled.mlc_n_pairs)
    stack.mlc_edges_off = iarr(compiled.mlc_edges_off)
    stack.mlc_tips_off = iarr(compiled.mlc_tips_off)
    stack.leaf_edges_v = farr(compiled.leaf_edges_v)
    stack.tips_neg = farr(compiled.tips_neg)
    stack.tips_pos = farr(compiled.tips_pos)
    return stack


@wp.func
def device_path_length(
    stack: StackArrays,
    j: int,
    p_u: float,
    p_v: float,
    p_w: float,
    d_u: float,
    d_v: float,
    d_w: float,
    from_origin: int,
) -> float:
    """Chord length of one local-frame ray through device ``j`` of the stack (cm).

    Dispatches on ``kind[j]`` to the jaw or MLC twin — the per-device half of
    :meth:`BeamLimitingStack.path_lengths`, in the kernel.
    """
    if stack.kind[j] == 0:
        if stack.jaw_axis[j] == 0:
            return jaw_path_length(
                p_u,
                d_u,
                p_w,
                d_w,
                stack.z_top[j],
                stack.z_bottom[j],
                stack.jaw_edge_neg[j],
                stack.jaw_edge_pos[j],
                stack.jaw_focused[j],
                from_origin,
            )
        return jaw_path_length(
            p_v,
            d_v,
            p_w,
            d_w,
            stack.z_top[j],
            stack.z_bottom[j],
            stack.jaw_edge_neg[j],
            stack.jaw_edge_pos[j],
            stack.jaw_focused[j],
            from_origin,
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
        stack.mlc_tips_off[j],
        stack.leaf_edges_v,
        stack.tips_neg,
        stack.tips_pos,
        stack.mlc_focused_sides[j],
        from_origin,
    )


@wp.kernel
def stack_path_lengths_kernel(
    stack: StackArrays,
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
    z: wp.array(dtype=float),
    ux: wp.array(dtype=float),
    uy: wp.array(dtype=float),
    uz: wp.array(dtype=float),
    from_origin: int,
    out: wp.array2d(dtype=float),
) -> None:
    """Per-device chord lengths for engine-frame rays; the device twin of ``path_lengths``.

    Maps each ray into the local frame (``rotation . (p - origin)``) then fills
    ``out[tid, j]`` for every device. A probe of the geometry path the pre-solve
    kernel walks; test-pinned against the host in ``tests/physics``.
    """
    tid = wp.tid()
    ox = stack.origin[0]
    oy = stack.origin[1]
    oz = stack.origin[2]
    rx = x[tid] - ox
    ry = y[tid] - oy
    rz = z[tid] - oz
    r0 = stack.rotation[0]
    r1 = stack.rotation[1]
    r2 = stack.rotation[2]
    r3 = stack.rotation[3]
    r4 = stack.rotation[4]
    r5 = stack.rotation[5]
    r6 = stack.rotation[6]
    r7 = stack.rotation[7]
    r8 = stack.rotation[8]
    p_u = r0 * rx + r1 * ry + r2 * rz
    p_v = r3 * rx + r4 * ry + r5 * rz
    p_w = r6 * rx + r7 * ry + r8 * rz
    d_u = r0 * ux[tid] + r1 * uy[tid] + r2 * uz[tid]
    d_v = r3 * ux[tid] + r4 * uy[tid] + r5 * uz[tid]
    d_w = r6 * ux[tid] + r7 * uy[tid] + r8 * uz[tid]
    for j in range(stack.n_devices):
        out[tid, j] = device_path_length(stack, j, p_u, p_v, p_w, d_u, d_v, d_w, from_origin)


@wp.struct
class MuArrays:
    """Macroscopic mu (1/cm) on a uniform log-energy grid, per stack/air material.

    The device twin of :class:`pyRadMC.geometry.head._MuTables`: ``log_total`` and
    ``log_compton`` hold the log of the density-scaled coefficient, row ``j`` for
    device ``j`` and the last row for the air column. Interpolated linearly in the
    log value then exponentiated (``_mu_lookup``), matching the host np.interp path
    so host and device evaluate the same physics.
    """

    log_e_min: float
    inv_dlog: float
    n_points: int
    log_total: wp.array2d(dtype=float)
    log_compton: wp.array2d(dtype=float)


@wp.struct
class ExitBuffer:
    """Append-only exit-plane particle columns filled by an atomic counter.

    ``count[0]`` is the running number of appended particles; a thread writes only
    while its index is below ``capacity`` (the caller sizes capacity to the exact
    worst case, so no particle is lost). Columns mirror the IAEA record the
    :class:`~pyRadMC.geometry.phasespace.InMemoryPhaseSpaceSource` consumes.
    """

    capacity: int
    count: wp.array(dtype=wp.int32)
    particle_type: wp.array(dtype=wp.int32)
    energy: wp.array(dtype=float)
    x: wp.array(dtype=float)
    y: wp.array(dtype=float)
    z: wp.array(dtype=float)
    ux: wp.array(dtype=float)
    uy: wp.array(dtype=float)
    uz: wp.array(dtype=float)
    weight: wp.array(dtype=float)


@wp.func
def _mu_lookup(
    table: wp.array2d(dtype=float),
    row: int,
    log_e_min: float,
    inv_dlog: float,
    n: int,
    energy: float,
) -> float:
    """Macroscopic mu at ``energy`` for ``row``: linear-in-log-value, clamped at edges."""
    t = (wp.log(energy) - log_e_min) * inv_dlog
    t = min(max(t, 0.0), float(n - 1))
    i = min(int(t), n - 2)
    frac = t - float(i)
    return wp.exp(table[row, i] * (1.0 - frac) + table[row, i + 1] * frac)


@wp.func
def _roulette_below(weight: float, floor: float, state: WarpRNGState) -> float:
    """Fair Russian roulette on a secondary below ``floor``; expected weight preserved."""
    if floor <= 0.0:
        return weight
    if weight <= 0.0:
        return weight
    if weight >= floor:
        return weight
    if uniform(state) < weight / floor:
        return floor
    return 0.0


@wp.func
def _emit(
    buf: ExitBuffer,
    stack: StackArrays,
    code: int,
    energy: float,
    x: float,
    y: float,
    z: float,
    ux: float,
    uy: float,
    uz: float,
    weight: float,
    exit_z: float,
    cutoff: float,
) -> int:
    """Slide a particle to the exit plane and append it if viable; returns 1 if kept.

    Kept when forward-going (local ``w`` direction positive), at or above ``cutoff``,
    and positive weight — the device form of
    :func:`pyRadMC.geometry.head._propagate_and_add`'s keep test.
    """
    kept = 0
    if weight > 0.0 and energy >= cutoff:
        r6 = stack.rotation[6]
        r7 = stack.rotation[7]
        r8 = stack.rotation[8]
        d_w = r6 * ux + r7 * uy + r8 * uz
        if d_w > 0.0:
            ox = stack.origin[0]
            oy = stack.origin[1]
            oz = stack.origin[2]
            p_w = r6 * (x - ox) + r7 * (y - oy) + r8 * (z - oz)
            t_exit = (exit_z - p_w) / d_w
            if t_exit >= 0.0:
                kept = 1
                idx = wp.atomic_add(buf.count, 0, 1)
                if idx < buf.capacity:
                    buf.particle_type[idx] = code
                    buf.energy[idx] = energy
                    buf.x[idx] = x + t_exit * ux
                    buf.y[idx] = y + t_exit * uy
                    buf.z[idx] = z + t_exit * uz
                    buf.ux[idx] = ux
                    buf.uy[idx] = uy
                    buf.uz[idx] = uz
                    buf.weight[idx] = weight
    return kept


@wp.kernel
def presolve_kernel(
    seed: int,
    slots: wp.array(dtype=wp.uint32),
    energy_in: wp.array(dtype=float),
    x_in: wp.array(dtype=float),
    y_in: wp.array(dtype=float),
    z_in: wp.array(dtype=float),
    ux_in: wp.array(dtype=float),
    uy_in: wp.array(dtype=float),
    uz_in: wp.array(dtype=float),
    w_in: wp.array(dtype=float),
    stack: StackArrays,
    mu: MuArrays,
    first_compton: int,
    has_air: int,
    air_row: int,
    air_z_start: float,
    air_z_end: float,
    exit_z: float,
    pcut: float,
    ecut: float,
    weight_floor: float,
    buf: ExitBuffer,
) -> None:
    """One thread per primary: stack + air forced first Compton, then the transmitted photon.

    The device twin of :func:`pyRadMC.geometry.head.presolve_head`'s body (see it for
    the physics and stated approximations). Each primary's own counter-based stream
    (``init_slot``) makes the result within-device reproducible; cross-target it agrees
    with the host only statistically (AGENTS.md 2.3).
    """
    tid = wp.tid()
    slots[tid] = init_slot(seed, tid)
    state = WarpRNGState()
    state.slots = slots
    state.idx = tid

    energy = energy_in[tid]
    w0 = w_in[tid]
    px = x_in[tid]
    py = y_in[tid]
    pz = z_in[tid]
    ux = ux_in[tid]
    uy = uy_in[tid]
    uz = uz_in[tid]

    ox = stack.origin[0]
    oy = stack.origin[1]
    oz = stack.origin[2]
    r0 = stack.rotation[0]
    r1 = stack.rotation[1]
    r2 = stack.rotation[2]
    r3 = stack.rotation[3]
    r4 = stack.rotation[4]
    r5 = stack.rotation[5]
    r6 = stack.rotation[6]
    r7 = stack.rotation[7]
    r8 = stack.rotation[8]
    rx = px - ox
    ry = py - oy
    rz = pz - oz
    p_u = r0 * rx + r1 * ry + r2 * rz
    p_v = r3 * rx + r4 * ry + r5 * rz
    p_w = r6 * rx + r7 * ry + r8 * rz
    d_u = r0 * ux + r1 * uy + r2 * uz
    d_v = r3 * ux + r4 * uy + r5 * uz
    d_w = r6 * ux + r7 * uy + r8 * uz

    # --- the beam-limiting stack ---------------------------------------------------
    tau_total = float(0.0)  # noqa: UP018 - Warp needs float() to declare a loop-mutated var
    for j in range(stack.n_devices):
        thickness = device_path_length(stack, j, p_u, p_v, p_w, d_u, d_v, d_w, 0)
        tau_total += (
            _mu_lookup(mu.log_total, j, mu.log_e_min, mu.inv_dlog, mu.n_points, energy) * thickness
        )
    transmitted_w = w0 * wp.exp(-tau_total)

    if first_compton == 1 and tau_total > 0.0:
        target = uniform(state) * tau_total
        cum = float(0.0)  # noqa: UP018 - Warp dynamic loop var (see above)
        jdev = int(0)  # noqa: UP018, RUF046 - Warp dynamic loop var
        found = int(0)  # noqa: UP018, RUF046 - Warp dynamic loop var
        for j in range(stack.n_devices):
            thickness = device_path_length(stack, j, p_u, p_v, p_w, d_u, d_v, d_w, 0)
            cum += (
                _mu_lookup(mu.log_total, j, mu.log_e_min, mu.inv_dlog, mu.n_points, energy)
                * thickness
            )
            if found == 0 and cum >= target:
                jdev = j
                found = 1
        mu_tot_j = _mu_lookup(mu.log_total, jdev, mu.log_e_min, mu.inv_dlog, mu.n_points, energy)
        mu_com_j = _mu_lookup(mu.log_compton, jdev, mu.log_e_min, mu.inv_dlog, mu.n_points, energy)
        compton_fraction = mu_com_j / mu_tot_j
        z_mid = 0.5 * (stack.z_top[jdev] + stack.z_bottom[jdev])
        t_mid = (z_mid - p_w) / d_w
        sx = px + t_mid * ux
        sy = py + t_mid * uy
        sz = pz + t_mid * uz
        ratio = sample_compton_energy_ratio(energy, state)
        e_out = ratio * energy
        cos_theta = compton_cos_theta(energy, ratio)
        phi = _TWO_PI * uniform(state)
        dox, doy, doz = rotate_direction(ux, uy, uz, cos_theta, phi)
        srx = sx - ox
        sry = sy - oy
        srz = sz - oz
        sp_u = r0 * srx + r1 * sry + r2 * srz
        sp_v = r3 * srx + r4 * sry + r5 * srz
        sp_w = r6 * srx + r7 * sry + r8 * srz
        do_u = r0 * dox + r1 * doy + r2 * doz
        do_v = r3 * dox + r4 * doy + r5 * doz
        do_w = r6 * dox + r7 * doy + r8 * doz
        tau_out = float(0.0)  # noqa: UP018 - Warp dynamic loop var (see above)
        for jj in range(stack.n_devices):
            thickness = device_path_length(stack, jj, sp_u, sp_v, sp_w, do_u, do_v, do_w, 1)
            tau_out += (
                _mu_lookup(mu.log_total, jj, mu.log_e_min, mu.inv_dlog, mu.n_points, e_out)
                * thickness
            )
        scatter_w = w0 * (1.0 - wp.exp(-tau_total)) * compton_fraction * wp.exp(-tau_out)
        scatter_w = _roulette_below(scatter_w, weight_floor, state)
        _emit(buf, stack, IAEA_PHOTON, e_out, sx, sy, sz, dox, doy, doz, scatter_w, exit_z, pcut)

    # --- the air column (primaries only) -------------------------------------------
    if has_air == 1:
        length = 0.0
        if d_w > 0.0:
            length = (air_z_end - air_z_start) / d_w
        mu_air = _mu_lookup(mu.log_total, air_row, mu.log_e_min, mu.inv_dlog, mu.n_points, energy)
        tau_air = mu_air * length
        pre_air_w = transmitted_w
        transmitted_w = transmitted_w * wp.exp(-tau_air)
        if first_compton == 1 and d_w > 0.0 and tau_air > 0.0:
            interact_w = pre_air_w * (1.0 - wp.exp(-tau_air))
            mu_com_air = _mu_lookup(
                mu.log_compton, air_row, mu.log_e_min, mu.inv_dlog, mu.n_points, energy
            )
            pair_w = interact_w * (mu_com_air / mu_air)
            u_depth = uniform(state)
            depth = -wp.log(1.0 - u_depth * (1.0 - wp.exp(-tau_air))) / mu_air
            t_start = (air_z_start - p_w) / d_w
            ax = px + (t_start + depth) * ux
            ay = py + (t_start + depth) * uy
            az = pz + (t_start + depth) * uz
            ratio = sample_compton_energy_ratio(energy, state)
            e_photon = ratio * energy
            e_electron = energy - e_photon
            phi = _TWO_PI * uniform(state)
            cos_theta = compton_cos_theta(energy, ratio)
            dpx, dpy, dpz = rotate_direction(ux, uy, uz, cos_theta, phi)
            cos_e = compton_electron_cos_theta(energy, ratio)
            dex, dey, dez = rotate_direction(ux, uy, uz, cos_e, phi + _PI)
            _emit(
                buf,
                stack,
                IAEA_PHOTON,
                e_photon,
                ax,
                ay,
                az,
                dpx,
                dpy,
                dpz,
                _roulette_below(pair_w, weight_floor, state),
                exit_z,
                pcut,
            )
            _emit(
                buf,
                stack,
                IAEA_ELECTRON,
                e_electron,
                ax,
                ay,
                az,
                dex,
                dey,
                dez,
                _roulette_below(pair_w, weight_floor, state),
                exit_z,
                ecut,
            )

    _emit(buf, stack, IAEA_PHOTON, energy, px, py, pz, ux, uy, uz, transmitted_w, exit_z, pcut)


@wp.kernel
def _scale_weights(weight: wp.array(dtype=float), scale: float) -> None:
    """Multiply each stored weight by the per-primary normalization factor in place."""
    weight[wp.tid()] *= scale


def _alloc_exit_buffer(capacity: int, device: str) -> ExitBuffer:
    """Allocate the append-only exit buffer on ``device``."""
    buf = ExitBuffer()
    buf.capacity = capacity
    buf.count = wp.zeros(1, dtype=wp.int32, device=device)
    buf.particle_type = wp.zeros(capacity, dtype=wp.int32, device=device)
    for field in ("energy", "x", "y", "z", "ux", "uy", "uz", "weight"):
        setattr(buf, field, wp.zeros(capacity, dtype=float, device=device))
    return buf


def _upload_mu(tables: _MuTables, device: str) -> MuArrays:
    """Upload the host log-mu tables (density-scaled) to ``device``."""
    log_energies = tables._log_energies  # uniform log grid (geomspace on the host)
    mu = MuArrays()
    mu.log_e_min = float(log_energies[0])
    mu.n_points = int(log_energies.size)
    mu.inv_dlog = (mu.n_points - 1) / (float(log_energies[-1]) - float(log_energies[0]))
    mu.log_total = wp.array(tables._log_total.astype(np.float32), dtype=float, device=device)
    mu.log_compton = wp.array(tables._log_compton.astype(np.float32), dtype=float, device=device)
    return mu


def presolve_head_device(
    source: Source,
    *,
    cross_sections,
    n_histories: int,
    seed: int,
    exit_z: float,
    stack: BeamLimitingStack,
    air: AirColumn | None = None,
    mode: str = "auto",
    device: str = "cuda:0",
    pcut: float = PCUT_MEV,
    ecut: float = ECUT_MEV,
    weight_floor: float = 1.0e-3,
    e_min: float = 0.010,
    n_table: int = 512,
    return_buffer: bool = False,
    return_device_source: bool = False,
):
    """Pre-solve the head on ``device`` and return the exit-plane phase space.

    The Warp twin of :func:`pyRadMC.geometry.head.presolve_head`: the primary
    sampling stays on the host (``source.sample_batch``, cheap and reused), while
    the head physics — stack and air-column tracing, the forced first Compton, the
    exit-plane scoring — runs as one kernel launch, one thread per primary. A
    beam-limiting ``stack`` is required (the stack-free passthrough has nothing to
    accelerate; use the host entry point).

    Return modes (the physics is identical; only the handoff differs):

    - default — the exit particles are copied back and wrapped in an
      :class:`~pyRadMC.geometry.phasespace.InMemoryPhaseSpaceSource`, exactly as the
      host does.
    - ``return_device_source=True`` — a :class:`DevicePhaseSpace` keeping the
      population on the device; ``WarpEngine.run`` transports it with no copy back or
      re-upload (the high-performance path). Only the ``count`` scalar is read back.
    - ``return_buffer=True`` — additionally returns the raw on-device
      :class:`ExitBuffer` and its filled count, for introspection.
    """
    if stack is None:
        raise ValueError("the device pre-solve requires a stack; use presolve_head on the host")
    resolved, _frame = resolve_presolve(stack, None, air, exit_z, mode)

    columns = source.sample_batch(seed, 0, n_histories)
    if not np.all(columns["particle_type"] == IAEA_PHOTON):
        raise ValueError("the pre-solve transports photon sources only")

    pairs: tuple[tuple[int, float], ...] = tuple(zip(stack.materials, stack.densities, strict=True))
    air_row = len(pairs)
    if air is not None:
        pairs = (*pairs, (air.material, air.density))
    mu_tables = _MuTables(cross_sections, pairs, e_min, source.max_energy, n_table)

    stack_dev = upload_stack(stack.flatten(), device)
    mu_dev = _upload_mu(mu_tables, device)

    def farr(name: str) -> wp.array:
        return wp.array(columns[name].astype(np.float32), dtype=float, device=device)

    first_compton = 1 if resolved == "first_compton" else 0
    has_air = 1 if air is not None else 0
    per_primary = 1 + first_compton + 2 * (first_compton * has_air)
    capacity = int(n_histories) * per_primary
    buf = _alloc_exit_buffer(capacity, device)
    slots = wp.zeros(n_histories, dtype=wp.uint32, device=device)

    wp.launch(
        presolve_kernel,
        dim=n_histories,
        inputs=[
            seed,
            slots,
            farr("energy"),
            farr("x"),
            farr("y"),
            farr("z"),
            farr("ux"),
            farr("uy"),
            farr("uz"),
            farr("weight"),
            stack_dev,
            mu_dev,
            first_compton,
            has_air,
            air_row,
            float(air.z_start) if air is not None else 0.0,
            float(air.z_end) if air is not None else 0.0,
            float(exit_z),
            float(pcut),
            float(ecut),
            float(weight_floor),
            buf,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    count = int(buf.count.numpy()[0])
    if count > capacity:
        raise RuntimeError(f"exit buffer overflow: {count} > capacity {capacity}")
    if count == 0:
        raise ValueError("the pre-solve kept no particles; check the geometry and cutoffs")

    # Normalize per original primary (the N/n factor of _Population.build) on the
    # device, so the exposed buffer and the copied-back phase space agree — the
    # no-copy transport path (DevicePhaseSpace) consumes buf.weight[:count] directly.
    wp.launch(_scale_weights, dim=count, inputs=[buf.weight, count / n_histories], device=device)
    wp.synchronize_device(device)

    device_ps = DevicePhaseSpace(buf, count, device, float(source.max_energy))
    if return_device_source:
        return device_ps
    phase_space = device_ps.to_phase_space()
    if return_buffer:
        return phase_space, buf, count
    return phase_space


def _buffer_to_phase_space(buf: ExitBuffer, count: int) -> InMemoryPhaseSpaceSource:
    """Copy the first ``count`` exit records back to the host phase space."""
    kept = slice(0, count)
    directions = np.stack(
        [buf.ux.numpy()[kept], buf.uy.numpy()[kept], buf.uz.numpy()[kept]], axis=1
    ).astype(np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)  # float32 round-off
    return InMemoryPhaseSpaceSource(
        particle_type=buf.particle_type.numpy()[kept].astype(np.int32),
        energy=buf.energy.numpy()[kept].astype(np.float64),
        x=buf.x.numpy()[kept].astype(np.float64),
        y=buf.y.numpy()[kept].astype(np.float64),
        z=buf.z.numpy()[kept].astype(np.float64),
        ux=directions[:, 0],
        uy=directions[:, 1],
        uz=directions[:, 2],
        weight=buf.weight.numpy()[kept].astype(np.float64),
    )


class DevicePhaseSpace:
    """A pre-solve exit population left on the device for no-copy transport.

    Returned by :func:`presolve_head_device` with ``return_device_source=True``.
    ``WarpEngine.run`` transports it directly — sampling records from ``buffer``
    on the device and seeding the transport queues, skipping the copy back to host
    and the re-upload the :class:`~pyRadMC.geometry.phasespace.InMemoryPhaseSpaceSource`
    path incurs. Weights are already normalized per original primary. It is a
    device handle, not a :class:`~pyRadMC.geometry.source.Source`: ``warp_sampler``
    is ``None`` and it carries no host ``emit``/``sample_batch``; ``to_phase_space``
    materializes the host source when the copy is actually wanted.
    """

    warp_sampler = None

    def __init__(self, buffer: ExitBuffer, count: int, device: str, max_energy: float) -> None:
        self.buffer = buffer
        self.count = count
        self.device = device
        self._max_energy = max_energy

    @property
    def max_energy(self) -> float:
        """Highest energy in the stored population (MeV), for the table upload."""
        return self._max_energy

    def __len__(self) -> int:
        """Return the number of stored exit-plane particles."""
        return self.count

    def to_phase_space(self) -> InMemoryPhaseSpaceSource:
        """Copy the population back to a host :class:`InMemoryPhaseSpaceSource`."""
        return _buffer_to_phase_space(self.buffer, self.count)
