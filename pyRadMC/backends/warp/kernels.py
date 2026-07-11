"""Transport kernels: the compiled mirror of the reference step loops.

Kernel bodies sequence the same single-source physics functions as
:mod:`pyRadMC.transport.photon` and :mod:`pyRadMC.transport.electron` — the
behavioural specification — with three structural substitutions, and nothing
physical decided here (AGENTS.md section 3):

data access
    Table lookups (:mod:`pyRadMC.data.tables`) instead of host method calls.
secondaries
    Persistent queues between separate photon and electron kernels instead of a
    per-history stack (the divergence mitigation the plan schedules first). Each
    queued particle carries its own RNG stream, derived from its parent's via
    ``spawn_stream``; combined with fixed-point scoring, the result is
    bit-reproducible on one device no matter how threads are scheduled.
scoring
    ``atomic_add`` of int64 fixed-point quanta (:data:`ENERGY_QUANTUM_MEV`).
    Integer addition is associative, so the accumulated maps are independent of
    thread interleaving — this is what makes the within-device reproducibility of
    AGENTS.md 2.3 *enforceable* on CUDA, where float atomics are not.

Numerical deviations from the reference loop, both documented where used:

- Transport arithmetic is float32; statistical equivalence with the float64
  reference is asserted by the chi-squared oracle, never bit equality (AGENTS.md
  2.3).
- The entry nudge is 1e-4 cm instead of the reference's 1e-9 cm: a nudge below
  float32 resolution at centimetre coordinates would be lost in rounding and
  photons entering exactly on the surface would be dropped. 1 micrometre remains
  far below any voxel size or mean free path this engine sees.
- The majorant sanity check cannot raise from a kernel; it increments a violation
  counter that the engine turns into the same ``RuntimeError`` after the batch.

This module is Warp kernel code: exempt from ``mypy --strict`` (AGENTS.md 5).
"""

from __future__ import annotations

import math

import warp as wp

from pyRadMC import ELECTRON_MASS_MEV
from pyRadMC.backends.warp.physics import warp_physics
from pyRadMC.data.interface import PhotonProcess
from pyRadMC.rng.warp_shim import WarpRNGState, init_slot, spawn_stream, uniform
from pyRadMC.transport.electron import STEP_ENERGY_FRACTION, STEP_VOXEL_FRACTION
from pyRadMC.transport.particles import ELECTRON, PHOTON, POSITRON

__all__ = [
    "ENERGY_QUANTUM_MEV",
    "GridInfo",
    "Queue",
    "Tables",
    "electron_kernel",
    "generate_parallel_beam",
    "generate_pencil_beam",
    "photon_kernel",
]

ENERGY_QUANTUM_MEV: float = 1.0e-9
"""Fixed-point scoring quantum. Deposits are rounded to the nearest quantum in
float64, so the per-deposit error is at most 5e-10 MeV — orders below the float32
transport arithmetic. int64 headroom: ~9e18 quanta = 9e9 MeV per voxel per batch."""

_INV_QUANTUM = 1.0 / ENERGY_QUANTUM_MEV

_ENTRY_NUDGE_CM = 1.0e-4
"""float32 entry nudge; see the module docstring for why it exceeds the reference's."""

_MAJORANT_TOLERANCE = 1.0e-9  # same relative headroom as the reference loop

_INFINITY_CM = 1.0e30
"""Sentinel threshold for a missed grid: slab_entry_distance returns +inf, and any
comparison against this finite bound classifies it without needing isinf in-kernel."""

_p = warp_physics()

# Bound at module level so the kernel bodies below read like the reference loops.
sample_path_length = _p.sample_path_length
select_photon_process = _p.select_photon_process
sample_compton_energy_ratio = _p.sample_compton_energy_ratio
compton_cos_theta = _p.compton_cos_theta
compton_electron_cos_theta = _p.compton_electron_cos_theta
sample_rayleigh_cos_theta = _p.sample_rayleigh_cos_theta
rotate_direction = _p.rotate_direction
sample_isotropic_direction = _p.sample_isotropic_direction
sample_moller_delta_energy = _p.sample_moller_delta_energy
moller_direction_cosines = _p.moller_direction_cosines
sample_hinge_cos_theta = _p.sample_hinge_cos_theta
bremsstrahlung_step_parameters = _p.bremsstrahlung_step_parameters
sample_bremsstrahlung_energy = _p.sample_bremsstrahlung_energy
lookup_1d = _p.lookup_loglinear_1d
lookup_2d = _p.lookup_loglinear_2d
point_inside = _p.point_inside
point_axis_index = _p.point_axis_index
slab_entry_distance = _p.slab_entry_distance


@wp.struct
class GridInfo:
    """Scalar grid metadata; the density/material arrays travel separately."""

    x_lo: float
    y_lo: float
    z_lo: float
    x_hi: float
    y_hi: float
    z_hi: float
    sx: float
    sy: float
    sz: float
    ny: int
    nz: int
    min_spacing: float


@wp.struct
class Tables:
    """Device copies of :class:`pyRadMC.data.tables.CrossSectionTables` (float32)."""

    mu_compton: wp.array2d(dtype=float)
    mu_photo: wp.array2d(dtype=float)
    mu_pair: wp.array2d(dtype=float)
    mu_rayleigh: wp.array2d(dtype=float)
    majorant: wp.array(dtype=float)
    stopping_restricted: wp.array2d(dtype=float)
    stopping_radiative: wp.array2d(dtype=float)
    moller: wp.array2d(dtype=float)
    csda_range: wp.array2d(dtype=float)
    scattering_power: wp.array2d(dtype=float)
    p_log_e_min: float
    p_inv_dlog: float
    e_log_e_min: float
    e_inv_dlog: float
    n_points: int


@wp.struct
class Queue:
    """Structure-of-arrays particle queue with an atomic append counter.

    Overflow does not write (the slot does not exist) but the counter keeps
    counting, so the engine detects ``count > capacity`` after the launch and
    raises — secondaries are never silently dropped.
    """

    kind: wp.array(dtype=wp.int32)
    energy: wp.array(dtype=float)
    x: wp.array(dtype=float)
    y: wp.array(dtype=float)
    z: wp.array(dtype=float)
    ux: wp.array(dtype=float)
    uy: wp.array(dtype=float)
    uz: wp.array(dtype=float)
    rng: wp.array(dtype=wp.uint32)
    count: wp.array(dtype=wp.int32)
    capacity: int


@wp.func
def _queue_push(
    q: Queue,
    kind: int,
    energy: float,
    x: float,
    y: float,
    z: float,
    ux: float,
    uy: float,
    uz: float,
    rng_state: wp.uint32,
):
    idx = wp.atomic_add(q.count, 0, 1)
    if idx < q.capacity:
        q.kind[idx] = kind
        q.energy[idx] = energy
        q.x[idx] = x
        q.y[idx] = y
        q.z[idx] = z
        q.ux[idx] = ux
        q.uy[idx] = uy
        q.uz[idx] = uz
        q.rng[idx] = rng_state


@wp.func
def _deposit(
    edep: wp.array(dtype=wp.int64), gi: GridInfo, ix: int, iy: int, iz: int, energy: float
):
    """Round-to-nearest fixed-point deposit; float64 scaling keeps it exact."""
    quanta = wp.int64(wp.float64(energy) * wp.float64(_INV_QUANTUM) + wp.float64(0.5))
    wp.atomic_add(edep, (ix * gi.ny + iy) * gi.nz + iz, quanta)


@wp.func
def _escape(escaped: wp.array(dtype=wp.int64), energy: float):
    quanta = wp.int64(wp.float64(energy) * wp.float64(_INV_QUANTUM) + wp.float64(0.5))
    wp.atomic_add(escaped, 0, quanta)


@wp.func
def _deposit_or_escape(
    edep: wp.array(dtype=wp.int64),
    escaped: wp.array(dtype=wp.int64),
    gi: GridInfo,
    x: float,
    y: float,
    z: float,
    amount: float,
):
    """Mirror of the reference electron loop's midpoint deposit helper."""
    if amount <= 0.0:
        return
    if point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
        ix = point_axis_index(x, gi.x_lo, gi.sx)
        iy = point_axis_index(y, gi.y_lo, gi.sy)
        iz = point_axis_index(z, gi.z_lo, gi.sz)
        _deposit(edep, gi, ix, iy, iz, amount)
    else:
        _escape(escaped, amount)


@wp.func
def _annihilate_at_rest(
    x: float,
    y: float,
    z: float,
    gi: GridInfo,
    state: WarpRNGState,
    q_photon: Queue,
    edep: wp.array(dtype=wp.int64),
    pcut: float,
):
    """Positron annihilation at rest; mirror of transport.particles.annihilate_at_rest.

    Called only for positions inside the grid. Each annihilation photon gets its
    own child stream, since the pair are transported by independent threads.
    """
    if pcut >= ELECTRON_MASS_MEV:
        ix = point_axis_index(x, gi.x_lo, gi.sx)
        iy = point_axis_index(y, gi.y_lo, gi.sy)
        iz = point_axis_index(z, gi.z_lo, gi.sz)
        _deposit(edep, gi, ix, iy, iz, 2.0 * ELECTRON_MASS_MEV)
        return
    ax, ay, az = sample_isotropic_direction(state)
    _queue_push(q_photon, PHOTON, ELECTRON_MASS_MEV, x, y, z, ax, ay, az, spawn_stream(state))
    _queue_push(q_photon, PHOTON, ELECTRON_MASS_MEV, x, y, z, -ax, -ay, -az, spawn_stream(state))


@wp.kernel
def photon_kernel(
    gi: GridInfo,
    density: wp.array3d(dtype=float),
    material: wp.array3d(dtype=wp.int32),
    tab: Tables,
    q_in: Queue,
    q_electron: Queue,
    q_photon_out: Queue,
    slots: wp.array(dtype=wp.uint32),
    edep: wp.array(dtype=wp.int64),
    escaped: wp.array(dtype=wp.int64),
    pcut: float,
    ecut: float,
    transport_electrons: int,
    majorant_violations: wp.array(dtype=wp.int32),
):
    """One thread transports one photon to termination; Woodcock tracking.

    Sequencing mirrors ``transport.photon.photon_steps`` statement for statement;
    consult that loop for the physics commentary.
    """
    tid = wp.tid()
    e = q_in.energy[tid]
    x = q_in.x[tid]
    y = q_in.y[tid]
    z = q_in.z[tid]
    ux = q_in.ux[tid]
    uy = q_in.uy[tid]
    uz = q_in.uz[tid]
    slots[tid] = q_in.rng[tid]
    state = WarpRNGState()
    state.slots = slots
    state.idx = tid

    if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
        t = slab_entry_distance(
            x, y, z, ux, uy, uz, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi
        )
        if t >= _INFINITY_CM:
            _escape(escaped, e)
            return
        t += _ENTRY_NUDGE_CM
        x += t * ux
        y += t * uy
        z += t * uz
        if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
            _escape(escaped, e)
            return

    while True:
        mu_majorant = lookup_1d(tab.majorant, tab.p_log_e_min, tab.p_inv_dlog, tab.n_points, e)
        step = sample_path_length(mu_majorant, state)
        x += step * ux
        y += step * uy
        z += step * uz
        if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
            _escape(escaped, e)
            return

        ix = point_axis_index(x, gi.x_lo, gi.sx)
        iy = point_axis_index(y, gi.y_lo, gi.sy)
        iz = point_axis_index(z, gi.z_lo, gi.sz)
        rho = density[ix, iy, iz]
        mat = material[ix, iy, iz]
        mu_compton = rho * lookup_2d(
            tab.mu_compton, mat, tab.p_log_e_min, tab.p_inv_dlog, tab.n_points, e
        )
        mu_photo = rho * lookup_2d(
            tab.mu_photo, mat, tab.p_log_e_min, tab.p_inv_dlog, tab.n_points, e
        )
        mu_pair = rho * lookup_2d(
            tab.mu_pair, mat, tab.p_log_e_min, tab.p_inv_dlog, tab.n_points, e
        )
        mu_rayleigh = rho * lookup_2d(
            tab.mu_rayleigh, mat, tab.p_log_e_min, tab.p_inv_dlog, tab.n_points, e
        )
        mu_real = mu_compton + mu_photo + mu_pair + mu_rayleigh
        if mu_real > mu_majorant * (1.0 + _MAJORANT_TOLERANCE):
            wp.atomic_add(majorant_violations, 0, 1)
            _escape(escaped, e)
            return

        if uniform(state) * mu_majorant >= mu_real:
            continue  # delta (fictitious) scattering

        process = select_photon_process(mu_compton, mu_photo, mu_pair, mu_rayleigh, state)

        if process == PhotonProcess.RAYLEIGH:
            cos_coherent = sample_rayleigh_cos_theta(state)
            phi = 2.0 * math.pi * uniform(state)
            ux, uy, uz = rotate_direction(ux, uy, uz, cos_coherent, phi)
        elif process == PhotonProcess.COMPTON:
            ratio = sample_compton_energy_ratio(e, state)
            recoil = e * (1.0 - ratio)
            phi = 2.0 * math.pi * uniform(state)
            if transport_electrons != 0 and recoil > ecut:
                cos_electron = compton_electron_cos_theta(e, ratio)
                ex, ey, ez = rotate_direction(ux, uy, uz, cos_electron, phi + math.pi)
                _queue_push(q_electron, ELECTRON, recoil, x, y, z, ex, ey, ez, spawn_stream(state))
            else:
                _deposit(edep, gi, ix, iy, iz, recoil)
            cos_gamma = compton_cos_theta(e, ratio)
            ux, uy, uz = rotate_direction(ux, uy, uz, cos_gamma, phi)
            e *= ratio
            if e <= pcut:
                _deposit(edep, gi, ix, iy, iz, e)
                return
        elif process == PhotonProcess.PHOTOELECTRIC:
            if transport_electrons != 0 and e > ecut:
                _queue_push(q_electron, ELECTRON, e, x, y, z, ux, uy, uz, spawn_stream(state))
            else:
                _deposit(edep, gi, ix, iy, iz, e)
            return
        else:  # pair production
            kinetic = e - 2.0 * ELECTRON_MASS_MEV
            if transport_electrons != 0:
                fraction = uniform(state)
                share_e = fraction * kinetic
                share_p = (1.0 - fraction) * kinetic
                if share_e > ecut:
                    _queue_push(
                        q_electron, ELECTRON, share_e, x, y, z, ux, uy, uz, spawn_stream(state)
                    )
                else:
                    _deposit(edep, gi, ix, iy, iz, share_e)
                if share_p > ecut:
                    _queue_push(
                        q_electron, POSITRON, share_p, x, y, z, ux, uy, uz, spawn_stream(state)
                    )
                else:
                    _deposit(edep, gi, ix, iy, iz, share_p)
                    _annihilate_at_rest(x, y, z, gi, state, q_photon_out, edep, pcut)
            else:
                _deposit(edep, gi, ix, iy, iz, kinetic)
                _annihilate_at_rest(x, y, z, gi, state, q_photon_out, edep, pcut)
            return


@wp.kernel
def electron_kernel(
    gi: GridInfo,
    density: wp.array3d(dtype=float),
    material: wp.array3d(dtype=wp.int32),
    tab: Tables,
    q_in: Queue,
    q_photon: Queue,
    q_electron_out: Queue,
    slots: wp.array(dtype=wp.uint32),
    edep: wp.array(dtype=wp.int64),
    escaped: wp.array(dtype=wp.int64),
    pcut: float,
    ecut: float,
):
    """One thread transports one electron or positron; Class II condensed history.

    Sequencing mirrors ``transport.electron.electron_steps`` statement for
    statement; delta rays go to the electron out-queue instead of a stack, and
    bremsstrahlung/annihilation photons to the photon queue.
    """
    tid = wp.tid()
    is_positron = q_in.kind[tid] == POSITRON
    e = q_in.energy[tid]
    x = q_in.x[tid]
    y = q_in.y[tid]
    z = q_in.z[tid]
    ux = q_in.ux[tid]
    uy = q_in.uy[tid]
    uz = q_in.uz[tid]
    slots[tid] = q_in.rng[tid]
    state = WarpRNGState()
    state.slots = slots
    state.idx = tid

    latent = 0.0
    if is_positron:
        latent = 2.0 * ELECTRON_MASS_MEV

    if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
        t = slab_entry_distance(
            x, y, z, ux, uy, uz, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi
        )
        if t >= _INFINITY_CM:
            _escape(escaped, e + latent)
            return
        t += _ENTRY_NUDGE_CM
        x += t * ux
        y += t * uy
        z += t * uz
        if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
            _escape(escaped, e + latent)
            return

    while True:
        if e <= ecut:
            ix = point_axis_index(x, gi.x_lo, gi.sx)
            iy = point_axis_index(y, gi.y_lo, gi.sy)
            iz = point_axis_index(z, gi.z_lo, gi.sz)
            _deposit(edep, gi, ix, iy, iz, e)
            if is_positron:
                _annihilate_at_rest(x, y, z, gi, state, q_photon, edep, pcut)
            return

        ix = point_axis_index(x, gi.x_lo, gi.sx)
        iy = point_axis_index(y, gi.y_lo, gi.sy)
        iz = point_axis_index(z, gi.z_lo, gi.sz)
        rho = density[ix, iy, iz]
        mat = material[ix, iy, iz]

        # --- substep length ----------------------------------------------------
        range_cm = (
            lookup_2d(tab.csda_range, mat, tab.e_log_e_min, tab.e_inv_dlog, tab.n_points, e) / rho
        )
        s_max = min(STEP_ENERGY_FRACTION * range_cm, STEP_VOXEL_FRACTION * gi.min_spacing)
        sigma_moller = rho * lookup_2d(
            tab.moller, mat, tab.e_log_e_min, tab.e_inv_dlog, tab.n_points, e
        )
        s = s_max
        moller_pending = False
        if sigma_moller > 0.0:
            s_interaction = sample_path_length(sigma_moller, state)
            if s_interaction <= s_max:
                s = s_interaction
                moller_pending = True

        # --- continuous loss over the substep -----------------------------------
        de = (
            lookup_2d(
                tab.stopping_restricted, mat, tab.e_log_e_min, tab.e_inv_dlog, tab.n_points, e
            )
            * rho
            * s
        )
        if de >= e - ecut:
            s *= (e - ecut) / de
            de = e - ecut
            moller_pending = False
        emit_probability, local_brems = bremsstrahlung_step_parameters(
            e,
            lookup_2d(
                tab.stopping_radiative, mat, tab.e_log_e_min, tab.e_inv_dlog, tab.n_points, e
            ),
            rho,
            s,
            pcut,
        )
        continuous = de + local_brems
        emit_brems = uniform(state) < emit_probability

        # --- random hinge: move, deflect, move ----------------------------------
        s1 = uniform(state) * s
        s2 = s - s1
        d1 = 0.0
        if s > 0.0:
            d1 = continuous * (s1 / s)
        d2 = continuous - d1

        _deposit_or_escape(
            edep, escaped, gi, x + ux * s1 / 2.0, y + uy * s1 / 2.0, z + uz * s1 / 2.0, d1
        )
        e -= d1
        x += ux * s1
        y += uy * s1
        z += uz * s1
        if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
            _escape(escaped, e + latent)
            return

        mean_square = (
            lookup_2d(tab.scattering_power, mat, tab.e_log_e_min, tab.e_inv_dlog, tab.n_points, e)
            * rho
            * s
        )
        cos_hinge = sample_hinge_cos_theta(mean_square, state)
        phi = 2.0 * math.pi * uniform(state)
        ux, uy, uz = rotate_direction(ux, uy, uz, cos_hinge, phi)

        if emit_brems and e > pcut:
            k = sample_bremsstrahlung_energy(e, pcut, state)
            k = min(k, e - d2)
            if k > pcut:
                _queue_push(q_photon, PHOTON, k, x, y, z, ux, uy, uz, spawn_stream(state))
            else:
                _deposit_or_escape(edep, escaped, gi, x, y, z, k)
            e -= k

        _deposit_or_escape(
            edep, escaped, gi, x + ux * s2 / 2.0, y + uy * s2 / 2.0, z + uz * s2 / 2.0, d2
        )
        e -= d2
        x += ux * s2
        y += uy * s2
        z += uz * s2
        if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
            _escape(escaped, e + latent)
            return

        # --- discrete Moller event at the end of the substep ---------------------
        if moller_pending and e > 2.0 * ecut:
            delta_energy = sample_moller_delta_energy(e, ecut, state)
            cos_delta, cos_primary = moller_direction_cosines(e, delta_energy)
            phi = 2.0 * math.pi * uniform(state)
            dx, dy, dz = rotate_direction(ux, uy, uz, cos_delta, phi)
            _queue_push(
                q_electron_out, ELECTRON, delta_energy, x, y, z, dx, dy, dz, spawn_stream(state)
            )
            ux, uy, uz = rotate_direction(ux, uy, uz, cos_primary, phi + math.pi)
            e -= delta_energy


@wp.kernel
def generate_pencil_beam(
    seed: int,
    history_offset: int,
    kind: int,
    energy: float,
    px: float,
    py: float,
    pz: float,
    dx: float,
    dy: float,
    dz: float,
    q: Queue,
):
    """Write primaries at fixed slots (no atomics: layout is deterministic).

    The engine presets the queue counter; ``emit`` consumes no draws, matching
    the host source.
    """
    tid = wp.tid()
    q.kind[tid] = kind
    q.energy[tid] = energy
    q.x[tid] = px
    q.y[tid] = py
    q.z[tid] = pz
    q.ux[tid] = dx
    q.uy[tid] = dy
    q.uz[tid] = dz
    q.rng[tid] = init_slot(seed, history_offset + tid)


@wp.kernel
def generate_parallel_beam(
    seed: int,
    history_offset: int,
    kind: int,
    energy: float,
    z0: float,
    x0: float,
    x_extent: float,
    y0: float,
    y_extent: float,
    slots: wp.array(dtype=wp.uint32),
    q: Queue,
):
    """Uniform rectangular field along +z; two uniforms per primary, like the host."""
    tid = wp.tid()
    slots[tid] = init_slot(seed, history_offset + tid)
    state = WarpRNGState()
    state.slots = slots
    state.idx = tid
    x = x0 + x_extent * uniform(state)
    y = y0 + y_extent * uniform(state)
    q.kind[tid] = kind
    q.energy[tid] = energy
    q.x[tid] = x
    q.y[tid] = y
    q.z[tid] = z0
    q.ux[tid] = 0.0
    q.uy[tid] = 0.0
    q.uz[tid] = 1.0
    q.rng[tid] = slots[tid]
