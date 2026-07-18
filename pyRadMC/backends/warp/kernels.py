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

Numerical deviations from the reference loop, each documented where used:

- Transport arithmetic is float32; statistical equivalence with the float64
  reference is asserted by the chi-squared oracle, never bit equality (AGENTS.md
  2.3).
- The module compiles with ``fast_math`` (approximate transcendentals and
  division; see the ``set_module_options`` site). Within-device bit
  reproducibility is unaffected — one fixed binary — and the water dose-to-water
  factor returns an exact 1.0 without dividing, preserving that pinned identity.
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

from pyRadMC import (
    ELECTRON_MASS_MEV,
    PHOTON_ROULETTE_MEV,
    PHOTON_ROULETTE_SURVIVAL,
    PHOTON_ROULETTE_WEIGHT_CAP,
    PHOTON_SPLIT_N,
)
from pyRadMC.backends.warp.physics import warp_physics
from pyRadMC.data.interface import PhotonProcess
from pyRadMC.data.materials import WATER
from pyRadMC.rng.warp_shim import WarpRNGState, init_slot, spawn_stream, uniform
from pyRadMC.transport.electron import (
    BOUNDARY_NUDGE_CM,
    STEP_ENERGY_FRACTION,
    STEP_VOXEL_FRACTION,
)
from pyRadMC.transport.particles import ELECTRON, PHOTON, POSITRON

__all__ = [
    "ENERGY_QUANTUM_MEV",
    "GridInfo",
    "Queue",
    "Tables",
    "accumulate_dij_batch",
    "accumulate_run_batch",
    "compact_column",
    "count_kept_per_column",
    "electron_kernel",
    "fill_int32",
    "fill_int64",
    "finalize_run",
    "generate_beamlet_lattice",
    "generate_from_exit_buffer",
    "generate_from_upload",
    "generate_parallel_beam",
    "generate_pencil_beam",
    "make_beamlet_generator_kernel",
    "make_generator_kernel",
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

# This engine never differentiates: skip warp's default adjoint (backward) kernel
# compilation, which costs ~3x on the cold module compile and buys nothing here.
#
# fast_math: approximate transcendentals/division on the transport kernels —
# measured 14-23% wall time on the latency-bound transport (2026-07-18). This is
# a *documented numerical deviation* in the sense of the module docstring: results
# stay bit-reproducible within one device (one fixed binary; AGENTS.md 2.3), the
# cross-target claim was always statistical-only, and the chi-squared and
# energy-balance suites gate the equivalence. The pre-solve module must NOT take
# this option: its attenuation mode is test-pinned float32-exact against the host.
wp.set_module_options({"enable_backward": False, "fast_math": True})

_p = warp_physics()

# Bound at module level so the kernel bodies below read like the reference loops.
sample_path_length = _p.sample_path_length
select_photon_process = _p.select_photon_process
sample_compton_energy_ratio = _p.sample_compton_energy_ratio
compton_cos_theta = _p.compton_cos_theta
compton_electron_cos_theta = _p.compton_electron_cos_theta
sample_rayleigh_cos_theta = _p.sample_rayleigh_cos_theta
sample_coherent_cos_theta_form_factor = _p.sample_coherent_cos_theta_form_factor
rotate_direction = _p.rotate_direction
sample_isotropic_direction = _p.sample_isotropic_direction
sample_moller_delta_energy = _p.sample_moller_delta_energy
moller_direction_cosines = _p.moller_direction_cosines
sample_hinge_cos_theta = _p.sample_hinge_cos_theta
bremsstrahlung_step_parameters = _p.bremsstrahlung_step_parameters
sample_bremsstrahlung_energy = _p.sample_bremsstrahlung_energy
roulette_weight = _p.roulette_weight
lookup_1d = _p.lookup_loglinear_1d
lookup_2d = _p.lookup_loglinear_2d
point_inside = _p.point_inside
point_axis_index = _p.point_axis_index
slab_entry_distance = _p.slab_entry_distance
distance_to_voxel_boundary = _p.distance_to_voxel_boundary


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
    nx: int
    ny: int
    nz: int
    n_voxels: int
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
    coherent_x: wp.array(dtype=float)
    coherent_cumulative: wp.array2d(dtype=float)
    p_log_e_min: float
    p_inv_dlog: float
    e_log_e_min: float
    e_inv_dlog: float
    n_points: int
    n_coherent: int


@wp.struct
class Queue:
    """Structure-of-arrays particle queue with an atomic append counter.

    Overflow does not write (the slot does not exist) but the counter keeps
    counting, so the engine detects ``count > capacity`` after the launch and
    raises — secondaries are never silently dropped.

    ``beamlet`` is the Dij tag: the group-local index of the beamlet whose
    primary this particle descends from. Secondaries inherit it unchanged, so a
    history's whole family scores into one Dij column. Plain dose runs leave it
    at zero everywhere, which makes them the one-column degenerate case.

    ``primary`` is 1 for a source photon and 0 for every secondary. Only a
    primary splits at its first Compton scatter (Compton splitting, Phase 4);
    the generators set it, ``_queue_push`` always pushes 0.
    """

    kind: wp.array(dtype=wp.int32)
    beamlet: wp.array(dtype=wp.int32)
    primary: wp.array(dtype=wp.int32)
    energy: wp.array(dtype=float)
    weight: wp.array(dtype=float)
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
    beamlet: int,
    energy: float,
    weight: float,
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
        q.beamlet[idx] = beamlet
        q.primary[idx] = 0  # every queued particle is a secondary, never a source
        q.energy[idx] = energy
        q.weight[idx] = weight
        q.x[idx] = x
        q.y[idx] = y
        q.z[idx] = z
        q.ux[idx] = ux
        q.uy[idx] = uy
        q.uz[idx] = uz
        q.rng[idx] = rng_state


@wp.func
def _push_source(
    q: Queue,
    kind: int,
    energy: float,
    weight: float,
    x: float,
    y: float,
    z: float,
    ux: float,
    uy: float,
    uz: float,
    rng_slot: wp.uint32,
):
    """Atomic-append a *source* primary (``primary=1``, ``beamlet=0``) to a queue.

    Mirror of :func:`_queue_push` for the wrapped-``wp.func`` generator route: the
    generator may emit either kind, so it pushes into the matching queue at an atomic
    slot rather than the thread's fixed index. ``primary=1`` marks it a source
    particle (the only one that may split at first Compton).
    """
    idx = wp.atomic_add(q.count, 0, 1)
    if idx < q.capacity:
        q.kind[idx] = kind
        q.beamlet[idx] = 0
        q.primary[idx] = 1
        q.energy[idx] = energy
        q.weight[idx] = weight
        q.x[idx] = x
        q.y[idx] = y
        q.z[idx] = z
        q.ux[idx] = ux
        q.uy[idx] = uy
        q.uz[idx] = uz
        q.rng[idx] = rng_slot


_generator_cache: dict = {}


def make_generator_kernel(sampler):
    """Wrap a user ``@wp.func`` primary sampler into a generator kernel (cached).

    ``sampler(history_index: int, state) -> (kind, energy, x, y, z, ux, uy, uz,
    weight)`` is the advanced Warp route
    (:attr:`pyRadMC.geometry.source.Source.warp_sampler`): the returned kernel sets up
    the per-history RNG slot exactly as the built-in generators do (so a primary's
    transport continues the same stream the sampler drew from), calls the user func,
    books the emitted weight-energy (plus a positron's ``2 m_e c^2`` rest mass) into a
    signed fixed-point counter, and pushes the primary into the photon or electron
    queue by its kind. Memoized per sampler object so each is compiled once.
    """
    cached = _generator_cache.get(sampler)
    if cached is not None:
        return cached

    @wp.kernel
    def generate(
        seed: int,
        history_offset: int,
        q_photon: Queue,
        q_electron: Queue,
        slots: wp.array(dtype=wp.uint32),
        emitted: wp.array(dtype=wp.int64),
    ):
        tid = wp.tid()
        h = history_offset + tid
        slots[tid] = init_slot(seed, h)
        state = WarpRNGState()
        state.slots = slots
        state.idx = tid
        kind, energy, x, y, z, ux, uy, uz, weight = sampler(h, state)
        latent = 0.0
        if kind == POSITRON:
            latent = 2.0 * ELECTRON_MASS_MEV
        _escape(emitted, weight * (energy + latent))  # signed fixed-point accumulator
        if kind == PHOTON:
            _push_source(q_photon, kind, energy, weight, x, y, z, ux, uy, uz, slots[tid])
        else:
            _push_source(q_electron, kind, energy, weight, x, y, z, ux, uy, uz, slots[tid])

    _generator_cache[sampler] = generate
    return generate


_beamlet_generator_cache: dict = {}


def make_beamlet_generator_kernel(sampler):
    """Wrap a user ``@wp.func`` beamlet sampler into a Dij generator kernel (cached).

    ``sampler(beamlet: int, within_index: int, state) -> (energy, x, y, z, ux, uy, uz,
    weight)`` is the advanced Warp Dij route
    (:attr:`pyRadMC.geometry.source.BeamletSource.warp_beamlet_sampler`): the returned
    kernel bakes the same one-batch block mapping as :func:`generate_beamlet_lattice` —
    group-local beamlet ``t // per_batch``, within-beamlet index
    ``r = batch*per_batch + i``, global history ``h = beamlet*n_per + r``, correlated key
    ``r`` — calls the sampler for the primary, and writes a photon at the sampled weight
    tagged with the column ``local`` (the weight carries deterministic source-side
    attenuation; it must match what the host ``emit`` returns or the backends disagree
    systematically). Emitted ``weight * energy`` is booked into a signed fixed-point
    counter. Memoized per sampler.
    """
    cached = _beamlet_generator_cache.get(sampler)
    if cached is not None:
        return cached

    @wp.kernel
    def generate(
        seed: int,
        group_start: int,
        n_per: int,
        per_batch: int,
        batch: int,
        t_offset: int,
        correlated: int,
        slots: wp.array(dtype=wp.uint32),
        emitted: wp.array(dtype=wp.int64),
        q: Queue,
    ):
        tid = wp.tid()
        t = t_offset + tid
        local = t // per_batch
        i = t - local * per_batch
        r = batch * per_batch + i
        beamlet = group_start + local
        h = beamlet * n_per + r
        key = h
        if correlated != 0:
            key = r
        slots[tid] = init_slot(seed, key)
        state = WarpRNGState()
        state.slots = slots
        state.idx = tid
        energy, x, y, z, ux, uy, uz, weight = sampler(beamlet, r, state)
        _escape(emitted, weight * energy)  # photon; no positron latent in a Dij
        q.kind[tid] = PHOTON
        q.beamlet[tid] = local
        q.primary[tid] = 1
        q.energy[tid] = energy
        q.weight[tid] = weight
        q.x[tid] = x
        q.y[tid] = y
        q.z[tid] = z
        q.ux[tid] = ux
        q.uy[tid] = uy
        q.uz[tid] = uz
        q.rng[tid] = slots[tid]

    _beamlet_generator_cache[sampler] = generate
    return generate


@wp.func
def _spr_factor(
    tab: Tables,
    dose_to_water: int,
    energy: float,
    mat: int,
    ecut: float,
) -> float:
    """Dose-to-water deposit weight from the flattened stopping tables.

    In-kernel mirror of :func:`pyRadMC.scoring.dose_to_water.water_spr`: the
    restricted collision stopping-power ratio water/medium at the deposit's
    cutoff-clamped energy (the lookup additionally clamps to the table domain).
    Exactly 1.0 under dose-to-medium — and for water deposits, where it is the
    ratio of two identical lookups. The water case returns the exact 1.0
    *without dividing*: the host mirror gets x/x == 1.0 from IEEE float64, but
    this module compiles with fast_math, whose approximate division would break
    the test-pinned water D_w == D_m bit identity.
    """
    if dose_to_water == 0:
        return 1.0
    if mat == WATER:
        return 1.0
    e = wp.max(energy, ecut)
    s_water = lookup_2d(
        tab.stopping_restricted, WATER, tab.e_log_e_min, tab.e_inv_dlog, tab.n_points, e
    )
    s_medium = lookup_2d(
        tab.stopping_restricted, mat, tab.e_log_e_min, tab.e_inv_dlog, tab.n_points, e
    )
    return s_water / s_medium


@wp.func
def _deposit(
    edep: wp.array(dtype=wp.int64),
    deposited: wp.array(dtype=wp.int64),
    unscored: wp.array(dtype=wp.int64),
    si: GridInfo,
    base: int,
    x: float,
    y: float,
    z: float,
    energy: float,
    factor: float,
    dose_to_water: int,
):
    """Round-to-nearest fixed-point deposit at a position inside the transport grid.

    Routes by position into the *scoring* grid ``si`` (Phase 5 decoupled dose
    grid), mirroring ``BatchedDoseScorer.deposit_at``: the containing scoring
    voxel of this particle's column, or the unscored ledger counter when the
    scoring grid does not cover the position — never a clamped edge voxel, which
    would corrupt edge dose. Quantization happens once, before routing, so
    ``deposited + unscored`` is invariant to the scoring grid in exact quanta.

    ``base`` is the flat offset of this particle's beamlet column
    (``beamlet * si.n_voxels``); zero for plain dose runs, whose ``edep`` is a
    single column. When the scoring grid is the transport grid (the default),
    the routing math is the identical ``point_axis_index`` arithmetic the
    transport loop used, so behaviour is byte-identical to indexed deposits.

    ``factor`` is the dose-to-water weight of this deposit (``_spr_factor``;
    exactly 1.0 under dose-to-medium, where multiplying by it is a bit-exact
    identity): the *tally* accumulates the weighted quanta, while the ledger —
    the unscored bucket, and under dose-to-water the ``deposited`` counter —
    books physical quanta, so the energy balance is scoring-mode invariant.
    Under dose-to-medium the host derives deposited energy from ``edep``
    directly and the counter is skipped.
    """
    if point_inside(x, y, z, si.x_lo, si.y_lo, si.z_lo, si.x_hi, si.y_hi, si.z_hi):
        quanta = wp.int64(
            wp.float64(energy) * wp.float64(factor) * wp.float64(_INV_QUANTUM) + wp.float64(0.5)
        )
        ix = point_axis_index(x, si.x_lo, si.sx, si.nx)
        iy = point_axis_index(y, si.y_lo, si.sy, si.ny)
        iz = point_axis_index(z, si.z_lo, si.sz, si.nz)
        wp.atomic_add(edep, base + (ix * si.ny + iy) * si.nz + iz, quanta)
        if dose_to_water != 0:
            physical = wp.int64(wp.float64(energy) * wp.float64(_INV_QUANTUM) + wp.float64(0.5))
            wp.atomic_add(deposited, 0, physical)
    else:
        physical = wp.int64(wp.float64(energy) * wp.float64(_INV_QUANTUM) + wp.float64(0.5))
        wp.atomic_add(unscored, 0, physical)


@wp.func
def _escape(escaped: wp.array(dtype=wp.int64), energy: float):
    """Signed ledger entry: roulette boosts book *negative* weight-energy here.

    ``floor(x + 0.5)`` is round-to-nearest for either sign (plain int64 casting
    truncates toward zero, which would round negative entries the wrong way); for
    the non-negative deposits it is bit-identical to the previous truncation.
    """
    quanta = wp.int64(wp.floor(wp.float64(energy) * wp.float64(_INV_QUANTUM) + wp.float64(0.5)))
    wp.atomic_add(escaped, 0, quanta)


@wp.func
def _deposit_or_escape(
    edep: wp.array(dtype=wp.int64),
    escaped: wp.array(dtype=wp.int64),
    deposited: wp.array(dtype=wp.int64),
    unscored: wp.array(dtype=wp.int64),
    gi: GridInfo,
    si: GridInfo,
    base: int,
    x: float,
    y: float,
    z: float,
    amount: float,
    factor: float,
    dose_to_water: int,
):
    """Mirror of the reference electron loop's midpoint deposit helper.

    Escape is decided against the *transport* grid ``gi`` (outside it is
    vacuum); a deposit inside it then routes by position into the scoring grid
    ``si`` or the unscored bucket, exactly as on the reference backend. The
    escaped ledger books the physical amount; ``factor`` weights the tally only.
    """
    if amount <= 0.0:
        return
    if point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
        _deposit(edep, deposited, unscored, si, base, x, y, z, amount, factor, dose_to_water)
    else:
        _escape(escaped, amount)


@wp.func
def _annihilate_at_rest(
    x: float,
    y: float,
    z: float,
    si: GridInfo,
    state: WarpRNGState,
    q_photon: Queue,
    edep: wp.array(dtype=wp.int64),
    deposited: wp.array(dtype=wp.int64),
    unscored: wp.array(dtype=wp.int64),
    pcut: float,
    beamlet: int,
    base: int,
    weight: float,
    factor: float,
    dose_to_water: int,
):
    """Positron annihilation at rest; mirror of transport.particles.annihilate_at_rest.

    Called only for positions inside the transport grid. Each annihilation photon
    gets its own child stream, since the pair are transported by independent
    threads; both inherit the positron's beamlet tag and statistical weight.
    ``factor`` is the caller's dose-to-water weight at the deposit site.
    """
    if pcut >= ELECTRON_MASS_MEV:
        _deposit(
            edep,
            deposited,
            unscored,
            si,
            base,
            x,
            y,
            z,
            weight * 2.0 * ELECTRON_MASS_MEV,
            factor,
            dose_to_water,
        )
        return
    ax, ay, az = sample_isotropic_direction(state)
    _queue_push(
        q_photon,
        PHOTON,
        beamlet,
        ELECTRON_MASS_MEV,
        weight,
        x,
        y,
        z,
        ax,
        ay,
        az,
        spawn_stream(state),
    )
    _queue_push(
        q_photon,
        PHOTON,
        beamlet,
        ELECTRON_MASS_MEV,
        weight,
        x,
        y,
        z,
        -ax,
        -ay,
        -az,
        spawn_stream(state),
    )


# --- batch reduction: fixed-point energy maps -> per-voxel dose mean & sigma ------
#
# Moves the scoring finalize (the mass divide and the batch mean/variance) onto the
# device, so only the reduced dose maps — not the dense per-group fixed-point buffer —
# cross back to the host. The two helpers below are shared by the open-field ``run``
# and the ``run_dij`` reductions so those two paths stay **bit-identical within a
# device** (AGENTS 2.3): a 1x1 Dij column reduces to the open-field dose bit for bit
# because both fold the same per-batch doses, in the same order, through the same
# float64 arithmetic. They deliberately mirror ``BatchedDoseScorer`` (host, NumPy
# float64); across host and device only statistical equivalence is claimed, never bit
# equality, so ``ref`` is unaffected.


@wp.func
def _batch_dose(
    q: wp.int64, mass: wp.float64, nhist: wp.float64, quantum: wp.float64
) -> wp.float64:
    """One voxel's per-batch dose, float64, matching ``BatchedDoseScorer.end_batch``.

    ``energy = quanta * quantum``; ``dose = energy / (mass * n_histories)``; exactly
    zero where the scoring voxel carries no mass (uncovered — the masked divide on the
    host). The operation order is fixed so the host and device agree bitwise on one
    target.
    """
    if mass <= wp.float64(0.0):
        return wp.float64(0.0)
    energy = wp.float64(q) * quantum
    return energy / (mass * nhist)


@wp.func
def _mean_sigma(s1: wp.float64, s2: wp.float64, n_batches: int):
    """Batch-mean dose and its standard error, mirror of ``BatchedDoseScorer.finalize``.

    ``mean = S1/n``; the sample variance of the batch means, clamped at zero against
    float cancellation; ``sigma = sqrt(var/n)``. A single batch has no spread, so its
    sigma is zero (degenerate, as on the host).
    """
    n = wp.float64(n_batches)
    mean = s1 / n
    if n_batches == 1:
        return mean, wp.float64(0.0)
    var = wp.max(wp.float64(0.0), s2 / n - mean * mean) * n / (n - wp.float64(1.0))
    return mean, wp.sqrt(var / n)


@wp.kernel
def accumulate_dij_batch(
    edep: wp.array(dtype=wp.int64),
    voxel_mass: wp.array(dtype=wp.float64),
    n_voxels: int,
    nhist: wp.float64,
    quantum: wp.float64,
    s1: wp.array(dtype=wp.float64),
    s2: wp.array(dtype=wp.float64),
    total_quanta: wp.array(dtype=wp.int64),
):
    """Fold one batch of a beamlet group into the running per-column dose sums.

    The Dij counterpart of :func:`accumulate_run_batch`: one thread per
    ``(local, voxel)`` over the group's ``group * n_voxels`` dense map, launched once
    per statistical batch. A voxel is touched by a single thread per launch and the
    launches are sequential in batch order, so ``s1`` accumulates ``sum_b dose_b``
    as the same left fold, term for term, that the batch-axis loop it replaces
    performed — which is what keeps ``run`` and ``run_dij`` bitwise equal for a 1x1
    lattice, and keeps the Dij invariant to the batch-vs-device-axis scheduling.

    ``total_quanta`` accumulates the group's fixed-point quanta (the dose-to-medium
    deposited-energy book; dose-to-water uses its physical counter). Integer
    addition is associative, so its atomic accumulation is order-independent.
    """
    tid = wp.tid()
    local = tid // n_voxels
    vox = tid - local * n_voxels
    q = edep[tid]
    d = _batch_dose(q, voxel_mass[vox], nhist, quantum)
    s1[tid] = s1[tid] + d
    s2[tid] = s2[tid] + d * d
    wp.atomic_add(total_quanta, 0, q)


@wp.kernel
def accumulate_run_batch(
    edep: wp.array(dtype=wp.int64),
    voxel_mass: wp.array(dtype=wp.float64),
    nhist: wp.float64,
    quantum: wp.float64,
    s1: wp.array(dtype=wp.float64),
    s2: wp.array(dtype=wp.float64),
    total_quanta: wp.array(dtype=wp.int64),
):
    """Fold one open-field batch into the running per-voxel dose sums.

    Launched once per statistical batch (one thread per voxel, no race — a voxel is
    touched by a single thread per launch, and launches are sequential), so ``s1``
    accumulates ``sum_b dose_b`` in batch order — the same left fold, term for term,
    that :func:`accumulate_dij_batch` performs for a 1x1 column, hence the bitwise
    equality between ``run`` and ``run_dij`` on one device.
    """
    vox = wp.tid()
    q = edep[vox]
    d = _batch_dose(q, voxel_mass[vox], nhist, quantum)
    s1[vox] = s1[vox] + d
    s2[vox] = s2[vox] + d * d
    wp.atomic_add(total_quanta, 0, q)


@wp.kernel
def finalize_run(
    s1: wp.array(dtype=wp.float64),
    s2: wp.array(dtype=wp.float64),
    n_batches: int,
    mean_out: wp.array(dtype=wp.float64),
    sigma_out: wp.array(dtype=wp.float64),
):
    """Turn the accumulated open-field sums into per-voxel dose mean and sigma."""
    vox = wp.tid()
    mean, sigma = _mean_sigma(s1[vox], s2[vox], n_batches)
    mean_out[vox] = mean
    sigma_out[vox] = sigma


# --- device truncation + compaction: dense dose maps -> sparse CSC columns --------
#
# The per-column truncation (AGENTS 2.8) and the CSC compaction run on the device, so
# only the surviving sparse entries — not the dense per-column dose maps — read back.
# Determinism (AGENTS 2.3): each column is handled by a single thread that scans
# voxels in ascending index order, so the kept indices come out sorted with no
# atomics and the layout is bit-reproducible. ``col_max`` is a plain maximum (no
# arithmetic), and the keep test ``mean >= truncation * col_max and mean > 0`` uses
# the same float64 product as ``DijAssembler.add_block``, so the sparse pattern and
# values are byte-identical to the host truncation on the same dose maps.


@wp.kernel
def fill_int32(a: wp.array(dtype=wp.int32), value: int):
    """Stream-aware int32 fill.

    ``array.zero_()`` and host uploads run on the device's *current* stream; a
    batch lane running on its own stream must set queue counts with an operation
    it can order on that stream, which a plain kernel launch is.
    """
    a[wp.tid()] = value


@wp.kernel
def fill_int64(a: wp.array(dtype=wp.int64), value: wp.int64):
    """Stream-aware int64 fill; the lane-stream counterpart of ``edep.zero_()``."""
    a[wp.tid()] = value


@wp.kernel
def count_kept_per_column(
    mean: wp.array(dtype=wp.float64),
    n_voxels: int,
    truncation: wp.float64,
    counts: wp.array(dtype=wp.int32),
):
    """Per-column count of voxels surviving truncation (one thread per column)."""
    local = wp.tid()
    base = local * n_voxels
    col_max = wp.float64(0.0)
    for v in range(n_voxels):
        m = mean[base + v]
        if m > col_max:
            col_max = m
    thr = truncation * col_max
    c = int(0)  # noqa: UP018, RUF046 - Warp needs a dynamic int var for the loop counter
    for v in range(n_voxels):
        m = mean[base + v]
        if m > wp.float64(0.0) and m >= thr:
            c += 1
    counts[local] = c


@wp.kernel
def compact_column(
    mean: wp.array(dtype=wp.float64),
    sigma: wp.array(dtype=wp.float64),
    n_voxels: int,
    truncation: wp.float64,
    offsets: wp.array(dtype=wp.int32),
    out_indices: wp.array(dtype=wp.int64),
    out_dose: wp.array(dtype=wp.float64),
    out_sigma: wp.array(dtype=wp.float64),
):
    """Scatter each column's surviving entries into the compact CSC arrays.

    ``offsets`` is the exclusive prefix sum of the counts, so column ``local`` writes
    a contiguous run; scanning voxels ascending makes the row indices sorted, exactly
    like ``np.flatnonzero`` on the host.
    """
    local = wp.tid()
    base = local * n_voxels
    col_max = wp.float64(0.0)
    for v in range(n_voxels):
        m = mean[base + v]
        if m > col_max:
            col_max = m
    thr = truncation * col_max
    cursor = offsets[local]
    for v in range(n_voxels):
        m = mean[base + v]
        if m > wp.float64(0.0) and m >= thr:
            out_indices[cursor] = wp.int64(v)
            out_dose[cursor] = m
            out_sigma[cursor] = sigma[base + v]
            cursor += 1


@wp.kernel
def photon_kernel(
    gi: GridInfo,
    si: GridInfo,
    density: wp.array3d(dtype=float),
    material: wp.array3d(dtype=wp.uint8),
    tab: Tables,
    q_in: Queue,
    q_electron: Queue,
    q_photon_out: Queue,
    slots: wp.array(dtype=wp.uint32),
    edep: wp.array(dtype=wp.int64),
    escaped: wp.array(dtype=wp.int64),
    unscored: wp.array(dtype=wp.int64),
    deposited: wp.array(dtype=wp.int64),
    pcut: float,
    ecut: float,
    transport_electrons: int,
    dose_to_water: int,
    majorant_violations: wp.array(dtype=wp.int32),
):
    """One thread transports one photon to termination; Woodcock tracking.

    Sequencing mirrors ``transport.photon.photon_steps`` statement for statement;
    consult that loop for the physics commentary. ``gi`` is the transport grid;
    ``si`` the scoring grid deposits route into (equal to ``gi`` by default).
    ``dose_to_water`` selects the tally weighting (``_spr_factor``); the books
    stay physical either way.
    """
    tid = wp.tid()
    e = q_in.energy[tid]
    beamlet = q_in.beamlet[tid]
    base = beamlet * si.n_voxels
    w = q_in.weight[tid]
    primary = q_in.primary[tid]
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
            _escape(escaped, w * e)
            return
        t += _ENTRY_NUDGE_CM
        x += t * ux
        y += t * uy
        z += t * uz
        if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
            _escape(escaped, w * e)
            return

    while True:
        mu_majorant = lookup_1d(tab.majorant, tab.p_log_e_min, tab.p_inv_dlog, tab.n_points, e)
        step = sample_path_length(mu_majorant, state)
        x += step * ux
        y += step * uy
        z += step * uz
        if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
            _escape(escaped, w * e)
            return

        ix = point_axis_index(x, gi.x_lo, gi.sx, gi.nx)
        iy = point_axis_index(y, gi.y_lo, gi.sy, gi.ny)
        iz = point_axis_index(z, gi.z_lo, gi.sz, gi.nz)
        rho = density[ix, iy, iz]
        mat = int(material[ix, iy, iz])  # uint8 voxel map -> table row index
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
            _escape(escaped, w * e)
            return

        if uniform(state) * mu_majorant >= mu_real:
            continue  # delta (fictitious) scattering

        process = select_photon_process(mu_compton, mu_photo, mu_pair, mu_rayleigh, state)

        if process == PhotonProcess.RAYLEIGH:
            cos_coherent = sample_coherent_cos_theta_form_factor(
                tab.coherent_cumulative[mat], tab.coherent_x, tab.n_coherent, e, state
            )
            phi = 2.0 * math.pi * uniform(state)
            ux, uy, uz = rotate_direction(ux, uy, uz, cos_coherent, phi)
        elif process == PhotonProcess.COMPTON:
            if PHOTON_SPLIT_N > 1 and primary != 0:
                # Compton splitting (Phase 4): the source photon samples
                # PHOTON_SPLIT_N independent final states, each copy (scattered
                # photon + recoil electron) weighted w/N and pushed as a
                # non-primary photon; the primary's thread ends here. Ships at
                # PHOTON_SPLIT_N=1 (analog), where the constant guard above
                # dead-codes this block and a primary's first Compton continues
                # in-thread like every later scatter — no re-queue, no child
                # stream; the N > 1 machinery compiles in unchanged whenever the
                # constant says so. Only the primary splits, so the population
                # is bounded.
                # Copies get child streams (spawn_stream), the warp counterpart
                # of the reference's sequential draws; each copy conserves energy
                # (scattered + recoil = e), so the balance stays exact.
                split_w = w / float(PHOTON_SPLIT_N)
                for _ in range(PHOTON_SPLIT_N):
                    ratio = sample_compton_energy_ratio(e, state)
                    recoil = e * (1.0 - ratio)
                    phi = 2.0 * math.pi * uniform(state)
                    if transport_electrons != 0 and recoil > ecut:
                        cos_electron = compton_electron_cos_theta(e, ratio)
                        ex, ey, ez = rotate_direction(ux, uy, uz, cos_electron, phi + math.pi)
                        _queue_push(
                            q_electron,
                            ELECTRON,
                            beamlet,
                            recoil,
                            split_w,
                            x,
                            y,
                            z,
                            ex,
                            ey,
                            ez,
                            spawn_stream(state),
                        )
                    else:
                        f = _spr_factor(tab, dose_to_water, recoil, mat, ecut)
                        _deposit(
                            edep,
                            deposited,
                            unscored,
                            si,
                            base,
                            x,
                            y,
                            z,
                            split_w * recoil,
                            f,
                            dose_to_water,
                        )
                    cos_gamma = compton_cos_theta(e, ratio)
                    gx, gy, gz = rotate_direction(ux, uy, uz, cos_gamma, phi)
                    e_scatter = e * ratio
                    if e_scatter <= pcut:
                        f = _spr_factor(tab, dose_to_water, e_scatter, mat, ecut)
                        _deposit(
                            edep,
                            deposited,
                            unscored,
                            si,
                            base,
                            x,
                            y,
                            z,
                            split_w * e_scatter,
                            f,
                            dose_to_water,
                        )
                    else:
                        copy_w = split_w
                        keep = True
                        if e_scatter < PHOTON_ROULETTE_MEV and copy_w < PHOTON_ROULETTE_WEIGHT_CAP:
                            new_w = roulette_weight(copy_w, PHOTON_ROULETTE_SURVIVAL, state)
                            _escape(escaped, (copy_w - new_w) * e_scatter)
                            if new_w == 0.0:
                                keep = False
                            else:
                                copy_w = new_w
                        if keep:
                            _queue_push(
                                q_photon_out,
                                PHOTON,
                                beamlet,
                                e_scatter,
                                copy_w,
                                x,
                                y,
                                z,
                                gx,
                                gy,
                                gz,
                                spawn_stream(state),
                            )
                return
            # Non-primary photon: single scatter, continue in-thread — the bulk
            # of the cascade. Only the source photon's first Compton re-queues
            # (splitting, above); everything here mirrors the reference loop.
            ratio = sample_compton_energy_ratio(e, state)
            recoil = e * (1.0 - ratio)
            phi = 2.0 * math.pi * uniform(state)
            if transport_electrons != 0 and recoil > ecut:
                cos_electron = compton_electron_cos_theta(e, ratio)
                ex, ey, ez = rotate_direction(ux, uy, uz, cos_electron, phi + math.pi)
                _queue_push(
                    q_electron,
                    ELECTRON,
                    beamlet,
                    recoil,
                    w,
                    x,
                    y,
                    z,
                    ex,
                    ey,
                    ez,
                    spawn_stream(state),
                )
            else:
                f = _spr_factor(tab, dose_to_water, recoil, mat, ecut)
                _deposit(edep, deposited, unscored, si, base, x, y, z, w * recoil, f, dose_to_water)
            cos_gamma = compton_cos_theta(e, ratio)
            ux, uy, uz = rotate_direction(ux, uy, uz, cos_gamma, phi)
            e *= ratio
            if e <= pcut:
                _deposit(
                    edep,
                    deposited,
                    unscored,
                    si,
                    base,
                    x,
                    y,
                    z,
                    w * e,
                    _spr_factor(tab, dose_to_water, e, mat, ecut),
                    dose_to_water,
                )
                return
            if e < PHOTON_ROULETTE_MEV and w < PHOTON_ROULETTE_WEIGHT_CAP:
                # Basic VR, mirror of the reference loop: roulette the softened
                # scattered photon; the signed ledger entry keeps the balance exact.
                new_w = roulette_weight(w, PHOTON_ROULETTE_SURVIVAL, state)
                _escape(escaped, (w - new_w) * e)
                if new_w == 0.0:
                    return
                w = new_w
        elif process == PhotonProcess.PHOTOELECTRIC:
            if transport_electrons != 0 and e > ecut:
                _queue_push(
                    q_electron, ELECTRON, beamlet, e, w, x, y, z, ux, uy, uz, spawn_stream(state)
                )
            else:
                _deposit(
                    edep,
                    deposited,
                    unscored,
                    si,
                    base,
                    x,
                    y,
                    z,
                    w * e,
                    _spr_factor(tab, dose_to_water, e, mat, ecut),
                    dose_to_water,
                )
            return
        else:  # pair production
            kinetic = e - 2.0 * ELECTRON_MASS_MEV
            if transport_electrons != 0:
                fraction = uniform(state)
                share_e = fraction * kinetic
                share_p = (1.0 - fraction) * kinetic
                if share_e > ecut:
                    _queue_push(
                        q_electron,
                        ELECTRON,
                        beamlet,
                        share_e,
                        w,
                        x,
                        y,
                        z,
                        ux,
                        uy,
                        uz,
                        spawn_stream(state),
                    )
                else:
                    f = _spr_factor(tab, dose_to_water, share_e, mat, ecut)
                    _deposit(
                        edep, deposited, unscored, si, base, x, y, z, w * share_e, f, dose_to_water
                    )
                if share_p > ecut:
                    _queue_push(
                        q_electron,
                        POSITRON,
                        beamlet,
                        share_p,
                        w,
                        x,
                        y,
                        z,
                        ux,
                        uy,
                        uz,
                        spawn_stream(state),
                    )
                else:
                    f = _spr_factor(tab, dose_to_water, share_p, mat, ecut)
                    _deposit(
                        edep, deposited, unscored, si, base, x, y, z, w * share_p, f, dose_to_water
                    )
                    _annihilate_at_rest(
                        x,
                        y,
                        z,
                        si,
                        state,
                        q_photon_out,
                        edep,
                        deposited,
                        unscored,
                        pcut,
                        beamlet,
                        base,
                        w,
                        _spr_factor(tab, dose_to_water, ELECTRON_MASS_MEV, mat, ecut),
                        dose_to_water,
                    )
            else:
                f = _spr_factor(tab, dose_to_water, kinetic, mat, ecut)
                _deposit(
                    edep, deposited, unscored, si, base, x, y, z, w * kinetic, f, dose_to_water
                )
                _annihilate_at_rest(
                    x,
                    y,
                    z,
                    si,
                    state,
                    q_photon_out,
                    edep,
                    deposited,
                    unscored,
                    pcut,
                    beamlet,
                    base,
                    w,
                    _spr_factor(tab, dose_to_water, ELECTRON_MASS_MEV, mat, ecut),
                    dose_to_water,
                )
            return


@wp.kernel
def electron_kernel(
    gi: GridInfo,
    si: GridInfo,
    density: wp.array3d(dtype=float),
    material: wp.array3d(dtype=wp.uint8),
    tab: Tables,
    q_in: Queue,
    q_photon: Queue,
    q_electron_out: Queue,
    slots: wp.array(dtype=wp.uint32),
    edep: wp.array(dtype=wp.int64),
    escaped: wp.array(dtype=wp.int64),
    unscored: wp.array(dtype=wp.int64),
    deposited: wp.array(dtype=wp.int64),
    pcut: float,
    ecut: float,
    dose_to_water: int,
):
    """One thread transports one electron or positron; Class II condensed history.

    Sequencing mirrors ``transport.electron.electron_steps`` statement for
    statement; delta rays go to the electron out-queue instead of a stack, and
    bremsstrahlung/annihilation photons to the photon queue. ``gi`` is the
    transport grid; ``si`` the scoring grid deposits route into.
    ``dose_to_water`` selects the tally weighting (``_spr_factor``): one factor
    per substep, at the substep's initial energy and voxel material — the same
    (E, material) the restricted stopping power charged the loss with.
    """
    tid = wp.tid()
    is_positron = q_in.kind[tid] == POSITRON
    e = q_in.energy[tid]
    beamlet = q_in.beamlet[tid]
    base = beamlet * si.n_voxels
    w = q_in.weight[tid]
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
            _escape(escaped, w * (e + latent))
            return
        t += _ENTRY_NUDGE_CM
        x += t * ux
        y += t * uy
        z += t * uz
        if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
            _escape(escaped, w * (e + latent))
            return

    while True:
        if e <= ecut:
            # Terminal: local deposit; the material lookup is needed for the
            # dose-to-water factor (a source may inject a sub-cutoff electron,
            # so the loop body's lookup may not have run yet).
            t_ix = point_axis_index(x, gi.x_lo, gi.sx, gi.nx)
            t_iy = point_axis_index(y, gi.y_lo, gi.sy, gi.ny)
            t_iz = point_axis_index(z, gi.z_lo, gi.sz, gi.nz)
            t_mat = int(material[t_ix, t_iy, t_iz])  # uint8 voxel map -> table row index
            f = _spr_factor(tab, dose_to_water, e, t_mat, ecut)
            _deposit(edep, deposited, unscored, si, base, x, y, z, w * e, f, dose_to_water)
            if is_positron:
                _annihilate_at_rest(
                    x,
                    y,
                    z,
                    si,
                    state,
                    q_photon,
                    edep,
                    deposited,
                    unscored,
                    pcut,
                    beamlet,
                    base,
                    w,
                    _spr_factor(tab, dose_to_water, ELECTRON_MASS_MEV, t_mat, ecut),
                    dose_to_water,
                )
            return

        ix = point_axis_index(x, gi.x_lo, gi.sx, gi.nx)
        iy = point_axis_index(y, gi.y_lo, gi.sy, gi.ny)
        iz = point_axis_index(z, gi.z_lo, gi.sz, gi.nz)
        rho = density[ix, iy, iz]
        mat = int(material[ix, iy, iz])  # uint8 voxel map -> table row index

        # --- substep length ----------------------------------------------------
        range_cm = (
            lookup_2d(tab.csda_range, mat, tab.e_log_e_min, tab.e_inv_dlog, tab.n_points, e) / rho
        )
        s_max = min(STEP_ENERGY_FRACTION * range_cm, STEP_VOXEL_FRACTION * gi.min_spacing)
        # Cap at the next voxel face (plus the nudge across it) so the substep's
        # density and material stay those of the voxel it starts in.
        s_boundary = (
            distance_to_voxel_boundary(
                x,
                y,
                z,
                ux,
                uy,
                uz,
                gi.x_lo,
                gi.y_lo,
                gi.z_lo,
                gi.sx,
                gi.sy,
                gi.sz,
                gi.nx,
                gi.ny,
                gi.nz,
            )
            + BOUNDARY_NUDGE_CM
        )
        s_geometry = min(s_max, s_boundary)
        sigma_moller = rho * lookup_2d(
            tab.moller, mat, tab.e_log_e_min, tab.e_inv_dlog, tab.n_points, e
        )
        s = s_geometry
        moller_pending = False
        if sigma_moller > 0.0:
            s_interaction = sample_path_length(sigma_moller, state)
            if s_interaction <= s_geometry:
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
        substep_factor = _spr_factor(tab, dose_to_water, e, mat, ecut)

        # --- random hinge: move, deflect, move ----------------------------------
        s1 = uniform(state) * s
        s2 = s - s1
        d1 = 0.0
        if s > 0.0:
            d1 = continuous * (s1 / s)
        d2 = continuous - d1

        _deposit_or_escape(
            edep,
            escaped,
            deposited,
            unscored,
            gi,
            si,
            base,
            x + ux * s1 / 2.0,
            y + uy * s1 / 2.0,
            z + uz * s1 / 2.0,
            w * d1,
            substep_factor,
            dose_to_water,
        )
        e -= d1
        x += ux * s1
        y += uy * s1
        z += uz * s1
        if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
            _escape(escaped, w * (e + latent))
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
                photon_w = w
                if k < PHOTON_ROULETTE_MEV and w < PHOTON_ROULETTE_WEIGHT_CAP:
                    # Basic VR, mirror of the reference loop: roulette the soft
                    # bremsstrahlung photon at birth; signed ledger keeps balance.
                    photon_w = roulette_weight(w, PHOTON_ROULETTE_SURVIVAL, state)
                    _escape(escaped, (w - photon_w) * k)
                if photon_w > 0.0:
                    _queue_push(
                        q_photon,
                        PHOTON,
                        beamlet,
                        k,
                        photon_w,
                        x,
                        y,
                        z,
                        ux,
                        uy,
                        uz,
                        spawn_stream(state),
                    )
            else:
                _deposit_or_escape(
                    edep,
                    escaped,
                    deposited,
                    unscored,
                    gi,
                    si,
                    base,
                    x,
                    y,
                    z,
                    w * k,
                    _spr_factor(tab, dose_to_water, k, mat, ecut),
                    dose_to_water,
                )
            e -= k

        _deposit_or_escape(
            edep,
            escaped,
            deposited,
            unscored,
            gi,
            si,
            base,
            x + ux * s2 / 2.0,
            y + uy * s2 / 2.0,
            z + uz * s2 / 2.0,
            w * d2,
            substep_factor,
            dose_to_water,
        )
        e -= d2
        x += ux * s2
        y += uy * s2
        z += uz * s2
        if not point_inside(x, y, z, gi.x_lo, gi.y_lo, gi.z_lo, gi.x_hi, gi.y_hi, gi.z_hi):
            _escape(escaped, w * (e + latent))
            return

        # --- discrete Moller event at the end of the substep ---------------------
        if moller_pending and e > 2.0 * ecut:
            delta_energy = sample_moller_delta_energy(e, ecut, state)
            cos_delta, cos_primary = moller_direction_cosines(e, delta_energy)
            phi = 2.0 * math.pi * uniform(state)
            dx, dy, dz = rotate_direction(ux, uy, uz, cos_delta, phi)
            _queue_push(
                q_electron_out,
                ELECTRON,
                beamlet,
                delta_energy,
                w,
                x,
                y,
                z,
                dx,
                dy,
                dz,
                spawn_stream(state),
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
    q.beamlet[tid] = 0
    q.primary[tid] = 1  # source particle: the only one that may split
    q.energy[tid] = energy
    q.weight[tid] = 1.0
    q.x[tid] = px
    q.y[tid] = py
    q.z[tid] = pz
    q.ux[tid] = dx
    q.uy[tid] = dy
    q.uz[tid] = dz
    q.rng[tid] = init_slot(seed, history_offset + tid)


@wp.kernel
def generate_from_upload(
    seed: int,
    hist: wp.array(dtype=wp.int32),
    beamlet: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    energy: wp.array(dtype=float),
    px: wp.array(dtype=float),
    py: wp.array(dtype=float),
    pz: wp.array(dtype=float),
    dx: wp.array(dtype=float),
    dy: wp.array(dtype=float),
    dz: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    q: Queue,
):
    """Write host-sampled primaries at fixed slots.

    Mirror of :func:`generate_pencil_beam` for host-sampled sources (a phase space, or
    a user pre-sampled source): kind, energy, weight, position and direction come
    per-particle from uploaded arrays instead of analytic parameters. ``beamlet`` is
    the per-particle Dij column tag (all zero for a plain ``run``). The transport rng
    is seeded from the uploaded ``hist`` key (the history index for a plain run, the
    correlated-sampling key for a Dij), so it is a pure function of ``(seed, key)`` and
    independent of chunk boundaries — the host drew the position from a separate stream.
    """
    tid = wp.tid()
    q.kind[tid] = kind[tid]
    q.beamlet[tid] = beamlet[tid]
    q.primary[tid] = 1  # source particle: the only one that may split
    q.energy[tid] = energy[tid]
    q.weight[tid] = weight[tid]
    q.x[tid] = px[tid]
    q.y[tid] = py[tid]
    q.z[tid] = pz[tid]
    q.ux[tid] = dx[tid]
    q.uy[tid] = dy[tid]
    q.uz[tid] = dz[tid]
    q.rng[tid] = init_slot(seed, hist[tid])


@wp.kernel
def generate_from_exit_buffer(
    seed: int,
    sampling_seed: int,
    history_offset: int,
    count: int,
    particle_type: wp.array(dtype=wp.int32),
    b_energy: wp.array(dtype=float),
    b_x: wp.array(dtype=float),
    b_y: wp.array(dtype=float),
    b_z: wp.array(dtype=float),
    b_ux: wp.array(dtype=float),
    b_uy: wp.array(dtype=float),
    b_uz: wp.array(dtype=float),
    b_weight: wp.array(dtype=float),
    q_photon: Queue,
    q_electron: Queue,
    slots: wp.array(dtype=wp.uint32),
    emitted: wp.array(dtype=wp.int64),
):
    """Sample one exit-plane record per history from a device pre-solve buffer.

    The device-resident twin of the phase-space path: instead of a host
    ``sample_batch`` + :func:`generate_from_upload`, each history draws a record
    index into the on-device population from a stream keyed on ``sampling_seed``
    (distinct from the transport stream ``init_slot(seed, h)``), books the emitted
    weight-energy, and atomic-appends the record into the photon or electron queue
    as a source primary. Sampling is with replacement over the ``count`` records,
    the same contract as :class:`~pyRadMC.geometry.phasespace.InMemoryPhaseSpaceSource`,
    so the no-copy path and the host-handoff path estimate one dose. The pre-solve
    emits only IAEA photons (1) and electrons (2); there is no positron latent term.
    """
    tid = wp.tid()
    h = history_offset + tid
    slots[tid] = init_slot(seed, h)
    sampler = wp.rand_init(sampling_seed, h)
    k = int(wp.randf(sampler) * float(count))
    if k >= count:
        k = count - 1
    energy = b_energy[k]
    weight = b_weight[k]
    _escape(emitted, weight * energy)
    if particle_type[k] == 1:
        _push_source(
            q_photon,
            PHOTON,
            energy,
            weight,
            b_x[k],
            b_y[k],
            b_z[k],
            b_ux[k],
            b_uy[k],
            b_uz[k],
            slots[tid],
        )
    else:
        _push_source(
            q_electron,
            ELECTRON,
            energy,
            weight,
            b_x[k],
            b_y[k],
            b_z[k],
            b_ux[k],
            b_uy[k],
            b_uz[k],
            slots[tid],
        )


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
    q.beamlet[tid] = 0
    q.primary[tid] = 1  # source particle: the only one that may split
    q.energy[tid] = energy
    q.weight[tid] = 1.0
    q.x[tid] = x
    q.y[tid] = y
    q.z[tid] = z0
    q.ux[tid] = 0.0
    q.uy[tid] = 0.0
    q.uz[tid] = 1.0
    q.rng[tid] = slots[tid]


@wp.kernel
def generate_beamlet_lattice(
    seed: int,
    group_start: int,
    n_histories_per_beamlet: int,
    per_batch: int,
    batch: int,
    t_offset: int,
    correlated: int,
    energy: float,
    z0: float,
    x_lo: wp.array(dtype=float),
    x_extent: wp.array(dtype=float),
    y_lo: wp.array(dtype=float),
    y_extent: wp.array(dtype=float),
    slots: wp.array(dtype=wp.uint32),
    q: Queue,
):
    """Primaries for one chunk of *one batch* of a beamlet group, tagged by beamlet.

    Thread ``tid`` handles flat index ``t = t_offset + tid`` of this batch's block:
    group-local beamlet ``t // per_batch``, within-batch index ``i = t % per_batch``,
    and so within-beamlet index ``r = batch * per_batch + i``. The global history
    index follows the project-wide mapping fixed by ``ReferenceEngine.run_dij`` —
    ``h = j * n_per + r`` — so streams, and with them the whole Dij, are invariant
    to grouping, batch merging, and chunking (test-pinned).

    The column tag is ``local`` alone. Batches are separated *in time* (one drain
    per batch, folded into running float64 sums by ``accumulate_dij_batch``) rather
    than along a device axis, so the group's dense buffer costs
    ``group * n_voxels`` int64 instead of ``group * n_batches * n_voxels``. Each
    history still draws exactly the stream its ``(seed, key)`` names, so this is a
    pure scheduling change.

    Correlated sampling (Phase 4) keys the stream on ``r`` instead of ``h`` —
    see ``ReferenceEngine.run_dij`` for the mapping's definition and caveats.
    Neither ``r`` nor ``h`` depends on the grouping or chunking, so the
    scheduling bit-inertness above holds in both modes (test-pinned).

    The bounds arrays are per group-local beamlet, precomputed on the host by
    ``BeamletGridSource.beamlet_bounds`` — the lattice geometry has exactly one
    definition. Two uniforms per primary (x, then y), the same arithmetic as
    ``generate_parallel_beam``, so a 1x1 lattice is bit-identical to the open
    field on this device.
    """
    tid = wp.tid()
    t = t_offset + tid
    local = t // per_batch
    i = t - local * per_batch
    r = batch * per_batch + i
    h = (group_start + local) * n_histories_per_beamlet + r
    key = h
    if correlated != 0:
        key = r
    slots[tid] = init_slot(seed, key)
    state = WarpRNGState()
    state.slots = slots
    state.idx = tid
    x = x_lo[local] + x_extent[local] * uniform(state)
    y = y_lo[local] + y_extent[local] * uniform(state)
    q.kind[tid] = PHOTON
    q.beamlet[tid] = local
    q.primary[tid] = 1  # source photon: the only one that may split
    q.energy[tid] = energy
    q.weight[tid] = 1.0
    q.x[tid] = x
    q.y[tid] = y
    q.z[tid] = z0
    q.ux[tid] = 0.0
    q.uy[tid] = 0.0
    q.uz[tid] = 1.0
    q.rng[tid] = slots[tid]
