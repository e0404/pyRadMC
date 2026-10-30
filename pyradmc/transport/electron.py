"""Class II condensed-history electron (and positron) transport.

One particle at a time, for the reference backend. The scheme, per docs/decisions.md:

- **Continuous** energy loss along each substep from the restricted collision
  stopping power (Berger-Seltzer, cut at ECUT), deposited at the two segment
  midpoints around the multiple-scattering hinge.
- **Discrete Moller events** above ECUT, distance-sampled with the restricted Moller
  cross-section; the delta ray is stacked, both outgoing directions from exact
  kinematics with back-to-back azimuths.
- **Random hinge** multiple scattering (PENELOPE-style; Salvat et al.,
  PENELOPE-2018, sec. 4.3, doi:10.1787/32da5043-en): the deflection for the whole
  substep is applied at a uniformly random point within it. The angular law is
  **Goudsmit-Saunderson** by default (exact for arbitrary step length, sampled
  from precomputed tables anchored to the scattering power; the shipped
  configuration since 2026-07-23); the Gaussian Fermi-Eyges hinge it replaced
  survives as the ``msc_model="gaussian"`` test instrument.
- **Discrete thin-target bremsstrahlung** (see :mod:`pyradmc.physics.brems`): at most
  one photon per substep, emitted forward from the hinge point, Bernoulli probability
  matched to the radiative stopping power; the sub-PCUT remainder joins the
  continuous deposit.
- **Positrons are transported as electrons** (no Bhabha distinction — a stated,
  maintainer-approved approximation) and annihilate at rest into two back-to-back
  511 keV photons wherever they stop; a positron leaving the grid carries its
  annihilation energy with it.
- **Static magnetic field** (:mod:`pyradmc.physics.lorentz`; Bielajew 1988,
  doi:10.1007/978-1-4613-1059-4_19; Malkov & Rogers 2016, doi:10.1118/1.4954318):
  the field is sampled once per substep and each of the hinge's two straight chords
  is taken along the direction at its own midpoint — rotate about B by half the
  chord's gyration angle, drift, rotate by the other half — with the gyroradius at
  the chord's mid-energy. The direction is then exact and the end point within
  ``s theta^2/24`` of the arc. Multiple scattering and the field are decoupled within
  a substep, as in every production code; the decoupling error is bounded by the
  gyration-angle cap :data:`LORENTZ_STEP_ANGLE_MAX`. Outside the grid there is no
  field: a charged particle born outside flies straight to the entry face.
- **Long exact-helix steps in exterior air**, a named approximation, taken only when the
  engine holds a map (:mod:`pyradmc.geometry.exterior`) under a uniform field, in a voxel
  whose safety distance is non-zero. The substep is then limited by the energy fraction
  and that safety alone: no gyration cap, no voxel face. Path bounds displacement, so
  the whole step stays in the exterior. Each chord is the exact helix
  (:func:`~pyradmc.physics.lorentz.helix_chord`) at the chord's mid-energy radius. Hinge,
  loss, Moller, brems and the ledger are the ordinary loop's. What is approximate, and
  only here: **the deposit of a long step is not spatially resolved** (it files at the two
  chord midpoints, which lie in exterior voxels and never in the body, or along the
  straight chords under ``deposit_resolution_cm``, never along the arc), and **the whole
  step is charged at its start voxel's density and material**, though it may cross
  exterior voxels of other densities below the region's bound. Rotation draws no random
  number, and the hinge decouples from the field exactly as above, over a larger angle.

Substeps are limited four ways: a fraction of the CSDA range (energy-loss
accuracy), a mean-square hinge-angle ceiling (multiple-scattering accuracy — the
angular cap that replaced the historic half-voxel-edge limit; see
:data:`STEP_HINGE_THETA2_MAX`), a gyration-angle ceiling in a magnetic field
(:data:`LORENTZ_STEP_ANGLE_MAX`), and truncation at the next voxel face
(heterogeneity: a substep's density and material are those of the voxel it
runs in — exactly so without a field, and to within the chord's bend with one).
A long exterior-air step (above) replaces the last two with its safety distance.
The reference backend keeps the caps conservative; the Warp backend
mirrors this loop step for step and is validated against it — step aggressiveness
beyond it must be bought with better transport mechanics, not by loosening the
oracle.

Electrons at or below ECUT deposit their kinetic energy locally and terminate.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pyradmc import (
    ELECTRON_MASS_MEV,
    PHOTON_ROULETTE_MEV,
    PHOTON_ROULETTE_SURVIVAL,
    PHOTON_ROULETTE_WEIGHT_CAP,
)
from pyradmc.data.interface import CrossSectionSource
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import (
    VoxelGrid,
    distance_to_voxel_boundary,
    nudge_across_face,
    voxel_face_axis,
)
from pyradmc.physics.brems import bremsstrahlung_step_parameters, sample_bremsstrahlung_energy
from pyradmc.physics.direction import rotate_direction
from pyradmc.physics.lorentz import (
    gyroradius_cm,
    helix_chord,
    lorentz_rotation_angle,
    rotate_about_axis,
)
from pyradmc.physics.moller import moller_direction_cosines, sample_moller_delta_energy
from pyradmc.physics.msc import sample_hinge_cos_theta
from pyradmc.physics.path import sample_path_length
from pyradmc.physics.roulette import roulette_weight
from pyradmc.rng import RNGState, uniform
from pyradmc.transport.particles import (
    ELECTRON,
    PHOTON,
    DepositFn,
    DepositWeightFn,
    SpawnFn,
    annihilate_at_rest,
    unit_weight,
)

__all__ = [
    "BOUNDARY_NUDGE_CM",
    "GS_STEP_ENERGY_FRACTION",
    "LORENTZ_STEP_ANGLE_MAX",
    "STEP_ENERGY_FRACTION",
    "STEP_HINGE_THETA2_MAX",
    "SUBSTEP_DEPOSIT_CAP",
    "default_step_energy_fraction",
    "electron_steps",
    "substep_deposit_count",
    "substep_pieces",
]

STEP_ENERGY_FRACTION: float = 0.05
"""Maximum fraction of the CSDA range per substep under the Gaussian instrument.

Bounds the relative energy change per substep so that evaluating the stopping power,
Moller cross-section and scattering power at the substep's *initial* energy stays a
sub-percent approximation. Conservative on purpose, and frozen at the value the
Gaussian hinge was validated at: the retained instrument must keep meaning what it
meant when every paired GS-vs-Gaussian comparison was made against it.
"""

GS_STEP_ENERGY_FRACTION: float = 0.20
"""Maximum fraction of the CSDA range per substep under Goudsmit-Saunderson.

The shipped configuration. With the exact angular law the Gaussian-validity cap
does not apply, and 0.20 was validated as a package (2026-07-21/22): 3.07x fewer
substeps on the realistic 3 mm schedule with zero significant depth-drift bins,
R50/detour and PDD-gamma gates passing, +0.23 percent end-to-end on the tabulated
6 MV spectral thorax, and cross-backend chi-squared on both backends. Larger
fractions were measured to need path-length corrections that 0.20 does not.
"""


def default_step_energy_fraction(msc_model: str) -> float:
    """Resolve the per-model substep-fraction default.

    The two knobs are coupled: each angular model is validated *at* its
    fraction, so an unspecified ``step_energy_fraction`` follows the model —
    :data:`GS_STEP_ENERGY_FRACTION` for the shipped ``"gs"`` configuration,
    :data:`STEP_ENERGY_FRACTION` for the ``"gaussian"`` instrument. A shared
    default would silently run one model at the other's schedule.
    """
    return GS_STEP_ENERGY_FRACTION if msc_model == "gs" else STEP_ENERGY_FRACTION


STEP_HINGE_THETA2_MAX: float = 0.10
"""Maximum mean-square hinge deflection per substep, in rad^2. Gaussian model only.

The Gaussian random-hinge model's validity is *angular*, so the limiter is
angular: the substep is capped at ``theta2_max / (T(E) rho)`` with the mass
scattering power ``T``, bounding each hinge's accumulated ``<theta^2>``
directly. Because ``T ~ 1/E^2`` outruns the shrinking range, a
fraction-of-range cap alone lets hinge angles *grow* as the electron slows —
the low-energy under-ranging the R50 validation gate caught when the historic
half-voxel-edge cap (whose protection was an accident of voxel size; see the
a6b6c07 revert) was removed without a replacement.

**Where it binds.** Measured 2026-07-20 under the Highland scattering power
then in use: at the ``STEP_ENERGY_FRACTION`` of 0.05 the cap was *inert* in
every soft tissue (``s_theta / s_E >= 1.02`` pointwise over 0.2-20 MeV in
water, air, lung and adipose; the ratio is density-independent — it is a
property of the medium), binding first only in cortical bone below ~2.5 MeV.
With the Class-II transport-moment scattering power (2026-09),
``s_theta / s_E`` in water is 0.96 at 0.2 MeV, 1.01 at 0.5 MeV and rises
steeply above (about 4.8 at 10 MeV), so the angular cap binds only at the soft
end of the transported range. It re-binds at higher energies only for larger
fractions; the exact crossover is energy-dependent.

**Not applied under ``msc_model="gs"`` — the shipped default**:
Goudsmit-Saunderson samples the exact multiple-scattering angle for an
arbitrary step length, so the Gaussian-validity limit does not apply and the
energy-limited step stands (validated with the cap lifted: R50/detour and
PDD-gamma gates, and the tabulated 6 MV spectral thorax at fraction 0.20).
The cap therefore protects only the ``"gaussian"`` instrument path. The value
is validated by the R50/R_CSDA detour-factor gates; do not change it without
rerunning them.
"""

_ENTRY_NUDGE_CM = 1.0e-9  # same convention as the photon loop

LORENTZ_STEP_ANGLE_MAX: float = 0.2
"""Largest gyration angle (radians) a charged-particle substep may turn through.

In a static magnetic field every substep is also capped at ``s_B = 0.2 r_g`` with
``r_g`` the gyroradius at the substep's start energy (:mod:`pyradmc.physics.lorentz`).
The cap is what keeps the two straight chords of the random hinge close to the
gyration arc: each chord is taken along the direction at its own midpoint (rotate by
half the chord's angle, drift, rotate by the other half), which places its end point
within ``s theta^2 / 24`` of the exact arc and leaves the direction exact, so the
residual is second order in the angle. The value is the one Malkov & Rogers (2016,
Med. Phys. 43, 4447, doi:10.1118/1.4954318) validated for their higher-order EGSnrc
scheme: it passes the Fano test in a field to 0.1 % at ``delta_u = 0.2`` and stays
stable to 0.4, where the historical first-order scheme needed 0.02 at two to three
times the cost.

Where it binds: in water at 1.5 T, ``s_B`` is 0.63 mm at 1 MeV against the 0.9 mm
energy-limited GS step, so it shortens substeps below roughly 1.5 MeV; at 0.35 T it is
inert in tissue. In air the energy-limited step is metres and the voxel face bound the
substep before; now ``s_B`` does, so air costs several substeps per voxel. An
:class:`~pyradmc.geometry.exterior.ExteriorAirMap` lifts it in air far from the body,
where the chords are exact instead; it is not relaxed anywhere else.

Accuracy-defining (AGENTS.md 2.8): changing it needs the Fano-in-field gate and a
measurement on realistic spectral, heterogeneous geometry, never water alone.
"""

BOUNDARY_NUDGE_CM: float = 1.0e-4
"""Over-step past a voxel face when a substep is boundary-limited, in cm.

Added to the distance returned by
:func:`pyradmc.geometry.grid.distance_to_voxel_boundary` so a boundary-capped substep
lands just inside the next voxel rather than exactly on the face, where the half-open
lower-inclusive convention could otherwise trap an inward-facing electron at zero
distance. Unlike the entry nudge (which differs by backend for float32 reasons), this
is shared verbatim with the Warp kernel so both backends truncate at identical
positions; 1 micrometre is far below any voxel edge, and its short over-step is charged
to the crossed voxel's stopping power, so it is physically consistent, not a leak."""


SUBSTEP_DEPOSIT_CAP = 1024
"""Upper bound on pieces per half-substep, so a mis-specified resolution cannot
explode the deposit count. Past it the deposit is coarser than asked — a resolution
shortfall, not a correctness failure."""


def substep_pieces(length: float, resolution: float) -> int:
    """Pieces to split a half-substep of ``length`` cm into; ``resolution <= 0`` gives 1.

    ``ceil(length / resolution)``, clamped to ``[1, SUBSTEP_DEPOSIT_CAP]``. Ceil, not
    round: a piece *longer* than the requested resolution is the failure this exists
    to prevent, so the count errs upward.

    Non-positive ``resolution`` is the "one midpoint deposit" sentinel rather than an
    error, because this is the form the Warp kernels compile (AGENTS.md 2.5) and a
    kernel cannot raise; the host API's ``None`` maps onto it, and the engines
    validate a caller's value before any launch. Keeping one function means host and
    device cannot disagree about how many deposits a step becomes.
    """
    if resolution <= 0.0:
        return 1
    # math.ceil returns an int on the host; under Warp it is a float ceil, and the
    # cast is what makes both return an index (as in point_axis_index).
    n = int(math.ceil(length / resolution))  # noqa: RUF046
    if n < 1:
        return 1
    if n > SUBSTEP_DEPOSIT_CAP:
        return SUBSTEP_DEPOSIT_CAP
    return n


def substep_deposit_count(length: float, resolution: float | None) -> int:
    """Host wrapper over :func:`substep_pieces`: ``None`` means one midpoint deposit.

    Refuses a non-positive number, which is a caller error rather than a request for
    the default; ``None`` is how the default is spelled.
    """
    if resolution is None:
        return 1
    if resolution <= 0.0:
        raise ValueError(f"deposit resolution must be positive, got {resolution}")
    return substep_pieces(length, resolution)


def electron_steps(
    is_positron: bool,
    energy: float,
    weight: float,
    x: float,
    y: float,
    z: float,
    ux: float,
    uy: float,
    uz: float,
    grid: VoxelGrid,
    cross_sections: CrossSectionSource,
    rng_state: RNGState,
    deposit: DepositFn,
    spawn: SpawnFn,
    pcut: float,
    ecut: float,
    deposit_weight: DepositWeightFn = unit_weight,
    step_energy_fraction: float | None = None,
    msc_model: str = "gs",
    deposit_resolution_cm: float | None = None,
    field: MagneticField | None = None,
    exterior_safety: npt.NDArray[np.float64] | None = None,
) -> float:
    """Transport one electron or positron; secondaries go to ``spawn``.

    Returns the weight-energy removed from the transported population: physical
    escape (for a positron including the 2 m_e c^2 of latent annihilation energy
    it carries out) plus the net ledger effect of rouletting sub-threshold
    bremsstrahlung photons at birth (see :mod:`pyradmc.transport.photon`). The
    electron itself never plays; its ``weight`` is constant and scales every
    deposit, so the engine's exact energy balance extends over this loop
    unchanged.

    ``deposit_weight`` (dose-to-water SPR; default dose-to-medium) is evaluated
    once per substep at the substep's *initial* energy and voxel material — the
    same (E, material) the restricted stopping power charged the continuous loss
    with, so the conversion undoes exactly the weighting the medium applied —
    and at the cutoff-clamped energy for local sub-threshold deposits.

    ``msc_model`` selects the multiple-scattering angular law: ``"gs"``
    (default — Goudsmit-Saunderson, the exact law for an arbitrary step
    length, adopted as the shipped configuration 2026-07-23 after the
    validation record in the R50/detour, PDD-gamma and tabulated-thorax
    gates) or ``"gaussian"`` (the small-angle random hinge it replaced,
    retained as the AGENTS.md 2.10 **test instrument**: a paired comparison
    against it isolates the angular model while the rest of the physics is
    held fixed, which is how the GS adoption itself was measured). Both
    consume exactly one uniform, so the random stream does not shift between
    them. Under ``"gs"`` the angular cap (:data:`STEP_HINGE_THETA2_MAX`) is
    **not applied** — it is a Gaussian-validity limit, and removing it is
    what lets the larger GS substep fraction actually grow the step.

    ``deposit_resolution_cm`` is the longest piece a half-substep's continuous
    energy loss may be filed as. ``None`` (the default) files the whole half-step
    as one point deposit at its midpoint — the behaviour every existing result was
    produced with, byte-identical. A value splits the half-step into equal pieces
    no longer than it and deposits an equal share at each piece's midpoint.

    This changes *where* energy is filed, never how much, and never the trajectory
    or the random stream: the substep length, the hinge, the scattering and the
    energy loss are all untouched, so a run with it set transports exactly the same
    histories. It matters only when dose is scored below the transport voxel scale,
    where the midpoint deposit prints the voxel lattice onto the dose profile (see
    the module tests). Set it to the finest bin the scorer resolves; leave it None
    when scoring at voxel resolution, where it would only cost time.

    ``step_energy_fraction`` overrides the per-model default
    (:func:`default_step_energy_fraction`; ``None`` resolves to the selected
    model's validated fraction) — a measurement instrument for substep
    resolution studies. Changing a *default* is a maintainer decision gated
    on the validation tier, not a knob turn.

    ``field`` is the static magnetic field the particle moves in, or ``None`` for
    the field-free world. It is world data handed down from the engine, never a
    per-run option (AGENTS.md 2.10). The Lorentz force does no work, so it enters
    only through the direction of each chord and the step cap that keeps the chord
    close to the gyration arc; the field-free path executes no field operation and
    draws no random number. An explicit zero field therefore reproduces ``None``
    byte for byte; the separate boundary and loss fixes can change historical
    field-free results (see CHANGELOG.md).

    ``exterior_safety`` is the per-voxel safety of an
    :class:`~pyradmc.geometry.exterior.ExteriorAirMap`, or ``None`` for the current
    stepping; the engines pass it only under a uniform non-zero field
    (:func:`~pyradmc.geometry.exterior.resolve_exterior_safety`). Where it is positive the
    substep is the long exact-helix step of the module docstring.
    """
    if msc_model not in ("gaussian", "gs"):
        raise ValueError(f"unknown msc_model {msc_model!r}; expected 'gaussian' or 'gs'")
    if step_energy_fraction is None:
        step_energy_fraction = default_step_energy_fraction(msc_model)
    if deposit_resolution_cm is not None and deposit_resolution_cm <= 0.0:
        raise ValueError(f"deposit resolution must be positive, got {deposit_resolution_cm}")
    # One float carries the option from here down, matching what the kernels take:
    # non-positive is the "single midpoint deposit" sentinel.
    resolution = 0.0 if deposit_resolution_cm is None else deposit_resolution_cm
    escaped = 0.0
    e = energy
    w = weight
    latent = 2.0 * ELECTRON_MASS_MEV if is_positron else 0.0
    # Path already walked in the current voxel that no substep has charged yet: the
    # nudge a face crossing carries the particle in by. It is charged here, at this
    # voxel's density, by the first substep taken in the voxel (see the hinge).
    carry = 0.0

    # Vacuum flight for a primary born outside the phantom (electron beams).
    if not grid.contains(x, y, z):
        t = grid.distance_to_entry(x, y, z, ux, uy, uz)
        if math.isinf(t):
            return w * (e + latent)
        t += _ENTRY_NUDGE_CM
        x += t * ux
        y += t * uy
        z += t * uz
        if not grid.contains(x, y, z):
            return w * (e + latent)

    def deposit_or_escape(px: float, py: float, pz: float, amount: float, factor: float) -> float:
        """Deposit at a point if it is inside the transport grid; else escaped.

        The scoring callback receives the position, not a voxel index: routing
        into the (possibly decoupled) scoring grid is the scorer's job.
        ``factor`` is the caller's deposit weight; escape stays physical.
        """
        if amount <= 0.0:
            return 0.0
        if grid.contains(px, py, pz):
            deposit(px, py, pz, amount, factor * amount)
            return 0.0
        return amount

    def deposit_spread(
        px: float,
        py: float,
        pz: float,
        dx: float,
        dy: float,
        dz: float,
        length: float,
        amount: float,
        factor: float,
    ) -> float:
        """File one half-substep's loss along its path; returns escaped energy.

        ``(px, py, pz)`` is the half-step's *start* and ``length`` its length. The
        pieces are equal, and each piece's share is deposited at its midpoint, so a
        single piece reproduces the plain midpoint deposit exactly (``length / 1.0``
        is exact and ``0.5 * length`` equals ``length / 2.0`` in IEEE arithmetic).
        """
        n = substep_pieces(length, resolution)
        share = amount / float(n)
        piece = length / float(n)
        out = 0.0
        for k in range(n):
            t = (float(k) + 0.5) * piece
            out += deposit_or_escape(px + dx * t, py + dy * t, pz + dz * t, share, factor)
        return out

    while True:
        if e <= ecut:
            # Terminal: local deposit; the current position is inside the grid.
            ix, iy, iz = grid.voxel_index(x, y, z)
            terminal_material = int(grid.material[ix, iy, iz])
            deposit(x, y, z, w * e, deposit_weight(e, terminal_material) * w * e)
            if is_positron:
                annihilate_at_rest(
                    x,
                    y,
                    z,
                    rng_state,
                    deposit,
                    spawn,
                    pcut,
                    w,
                    deposit_weight(ELECTRON_MASS_MEV, terminal_material),
                )
            return escaped

        ix, iy, iz = grid.voxel_index(x, y, z)
        rho = float(grid.density[ix, iy, iz])
        material = int(grid.material[ix, iy, iz])

        # --- magnetic field at the substep start -----------------------------------
        # Sampled once per substep and held over both chords (with a face-capped
        # substep a per-voxel map is then exact). A zero vector is the field-free
        # path: no cap, no rotation, no random number — byte-identical to ``None``.
        has_field = False
        b_mag = 0.0
        b_ux = 0.0
        b_uy = 0.0
        b_uz = 0.0
        if field is not None:
            bx, by, bz = field.sample(x, y, z)
            b2 = bx * bx + by * by + bz * bz
            if b2 > 0.0:
                b_mag = math.sqrt(b2)
                b_ux = bx / b_mag
                b_uy = by / b_mag
                b_uz = bz / b_mag
                has_field = True

        # Long exact-helix step in exterior air: positive means the whole of a path this
        # long from here is exterior (pyradmc.geometry.exterior). Zero is the current
        # stepping, and a field-free substep never reads the map.
        exterior = 0.0
        if has_field and exterior_safety is not None:
            exterior = float(exterior_safety[ix, iy, iz])

        # --- substep length --------------------------------------------------------
        range_cm = cross_sections.csda_range(e, material) / rho
        s_max = step_energy_fraction * range_cm
        if msc_model != "gs":
            # The angular cap is a *Gaussian-validity* limit (see
            # STEP_HINGE_THETA2_MAX): it exists because the small-angle hinge is
            # wrong at large per-step angles. Goudsmit-Saunderson is exact at
            # arbitrary angle, so under it the energy-limited step stands
            # (maintainer decision 2026-07-21; validated with the cap lifted on
            # the R50/detour and PDD-gamma gates and the tabulated spectral
            # thorax). The Gaussian branch is unchanged to the bit.
            s_theta = STEP_HINGE_THETA2_MAX / (
                cross_sections.scattering_power(e, material, ecut) * rho
            )
            s_max = min(s_max, s_theta)
        if has_field:
            # Curvature cap (LORENTZ_STEP_ANGLE_MAX): bound the gyration angle per
            # substep so each straight chord stays close to the arc it stands for.
            if exterior > 0.0:
                # The safety replaces the curvature cap: the chords below are exact at
                # any angle, so the cap has nothing left to protect.
                s_max = min(s_max, exterior)
            else:
                s_max = min(s_max, LORENTZ_STEP_ANGLE_MAX * gyroradius_cm(e, b_mag))
        # Cap at the next voxel face (plus the nudge across it) so the substep's
        # density and material stay those of the voxel it starts in. The distance
        # is taken along the pre-rotation direction: a bent chord can end up to
        # s*theta/2 (at most delta_u^2 r_g / 2, tens of microns at 1 MeV, 1.5 T)
        # off the straight-line prediction, so under a field the one-voxel
        # invariant holds to that tolerance and the next substep re-reads its
        # voxel — the stated approximation of this scheme.
        # A long step has no face cap: it crosses faces between exterior voxels freely.
        s_boundary = math.inf
        if exterior <= 0.0:
            s_boundary = (
                distance_to_voxel_boundary(
                    x, y, z, ux, uy, uz, *grid.origin, *grid.spacing, *grid.shape
                )
                + BOUNDARY_NUDGE_CM
            )
        s_geometry = min(s_max, s_boundary)
        sigma_moller = rho * cross_sections.moller_cross_section(e, material, ecut)
        s_interaction = (
            sample_path_length(sigma_moller, rng_state) if sigma_moller > 0.0 else math.inf
        )
        s = min(s_interaction, s_geometry)
        # A Moller event fires only if the sampled flight is the actual limiter — a
        # geometry-truncated substep ends at the boundary with no interaction.
        moller_pending = s_interaction <= s_geometry

        # The face cap ends the substep a nudge past the face. That micron is a
        # geometric device for crossing, not path in this voxel: the chords split
        # and the loss covers the path *inside* this voxel, ``s_in``, plus any
        # ``carry`` — the nudge the particle entered this voxel by, charged now at
        # this voxel's density. (Charging the nudge at the old voxel's density let
        # electrons that range out in the last micron before an air face be carried
        # into the air with a sliver above ECUT and terminate there at water-density
        # rate: the last 10 % of the Fano violation in a 2 mm air gap. Splitting the
        # full s let the first chord overshoot the face on short final steps.)
        face_limited = s == s_boundary
        s_in = s - BOUNDARY_NUDGE_CM if face_limited else s
        crossing = False

        # --- continuous loss over the substep (exact, DPM-style) --------------------
        # E_end = r^-1(r(E) - rho*path) with the restricted-collision range, replacing
        # the first-order S(E_start)*rho*path linearization; the range-out shortening
        # is exact by the same map. Sempau et al. 2000, doi:10.1088/0031-9155/45/8/315.
        path = s_in + carry
        available = cross_sections.restricted_range(e, material, ecut)
        if rho * path >= available:
            # The electron ranges out inside this substep, in this voxel.
            s_in = max(available / rho - carry, 0.0)
            path = s_in + carry
            de = e - ecut
            moller_pending = False
        else:
            de = e - cross_sections.energy_after_mass_path(e, material, ecut, rho * path)
        emit_probability, local_brems = bremsstrahlung_step_parameters(
            e, cross_sections.radiative_stopping_power(e, material), rho, path, pcut
        )
        continuous = de + local_brems
        emit_brems = uniform(rng_state) < emit_probability
        # Dose-to-water: one factor per substep, at the (E, material) that set the
        # stopping power above; 1.0 exactly under the dose-to-medium default.
        substep_factor = deposit_weight(e, material)

        # --- random hinge: move, deflect, move --------------------------------------
        # A chord that reaches a face stops on it and the particle is carried across
        # afterwards (or left just inside if terminal there). The carry's share of
        # the loss rides with the first chord: it was walked just behind it.
        s1 = uniform(rng_state) * s_in
        s2 = s_in - s1
        d1 = continuous * ((s1 + carry) / path) if path > 0.0 else 0.0
        d2 = continuous - d1
        carry = 0.0

        # Under a field each chord is taken along the direction at its midpoint:
        # rotate by half the chord's gyration angle, drift, rotate by the other
        # half. The gyroradius is read at the chord's mid-energy (the hinge already
        # knows the chord's share of the loss), which is the energy half of the
        # midpoint rule. The Lorentz force does no work, so the loss is untouched.
        half = 0.0
        cut = False
        if has_field:
            half = 0.5 * lorentz_rotation_angle(s1, gyroradius_cm(e - 0.5 * d1, b_mag), is_positron)
            ux, uy, uz = rotate_about_axis(ux, uy, uz, b_ux, b_uy, b_uz, half)
            if exterior <= 0.0:
                # The face cap was taken along the pre-rotation direction; the bent
                # chord may reach a face sooner. Stop there and end the substep (the
                # PENELOPE rule for an interface met before the hinge): the remainder
                # is transported from the new voxel with that voxel's medium.
                s1_face = distance_to_voxel_boundary(
                    x, y, z, ux, uy, uz, *grid.origin, *grid.spacing, *grid.shape
                )
                if s1 >= s1_face:
                    d1 = d1 * (s1_face / s1) if s1 > 0.0 else 0.0
                    s1 = s1_face
                    cut = True
                    axis = voxel_face_axis(
                        x, y, z, ux, uy, uz, *grid.origin, *grid.spacing, *grid.shape
                    )
                    # The first half-turn was the full chord's; make the whole turn the
                    # cut chord's, so direction and path stay on the same arc.
                    half = (
                        lorentz_rotation_angle(s1, gyroradius_cm(e - 0.5 * d1, b_mag), is_positron)
                        - half
                    )
        # A chord runs along the direction at its midpoint; a long step's is shortened
        # to the true helix displacement, whose midpoint is inside the safety ball, so
        # its deposit lands in an exterior voxel and never in the body.
        cx, cy, cz = ux, uy, uz
        if exterior > 0.0:
            cx, cy, cz = helix_chord(ux, uy, uz, b_ux, b_uy, b_uz, half)
        escaped += deposit_spread(x, y, z, cx, cy, cz, s1, w * d1, substep_factor)
        e -= d1
        x += cx * s1
        y += cy * s1
        z += cz * s1
        if has_field:
            ux, uy, uz = rotate_about_axis(ux, uy, uz, b_ux, b_uy, b_uz, half)
        if cut:
            # On the face: carry across, or stop just inside if terminal. The
            # micron walked across is charged by the next substep, in that voxel.
            # Along the face *normal*: along the direction, a grazing crossing moves
            # the coordinate by less than a float32 ulp and the particle never
            # leaves the face (see ``voxel_face_axis``).
            nudge = -BOUNDARY_NUDGE_CM if e <= ecut else BOUNDARY_NUDGE_CM
            dx, dy, dz = nudge_across_face(axis, ux, uy, uz, nudge)
            x += dx
            y += dy
            z += dz
            carry = BOUNDARY_NUDGE_CM
            if not grid.contains(x, y, z):
                return escaped + w * (e + latent)
            continue
        if not grid.contains(x, y, z):
            return escaped + w * (e + latent)

        mean_square = cross_sections.scattering_power(e, material, ecut) * rho * path
        # Both laws take the same <theta^2> and one uniform; see ``msc_model``.
        if msc_model == "gs":
            cos_hinge = cross_sections.sample_gs_cos_theta(mean_square, e, material, rng_state)
        else:
            cos_hinge = sample_hinge_cos_theta(mean_square, rng_state)
        phi = 2.0 * math.pi * uniform(rng_state)
        ux, uy, uz = rotate_direction(ux, uy, uz, cos_hinge, phi)

        if emit_brems and e > pcut:
            k = sample_bremsstrahlung_energy(e, pcut, rng_state)
            # Rare tail: the sampled photon would overdraw the energy still owed to
            # the second continuous deposit. Cap it; a capped photon at or below
            # PCUT is not transportable and deposits at the hinge instead.
            k = min(k, e - d2)
            if k > pcut:
                photon_w = w
                if k < PHOTON_ROULETTE_MEV and w < PHOTON_ROULETTE_WEIGHT_CAP:
                    # Basic VR: roulette the soft bremsstrahlung photon at birth;
                    # the ledger entry keeps the run's energy balance exact.
                    photon_w = roulette_weight(w, PHOTON_ROULETTE_SURVIVAL, rng_state)
                    escaped += (w - photon_w) * k
                if photon_w > 0.0:
                    spawn((PHOTON, k, photon_w, x, y, z, ux, uy, uz))  # forward
            else:
                escaped += deposit_or_escape(x, y, z, w * k, deposit_weight(k, material))
            e -= k

        half = 0.0
        if has_field:
            half = 0.5 * lorentz_rotation_angle(s2, gyroradius_cm(e - 0.5 * d2, b_mag), is_positron)
            ux, uy, uz = rotate_about_axis(ux, uy, uz, b_ux, b_uy, b_uz, half)
        # The hinge turned the particle; the face cap was taken along the old
        # direction, so the second chord can cross into the neighbouring voxel and
        # would carry this voxel's energy loss (and a range-out termination) into a
        # medium that may be a thousand times lighter. Stop at the face instead:
        # the loss along the shortened chord scales with its length, the untaken
        # part stays kinetic, and a Moller event sampled beyond the face is simply
        # not reached (the flight is re-sampled next substep; memoryless). This is
        # the second-leg interface rule of the random hinge (Salvat et al.,
        # PENELOPE-2018, sec. 4.3) and it closes the Fano test at air contrast.
        # A long step has no interface to stop at: the safety holds for the whole path.
        cx, cy, cz = ux, uy, uz
        if exterior > 0.0:
            cx, cy, cz = helix_chord(ux, uy, uz, b_ux, b_uy, b_uz, half)
        else:
            s2_face = distance_to_voxel_boundary(
                x, y, z, ux, uy, uz, *grid.origin, *grid.spacing, *grid.shape
            )
            if s2 >= s2_face or (face_limited and s2_face - s2 <= BOUNDARY_NUDGE_CM):
                # A face-limited substep ends on its face unless the hinge turned the
                # particle away from it by more than a nudge. The exact test alone is
                # not enough: a chord split leaves the particle a rounding residue
                # short of the face, the next substep then has no in-voxel path at all
                # and the exact test is false for ever, so the particle never moves.
                d2 = d2 * (s2_face / s2) if s2 > 0.0 else 0.0
                s2 = s2_face
                moller_pending = False
                crossing = True
                axis = voxel_face_axis(
                    x, y, z, ux, uy, uz, *grid.origin, *grid.spacing, *grid.shape
                )
                if has_field:
                    half = (
                        lorentz_rotation_angle(s2, gyroradius_cm(e - 0.5 * d2, b_mag), is_positron)
                        - half
                    )
        escaped += deposit_spread(x, y, z, cx, cy, cz, s2, w * d2, substep_factor)
        e -= d2
        x += cx * s2
        y += cy * s2
        z += cz * s2
        if has_field:
            ux, uy, uz = rotate_about_axis(ux, uy, uz, b_ux, b_uy, b_uz, half)
        if crossing:
            # On the face: carry across (along the normal, see above), or stop just
            # inside if terminal; the micron is charged by the next substep there.
            nudge = -BOUNDARY_NUDGE_CM if e <= ecut else BOUNDARY_NUDGE_CM
            dx, dy, dz = nudge_across_face(axis, ux, uy, uz, nudge)
            x += dx
            y += dy
            z += dz
            carry = BOUNDARY_NUDGE_CM
        if not grid.contains(x, y, z):
            return escaped + w * (e + latent)

        # --- discrete Moller event at the end of the substep -------------------------
        if moller_pending and e > 2.0 * ecut:
            delta_energy = sample_moller_delta_energy(e, ecut, rng_state)
            cos_delta, cos_primary = moller_direction_cosines(e, delta_energy)
            phi = 2.0 * math.pi * uniform(rng_state)
            delta_dir = rotate_direction(ux, uy, uz, cos_delta, phi)
            spawn((ELECTRON, delta_energy, w, x, y, z, *delta_dir))
            ux, uy, uz = rotate_direction(ux, uy, uz, cos_primary, phi + math.pi)
            e -= delta_energy
