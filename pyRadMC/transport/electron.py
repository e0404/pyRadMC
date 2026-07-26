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
- **Discrete thin-target bremsstrahlung** (see :mod:`pyRadMC.physics.brems`): at most
  one photon per substep, emitted forward from the hinge point, Bernoulli probability
  matched to the radiative stopping power; the sub-PCUT remainder joins the
  continuous deposit.
- **Positrons are transported as electrons** (no Bhabha distinction — a stated,
  maintainer-approved approximation) and annihilate at rest into two back-to-back
  511 keV photons wherever they stop; a positron leaving the grid carries its
  annihilation energy with it.

Substeps are limited three ways: a fraction of the CSDA range (energy-loss
accuracy), a mean-square hinge-angle ceiling (multiple-scattering accuracy — the
angular cap that replaced the historic half-voxel-edge limit; see
:data:`STEP_HINGE_THETA2_MAX`), and truncation at the next voxel face
(heterogeneity: a substep's density and material are exactly those of the voxel
it runs in). The reference backend keeps the caps conservative; the Warp backend
mirrors this loop step for step and is validated against it — step aggressiveness
beyond it must be bought with better transport mechanics, not by loosening the
oracle.

Electrons at or below ECUT deposit their kinetic energy locally and terminate.
"""

from __future__ import annotations

import math

from pyRadMC import (
    ELECTRON_MASS_MEV,
    PHOTON_ROULETTE_MEV,
    PHOTON_ROULETTE_SURVIVAL,
    PHOTON_ROULETTE_WEIGHT_CAP,
)
from pyRadMC.data.interface import CrossSectionSource
from pyRadMC.geometry.grid import VoxelGrid, distance_to_voxel_boundary
from pyRadMC.physics.brems import bremsstrahlung_step_parameters, sample_bremsstrahlung_energy
from pyRadMC.physics.direction import rotate_direction
from pyRadMC.physics.moller import moller_direction_cosines, sample_moller_delta_energy
from pyRadMC.physics.msc import sample_hinge_cos_theta
from pyRadMC.physics.path import sample_path_length
from pyRadMC.physics.roulette import roulette_weight
from pyRadMC.rng import RNGState, uniform
from pyRadMC.transport.particles import (
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
    "STEP_ENERGY_FRACTION",
    "STEP_HINGE_THETA2_MAX",
    "default_step_energy_fraction",
    "electron_steps",
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

**Where it binds — measured 2026-07-20, correcting an earlier claim here.** At
the shipped ``STEP_ENERGY_FRACTION`` of 0.05 this cap is *inert* in every soft
tissue: ``s_theta / s_E >= 1.02`` pointwise over 0.2-20 MeV in water, air, lung
and adipose (the ratio is density-independent — it is a property of the
medium), so the energy cap binds first everywhere except cortical bone below
~2.5 MeV. The cap's protection matters for *fractions above* ~0.057, where it
re-binds and clamps the effective step — the reason raising the fraction alone
was measured to buy almost nothing under the Gaussian model.

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

BOUNDARY_NUDGE_CM: float = 1.0e-4
"""Over-step past a voxel face when a substep is boundary-limited, in cm.

Added to the distance returned by
:func:`pyRadMC.geometry.grid.distance_to_voxel_boundary` so a boundary-capped substep
lands just inside the next voxel rather than exactly on the face, where the half-open
lower-inclusive convention could otherwise trap an inward-facing electron at zero
distance. Unlike the entry nudge (which differs by backend for float32 reasons), this
is shared verbatim with the Warp kernel so both backends truncate at identical
positions; 1 micrometre is far below any voxel edge, and its short over-step is charged
to the crossed voxel's stopping power, so it is physically consistent, not a leak."""


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
) -> float:
    """Transport one electron or positron; secondaries go to ``spawn``.

    Returns the weight-energy removed from the transported population: physical
    escape (for a positron including the 2 m_e c^2 of latent annihilation energy
    it carries out) plus the net ledger effect of rouletting sub-threshold
    bremsstrahlung photons at birth (see :mod:`pyRadMC.transport.photon`). The
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

    ``step_energy_fraction`` overrides the per-model default
    (:func:`default_step_energy_fraction`; ``None`` resolves to the selected
    model's validated fraction) — a measurement instrument for substep
    resolution studies. Changing a *default* is a maintainer decision gated
    on the validation tier, not a knob turn.
    """
    if msc_model not in ("gaussian", "gs"):
        raise ValueError(f"unknown msc_model {msc_model!r}; expected 'gaussian' or 'gs'")
    if step_energy_fraction is None:
        step_energy_fraction = default_step_energy_fraction(msc_model)
    escaped = 0.0
    e = energy
    w = weight
    latent = 2.0 * ELECTRON_MASS_MEV if is_positron else 0.0

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
            s_theta = STEP_HINGE_THETA2_MAX / (cross_sections.scattering_power(e, material) * rho)
            s_max = min(s_max, s_theta)
        # Cap at the next voxel face (plus the nudge across it) so the substep's
        # density and material stay those of the voxel it starts in.
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

        # --- continuous loss over the substep (exact, DPM-style) --------------------
        # E_end = r^-1(r(E) - rho*s) with the restricted-collision range, replacing
        # the first-order S(E_start)*rho*s linearization; the range-out shortening
        # is exact by the same map. Sempau et al. 2000, doi:10.1088/0031-9155/45/8/315.
        available = cross_sections.restricted_range(e, material, ecut)
        if rho * s >= available:
            # The electron ranges out inside this substep.
            s = available / rho
            de = e - ecut
            moller_pending = False
        else:
            de = e - cross_sections.energy_after_mass_path(e, material, ecut, rho * s)
        emit_probability, local_brems = bremsstrahlung_step_parameters(
            e, cross_sections.radiative_stopping_power(e, material), rho, s, pcut
        )
        continuous = de + local_brems
        emit_brems = uniform(rng_state) < emit_probability
        # Dose-to-water: one factor per substep, at the (E, material) that set the
        # stopping power above; 1.0 exactly under the dose-to-medium default.
        substep_factor = deposit_weight(e, material)

        # --- random hinge: move, deflect, move --------------------------------------
        s1 = uniform(rng_state) * s
        s2 = s - s1
        d1 = continuous * (s1 / s) if s > 0.0 else 0.0
        d2 = continuous - d1

        escaped += deposit_or_escape(
            x + ux * s1 / 2.0, y + uy * s1 / 2.0, z + uz * s1 / 2.0, w * d1, substep_factor
        )
        e -= d1
        x += ux * s1
        y += uy * s1
        z += uz * s1
        if not grid.contains(x, y, z):
            return escaped + w * (e + latent)

        mean_square = cross_sections.scattering_power(e, material) * rho * s
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

        escaped += deposit_or_escape(
            x + ux * s2 / 2.0, y + uy * s2 / 2.0, z + uz * s2 / 2.0, w * d2, substep_factor
        )
        e -= d2
        x += ux * s2
        y += uy * s2
        z += uz * s2
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
