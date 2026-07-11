"""Class II condensed-history electron (and positron) transport.

One particle at a time, for the reference backend. The scheme, per AGENTS.md 7.2:

- **Continuous** energy loss along each substep from the restricted collision
  stopping power (Berger-Seltzer, cut at ECUT), deposited at the two segment
  midpoints around the multiple-scattering hinge.
- **Discrete Moller events** above ECUT, distance-sampled with the restricted Moller
  cross-section; the delta ray is stacked, both outgoing directions from exact
  kinematics with back-to-back azimuths.
- **Random hinge** multiple scattering (PENELOPE-style; Salvat et al.,
  PENELOPE-2018, sec. 4.3, doi:10.1787/32da5043-en): the Gaussian Fermi-Eyges
  deflection for the whole substep is applied at a uniformly random point within it.
- **Discrete thin-target bremsstrahlung** (see :mod:`pyRadMC.physics.brems`): at most
  one photon per substep, emitted forward from the hinge point, Bernoulli probability
  matched to the radiative stopping power; the sub-PCUT remainder joins the
  continuous deposit.
- **Positrons are transported as electrons** (no Bhabha distinction — a stated,
  maintainer-approved approximation) and annihilate at rest into two back-to-back
  511 keV photons wherever they stop; a positron leaving the grid carries its
  annihilation energy with it.

Substeps are limited to a fraction of the CSDA range (energy-loss accuracy) and of
the smallest voxel edge (heterogeneity accuracy). The reference backend keeps both
conservative; the Warp backend mirrors this loop step for step and is validated
against it — step aggressiveness beyond it must be bought with better transport
mechanics, not by loosening the oracle.

Electrons at or below ECUT deposit their kinetic energy locally and terminate.
"""

from __future__ import annotations

import math

from pyRadMC import ELECTRON_MASS_MEV
from pyRadMC.data.interface import CrossSectionSource
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.physics.brems import bremsstrahlung_step_parameters, sample_bremsstrahlung_energy
from pyRadMC.physics.direction import rotate_direction
from pyRadMC.physics.moller import moller_direction_cosines, sample_moller_delta_energy
from pyRadMC.physics.msc import sample_hinge_cos_theta
from pyRadMC.physics.path import sample_path_length
from pyRadMC.rng import RNGState, uniform
from pyRadMC.transport.particles import (
    ELECTRON,
    PHOTON,
    DepositFn,
    SpawnFn,
    annihilate_at_rest,
)

__all__ = ["STEP_ENERGY_FRACTION", "STEP_VOXEL_FRACTION", "electron_steps"]

STEP_ENERGY_FRACTION: float = 0.05
"""Maximum fraction of the CSDA range per substep.

Bounds the relative energy change per substep so that evaluating the stopping power,
Moller cross-section and scattering power at the substep's *initial* energy stays a
sub-percent approximation. Conservative on purpose: this is the oracle.
"""

STEP_VOXEL_FRACTION: float = 0.5
"""Maximum substep length as a fraction of the smallest voxel edge.

Keeps the two continuous-deposit midpoints resolving the voxel structure and bounds
the error of using one voxel's density for a substep that grazes a neighbour.
"""

_ENTRY_NUDGE_CM = 1.0e-9  # same convention as the photon loop


def electron_steps(
    is_positron: bool,
    energy: float,
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
) -> float:
    """Transport one electron or positron; secondaries go to ``spawn``.

    Returns the energy escaping the grid (for a positron this includes the
    2 m_e c^2 of latent annihilation energy it carries out). Every MeV of kinetic
    energy is either deposited, handed to a spawned secondary, or escapes — the
    engine's exact energy balance extends over this loop unchanged.
    """
    escaped = 0.0
    e = energy
    min_spacing = min(grid.spacing)
    latent = 2.0 * ELECTRON_MASS_MEV if is_positron else 0.0

    # Vacuum flight for a primary born outside the phantom (electron beams).
    if not grid.contains(x, y, z):
        t = grid.distance_to_entry(x, y, z, ux, uy, uz)
        if math.isinf(t):
            return e + latent
        t += _ENTRY_NUDGE_CM
        x += t * ux
        y += t * uy
        z += t * uz
        if not grid.contains(x, y, z):
            return e + latent

    def deposit_or_escape(px: float, py: float, pz: float, amount: float) -> float:
        """Deposit at a point if it is inside; otherwise report it as escaped."""
        if amount <= 0.0:
            return 0.0
        if grid.contains(px, py, pz):
            ix, iy, iz = grid.voxel_index(px, py, pz)
            deposit(ix, iy, iz, amount)
            return 0.0
        return amount

    while True:
        if e <= ecut:
            # Terminal: local deposit; the current position is inside the grid.
            ix, iy, iz = grid.voxel_index(x, y, z)
            deposit(ix, iy, iz, e)
            if is_positron:
                annihilate_at_rest(x, y, z, grid, rng_state, deposit, spawn, pcut)
            return escaped

        ix, iy, iz = grid.voxel_index(x, y, z)
        rho = float(grid.density[ix, iy, iz])
        material = int(grid.material[ix, iy, iz])

        # --- substep length --------------------------------------------------------
        range_cm = cross_sections.csda_range(e, material) / rho
        s_max = min(STEP_ENERGY_FRACTION * range_cm, STEP_VOXEL_FRACTION * min_spacing)
        sigma_moller = rho * cross_sections.moller_cross_section(e, material, ecut)
        s_interaction = (
            sample_path_length(sigma_moller, rng_state) if sigma_moller > 0.0 else math.inf
        )
        s = min(s_interaction, s_max)
        moller_pending = s_interaction <= s_max

        # --- continuous loss over the substep ---------------------------------------
        de = cross_sections.restricted_stopping_power(e, material, ecut) * rho * s
        if de >= e - ecut:
            # The electron ranges out inside this substep: shorten it linearly.
            s *= (e - ecut) / de
            de = e - ecut
            moller_pending = False
        emit_probability, local_brems = bremsstrahlung_step_parameters(
            e, cross_sections.radiative_stopping_power(e, material), rho, s, pcut
        )
        continuous = de + local_brems
        emit_brems = uniform(rng_state) < emit_probability

        # --- random hinge: move, deflect, move --------------------------------------
        s1 = uniform(rng_state) * s
        s2 = s - s1
        d1 = continuous * (s1 / s) if s > 0.0 else 0.0
        d2 = continuous - d1

        escaped += deposit_or_escape(x + ux * s1 / 2.0, y + uy * s1 / 2.0, z + uz * s1 / 2.0, d1)
        e -= d1
        x += ux * s1
        y += uy * s1
        z += uz * s1
        if not grid.contains(x, y, z):
            return escaped + e + latent

        mean_square = cross_sections.scattering_power(e, material) * rho * s
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
                spawn((PHOTON, k, x, y, z, ux, uy, uz))  # forward, stated approx
            else:
                escaped += deposit_or_escape(x, y, z, k)
            e -= k

        escaped += deposit_or_escape(x + ux * s2 / 2.0, y + uy * s2 / 2.0, z + uz * s2 / 2.0, d2)
        e -= d2
        x += ux * s2
        y += uy * s2
        z += uz * s2
        if not grid.contains(x, y, z):
            return escaped + e + latent

        # --- discrete Moller event at the end of the substep -------------------------
        if moller_pending and e > 2.0 * ecut:
            delta_energy = sample_moller_delta_energy(e, ecut, rng_state)
            cos_delta, cos_primary = moller_direction_cosines(e, delta_energy)
            phi = 2.0 * math.pi * uniform(rng_state)
            delta_dir = rotate_direction(ux, uy, uz, cos_delta, phi)
            spawn((ELECTRON, delta_energy, x, y, z, *delta_dir))
            ux, uy, uz = rotate_direction(ux, uy, uz, cos_primary, phi + math.pi)
            e -= delta_energy
