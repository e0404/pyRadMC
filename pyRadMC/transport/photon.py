"""Photon transport with Woodcock tracking.

One photon at a time: this loop serves the reference backend and is the behavioural
specification for the Warp kernels (Phase 2). Sampling lives in :mod:`pyRadMC.physics`;
data access goes through :class:`pyRadMC.data.interface.CrossSectionSource`; this
module only sequences them.

Tracking is Woodcock (delta) tracking: free paths are sampled with the majorant
cross-section and interactions are accepted with probability mu_real / mu_majorant,
so voxel boundaries never need to be ray-traced. Woodcock et al. (1965), ANL-7050.

Interactions hand their charged secondaries to the caller through ``spawn`` when
electron transport is on (Phase 1); with it off, the electron energy is deposited at
the interaction voxel — the Phase 0 KERMA approximation, kept as an explicit option
for photon-only physics tests (AGENTS.md 7.2).

Stated approximations, named here because this is where they are implemented:

- **No fluorescence**: the photoelectric event hands the full photon energy to the
  photoelectron (or the voxel); characteristic x-rays (< 1 keV in water) are not
  emitted. The photoelectron is emitted **forward** (maintainer-approved; the Sauter
  distribution is strongly forward at these energies anyway).
- **Pair energy split sampled uniformly** between the electron and positron
  (maintainer-approved), both emitted **forward**; the positron annihilates at rest
  (see :mod:`pyRadMC.transport.particles`).
- **No Rayleigh by default** is a property of the *data*, not of this loop: the loop
  is channel-complete (AGENTS.md 2.10) and samples the coherent channel whenever the
  data source reports it nonzero (see :mod:`pyRadMC.physics.rayleigh` for the angular
  model). The analytic source keeps the coherent column at zero.

Photons at or below ``pcut`` deposit their energy locally and terminate.
"""

from __future__ import annotations

import math

from pyRadMC import ELECTRON_MASS_MEV
from pyRadMC.data.interface import CrossSectionSource, PhotonProcess
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.physics.channel import select_photon_process
from pyRadMC.physics.compton import (
    compton_cos_theta,
    compton_electron_cos_theta,
    sample_compton_energy_ratio,
)
from pyRadMC.physics.direction import rotate_direction
from pyRadMC.physics.path import sample_path_length
from pyRadMC.physics.rayleigh import sample_rayleigh_cos_theta
from pyRadMC.rng import RNGState, uniform
from pyRadMC.transport.particles import (
    ELECTRON,
    PHOTON,
    POSITRON,
    DepositFn,
    SpawnFn,
    StackEntry,
    annihilate_at_rest,
)

__all__ = ["DepositFn", "photon_steps", "transport_photon"]

# Nudge past the grid surface after the vacuum flight, so the entry position is
# strictly inside under the half-open convention. 1e-9 cm is float-noise relative to
# any voxel this engine will see, and vastly below a photon mean free path.
_ENTRY_NUDGE_CM = 1.0e-9

# Relative headroom for the majorant sanity check. The majorant must bound the real
# cross-section exactly; the epsilon only forgives float evaluation-order noise.
_MAJORANT_TOLERANCE = 1.0e-9


def photon_steps(
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
    transport_electrons: bool,
) -> float:
    """Transport one photon; secondaries go to ``spawn``. Returns escaped energy.

    With ``transport_electrons`` false, charged secondaries deposit locally (KERMA);
    annihilation photons are spawned in either mode. Charged secondaries at or below
    ``ecut`` deposit locally in either mode (production threshold).
    """
    escaped = 0.0
    e = energy

    # Fly through the vacuum outside the grid, if born there.
    if not grid.contains(x, y, z):
        t = grid.distance_to_entry(x, y, z, ux, uy, uz)
        if math.isinf(t):
            return e
        t += _ENTRY_NUDGE_CM
        x += t * ux
        y += t * uy
        z += t * uz
        if not grid.contains(x, y, z):  # grazing-corner numerics
            return e

    while True:
        mu_majorant = cross_sections.majorant(e)
        step = sample_path_length(mu_majorant, rng_state)
        x += step * ux
        y += step * uy
        z += step * uz
        if not grid.contains(x, y, z):
            return escaped + e

        ix, iy, iz = grid.voxel_index(x, y, z)
        rho = float(grid.density[ix, iy, iz])
        material = int(grid.material[ix, iy, iz])
        mu_compton = rho * cross_sections.mu_over_rho(e, material, PhotonProcess.COMPTON)
        mu_photo = rho * cross_sections.mu_over_rho(e, material, PhotonProcess.PHOTOELECTRIC)
        mu_pair = rho * cross_sections.mu_over_rho(e, material, PhotonProcess.PAIR)
        mu_rayleigh = rho * cross_sections.mu_over_rho(e, material, PhotonProcess.RAYLEIGH)
        mu_real = mu_compton + mu_photo + mu_pair + mu_rayleigh
        if mu_real > mu_majorant * (1.0 + _MAJORANT_TOLERANCE):
            raise RuntimeError(
                f"Woodcock majorant violated: mu_real={mu_real:.6e} > "
                f"majorant={mu_majorant:.6e} 1/cm at E={e:.4f} MeV, "
                f"voxel ({ix}, {iy}, {iz}). The geometry contains material or "
                "density the majorant declaration did not cover."
            )

        # Delta (fictitious) scattering: no interaction, keep flying.
        if uniform(rng_state) * mu_majorant >= mu_real:
            continue

        process = select_photon_process(mu_compton, mu_photo, mu_pair, mu_rayleigh, rng_state)

        if process == PhotonProcess.RAYLEIGH:
            # Coherent: direction changes, energy does not. Unreachable with the
            # analytic data source (zero coherent column); see physics.rayleigh.
            cos_coherent = sample_rayleigh_cos_theta(rng_state)
            phi = 2.0 * math.pi * uniform(rng_state)
            ux, uy, uz = rotate_direction(ux, uy, uz, cos_coherent, phi)
        elif process == PhotonProcess.COMPTON:
            ratio = sample_compton_energy_ratio(e, rng_state)
            recoil = e * (1.0 - ratio)
            phi = 2.0 * math.pi * uniform(rng_state)
            if transport_electrons and recoil > ecut:
                cos_electron = compton_electron_cos_theta(e, ratio)
                edir = rotate_direction(ux, uy, uz, cos_electron, phi + math.pi)
                spawn((ELECTRON, recoil, x, y, z, *edir))
            else:
                deposit(ix, iy, iz, recoil)
            cos_gamma = compton_cos_theta(e, ratio)
            ux, uy, uz = rotate_direction(ux, uy, uz, cos_gamma, phi)
            e *= ratio
            if e <= pcut:
                deposit(ix, iy, iz, e)
                return escaped
        elif process == PhotonProcess.PHOTOELECTRIC:
            if transport_electrons and e > ecut:
                spawn((ELECTRON, e, x, y, z, ux, uy, uz))  # forward, no fluorescence
            else:
                deposit(ix, iy, iz, e)
            return escaped
        else:  # pair production
            kinetic = e - 2.0 * ELECTRON_MASS_MEV
            if transport_electrons:
                fraction = uniform(rng_state)  # uniform split, stated approximation
                for kind, share in (
                    (ELECTRON, fraction * kinetic),
                    (POSITRON, (1.0 - fraction) * kinetic),
                ):
                    if share > ecut:
                        spawn((kind, share, x, y, z, ux, uy, uz))  # forward
                    else:
                        deposit(ix, iy, iz, share)
                        if kind == POSITRON:
                            annihilate_at_rest(x, y, z, grid, rng_state, deposit, spawn, pcut)
            else:
                deposit(ix, iy, iz, kinetic)
                annihilate_at_rest(x, y, z, grid, rng_state, deposit, spawn, pcut)
            return escaped


def transport_photon(
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
    pcut: float,
) -> float:
    """Transport one photon history in KERMA mode (Phase 0 behaviour).

    Electron energy deposits at the interaction voxel; annihilation photons are
    followed. Kept as the photon-only entry point for attenuation and scatter tests,
    where local deposition *is* the observable being tested. Full histories with
    electron transport go through :func:`pyRadMC.transport.history.transport_history`.
    """
    stack: list[StackEntry] = []
    escaped = photon_steps(
        energy,
        x,
        y,
        z,
        ux,
        uy,
        uz,
        grid,
        cross_sections,
        rng_state,
        deposit,
        stack.append,
        pcut,
        ecut=math.inf,  # unused in KERMA mode: every charged secondary deposits
        transport_electrons=False,
    )
    while stack:
        kind, e, px, py, pz, dx, dy, dz = stack.pop()
        assert kind == PHOTON  # KERMA mode spawns nothing else
        escaped += photon_steps(
            e,
            px,
            py,
            pz,
            dx,
            dy,
            dz,
            grid,
            cross_sections,
            rng_state,
            deposit,
            stack.append,
            pcut,
            ecut=math.inf,
            transport_electrons=False,
        )
    return escaped
