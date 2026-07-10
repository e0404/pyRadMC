"""Photon transport with Woodcock tracking and local electron energy deposition.

One history at a time: this loop serves the reference backend and is the behavioural
specification for the Warp kernels (Phase 2). Sampling lives in :mod:`pyRadMC.physics`;
data access goes through :class:`pyRadMC.data.interface.CrossSectionSource`; this
module only sequences them.

Tracking is Woodcock (delta) tracking: free paths are sampled with the majorant
cross-section and interactions are accepted with probability mu_real / mu_majorant,
so voxel boundaries never need to be ray-traced. Woodcock et al. (1965), ANL-7050.

Stated approximations (Phase 0, AGENTS.md section 7), named here because this is where
they are implemented:

- **KERMA / local deposition**: secondary electrons and positrons are not transported;
  their kinetic energy is deposited in the interaction voxel. Valid where charged-
  particle equilibrium holds; systematically wrong within an electron range of
  interfaces and the surface. Electron transport is Phase 1.
- **No fluorescence**: the photoelectric event deposits the full photon energy
  locally; characteristic x-rays (< 1 keV in water) are not emitted.
- **Positron annihilation at rest**: pair events deposit E - 2 m_e c^2 locally and
  emit two back-to-back 511 keV photons, isotropically oriented, from the pair vertex.
  Annihilation in flight is neglected.
- **No Rayleigh by default** (see :mod:`pyRadMC.data.analytic`).

Photons at or below ``pcut`` deposit their energy locally and terminate.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from pyRadMC import ELECTRON_MASS_MEV
from pyRadMC.data.interface import CrossSectionSource, PhotonProcess
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.physics.channel import select_photon_process
from pyRadMC.physics.compton import compton_cos_theta, sample_compton_energy_ratio
from pyRadMC.physics.direction import rotate_direction, sample_isotropic_direction
from pyRadMC.physics.path import sample_path_length
from pyRadMC.rng import RNGState, uniform

__all__ = ["DepositFn", "transport_photon"]

DepositFn = Callable[[int, int, int, float], None]
"""Callback ``(ix, iy, iz, energy_mev)`` scoring one energy deposit."""

# Nudge past the grid surface after the vacuum flight, so the entry position is
# strictly inside under the half-open convention. 1e-9 cm is float-noise relative to
# any voxel this engine will see, and vastly below a photon mean free path.
_ENTRY_NUDGE_CM = 1.0e-9

# Relative headroom for the majorant sanity check. The majorant must bound the real
# cross-section exactly; the epsilon only forgives float evaluation-order noise.
_MAJORANT_TOLERANCE = 1.0e-9


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
    """Transport one photon history (primary plus its annihilation photons).

    Parameters
    ----------
    energy
        Photon energy in MeV.
    x, y, z
        Start position in cm; may be outside the grid (the outside is vacuum).
    ux, uy, uz
        Unit direction.
    grid
        The voxel phantom.
    cross_sections
        Interaction data source; also supplies the Woodcock majorant, which is
        sanity-checked against the local cross-section at every real-interaction
        candidate (a violated majorant biases dose smoothly and invisibly).
    rng_state
        Per-history RNG state.
    deposit
        Scoring callback receiving voxel indices and the deposited energy in MeV.
    pcut
        Photon cutoff in MeV (see ``pyRadMC.PCUT_MEV``); accuracy-defining, so it is
        an explicit argument, never defaulted here.

    Returns
    -------
    float
        Energy escaping the grid, in MeV. The caller balances this against emitted
        and deposited energy; the sum is exact.
    """
    escaped = 0.0
    # The history's particle stack: the primary, then annihilation photons.
    stack = [(energy, x, y, z, ux, uy, uz)]

    while stack:
        e, px, py, pz, dx, dy, dz = stack.pop()

        # Fly through the vacuum outside the grid, if born there.
        if not grid.contains(px, py, pz):
            t = grid.distance_to_entry(px, py, pz, dx, dy, dz)
            if math.isinf(t):
                escaped += e
                continue
            t += _ENTRY_NUDGE_CM
            px += t * dx
            py += t * dy
            pz += t * dz
            if not grid.contains(px, py, pz):  # grazing-corner numerics
                escaped += e
                continue

        while True:
            mu_majorant = cross_sections.majorant(e)
            step = sample_path_length(mu_majorant, rng_state)
            px += step * dx
            py += step * dy
            pz += step * dz
            if not grid.contains(px, py, pz):
                escaped += e
                break

            ix, iy, iz = grid.voxel_index(px, py, pz)
            rho = float(grid.density[ix, iy, iz])
            material = int(grid.material[ix, iy, iz])
            mu_compton = rho * cross_sections.mu_over_rho(e, material, PhotonProcess.COMPTON)
            mu_photo = rho * cross_sections.mu_over_rho(e, material, PhotonProcess.PHOTOELECTRIC)
            mu_pair = rho * cross_sections.mu_over_rho(e, material, PhotonProcess.PAIR)
            mu_real = mu_compton + mu_photo + mu_pair
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

            process = select_photon_process(mu_compton, mu_photo, mu_pair, rng_state)

            if process == PhotonProcess.COMPTON:
                ratio = sample_compton_energy_ratio(e, rng_state)
                deposit(ix, iy, iz, e * (1.0 - ratio))  # electron energy, locally
                cos_theta = compton_cos_theta(e, ratio)
                phi = 2.0 * math.pi * uniform(rng_state)
                dx, dy, dz = rotate_direction(dx, dy, dz, cos_theta, phi)
                e *= ratio
                if e <= pcut:
                    deposit(ix, iy, iz, e)
                    break
            elif process == PhotonProcess.PHOTOELECTRIC:
                deposit(ix, iy, iz, e)  # no fluorescence
                break
            else:  # pair production
                deposit(ix, iy, iz, e - 2.0 * ELECTRON_MASS_MEV)
                if pcut >= ELECTRON_MASS_MEV:
                    deposit(ix, iy, iz, 2.0 * ELECTRON_MASS_MEV)
                else:
                    ax, ay, az = sample_isotropic_direction(rng_state)
                    stack.append((ELECTRON_MASS_MEV, px, py, pz, ax, ay, az))
                    stack.append((ELECTRON_MASS_MEV, px, py, pz, -ax, -ay, -az))
                break

    return escaped
