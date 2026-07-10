"""Particle kinds and shared transport callbacks.

Integer kinds (not an enum) so the values cross unchanged into Warp kernels, exactly
as :class:`pyRadMC.data.interface.PhotonProcess` does.
"""

from __future__ import annotations

from collections.abc import Callable

from pyRadMC import ELECTRON_MASS_MEV
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.physics.direction import sample_isotropic_direction
from pyRadMC.rng import RNGState

__all__ = [
    "ELECTRON",
    "PHOTON",
    "POSITRON",
    "DepositFn",
    "SpawnFn",
    "StackEntry",
    "annihilate_at_rest",
]

PHOTON = 0
ELECTRON = 1
POSITRON = 2

DepositFn = Callable[[int, int, int, float], None]
"""Scoring callback ``(ix, iy, iz, energy_mev)`` for one energy deposit."""

StackEntry = tuple[int, float, float, float, float, float, float, float]
"""One stacked particle: ``(kind, energy, x, y, z, ux, uy, uz)``."""

SpawnFn = Callable[[StackEntry], None]
"""Stack push for a secondary particle."""


def annihilate_at_rest(
    x: float,
    y: float,
    z: float,
    grid: VoxelGrid,
    rng_state: RNGState,
    deposit: DepositFn,
    spawn: SpawnFn,
    pcut: float,
) -> None:
    """Positron annihilation at rest: two back-to-back 511 keV photons, isotropic.

    A stated approximation (annihilation in flight neglected; AGENTS.md 7.2). Called
    only for positions inside the grid. If PCUT is at or above 511 keV the photons
    would die immediately, so the 1.022 MeV is deposited instead.
    """
    if pcut >= ELECTRON_MASS_MEV:
        ix, iy, iz = grid.voxel_index(x, y, z)
        deposit(ix, iy, iz, 2.0 * ELECTRON_MASS_MEV)
        return
    ax, ay, az = sample_isotropic_direction(rng_state)
    spawn((PHOTON, ELECTRON_MASS_MEV, x, y, z, ax, ay, az))
    spawn((PHOTON, ELECTRON_MASS_MEV, x, y, z, -ax, -ay, -az))
