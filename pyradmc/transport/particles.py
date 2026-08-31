"""Particle kinds and shared transport callbacks.

Integer kinds (not an enum) so the values cross unchanged into Warp kernels, exactly
as :class:`pyradmc.data.interface.PhotonProcess` does.
"""

from __future__ import annotations

from collections.abc import Callable

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.physics.direction import sample_isotropic_direction
from pyradmc.rng import RNGState

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

DepositFn = Callable[[float, float, float, float, float], None]
"""Scoring callback ``(x, y, z, energy_mev, scored_mev)`` for one energy deposit.

The position (cm) is the deposit site, guaranteed inside the *transport* grid by
the caller; the scorer routes it to a scoring-grid voxel — or to the unscored
ledger bucket when the scoring grid does not cover it. Deposits are keyed by
position, not voxel index, precisely so that the scoring grid may differ from the
transport grid.

``energy_mev`` is the physical energy — what the ledger books — and
``scored_mev`` is what the dose tally accumulates: identical under
dose-to-medium, SPR-weighted under dose-to-water (see
:mod:`pyradmc.scoring.dose_to_water`). Carrying both keeps the energy balance
exact in every scoring mode.

Under variance reduction, both energies are already weight-scaled: the
scorer sees expected energy, never per-particle energy.
"""

DepositWeightFn = Callable[[float, int], float]
"""Scoring-output weight ``(energy_mev, material) -> factor`` for one deposit.

Evaluated by the transport loops at the deposit's particle energy and local
material; the deposit callback receives ``factor * energy`` as its scored
amount. :func:`unit_weight` (dose-to-medium, the default) returns exactly 1.0,
which is a bit-exact identity under IEEE multiplication — the dose-to-medium
path is byte-identical to scoring without a weight.
"""


def unit_weight(energy: float, material: int) -> float:
    """Dose-to-medium deposit weight: exactly 1 for every deposit."""
    return 1.0


StackEntry = tuple[int, float, float, float, float, float, float, float, float]
"""One stacked particle: ``(kind, energy, weight, x, y, z, ux, uy, uz)``.

``weight`` is the statistical weight (1.0 for analog transport); secondaries
inherit their parent's weight unless a roulette game changed it.
"""

SpawnFn = Callable[[StackEntry], None]
"""Stack push for a secondary particle."""


def annihilate_at_rest(
    x: float,
    y: float,
    z: float,
    rng_state: RNGState,
    deposit: DepositFn,
    spawn: SpawnFn,
    pcut: float,
    weight: float,
    scored_factor: float = 1.0,
) -> None:
    """Positron annihilation at rest: two back-to-back 511 keV photons, isotropic.

    A stated approximation (annihilation in flight neglected; docs/decisions.md). Called
    only for positions inside the transport grid. If PCUT is at or above 511 keV the
    photons would die immediately, so the 1.022 MeV is deposited instead — weighted
    by ``scored_factor``, the caller's dose-to-water SPR at the deposit site (1.0
    for dose-to-medium). The photons carry the positron's statistical weight; at
    511 keV they sit above the photon roulette threshold by construction (see
    ``PHOTON_ROULETTE_MEV``).
    """
    if pcut >= ELECTRON_MASS_MEV:
        amount = weight * 2.0 * ELECTRON_MASS_MEV
        deposit(x, y, z, amount, scored_factor * amount)
        return
    ax, ay, az = sample_isotropic_direction(rng_state)
    spawn((PHOTON, ELECTRON_MASS_MEV, weight, x, y, z, ax, ay, az))
    spawn((PHOTON, ELECTRON_MASS_MEV, weight, x, y, z, -ax, -ay, -az))
