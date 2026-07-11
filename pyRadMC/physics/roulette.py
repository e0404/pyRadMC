"""Russian roulette: the unbiased particle-population control of Phase 3.

Russian roulette removes a particle with probability ``1 - survival`` and boosts
a survivor's statistical weight by ``1 / survival``, so the expected weight — and
with it every scored expectation — is exactly preserved:

    E[w_out] = survival * (w / survival) + (1 - survival) * 0 = w.

Standard variance-reduction technique; see Salvat et al., PENELOPE-2018, sec. 1.6.2
(doi:10.1787/32da5043-en) and Kawrakow & Fippel, Phys. Med. Biol. 45 (2000) 2163
(doi:10.1088/0031-9155/45/8/308) for its use in photon dose calculation.

Where the game is *played* — which particles, below which energy, up to which
weight — is transport-loop policy, controlled by the ``PHOTON_ROULETTE_*``
constants in :mod:`pyRadMC`; this module owns only the fair game itself.
"""

from __future__ import annotations

from pyRadMC.rng import RNGState, uniform

__all__ = ["roulette_weight"]


def roulette_weight(weight: float, survival: float, rng_state: RNGState) -> float:
    """Play one round: return 0.0 (killed) or ``weight / survival`` (survivor).

    Consumes exactly one uniform. ``survival`` must lie in (0, 1]; the caller
    guards the degenerate cases (it never invokes the game with survival <= 0).
    """
    survivor = 0.0
    if uniform(rng_state) < survival:
        survivor = weight / survival
    return survivor
