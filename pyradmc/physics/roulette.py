"""Russian roulette: the unbiased particle-population control.

Russian roulette removes a particle with probability ``1 - survival`` and boosts
a survivor's statistical weight by ``1 / survival``, so the expected weight — and
with it every scored expectation — is exactly preserved:

    E[w_out] = survival * (w / survival) + (1 - survival) * 0 = w.

Standard variance-reduction technique; see Salvat et al., PENELOPE-2018, sec. 1.6.2
(doi:10.1787/32da5043-en) and Kawrakow & Fippel, Phys. Med. Biol. 45 (2000) 2163
(doi:10.1088/0031-9155/45/8/308) for its use in photon dose calculation.

Where the game is *played* — which particles, below which energy, up to which
weight — is transport-loop policy, controlled by the ``PHOTON_ROULETTE_*``
constants in :mod:`pyradmc`; this module owns only the fair game itself.

The same game is played on the entries of a dose influence matrix by
:func:`roulette_dij_entry`, the per-entry rule of the Dij's stochastic truncation
mode; which entries, against which threshold, is the assembler's policy
(:class:`pyradmc.scoring.dij.DijAssembler`).
"""

from __future__ import annotations

import math

from pyradmc.rng import RNGState, uniform

__all__ = ["roulette_dij_entry", "roulette_weight"]


def roulette_weight(weight: float, survival: float, rng_state: RNGState) -> float:
    """Play one round: return 0.0 (killed) or ``weight / survival`` (survivor).

    Consumes exactly one uniform. ``survival`` must lie in (0, 1]; the caller
    guards the degenerate cases (it never invokes the game with survival <= 0).
    """
    survivor = 0.0
    if uniform(rng_state) < survival:
        survivor = weight / survival
    return survivor


def roulette_dij_entry(
    mean: float, sigma: float, threshold: float, u: float
) -> tuple[float, float]:
    """Truncate one Dij entry stochastically: return its stored ``(value, sigma)``.

    The importance sampling of the dose calculation matrix of Thieke, Nill,
    Oelfke and Bortfeld, Med. Phys. 29 (2002) 676 (doi:10.1118/1.1469633), which is
    Russian roulette on the entry with survival ``p = mean / threshold``:

    - ``mean >= threshold`` (``threshold = 0`` included): kept unchanged,
      ``(mean, sigma)`` — exactly what the hard rule keeps;
    - ``0 < mean < threshold``: kept with probability ``p``, i.e. when ``u < p``,
      and then stored with value ``threshold``; otherwise dropped;
    - ``mean <= 0``: dropped, as under the hard rule.

    A dropped entry returns ``(0.0, 0.0)``; every stored value is positive, so a
    zero value is the drop. The stored value's expectation over the draw is
    ``p * threshold = mean`` for every entry, so column sums and any plan dose
    built from the matrix are unbiased — unlike the hard cut, whose removed dose
    depends on the statistics the tail was scored with.

    Stored sigma of a surviving tail entry, with ``m = mean``, ``s = sigma``,
    ``t = threshold``::

        sigma_stored = sqrt(s^2 t / m + t (t - m))

    Over the draw its expected square is ``p * sigma_stored^2 = s^2 + m (t - m)``:
    the Monte Carlo variance plus the Bernoulli variance ``t^2 p (1 - p)`` of the
    draw itself, which is the total variance of the stored entry. A quadrature
    plan-dose sigma built from the stored sigmas therefore stays unbiased *in
    expectation* wherever it was valid before (an independent Dij, AGENTS.md
    section 8). The unbiasedness of the value is exact for any threshold, also the
    Monte Carlo-estimated ``t = truncation * column_max``, since it holds
    conditionally on the scored column. Stated approximation: the variance terms
    ``s^2`` and ``m (t - m)`` are evaluated at the scored mean, like every
    per-entry sigma of the Dij, so the variance identity is one in expectation
    over the draw given the scored column.

    ``u`` is one uniform from U[0, 1) drawn through the RNG interface by the
    caller, which owns the stream (a pure function of seed, column and voxel; see
    :class:`~pyradmc.scoring.dij.DijAssembler`). Taking the uniform rather than a
    state lets the caller key it per entry, so the decision is the same however
    the matrix is partitioned or how many passes read it.
    """
    value = 0.0
    stored_sigma = 0.0
    if mean > 0.0:
        if mean >= threshold:
            value = mean
            stored_sigma = sigma
        elif u < mean / threshold:
            value = threshold
            variance = sigma * sigma * threshold / mean + threshold * (threshold - mean)
            stored_sigma = math.sqrt(variance)
    return value, stored_sigma
