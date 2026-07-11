"""pyRadMC: fast photon Monte Carlo for beamlet-resolved treatment planning.

See AGENTS.md for the development contract and the current phase.

Physical constants and defaults that are accuracy-defining live here so that there is
exactly one place to change them, and so that a change is visible in a diff. They are
not tuning knobs; see AGENTS.md section 2.8.
"""

from __future__ import annotations

__version__ = "0.0.1.dev0"

# --- accuracy-defining defaults ------------------------------------------------
# Changing any of these requires a test demonstrating the dosimetric effect.

ECUT_MEV: float = 0.200
"""Electron transport and production cutoff, kinetic energy in MeV (DPM default)."""

PCUT_MEV: float = 0.050
"""Photon transport cutoff in MeV. Below this, energy is deposited locally."""

DIJ_TRUNCATION_RELATIVE: float = 1.0e-3
"""Dij column truncation, relative to that beamlet column's maximum.

This biases the low-dose tail, which is where NTCP and LET-guided objectives operate.
It is tested against DVH endpoints, never against a matrix norm.
"""

# --- variance-reduction parameters (Phase 3) -------------------------------------
# Russian roulette on low-energy photons (see pyRadMC.physics.roulette and the
# transport loops). Unbiased by construction — these change realizations and
# efficiency, never expectations; unbiasedness is test-pinned against a
# roulette-free run. They are still fixed project-wide values, not per-run options
# (AGENTS.md 2.10): one configuration is what the validation tier certifies.

PHOTON_ROULETTE_MEV: float = 0.5
"""Photons below this energy play Russian roulette at their creation or scatter.

Chosen just below the 511 keV annihilation line so annihilation photons are exempt
and positron energy accounting stays analog.
"""

PHOTON_ROULETTE_SURVIVAL: float = 0.5
"""Survival probability per game; a survivor's weight is boosted by its inverse."""

PHOTON_ROULETTE_WEIGHT_CAP: float = 4.0
"""No roulette at or above this weight (the weight-window ceiling).

Caps the boost cascade at two consecutive survivals (1 -> 2 -> 4), bounding the
graininess a single high-weight deposit can leave in the low-dose tail.
"""

PHOTON_SPLIT_N: int = 2
"""Compton splitting multiplicity at a *primary* photon's first Compton scatter.

The primary's Compton final state is sampled ``PHOTON_SPLIT_N`` times, each copy
(scattered photon + recoil electron) carrying weight ``1 / PHOTON_SPLIT_N``: N
independent samples of the dominant scatter source, averaged, so the scattered
dose variance falls while the expectation is exactly preserved and energy is
conserved per realization. Only the primary splits, so the photon population is
bounded (at most N first-generation scattered photons per source photon, and
the transport cost grows about linearly in N); the soft-photon roulette above
then culls the degraded copies, so the splitting and the culling are paired —
the roulette earns its keep (AGENTS.md 7.2).

Like the roulette parameters this is a variance-reduction efficiency knob, not
accuracy-defining: it changes realizations and cost, never expectations.
Unbiasedness is test-pinned against a split-free instrument (N = 1). It is a
fixed project-wide value, one configuration (AGENTS.md 2.10).

The default is deliberately conservative: N = 2 is the minimal genuine split
(it doubles the scattered-photon statistics and pairs with the roulette) at
roughly double the reference-transport cost. The variance-reduction *efficiency*
— the figure of merit ``1 / (sigma^2 * time)`` as a function of N — is settled
by the Phase 4 efficiency measurement; raise N only on that evidence, with the
measurement rerun.
"""

# --- physical constants --------------------------------------------------------

ELECTRON_MASS_MEV: float = 0.510_998_950_69
"""Electron rest mass energy in MeV (CODATA 2022)."""

__all__ = [
    "DIJ_TRUNCATION_RELATIVE",
    "ECUT_MEV",
    "ELECTRON_MASS_MEV",
    "PCUT_MEV",
    "PHOTON_ROULETTE_MEV",
    "PHOTON_ROULETTE_SURVIVAL",
    "PHOTON_ROULETTE_WEIGHT_CAP",
    "PHOTON_SPLIT_N",
    "__version__",
]
