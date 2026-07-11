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
    "__version__",
]
