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

# --- physical constants --------------------------------------------------------

ELECTRON_MASS_MEV: float = 0.510_998_950_69
"""Electron rest mass energy in MeV (CODATA 2022)."""

__all__ = [
    "DIJ_TRUNCATION_RELATIVE",
    "ECUT_MEV",
    "ELECTRON_MASS_MEV",
    "PCUT_MEV",
    "__version__",
]
