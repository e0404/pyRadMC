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

PHOTON_SPLIT_N: int = 1
"""Compton splitting multiplicity at a *primary* photon's first Compton scatter.

``N = 1`` is **splitting off — the shipped configuration.** At N == 1 the
primary Comptons into a single full-weight copy, i.e. exactly analog transport.
For ``N > 1`` the primary's Compton final state is sampled N times, each copy
(scattered photon + recoil electron) carrying weight ``1 / N``: N independent
samples of the dominant scatter source, exactly unbiased and energy-conserving
per realization, with cost growing about linearly in N. Only the primary
splits, so the population is bounded and the soft-photon roulette culls the
degraded copies.

This is a variance-reduction efficiency knob, not accuracy-defining: it changes
realizations and cost, never expectations. Correctness of the ``N > 1`` path
(unbiasedness, energy books, N-fold fair copies, variance reduction) is
test-pinned with N = 2 as the instrument, so the mechanism stays validated
though it is dormant.

**Why it ships off (Phase 4 efficiency measurement).** For the analytic-water
Dij, splitting does not earn its keep: the figure of merit ``1/(sigma^2*time)``
is < 1 on the reference CPU (variance falls to ~0.67 in the high/mid-dose
region but cost rises ~1.7x) and roughly neutral on the GPU (a warp retires
with its longest thread). Worse, it does **not** help the low-dose tail — the
Dij's NTCP/LET region — because that tail is fed by rare wide-angle multiple
scatters that uniform primary splitting cannot target; splitting deeper only
degrades the FOM further (measured).

**Phase 5 re-measured it on a phase-space source and it still ships off.** On
now-stable-power hardware the FOM ratio split/no-split was 0.75, 0.48, 0.28 at
N = 2, 4, 8 — worse, monotonically. First-Compton splitting decorrelates copies
only after that scatter (variance saturates far below 1/N) while cost grows
~linearly, and emitting a phase-space primary is as cheap as an analytic beam,
so the cost structure matches. The N > 1 path stays retained and N=2-pinned. See
AGENTS.md 7.2 for the full record and the emission-time-splitting alternative.
"""

# --- physical constants --------------------------------------------------------

ELECTRON_MASS_MEV: float = 0.510_998_950_69
"""Electron rest mass energy in MeV (CODATA 2022)."""

RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV: float = 80.65543
"""Coherent-scattering momentum-transfer coefficient: the tabulated form-factor abscissa
is ``x [1/angstrom] = this * E[MeV] * sin(theta/2)``, i.e. ``1/hc`` with
``hc = 0.012_398_42 MeV*angstrom`` (CODATA 2022). EPDL MF=27 tabulates ``F`` against ``x``."""

__all__ = [
    "DIJ_TRUNCATION_RELATIVE",
    "ECUT_MEV",
    "ELECTRON_MASS_MEV",
    "PCUT_MEV",
    "PHOTON_ROULETTE_MEV",
    "PHOTON_ROULETTE_SURVIVAL",
    "PHOTON_ROULETTE_WEIGHT_CAP",
    "PHOTON_SPLIT_N",
    "RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV",
    "__version__",
]
