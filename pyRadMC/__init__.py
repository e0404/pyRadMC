"""pyRadMC: fast photon Monte Carlo for beamlet-resolved treatment planning.

See AGENTS.md for the development contract.

This module is the public API: everything named in ``__all__`` is supported and
versioned, and everything else is an implementation detail that may move between
releases. Physical constants and defaults that are accuracy-defining live here too,
so that there is exactly one place to change them and so that a change is visible in
a diff. They are not tuning knobs; see AGENTS.md section 2.8.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

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

# --- variance-reduction parameters -------------------------------------
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

**Why it ships off (measured).** For the analytic-water
Dij, splitting does not earn its keep: the figure of merit ``1/(sigma^2*time)``
is < 1 on the reference CPU (variance falls to ~0.67 in the high/mid-dose
region but cost rises ~1.7x) and roughly neutral on the GPU (a warp retires
with its longest thread). Worse, it does **not** help the low-dose tail — the
Dij's NTCP/LET region — because that tail is fed by rare wide-angle multiple
scatters that uniform primary splitting cannot target; splitting deeper only
degrades the FOM further (measured).

**Re-measured on a phase-space source, it still ships off.** On
now-stable-power hardware the FOM ratio split/no-split was 0.75, 0.48, 0.28 at
N = 2, 4, 8 — worse, monotonically. First-Compton splitting decorrelates copies
only after that scatter (variance saturates far below 1/N) while cost grows
~linearly, and emitting a phase-space primary is as cheap as an analytic beam,
so the cost structure matches. The N > 1 path stays retained and N=2-pinned. See
docs/decisions.md for the full record and the emission-time-splitting alternative.
"""

# --- physical constants --------------------------------------------------------

ELECTRON_MASS_MEV: float = 0.510_998_950_69
"""Electron rest mass energy in MeV (CODATA 2022)."""

GY_PER_MEV_PER_G: float = 1.602_176_634e-10
"""Absolute-dose calibration: 1 MeV/g = this many gray.

Exact by SI definition: 1 MeV = e x 1e6 J with the elementary charge fixed at
1.602176634e-19 C (SI 2019), and per gram -> per kilogram is 1e3. The engines
score dose in MeV/g per emitted history; a planning consumer multiplies by this
constant for Gy per history and applies its own particles-per-MU scaling on top
(see :meth:`pyRadMC.scoring.dij.DijResult.dose_csc`)."""

RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV: float = 80.65543
"""Coherent-scattering momentum-transfer coefficient: the tabulated form-factor abscissa
is ``x [1/angstrom] = this * E[MeV] * sin(theta/2)``, i.e. ``1/hc`` with
``hc = 0.012_398_42 MeV*angstrom`` (CODATA 2022). EPDL MF=27 tabulates ``F`` against ``x``."""

# --- public API ----------------------------------------------------------------
# Re-exported lazily (PEP 562). ``import pyRadMC`` must stay cheap and dependency-free:
# a consumer that wants only the constants must not pay for NumPy, and a core-only
# install (no ``warp`` extra) must not fail at import merely because ``WarpEngine`` is
# a public name. The ``TYPE_CHECKING`` block below is what mypy and IDEs resolve
# against; ``_LAZY_EXPORTS`` is what actually runs.
#
# Names NOT promoted here are still importable from their modules, but they are not
# the public API: beam-limiting devices (``pyRadMC.geometry.collimation``), the
# treatment-head pre-solve (``pyRadMC.geometry.head``), the CT adapter
# (``pyRadMC.adapters.ct``), the tabulated-table precompiler
# (``pyRadMC.data.tabulated``) and the toy optimizer (``pyRadMC.study``) are
# documented at subpackage level because each is a coherent subsystem with its own
# vocabulary, not a name a first script reaches for.

if TYPE_CHECKING:
    from pyRadMC.backends.ref.engine import ReferenceEngine
    from pyRadMC.backends.results import TransportResult
    from pyRadMC.backends.warp.engine import WarpEngine
    from pyRadMC.data.analytic import AnalyticCrossSections
    from pyRadMC.data.interface import CrossSectionSource, PhotonProcess
    from pyRadMC.data.materials import (
        ADIPOSE,
        AIR,
        CORTICAL_BONE,
        LUNG,
        MATERIALS,
        TUNGSTEN,
        WATER,
        MaterialData,
    )
    from pyRadMC.data.tabulated.source import TabulatedCrossSections
    from pyRadMC.geometry.fluence import RadialFluence
    from pyRadMC.geometry.grid import VoxelGrid
    from pyRadMC.geometry.phasespace import InMemoryPhaseSpaceSource, PhaseSpaceSource
    from pyRadMC.geometry.source import (
        BeamletGridSource,
        BeamletSource,
        CompositeBeamletSource,
        CompositeSource,
        GaussianSpotBeamletSource,
        GaussianSpotBeamSource,
        ParallelBeamSource,
        PencilBeamSource,
        Primary,
        PrimaryFluenceBeamletSource,
        PrimaryFluenceBeamSource,
        Source,
        SpectralBeamletSource,
        SpectralBeamSource,
    )
    from pyRadMC.geometry.spectrum import ALI_ROGERS_BEAMS, Spectrum, ali_rogers_mv
    from pyRadMC.rng.host import HostRNG
    from pyRadMC.scoring.cylinder import (
        CylindricalScoringGrid,
        geometric_edges,
        graded_edges,
        uniform_edges,
    )
    from pyRadMC.scoring.dij import DijResult
    from pyRadMC.scoring.grid import ScoringGrid

_LAZY_EXPORTS: dict[str, str] = {
    "ADIPOSE": "pyRadMC.data.materials",
    "AIR": "pyRadMC.data.materials",
    "ALI_ROGERS_BEAMS": "pyRadMC.geometry.spectrum",
    "AnalyticCrossSections": "pyRadMC.data.analytic",
    "BeamletGridSource": "pyRadMC.geometry.source",
    "BeamletSource": "pyRadMC.geometry.source",
    "CORTICAL_BONE": "pyRadMC.data.materials",
    "CompositeBeamletSource": "pyRadMC.geometry.source",
    "CompositeSource": "pyRadMC.geometry.source",
    "CrossSectionSource": "pyRadMC.data.interface",
    "CylindricalScoringGrid": "pyRadMC.scoring.cylinder",
    "DijResult": "pyRadMC.scoring.dij",
    "GaussianSpotBeamSource": "pyRadMC.geometry.source",
    "GaussianSpotBeamletSource": "pyRadMC.geometry.source",
    "HostRNG": "pyRadMC.rng.host",
    "InMemoryPhaseSpaceSource": "pyRadMC.geometry.phasespace",
    "LUNG": "pyRadMC.data.materials",
    "MATERIALS": "pyRadMC.data.materials",
    "MaterialData": "pyRadMC.data.materials",
    "ParallelBeamSource": "pyRadMC.geometry.source",
    "PencilBeamSource": "pyRadMC.geometry.source",
    "PhaseSpaceSource": "pyRadMC.geometry.phasespace",
    "PhotonProcess": "pyRadMC.data.interface",
    "Primary": "pyRadMC.geometry.source",
    "PrimaryFluenceBeamSource": "pyRadMC.geometry.source",
    "PrimaryFluenceBeamletSource": "pyRadMC.geometry.source",
    "RadialFluence": "pyRadMC.geometry.fluence",
    "ReferenceEngine": "pyRadMC.backends.ref.engine",
    "ScoringGrid": "pyRadMC.scoring.grid",
    "Source": "pyRadMC.geometry.source",
    "Spectrum": "pyRadMC.geometry.spectrum",
    "SpectralBeamSource": "pyRadMC.geometry.source",
    "SpectralBeamletSource": "pyRadMC.geometry.source",
    "TUNGSTEN": "pyRadMC.data.materials",
    "TabulatedCrossSections": "pyRadMC.data.tabulated.source",
    "TransportResult": "pyRadMC.backends.results",
    "VoxelGrid": "pyRadMC.geometry.grid",
    "WATER": "pyRadMC.data.materials",
    "WarpEngine": "pyRadMC.backends.warp.engine",
    "ali_rogers_mv": "pyRadMC.geometry.spectrum",
    "geometric_edges": "pyRadMC.scoring.cylinder",
    "graded_edges": "pyRadMC.scoring.cylinder",
    "uniform_edges": "pyRadMC.scoring.cylinder",
}

# Third-party module whose absence means an optional extra is not installed -> the
# extra that provides it. Only consulted when the import actually failed on that
# module, so a genuine ImportError from inside our own code still propagates as-is.
_EXTRA_FOR_MISSING_MODULE: dict[str, str] = {"warp": "warp", "SimpleITK": "ct"}


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access, then cache it in the module globals."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        missing = (getattr(exc, "name", "") or "").split(".")[0]
        extra = _EXTRA_FOR_MISSING_MODULE.get(missing)
        if extra is None:
            raise
        raise ImportError(
            f"pyRadMC.{name} needs the optional {extra!r} extra: pip install 'pyRadMC[{extra}]'"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # later lookups find it directly and skip __getattr__
    return value


def __dir__() -> list[str]:
    """List the public API, so tab-completion sees names not yet imported."""
    return sorted(__all__)


__all__ = [
    "ADIPOSE",
    "AIR",
    "ALI_ROGERS_BEAMS",
    "CORTICAL_BONE",
    "DIJ_TRUNCATION_RELATIVE",
    "ECUT_MEV",
    "ELECTRON_MASS_MEV",
    "GY_PER_MEV_PER_G",
    "LUNG",
    "MATERIALS",
    "PCUT_MEV",
    "PHOTON_ROULETTE_MEV",
    "PHOTON_ROULETTE_SURVIVAL",
    "PHOTON_ROULETTE_WEIGHT_CAP",
    "PHOTON_SPLIT_N",
    "RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV",
    "TUNGSTEN",
    "WATER",
    "AnalyticCrossSections",
    "BeamletGridSource",
    "BeamletSource",
    "CompositeBeamletSource",
    "CompositeSource",
    "CrossSectionSource",
    "CylindricalScoringGrid",
    "DijResult",
    "GaussianSpotBeamSource",
    "GaussianSpotBeamletSource",
    "HostRNG",
    "InMemoryPhaseSpaceSource",
    "MaterialData",
    "ParallelBeamSource",
    "PencilBeamSource",
    "PhaseSpaceSource",
    "PhotonProcess",
    "Primary",
    "PrimaryFluenceBeamSource",
    "PrimaryFluenceBeamletSource",
    "RadialFluence",
    "ReferenceEngine",
    "ScoringGrid",
    "Source",
    "SpectralBeamSource",
    "SpectralBeamletSource",
    "Spectrum",
    "TabulatedCrossSections",
    "TransportResult",
    "VoxelGrid",
    "WarpEngine",
    "__version__",
    "ali_rogers_mv",
    "geometric_edges",
    "graded_edges",
    "uniform_edges",
]
