"""Material definitions for the cross-section backends.

The engine began as water-only. The registry is nevertheless a registry, not
a constant, so that adding lung/bone-like media in later phases is a data change and not
a code change — that change is the multi-material work, which gave every entry its
elemental composition (to mix per-element EPDL/EEDL data into per-material tables), its
mean excitation energy and Sternheimer density-effect coefficients (the Berger-Seltzer
collision stopping inputs), and its ESTAR radiative anchors (the log-quadratic radiative
fit is a calibration, and each material carries its own calibration points).

Materials are referred to everywhere by integer index (see
:class:`pyradmc.data.interface.CrossSectionSource`); the index crosses unchanged into
kernels, where a Python object cannot.

Data provenance:

- Atomic weights: IUPAC 2021 standard atomic weights (Prohaska et al., Pure Appl.
  Chem. 94, 573 (2022), doi:10.1515/pac-2019-0603), abridged values.
- Water: composition from the H2O formula; I = 75 eV from ICRU Report 37 (1984);
  Sternheimer coefficients from Sternheimer, Berger & Seltzer, At. Data Nucl. Data
  Tables 30, 261 (1984), doi:10.1016/0092-640X(84)90002-0; radiative anchors from
  NIST ESTAR (Berger & Seltzer's data; see the analytic backend).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

__all__ = [
    "ADIPOSE",
    "AIR",
    "CORTICAL_BONE",
    "LUNG",
    "MATERIALS",
    "STANDARD_ATOMIC_WEIGHT",
    "TUNGSTEN",
    "TUNGSTEN_ALLOY",
    "WATER",
    "MaterialData",
    "SternheimerParameters",
    "electrons_per_gram_from_composition",
]

AVOGADRO: float = 6.022_140_76e23
"""Avogadro constant in 1/mol (exact, SI 2019)."""

# Standard (conventional) atomic weights, g/mol; IUPAC 2021 (Prohaska et al.,
# doi:10.1515/pac-2019-0603). The elements of the ICRP/ICRU reference media (H..Ca
# for the bulk constituents plus the Fe/Zn traces of the ICRP tissues) and of the
# beam-limiting-device materials (W, with the Ni/Cu binders of the common heavy
# alloys, and Al for the flattening-filter attenuation the Ali-Rogers spectra carry).
# This is the canonical table; the EPDL/EEDL parsers import it from here.
STANDARD_ATOMIC_WEIGHT: dict[int, float] = {
    1: 1.008,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    11: 22.98976928,
    12: 24.305,
    13: 26.9815384,
    15: 30.973761998,
    16: 32.06,
    17: 35.45,
    18: 39.95,
    19: 39.0983,
    20: 40.078,
    26: 55.845,
    28: 58.693,
    29: 63.546,
    30: 65.38,
    74: 183.84,
}


class SternheimerParameters(NamedTuple):
    """Sternheimer density-effect coefficients (a, m, x0, x1, Cbar, delta0).

    The piecewise parameterization of the density-effect correction delta(x) with
    x = log10(p / m_e c); Sternheimer, Berger & Seltzer (1984),
    doi:10.1016/0092-640X(84)90002-0. ``delta0`` is the small conduction-electron
    term below ``x0`` (zero for insulators in the SBS tables that omit it).
    """

    a: float
    m: float
    x0: float
    x1: float
    cbar: float
    delta0: float


def electrons_per_gram_from_composition(composition: tuple[tuple[int, float], ...]) -> float:
    """Electron density per unit mass, ``N_A * sum_i w_i Z_i / M_i``, in 1/g.

    ``composition`` is ``((Z, mass_fraction), ...)``; weights come from
    :data:`STANDARD_ATOMIC_WEIGHT`.
    """
    if not composition:
        raise ValueError("empty composition")
    return AVOGADRO * sum(fraction * z / STANDARD_ATOMIC_WEIGHT[z] for z, fraction in composition)


@dataclass(frozen=True)
class MaterialData:
    """Host-side physical data for one material.

    Attributes
    ----------
    name
        Human-readable identifier.
    density
        Reference mass density in g/cm^3. Voxel densities in the geometry scale
        macroscopic cross-sections relative to this via the mass quantities, so this
        value is informational for water-like transport, not accuracy-defining.
        (The Sternheimer coefficients below *were computed at* this density; evaluating
        them for a voxel at a different density is the standard density-scaling
        approximation every mass-quantity lookup in this engine already makes.)
    electrons_per_gram
        Electron density per unit mass in 1/g, i.e. ``N_A * <Z/A>``. Multiplies
        per-electron cross-sections (Klein-Nishina, Moller, Berger-Seltzer) into mass
        quantities. Stored explicitly rather than derived so that water can keep its
        historical molecular-weight value (10 N_A / 18.01528) bit-exactly.
    composition
        Elemental mass fractions ``((Z, w), ...)``, ascending in Z, summing to one
        within the rounding of the published table they were transcribed from. This is
        the compile-time mixing key for per-element photon/electron data.
    mean_excitation_mev
        Mean excitation energy I in MeV (ICRU-37 vintage, consistent with the
        Sternheimer coefficients and with NIST ESTAR).
    sternheimer
        Density-effect coefficients, computed at ``density``; see
        :class:`SternheimerParameters`.
    radiative_anchors
        ``((E_MeV, S_rad_MeV_cm2_g), ...)`` NIST ESTAR radiative stopping anchors the
        log-quadratic radiative fit passes through exactly (three or more, ascending).
    """

    name: str
    density: float
    electrons_per_gram: float
    composition: tuple[tuple[int, float], ...]
    mean_excitation_mev: float
    sternheimer: SternheimerParameters
    radiative_anchors: tuple[tuple[float, float], ...]


WATER: int = 0
"""Material index of liquid water."""

AIR: int = 1
"""Material index of dry air (near sea level, 20 C, 1 atm)."""

LUNG: int = 2
"""Material index of ICRP lung tissue (deflated parenchyma; voxel density scales it)."""

ADIPOSE: int = 3
"""Material index of ICRP adipose tissue."""

CORTICAL_BONE: int = 4
"""Material index of ICRP cortical bone."""

TUNGSTEN: int = 5
"""Material index of pure tungsten (beam-limiting devices: jaws, MLC leaves).

Real jaws and leaves are often 90-97 percent W heavy alloys with Ni/Cu/Fe binders;
v1 of the collimation module models them as pure tungsten at a caller-supplied
density, which moves mu/rho by ~1-2 percent at MV energies. :data:`TUNGSTEN_ALLOY`
is that anticipated alloy entry; this pure entry remains for callers that want
elemental tungsten.
"""

TUNGSTEN_ALLOY: int = 6
"""Material index of W95/Ni3.5/Cu1.5 heavy alloy at 18.0 g/cm^3 (MLC leaves).

The composition and density are the published surrogate for Halcyon-class dual-layer
MLC leaves (nominal; real machines vary by a few percent and transmission
commissioning may adjust the density). Retires the pure-W-at-alloy-density
approximation recorded on :data:`TUNGSTEN`.
"""

# H2O: 10 electrons per molecule, M = 18.01528 g/mol (molecular standard); the
# composition uses the IUPAC elemental weights (2*1.008 + 15.999 = 18.015 g/mol),
# a 1.5e-5 relative difference the consistency test documents.
_WATER_ELECTRONS_PER_GRAM = 10.0 * AVOGADRO / 18.01528
_WATER_MOLAR_MASS = 2.0 * STANDARD_ATOMIC_WEIGHT[1] + STANDARD_ATOMIC_WEIGHT[8]

# The ICRP reference media below are transcribed from the PDG 2024 atomic-properties
# compilation (pdg.lbl.gov/2024/AtomicNuclearProperties: compositions, densities,
# I-values, and the Sternheimer, Berger & Seltzer (1984) density-effect rows it
# reproduces) and NIST ESTAR (radiative anchors; materials 104/190/103/120). These are
# the ICRP formulations ESTAR itself computes with, so the per-material validation
# gates in tests/unit/test_icrp_media.py compare like with like. Mass fractions are
# stored exactly as published (summing to one within table rounding), trace elements
# included — dropping or renormalizing them would be an undocumented data alteration.
_AIR_COMPOSITION: tuple[tuple[int, float], ...] = (
    (6, 0.000124),
    (7, 0.755267),
    (8, 0.231781),
    (18, 0.012827),
)
_LUNG_COMPOSITION: tuple[tuple[int, float], ...] = (
    (1, 0.101278),
    (6, 0.102310),
    (7, 0.028650),
    (8, 0.757072),
    (11, 0.001840),
    (12, 0.000730),
    (15, 0.000800),
    (16, 0.002250),
    (17, 0.002660),
    (19, 0.001940),
    (20, 0.000090),
    (26, 0.000370),
    (30, 0.000010),
)
_ADIPOSE_COMPOSITION: tuple[tuple[int, float], ...] = (
    (1, 0.119477),
    (6, 0.637240),
    (7, 0.007970),
    (8, 0.232333),
    (11, 0.000500),
    (12, 0.000020),
    (15, 0.000160),
    (16, 0.000730),
    (17, 0.001190),
    (19, 0.000320),
    (20, 0.000020),
    (26, 0.000020),
    (30, 0.000020),
)
_CORTICAL_BONE_COMPOSITION: tuple[tuple[int, float], ...] = (
    (1, 0.047234),
    (6, 0.144330),
    (7, 0.041990),
    (8, 0.446096),
    (12, 0.002200),
    (15, 0.104970),
    (16, 0.003150),
    (20, 0.209930),
    (30, 0.000100),
)

MATERIALS: tuple[MaterialData, ...] = (
    MaterialData(
        name="water",
        density=1.0,
        electrons_per_gram=_WATER_ELECTRONS_PER_GRAM,
        composition=(
            (1, 2.0 * STANDARD_ATOMIC_WEIGHT[1] / _WATER_MOLAR_MASS),
            (8, STANDARD_ATOMIC_WEIGHT[8] / _WATER_MOLAR_MASS),
        ),
        mean_excitation_mev=75.0e-6,  # ICRU Report 37 (1984)
        # Sternheimer, Berger & Seltzer (1984), liquid water row (exact, delta0 included).
        sternheimer=SternheimerParameters(
            a=0.09116, m=3.4773, x0=0.2400, x1=2.8004, cbar=3.5017, delta0=0.097
        ),
        # NIST ESTAR radiative mass stopping power, liquid water (MeV -> MeV cm^2/g).
        radiative_anchors=((1.0, 0.0128), (10.0, 0.1813), (20.0, 0.4008)),
    ),
    MaterialData(
        name="air",
        density=1.205e-3,
        electrons_per_gram=electrons_per_gram_from_composition(_AIR_COMPOSITION),
        composition=_AIR_COMPOSITION,
        mean_excitation_mev=85.7e-6,
        sternheimer=SternheimerParameters(
            a=0.1091, m=3.3994, x0=1.7418, x1=4.2759, cbar=10.5961, delta0=0.0
        ),
        radiative_anchors=((1.0, 1.271e-2), (10.0, 1.795e-1), (20.0, 4.042e-1)),
    ),
    MaterialData(
        name="lung",
        density=1.050,
        electrons_per_gram=electrons_per_gram_from_composition(_LUNG_COMPOSITION),
        composition=_LUNG_COMPOSITION,
        mean_excitation_mev=75.3e-6,
        sternheimer=SternheimerParameters(
            a=0.0859, m=3.5353, x0=0.2261, x1=2.8001, cbar=3.4708, delta0=0.0
        ),
        radiative_anchors=((1.0, 1.265e-2), (10.0, 1.793e-1), (20.0, 4.038e-1)),
    ),
    MaterialData(
        name="adipose",
        density=0.920,
        electrons_per_gram=electrons_per_gram_from_composition(_ADIPOSE_COMPOSITION),
        composition=_ADIPOSE_COMPOSITION,
        mean_excitation_mev=63.2e-6,
        sternheimer=SternheimerParameters(
            a=0.1028, m=3.4817, x0=0.1827, x1=2.6530, cbar=3.2367, delta0=0.0
        ),
        radiative_anchors=((1.0, 1.070e-2), (10.0, 1.542e-1), (20.0, 3.485e-1)),
    ),
    MaterialData(
        name="cortical_bone",
        density=1.850,
        electrons_per_gram=electrons_per_gram_from_composition(_CORTICAL_BONE_COMPOSITION),
        composition=_CORTICAL_BONE_COMPOSITION,
        mean_excitation_mev=106.4e-6,
        sternheimer=SternheimerParameters(
            a=0.0620, m=3.5919, x0=0.1161, x1=3.0919, cbar=3.6488, delta0=0.0
        ),
        radiative_anchors=((1.0, 1.824e-2), (10.0, 2.476e-1), (20.0, 5.525e-1)),
    ),
    # Tungsten (BLD workstream, 2026-07-15): density and I from the PDG 2024
    # atomic-properties page; Sternheimer row from the same page's muE header (the
    # SBS-1984 tungsten row); radiative anchors from NIST ESTAR element 074. See
    # the TUNGSTEN docstring for the pure-W-at-alloy-density approximation.
    MaterialData(
        name="tungsten",
        density=19.30,
        electrons_per_gram=electrons_per_gram_from_composition(((74, 1.0),)),
        composition=((74, 1.0),),
        mean_excitation_mev=727.0e-6,  # ICRU-37 vintage, as in ESTAR and SBS-1984
        sternheimer=SternheimerParameters(
            a=0.1551, m=2.8447, x0=0.2167, x1=3.4960, cbar=5.4059, delta0=0.14
        ),
        radiative_anchors=((1.0, 1.159e-1), (10.0, 1.132), (20.0, 2.406)),
    ),
    # Tungsten heavy alloy W95/Ni3.5/Cu1.5 (Halcyon commissioning workstream,
    # 2026-08-11): the published composition/density surrogate for dual-layer MLC
    # leaves. I = 692.5 eV and the radiative anchors are NIST ESTAR's *user-defined
    # material* output for exactly this composition and density (ESTAR applies the
    # ICRU-37 Bragg-additivity rule; doi:10.18434/T4NC7P). The Sternheimer row is not
    # in SBS-1984: cbar is the exact plasma-energy relation 2 ln(I/hw_p) + 1 (which
    # reproduces the published tungsten row's 5.4059 from tungsten's own I and
    # density), a/m/x0/x1 are least-squares fitted to ESTAR's exact density-effect
    # column over its full 0.01-1000 MeV grid (max |delta error| 0.028), and
    # delta0 = 0.14 is carried from the tungsten row (metallic conduction term).
    # Gates and provenance: tests/unit/test_tungsten_alloy.py.
    MaterialData(
        name="tungsten_alloy",
        density=18.0,
        electrons_per_gram=electrons_per_gram_from_composition(
            ((28, 0.035), (29, 0.015), (74, 0.95))
        ),
        composition=((28, 0.035), (29, 0.015), (74, 0.95)),
        mean_excitation_mev=692.5e-6,
        sternheimer=SternheimerParameters(
            a=0.1564, m=2.8413, x0=0.2323, x1=3.4878, cbar=5.3699, delta0=0.14
        ),
        radiative_anchors=((1.0, 1.124e-1), (10.0, 1.104), (20.0, 2.348)),
    ),
)
"""Material registry, indexed by the ``WATER``-style integer constants."""
