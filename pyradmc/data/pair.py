r"""Screened Bethe-Heitler pair-production data: screening parameters and the DCS.

Host-side companion of :mod:`pyradmc.physics.pair`. The *total* pair cross-section
is whatever the :class:`~pyradmc.data.interface.CrossSectionSource` reports for
``PhotonProcess.PAIR``; this module supplies only what fixes the **shape** of the
energy sharing between the electron and the positron:

- the per-material screening parameters, derived from the elemental composition
  (:func:`pair_screening_parameters`), and
- the differential cross-section in the lepton's energy fraction
  (:func:`pair_sharing_dcs`), the analytic oracle the sampler is tested against.

Model
-----
Bethe-Heitler in the Born approximation with Thomas-Fermi screening, in the
Butcher-Messel parameterization used by EGS4 and by Geant4's
``G4BetheHeitlerModel``. With :math:`\varepsilon = E_\pm / k` the *total* energy
fraction of one lepton and :math:`\varepsilon_0 = m_e c^2 / k`:

.. math::

    \frac{d\sigma}{d\varepsilon} \propto
        \left[\varepsilon^2 + (1-\varepsilon)^2\right]
        \left[\Phi_1(\delta) - \tfrac12 F(Z)\right]
      + \tfrac23\,\varepsilon(1-\varepsilon)
        \left[\Phi_2(\delta) - \tfrac12 F(Z)\right],
    \qquad
    \delta = \frac{136}{Z^{1/3}}\,\frac{\varepsilon_0}{\varepsilon(1-\varepsilon)},

on :math:`\varepsilon_0 \le \varepsilon \le 1 - \varepsilon_0`, with

.. math::

    F(Z) = \begin{cases}
        \tfrac83 \ln Z, & k < 50\ \mathrm{MeV} \\
        \tfrac83 \ln Z + 8 f_c(Z), & k \ge 50\ \mathrm{MeV}
    \end{cases}

and :math:`f_c` the Coulomb correction (:func:`coulomb_correction`).

References: Bethe & Heitler, Proc. R. Soc. A 146, 83 (1934),
doi:10.1098/rspa.1934.0140; Butcher & Messel, Nucl. Phys. 20, 15 (1960),
doi:10.1016/0029-5582(60)90162-0 (screening-function fits); Davies, Bethe &
Maximon, Phys. Rev. 93, 788 (1954), doi:10.1103/PhysRev.93.788 (Coulomb
correction); Nelson, Hirayama & Rogers, *The EGS4 Code System*, SLAC-265 (1985),
sec. 2.7 (mixture rule, 50 MeV switch); Agostinelli et al., NIM A 506, 250 (2003),
doi:10.1016/S0168-9002(03)01368-8 with the Geant4 Physics Reference Manual,
"Gamma conversion into e+e- pair" (the form transcribed here).

Stated approximations
---------------------
- **Below 2 MeV the sharing is uniform** in kinetic energy. The Born/screened
  expression above turns negative over the whole kinematic interval close to
  threshold (below about 1.8 MeV for every element), so it cannot be used there;
  EGS4 (below 2.1 MeV) and Geant4 (below 2 MeV) make the same substitution. The
  2 MeV switch is a discontinuity of the *shape*, not of the total.
- Where the expression is negative above 2 MeV (energy fractions near the
  kinematic endpoints at a few MeV) the DCS is taken as zero, as both codes do.
- **Symmetric sharing.** The Born approximation does not distinguish the charges;
  the true low-energy asymmetry (the positron is repelled by the nucleus) is
  neglected.
- The Coulomb correction is applied only at and above 50 MeV, a step in the shape
  that is negligible for low-Z media and several percent of the DCS in tungsten.
- **Triplet production is not distinguished** (existing approximation of the
  photon channels): the field-of-the-electron contribution enters only through the
  :math:`Z(Z+\xi)` element weights of the mixture rule.
- No Landau-Pomeranchuk-Migdal suppression; irrelevant below the GeV scale.

Validity: threshold to a few hundred MeV for the shape as described; intended use
1.022 to 300 MeV.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data.materials import STANDARD_ATOMIC_WEIGHT

__all__ = [
    "FINE_STRUCTURE_CONSTANT",
    "PAIR_COULOMB_ENERGY_MEV",
    "PAIR_LOW_ENERGY_LIMIT_MEV",
    "PairScreening",
    "coulomb_correction",
    "pair_screening_parameters",
    "pair_sharing_dcs",
    "radiation_logarithms",
    "screening_phi1",
    "screening_phi2",
]

FINE_STRUCTURE_CONSTANT: float = 7.297_352_5693e-3
"""Fine-structure constant alpha (CODATA 2018)."""

PAIR_LOW_ENERGY_LIMIT_MEV: float = 2.0
"""Photon energy below which the sharing is uniform; see the module docstring.

Must equal :data:`pyradmc.physics.pair.PAIR_LOW_ENERGY_LIMIT_MEV` (test-pinned
through the sampler's chi-squared against :func:`pair_sharing_dcs`)."""

PAIR_COULOMB_ENERGY_MEV: float = 50.0
"""Photon energy at and above which the Coulomb correction is applied."""

_SCREENING_PREFACTOR: float = 136.0
"""The 136 of delta = 136 Z^(-1/3) eps0 / (eps (1 - eps)) (Butcher & Messel)."""

# Tsai, Rev. Mod. Phys. 46, 815 (1974), doi:10.1103/RevModPhys.46.815, table B.2:
# (L_rad, L'_rad) for the light elements where the Thomas-Fermi model fails.
_TSAI_LIGHT_ELEMENTS: dict[int, tuple[float, float]] = {
    1: (5.31, 6.144),
    2: (4.79, 5.621),
    3: (4.74, 5.805),
    4: (4.71, 5.924),
}


class PairScreening(NamedTuple):
    """Per-material inputs of the pair energy-sharing sampler; all dimensionless.

    Three scalars per material, so they cross into a kernel as a small per-material
    array, not as an energy table.

    Attributes
    ----------
    screening_constant
        ``136 Z_eff^(-1/3)``: multiplies ``eps0 / (eps (1 - eps))`` into the
        screening variable delta.
    coulomb_low
        ``F`` below 50 MeV, ``(8/3) <ln Z>``.
    coulomb_high
        ``F`` at and above 50 MeV, ``(8/3) <ln Z> + 8 <f_c>``.
    """

    screening_constant: float
    coulomb_low: float
    coulomb_high: float


def coulomb_correction(z: float) -> float:
    r"""Coulomb correction :math:`f_c(Z)` to the Born pair cross-section.

    .. math::

        f_c(Z) = a^2 \left[ \frac{1}{1+a^2} + 0.20206 - 0.0369\,a^2
                 + 0.0083\,a^4 - 0.002\,a^6 \right], \qquad a = \alpha Z,

    Davies, Bethe & Maximon (1954), doi:10.1103/PhysRev.93.788: their four-figure
    approximation to :math:`a^2 \sum_{n\ge1} 1/[n(n^2+a^2)]`, valid to ``a = 2/3``
    (the series itself is the unit-test oracle).
    """
    a2 = (FINE_STRUCTURE_CONSTANT * z) ** 2
    return a2 * (1.0 / (1.0 + a2) + 0.20206 - 0.0369 * a2 + 0.0083 * a2 * a2 - 0.002 * a2**3)


def radiation_logarithms(z: int) -> tuple[float, float]:
    r"""Tsai's radiation logarithms ``(L_rad, L'_rad)`` for element ``z``.

    :math:`L_{rad} = \ln(184.15\,Z^{-1/3})` and
    :math:`L'_{rad} = \ln(1194\,Z^{-2/3})` for ``Z > 4``; tabulated values for
    ``Z <= 4``. Tsai (1974), doi:10.1103/RevModPhys.46.815, table B.2. They are the
    nuclear-field and electron-field screening logarithms of the complete-screening
    limit; here they only weight the elements of a mixture.
    """
    if z < 1:
        raise ValueError(f"atomic number must be positive, got {z}")
    light = _TSAI_LIGHT_ELEMENTS.get(z)
    if light is not None:
        return light
    return math.log(184.15 * z ** (-1.0 / 3.0)), math.log(1194.0 * z ** (-2.0 / 3.0))


def pair_screening_parameters(composition: tuple[tuple[int, float], ...]) -> PairScreening:
    r"""Screening parameters of a material from its elemental mass fractions.

    The EGS4 mixture rule (SLAC-265, sec. 2.7): each element contributes to the
    pair yield in proportion to :math:`p_i Z_i (Z_i + \xi_i)`, with :math:`p_i` the
    atom-number fraction and

    .. math::

        \xi_i = \frac{L'_{rad}(Z_i)}{L_{rad}(Z_i) - f_c(Z_i)}

    the electron-field (triplet) to nuclear-field ratio. The DCS is linear in
    :math:`\ln Z` and :math:`f_c`, so those average exactly; the screening
    variable is not linear in :math:`Z^{-1/3}`, and the rule replaces it by its
    weighted geometric mean, :math:`Z_{eff}^{-1/3} = \exp(-\tfrac13 \langle \ln Z
    \rangle)` — the approximation EGS4 makes, exact for an element. (Geant4
    instead picks one element per event; the two agree to well under a percent of
    the DCS for tissue-like mixtures, where one element dominates the weights.)

    Parameters
    ----------
    composition
        ``((Z, mass_fraction), ...)`` as on
        :class:`~pyradmc.data.materials.MaterialData`; need not be normalised.
    """
    if not composition:
        raise ValueError("empty composition")
    total = 0.0
    mean_log_z = 0.0
    mean_coulomb = 0.0
    for z, mass_fraction in composition:
        l_rad, l_rad_prime = radiation_logarithms(z)
        f_c = coulomb_correction(z)
        xi = l_rad_prime / (l_rad - f_c)
        weight = mass_fraction / STANDARD_ATOMIC_WEIGHT[z] * z * (z + xi)
        total += weight
        mean_log_z += weight * math.log(z)
        mean_coulomb += weight * f_c
    if total <= 0.0:
        raise ValueError("composition has no positive mass fraction")
    mean_log_z /= total
    mean_coulomb /= total
    coulomb_low = 8.0 / 3.0 * mean_log_z
    return PairScreening(
        screening_constant=_SCREENING_PREFACTOR * math.exp(-mean_log_z / 3.0),
        coulomb_low=coulomb_low,
        coulomb_high=coulomb_low + 8.0 * mean_coulomb,
    )


def screening_phi1(delta: float) -> float:
    r"""Butcher-Messel screening function :math:`\Phi_1(\delta)`.

    .. math::

        \Phi_1(\delta) = \begin{cases}
            20.867 - 3.242\,\delta + 0.625\,\delta^2, & \delta \le 1 \\
            21.12 - 4.184 \ln(\delta + 0.952), & \delta > 1
        \end{cases}

    Butcher & Messel (1960), doi:10.1016/0029-5582(60)90162-0: fits to the
    Thomas-Fermi screening integrals of Bethe & Heitler; :math:`\Phi_1(0)`
    reproduces the complete-screening :math:`4 \ln 183`. The two branches differ by
    about 0.07 at ``delta = 1`` — a property of the published fit, kept as is.
    """
    if delta > 1.0:
        return 21.12 - 4.184 * math.log(delta + 0.952)
    return 20.867 - 3.242 * delta + 0.625 * delta * delta


def screening_phi2(delta: float) -> float:
    r"""Butcher-Messel screening function :math:`\Phi_2(\delta)`.

    .. math::

        \Phi_2(\delta) = \begin{cases}
            20.209 - 1.930\,\delta - 0.086\,\delta^2, & \delta \le 1 \\
            21.12 - 4.184 \ln(\delta + 0.952), & \delta > 1
        \end{cases}

    See :func:`screening_phi1`; :math:`\Phi_1(0) - \Phi_2(0) = 2/3` to the fit's
    accuracy, and the branches differ by about 0.13 at ``delta = 1``.
    """
    if delta > 1.0:
        return 21.12 - 4.184 * math.log(delta + 0.952)
    return 20.209 - 1.930 * delta - 0.086 * delta * delta


def pair_sharing_dcs(energy: float, epsilon: float, screening: PairScreening) -> float:
    r"""Pair DCS in the lepton total-energy fraction, unnormalised and dimensionless.

    The bracket of the module docstring's governing equation,

    .. math::

        \left[\varepsilon^2 + (1-\varepsilon)^2\right]
        \left[\Phi_1(\delta) - \tfrac12 F\right]
        + \tfrac23\,\varepsilon(1-\varepsilon)\left[\Phi_2(\delta) - \tfrac12 F\right],

    clipped at zero, and zero outside :math:`[\varepsilon_0, 1-\varepsilon_0]`.
    Below :data:`PAIR_LOW_ENERGY_LIMIT_MEV` it is 1 on the kinematic interval (the
    uniform split); below threshold it is 0. Only the shape is meaningful: the
    absolute pair cross-section comes from the ``CrossSectionSource``.

    Parameters
    ----------
    energy
        Photon energy k in MeV.
    epsilon
        Total energy of one lepton over k, ``(T + m_e c^2) / k``.
    screening
        The material's :class:`PairScreening`.
    """
    if energy <= 2.0 * ELECTRON_MASS_MEV:
        return 0.0
    eps0 = ELECTRON_MASS_MEV / energy
    if epsilon < eps0 or epsilon > 1.0 - eps0:
        return 0.0
    if energy < PAIR_LOW_ENERGY_LIMIT_MEV:
        return 1.0
    half_f = 0.5 * (
        screening.coulomb_high if energy >= PAIR_COULOMB_ENERGY_MEV else screening.coulomb_low
    )
    product = epsilon * (1.0 - epsilon)
    delta = screening.screening_constant * eps0 / product
    value = (epsilon**2 + (1.0 - epsilon) ** 2) * (
        screening_phi1(delta) - half_f
    ) + 2.0 / 3.0 * product * (screening_phi2(delta) - half_f)
    return max(0.0, value)
