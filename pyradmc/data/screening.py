r"""Atomic screening constants shared by the two Bethe-Heitler models.

Bremsstrahlung (:mod:`pyradmc.data.bethe_heitler`) and pair production
(:mod:`pyradmc.data.pair`) are one matrix element read in two channels, and both
need the same three things about an element: the fine-structure constant, the
Coulomb correction to the Born approximation, and the radiation logarithms of the
complete-screening limit. They are defined here once, so a medium cannot be
screened by two slightly different numbers for the photon that makes a pair and the
lepton that radiates (``tests/unit/test_screening_shared.py``).

What is *not* shared, on purpose: the two models' mixture rules
(:func:`pyradmc.data.bethe_heitler.screening_parameters`,
:func:`pyradmc.data.pair.pair_screening_parameters`) and their screening functions
(Tsai's Thomas-Fermi fits against the Butcher-Messel fits). Those are different
quantities, not duplicates.

Host-side only; nothing here is called from a kernel.
"""

from __future__ import annotations

import math

__all__ = [
    "FINE_STRUCTURE_CONSTANT",
    "TSAI_PHI1_COMPLETE",
    "TSAI_PSI1_COMPLETE",
    "coulomb_correction",
    "radiation_logarithms",
]

FINE_STRUCTURE_CONSTANT: float = 7.297_352_5693e-3
"""Fine-structure constant alpha (CODATA 2018)."""

TSAI_PHI1_COMPLETE: float = 20.863
"""``phi_1(0)`` of Tsai's Thomas-Fermi fit, his rounded ``4 ln 184.15``."""

TSAI_PSI1_COMPLETE: float = 28.340
"""``psi_1(0)`` of Tsai's Thomas-Fermi fit, his rounded ``4 ln 1194``."""

# Tsai, Rev. Mod. Phys. 46, 815 (1974), doi:10.1103/RevModPhys.46.815, table B.2
# (reproduced as table 34.2 of the Particle Data Group review of the passage of
# particles through matter): (L_rad, L'_rad) for the light elements, where the
# Thomas-Fermi model fails.
_TSAI_LIGHT_ELEMENTS: dict[int, tuple[float, float]] = {
    1: (5.31, 6.144),
    2: (4.79, 5.621),
    3: (4.74, 5.805),
    4: (4.71, 5.924),
}


def coulomb_correction(z: float) -> float:
    r"""Davies-Bethe-Maximon Coulomb correction :math:`f_c(Z)` to the Born approximation.

    .. math::

        f_c(Z) = a^2 \left[ \frac{1}{1+a^2} + 0.20206 - 0.0369\,a^2
                 + 0.0083\,a^4 - 0.002\,a^6 \right], \qquad a = \alpha Z,

    Davies, Bethe & Maximon, Phys. Rev. 93, 788 (1954), doi:10.1103/PhysRev.93.788:
    their four-figure approximation to :math:`a^2 \sum_{n\ge1} 1/[n(n^2+a^2)]`,
    valid to ``a = 2/3`` (the polynomial as given by Tsai (1974),
    doi:10.1103/RevModPhys.46.815; the series itself is the unit-test oracle).

    It is a high-energy result. *Where* it is applied is each model's own stated
    choice, not this function's: the bremsstrahlung model applies it at every
    energy (:mod:`pyradmc.physics.brems_bh`), the pair model at and above
    :data:`pyradmc.physics.pair.PAIR_COULOMB_ENERGY_MEV` only.
    """
    a2 = (FINE_STRUCTURE_CONSTANT * z) ** 2
    return a2 * (1.0 / (1.0 + a2) + 0.20206 - 0.0369 * a2 + 0.0083 * a2 * a2 - 0.002 * a2**3)


def radiation_logarithms(z: int) -> tuple[float, float]:
    r"""Tsai's radiation logarithms ``(L_rad, L'_rad)`` for element ``z``.

    :math:`L_{rad} = \ln(184.15\,Z^{-1/3})` and
    :math:`L'_{rad} = \ln(1194\,Z^{-2/3})` for ``Z > 4``; tabulated values for
    ``Z <= 4``. Tsai (1974), doi:10.1103/RevModPhys.46.815, table B.2. They are the
    nuclear-field (elastic) and electron-field (inelastic) screening logarithms of
    the complete-screening limit.

    Evaluated as :math:`\tfrac14\phi_1(0) - \tfrac13\ln Z` and
    :math:`\tfrac14\psi_1(0) - \tfrac23\ln Z` with the constants of Tsai's
    Thomas-Fermi fit as he rounds them (:data:`TSAI_PHI1_COMPLETE`,
    :data:`TSAI_PSI1_COMPLETE`). The rounding moves the logarithms by 6.4e-7 and
    6.4e-5. This form is used because the bremsstrahlung shape function evaluates
    its screening from those same two constants; the values are pinned literally in
    ``tests/unit/test_pair_screening.py``.
    """
    if z < 1:
        raise ValueError(f"atomic number must be positive, got {z}")
    light = _TSAI_LIGHT_ELEMENTS.get(z)
    if light is not None:
        return light
    log_z = math.log(z)
    return (
        0.25 * TSAI_PHI1_COMPLETE - log_z / 3.0,
        0.25 * TSAI_PSI1_COMPLETE - 2.0 * log_z / 3.0,
    )
