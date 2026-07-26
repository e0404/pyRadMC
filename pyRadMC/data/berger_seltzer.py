"""ICRU-37 electron stopping quantities as functions of a material.

The Berger-Seltzer restricted collision stopping power, the Sternheimer density
effect, the restricted Moller cross-section, the log-quadratic ESTAR radiative fit,
and the CSDA range integral — the formulas the analytic backend evaluated for water
— parameterized by :class:`~pyRadMC.data.materials.MaterialData` so the
tabulated precompiler can evaluate them for any registry material (multi-material
work). The analytic backend delegates here with the water entry; the delegation
identity is test-pinned (``tests/unit/test_berger_seltzer.py``), because these
formulas *are* the electron oracle and moving them must not change them
(AGENTS.md 2.2).

These are host-side, compile-time functions (construction of tables and compiled
data); no kernel ever calls them. Governing equations and citations sit on each
function. All energies in MeV; mass stopping powers in MeV cm^2/g; cross-sections
per unit mass in cm^2/g; ranges in g/cm^2.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pyRadMC import ELECTRON_MASS_MEV
from pyRadMC.data.materials import MaterialData, SternheimerParameters

__all__ = [
    "CLASSICAL_ELECTRON_RADIUS_CM",
    "csda_range_table",
    "density_effect",
    "moller_dcs_per_electron",
    "radiative_fit_coefficients",
    "radiative_stopping",
    "restricted_collision_stopping",
    "restricted_moller_cross_section",
]

CLASSICAL_ELECTRON_RADIUS_CM: float = 2.817_940_3262e-13
"""Classical electron radius in cm (CODATA 2018)."""

TWO_PI_RE2_MC2: float = 2.0 * math.pi * CLASSICAL_ELECTRON_RADIUS_CM**2 * ELECTRON_MASS_MEV
"""2 pi r_e^2 m_e c^2 in MeV cm^2: the Moller/Bethe prefactor per electron."""


def density_effect(tau: float, params: SternheimerParameters) -> float:
    """Sternheimer density-effect correction delta.

    Piecewise in x = log10(p / m_e c) = log10(sqrt(tau (tau + 2))): zero-ish below
    x0 (insulator form with the small delta0 conduction term), the standard
    a (x1 - x)^m interpolation between x0 and x1, and the asymptotic 4.6052 x - Cbar
    above. Sternheimer, Berger & Seltzer (1984), doi:10.1016/0092-640X(84)90002-0.
    The coefficients were computed at the material's reference density; evaluating
    them for a voxel at a different density is the density-scaling approximation
    every mass-quantity lookup in this engine makes (see ``MaterialData.density``).
    """
    a, m, x0, x1, cbar, delta0 = params
    x = 0.5 * math.log10(tau * (tau + 2.0))
    if x >= x1:
        return 4.6052 * x - cbar
    if x >= x0:
        return 4.6052 * x - cbar + a * math.pow(x1 - x, m)
    return delta0 * math.pow(10.0, 2.0 * (x - x0))


def moller_dcs_per_electron(energy: float, energy_transfer: float) -> float:
    r"""Moller differential cross-section per electron, in cm^2/MeV.

    For primary kinetic energy T and delta kinetic energy W, with x = W/T:

    .. math::

        \frac{d\sigma}{dW} = \frac{2\pi r_e^2 mc^2}{\beta^2 T^2}
            \left[ \frac{1}{x^2} + \frac{1}{(1-x)^2}
                 + \left(\frac{\tau}{\tau+1}\right)^2
                 - \frac{2\tau+1}{(\tau+1)^2}\,\frac{1}{x(1-x)} \right],
        \qquad 0 < x \le \tfrac12.

    Moller (1932), doi:10.1002/andp.19324060506; e.g. Salvat et al., PENELOPE-2018,
    eq. (3.111). Zero outside the kinematic interval. Material-free (per electron);
    exposed so tests can integrate it independently of the closed forms below.
    """
    if energy_transfer <= 0.0 or energy_transfer > energy / 2.0:
        return 0.0
    tau = energy / ELECTRON_MASS_MEV
    beta_sq = tau * (tau + 2.0) / (tau + 1.0) ** 2
    x = energy_transfer / energy
    f = (
        1.0 / x**2
        + 1.0 / (1.0 - x) ** 2
        + (tau / (tau + 1.0)) ** 2
        - (2.0 * tau + 1.0) / (tau + 1.0) ** 2 / (x * (1.0 - x))
    )
    return TWO_PI_RE2_MC2 / (beta_sq * energy**2) * f


def restricted_collision_stopping(energy: float, material: MaterialData, delta_cut: float) -> float:
    r"""Restricted collision stopping power, Berger-Seltzer form, in MeV cm^2/g.

    .. math::

        S(T, \Delta)/\rho = \frac{2\pi r_e^2 m c^2 N_e}{\beta^2}
            \left[ \ln\frac{\tau^2(\tau+2)}{2 (I/mc^2)^2}
                   + G^-(\tau, w) - \delta \right],
        \qquad w = \Delta/T,

    with the electron term (ICRU Report 37 (1984); Salvat et al., PENELOPE-2018,
    sec. 3.2.3, doi:10.1787/32da5043-en)

    .. math::

        G^-(\tau, w) = -1 - \beta^2 + \ln\bigl(4(1-w)w\bigr) + \frac{1}{1-w}
            + \frac{(\tau w)^2/2 + (2\tau+1)\ln(1-w)}{(\tau+1)^2}

    and the Sternheimer density effect :math:`\delta` from the material's
    coefficients. At ``delta_cut = T/2`` this is the unrestricted collision
    stopping power (the ESTAR collision column); that limit and the Moller
    consistency identity are both test-pinned.

    Shell corrections are neglected — percent-level below ~50 keV, far below ECUT's
    influence on dose at this engine's energies (stated approximation, carried over
    from the water-only implementation this generalizes).
    """
    if energy <= 0.0:
        raise ValueError(f"non-positive electron kinetic energy {energy} MeV")
    tau = energy / ELECTRON_MASS_MEV
    beta_sq = tau * (tau + 2.0) / (tau + 1.0) ** 2
    w = min(delta_cut / energy, 0.5)
    if w <= 0.0:
        raise ValueError(f"non-positive delta_cut {delta_cut} MeV")

    g_minus = (
        -1.0
        - beta_sq
        + math.log(4.0 * (1.0 - w) * w)
        + 1.0 / (1.0 - w)
        + ((tau * w) ** 2 / 2.0 + (2.0 * tau + 1.0) * math.log(1.0 - w)) / (tau + 1.0) ** 2
    )
    log_term = math.log(
        tau**2 * (tau + 2.0) / (2.0 * (material.mean_excitation_mev / ELECTRON_MASS_MEV) ** 2)
    )
    prefactor = TWO_PI_RE2_MC2 * material.electrons_per_gram / beta_sq
    return prefactor * (log_term + g_minus - density_effect(tau, material.sternheimer))


def restricted_moller_cross_section(
    energy: float, material: MaterialData, delta_cut: float
) -> float:
    r"""Restricted Moller cross-section per unit mass, in cm^2/g.

    Closed-form integral of the Moller DCS (:func:`moller_dcs_per_electron`) from
    ``delta_cut`` to ``T/2``:

    .. math::

        \sigma(T, \Delta) = \frac{2\pi r_e^2 mc^2 N_e}{\beta^2 T}
            \left[ \left(\frac{\tau}{\tau+1}\right)^2\left(\tfrac12 - w\right)
            + \frac{1}{w} - \frac{1}{1-w}
            + \frac{2\tau+1}{(\tau+1)^2} \ln\frac{w}{1-w} \right]

    Moller (1932), doi:10.1002/andp.19324060506. Zero at or below ``2 delta_cut``.
    """
    if energy <= 0.0:
        raise ValueError(f"non-positive electron kinetic energy {energy} MeV")
    if energy <= 2.0 * delta_cut:
        return 0.0
    tau = energy / ELECTRON_MASS_MEV
    beta_sq = tau * (tau + 2.0) / (tau + 1.0) ** 2
    w = delta_cut / energy

    bracket = (
        (tau / (tau + 1.0)) ** 2 * (0.5 - w)
        + 1.0 / w
        - 1.0 / (1.0 - w)
        + (2.0 * tau + 1.0) / (tau + 1.0) ** 2 * math.log(w / (1.0 - w))
    )
    return TWO_PI_RE2_MC2 * material.electrons_per_gram / (beta_sq * energy) * bracket


def radiative_fit_coefficients(
    anchors: tuple[tuple[float, float], ...],
) -> tuple[float, float, float]:
    """Exact log-quadratic through three radiative stopping anchors.

    The radiative channel is a calibration, not a theory (see the analytic backend's
    module docstring): each material carries three NIST ESTAR anchors
    (``MaterialData.radiative_anchors``) and the fit passes through them exactly.
    """
    if len(anchors) != 3:
        raise ValueError(f"the exact log-quadratic needs exactly 3 anchors, got {len(anchors)}")
    log_e = np.log([e for e, _ in anchors])
    log_s = np.log([s for _, s in anchors])
    design = np.stack([np.ones_like(log_e), log_e, log_e**2], axis=1)
    c = np.linalg.solve(design, log_s)
    return float(c[0]), float(c[1]), float(c[2])


def radiative_stopping(energy: float, coefficients: tuple[float, float, float]) -> float:
    """Radiative stopping power from the log-quadratic fit, in MeV cm^2/g.

    Below the lowest anchor (1 MeV) it extrapolates smoothly; there the radiative
    share of the total stopping power is under one percent, so the extrapolation
    error is dosimetrically irrelevant.
    """
    if energy <= 0.0:
        raise ValueError(f"non-positive electron kinetic energy {energy} MeV")
    log_e = math.log(energy)
    c0, c1, c2 = coefficients
    return math.exp(c0 + c1 * log_e + c2 * log_e**2)


def csda_range_table(
    material: MaterialData,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """CSDA range lookup ``(log_energies, ranges)`` in (log MeV, g/cm^2).

    Trapezoidal cumulative integration of ``1/S_total`` on a dense logarithmic grid
    from 1 keV to 30 MeV, with the total the unrestricted collision plus radiative
    stopping. Errs on the *under*-estimating side (the grid starts above zero,
    truncating the sub-keV tail) by well under 1e-4 g/cm^2; see the analytic
    backend's ``csda_range`` docstring for why that side is acceptable. 600
    log-spaced points hold the interpolation error well under the 3 percent test
    tolerance.
    """
    coefficients = radiative_fit_coefficients(material.radiative_anchors)
    energies = np.geomspace(1.0e-3, 30.0, 600)
    inverse_total = np.array(
        [
            1.0
            / (
                restricted_collision_stopping(float(e), material, delta_cut=float(e) / 2.0)
                + radiative_stopping(float(e), coefficients)
            )
            for e in energies
        ]
    )
    steps = np.diff(energies) * 0.5 * (inverse_total[1:] + inverse_total[:-1])
    ranges = np.concatenate(([0.0], np.cumsum(steps)))
    return np.log(energies), ranges
