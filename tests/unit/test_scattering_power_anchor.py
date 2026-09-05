r"""The multiple-scattering *strength* must be the first transport moment.

Goudsmit-Saunderson pins ``<cos theta> = exp(-s N sigma_tr)`` exactly, with
``sigma_tr = sigma_el G_1`` the transport cross-section of the single-scattering
law. The engine anchors the GS deflection to ``scattering_power`` (``T rho s`` is
the small-step ``<theta^2>``), so ``T`` *is* ``2 (N_A/A) sigma_tr`` — nothing else
is consistent with the screened-Rutherford shape the GS tables are built from.

Highland's ``(14.1 MeV / pv)^2 / X_0`` is a fit to the width of the Moliere
*core*; it lacks the ``ln(1/eta) - 1`` factor of the transport moment, which grows
with energy because ``eta ~ 1/p^2``. The full ``Z(Z+1)`` moment includes soft
electron-electron deflections, but its above-ECUT part is already transported as
explicit Moller events. This module pins the exact Class-II partition: keep the
soft electron moment and subtract only the hard moment.
"""

from __future__ import annotations

import math

import pytest
from scipy import integrate

from pyradmc import ECUT_MEV, ELECTRON_MASS_MEV
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.berger_seltzer import moller_dcs_per_electron
from pyradmc.data.goudsmit_saunderson import (
    first_transport_moment,
    hard_moller_transport_cross_section,
    moliere_screening,
)
from pyradmc.data.materials import MATERIALS, STANDARD_ATOMIC_WEIGHT, WATER
from pyradmc.physics.moller import moller_direction_cosines

_AVOGADRO = 6.02214076e23
_CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403262e-13


def _hard_moller_transport_cross_section(energy: float, delta_cut: float) -> float:
    """Independent quadrature of ``integral d sigma/dW (1-cos theta_primary)``."""
    if energy <= 2.0 * delta_cut:
        return 0.0
    value, _ = integrate.quad(
        lambda transfer: (
            moller_dcs_per_electron(energy, transfer)
            * (1.0 - moller_direction_cosines(energy, transfer)[1])
        ),
        delta_cut,
        0.5 * energy,
        epsabs=1.0e-40,
        epsrel=1.0e-11,
    )
    return float(value)


def _transport_moment_scattering_power(energy: float, delta_cut: float) -> float:
    r"""Independent Class-II moment: nuclear plus soft electron scattering.

    With the unit-charge screened-Rutherford transport cross section

    .. math::

        \sigma_{tr} = \frac{\pi r_e^2 (m c^2/pv)^2}{\eta(1+\eta)}G_1(\eta),

    the per-element contribution is
    ``Z^2 sigma_tr + Z(sigma_tr - sigma_tr,hard-Moller)``. The hard term is
    integrated numerically above rather than sharing its implementation.
    """
    tau = energy / ELECTRON_MASS_MEV
    pv = ELECTRON_MASS_MEV * tau * (tau + 2.0) / (tau + 1.0)
    composition = MATERIALS[WATER].composition
    eta = moliere_screening(composition, energy)
    g1 = first_transport_moment(eta)
    sigma_el = (
        math.pi
        * _CLASSICAL_ELECTRON_RADIUS_CM**2
        * (ELECTRON_MASS_MEV / pv) ** 2
        / (eta * (1.0 + eta))
    )
    sigma_transport = sigma_el * g1
    hard_moller = _hard_moller_transport_cross_section(energy, delta_cut)
    total = 0.0
    for z, fraction in composition:
        total += (
            2.0
            * fraction
            * _AVOGADRO
            / STANDARD_ATOMIC_WEIGHT[z]
            * (z * z * sigma_transport + z * (sigma_transport - hard_moller))
        )
    return total


@pytest.mark.parametrize("energy", [0.2, 1.0, 6.0, 10.0, 15.0, 20.0])
def test_analytic_scattering_power_is_the_transport_moment(energy: float) -> None:
    """``T`` equals ``2 (N_A/A) sigma_el G_1`` of the Moliere-screened law.

    1e-6 rather than machine precision only because the package and the GS data
    module carry the electron mass to different digits (3e-9 relative); a wrong
    formula misses by orders of magnitude.
    """
    xs = AnalyticCrossSections()
    assert xs.scattering_power(energy, WATER, ECUT_MEV) == pytest.approx(
        _transport_moment_scattering_power(energy, ECUT_MEV), rel=1.0e-6
    )


@pytest.mark.parametrize(
    ("energy", "delta_cut"),
    [(0.4001, 0.2), (0.5, 0.2), (1.0, 0.05), (6.0, 0.2), (10.0, 1.0), (20.0, 0.2)],
)
def test_hard_moller_transport_cross_section_matches_quadrature(
    energy: float, delta_cut: float
) -> None:
    """The scalar closed form equals direct DCS/kinematics quadrature."""
    assert hard_moller_transport_cross_section(energy, delta_cut) == pytest.approx(
        _hard_moller_transport_cross_section(energy, delta_cut), rel=3.0e-9
    )


def test_scattering_power_tracks_the_discrete_moller_cut() -> None:
    """Lower ECUT moves more electron scattering into explicit hard events."""
    xs = AnalyticCrossSections()
    energy = 6.0
    low_cut = xs.scattering_power(energy, WATER, 0.05)
    default_cut = xs.scattering_power(energy, WATER, ECUT_MEV)
    closed_channel = xs.scattering_power(energy, WATER, 0.5 * energy)
    assert low_cut < default_cut < closed_channel


def test_transport_moment_exceeds_the_highland_core_width_at_high_energy() -> None:
    """The log factor Highland lacks: ``T / T_Highland`` rises with energy, >1.5 at 10 MeV.

    Records the size of the correction so a future re-fit cannot silently shrink
    it. Highland's ``(14.1/pv)^2 / X_0`` with ``X_0 = 36.08 g/cm^2`` is written out
    as the historical reference, not imported from anywhere.
    """
    xs = AnalyticCrossSections()
    ratios = []
    for energy in (1.0, 6.0, 10.0, 15.0):
        tau = energy / ELECTRON_MASS_MEV
        pv = ELECTRON_MASS_MEV * tau * (tau + 2.0) / (tau + 1.0)
        highland = (14.1 / pv) ** 2 / 36.08
        ratios.append(xs.scattering_power(energy, WATER, ECUT_MEV) / highland)
    assert ratios == sorted(ratios), f"T / T_Highland must rise with energy, got {ratios}"
    assert ratios[0] > 1.1, f"1 MeV: expected the transport moment above Highland, got {ratios[0]}"
    assert ratios[2] > 1.5, f"10 MeV: expected T / T_Highland > 1.5, got {ratios[2]}"
