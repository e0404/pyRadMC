"""Electron interaction data for water.

Reference values are NIST ESTAR (https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html)
collision and radiative stopping powers and CSDA ranges for liquid water. As with the
XCOM values in ``test_cross_sections_water.py``, they are transcribed here to a few
percent of tolerance, not as authoritative data — verify against ESTAR before trusting
them further.

Two kinds of test live here and they carry different weight:

- The **collision** stopping-power and CSDA anchors are genuine validation: the
  Berger-Seltzer formula and the density effect are implemented independently of the
  anchor values.
- The **radiative** stopping-power anchors are calibration pins: the analytic backend
  is *fitted* to them (as the pair channel is fitted to the XCOM totals), so the test
  detects regression, not truth.

The Moller consistency identity is the strongest test in the file: the closed-form
restricted stopping power and the Moller differential cross-section are independent
algebra, and the identity ties them together exactly.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import integrate

pytest.importorskip(
    "pyradmc.data.analytic",
    reason="analytic backend must exist",
)

from pyradmc import ECUT_MEV, ELECTRON_MASS_MEV
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER

# Kinetic energy (MeV) -> collision stopping power (MeV cm^2/g), water, NIST ESTAR.
ESTAR_WATER_COLLISION: dict[float, float] = {
    0.2: 2.793,
    0.5: 2.034,
    1.0: 1.849,
    2.0: 1.824,
    10.0: 1.968,
    20.0: 2.046,
}

# Kinetic energy (MeV) -> radiative stopping power (MeV cm^2/g), water, NIST ESTAR.
# Calibration anchors: the backend is fitted to these. Regression pin only.
ESTAR_WATER_RADIATIVE: dict[float, float] = {
    1.0: 0.0128,
    10.0: 0.1813,
    20.0: 0.4008,
}

# Kinetic energy (MeV) -> CSDA range (g/cm^2), water, NIST ESTAR.
ESTAR_WATER_CSDA_RANGE: dict[float, float] = {
    0.5: 0.1766,
    1.0: 0.4367,
    10.0: 4.975,
    20.0: 9.320,
}

COLLISION_TOLERANCE = 0.03
RADIATIVE_TOLERANCE = 0.05
RANGE_TOLERANCE = 0.03


@pytest.fixture(scope="module")
def xs() -> AnalyticCrossSections:
    return AnalyticCrossSections()


@pytest.mark.parametrize(("energy", "expected"), sorted(ESTAR_WATER_COLLISION.items()))
def test_unrestricted_collision_stopping_power_matches_estar(
    xs: AnalyticCrossSections, energy: float, expected: float
) -> None:
    """Berger-Seltzer at delta_cut = T/2 reproduces the ESTAR collision column.

    The unrestricted collision stopping power is the restricted one evaluated at the
    Moller kinematic limit (the delta ray can carry at most half the kinetic energy,
    by indistinguishability).
    """
    actual = xs.restricted_stopping_power(energy, WATER, delta_cut=energy / 2.0)
    relative_error = abs(actual - expected) / expected
    assert relative_error < COLLISION_TOLERANCE, (
        f"S_col(water, {energy} MeV) = {actual:.4f} MeV cm^2/g, "
        f"ESTAR = {expected:.4f}, relative error {relative_error:.1%}"
    )


def test_restriction_reduces_stopping_power_monotonically(xs: AnalyticCrossSections) -> None:
    """Lowering delta_cut removes energy transfers from the continuous part."""
    energy = 5.0
    cuts = [0.010, 0.050, 0.200, 1.000, 2.500]
    values = [xs.restricted_stopping_power(energy, WATER, delta_cut=c) for c in cuts]
    assert all(a < b for a, b in itertools.pairwise(values)), (
        f"restricted stopping power not monotonic in delta_cut: {values}"
    )


@pytest.mark.parametrize(("energy", "expected"), sorted(ESTAR_WATER_RADIATIVE.items()))
def test_radiative_stopping_power_pins_calibration(
    xs: AnalyticCrossSections, energy: float, expected: float
) -> None:
    """Regression pin on the radiative fit; see the module docstring."""
    actual = xs.radiative_stopping_power(energy, WATER)
    relative_error = abs(actual - expected) / expected
    assert relative_error < RADIATIVE_TOLERANCE, (
        f"S_rad(water, {energy} MeV) = {actual:.5f} MeV cm^2/g, "
        f"ESTAR = {expected:.5f}, relative error {relative_error:.1%}"
    )


@pytest.mark.parametrize(("energy", "expected"), sorted(ESTAR_WATER_CSDA_RANGE.items()))
def test_csda_range_matches_estar(
    xs: AnalyticCrossSections, energy: float, expected: float
) -> None:
    """The range integral of the total stopping power reproduces ESTAR.

    This compounds the collision and radiative parameterizations over the whole
    slowing-down history, so it is a stronger check than any single-energy anchor.
    """
    actual = xs.csda_range(energy, WATER)
    relative_error = abs(actual - expected) / expected
    assert relative_error < RANGE_TOLERANCE, (
        f"R_CSDA(water, {energy} MeV) = {actual:.4f} g/cm^2, "
        f"ESTAR = {expected:.4f}, relative error {relative_error:.1%}"
    )


class TestMollerConsistency:
    """S_unrestricted - S_restricted(cut) must equal the Moller moment above the cut.

    The closed-form Berger-Seltzer G term and the Moller differential cross-section
    are independent algebra; the identity

        S(T, T/2) - S(T, cut) = N_e * integral_cut^{T/2} W (dsigma/dW) dW

    holds exactly by construction of the Class II split. A sign or factor error in
    either piece breaks it at the percent level or worse.
    """

    @pytest.mark.parametrize("energy", [0.5, 2.0, 10.0])
    @pytest.mark.parametrize("cut", [0.05, 0.2])
    def test_identity(self, xs: AnalyticCrossSections, energy: float, cut: float) -> None:
        from pyradmc.data.analytic import moller_dcs_per_electron
        from pyradmc.data.materials import MATERIALS

        lhs = xs.restricted_stopping_power(
            energy, WATER, delta_cut=energy / 2.0
        ) - xs.restricted_stopping_power(energy, WATER, delta_cut=cut)

        electrons = MATERIALS[WATER].electrons_per_gram
        moment, _ = integrate.quad(
            lambda w: w * moller_dcs_per_electron(energy, w), cut, energy / 2.0
        )
        rhs = electrons * moment

        assert lhs == pytest.approx(rhs, rel=1e-4), (
            f"Moller consistency broken at T={energy} MeV, cut={cut} MeV: "
            f"stopping-power difference {lhs:.6e}, Moller moment {rhs:.6e}"
        )


class TestMollerCrossSection:
    """The restricted Moller cross-section accessor on the cross-section interface."""

    def test_zero_below_threshold(self, xs: AnalyticCrossSections) -> None:
        """No delta above the cut is kinematically possible unless T > 2 * cut."""
        assert xs.moller_cross_section(0.3, WATER, delta_cut=0.2) == 0.0
        assert xs.moller_cross_section(0.4, WATER, delta_cut=0.2) == 0.0

    def test_matches_quadrature_of_the_dcs(self, xs: AnalyticCrossSections) -> None:
        """The closed-form total equals the numeric integral of the DCS."""
        from pyradmc.data.analytic import moller_dcs_per_electron
        from pyradmc.data.materials import MATERIALS

        energy, cut = 5.0, 0.2
        total, _ = integrate.quad(lambda w: moller_dcs_per_electron(energy, w), cut, energy / 2.0)
        expected = MATERIALS[WATER].electrons_per_gram * total
        actual = xs.moller_cross_section(energy, WATER, delta_cut=cut)
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_grows_as_the_cut_drops(self, xs: AnalyticCrossSections) -> None:
        """More of the delta spectrum is above a lower cut."""
        energy = 5.0
        sigma_low = xs.moller_cross_section(energy, WATER, delta_cut=0.05)
        sigma_high = xs.moller_cross_section(energy, WATER, delta_cut=0.5)
        assert sigma_low > sigma_high > 0.0


def test_scattering_power_magnitude_and_trend(xs: AnalyticCrossSections) -> None:
    """Mass scattering power decreases steeply with energy and has a sane magnitude.

    Sanity bounds, not a precision pin: T/rho for water at 1 MeV is of order a few
    rad^2 cm^2/g (Rossi formula with X0 = 36.08 g/cm^2), and falls roughly as 1/(p v)^2.
    """
    t_1mev = xs.scattering_power(1.0, WATER, ECUT_MEV)
    t_10mev = xs.scattering_power(10.0, WATER, ECUT_MEV)
    assert 1.0 < t_1mev < 20.0
    assert t_10mev < t_1mev / 10.0

    energies = np.geomspace(0.2, 20.0, 30)
    values = [xs.scattering_power(float(e), WATER, ECUT_MEV) for e in energies]
    assert all(a > b for a, b in itertools.pairwise(values))


def test_electron_mass_constant_is_consistent() -> None:
    """The tests above assume ELECTRON_MASS_MEV; keep the assumption visible."""
    assert pytest.approx(0.511, abs=1e-3) == ELECTRON_MASS_MEV
