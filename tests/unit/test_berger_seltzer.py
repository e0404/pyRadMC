"""Material-parameterized Berger-Seltzer electron stopping .

The formulas the analytic backend has always evaluated for water — Berger-Seltzer
restricted collision stopping, the Sternheimer density effect, the restricted Moller
cross-section, the log-quadratic ESTAR radiative fit, and the CSDA range integral —
move to ``pyradmc.data.berger_seltzer`` as pure functions of a ``MaterialData``, so the
precompiler can evaluate them for any registry material. The analytic backend delegates
with the water entry; the delegation-identity tests here pin that the refactor moved
the oracle, verbatim, rather than changing it (AGENTS.md 2.2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.berger_seltzer import (
    csda_range_table,
    density_effect,
    radiative_fit_coefficients,
    radiative_stopping,
    restricted_collision_stopping,
    restricted_moller_cross_section,
)
from pyradmc.data.materials import MATERIALS, WATER, MaterialData, SternheimerParameters

DELTA_CUT = 0.2
ENERGIES = (0.05, 0.2, 0.5, 1.0, 2.0, 6.0, 10.0, 20.0)

WATER_DATA = MATERIALS[WATER]


def _synthetic(mean_excitation_mev: float) -> MaterialData:
    """A water-like material whose I-value can be dialed."""
    return MaterialData(
        name="synthetic",
        density=1.0,
        electrons_per_gram=WATER_DATA.electrons_per_gram,
        composition=WATER_DATA.composition,
        mean_excitation_mev=mean_excitation_mev,
        sternheimer=WATER_DATA.sternheimer,
        radiative_anchors=WATER_DATA.radiative_anchors,
    )


@pytest.fixture(scope="module")
def xs() -> AnalyticCrossSections:
    return AnalyticCrossSections()


class TestDelegationIdentity:
    """The analytic backend's water answers are these functions with the water entry.

    Exact equality, not approx: the analytic source *delegates*, so any difference
    means two diverging copies of the oracle exist — the failure mode this module
    exists to prevent.
    """

    @pytest.mark.parametrize("energy", ENERGIES)
    def test_restricted_stopping(self, xs: AnalyticCrossSections, energy: float) -> None:
        assert restricted_collision_stopping(
            energy, WATER_DATA, DELTA_CUT
        ) == xs.restricted_stopping_power(energy, WATER, DELTA_CUT)

    @pytest.mark.parametrize("energy", ENERGIES)
    def test_moller(self, xs: AnalyticCrossSections, energy: float) -> None:
        assert restricted_moller_cross_section(
            energy, WATER_DATA, DELTA_CUT
        ) == xs.moller_cross_section(energy, WATER, DELTA_CUT)

    @pytest.mark.parametrize("energy", ENERGIES)
    def test_radiative(self, xs: AnalyticCrossSections, energy: float) -> None:
        coeffs = radiative_fit_coefficients(WATER_DATA.radiative_anchors)
        assert radiative_stopping(energy, coeffs) == xs.radiative_stopping_power(energy, WATER)

    @pytest.mark.parametrize("energy", ENERGIES)
    def test_csda_range(self, xs: AnalyticCrossSections, energy: float) -> None:
        log_energies, ranges = csda_range_table(WATER_DATA)
        table_value = float(np.interp(math.log(energy), log_energies, ranges))
        assert table_value == xs.csda_range(energy, WATER)


class TestMaterialDependence:
    """The formulas respond to the material data the way the physics says they must."""

    def test_higher_mean_excitation_lowers_stopping(self) -> None:
        """dS/dI < 0: a tighter-bound medium stops less (Bethe logarithm)."""
        soft = restricted_collision_stopping(1.0, _synthetic(63.2e-6), DELTA_CUT)
        hard = restricted_collision_stopping(1.0, _synthetic(106.4e-6), DELTA_CUT)
        assert soft > hard

    def test_stopping_scales_with_electron_density(self) -> None:
        """The prefactor is linear in N_e at fixed I and delta parameters."""
        doubled = MaterialData(
            name="doubled",
            density=1.0,
            electrons_per_gram=2.0 * WATER_DATA.electrons_per_gram,
            composition=WATER_DATA.composition,
            mean_excitation_mev=WATER_DATA.mean_excitation_mev,
            sternheimer=WATER_DATA.sternheimer,
            radiative_anchors=WATER_DATA.radiative_anchors,
        )
        assert restricted_collision_stopping(1.0, doubled, DELTA_CUT) == pytest.approx(
            2.0 * restricted_collision_stopping(1.0, WATER_DATA, DELTA_CUT), rel=1e-12
        )
        assert restricted_moller_cross_section(1.0, doubled, DELTA_CUT) == pytest.approx(
            2.0 * restricted_moller_cross_section(1.0, WATER_DATA, DELTA_CUT), rel=1e-12
        )

    def test_radiative_fit_passes_through_anchors(self) -> None:
        """The log-quadratic is exact at its three anchors, whatever they are."""
        anchors = ((1.0, 0.0182), (10.0, 0.2476), (20.0, 0.5525))  # ESTAR cortical bone
        coeffs = radiative_fit_coefficients(anchors)
        for energy, expected in anchors:
            assert radiative_stopping(energy, coeffs) == pytest.approx(expected, rel=1e-12)

    def test_csda_range_is_monotone(self) -> None:
        _, ranges = csda_range_table(WATER_DATA)
        assert np.all(np.diff(ranges) > 0.0)


class TestDensityEffect:
    """The Sternheimer piecewise form, on the water row."""

    PARAMS = WATER_DATA.sternheimer

    def test_asymptotic_form_above_x1(self) -> None:
        """delta -> 4.6052 x - Cbar for x >= x1 (complete screening)."""
        tau = 1000.0 / 0.51099895  # ~1 GeV: x ~ 3.3, above water's x1 = 2.8
        x = 0.5 * math.log10(tau * (tau + 2.0))
        assert x >= self.PARAMS.x1
        assert density_effect(tau, self.PARAMS) == pytest.approx(
            4.6052 * x - self.PARAMS.cbar, rel=1e-12
        )

    def test_conduction_term_below_x0(self) -> None:
        """Water's delta0 = 0.097 gives a small positive delta below x0."""
        tau = 0.05 / 0.51099895  # 50 keV: x below water's x0 = 0.24
        x = 0.5 * math.log10(tau * (tau + 2.0))
        assert x < self.PARAMS.x0
        delta = density_effect(tau, self.PARAMS)
        assert 0.0 < delta < 0.097

    def test_zero_delta0_kills_the_conduction_term(self) -> None:
        params = SternheimerParameters(
            a=self.PARAMS.a,
            m=self.PARAMS.m,
            x0=self.PARAMS.x0,
            x1=self.PARAMS.x1,
            cbar=self.PARAMS.cbar,
            delta0=0.0,
        )
        assert density_effect(0.05 / 0.51099895, params) == 0.0
