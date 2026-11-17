"""Positron interaction data: Bhabha, positron stopping, annihilation in flight.

No tabulated positron stopping powers are pinned here. ICRU Report 37 tabulates
them, but NIST ESTAR does not (it is electron-only) and no transcription of the
ICRU 37 positron tables was available when this was written; inventing anchor
values would be worse than having none. What stands in for them:

- the **unrestricted** positron collision stopping power against the closed
  Rohrlich-Carlson / ICRU 37 form — different algebra from the restricted
  expression the code evaluates;
- the **Bhabha consistency identity**: restricted stopping plus the energy moment
  of the Bhabha DCS above the cut equals the unrestricted stopping, with the DCS
  integrated numerically — the positron twin of the Moller identity that is the
  strongest test in ``test_electron_data.py``;
- the Bhabha DCS against an independent transcription (PENELOPE's ``b_i`` form
  against the ``B_i(y)`` form the code uses);
- the electron/positron stopping difference against the closed-form limits of
  the two ``F`` functions;
- the Heitler annihilation cross-section against its two analytic limits and
  against the integral of the differential cross-section the sampler draws from.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import integrate

from pyradmc import ECUT_MEV, ELECTRON_MASS_MEV
from pyradmc.data import berger_seltzer
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.berger_seltzer import CLASSICAL_ELECTRON_RADIUS_CM, TWO_PI_RE2_MC2
from pyradmc.data.interface import DOMAIN_EDGE_RTOL, CrossSectionSource
from pyradmc.data.materials import CORTICAL_BONE, MATERIALS, WATER
from pyradmc.data.positron import (
    annihilation_cross_section_per_electron,
    annihilation_dcs_per_electron,
    bhabha_dcs_per_electron,
    positron_restricted_collision_stopping,
    restricted_bhabha_cross_section,
)

ENERGIES = (0.25, 0.5, 1.0, 6.0, 20.0, 60.0, 200.0)
PI_RE2 = math.pi * CLASSICAL_ELECTRON_RADIUS_CM**2


@pytest.fixture(scope="module")
def xs() -> AnalyticCrossSections:
    return AnalyticCrossSections()


class _WideAnalytic(AnalyticCrossSections):
    """Analytic water claiming a 250 MeV domain: a grid-extension instrument.

    The twin of the one in ``test_energy_domain.py``. The positron collision
    stopping it integrates is the Berger-Seltzer closed form, which holds at
    these energies; nothing here asserts the photon data is valid up there.
    """

    @property
    def energy_max(self) -> float:
        return 250.0


def _bhabha_dcs_penelope(energy: float, w: float) -> float:
    """Bhabha DCS per electron, PENELOPE-2018 form (doi:10.1787/32da5043-en, sec. 3.2).

    d sigma / dW = 2 pi r_e^2 mc^2 / (beta^2 W^2)
                   [1 - b1 x + b2 x^2 - b3 x^3 + b4 x^4],  x = W / T.
    """
    gamma = 1.0 + energy / ELECTRON_MASS_MEV
    beta_sq = 1.0 - 1.0 / gamma**2
    common = ((gamma - 1.0) / gamma) ** 2
    b1 = common * (2.0 * (gamma + 1.0) ** 2 - 1.0) / (gamma**2 - 1.0)
    b2 = common * (3.0 * (gamma + 1.0) ** 2 + 1.0) / (gamma + 1.0) ** 2
    b3 = common * 2.0 * gamma * (gamma - 1.0) / (gamma + 1.0) ** 2
    b4 = common * (gamma - 1.0) ** 2 / (gamma + 1.0) ** 2
    x = w / energy
    return TWO_PI_RE2_MC2 / (beta_sq * w * w) * (1.0 - b1 * x + b2 * x**2 - b3 * x**3 + b4 * x**4)


def _unrestricted_positron_stopping(energy: float, material: int) -> float:
    """ICRU 37 unrestricted positron collision stopping power, MeV cm^2/g.

    F+ = 2 ln 2 - (beta^2 / 12) [23 + 14/(tau+2) + 10/(tau+2)^2 + 4/(tau+2)^3]
    (Rohrlich & Carlson, Phys. Rev. 93, 38 (1954), doi:10.1103/PhysRev.93.38).
    """
    data = MATERIALS[material]
    tau = energy / ELECTRON_MASS_MEV
    beta_sq = tau * (tau + 2.0) / (tau + 1.0) ** 2
    f_plus = 2.0 * math.log(2.0) - beta_sq / 12.0 * (
        23.0 + 14.0 / (tau + 2.0) + 10.0 / (tau + 2.0) ** 2 + 4.0 / (tau + 2.0) ** 3
    )
    log_term = math.log(
        tau**2 * (tau + 2.0) / (2.0 * (data.mean_excitation_mev / ELECTRON_MASS_MEV) ** 2)
    )
    return (
        TWO_PI_RE2_MC2
        * data.electrons_per_gram
        / beta_sq
        * (log_term + f_plus - berger_seltzer.density_effect(tau, data.sternheimer))
    )


class TestBhabhaDcs:
    @pytest.mark.parametrize("energy", [0.01, *ENERGIES])
    def test_matches_independent_transcription(self, energy: float) -> None:
        for x in (1e-4, 0.01, 0.2, 0.5, 0.8, 1.0):
            w = x * energy
            assert bhabha_dcs_per_electron(energy, w) == pytest.approx(
                _bhabha_dcs_penelope(energy, w), rel=1e-11
            )

    def test_zero_outside_the_kinematic_interval(self) -> None:
        """Distinguishable particles: the positron may hand over all of T, no more."""
        assert bhabha_dcs_per_electron(2.0, 0.0) == 0.0
        assert bhabha_dcs_per_electron(2.0, -0.1) == 0.0
        assert bhabha_dcs_per_electron(2.0, 2.0) > 0.0
        assert bhabha_dcs_per_electron(2.0, 2.0 + 1e-9) == 0.0

    def test_extends_beyond_the_moller_limit(self) -> None:
        """Where Moller stops (T/2) Bhabha continues to T."""
        assert berger_seltzer.moller_dcs_per_electron(2.0, 1.5) == 0.0
        assert bhabha_dcs_per_electron(2.0, 1.5) > 0.0

    def test_soft_collisions_are_rutherford_like(self) -> None:
        """For W << T both Moller and Bhabha tend to 2 pi r_e^2 mc^2 / (beta^2 W^2)."""
        energy, w = 10.0, 1e-5
        gamma = 1.0 + energy / ELECTRON_MASS_MEV
        rutherford = TWO_PI_RE2_MC2 / ((1.0 - 1.0 / gamma**2) * w * w)
        assert bhabha_dcs_per_electron(energy, w) == pytest.approx(rutherford, rel=1e-5)
        assert berger_seltzer.moller_dcs_per_electron(energy, w) == pytest.approx(
            rutherford, rel=1e-5
        )


class TestBhabhaCrossSection:
    @pytest.mark.parametrize("energy", ENERGIES)
    @pytest.mark.parametrize("cut", [0.01, ECUT_MEV])
    def test_closed_form_equals_integrated_dcs(self, energy: float, cut: float) -> None:
        integral, _ = integrate.quad(
            lambda w: bhabha_dcs_per_electron(energy, w), cut, energy, epsabs=0.0, epsrel=1e-11
        )
        expected = MATERIALS[WATER].electrons_per_gram * integral
        assert restricted_bhabha_cross_section(energy, MATERIALS[WATER], cut) == pytest.approx(
            expected, rel=1e-9
        )

    def test_zero_at_and_below_the_cut(self) -> None:
        """A positron needs only T > cut to make a delta, where Moller needs 2 cut."""
        water = MATERIALS[WATER]
        assert restricted_bhabha_cross_section(0.2, water, 0.2) == 0.0
        assert restricted_bhabha_cross_section(0.1, water, 0.2) == 0.0
        assert restricted_bhabha_cross_section(0.3, water, 0.2) > 0.0
        assert berger_seltzer.restricted_moller_cross_section(0.3, water, 0.2) == 0.0

    def test_rejects_non_positive_energy(self) -> None:
        with pytest.raises(ValueError, match="energy"):
            restricted_bhabha_cross_section(0.0, MATERIALS[WATER], 0.2)


class TestPositronStopping:
    @pytest.mark.parametrize("energy", [0.01, 0.1, *ENERGIES])
    @pytest.mark.parametrize("material", [WATER, CORTICAL_BONE])
    def test_unrestricted_limit_is_the_icru37_closed_form(
        self, energy: float, material: int
    ) -> None:
        """At delta_cut >= T the restricted formula is the Rohrlich-Carlson total."""
        for cut in (energy, 2.0 * energy):
            actual = positron_restricted_collision_stopping(energy, MATERIALS[material], cut)
            assert actual == pytest.approx(
                _unrestricted_positron_stopping(energy, material), rel=1e-11
            )

    @pytest.mark.parametrize("energy", ENERGIES)
    @pytest.mark.parametrize("cut", [0.01, ECUT_MEV])
    def test_bhabha_consistency_identity(self, energy: float, cut: float) -> None:
        """S_restricted(T, cut) + N_e int_cut^T W dsigma_Bhabha = S_unrestricted(T).

        The restricted stopping power and the Bhabha DCS are independent algebra;
        this identity is what makes the Class II split of positron collisions
        (continuous below the cut, discrete above) lose no energy.
        """
        water = MATERIALS[WATER]
        moment, _ = integrate.quad(
            lambda w: w * bhabha_dcs_per_electron(energy, w),
            cut,
            energy,
            epsabs=0.0,
            epsrel=1e-11,
        )
        restricted = positron_restricted_collision_stopping(energy, water, cut)
        total = _unrestricted_positron_stopping(energy, WATER)
        assert restricted + water.electrons_per_gram * moment == pytest.approx(total, rel=1e-9)
        assert restricted < total

    def test_differs_from_the_electron_value_by_a_few_percent(self) -> None:
        """Positron collision stopping: above the electron's at low T, below at high T.

        The two differ only through F+ and F-. As tau -> 0, F+ - F- -> 3 ln 2 - 1
        (> 0); as tau -> infinity, F+ - F- -> 2 ln 2 - 23/12 - 1/8 (< 0). In water
        that is roughly +10 percent at 10 keV and -2 to -3 percent from 1 MeV up.
        The bands below are bounds on that formula-level expectation, not
        tabulated reference data.
        """
        water = MATERIALS[WATER]

        def ratio(energy: float) -> float:
            positron = positron_restricted_collision_stopping(energy, water, energy)
            electron = berger_seltzer.restricted_collision_stopping(energy, water, energy / 2.0)
            return positron / electron

        assert 1.05 < ratio(0.01) < 1.15
        assert 1.0 < ratio(0.1) < 1.08
        for energy in (1.0, 6.0, 20.0, 200.0):
            assert 0.96 < ratio(energy) < 0.99
        # One crossover, from above to below the electron value.
        signs = [ratio(float(e)) > 1.0 for e in np.geomspace(0.01, 300.0, 200)]
        assert signs[0]
        assert not signs[-1]
        assert sum(a != b for a, b in itertools.pairwise(signs)) == 1

    def test_asymptotic_difference_of_the_f_functions(self) -> None:
        """(S+ - S-) beta^2 / (2 pi r_e^2 mc^2 N_e) -> the closed-form limits."""
        water = MATERIALS[WATER]

        def f_difference(energy: float) -> float:
            tau = energy / ELECTRON_MASS_MEV
            beta_sq = tau * (tau + 2.0) / (tau + 1.0) ** 2
            positron = positron_restricted_collision_stopping(energy, water, energy)
            electron = berger_seltzer.restricted_collision_stopping(energy, water, energy / 2.0)
            return (positron - electron) * beta_sq / (TWO_PI_RE2_MC2 * water.electrons_per_gram)

        assert f_difference(1e-6) == pytest.approx(3.0 * math.log(2.0) - 1.0, abs=1e-4)
        assert f_difference(1e6) == pytest.approx(
            2.0 * math.log(2.0) - 23.0 / 12.0 - 0.125, abs=1e-4
        )

    def test_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="energy"):
            positron_restricted_collision_stopping(0.0, MATERIALS[WATER], 0.2)
        with pytest.raises(ValueError, match="delta_cut"):
            positron_restricted_collision_stopping(1.0, MATERIALS[WATER], 0.0)


class TestAnnihilationCrossSection:
    def test_low_energy_one_over_v_law(self) -> None:
        """sigma -> pi r_e^2 / beta as T -> 0 (Dirac's non-relativistic limit)."""
        for energy in (1e-8, 1e-6, 1e-5):
            gamma = 1.0 + energy / ELECTRON_MASS_MEV
            beta = math.sqrt(1.0 - 1.0 / gamma**2)
            assert annihilation_cross_section_per_electron(energy) * beta / PI_RE2 == pytest.approx(
                1.0, rel=1e-3
            )

    def test_high_energy_logarithmic_falloff(self) -> None:
        """sigma -> pi r_e^2 [ln(2 gamma) - 1] / gamma for gamma >> 1."""
        for energy in (5.0e3, 5.0e4):
            gamma = 1.0 + energy / ELECTRON_MASS_MEV
            asymptote = PI_RE2 * (math.log(2.0 * gamma) - 1.0) / gamma
            assert annihilation_cross_section_per_electron(energy) == pytest.approx(
                asymptote, rel=2e-3
            )

    def test_falls_monotonically_with_energy(self) -> None:
        values = [
            annihilation_cross_section_per_electron(float(e)) for e in np.geomspace(1e-4, 1e3, 200)
        ]
        assert all(v > 0.0 for v in values)
        assert values == sorted(values, reverse=True)

    @pytest.mark.parametrize("energy", [0.001, *ENERGIES])
    def test_equals_the_integrated_differential_cross_section(self, energy: float) -> None:
        """The total is the integral of the DCS the sampler draws from.

        The DCS is the one-photon-labelled form: integrating it over the whole
        allowed photon-energy interval counts each annihilation once.
        """
        total = energy + 2.0 * ELECTRON_MASS_MEV
        half_width = 0.5 * math.sqrt(energy / total)
        k_min, k_max = total * (0.5 - half_width), total * (0.5 + half_width)
        integral, _ = integrate.quad(
            lambda k: annihilation_dcs_per_electron(energy, k),
            k_min,
            k_max,
            epsabs=0.0,
            epsrel=1e-11,
        )
        assert annihilation_cross_section_per_electron(energy) == pytest.approx(integral, rel=1e-8)

    def test_dcs_zero_outside_the_kinematic_interval(self) -> None:
        energy = 2.0
        total = energy + 2.0 * ELECTRON_MASS_MEV
        half_width = 0.5 * math.sqrt(energy / total)
        assert annihilation_dcs_per_electron(energy, total * (0.5 - half_width) * 0.999) == 0.0
        assert annihilation_dcs_per_electron(energy, total * (0.5 + half_width) * 1.001) == 0.0
        assert annihilation_dcs_per_electron(energy, 0.5 * total) > 0.0

    def test_rejects_non_positive_energy(self) -> None:
        with pytest.raises(ValueError, match="energy"):
            annihilation_cross_section_per_electron(0.0)


class TestInterface:
    """The positron accessors of ``CrossSectionSource``."""

    def test_concrete_on_the_interface(self) -> None:
        for name in (
            "positron_restricted_stopping_power",
            "bhabha_cross_section",
            "annihilation_cross_section",
            "positron_restricted_range",
            "positron_energy_after_mass_path",
        ):
            assert name not in CrossSectionSource.__abstractmethods__

    @pytest.mark.parametrize("energy", ENERGIES)
    def test_delegates_with_the_registry_material(
        self, xs: AnalyticCrossSections, energy: float
    ) -> None:
        water = MATERIALS[WATER]
        assert xs.positron_restricted_stopping_power(energy, WATER, ECUT_MEV) == (
            positron_restricted_collision_stopping(energy, water, ECUT_MEV)
        )
        assert xs.bhabha_cross_section(energy, WATER, ECUT_MEV) == (
            restricted_bhabha_cross_section(energy, water, ECUT_MEV)
        )
        assert xs.annihilation_cross_section(energy, WATER) == pytest.approx(
            water.electrons_per_gram * annihilation_cross_section_per_electron(energy), rel=1e-14
        )

    def test_rejects_unknown_material(self, xs: AnalyticCrossSections) -> None:
        bad = xs.n_materials
        with pytest.raises(ValueError, match="material"):
            xs.positron_restricted_stopping_power(1.0, bad, ECUT_MEV)
        with pytest.raises(ValueError, match="material"):
            xs.bhabha_cross_section(1.0, bad, ECUT_MEV)
        with pytest.raises(ValueError, match="material"):
            xs.annihilation_cross_section(1.0, bad)
        with pytest.raises(ValueError, match="material"):
            xs.positron_restricted_range(1.0, bad, ECUT_MEV)

    def test_annihilation_mean_free_path_is_long(self, xs: AnalyticCrossSections) -> None:
        """In water at 1 MeV, annihilation in flight is a percent-level channel.

        N_e sigma ~ 3.3e23 /g x 1.7e-25 cm^2 ~ 0.06 cm^2/g against a stopping
        power of ~1.8 MeV cm^2/g: a few percent chance per MeV lost.
        """
        mu = xs.annihilation_cross_section(1.0, WATER)
        assert 0.02 < mu < 0.08


class TestPositronRange:
    """Restricted positron range: the positron twin of ``test_restricted_range.py``."""

    @pytest.mark.parametrize("energy", [0.5, 1.0, 3.0, 6.0, 18.0])
    def test_derivative_is_inverse_restricted_stopping(
        self, xs: AnalyticCrossSections, energy: float
    ) -> None:
        """dr/dE = 1/S+ at the energies and tolerance of the electron pin.

        The shared grid interpolates the range linearly in log E, so a 0.1 percent
        central difference carries a cell-position-dependent error of up to about
        1e-4 that is identical for electrons and positrons (measured: -1.09e-4
        for both at 20 MeV, on the wide-domain instrument now that the analytic
        source refuses 20.02 MeV); the electron test's energies are kept so the
        two pins say the same thing.
        """
        h = 1e-3 * energy
        slope = (
            xs.positron_restricted_range(energy + h, WATER, ECUT_MEV)
            - xs.positron_restricted_range(energy - h, WATER, ECUT_MEV)
        ) / (2.0 * h)
        expected = 1.0 / xs.positron_restricted_stopping_power(energy, WATER, ECUT_MEV)
        assert slope == pytest.approx(expected, rel=1e-4)

    def test_zero_at_the_cut_and_increasing(self, xs: AnalyticCrossSections) -> None:
        assert xs.positron_restricted_range(ECUT_MEV, WATER, ECUT_MEV) == 0.0
        ranges = [
            xs.positron_restricted_range(float(e), WATER, ECUT_MEV)
            for e in np.geomspace(ECUT_MEV, xs.energy_max, 50)
        ]
        assert ranges == sorted(ranges)

    @pytest.mark.parametrize("energy", [0.5, 6.0, 20.0, 50.0])
    @pytest.mark.parametrize("fraction", [0.05, 0.5, 0.95])
    def test_round_trip(self, energy: float, fraction: float) -> None:
        """On the wide-domain instrument, so the pin also covers a node above 20 MeV."""
        xs = _WideAnalytic()
        available = xs.positron_restricted_range(energy, WATER, ECUT_MEV)
        e_end = xs.positron_energy_after_mass_path(energy, WATER, ECUT_MEV, fraction * available)
        assert ECUT_MEV < e_end < energy
        remaining = xs.positron_restricted_range(e_end, WATER, ECUT_MEV)
        assert remaining == pytest.approx((1.0 - fraction) * available, rel=1e-4)

    def test_above_the_domain_raises_like_the_electron_range(
        self, xs: AnalyticCrossSections
    ) -> None:
        """Fail closed above ``energy_max``, with the electron lookup's edge allowance.

        The positron maps used to clamp to a hard 64 MeV grid top; the electron
        ones now refuse (``test_energy_domain.py``) and the twins must not differ.
        """
        top = xs.energy_max
        for query in (xs.restricted_range, xs.positron_restricted_range):
            query(top, WATER, ECUT_MEV)
            query(top * (1.0 + 0.5 * DOMAIN_EDGE_RTOL), WATER, ECUT_MEV)
            with pytest.raises(ValueError, match="energy domain"):
                query(top * (1.0 + 2.0 * DOMAIN_EDGE_RTOL), WATER, ECUT_MEV)
            with pytest.raises(ValueError, match="energy domain"):
                query(30.0, WATER, ECUT_MEV)
        for step in (xs.energy_after_mass_path, xs.positron_energy_after_mass_path):
            with pytest.raises(ValueError, match="energy domain"):
                step(30.0, WATER, ECUT_MEV, 0.1)

    def test_grid_extends_to_a_wide_domain(self) -> None:
        """At 200 MeV the range is the integral of 1/S+, not the 64 MeV value."""
        xs = _WideAnalytic()

        def integrand(e: float) -> float:
            return 1.0 / xs.positron_restricted_stopping_power(e, WATER, ECUT_MEV)

        for energy in (100.0, 200.0, 250.0):
            expected, _ = integrate.quad(integrand, ECUT_MEV, energy, limit=200)
            assert xs.positron_restricted_range(energy, WATER, ECUT_MEV) == pytest.approx(
                expected, rel=1e-4
            )

    def test_grid_extension_is_append_only(self, xs: AnalyticCrossSections) -> None:
        """A wider domain leaves every positron range inside the narrow one bit-identical."""
        wide = _WideAnalytic()
        for e in np.geomspace(ECUT_MEV, 20.0, 37):
            assert wide.positron_restricted_range(
                float(e), WATER, ECUT_MEV
            ) == xs.positron_restricted_range(float(e), WATER, ECUT_MEV)
            for path in (1.0e-3, 0.3, 2.0):
                assert wide.positron_energy_after_mass_path(
                    float(e), WATER, ECUT_MEV, path
                ) == xs.positron_energy_after_mass_path(float(e), WATER, ECUT_MEV, path)

    def test_shares_the_electron_range_nodes(self, xs: AnalyticCrossSections) -> None:
        """One grid rule for both particles: the same log-energy nodes, key for key."""
        for source in (xs, _WideAnalytic()):
            electron_nodes, _ = source._range_grid(WATER, ECUT_MEV)
            positron_nodes, _ = source._positron_range_grid(WATER, ECUT_MEV)
            np.testing.assert_array_equal(positron_nodes, electron_nodes)

    def test_ranging_out_returns_the_cut(self, xs: AnalyticCrossSections) -> None:
        available = xs.positron_restricted_range(2.0, WATER, ECUT_MEV)
        assert xs.positron_energy_after_mass_path(2.0, WATER, ECUT_MEV, available) == ECUT_MEV
        assert xs.positron_energy_after_mass_path(2.0, WATER, ECUT_MEV, 2 * available) == ECUT_MEV

    def test_is_not_the_electron_range(self, xs: AnalyticCrossSections) -> None:
        """Different restricted stopping (Bhabha vs Moller cut): about a percent longer.

        Above ~0.5 MeV the positron's restricted stopping power at the 200 keV cut
        is 0.1 to 1.7 percent below the electron's, so its range is longer.
        """
        for energy in (1.0, 6.0, 20.0):
            positron = xs.positron_restricted_range(energy, WATER, ECUT_MEV)
            electron = xs.restricted_range(energy, WATER, ECUT_MEV)
            assert 1.002 < positron / electron < 1.02
