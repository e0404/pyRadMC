"""Screened Bethe-Heitler bremsstrahlung sampling: photon energy and emission angle.

Same pattern as the Moller and Compton sampler tests: hard kinematic bounds, then
chi-squared of the sampled spectrum against the differential cross-section the
sampler claims to draw from, with a reported p-value (AGENTS.md section 4). Each
distribution test is paired with a deliberately wrong sampler that the same oracle
must reject, so a pass means something.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy import integrate, stats

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data.bethe_heitler import screening_parameters
from pyradmc.data.materials import AIR, CORTICAL_BONE, MATERIALS, TUNGSTEN, WATER
from pyradmc.physics.brems import sample_bremsstrahlung_energy
from pyradmc.physics.brems_bh import (
    bethe_heitler_rejection_bound,
    bethe_heitler_shape,
    sample_bethe_heitler_photon_energy,
    sample_bremsstrahlung_cos_theta,
    sample_bremsstrahlung_one_minus_cos_theta,
)
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

K_CUT = 0.05  # the shipped PCUT, MeV

# (kinetic energy in MeV, material): the 0.2-300 MeV span of the stated validity
# range in the soft-tissue, bone, gas and high-Z media the registry carries.
SPECTRUM_CASES = [
    (0.2, WATER),
    (2.0, WATER),
    (20.0, WATER),
    (300.0, WATER),
    (6.0, CORTICAL_BONE),
    (200.0, CORTICAL_BONE),
    (20.0, AIR),
    (2.0, TUNGSTEN),
    (100.0, TUNGSTEN),
]
EFFICIENCY_ENERGIES = (0.2, 0.5, 1.0, 2.0, 6.0, 20.0, 50.0, 100.0, 200.0, 300.0)
EFFICIENCY_MATERIALS = (WATER, CORTICAL_BONE, AIR, TUNGSTEN)


def _shape_args(material: int) -> tuple[float, float, float, float, float]:
    s = screening_parameters(MATERIALS[material].composition)
    return (s.z_ratio, s.c_nuc, s.c_el, s.g_nuc, s.g_el)


def _spectrum_p_value(
    samples: np.ndarray, energy: float, args: tuple[float, float, float, float, float]
) -> tuple[float, float, int]:
    """Chi-squared of ``samples`` against ``G(k)/k`` on [K_CUT, energy]."""
    edges = np.geomspace(K_CUT, energy, 25)

    def pdf(k: float) -> float:
        return bethe_heitler_shape(energy, k, *args) / k

    expected = np.array(
        [integrate.quad(pdf, lo, hi, epsabs=0.0)[0] for lo, hi in itertools.pairwise(edges)]
    )
    observed, _ = np.histogram(samples, bins=edges)
    assert observed.sum() == len(samples), "a sample fell outside [k_cut, E]"
    expected = expected / expected.sum() * len(samples)
    # Merge the sparse tip bins into their lower neighbour: chi-squared needs counts.
    while expected[-1] < 20.0:
        expected = np.concatenate((expected[:-2], [expected[-2] + expected[-1]]))
        observed = np.concatenate((observed[:-2], [observed[-2] + observed[-1]]))
    assert np.all(expected > 20.0), "rebin: chi-squared needs counts"
    chi2, p_value = stats.chisquare(observed, expected)
    return float(chi2), float(p_value), len(observed) - 1


def _draw(sampler: Callable[[np.random.Generator], float], n: int, stream: int) -> np.ndarray:
    state = HostRNG().init_state(SEED, stream)
    return np.array([sampler(state) for _ in range(n)])


class TestPhotonEnergySampling:
    N = 20_000

    @pytest.mark.parametrize(("energy", "material"), SPECTRUM_CASES)
    def test_photon_energy_within_kinematic_bounds(self, energy: float, material: int) -> None:
        """k_cut <= k <= T: the photon cannot carry more than the kinetic energy."""
        args = _shape_args(material)
        k = _draw(
            lambda s: sample_bethe_heitler_photon_energy(energy, K_CUT, *args, s),
            4_000,
            stream=700,
        )
        assert np.all(k >= K_CUT)
        assert np.all(k <= energy)

    @pytest.mark.parametrize(("energy", "material"), SPECTRUM_CASES)
    def test_spectrum_matches_screened_dcs(self, energy: float, material: int) -> None:
        """Chi-squared of sampled photon energies against the screened DCS."""
        args = _shape_args(material)
        k = _draw(
            lambda s: sample_bethe_heitler_photon_energy(energy, K_CUT, *args, s),
            self.N,
            stream=701,
        )
        chi2, p_value, dof = _spectrum_p_value(k, energy, args)
        assert p_value > 0.01, (
            f"Bethe-Heitler spectrum off at T={energy} MeV in {MATERIALS[material].name}: "
            f"chi2={chi2:.1f} on {dof} dof, p={p_value:.2e}"
        )

    @pytest.mark.parametrize(("energy", "material"), SPECTRUM_CASES)
    def test_oracle_rejects_the_thin_target_spectrum(self, energy: float, material: int) -> None:
        """Power: the shipped 1/k sampler must fail the very same chi-squared.

        If the oracle could not tell 1/k from the screened shape, the test above
        would pass for the model this one is meant to replace.
        """
        args = _shape_args(material)
        k = _draw(lambda s: sample_bremsstrahlung_energy(energy, K_CUT, s), self.N, stream=701)
        _, p_value, _ = _spectrum_p_value(k, energy, args)
        assert p_value < 1.0e-6, f"oracle has no power against 1/k: p={p_value:.2e}"

    @pytest.mark.parametrize("material", EFFICIENCY_MATERIALS)
    @pytest.mark.parametrize("energy", EFFICIENCY_ENERGIES)
    def test_rejection_bound_dominates_the_shape(self, energy: float, material: int) -> None:
        """The envelope constant is a true upper bound on [k_cut, T].

        A violated bound is a silent truncation of the spectrum, which a
        chi-squared at moderate statistics could miss in a narrow k range.
        """
        args = _shape_args(material)
        bound = bethe_heitler_rejection_bound(energy, K_CUT, *args)
        k = np.geomspace(K_CUT, energy, 4001)
        shape = np.array([bethe_heitler_shape(energy, float(x), *args) for x in k])
        assert bound > 0.0
        assert np.all(shape <= bound * (1.0 + 1.0e-12))

    @pytest.mark.parametrize("material", EFFICIENCY_MATERIALS)
    @pytest.mark.parametrize("energy", EFFICIENCY_ENERGIES)
    def test_rejection_efficiency_is_bounded(self, energy: float, material: int) -> None:
        """Acceptance probability stays above 0.6 over 0.2-300 MeV.

        Computed exactly, not sampled: the efficiency of rejection from the
        uniform-in-ln(k) envelope is the mean of shape/bound over ln k. Measured:
        0.71 at 0.2 MeV, a minimum of 0.61 near 1 MeV, rising to 0.92 at 300 MeV,
        nearly the same in water, cortical bone, air and tungsten.
        """
        args = _shape_args(material)
        bound = bethe_heitler_rejection_bound(energy, K_CUT, *args)
        log_ratio = math.log(energy / K_CUT)
        mean, _ = integrate.quad(
            lambda t: bethe_heitler_shape(energy, K_CUT * math.exp(t), *args),
            0.0,
            log_ratio,
            limit=200,
        )
        efficiency = mean / (log_ratio * bound)
        assert 0.6 < efficiency <= 1.0, (
            f"efficiency {efficiency:.3f} at T={energy} MeV in {MATERIALS[material].name}"
        )


class TestEmissionAngle:
    N = 20_000

    @staticmethod
    def _beta(energy: float) -> float:
        total = energy + ELECTRON_MASS_MEV
        return math.sqrt(energy * (energy + 2.0 * ELECTRON_MASS_MEV)) / total

    @classmethod
    def _cdf(cls, energy: float, mu: np.ndarray) -> np.ndarray:
        r"""CDF of :math:`p(\mu) \propto (1 - \beta\mu)^{-2}` on [-1, 1]."""
        beta = cls._beta(energy)
        return (1.0 / (1.0 - beta * mu) - 1.0 / (1.0 + beta)) * (1.0 - beta * beta) / (2.0 * beta)

    def _p_value(self, mu: np.ndarray, energy: float) -> float:
        # Equal-probability bins from the analytic inverse CDF.
        beta = self._beta(energy)
        q = np.linspace(0.0, 1.0, 41)
        edges = (2.0 * q - 1.0 + beta) / (1.0 - beta + 2.0 * beta * q)
        edges[0], edges[-1] = -1.0, 1.0
        observed, _ = np.histogram(mu, bins=edges)
        expected = np.diff(self._cdf(energy, edges)) * len(mu)
        assert observed.sum() == len(mu)
        return float(stats.chisquare(observed, expected * len(mu) / expected.sum())[1])

    @pytest.mark.parametrize("energy", [0.2, 2.0, 20.0, 300.0])
    def test_cosine_matches_analytic_cdf(self, energy: float) -> None:
        """Chi-squared of sampled cosines in 40 equal-probability bins."""
        mu = _draw(lambda s: sample_bremsstrahlung_cos_theta(energy, s), self.N, stream=710)
        assert np.all(mu >= -1.0)
        assert np.all(mu <= 1.0)
        p_value = self._p_value(mu, energy)
        assert p_value > 0.01, f"emission cosine off at T={energy} MeV: p={p_value:.2e}"

    @pytest.mark.parametrize("energy", [0.2, 2.0, 20.0, 300.0])
    def test_oracle_rejects_a_wrong_velocity(self, energy: float) -> None:
        """Power: cosines sampled at 1.2 times the kinetic energy must fail."""
        mu = _draw(lambda s: sample_bremsstrahlung_cos_theta(1.2 * energy, s), self.N, stream=710)
        p_value = self._p_value(mu, energy)
        assert p_value < 1.0e-6, f"angular oracle has no power: p={p_value:.2e}"

    @pytest.mark.parametrize("energy", [2.0, 20.0, 300.0])
    def test_characteristic_angle_is_one_over_gamma(self, energy: float) -> None:
        r"""The median polar angle is :math:`1/\gamma` to leading order.

        Exactly, the median of the law is :math:`1 - \mu = 1 - \beta`, i.e.
        :math:`\theta_{1/2} = \arccos\beta \approx 1/\gamma`.
        """
        mu = _draw(lambda s: sample_bremsstrahlung_cos_theta(energy, s), self.N, stream=711)
        gamma = 1.0 + energy / ELECTRON_MASS_MEV
        one_minus_mu = np.sort(1.0 - mu)
        exact = 1.0 - self._beta(energy)
        # Order-statistic interval for the median at ~4 sigma: binomial ranks.
        half_width = int(4.0 * math.sqrt(self.N) / 2.0)
        lo = one_minus_mu[self.N // 2 - half_width]
        hi = one_minus_mu[self.N // 2 + half_width]
        assert lo <= exact <= hi
        median_angle = math.acos(1.0 - float(np.median(one_minus_mu)))
        assert 0.8 / gamma < median_angle < 1.25 / gamma

    @pytest.mark.parametrize("energy", [0.2, 20.0, 300.0])
    def test_one_minus_cosine_is_the_same_draw(self, energy: float) -> None:
        """The two forms consume one uniform each and describe the same angle."""
        mu = _draw(lambda s: sample_bremsstrahlung_cos_theta(energy, s), 200, stream=712)
        x = _draw(lambda s: sample_bremsstrahlung_one_minus_cos_theta(energy, s), 200, stream=712)
        assert np.all(x >= 0.0)
        assert np.all(x <= 2.0)
        np.testing.assert_array_equal(mu, 1.0 - x)

    def test_endpoints_of_the_inversion(self) -> None:
        """u -> 0 maps to backward, u -> 1 to forward emission."""

        class _Fixed:
            def __init__(self, value: float) -> None:
                self._value = value

            def random(self) -> float:
                return self._value

        assert sample_bremsstrahlung_cos_theta(2.0, _Fixed(0.0)) == pytest.approx(-1.0, abs=1e-14)
        almost_one = math.nextafter(1.0, 0.0)
        assert sample_bremsstrahlung_cos_theta(2.0, _Fixed(almost_one)) == pytest.approx(
            1.0, abs=1e-12
        )
