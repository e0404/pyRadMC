"""Positron interaction sampling: Bhabha deltas and annihilation in flight.

Same pattern as ``test_electron_sampling.py``: hard kinematic bounds, exact
energy-momentum identities, and chi-squared of each sampled spectrum against the
differential cross-section the data layer exposes — with a deliberately wrong
sampler run through the same oracle to show the test can fail.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy import integrate, stats

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data.positron import annihilation_dcs_per_electron, bhabha_dcs_per_electron
from pyradmc.physics.annihilation import (
    annihilation_direction_cosines,
    sample_annihilation_photon_energy,
)
from pyradmc.physics.bhabha import bhabha_direction_cosines, sample_bhabha_delta_energy
from pyradmc.physics.moller import sample_moller_delta_energy
from pyradmc.rng import uniform
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

N = 40_000
TWO_M = 2.0 * ELECTRON_MASS_MEV


def _momentum(kinetic: float) -> float:
    return math.sqrt(kinetic * (kinetic + TWO_M))


def _spectrum_p_value(
    samples: np.ndarray, density: Callable[[float], float], edges: np.ndarray
) -> tuple[float, float]:
    """Chi-squared of ``samples`` against ``density`` integrated over ``edges``.

    Samples outside the edges are counted into the end bins, so a sampler that
    leaves the allowed interval is penalised rather than ignored.
    """
    observed, _ = np.histogram(np.clip(samples, edges[0], edges[-1]), bins=edges)
    expected = np.array(
        [
            integrate.quad(density, lo, hi, epsabs=0.0, epsrel=1e-10)[0]
            for lo, hi in itertools.pairwise(edges)
        ]
    )
    expected *= len(samples) / expected.sum()
    assert np.all(expected > 20.0), "rebin: chi-squared needs counts"
    chi2, p_value = stats.chisquare(observed, expected)
    return float(chi2), float(p_value)


# -- Bhabha ------------------------------------------------------------------------


def _bhabha_envelope_only(energy: float, cut: float, state: object) -> float:
    """Mutant: the 1/W^2 envelope without the rejection (pure Rutherford)."""
    w_min = cut / energy
    return energy * w_min / (1.0 - uniform(state) * (1.0 - w_min))


class TestBhabhaSampling:
    CUT = 0.2
    CASES = ((0.5, 400), (2.0, 401), (20.0, 402), (200.0, 403))

    def _p_value(self, sampler: Callable[..., float], energy: float, stream: int) -> float:
        state = HostRNG().init_state(SEED, stream)
        w = np.array([sampler(energy, self.CUT, state) for _ in range(N)])
        # Log-spaced bins follow the ~1/W^2 fall; twelve of them keep the expected
        # count of the hardest bin above 20 even across three decades at 200 MeV.
        edges = np.geomspace(self.CUT, energy, 13)
        _, p = _spectrum_p_value(w, lambda x: bhabha_dcs_per_electron(energy, x), edges)
        return p

    @pytest.mark.parametrize(("energy", "stream"), CASES)
    def test_delta_energy_within_kinematic_bounds(self, energy: float, stream: int) -> None:
        """cut <= W <= T: unlike Moller, the positron may transfer everything."""
        state = HostRNG().init_state(SEED, stream)
        w = np.array([sample_bhabha_delta_energy(energy, self.CUT, state) for _ in range(5_000)])
        assert np.all(w >= self.CUT)
        assert np.all(w <= energy)
        assert np.any(w > 0.5 * energy), "no transfer above T/2: that is Moller's limit"

    @pytest.mark.parametrize(("energy", "stream"), CASES)
    def test_spectrum_matches_bhabha_dcs(self, energy: float, stream: int) -> None:
        p_value = self._p_value(sample_bhabha_delta_energy, energy, stream)
        assert p_value > 0.01, f"Bhabha delta spectrum at {energy} MeV off: p={p_value:.2e}"

    @pytest.mark.parametrize(("energy", "stream"), [(2.0, 401), (20.0, 402), (200.0, 403)])
    def test_oracle_rejects_moller_in_place_of_bhabha(self, energy: float, stream: int) -> None:
        """Power: the electron sampler (today's positron treatment) fails."""
        p_value = self._p_value(sample_moller_delta_energy, energy, stream)
        assert p_value < 1e-6, f"oracle passes Moller at {energy} MeV: p={p_value:.2e}"

    @pytest.mark.parametrize(("energy", "stream"), CASES)
    def test_oracle_rejects_the_bare_envelope(self, energy: float, stream: int) -> None:
        """Power against a subtler error: right support, missing the Bhabha factor."""
        p_value = self._p_value(_bhabha_envelope_only, energy, stream)
        assert p_value < 1e-6, f"oracle passes pure 1/W^2 at {energy} MeV: p={p_value:.2e}"

    def test_rejection_function_is_a_probability(self) -> None:
        """0 < 1 - b1 x + b2 x^2 - b3 x^3 + b4 x^4 <= 1 on (0, 1] at every energy.

        The sampler rejects against 1; this is the bound that makes that exact.
        Written from PENELOPE's b_i (doi:10.1787/32da5043-en), independently of the
        sampler's nested form.
        """
        x = np.linspace(1e-6, 1.0, 2001)
        for energy in np.geomspace(1e-4, 1e4, 81):
            gamma = 1.0 + energy / ELECTRON_MASS_MEV
            common = ((gamma - 1.0) / gamma) ** 2
            b1 = common * (2.0 * (gamma + 1.0) ** 2 - 1.0) / (gamma**2 - 1.0)
            b2 = common * (3.0 * (gamma + 1.0) ** 2 + 1.0) / (gamma + 1.0) ** 2
            b3 = common * 2.0 * gamma * (gamma - 1.0) / (gamma + 1.0) ** 2
            b4 = common * (gamma - 1.0) ** 2 / (gamma + 1.0) ** 2
            g = 1.0 - b1 * x + b2 * x**2 - b3 * x**3 + b4 * x**4
            assert np.all(g > 0.0)
            assert np.all(g <= 1.0 + 1e-12)

    @pytest.mark.parametrize("energy", [0.3, 2.0, 20.0, 200.0])
    @pytest.mark.parametrize("fraction", [1e-3, 0.1, 0.5, 0.75, 0.999, 1.0])
    def test_direction_cosines_conserve_momentum(self, energy: float, fraction: float) -> None:
        """Two-body kinematics on a stationary electron, valid up to W = T.

        Energy is conserved by construction (the primary keeps T - W); the
        longitudinal momenta sum to the incident momentum and the transverse ones
        cancel. Algebra: float tolerance only.
        """
        w = fraction * energy
        cos_delta, cos_primary = bhabha_direction_cosines(energy, w)
        assert 0.0 <= cos_delta <= 1.0
        assert 0.0 <= cos_primary <= 1.0

        p_in, p_delta, p_primary = _momentum(energy), _momentum(w), _momentum(energy - w)
        assert p_delta * cos_delta + p_primary * cos_primary == pytest.approx(p_in, rel=1e-11)
        sin_delta = math.sqrt(max(0.0, 1.0 - cos_delta**2))
        sin_primary = math.sqrt(max(0.0, 1.0 - cos_primary**2))
        assert p_delta * sin_delta == pytest.approx(
            p_primary * sin_primary, rel=1e-6, abs=1e-9 * p_in
        )

    def test_sampled_events_conserve_energy_and_momentum(self) -> None:
        """The identity over sampled events at random energies, not hand-picked ones."""
        state = HostRNG().init_state(SEED, 410)
        for _ in range(3_000):
            energy = 0.21 * math.exp(uniform(state) * math.log(300.0 / 0.21))
            w = sample_bhabha_delta_energy(energy, 0.2, state)
            cos_delta, cos_primary = bhabha_direction_cosines(energy, w)
            p_delta, p_primary = _momentum(w), _momentum(energy - w)
            assert p_delta * cos_delta + p_primary * cos_primary == pytest.approx(
                _momentum(energy), rel=1e-10
            )
            transverse = p_delta * math.sqrt(max(0.0, 1.0 - cos_delta**2)) - p_primary * math.sqrt(
                max(0.0, 1.0 - cos_primary**2)
            )
            assert abs(transverse) < 1e-6 * _momentum(energy)


# -- annihilation in flight --------------------------------------------------------


def _annihilation_envelope_only(energy: float, state: object) -> float:
    """Mutant: the 1/k envelope without the rejection."""
    total = energy + TWO_M
    half_width = 0.5 * math.sqrt(energy / total)
    low, high = 0.5 - half_width, 0.5 + half_width
    return total * low * math.exp(uniform(state) * math.log(high / low))


class TestAnnihilationSampling:
    CASES = ((0.3, 420), (2.0, 421), (20.0, 422), (200.0, 423))

    @staticmethod
    def _limits(energy: float) -> tuple[float, float]:
        total = energy + TWO_M
        half_width = 0.5 * math.sqrt(energy / total)
        return total * (0.5 - half_width), total * (0.5 + half_width)

    def _p_value(self, sampler: Callable[..., float], energy: float, stream: int) -> float:
        state = HostRNG().init_state(SEED, stream)
        k = np.array([sampler(energy, state) for _ in range(N)])
        k_min, k_max = self._limits(energy)
        edges = np.geomspace(k_min, k_max, 21)
        _, p = _spectrum_p_value(k, lambda x: annihilation_dcs_per_electron(energy, x), edges)
        return p

    @pytest.mark.parametrize(("energy", "stream"), CASES)
    def test_photon_energy_within_kinematic_bounds(self, energy: float, stream: int) -> None:
        state = HostRNG().init_state(SEED, stream)
        k_min, k_max = self._limits(energy)
        k = np.array([sample_annihilation_photon_energy(energy, state) for _ in range(5_000)])
        assert np.all(k >= k_min * (1.0 - 1e-12))
        assert np.all(k <= k_max * (1.0 + 1e-12))

    @pytest.mark.parametrize(("energy", "stream"), CASES)
    def test_spectrum_matches_heitler_dcs(self, energy: float, stream: int) -> None:
        p_value = self._p_value(sample_annihilation_photon_energy, energy, stream)
        assert p_value > 0.01, f"annihilation spectrum at {energy} MeV off: p={p_value:.2e}"

    @pytest.mark.parametrize(("energy", "stream"), CASES)
    def test_oracle_rejects_the_bare_envelope(self, energy: float, stream: int) -> None:
        p_value = self._p_value(_annihilation_envelope_only, energy, stream)
        assert p_value < 1e-6, f"oracle passes pure 1/k at {energy} MeV: p={p_value:.2e}"

    def test_rejection_function_is_bounded_by_its_maximum(self) -> None:
        """0 < g(eps) <= 1 - 2/(gamma+1)^2 on the allowed interval, at every energy."""
        for energy in np.geomspace(1e-5, 1e4, 91):
            gamma = 1.0 + energy / ELECTRON_MASS_MEV
            s = math.sqrt((gamma - 1.0) / (gamma + 1.0))
            eps = np.linspace(0.5 * (1.0 - s), 0.5 * (1.0 + s), 4001)
            g = 1.0 - eps + (2.0 * gamma * eps - 1.0) / (eps * (gamma + 1.0) ** 2)
            bound = 1.0 - 2.0 / (gamma + 1.0) ** 2
            assert np.all(g > 0.0)
            assert np.all(g <= bound * (1.0 + 1e-12))

    def test_sampled_events_conserve_energy_and_momentum(self) -> None:
        """k1 + k2 = T + 2 m c^2 and the photon momenta sum to the positron's.

        Over sampled events at log-random energies from 1 keV to 300 MeV. The
        azimuths are back to back, so transverse balance is k1 sin1 = k2 sin2.
        """
        state = HostRNG().init_state(SEED, 430)
        for _ in range(5_000):
            energy = 1e-3 * math.exp(uniform(state) * math.log(300.0 / 1e-3))
            total = energy + TWO_M
            k1 = sample_annihilation_photon_energy(energy, state)
            k2 = total - k1
            assert k1 > 0.0
            assert k2 > 0.0
            cos1, cos2 = annihilation_direction_cosines(energy, k1)
            assert -1.0 <= cos1 <= 1.0
            assert -1.0 <= cos2 <= 1.0
            p = _momentum(energy)
            assert k1 * cos1 + k2 * cos2 == pytest.approx(p, rel=1e-9)
            transverse = k1 * math.sqrt(max(0.0, 1.0 - cos1**2)) - k2 * math.sqrt(
                max(0.0, 1.0 - cos2**2)
            )
            assert abs(transverse) < 1e-6 * total

    def test_direction_cosines_are_symmetric_in_the_photons(self) -> None:
        energy = 3.0
        total = energy + TWO_M
        k1 = 0.3 * total
        cos1, cos2 = annihilation_direction_cosines(energy, k1)
        swapped2, swapped1 = annihilation_direction_cosines(energy, total - k1)
        assert (cos1, cos2) == pytest.approx((swapped1, swapped2), rel=1e-13)

    def test_slow_positron_gives_two_near_511_kev_photons(self) -> None:
        """The in-flight sampler joins the at-rest limit continuously."""
        state = HostRNG().init_state(SEED, 431)
        energy = 1e-6
        k = np.array([sample_annihilation_photon_energy(energy, state) for _ in range(2_000)])
        assert np.all(np.abs(k - ELECTRON_MASS_MEV) < 1e-3 * ELECTRON_MASS_MEV)

    def test_hard_photon_goes_forward_at_high_energy(self) -> None:
        energy = 200.0
        total = energy + TWO_M
        cos_hard, cos_soft = annihilation_direction_cosines(energy, 0.99 * total)
        assert cos_hard > 0.9999
        assert cos_soft < cos_hard
