"""Pair-production kinematics: screened Bethe-Heitler sharing and lepton angles.

Same pattern as the Compton and Moller samplers: hard kinematic bounds, the exact
conservation identity, and chi-squared of the sampled spectrum against the
differential cross-section the data layer exposes
(:func:`pyradmc.data.pair.pair_sharing_dcs`, written from the Phi1/Phi2 form while
the sampler works from the composition-rejection decomposition, so the comparison
also pins that algebra). Each distribution test is shown to have power by running a
deliberately wrong sampler through the same oracle.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy import integrate, stats

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data.materials import MATERIALS, TUNGSTEN, WATER
from pyradmc.data.pair import PairScreening, pair_screening_parameters, pair_sharing_dcs
from pyradmc.physics.pair import (
    pair_screening_f1,
    pair_screening_f2,
    sample_pair_cos_theta,
    sample_pair_electron_energy,
)
from pyradmc.rng import uniform
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

N = 40_000
N_BINS = 24

WATER_PARAMS = pair_screening_parameters(MATERIALS[WATER].composition)
TUNGSTEN_PARAMS = pair_screening_parameters(MATERIALS[TUNGSTEN].composition)

Sampler = Callable[[float, PairScreening, object], float]


def _model(energy: float, params: PairScreening, state: object) -> float:
    return sample_pair_electron_energy(
        energy, params.screening_constant, params.coulomb_low, params.coulomb_high, state
    )


def _uniform_split(energy: float, params: PairScreening, state: object) -> float:
    """The mutant: today's uniform kinetic-energy split."""
    return uniform(state) * (energy - 2.0 * ELECTRON_MASS_MEV)


def _sharing_p_value(
    sampler: Sampler, energy: float, params: PairScreening, stream: int
) -> tuple[float, float]:
    """Chi-squared of sampled total-energy fractions against the sharing DCS."""
    state = HostRNG().init_state(SEED, stream)
    kinetic = np.array([sampler(energy, params, state) for _ in range(N)])
    eps = (kinetic + ELECTRON_MASS_MEV) / energy

    eps0 = ELECTRON_MASS_MEV / energy
    grid = np.linspace(eps0, 1.0 - eps0, 4001)
    support = grid[[pair_sharing_dcs(energy, float(e), params) > 0.0 for e in grid]]
    edges = np.linspace(support[0], support[-1], N_BINS + 1)
    # Samples outside the DCS support are impossible under the model; count them
    # into the edge bins so a sampler that leaks there is penalised, not ignored.
    observed, _ = np.histogram(np.clip(eps, edges[0], edges[-1]), bins=edges)
    expected = np.array(
        [
            integrate.quad(lambda e: pair_sharing_dcs(energy, e, params), lo, hi)[0]
            for lo, hi in itertools.pairwise(edges)
        ]
    )
    expected *= N / expected.sum()
    assert np.all(expected > 20.0), "rebin: chi-squared needs counts"
    chi2, p_value = stats.chisquare(observed, expected)
    return float(chi2), float(p_value)


class TestEnergySharing:
    CASES = (
        (2.0, WATER_PARAMS, 300),
        (2.5, WATER_PARAMS, 301),
        (6.0, WATER_PARAMS, 302),
        (20.0, WATER_PARAMS, 303),
        (100.0, WATER_PARAMS, 304),
        (300.0, WATER_PARAMS, 305),
        (20.0, TUNGSTEN_PARAMS, 306),
        (100.0, TUNGSTEN_PARAMS, 307),
    )

    @pytest.mark.parametrize(("energy", "params", "stream"), CASES)
    def test_spectrum_matches_screened_bethe_heitler(
        self, energy: float, params: PairScreening, stream: int
    ) -> None:
        chi2, p_value = _sharing_p_value(_model, energy, params, stream)
        assert p_value > 0.01, (
            f"pair sharing at {energy} MeV off: chi2={chi2:.1f} on {N_BINS - 1} dof, "
            f"p={p_value:.2e}"
        )

    @pytest.mark.parametrize(("energy", "params", "stream"), CASES)
    def test_oracle_rejects_the_uniform_split(
        self, energy: float, params: PairScreening, stream: int
    ) -> None:
        """Power: the uniform split the model replaces fails the same test."""
        chi2, p_value = _sharing_p_value(_uniform_split, energy, params, stream)
        assert p_value < 1e-6, (
            f"oracle has no power at {energy} MeV: uniform split passes with "
            f"chi2={chi2:.1f}, p={p_value:.2e}"
        )

    @pytest.mark.parametrize(
        "energy", [1.0221, 1.03, 1.2, 1.9999, 2.0, 2.0001, 3.0, 49.999, 50.0, 300.0]
    )
    @pytest.mark.parametrize("params", [WATER_PARAMS, TUNGSTEN_PARAMS])
    def test_both_leptons_inside_the_kinematic_limits(
        self, energy: float, params: PairScreening
    ) -> None:
        """0 <= T- <= k - 2 m, so T+ = k - 2 m - T- is too; energy is conserved exactly."""
        kinetic = energy - 2.0 * ELECTRON_MASS_MEV
        state = HostRNG().init_state(SEED, 310)
        for _ in range(4_000):
            t_minus = _model(energy, params, state)
            t_plus = kinetic - t_minus
            assert 0.0 <= t_minus <= kinetic
            assert 0.0 <= t_plus <= kinetic
            assert t_minus + t_plus + 2.0 * ELECTRON_MASS_MEV == pytest.approx(energy, rel=1e-14)

    @pytest.mark.parametrize("energy", [1.5, 2.5, 20.0, 300.0])
    def test_sharing_is_symmetric_between_the_leptons(self, energy: float) -> None:
        """The model does not distinguish the charges: <T-> = (k - 2 m) / 2.

        (The Born-approximation DCS is exactly symmetric; the real low-energy
        asymmetry from the nuclear Coulomb field is a neglected effect.)
        """
        kinetic = energy - 2.0 * ELECTRON_MASS_MEV
        state = HostRNG().init_state(SEED, 311)
        t = np.array([_model(energy, WATER_PARAMS, state) for _ in range(N)])
        z = abs(t.mean() - 0.5 * kinetic) / (t.std(ddof=1) / math.sqrt(N))
        assert z < 4.0, f"<T-> = {t.mean():.4f} vs {0.5 * kinetic:.4f}: z = {z:.2f}"
        lower = int(np.count_nonzero(t < 0.5 * kinetic))
        assert stats.binomtest(lower, N, 0.5).pvalue > 0.01

    def test_uniform_below_the_low_energy_limit(self) -> None:
        """Below 2 MeV the sharing is the uniform split (stated approximation)."""
        energy = 1.5
        state = HostRNG().init_state(SEED, 312)
        fraction = np.array([_model(energy, WATER_PARAMS, state) for _ in range(N)]) / (
            energy - 2.0 * ELECTRON_MASS_MEV
        )
        observed, _ = np.histogram(fraction, bins=np.linspace(0.0, 1.0, 21))
        chi2, p_value = stats.chisquare(observed)
        assert p_value > 0.01, f"not uniform below 2 MeV: chi2={chi2:.1f}, p={p_value:.2e}"

    def test_rejection_envelope_bounds_the_screening_functions(self) -> None:
        """max over delta >= delta_min of F_i is the bound the sampler uses.

        The Butcher-Messel fit steps *up* by ~0.1 at delta = 1, so F_i(delta_min)
        alone is not a bound when delta_min < 1; the sampler takes the larger of it
        and the value just above the step.
        """
        from pyradmc.physics.pair import _F_ABOVE_STEP

        step = 42.24 - 8.368 * math.log(1.952)
        assert step <= _F_ABOVE_STEP == pytest.approx(step, rel=1e-9)
        for f in (pair_screening_f1, pair_screening_f2):
            assert f(1.0 + 1e-12) == pytest.approx(step, rel=1e-9)
            assert f(1.0) < step
            for delta_min in np.linspace(0.0, 1.0, 51):
                bound = max(f(float(delta_min)), step)
                deltas = np.concatenate(
                    [np.linspace(delta_min, 1.0, 200), np.linspace(1.0 + 1e-9, 60.0, 2000)]
                )
                assert all(f(float(d)) <= bound for d in deltas)


class TestPolarAngle:
    @staticmethod
    def _cdf(mu: np.ndarray, beta: float) -> np.ndarray:
        """CDF of p(mu) ~ (1 - beta mu)^-2 on [-1, 1]."""
        low = 1.0 / (1.0 + beta)
        return (1.0 / (1.0 - beta * mu) - low) / (1.0 / (1.0 - beta) - low)

    @staticmethod
    def _beta(kinetic: float) -> float:
        gamma = 1.0 + kinetic / ELECTRON_MASS_MEV
        return math.sqrt(1.0 - 1.0 / gamma**2)

    def _p_value(self, kinetic_sampled: float, kinetic_tested: float, stream: int) -> float:
        state = HostRNG().init_state(SEED, stream)
        mu = np.array([sample_pair_cos_theta(kinetic_sampled, state) for _ in range(N)])
        assert np.all(mu >= -1.0)
        assert np.all(mu <= 1.0)
        observed, _ = np.histogram(
            self._cdf(mu, self._beta(kinetic_tested)), bins=np.linspace(0.0, 1.0, 21)
        )
        return float(stats.chisquare(observed).pvalue)

    @pytest.mark.parametrize(
        ("kinetic", "stream"), [(0.05, 320), (0.5, 321), (3.0, 322), (30.0, 323)]
    )
    def test_matches_leading_order_distribution(self, kinetic: float, stream: int) -> None:
        p_value = self._p_value(kinetic, kinetic, stream)
        assert p_value > 0.01, f"pair polar angle at T={kinetic} MeV off: p={p_value:.2e}"

    @pytest.mark.parametrize(
        ("kinetic", "stream"), [(0.05, 320), (0.5, 321), (3.0, 322), (30.0, 323)]
    )
    def test_oracle_rejects_a_twenty_percent_energy_error(
        self, kinetic: float, stream: int
    ) -> None:
        """Power: the same samples tested against a 20 percent different energy fail.

        (At 50 keV the distribution is nearly isotropic and depends weakly on the
        energy: a 10 percent error is only marginally detected there, p = 2e-3 at
        this seed and sample size, against p < 1e-20 from 0.5 MeV up.)
        """
        p_value = self._p_value(kinetic, 1.2 * kinetic, stream)
        assert p_value < 1e-6, f"angle oracle has no power at T={kinetic} MeV: p={p_value:.2e}"

    def test_isotropic_at_rest_and_forward_at_high_energy(self) -> None:
        state = HostRNG().init_state(SEED, 324)
        slow = np.array([sample_pair_cos_theta(1e-9, state) for _ in range(N)])
        assert abs(slow.mean()) < 4.0 / math.sqrt(3.0 * N)
        fast = np.array([sample_pair_cos_theta(300.0, state) for _ in range(N)])
        # Characteristic angle ~ 1/gamma: the median polar angle is exactly 1/gamma
        # for this distribution in the small-angle limit.
        gamma = 1.0 + 300.0 / ELECTRON_MASS_MEV
        median_theta = float(np.median(np.arccos(fast)))
        assert median_theta == pytest.approx(1.0 / gamma, rel=0.05)
