"""Electron interaction sampling: Moller deltas, bremsstrahlung, the MS hinge.

Same testing pattern as the Compton sampler: hard kinematic bounds, exact
conservation identities, and chi-squared of the sampled spectrum against the same
differential cross-section the data layer exposes.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import integrate, stats

from pyRadMC import ELECTRON_MASS_MEV
from pyRadMC.data.analytic import moller_dcs_per_electron
from pyRadMC.physics.brems import (
    bremsstrahlung_step_parameters,
    sample_bremsstrahlung_energy,
)
from pyRadMC.physics.moller import moller_direction_cosines, sample_moller_delta_energy
from pyRadMC.physics.msc import sample_hinge_cos_theta
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED


class TestMollerSampling:
    ENERGY = 2.0
    CUT = 0.2

    def _samples(self, n: int, stream: int) -> np.ndarray:
        state = HostRNG().init_state(SEED, stream)
        return np.array(
            [sample_moller_delta_energy(self.ENERGY, self.CUT, state) for _ in range(n)]
        )

    def test_delta_energy_within_kinematic_bounds(self) -> None:
        """The delta is the lower-energy outgoing electron: cut <= W <= T/2."""
        w = self._samples(5_000, stream=50)
        assert np.all(w >= self.CUT)
        assert np.all(w <= self.ENERGY / 2.0)

    def test_spectrum_matches_moller_dcs(self) -> None:
        """Chi-squared of sampled delta energies against the Moller DCS.

        Log-spaced bins track the ~1/W^2 fall of the spectrum so every bin keeps a
        healthy expected count.
        """
        n = 30_000
        w = self._samples(n, stream=51)
        edges = np.geomspace(self.CUT, self.ENERGY / 2.0, 25)
        observed, _ = np.histogram(w, bins=edges)

        norm, _ = integrate.quad(
            lambda x: moller_dcs_per_electron(self.ENERGY, x), self.CUT, self.ENERGY / 2.0
        )
        expected = (
            np.array(
                [
                    integrate.quad(lambda x: moller_dcs_per_electron(self.ENERGY, x), lo, hi)[0]
                    for lo, hi in itertools.pairwise(edges)
                ]
            )
            / norm
            * n
        )
        assert np.all(expected > 20.0), "rebin: chi-squared needs counts"

        chi2, p_value = stats.chisquare(observed, expected * observed.sum() / expected.sum())
        assert p_value > 0.01, (
            f"Moller delta spectrum off: chi2={chi2:.1f} on {len(observed) - 1} dof, "
            f"p={p_value:.2e}"
        )

    @pytest.mark.parametrize("delta_energy", [0.2, 0.5, 0.99])
    def test_direction_cosines_conserve_momentum(self, delta_energy: float) -> None:
        """Longitudinal and transverse momentum balance exactly.

        With p(T) = sqrt(T (T + 2 m c^2)): the two outgoing longitudinal components
        sum to the incident momentum and the transverse components cancel. This is
        algebra; float tolerance only.
        """
        t = self.ENERGY
        m2 = 2.0 * ELECTRON_MASS_MEV

        def p(kinetic: float) -> float:
            return math.sqrt(kinetic * (kinetic + m2))

        cos_delta, cos_primary = moller_direction_cosines(t, delta_energy)
        assert 0.0 < cos_delta <= 1.0
        assert 0.0 < cos_primary <= 1.0

        p_in = p(t)
        p_d = p(delta_energy)
        p_p = p(t - delta_energy)
        longitudinal = p_d * cos_delta + p_p * cos_primary
        assert longitudinal == pytest.approx(p_in, rel=1e-12)

        sin_d = math.sqrt(1.0 - cos_delta**2)
        sin_p = math.sqrt(1.0 - cos_primary**2)
        assert p_d * sin_d == pytest.approx(p_p * sin_p, rel=1e-9)


class TestBremsstrahlung:
    ENERGY = 5.0
    PCUT = 0.05

    def test_photon_energy_bounds_and_log_uniformity(self) -> None:
        """k from the 1/k spectrum: bounded by [PCUT, E], uniform in ln k."""
        n = 30_000
        state = HostRNG().init_state(SEED, 60)
        k = np.array(
            [sample_bremsstrahlung_energy(self.ENERGY, self.PCUT, state) for _ in range(n)]
        )
        assert np.all(k >= self.PCUT)
        assert np.all(k <= self.ENERGY)

        observed, _ = np.histogram(
            np.log(k), bins=np.linspace(math.log(self.PCUT), math.log(self.ENERGY), 17)
        )
        chi2, p_value = stats.chisquare(observed)
        assert p_value > 0.01, f"1/k spectrum not log-uniform: chi2={chi2:.1f}, p={p_value:.2e}"

    def test_step_parameters_conserve_radiative_loss_in_expectation(self) -> None:
        """prob * <k> + local deposit == S_rad * rho * step, by construction.

        The identity that makes the thin-target model honest: whatever is not
        emitted above PCUT is deposited locally, and the expected emitted energy
        matches the radiative stopping power exactly.
        """
        s_rad, rho, step = 0.12, 1.0, 0.05
        prob, local = bremsstrahlung_step_parameters(self.ENERGY, s_rad, rho, step, self.PCUT)
        assert 0.0 <= prob < 1.0
        assert local >= 0.0

        mean_k = (self.ENERGY - self.PCUT) / math.log(self.ENERGY / self.PCUT)
        assert prob * mean_k + local == pytest.approx(s_rad * rho * step, rel=1e-12)

    def test_sampled_mean_matches_spectrum_mean(self) -> None:
        """<k> of the sampler agrees with the analytic mean of the 1/k spectrum."""
        n = 30_000
        state = HostRNG().init_state(SEED, 61)
        k = np.array(
            [sample_bremsstrahlung_energy(self.ENERGY, self.PCUT, state) for _ in range(n)]
        )
        mean_exact = (self.ENERGY - self.PCUT) / math.log(self.ENERGY / self.PCUT)
        z = abs(k.mean() - mean_exact) / (k.std(ddof=1) / math.sqrt(n))
        assert z < 4.0, f"<k> = {k.mean():.4f} vs exact {mean_exact:.4f}: z = {z:.2f}"

    def test_below_cutoff_everything_is_local(self) -> None:
        """An electron below PCUT cannot emit a transportable photon."""
        prob, local = bremsstrahlung_step_parameters(0.04, 0.001, 1.0, 0.05, self.PCUT)
        assert prob == 0.0
        assert local == pytest.approx(0.001 * 1.0 * 0.05)


class TestHinge:
    def test_cos_theta_bounds(self) -> None:
        state = HostRNG().init_state(SEED, 70)
        for _ in range(10_000):
            c = sample_hinge_cos_theta(0.05, state)
            assert -1.0 <= c <= 1.0

    def test_mean_square_angle_is_reproduced(self) -> None:
        """<theta^2> of the sampled hinge equals the Fermi-Eyges input variance.

        theta^2 is exponentially distributed (2D Gaussian in the projected angles),
        so the sample mean is an unbiased estimate of the input.
        """
        mean_sq = 0.02  # rad^2; small enough that the pi clamp never fires
        n = 50_000
        state = HostRNG().init_state(SEED, 71)
        theta_sq = np.array(
            [math.acos(sample_hinge_cos_theta(mean_sq, state)) ** 2 for _ in range(n)]
        )
        z = abs(theta_sq.mean() - mean_sq) / (theta_sq.std(ddof=1) / math.sqrt(n))
        assert z < 4.0, f"<theta^2> = {theta_sq.mean():.5f} vs {mean_sq:.5f}: z = {z:.2f}"

    def test_theta_squared_is_exponential(self) -> None:
        """Chi-squared against Exp(mean_sq) in equal-probability bins."""
        mean_sq = 0.01
        n = 30_000
        state = HostRNG().init_state(SEED, 72)
        theta_sq = np.array(
            [math.acos(sample_hinge_cos_theta(mean_sq, state)) ** 2 for _ in range(n)]
        )
        edges = stats.expon.ppf(np.linspace(0.0, 1.0, 21), scale=mean_sq)
        observed, _ = np.histogram(theta_sq, bins=edges)
        chi2, p_value = stats.chisquare(observed)
        assert p_value > 0.01, f"hinge theta^2 not exponential: chi2={chi2:.1f}, p={p_value:.2e}"
