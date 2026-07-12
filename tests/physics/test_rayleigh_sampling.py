"""Coherent (Rayleigh) angular sampling against the Thomson distribution.

The channel is data-disabled in the analytic source (zero coherent column), but the
transport loops are channel-complete (AGENTS.md 2.10), so the sampler is real physics
that must be right *before* any data source turns the channel on.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import integrate, stats

from pyRadMC.physics.rayleigh import (
    sample_coherent_cos_theta_form_factor,
    sample_rayleigh_cos_theta,
)
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED


def _samples(n: int, stream: int) -> np.ndarray:
    state = HostRNG().init_state(SEED, stream)
    return np.array([sample_rayleigh_cos_theta(state) for _ in range(n)])


def _cumulative(form_factor_squared: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """A(x) = integral_0^x F^2(x') x' dx', the form-factor cumulative the sampler inverts."""
    integrand = form_factor_squared * x_grid
    steps = np.diff(x_grid) * 0.5 * (integrand[1:] + integrand[:-1])
    return np.concatenate(([0.0], np.cumsum(steps)))


def _ff_samples(
    cumulative: np.ndarray, x_grid: np.ndarray, energy: float, n: int, stream: int
) -> np.ndarray:
    state = HostRNG().init_state(SEED, stream)
    return np.array(
        [
            sample_coherent_cos_theta_form_factor(cumulative, x_grid, len(x_grid), energy, state)
            for _ in range(n)
        ]
    )


def test_cosine_within_bounds() -> None:
    mu = _samples(5_000, stream=0)
    assert np.all(mu >= -1.0)
    assert np.all(mu <= 1.0)


def test_sampled_spectrum_matches_thomson() -> None:
    """Chi-squared of the cosine histogram against (1 + mu^2)/2.

    The detection oracle of AGENTS.md section 4 applied to the sampler: a wrong
    envelope or acceptance function shifts bin contents far beyond Poisson noise.
    """
    n = 30_000
    mu = _samples(n, stream=1)
    edges = np.linspace(-1.0, 1.0, 33)
    observed, _ = np.histogram(mu, bins=edges)

    def pdf(x: float) -> float:
        return 1.0 + x * x

    norm, _ = integrate.quad(pdf, -1.0, 1.0)
    expected = (
        np.array([integrate.quad(pdf, lo, hi)[0] for lo, hi in itertools.pairwise(edges)])
        / norm
        * n
    )
    assert np.all(expected > 20.0)
    chi2, p_value = stats.chisquare(observed, expected * observed.sum() / expected.sum())
    assert p_value > 0.01, f"chi2={chi2:.1f}, p={p_value:.2e}"


def test_moments_match_quadrature() -> None:
    """<mu> = 0 by symmetry and <mu^2> = 2/5 for the Thomson form; 4-sigma gates."""
    n = 30_000
    mu = _samples(n, stream=2)
    se_mean = float(np.std(mu, ddof=1)) / np.sqrt(n)
    assert abs(float(np.mean(mu))) < 4.0 * se_mean

    mu_sq = mu**2
    se_sq = float(np.std(mu_sq, ddof=1)) / np.sqrt(n)
    assert abs(float(np.mean(mu_sq)) - 0.4) < 4.0 * se_sq


# -- form-factor coherent sampler -----------------------------------------------


def test_flat_form_factor_reduces_to_thomson() -> None:
    """A constant F^2 makes the cumulative proportional to x^2 -> Thomson exactly.

    The form-factor sampler must reproduce the Thomson moments <mu>=0, <mu^2>=2/5 when
    the form factor carries no angular information; a fine grid keeps the linear
    inversion of the (quadratic) cumulative below the statistical gate.
    """
    x_grid = np.linspace(0.0, 200.0, 8000)
    cumulative = _cumulative(np.ones_like(x_grid), x_grid)
    mu = _ff_samples(cumulative, x_grid, energy=1.0, n=30_000, stream=3)

    se_mean = float(np.std(mu, ddof=1)) / np.sqrt(mu.size)
    assert abs(float(np.mean(mu))) < 4.0 * se_mean
    se_sq = float(np.std(mu**2, ddof=1)) / np.sqrt(mu.size)
    assert abs(float(np.mean(mu**2)) - 0.4) < 4.0 * se_sq


def test_form_factor_is_forward_peaked_and_grows_with_energy() -> None:
    """A decaying form factor peaks coherent scattering forward, more so at higher energy.

    F^2(x) = exp(-(x/5)^2) cuts the accessible momentum transfer off near x~5; at higher
    energy x_max = C*E grows, so the same cutoff maps to smaller angles -> larger <mu>.
    """
    x_grid = np.linspace(0.0, 600.0, 12_000)
    cumulative = _cumulative(np.exp(-((x_grid / 5.0) ** 2)), x_grid)

    mu_low = _ff_samples(cumulative, x_grid, energy=0.1, n=20_000, stream=4)
    mu_high = _ff_samples(cumulative, x_grid, energy=2.0, n=20_000, stream=5)

    assert float(np.mean(mu_low)) > 0.0  # forward-biased already
    assert float(np.mean(mu_high)) > float(np.mean(mu_low))  # more forward at higher E
    assert float(np.mean(mu_high)) > 0.99  # strongly forward at 2 MeV
