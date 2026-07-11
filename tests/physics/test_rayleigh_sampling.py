"""Coherent (Rayleigh) angular sampling against the Thomson distribution.

The channel is data-disabled in the analytic source (zero coherent column), but the
transport loops are channel-complete (AGENTS.md 2.10), so the sampler is real physics
that must be right *before* any data source turns the channel on.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import integrate, stats

from pyRadMC.physics.rayleigh import sample_rayleigh_cos_theta
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED


def _samples(n: int, stream: int) -> np.ndarray:
    state = HostRNG().init_state(SEED, stream)
    return np.array([sample_rayleigh_cos_theta(state) for _ in range(n)])


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
