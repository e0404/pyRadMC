"""Free-path sampling: the exponential distribution, exactly and statistically."""

from __future__ import annotations

import numpy as np
from scipy import stats

from pyRadMC.physics.path import sample_path_length
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED


def test_path_lengths_are_positive_and_finite() -> None:
    """u in [0, 1) makes -log(1 - u) finite and non-negative, always.

    The u = 1 case would be an infinite path; the RNG contract excludes it, and this
    checks the sampler does not reintroduce it via 1 - u == 0 rounding.
    """
    state = HostRNG().init_state(SEED, 20)
    mu = 0.07  # 1/cm, water-ish at 1 MeV
    for _ in range(100_000):
        s = sample_path_length(mu, state)
        assert s >= 0.0
        assert np.isfinite(s)


def test_path_lengths_are_exponential() -> None:
    """Chi-squared of the sampled histogram against Exp(mu), plus the mean.

    Uses equal-probability bins (deciles of the exact CDF) so every bin carries the
    same statistical weight out to the tail.
    """
    n = 50_000
    mu = 0.05
    state = HostRNG().init_state(SEED, 21)
    s = np.array([sample_path_length(mu, state) for _ in range(n)])

    quantiles = np.linspace(0.0, 1.0, 21)
    edges = stats.expon.ppf(quantiles, scale=1.0 / mu)
    observed, _ = np.histogram(s, bins=edges)
    chi2, p_value = stats.chisquare(observed)
    assert p_value > 0.01, (
        f"free paths not Exp({mu}): chi2={chi2:.1f} on {len(observed) - 1} dof, p={p_value:.2e}"
    )

    standard_error = float(np.std(s, ddof=1)) / np.sqrt(n)
    z = abs(float(np.mean(s)) - 1.0 / mu) / standard_error
    assert z < 4.0, f"mean free path {np.mean(s):.3f} vs {1.0 / mu:.3f} cm: z = {z:.2f}"


def test_path_length_scales_inversely_with_mu() -> None:
    """The same random stream through two mu values gives paths in exact ratio.

    s = -log(1-u)/mu is deterministic in (u, mu); this pins the parameterization
    (mu multiplies, not divides) without any statistics.
    """
    rng = HostRNG()
    state_a = rng.init_state(SEED, 22)
    state_b = rng.init_state(SEED, 22)
    for _ in range(1_000):
        s1 = sample_path_length(0.02, state_a)
        s2 = sample_path_length(0.08, state_b)
        assert abs(s1 - 4.0 * s2) < 1e-9 * max(s1, 1.0)
