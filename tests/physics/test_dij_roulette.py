"""Russian roulette on one Dij entry: the per-entry rule of stochastic truncation.

The rule (``roulette_dij_entry``) is the importance sampling of the dose
calculation matrix of Thieke et al., Med. Phys. 29 (2002) 676
(doi:10.1118/1.1469633): with threshold ``t``, an entry of mean ``m``

- at or above ``t`` is kept unchanged,
- below it is kept with probability ``m / t`` and then stored as exactly ``t``,

so the stored value has expectation ``m`` for every ``m``. The stored sigma of a
surviving tail entry, ``sqrt(s^2 t / m + t (t - m))``, makes the *expected stored
variance* equal the total variance of the stored entry, ``s^2 + m (t - m)``.

The statistical checks draw their uniforms through the host RNG interface with the
fixed fast-tier seed, so every statistic here is deterministic.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from pyradmc.physics.roulette import roulette_dij_entry
from pyradmc.rng import uniform
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

# Two-sided normal quantile for a false-alarm probability of 1e-4 per check.
K_SIGMA = 3.89


@functools.cache
def _uniforms(n: int) -> np.ndarray:
    """``n`` draws through the scalar interface, one fresh counter-based state each.

    Cached: the draws are a pure function of ``(SEED, h)``; callers must not mutate.
    """
    rng = HostRNG()
    return np.array([uniform(rng.init_state(SEED, h)) for h in range(n)])


def _tail_sigma(m: float, s: float, t: float) -> float:
    return math.sqrt(s * s * t / m + t * (t - m))


class TestDeterministicCases:
    @pytest.mark.parametrize("u", [0.0, 0.3, 0.999999])
    @pytest.mark.parametrize(("m", "s", "t"), [(2.0, 0.1, 1.0), (1.0, 0.2, 1.0), (5.0, 0.0, 0.0)])
    def test_at_or_above_threshold_is_unchanged(self, m: float, s: float, t: float, u: float):
        """``m >= t`` (including ``t = 0``) returns the entry bit for bit, whatever ``u``."""
        assert roulette_dij_entry(m, s, t, u) == (m, s)

    @pytest.mark.parametrize("u", [0.0, 0.5])
    @pytest.mark.parametrize("t", [0.0, 1.0])
    def test_zero_mean_is_dropped(self, t: float, u: float) -> None:
        """A voxel without dose is never stored, as under the hard rule."""
        assert roulette_dij_entry(0.0, 0.0, t, u) == (0.0, 0.0)

    def test_tail_entry_survives_as_exactly_the_threshold(self) -> None:
        m, s, t = 0.3e-3, 0.05e-3, 1.0e-3
        value, sigma = roulette_dij_entry(m, s, t, 0.1)  # 0.1 < m / t = 0.3
        assert value == t
        assert sigma == _tail_sigma(m, s, t)

    def test_tail_entry_is_otherwise_dropped(self) -> None:
        assert roulette_dij_entry(0.3e-3, 0.05e-3, 1.0e-3, 0.3) == (0.0, 0.0)
        assert roulette_dij_entry(0.3e-3, 0.05e-3, 1.0e-3, 0.9) == (0.0, 0.0)

    def test_outcomes_are_only_drop_or_threshold(self) -> None:
        m, s, t = 0.42, 0.07, 1.0
        for u in _uniforms(2_000):
            value, sigma = roulette_dij_entry(m, s, t, float(u))
            assert (value, sigma) in ((0.0, 0.0), (t, _tail_sigma(m, s, t)))


class TestFairness:
    """Over the keep/drop draw: frequency, mean and stored variance.

    ``n = 40000`` draws of one tail entry at ``p = m / t = 0.3``. Each assertion is a
    ``K_SIGMA = 3.89`` bound (normal approximation, valid at ``n p (1 - p) = 8400``),
    a false-alarm probability of 1e-4 each.
    """

    N = 40_000
    M, S, T = 0.3, 0.04, 1.0

    def _outcomes(self) -> tuple[np.ndarray, np.ndarray]:
        out = [roulette_dij_entry(self.M, self.S, self.T, float(u)) for u in _uniforms(self.N)]
        values = np.array([v for v, _ in out])
        sigmas = np.array([s for _, s in out])
        return values, sigmas

    def test_keep_frequency_is_m_over_t(self) -> None:
        values, _ = self._outcomes()
        p = self.M / self.T
        kept = int(np.count_nonzero(values))
        sigma = math.sqrt(self.N * p * (1.0 - p))
        assert abs(kept - self.N * p) < K_SIGMA * sigma, (kept, self.N * p, sigma)

    def test_stored_value_is_unbiased(self) -> None:
        values, _ = self._outcomes()
        # Var of one stored value is m (t - m), the Bernoulli draw.
        sigma_mean = math.sqrt(self.M * (self.T - self.M) / self.N)
        assert abs(values.mean() - self.M) < K_SIGMA * sigma_mean

    def test_expected_stored_variance_is_the_total_variance(self) -> None:
        """E[stored sigma^2] = s^2 + m (t - m): MC variance plus the draw's own."""
        _, sigmas = self._outcomes()
        var_kept = _tail_sigma(self.M, self.S, self.T) ** 2
        p = self.M / self.T
        target = self.S**2 + self.M * (self.T - self.M)
        # Stored variance is var_kept with probability p, else 0.
        sigma_mean = var_kept * math.sqrt(p * (1.0 - p) / self.N)
        assert abs(float(np.mean(sigmas**2)) - target) < K_SIGMA * sigma_mean
        # And exactly, as an identity of the formula: p * var_kept == target.
        assert p * var_kept == pytest.approx(target, rel=1e-14)
