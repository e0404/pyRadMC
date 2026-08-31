"""Russian roulette weight sampling: the basic variance reduction.

The game must be exactly fair: E[weight out] = weight in, with outcomes only 0
(killed) or weight/survival (survivor). Fairness is what makes every roulette
site unbiased regardless of where the transport loops invoke it; the loops'
bookkeeping (the escaped-energy ledger) is tested at the integration level.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.physics.roulette import roulette_weight
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED


def _states(n: int):
    rng = HostRNG()
    return (rng.init_state(SEED, h) for h in range(n))


class TestOutcomes:
    def test_only_two_outcomes(self) -> None:
        for state in _states(500):
            out = roulette_weight(1.5, 0.4, state)
            assert out == 0.0 or out == pytest.approx(1.5 / 0.4)

    def test_survivor_weight_preserves_expectation_exactly(self) -> None:
        """The boosted weight is weight/survival by construction, not approximately."""
        state = next(s for s in _states(200) if roulette_weight(1.0, 0.5, s) != 0.0)
        # A fresh state that survives: replay with a different weight.
        assert roulette_weight(3.0, 0.5, state) in (0.0, 6.0)

    def test_certain_survival_is_the_identity(self) -> None:
        for state in _states(100):
            assert roulette_weight(2.0, 1.0, state) == 2.0


class TestFairness:
    def test_mean_weight_is_preserved(self) -> None:
        """Binomial: with n=20000, p=0.5, the mean survives within 5 sigma."""
        n, p, w = 20_000, 0.5, 1.0
        outcomes = np.array([roulette_weight(w, p, s) for s in _states(n)])
        # Var[w_out] = w^2 (1 - p) / p; sigma of the mean follows.
        sigma_mean = w * np.sqrt((1.0 - p) / p / n)
        assert abs(outcomes.mean() - w) < 5.0 * sigma_mean

    def test_survival_fraction_matches_probability(self) -> None:
        n, p = 20_000, 0.3
        survived = sum(roulette_weight(1.0, p, s) != 0.0 for s in _states(n))
        sigma = np.sqrt(n * p * (1.0 - p))
        assert abs(survived - n * p) < 5.0 * sigma


def test_consumes_exactly_one_uniform() -> None:
    """Stream accounting: one draw per game, so mirrored loops stay in lockstep."""
    from pyradmc.rng import uniform

    rng = HostRNG()
    a, b = rng.init_state(SEED, 7), rng.init_state(SEED, 7)
    roulette_weight(1.0, 0.5, a)
    uniform(b)
    assert uniform(a) == uniform(b)
