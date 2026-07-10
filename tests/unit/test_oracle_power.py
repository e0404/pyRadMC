"""Pin the statistical power of the test oracles themselves.

An oracle that passes everything is worse than no oracle: it converts an unverified code
into an apparently verified one. These tests measure what the oracles in ``conftest.py``
can and cannot see, and they pin the ordering asserted in AGENTS.md section 4.

They test the *tests*. They are cheap. They should never be deleted.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import assert_chi2_consistent, assert_z_consistent

N_VOXELS = 20_000
SIGMA_VOXEL = 0.02  # 2 percent, the planning-grade per-beamlet target


def _caught(oracle, a, sigma_a, b, sigma_b) -> bool:
    """True if the oracle rejects consistency."""
    try:
        oracle(a, sigma_a, b, sigma_b)
    except AssertionError:
        return True
    return False


@pytest.fixture
def pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """An unbiased estimate and its uncertainty array."""
    truth = np.ones(N_VOXELS)
    sigma = np.full(N_VOXELS, SIGMA_VOXEL)
    a = truth + rng.normal(0.0, SIGMA_VOXEL, N_VOXELS)
    return a, sigma


def test_oracles_accept_unbiased_estimates(pair, rng) -> None:
    """Two unbiased noisy estimates of the same truth must not be rejected.

    If this fails, the oracles are too strict and every backend comparison will flake.
    """
    a, sigma = pair
    b = np.ones(N_VOXELS) + rng.normal(0.0, SIGMA_VOXEL, N_VOXELS)
    assert_z_consistent(a, sigma, b, sigma)
    assert_chi2_consistent(a, sigma, b, sigma)


def test_chi2_false_positive_rate_is_calibrated() -> None:
    """The chi-squared oracle rejects unbiased pairs at approximately alpha.

    A test whose false-positive rate is not the alpha you asked for is not a statistical
    test, it is a coin flip with extra steps.
    """
    sigma = np.full(N_VOXELS, SIGMA_VOXEL)
    truth = np.ones(N_VOXELS)
    alpha = 0.01
    trials = 200

    rejections = 0
    for s in range(trials):
        r = np.random.default_rng(s)
        a = truth + r.normal(0.0, SIGMA_VOXEL, N_VOXELS)
        b = truth + r.normal(0.0, SIGMA_VOXEL, N_VOXELS)
        if _caught(assert_chi2_consistent, a, sigma, b, sigma):
            rejections += 1

    # Binomial 99.9 percent upper bound on 200 trials at p = 0.01 is about 8.
    assert rejections <= 8, (
        f"chi-squared rejected {rejections}/{trials} unbiased pairs; "
        f"expected about {alpha * trials:.0f}. The oracle is mis-calibrated."
    )


@pytest.mark.parametrize(
    ("bias", "chi2_should_catch", "zscore_should_catch"),
    [
        (0.010, True, True),  # 0.50 sigma_voxel
        (0.005, True, False),  # 0.25 sigma_voxel: chi2 sees it, tail-counting does not
        (0.002, False, False),  # 0.10 sigma_voxel: below the detection floor at this n
    ],
)
def test_uniform_bias_detection_floor(
    pair, rng, bias: float, chi2_should_catch: bool, zscore_should_catch: bool
) -> None:
    """Pin the measured detection floor for a uniform systematic bias.

    This is the table in AGENTS.md section 4. Chi-squared dominates the z-score outlier
    count against uniform bias, because it aggregates evidence rather than counting tail
    events. The z-score's value is diagnostic (it localizes), not detective.

    Neither oracle resolves a uniform bias below roughly 0.1 sigma_voxel at 20000 voxels.
    The floor scales as 1/sqrt(n_voxels): to see smaller effects, enlarge the region of
    interest. **Never loosen a tolerance instead.**
    """
    a, sigma = pair
    b = np.ones(N_VOXELS) * (1.0 + bias) + rng.normal(0.0, SIGMA_VOXEL, N_VOXELS)

    assert _caught(assert_chi2_consistent, a, sigma, b, sigma) is chi2_should_catch
    assert _caught(assert_z_consistent, a, sigma, b, sigma) is zscore_should_catch


@pytest.mark.parametrize(("fraction", "shift_sigmas"), [(0.01, 5.0), (0.005, 5.0), (0.01, 3.0)])
def test_localized_bias_is_caught_by_both(pair, rng, fraction: float, shift_sigmas: float) -> None:
    """A bias confined to a small region, e.g. a heterogeneity-interface bug.

    Both oracles catch this. Use the z-score to find out *which* voxels.
    """
    a, sigma = pair
    b = np.ones(N_VOXELS) + rng.normal(0.0, SIGMA_VOXEL, N_VOXELS)
    k = int(fraction * N_VOXELS)
    b[:k] += shift_sigmas * SIGMA_VOXEL

    assert _caught(assert_chi2_consistent, a, sigma, b, sigma)
    assert _caught(assert_z_consistent, a, sigma, b, sigma)
