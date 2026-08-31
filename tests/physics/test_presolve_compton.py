"""The pre-solve's vectorized Klein-Nishina sampler against the scalar oracle.

``physics/`` functions are scalar by contract (AGENTS.md 2.5); the head pre-solve
needs array-rate sampling, so ``geometry/head.py`` carries a vectorized Kahn
implementation — deliberate, contained duplication, pinned here against the
scalar :func:`sample_compton_energy_ratio` oracle so the two can never drift.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pyradmc.geometry.head import _sample_compton_ratios
from pyradmc.physics.compton import compton_cos_theta, sample_compton_energy_ratio
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED


@pytest.mark.parametrize("energy", [0.5, 2.0, 6.0])
def test_vectorized_ratios_match_the_scalar_oracle(energy: float) -> None:
    """Two-sample chi-squared between vectorized and scalar Kahn samples."""
    n = 40_000
    vectorized = _sample_compton_ratios(
        np.full(n, energy), np.random.Generator(np.random.PCG64(SEED))
    )
    rng = HostRNG()
    scalar = np.array(
        [sample_compton_energy_ratio(energy, rng.init_state(SEED, i)) for i in range(n)]
    )

    r_min = 1.0 / (1.0 + 2.0 * energy / 0.51099895069)
    edges = np.linspace(r_min, 1.0, 21)
    counts_a, _ = np.histogram(vectorized, bins=edges)
    counts_b, _ = np.histogram(scalar, bins=edges)
    # Standard two-sample chi-squared on the binned counts.
    total = counts_a + counts_b
    keep = total > 10
    expected_a = total[keep] * counts_a.sum() / (counts_a.sum() + counts_b.sum())
    expected_b = total[keep] * counts_b.sum() / (counts_a.sum() + counts_b.sum())
    chi2 = float(
        np.sum((counts_a[keep] - expected_a) ** 2 / expected_a)
        + np.sum((counts_b[keep] - expected_b) ** 2 / expected_b)
    )
    p_value = float(stats.chi2.sf(chi2, keep.sum() - 1))
    assert p_value > 0.01, f"chi2={chi2:.1f}, p={p_value:.2e} at {energy} MeV"


def test_ratios_respect_the_kinematic_range() -> None:
    energies = np.random.default_rng(3).uniform(0.3, 8.0, 5_000)
    ratios = _sample_compton_ratios(energies, np.random.Generator(np.random.PCG64(SEED)))
    r_min = 1.0 / (1.0 + 2.0 * energies / 0.51099895069)
    assert np.all(ratios <= 1.0)
    assert np.all(ratios >= r_min - 1e-12)


def test_cos_theta_is_the_scalar_compton_relation() -> None:
    """The vectorized angle mapping equals the scalar oracle value for value."""
    from pyradmc.geometry.head import _compton_cos_thetas

    energies = np.random.default_rng(4).uniform(0.3, 8.0, 512)
    ratios = _sample_compton_ratios(energies, np.random.Generator(np.random.PCG64(SEED)))
    vectorized = _compton_cos_thetas(energies, ratios)
    scalar = np.array(
        [compton_cos_theta(float(e), float(r)) for e, r in zip(energies, ratios, strict=True)]
    )
    np.testing.assert_array_equal(vectorized, scalar)
