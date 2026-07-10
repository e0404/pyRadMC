"""Direction sampling and rotation: exact invariants plus uniformity oracles."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from pyRadMC.physics.direction import rotate_direction, sample_isotropic_direction
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED


def test_isotropic_directions_are_unit_vectors() -> None:
    """Unit norm is algebra, not statistics: tight tolerance."""
    state = HostRNG().init_state(SEED, 10)
    for _ in range(2_000):
        ux, uy, uz = sample_isotropic_direction(state)
        assert abs(ux * ux + uy * uy + uz * uz - 1.0) < 1e-12


def test_isotropic_cos_theta_and_phi_are_uniform() -> None:
    """Isotropy means cos(theta) ~ U[-1, 1] and phi ~ U[0, 2 pi), independently.

    Chi-squared on both marginals. n = 20000 puts ~1250 counts in each of 16 bins;
    a common error (sampling theta uniformly instead of cos theta) fails at p < 1e-100.
    """
    n = 20_000
    state = HostRNG().init_state(SEED, 11)
    directions = np.array([sample_isotropic_direction(state) for _ in range(n)])

    cos_theta = directions[:, 2]
    phi = np.arctan2(directions[:, 1], directions[:, 0])

    for values, lo, hi, label in [
        (cos_theta, -1.0, 1.0, "cos(theta)"),
        (phi, -math.pi, math.pi, "phi"),
    ]:
        observed, _ = np.histogram(values, bins=np.linspace(lo, hi, 17))
        chi2, p_value = stats.chisquare(observed)
        assert p_value > 0.01, f"{label} not uniform: chi2={chi2:.1f}, p={p_value:.2e}"


@pytest.mark.parametrize(
    ("ux", "uy", "uz"),
    [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.577350269189626, 0.577350269189626, 0.577350269189626),
        # Near-pole: the rotation formula's denominator sqrt(1 - uz^2) is smallest here.
        (1.0e-7, 0.0, 0.9999999999999949),
    ],
)
def test_rotation_preserves_norm_and_opening_angle(ux: float, uy: float, uz: float) -> None:
    """Rotating u by (theta, phi) yields a unit vector at angle theta from u.

    Both are exact geometric identities, so the tolerance is float-precision (relaxed
    only through the near-pole denominator). A wrong sign or a swapped sine/cosine
    fails by O(1).
    """
    state = HostRNG().init_state(SEED, 12)
    for _ in range(500):
        cos_theta = 1.0 - 2.0 * state.random()
        phi = 2.0 * math.pi * state.random()
        vx, vy, vz = rotate_direction(ux, uy, uz, cos_theta, phi)

        assert abs(vx * vx + vy * vy + vz * vz - 1.0) < 1e-9
        dot = vx * ux + vy * uy + vz * uz
        assert abs(dot - cos_theta) < 1e-9


def test_rotation_azimuth_is_uniform_about_the_axis() -> None:
    """For fixed opening angle, the rotated vectors sweep the cone uniformly in phi.

    Guards against an azimuth convention bug (e.g. phi measured in a non-orthonormal
    frame) that norm and opening-angle checks cannot see.
    """
    n = 20_000
    axis = (0.6, -0.48, 0.64)  # deliberately not axis-aligned
    state = HostRNG().init_state(SEED, 13)

    # Build an orthonormal frame (e1, e2, axis) to measure the azimuth in.
    e1 = np.cross(axis, (0.0, 0.0, 1.0))
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    cos_theta = 0.3
    phis = np.empty(n)
    for i in range(n):
        v = rotate_direction(*axis, cos_theta, 2.0 * math.pi * state.random())
        phis[i] = math.atan2(float(np.dot(v, e2)), float(np.dot(v, e1)))

    observed, _ = np.histogram(phis, bins=np.linspace(-math.pi, math.pi, 17))
    chi2, p_value = stats.chisquare(observed)
    assert p_value > 0.01, f"cone azimuth not uniform: chi2={chi2:.1f}, p={p_value:.2e}"
