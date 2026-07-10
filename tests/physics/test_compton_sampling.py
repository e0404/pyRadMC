"""Compton sampling: Kahn rejection against the exact Klein-Nishina differential.

The sampler is the first physics routine in the code and the template for all others:
pure, scalar, RNG state passed in. These tests pin (a) hard kinematic bounds, (b) the
energy-angle relation as an exact identity, and (c) the sampled spectrum against the
analytic Klein-Nishina pdf with the chi-squared detection oracle.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import integrate, stats

from pyRadMC import ELECTRON_MASS_MEV
from pyRadMC.physics.compton import compton_cos_theta, sample_compton_energy_ratio
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED

ENERGIES_MEV = [0.2, 1.0, 6.0, 20.0]


def _kn_pdf_unnormalized(r: np.ndarray, alpha: float) -> np.ndarray:
    """Klein-Nishina dsigma/dr up to a constant, r = E'/E in [1/(1+2alpha), 1].

    From dsigma/dOmega proportional to r^2 (r + 1/r - sin^2 theta) and the Compton
    relation cos theta = 1 + 1/alpha - 1/(alpha r), whose Jacobian dcos(theta)/dr =
    1/(alpha r^2) cancels the r^2.
    """
    cos_theta = 1.0 + 1.0 / alpha - 1.0 / (alpha * r)
    return r + 1.0 / r - (1.0 - cos_theta**2)


def _samples(energy: float, n: int, stream: int) -> np.ndarray:
    state = HostRNG().init_state(SEED, stream)
    return np.array([sample_compton_energy_ratio(energy, state) for _ in range(n)])


@pytest.mark.parametrize("energy", ENERGIES_MEV)
def test_energy_ratio_within_kinematic_bounds(energy: float) -> None:
    """r = E'/E lies in [1/(1+2 alpha), 1]: full backscatter to forward scatter."""
    alpha = energy / ELECTRON_MASS_MEV
    r = _samples(energy, 5_000, stream=0)
    assert np.all(r >= 1.0 / (1.0 + 2.0 * alpha) - 1e-15)
    assert np.all(r <= 1.0)


@pytest.mark.parametrize("energy", ENERGIES_MEV)
def test_energy_angle_relation_is_exact(energy: float) -> None:
    """cos theta reconstructed from r satisfies the Compton relation identically.

    r = 1 / (1 + alpha (1 - cos theta)) round-trips to float precision; this is
    algebra, not physics, and gets a tight tolerance.
    """
    alpha = energy / ELECTRON_MASS_MEV
    r = _samples(energy, 2_000, stream=1)
    cos_theta = np.array([compton_cos_theta(energy, float(ri)) for ri in r])

    assert np.all(cos_theta >= -1.0 - 1e-12)
    assert np.all(cos_theta <= 1.0 + 1e-12)
    round_trip = 1.0 / (1.0 + alpha * (1.0 - cos_theta))
    np.testing.assert_allclose(round_trip, r, rtol=1e-12)


@pytest.mark.parametrize("energy", ENERGIES_MEV)
def test_sampled_spectrum_matches_klein_nishina(energy: float) -> None:
    """Chi-squared of the sampled r histogram against the analytic pdf.

    The detection oracle of AGENTS.md section 4, applied to the sampler itself. A bias
    in the rejection logic (wrong branch weight, wrong acceptance function) shifts bin
    contents far beyond the Poisson noise at n = 30000.
    """
    alpha = energy / ELECTRON_MASS_MEV
    n = 30_000
    r = _samples(energy, n, stream=2)

    r_min = 1.0 / (1.0 + 2.0 * alpha)
    edges = np.linspace(r_min, 1.0, 33)
    observed, _ = np.histogram(r, bins=edges)

    norm, _ = integrate.quad(lambda x: float(_kn_pdf_unnormalized(np.array(x), alpha)), r_min, 1.0)
    expected = (
        np.array(
            [
                integrate.quad(lambda x: float(_kn_pdf_unnormalized(np.array(x), alpha)), lo, hi)[0]
                for lo, hi in itertools.pairwise(edges)
            ]
        )
        / norm
        * n
    )
    # Guard the chi-squared approximation, not the physics.
    assert np.all(expected > 20.0), "bin layout leaves low-count bins; rebin"

    chi2, p_value = stats.chisquare(observed, expected * observed.sum() / expected.sum())
    assert p_value > 0.01, (
        f"sampled Compton spectrum at {energy} MeV deviates from Klein-Nishina: "
        f"chi2={chi2:.1f} on {len(observed) - 1} dof, p={p_value:.2e}"
    )


@pytest.mark.parametrize("energy", ENERGIES_MEV)
def test_mean_energy_ratio_matches_quadrature(energy: float) -> None:
    """<E'/E> agrees with dense quadrature of the Klein-Nishina pdf.

    A scalar moment test: catches smooth distortions that histogram binning dilutes.
    Tolerance is 4 standard errors of the sample mean; the seed is fixed, so this is
    deterministic (AGENTS.md section 2.4).
    """
    alpha = energy / ELECTRON_MASS_MEV
    n = 30_000
    r = _samples(energy, n, stream=3)

    r_min = 1.0 / (1.0 + 2.0 * alpha)
    norm, _ = integrate.quad(lambda x: float(_kn_pdf_unnormalized(np.array(x), alpha)), r_min, 1.0)
    mean_exact, _ = integrate.quad(
        lambda x: x * float(_kn_pdf_unnormalized(np.array(x), alpha)) / norm, r_min, 1.0
    )

    standard_error = float(np.std(r, ddof=1)) / np.sqrt(n)
    z = abs(float(np.mean(r)) - mean_exact) / standard_error
    assert z < 4.0, f"<r> = {np.mean(r):.6f} vs exact {mean_exact:.6f} at {energy} MeV: z = {z:.2f}"


@pytest.mark.parametrize("energy", ENERGIES_MEV)
def test_recoil_electron_angle_conserves_momentum(energy: float) -> None:
    """Photon + electron momenta balance exactly in both components.

    Longitudinal: E == E' cos(theta_gamma) + p_e cos(theta_e). Transverse:
    E' sin(theta_gamma) == p_e sin(theta_e). Algebraic identities of the free-electron
    kinematics; float tolerance only.
    """
    import math

    from pyRadMC.physics.compton import compton_electron_cos_theta

    for sampled in _samples(energy, 1_000, stream=4):
        ratio = float(sampled)
        if ratio > 0.999999:  # no recoil; the electron angle is undefined
            continue
        cos_g = compton_cos_theta(energy, ratio)
        cos_e = compton_electron_cos_theta(energy, ratio)
        recoil = energy * (1.0 - ratio)
        p_e = math.sqrt(recoil * (recoil + 2.0 * ELECTRON_MASS_MEV))

        longitudinal = ratio * energy * cos_g + p_e * cos_e
        assert longitudinal == pytest.approx(energy, rel=1e-9)
        transverse_gamma = ratio * energy * math.sqrt(max(0.0, 1.0 - cos_g * cos_g))
        transverse_e = p_e * math.sqrt(max(0.0, 1.0 - cos_e * cos_e))
        assert transverse_e == pytest.approx(transverse_gamma, rel=1e-6, abs=1e-9)
