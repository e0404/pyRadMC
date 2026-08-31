"""Compton scattering: Klein-Nishina sampling and kinematics.

Pure scalar functions (AGENTS.md section 2.5): no allocation, no objects, RNG state
passed in. Electron binding and Doppler broadening are neglected — free electrons at
rest — a stated approximation adequate above ~100 keV in low-Z media (this engine's
PCUT is 50 keV, where the error on the incoherent channel is a few percent).
"""

from __future__ import annotations

import math

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.rng import RNGState, uniform

__all__ = ["compton_cos_theta", "compton_electron_cos_theta", "sample_compton_energy_ratio"]


def sample_compton_energy_ratio(energy: float, rng_state: RNGState) -> float:
    r"""Sample the scattered/incident photon energy ratio r = E'/E via Kahn rejection.

    Samples exactly from the Klein-Nishina differential cross-section

    .. math::

        \frac{d\sigma}{dr} \propto r + \frac{1}{r} - \sin^2\theta(r),
        \qquad r \in \left[\frac{1}{1 + 2\alpha},\, 1\right],
        \quad \alpha = E / m_e c^2,

    by Kahn's two-branch mixture in x = 1/r: with probability
    (1 + 2 alpha)/(9 + 2 alpha), x is drawn uniformly on [1, 1 + 2 alpha] and accepted
    with 4 (x - 1)/x^2; otherwise x = (1 + 2 alpha)/(1 + 2 alpha u) and accepted with
    (cos^2 theta + 1/x)/2.

    Kahn (1956), RAND AECU-3259; see also Salvat et al., PENELOPE-2018, sec. 2.3
    (doi:10.1787/32da5043-en). Exact at all energies; the rejection *efficiency*
    degrades above a few MeV, where Koblinger's direct method is preferred — an
    acceptable cost in the reference backend; the Warp kernels adopted it
    unchanged, since rejection efficiency is not their bottleneck.

    Parameters
    ----------
    energy
        Incident photon energy in MeV.
    rng_state
        Per-history RNG state; consumed three uniforms per rejection round.
    """
    alpha = energy / ELECTRON_MASS_MEV
    two_alpha = 2.0 * alpha
    branch_probability = (1.0 + two_alpha) / (9.0 + two_alpha)

    while True:
        u1 = uniform(rng_state)
        u2 = uniform(rng_state)
        u3 = uniform(rng_state)
        if u1 <= branch_probability:
            x = 1.0 + two_alpha * u2
            if u3 <= 4.0 * (x - 1.0) / (x * x):
                return 1.0 / x
        else:
            x = (1.0 + two_alpha) / (1.0 + two_alpha * u2)
            cos_theta = 1.0 - (x - 1.0) / alpha
            if u3 <= 0.5 * (cos_theta * cos_theta + 1.0 / x):
                return 1.0 / x


def compton_cos_theta(energy: float, energy_ratio: float) -> float:
    r"""Polar scattering angle cosine from the Compton relation.

    .. math::

        \cos\theta = 1 + \frac{1}{\alpha} - \frac{1}{\alpha r},
        \qquad \alpha = E / m_e c^2, \quad r = E'/E.

    Compton (1923), doi:10.1103/PhysRev.21.483. Clamped to [-1, 1] against float
    round-off at the kinematic endpoints.

    Parameters
    ----------
    energy
        Incident photon energy in MeV.
    energy_ratio
        Scattered/incident energy ratio r = E'/E from
        :func:`sample_compton_energy_ratio`.
    """
    alpha = energy / ELECTRON_MASS_MEV
    cos_theta = 1.0 + 1.0 / alpha - 1.0 / (alpha * energy_ratio)
    return min(1.0, max(-1.0, cos_theta))


def compton_electron_cos_theta(energy: float, energy_ratio: float) -> float:
    r"""Polar angle cosine of the Compton recoil electron.

    Momentum conservation along the incident direction, with photon momenta equal to
    their energies and the electron momentum :math:`p_e = \sqrt{T (T + 2 m_e c^2)}`
    for the recoil kinetic energy :math:`T = E (1 - r)`:

    .. math::

        \cos\theta_e = \frac{E - E' \cos\theta_\gamma}{p_e}.

    Exact free-electron kinematics; the transverse balance
    :math:`p_e \sin\theta_e = E' \sin\theta_\gamma` is test-pinned. The electron
    azimuth is opposite the photon azimuth. Undefined at r = 1 (no recoil): callers
    only spawn an electron when T is above the production threshold.

    Parameters
    ----------
    energy
        Incident photon energy E in MeV.
    energy_ratio
        Scattered/incident energy ratio r = E'/E, strictly below 1.
    """
    recoil = energy * (1.0 - energy_ratio)
    p_electron = math.sqrt(recoil * (recoil + 2.0 * ELECTRON_MASS_MEV))
    cos_gamma = compton_cos_theta(energy, energy_ratio)
    cos_electron = (energy - energy_ratio * energy * cos_gamma) / p_electron
    return min(1.0, max(-1.0, cos_electron))
