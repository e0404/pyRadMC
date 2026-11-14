"""Two-photon positron annihilation in flight: photon energies and directions.

Pure scalar functions (AGENTS.md section 2.5). Whether an annihilation happens is
the data layer's cross-section
(:meth:`pyradmc.data.interface.CrossSectionSource.annihilation_cross_section`);
these functions distribute the available energy ``T + 2 m_e c^2`` and the momentum
between the two photons. Annihilation *at rest* (a positron that reached the
transport cutoff) stays :func:`pyradmc.transport.particles.annihilate_at_rest`.

Stated approximations: free target electron at rest; one- and three-photon
annihilation neglected (see :mod:`pyradmc.data.positron`).
"""

from __future__ import annotations

import math

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.rng import RNGState, uniform

__all__ = ["annihilation_direction_cosines", "sample_annihilation_photon_energy"]


def sample_annihilation_photon_energy(energy: float, rng_state: RNGState) -> float:
    r"""Sample the energy of one annihilation photon for a positron in flight, in MeV.

    With :math:`\varepsilon = k_1/(T + 2mc^2)` and :math:`\gamma = 1 + T/mc^2`, the
    Heitler differential cross-section in its one-photon-labelled form is

    .. math::

        \frac{d\sigma}{d\varepsilon} \propto \frac{1}{\varepsilon}\,g(\varepsilon),
        \qquad
        g(\varepsilon) = 1 - \varepsilon
            + \frac{2\gamma\varepsilon - 1}{\varepsilon\,(\gamma+1)^2},

    on :math:`\varepsilon_{min,max} = \tfrac12 \mp \tfrac12\sqrt{(\gamma-1)/(\gamma+1)}`.
    :math:`\varepsilon` is drawn from the :math:`1/\varepsilon` envelope,
    :math:`\varepsilon = \varepsilon_{min}(\varepsilon_{max}/\varepsilon_{min})^{\xi}`,
    and accepted with probability :math:`g/g_{max}`, where
    :math:`g_{max} = 1 - 2/(\gamma+1)^2` is the exact maximum, attained at
    :math:`\varepsilon = 1/(\gamma+1)` inside the interval (test-pinned). This is
    the EGS4 ``ANNIH`` / Geant4 ``G4eeToTwoGammaModel`` scheme with the tight
    bound. Heitler, *The Quantum Theory of Radiation*, 3rd ed. (1954); Nelson,
    Hirayama & Rogers, *The EGS4 Code System*, SLAC-265 (1985); Agostinelli et
    al., NIM A 506, 250 (2003), doi:10.1016/S0168-9002(03)01368-8 (Geant4
    Physics Reference Manual, "Positron annihilation into two photons").

    :math:`\varepsilon_{min}` is evaluated as
    :math:`1/[2(\gamma+1)\varepsilon_{max}]` (the two limits multiply to
    :math:`1/[2(\gamma+1)]`), which does not cancel at high energy in single
    precision.

    Parameters
    ----------
    energy
        Positron kinetic energy T in MeV; must be positive.
    rng_state
        Per-history RNG state; two uniforms per rejection round.

    Returns
    -------
    float
        Energy :math:`k_1` of the labelled photon. The other photon takes
        :math:`k_2 = T + 2 m_e c^2 - k_1`, so energy is conserved exactly by
        construction; directions from :func:`annihilation_direction_cosines`.
    """
    tau = energy / ELECTRON_MASS_MEV
    gamma = tau + 1.0
    gamma_plus = tau + 2.0
    eps_max = 0.5 + 0.5 * math.sqrt(tau / gamma_plus)
    eps_min = 0.5 / (gamma_plus * eps_max)
    log_ratio = math.log(eps_max / eps_min)
    inv_gp_sq = 1.0 / (gamma_plus * gamma_plus)
    bound = 1.0 - 2.0 * inv_gp_sq

    while True:
        eps = eps_min * math.exp(uniform(rng_state) * log_ratio)
        accept = 1.0 - eps + (2.0 * gamma * eps - 1.0) * inv_gp_sq / eps
        if uniform(rng_state) * bound <= accept:
            return eps * (energy + 2.0 * ELECTRON_MASS_MEV)


def annihilation_direction_cosines(energy: float, photon_energy: float) -> tuple[float, float]:
    r"""Polar angle cosines of the two annihilation photons about the positron direction.

    From energy-momentum conservation on a stationary electron, with
    :math:`E = T + 2mc^2` the available energy, :math:`p = \sqrt{T(T+2mc^2)}` the
    positron momentum (times c) and :math:`k_2 = E - k_1`:

    .. math::

        \cos\theta_i = \frac{(k_i - mc^2)\,E}{k_i\,p}, \qquad i = 1, 2.

    (Equivalent to the Compton-like relation :math:`k_i (E - p\cos\theta_i) =
    mc^2 E`.) The photon momenta then sum to the positron's exactly: the
    longitudinal components add to p and the transverse ones cancel with
    back-to-back azimuths — the caller draws one uniform azimuth and applies it
    with opposite signs. Test-pinned over sampled events from 1 keV to 300 MeV.

    Parameters
    ----------
    energy
        Positron kinetic energy T in MeV; must be positive.
    photon_energy
        Energy :math:`k_1` of the first photon in MeV, inside the kinematic
        interval (as returned by :func:`sample_annihilation_photon_energy`).

    Returns
    -------
    tuple[float, float]
        ``(cos_theta_1, cos_theta_2)``, clamped to [-1, 1] against round-off at
        the interval ends.
    """
    total = energy + 2.0 * ELECTRON_MASS_MEV
    momentum = math.sqrt(energy * total)
    other = total - photon_energy
    cos_first = (photon_energy - ELECTRON_MASS_MEV) * total / (photon_energy * momentum)
    cos_second = (other - ELECTRON_MASS_MEV) * total / (other * momentum)
    return (min(1.0, max(-1.0, cos_first)), min(1.0, max(-1.0, cos_second)))
