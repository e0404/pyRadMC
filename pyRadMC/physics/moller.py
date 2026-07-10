"""Moller (electron-electron knock-on) sampling and kinematics.

Pure scalar functions (AGENTS.md section 2.5). The delta ray is by convention the
*lower*-energy outgoing electron, so the transfer W lies in (0, T/2]; both outgoing
electrons are above ECUT by construction when the cut is ECUT.
"""

from __future__ import annotations

import math

from pyRadMC import ELECTRON_MASS_MEV
from pyRadMC.rng import RNGState, uniform

__all__ = ["moller_direction_cosines", "sample_moller_delta_energy"]


def sample_moller_delta_energy(energy: float, delta_cut: float, rng_state: RNGState) -> float:
    r"""Sample the delta-ray kinetic energy W from the restricted Moller DCS.

    Rejection from a :math:`1/x^2` envelope, x = W/T on [delta_cut/T, 1/2]: the
    envelope is inverted analytically and the acceptance function is

    .. math::

        x^2 f(x) = 1 + \left(\frac{x}{1-x}\right)^2
                 + x^2 \left(\frac{\tau}{\tau+1}\right)^2
                 - \frac{2\tau+1}{(\tau+1)^2}\,\frac{x}{1-x}
                 \; \le \; 2 + \frac{1}{4}\left(\frac{\tau}{\tau+1}\right)^2,

    the bound following from :math:`x/(1-x) \le 1` and :math:`x^2 \le 1/4` on the
    interval. Moller (1932), doi:10.1002/andp.19324060506; sampling as in Salvat et
    al., PENELOPE-2018, sec. 3.2 (doi:10.1787/32da5043-en).

    Parameters
    ----------
    energy
        Primary kinetic energy T in MeV; must exceed ``2 * delta_cut``.
    delta_cut
        Minimum delta kinetic energy in MeV (ECUT in the Class II scheme).
    rng_state
        Per-history RNG state; two uniforms per rejection round.
    """
    tau = energy / ELECTRON_MASS_MEV
    tau_ratio = tau / (tau + 1.0)
    tau_ratio_sq = tau_ratio * tau_ratio
    two_tau_term = (2.0 * tau + 1.0) / ((tau + 1.0) * (tau + 1.0))
    w_min = delta_cut / energy
    bound = 2.0 + 0.25 * tau_ratio_sq

    while True:
        u = uniform(rng_state)
        x = w_min / (1.0 - u * (1.0 - 2.0 * w_min))
        ratio = x / (1.0 - x)
        accept = 1.0 + ratio * ratio + x * x * tau_ratio_sq - two_tau_term * ratio
        if uniform(rng_state) * bound <= accept:
            return x * energy


def moller_direction_cosines(energy: float, delta_energy: float) -> tuple[float, float]:
    r"""Polar angle cosines of the delta ray and the surviving primary.

    From relativistic two-body kinematics on a stationary electron (e.g. Salvat et
    al., PENELOPE-2018, eq. 3.109):

    .. math::

        \cos\theta_\delta = \sqrt{\frac{W (T + 2mc^2)}{T (W + 2mc^2)}},
        \qquad
        \cos\theta_p = \sqrt{\frac{(T - W)(T + 2mc^2)}{T (T - W + 2mc^2)}}.

    Both longitudinal momenta sum to the incident momentum and the transverse
    momenta cancel exactly; this identity is test-pinned. The azimuths are back to
    back — the caller draws one uniform azimuth and applies it with opposite signs.

    Parameters
    ----------
    energy
        Primary kinetic energy T in MeV.
    delta_energy
        Delta-ray kinetic energy W in MeV, in (0, T/2].
    """
    two_m = 2.0 * ELECTRON_MASS_MEV
    cos_delta = math.sqrt(delta_energy * (energy + two_m) / (energy * (delta_energy + two_m)))
    surviving = energy - delta_energy
    cos_primary = math.sqrt(surviving * (energy + two_m) / (energy * (surviving + two_m)))
    return (min(1.0, cos_delta), min(1.0, cos_primary))
