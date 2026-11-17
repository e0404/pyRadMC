"""Bhabha (positron-electron knock-on) sampling and kinematics.

Pure scalar functions (AGENTS.md section 2.5), mirroring
:mod:`pyradmc.physics.moller`. The differences from Moller are physical: the
particles are distinguishable, so the delta ray is *the electron* and the transfer
W lies in (0, T] — the positron may hand over everything and come out below the
cut, which the transport must treat as a stopped positron (local deposit, then
annihilation at rest).
"""

from __future__ import annotations

import math

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.rng import RNGState, uniform

__all__ = ["bhabha_direction_cosines", "sample_bhabha_delta_energy"]


def sample_bhabha_delta_energy(energy: float, delta_cut: float, rng_state: RNGState) -> float:
    r"""Sample the delta-ray kinetic energy W from the restricted Bhabha DCS.

    Rejection from a :math:`1/x^2` envelope, x = W/T on [delta_cut/T, 1]: the
    envelope is inverted analytically and the acceptance function is

    .. math::

        \beta^2 x^2 f(x) = 1 - \beta^2 x \bigl( B_1 - x ( B_2 - x ( B_3 - x B_4 ))\bigr)
        \; \in \; (0, 1],

    with :math:`f` the bracket of the Bhabha DCS and, for :math:`y = 1/(\gamma+1)`,

    .. math::

        B_1 = 2 - y^2,\quad B_2 = (1-2y)(3+y^2),\quad
        B_3 = (1-2y)^2 + (1-2y)^3,\quad B_4 = (1-2y)^3 .

    The bound 1 is attained at x -> 0 (and, in the ultra-relativistic limit where
    the function tends to :math:`(1 - x + x^2)^2`, again at x = 1); that the
    function stays inside (0, 1] at every energy is test-pinned. Bhabha, Proc. R.
    Soc. A 154, 195 (1936), doi:10.1098/rspa.1936.0046; DCS as in ICRU Report 37
    (1984) and :func:`pyradmc.data.positron.bhabha_dcs_per_electron`; sampling as
    in Salvat et al., PENELOPE-2018, sec. 3.2 (doi:10.1787/32da5043-en).

    Free electron at rest (binding neglected), as for Moller.

    Parameters
    ----------
    energy
        Positron kinetic energy T in MeV; must exceed ``delta_cut``.
    delta_cut
        Minimum delta kinetic energy in MeV (ECUT in the Class II scheme).
    rng_state
        Per-history RNG state; two uniforms per rejection round.

    Returns
    -------
    float
        Delta-ray (electron) kinetic energy W in ``[delta_cut, T]``. The positron
        keeps ``T - W``, which may be below ``delta_cut``.
    """
    tau = energy / ELECTRON_MASS_MEV
    gamma = tau + 1.0
    beta_sq = tau * (tau + 2.0) / (gamma * gamma)
    y = 1.0 / (gamma + 1.0)
    q = 1.0 - 2.0 * y
    b1 = 2.0 - y * y
    b2 = q * (3.0 + y * y)
    b4 = q * q * q
    b3 = q * q + b4
    w_min = delta_cut / energy

    while True:
        u = uniform(rng_state)
        x = w_min / (1.0 - u * (1.0 - w_min))
        accept = 1.0 - beta_sq * x * (b1 - x * (b2 - x * (b3 - x * b4)))
        if uniform(rng_state) <= accept:
            return min(x * energy, energy)


def bhabha_direction_cosines(energy: float, delta_energy: float) -> tuple[float, float]:
    r"""Polar angle cosines of the delta electron and the scattered positron.

    Relativistic two-body kinematics on a stationary electron — the equal-mass
    relations of :func:`pyradmc.physics.moller.moller_direction_cosines`, here
    valid over the whole Bhabha range :math:`0 < W \le T`:

    .. math::

        \cos\theta_\delta = \sqrt{\frac{W (T + 2mc^2)}{T (W + 2mc^2)}},
        \qquad
        \cos\theta_+ = \sqrt{\frac{(T - W)(T + 2mc^2)}{T (T - W + 2mc^2)}}.

    Energy and momentum are conserved exactly (test-pinned): the longitudinal
    momenta sum to the incident momentum and the transverse momenta cancel. The
    azimuths are back to back — the caller draws one uniform azimuth and applies
    it with opposite signs. At W = T the positron stops (zero momentum; its cosine
    is returned as 0 and is irrelevant).

    Parameters
    ----------
    energy
        Positron kinetic energy T in MeV.
    delta_energy
        Delta-ray kinetic energy W in MeV, in (0, T].

    Returns
    -------
    tuple[float, float]
        ``(cos_delta, cos_positron)``.
    """
    two_m = 2.0 * ELECTRON_MASS_MEV
    cos_delta = math.sqrt(delta_energy * (energy + two_m) / (energy * (delta_energy + two_m)))
    surviving = max(energy - delta_energy, 0.0)
    cos_positron = math.sqrt(surviving * (energy + two_m) / (energy * (surviving + two_m)))
    return (min(1.0, cos_delta), min(1.0, cos_positron))
