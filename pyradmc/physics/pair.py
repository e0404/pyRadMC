r"""Pair-production kinematics: energy sharing and lepton polar angles.

Pure scalar functions (AGENTS.md section 2.5). The *whether* of pair production is
the ``CrossSectionSource``'s pair channel; these functions decide only how the
available energy and the directions are distributed. The per-material screening
scalars come from the data layer
(:meth:`pyradmc.data.interface.CrossSectionSource.pair_screening`,
:mod:`pyradmc.data.pair`, where the model, its references and its stated
approximations are written out); nothing here knows a material.

The screening functions are public only because the Warp loader compiles a
module's ``__all__`` and the sampler calls them; they are not a second access path
to cross-section data.
"""

from __future__ import annotations

import math

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.rng import RNGState, uniform

__all__ = [
    "pair_screening_f1",
    "pair_screening_f2",
    "sample_pair_cos_theta",
    "sample_pair_electron_energy",
    "sample_pair_one_minus_cos_theta",
]

PAIR_LOW_ENERGY_LIMIT_MEV: float = 2.0
"""Photon energy below which the sharing is uniform (EGS4/Geant4 convention; the
screened Born DCS is negative near threshold). See :mod:`pyradmc.data.pair`."""

PAIR_COULOMB_ENERGY_MEV: float = 50.0
"""Photon energy at and above which the Coulomb-corrected ``F`` is used."""

_F_ABOVE_STEP: float = 36.643_025_644_5
"""``42.24 - 8.368 ln(1.952)``: both screening combinations just above delta = 1."""


def pair_screening_f1(delta: float) -> float:
    r""":math:`3\Phi_1(\delta) - \Phi_2(\delta)` of the Butcher-Messel screening fits.

    ``42.392 - 7.796 delta + 1.961 delta^2`` for ``delta <= 1`` and
    ``42.24 - 8.368 ln(delta + 0.952)`` above. Butcher & Messel (1960),
    doi:10.1016/0029-5582(60)90162-0; the :math:`\Phi_i` themselves are
    :func:`pyradmc.data.pair.screening_phi1` and ``screening_phi2``.
    """
    if delta > 1.0:
        return 42.24 - 8.368 * math.log(delta + 0.952)
    return 42.392 - delta * (7.796 - 1.961 * delta)


def pair_screening_f2(delta: float) -> float:
    r""":math:`\tfrac32\Phi_1(\delta) + \tfrac12\Phi_2(\delta)`; see :func:`pair_screening_f1`.

    ``41.405 - 5.828 delta + 0.8945 delta^2`` for ``delta <= 1`` and
    ``42.24 - 8.368 ln(delta + 0.952)`` above.
    """
    if delta > 1.0:
        return 42.24 - 8.368 * math.log(delta + 0.952)
    return 41.405 - delta * (5.828 - 0.8945 * delta)


def sample_pair_electron_energy(
    energy: float,
    screening_constant: float,
    coulomb_low: float,
    coulomb_high: float,
    rng_state: RNGState,
) -> float:
    r"""Sample the electron's kinetic energy in pair production, in MeV.

    Screened Bethe-Heitler (Butcher-Messel screening, Coulomb correction at and
    above 50 MeV) by composition-rejection, as in EGS4 and Geant4's
    ``G4BetheHeitlerModel``. With :math:`\varepsilon` the total-energy fraction of
    one lepton, :math:`u = \tfrac12 - \varepsilon` and
    :math:`\delta = c\,\varepsilon_0/[\varepsilon(1-\varepsilon)]`
    (:math:`c` = ``screening_constant``, :math:`\varepsilon_0 = m_e c^2/k`), the
    DCS of :func:`pyradmc.data.pair.pair_sharing_dcs` is identically

    .. math::

        \frac{d\sigma}{d\varepsilon} \propto
            \tfrac23\,u^2\,\bigl[F_1(\delta) - F\bigr]
          + \tfrac13\,\bigl[F_2(\delta) - F\bigr],
        \qquad F_1 = 3\Phi_1 - \Phi_2,\quad F_2 = \tfrac32\Phi_1 + \tfrac12\Phi_2 .

    Both brackets fall with :math:`\delta`, which is smallest at
    :math:`\varepsilon = \tfrac12`, and vanish together at
    :math:`\delta_{max} = \exp[(42.24 - F)/8.368] - 0.952`; so
    :math:`\varepsilon` is sampled on
    :math:`[\max(\varepsilon_0, \varepsilon_1), \tfrac12]` with
    :math:`\varepsilon_1 = \tfrac12 - \tfrac12\sqrt{1 - \delta_{min}/\delta_{max}}`,
    choosing the :math:`u^2` or the flat term in proportion to their integrals and
    rejecting on the bracket over its bound. The bound is the bracket at
    :math:`\delta_{min}` — or just above :math:`\delta = 1` when
    :math:`\delta_{min} \le 1`, because the published fit steps *up* by about 0.1
    there (without this the rejection would silently clip the DCS). The sampled
    fraction is assigned to either lepton with equal probability: the Born DCS is
    symmetric.

    Below 2 MeV (and wherever :math:`\delta_{min} \ge \delta_{max}`, which cannot
    happen above 2 MeV for physical parameters) the kinetic energy is split
    uniformly — the stated near-threshold approximation of the model.

    References and approximations: :mod:`pyradmc.data.pair`. Valid from threshold
    to a few hundred MeV (no LPM effect).

    Parameters
    ----------
    energy
        Photon energy k in MeV; must exceed ``2 m_e c^2``.
    screening_constant, coulomb_low, coulomb_high
        The material's :class:`pyradmc.data.pair.PairScreening` scalars.
    rng_state
        Per-history RNG state. One uniform below 2 MeV; three per rejection round
        plus one for the charge assignment above.

    Returns
    -------
    float
        Electron kinetic energy :math:`T_-` in ``[0, k - 2 m_e c^2]``. The positron
        takes the rest, :math:`T_+ = k - 2 m_e c^2 - T_-`, so energy is conserved
        exactly by construction.
    """
    kinetic = energy - 2.0 * ELECTRON_MASS_MEV
    if energy < PAIR_LOW_ENERGY_LIMIT_MEV:
        return uniform(rng_state) * kinetic

    eps0 = ELECTRON_MASS_MEV / energy
    fz = coulomb_low
    if energy >= PAIR_COULOMB_ENERGY_MEV:
        fz = coulomb_high
    delta_min = 4.0 * screening_constant * eps0
    delta_max = math.exp((42.24 - fz) / 8.368) - 0.952
    if delta_min >= delta_max:
        return uniform(rng_state) * kinetic

    eps_min = 0.5 - 0.5 * math.sqrt(1.0 - delta_min / delta_max)
    if eps_min < eps0:
        eps_min = eps0
    u_max = 0.5 - eps_min

    f1_bound = pair_screening_f1(delta_min)
    f2_bound = pair_screening_f2(delta_min)
    if delta_min <= 1.0:
        f1_bound = max(f1_bound, _F_ABOVE_STEP)
        f2_bound = max(f2_bound, _F_ABOVE_STEP)
    f1_bound -= fz
    f2_bound -= fz
    n1 = u_max * u_max * f1_bound
    n2 = 1.5 * f2_bound

    # Loop body written without loop-carried or branch-defined variables and with
    # the return inside it: the form Warp's codegen accepts for a dynamic loop.
    while True:
        pick = uniform(rng_state) * (n1 + n2)
        xi = uniform(rng_state)
        u = u_max * xi
        if pick < n1:
            u = u_max * math.pow(xi, 1.0 / 3.0)
        eps = 0.5 - u
        delta = screening_constant * eps0 / (eps * (1.0 - eps))
        accept = (pair_screening_f2(delta) - fz) / f2_bound
        if pick < n1:
            accept = (pair_screening_f1(delta) - fz) / f1_bound
        if uniform(rng_state) <= accept:
            if uniform(rng_state) < 0.5:
                eps = 1.0 - eps
            # eps >= eps0 analytically; the clamp only guards float round-off.
            return min(max(eps * energy - ELECTRON_MASS_MEV, 0.0), kinetic)


def sample_pair_one_minus_cos_theta(kinetic_energy: float, rng_state: RNGState) -> float:
    r"""Sample :math:`1 - \cos\theta` of one pair lepton about the photon direction.

    Leading-order angular distribution

    .. math::

        p(\mu) \propto (1 - \beta\mu)^{-2}, \qquad -1 \le \mu \le 1,

    with :math:`\beta` the lepton's speed — the leading term of the
    Sauter-Gluckstern-Hull distribution (Motz, Olsen & Koch, Rev. Mod. Phys. 41,
    581 (1969), doi:10.1103/RevModPhys.41.581), as used by PENELOPE (Salvat et
    al., PENELOPE-2018, sec. 2.4, doi:10.1787/32da5043-en) and offered by EGSnrc as
    its leading-term pair angular option. Inverted analytically,
    :math:`\mu = (2\xi - 1 + \beta)/[(2\xi - 1)\beta + 1]`, evaluated as

    .. math::

        1 - \mu = \frac{2(1-\beta)(1-\xi)}{(1-\beta) + 2\beta\xi},
        \qquad 1 - \beta = \frac{1}{\gamma^2(1+\beta)},

    and returned in that form: at 300 MeV the cone is :math:`1 - \mu \sim 10^{-6}`,
    a couple of dozen steps of a single-precision cosine. The transport loops hand
    it to :func:`pyradmc.physics.direction.rotate_direction_one_minus_cos`, which
    never forms the cosine.

    Stated approximations: each lepton is sampled independently and their azimuths
    are independent and uniform (the caller draws them). The nucleus takes up the
    recoil, so the pair alone does **not** balance the photon's momentum — only
    energy is conserved by the pair.

    Parameters
    ----------
    kinetic_energy
        The lepton's kinetic energy in MeV (non-negative; isotropic at zero).
    rng_state
        Per-history RNG state; one uniform.

    Returns
    -------
    float
        :math:`1 - \cos\theta` in [0, 2].
    """
    tau = kinetic_energy / ELECTRON_MASS_MEV
    gamma = tau + 1.0
    beta = math.sqrt(tau * (tau + 2.0)) / gamma
    one_minus_beta = 1.0 / (gamma * gamma * (1.0 + beta))
    xi = uniform(rng_state)
    return 2.0 * one_minus_beta * (1.0 - xi) / (one_minus_beta + 2.0 * beta * xi)


def sample_pair_cos_theta(kinetic_energy: float, rng_state: RNGState) -> float:
    r"""Sample the polar-angle cosine of one pair lepton about the photon direction.

    :func:`sample_pair_one_minus_cos_theta` (distribution, references and stated
    approximations there), returned as the cosine. One uniform. The cosine form
    is for double-precision callers and tests; a kernel keeps ``1 - cos``.
    """
    return 1.0 - sample_pair_one_minus_cos_theta(kinetic_energy, rng_state)
