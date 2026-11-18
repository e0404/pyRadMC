r"""Screened Bethe-Heitler bremsstrahlung: photon-energy and emission-angle sampling.

Pure scalar functions (AGENTS.md section 2.5), the second bremsstrahlung model next
to the thin-target ``1/k`` one of :mod:`pyradmc.physics.brems`. It supplies the
*shape* of the photon spectrum; the absolute scale is fixed in the data layer
(:mod:`pyradmc.data.bethe_heitler`) by the radiative stopping power.

Governing equation
------------------
For an electron of kinetic energy :math:`T` and total energy :math:`E = T + m_e c^2`
radiating a photon of energy :math:`k` (:math:`y = k/E`, :math:`E' = E - k`), the
Born-approximation cross-section per atom with atomic screening, differential in
:math:`k` and integrated over all angles, is (Tsai, Rev. Mod. Phys. 46, 815 (1974),
doi:10.1103/RevModPhys.46.815, sec. III.D; erratum 49, 421 (1977),
doi:10.1103/RevModPhys.49.421)

.. math::

    \frac{d\sigma}{dk} = \frac{16 \alpha r_e^2 Z^2}{3 k} \Big\{
        \left(1 - y + \tfrac34 y^2\right)
        \Big[\tfrac14 \phi_1(\gamma) - \tfrac13 \ln Z - f(Z)
             + \tfrac{1}{Z}\big(\tfrac14 \psi_1(\epsilon) - \tfrac23 \ln Z\big)\Big]
        + \tfrac18 (1 - y)
        \Big[\phi_1(\gamma) - \phi_2(\gamma)
             + \tfrac{1}{Z}\big(\psi_1(\epsilon) - \psi_2(\epsilon)\big)\Big]
    \Big\},

with the screening variables

.. math::

    \gamma = \frac{100\, m_e c^2\, k}{E E'} Z^{-1/3}, \qquad
    \epsilon = \frac{100\, m_e c^2\, k}{E E'} Z^{-2/3},

and Tsai's closed-form fits to the Thomas-Fermi screening functions (his eqs.
3.38-3.41):

.. math::

    \phi_1(\gamma) &= 20.863 - 2 \ln\!\left[1 + (0.55846 \gamma)^2\right]
        - 4 \left[1 - 0.6 e^{-0.9 \gamma} - 0.4 e^{-1.5 \gamma}\right], \\
    \phi_1 - \phi_2 &= \tfrac23 \left(1 + 6.5 \gamma + 6 \gamma^2\right)^{-1}, \\
    \psi_1(\epsilon) &= 28.340 - 2 \ln\!\left[1 + (3.621 \epsilon)^2\right]
        - 4 \left[1 - 0.7 e^{-8 \epsilon} - 0.3 e^{-29.2 \epsilon}\right], \\
    \psi_1 - \psi_2 &= \tfrac23 \left(1 + 40 \epsilon + 400 \epsilon^2\right)^{-1}.

The :math:`\phi` terms are the nuclear field screened by the atomic electrons, the
:math:`\psi` terms radiation in the field of the atomic electrons, and :math:`f(Z)`
is the Davies-Bethe-Maximon Coulomb correction (Phys. Rev. 93, 788 (1954),
doi:10.1103/PhysRev.93.788). This is the same physics as the Koch & Motz formula
3BS(e) that EGS4/EGSnrc sample (Koch & Motz, Rev. Mod. Phys. 31, 920 (1959),
doi:10.1103/RevModPhys.31.920), in the formulation Geant4's relativistic model uses
below the LPM regime. For :math:`\gamma, \epsilon \to 0` it reduces to the
complete-screening spectrum :math:`\propto (\tfrac43 - \tfrac43 y + y^2)` (test-pinned
against the closed form), for :math:`\gamma, \epsilon \to \infty` to the unscreened
Bethe-Heitler logarithm :math:`\ln(2 E E'/(m_e c^2 k)) - \tfrac12`.

What the functions here evaluate is the dimensionless shape

.. math::

    G(k) = \left(1 - y + \tfrac34 y^2\right)
           \left[X_n(\gamma) + r X_e(\epsilon)\right]
           + \tfrac18 (1 - y)\left[D_n(\gamma) + r D_e(\epsilon)\right],
    \qquad \frac{d\sigma}{dk} \propto \frac{G(k)}{k},

with :math:`X_n = \tfrac14 \phi_1 - c_n`, :math:`X_e = \tfrac14 \psi_1 - c_e`,
:math:`D_n = \phi_1 - \phi_2`, :math:`D_e = \psi_1 - \psi_2`, and five per-material
numbers :math:`(r, c_n, c_e, g_n, g_e)` that carry the composition:
:math:`\gamma = g_n\, 100 m_e c^2 k / (E E')` and
:math:`\epsilon = g_e\, 100 m_e c^2 k / (E E')`. For an element,
:math:`r = 1/Z`, :math:`c_n = \tfrac13 \ln Z + f(Z)`, :math:`c_e = \tfrac23 \ln Z`,
:math:`g_n = Z^{-1/3}`, :math:`g_e = Z^{-2/3}`; the compound rule that produces them
is documented, with its measured error, in
:func:`pyradmc.data.bethe_heitler.screening_parameters`. The parameters are *data*
(AGENTS.md 2.6): they reach these functions as arguments, from the
``CrossSectionSource``.

Stated approximations
---------------------
- **Born approximation with the high-energy Coulomb correction at every energy.**
  The shape is not reliable near the photon tip :math:`k \to T` (no Elwert factor;
  the true cross-section is finite there) nor, as a whole, below roughly 1-2 MeV,
  where partial-wave shapes (Seltzer & Berger, At. Data Nucl. Data Tables 35, 345
  (1986), doi:10.1016/0092-640X(86)90014-8) differ from it. The radiative *total* is
  unaffected, being imposed by the normalization. Where the Coulomb correction
  drives the bracket negative (high Z, within a few electron masses of the tip) the
  shape is clamped at zero.
- **No Landau-Pomeranchuk-Migdal and no dielectric (Ter-Mikaelian) suppression.**
  Both act on :math:`k \lesssim 10^{-4} E` at the upper end of the intended range
  and matter only for :math:`E` well above 1 GeV in condensed media.
- **Electron-electron bremsstrahlung only through the** :math:`\psi` **terms**, i.e.
  the usual :math:`Z(Z + \xi)` treatment; no explicit electron-electron kinematics,
  no recoil-electron ejection.
- **No electron deflection on emission** (the photon's transverse momentum is not
  balanced on the electron) and **no polarization**.
- **Emission angle to leading order, independent of** :math:`k`:
  :func:`sample_bremsstrahlung_one_minus_cos_theta`.

Stated validity: 0.2-300 MeV kinetic energy for the sampling machinery (bounded
rejection efficiency, test-pinned); the physical accuracy of the *shape* is that of
the screened Born cross-section, i.e. good above a few MeV and degrading below.
"""

from __future__ import annotations

import math

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.rng import RNGState, uniform

__all__ = [
    "bethe_heitler_rejection_bound",
    "bethe_heitler_screening_terms",
    "bethe_heitler_shape",
    "sample_bethe_heitler_photon_energy",
    "sample_bremsstrahlung_cos_theta",
    "sample_bremsstrahlung_one_minus_cos_theta",
]


def bethe_heitler_screening_terms(
    energy: float,
    photon_energy: float,
    z_ratio: float,
    c_nuc: float,
    c_el: float,
    g_nuc: float,
    g_el: float,
) -> tuple[float, float]:
    r"""Screening brackets ``(X_n + r X_e, (D_n + r D_e) / 8)`` of the shape function.

    Tsai's Thomas-Fermi fits (module docstring). Both returned terms decrease
    monotonically with the photon energy at fixed electron energy — every
    :math:`\gamma`-dependent piece is decreasing and :math:`k/(E E')` increases with
    :math:`k` — which is what the rejection envelope relies on.

    Parameters
    ----------
    energy
        Electron kinetic energy T in MeV.
    photon_energy
        Photon energy k in MeV, in (0, T].
    z_ratio, c_nuc, c_el, g_nuc, g_el
        Per-material screening parameters ``(r, c_n, c_e, g_n, g_e)``; see the module
        docstring and :func:`pyradmc.data.bethe_heitler.screening_parameters`.
    """
    total = energy + ELECTRON_MASS_MEV
    reduced = 100.0 * ELECTRON_MASS_MEV * photon_energy / (total * (total - photon_energy))
    gamma = reduced * g_nuc
    epsilon = reduced * g_el

    gamma_scaled = 0.55846 * gamma
    phi1 = (
        20.863
        - 2.0 * math.log(1.0 + gamma_scaled * gamma_scaled)
        - 4.0 * (1.0 - 0.6 * math.exp(-0.9 * gamma) - 0.4 * math.exp(-1.5 * gamma))
    )
    phi_difference = (2.0 / 3.0) / (1.0 + 6.5 * gamma + 6.0 * gamma * gamma)

    epsilon_scaled = 3.621 * epsilon
    psi1 = (
        28.340
        - 2.0 * math.log(1.0 + epsilon_scaled * epsilon_scaled)
        - 4.0 * (1.0 - 0.7 * math.exp(-8.0 * epsilon) - 0.3 * math.exp(-29.2 * epsilon))
    )
    psi_difference = (2.0 / 3.0) / (1.0 + 40.0 * epsilon + 400.0 * epsilon * epsilon)

    main = 0.25 * phi1 - c_nuc + z_ratio * (0.25 * psi1 - c_el)
    tip = 0.125 * (phi_difference + z_ratio * psi_difference)
    return (main, tip)


def bethe_heitler_shape(
    energy: float,
    photon_energy: float,
    z_ratio: float,
    c_nuc: float,
    c_el: float,
    g_nuc: float,
    g_el: float,
) -> float:
    r"""Dimensionless spectral shape ``G(k)``, with ``dsigma/dk`` proportional to ``G(k)/k``.

    .. math::

        G(k) = \max\!\Big\{0,\ \left(1 - y + \tfrac34 y^2\right)
               \left[X_n + r X_e\right]
               + \tfrac18 (1 - y)\left[D_n + r D_e\right]\Big\},
        \qquad y = \frac{k}{T + m_e c^2}.

    ``G`` is ``k dsigma/dk`` in units of ``(16/3) alpha r_e^2`` times the
    atom-number-weighted mean of ``Z^2``: the energy radiated per unit photon energy.
    The clamp at zero is the stated treatment of the region where the Coulomb
    correction overshoots (module docstring).

    Parameters
    ----------
    energy
        Electron kinetic energy T in MeV.
    photon_energy
        Photon energy k in MeV, in (0, T].
    z_ratio, c_nuc, c_el, g_nuc, g_el
        Per-material screening parameters; see :func:`bethe_heitler_screening_terms`.
    """
    main, tip = bethe_heitler_screening_terms(
        energy, photon_energy, z_ratio, c_nuc, c_el, g_nuc, g_el
    )
    y = photon_energy / (energy + ELECTRON_MASS_MEV)
    value = (1.0 - y + 0.75 * y * y) * main + (1.0 - y) * tip
    return max(value, 0.0)


def bethe_heitler_rejection_bound(
    energy: float,
    k_cut: float,
    z_ratio: float,
    c_nuc: float,
    c_el: float,
    g_nuc: float,
    g_el: float,
) -> float:
    r"""Upper bound of ``G(k)`` on ``[k_cut, T]``, the rejection envelope constant.

    Both screening brackets decrease with ``k``, so they are largest at ``k_cut``;
    :math:`1 - y` is largest there too; and :math:`p(y) = 1 - y + \tfrac34 y^2` is
    convex, so on the interval it is bounded by the larger of its two end values.
    Hence

    .. math::

        G(k) \le \max\{p(y_{cut}), p(y_{max})\}\, \max\{0, X(k_{cut})\}
                 + (1 - y_{cut})\, \tfrac18 D(k_{cut}),

    a rigorous bound (test-pinned on a dense grid), tight to within the variation
    of :math:`p`, i.e. never looser than a factor 3/2.

    Parameters
    ----------
    energy
        Electron kinetic energy T in MeV; must exceed ``k_cut``.
    k_cut
        Lowest sampled photon energy in MeV (PCUT in the transport loop).
    z_ratio, c_nuc, c_el, g_nuc, g_el
        Per-material screening parameters; see :func:`bethe_heitler_screening_terms`.
    """
    total = energy + ELECTRON_MASS_MEV
    main, tip = bethe_heitler_screening_terms(energy, k_cut, z_ratio, c_nuc, c_el, g_nuc, g_el)
    y_cut = k_cut / total
    y_max = energy / total
    p_cut = 1.0 - y_cut + 0.75 * y_cut * y_cut
    p_max = 1.0 - y_max + 0.75 * y_max * y_max
    return max(p_cut, p_max) * max(main, 0.0) + (1.0 - y_cut) * tip


def sample_bethe_heitler_photon_energy(
    energy: float,
    k_cut: float,
    z_ratio: float,
    c_nuc: float,
    c_el: float,
    g_nuc: float,
    g_el: float,
    rng_state: RNGState,
) -> float:
    r"""Sample the photon energy ``k`` on ``[k_cut, T]`` from ``G(k)/k``.

    Rejection from the ``1/k`` envelope: ``k = k_cut (T/k_cut)^u`` is drawn by exact
    inversion and accepted with probability ``G(k) / G_max``, with ``G_max`` from
    :func:`bethe_heitler_rejection_bound`. The acceptance probability is the mean of
    ``G/G_max`` over ``ln k``; it is 0.61-0.92 over 0.2-300 MeV in water, cortical
    bone, air and tungsten at ``k_cut`` = 50 keV (computed exactly; the floor of 0.6
    is test-pinned). Two uniforms per rejection round.

    The caller must not sample where the discrete cross-section vanishes
    (``T <= k_cut``); a degenerate envelope returns ``k_cut`` rather than looping.

    Parameters
    ----------
    energy
        Electron kinetic energy T in MeV; must exceed ``k_cut``.
    k_cut
        Lowest photon energy in MeV (PCUT in the transport loop).
    z_ratio, c_nuc, c_el, g_nuc, g_el
        Per-material screening parameters; see :func:`bethe_heitler_screening_terms`.
    rng_state
        Per-history RNG state.
    """
    bound = bethe_heitler_rejection_bound(energy, k_cut, z_ratio, c_nuc, c_el, g_nuc, g_el)
    if bound <= 0.0:
        return k_cut
    log_ratio = math.log(energy / k_cut)
    while True:
        k = min(k_cut * math.exp(uniform(rng_state) * log_ratio), energy)
        shape = bethe_heitler_shape(energy, k, z_ratio, c_nuc, c_el, g_nuc, g_el)
        if uniform(rng_state) * bound <= shape:
            return k


def sample_bremsstrahlung_one_minus_cos_theta(energy: float, rng_state: RNGState) -> float:
    r"""Sample ``1 - cos(theta)`` of the photon relative to the electron direction.

    Leading-order angular law: emission isotropic in the electron rest frame,
    Lorentz-transformed to the laboratory,

    .. math::

        p(\mu) = \frac{1 - \beta^2}{2 (1 - \beta \mu)^2}, \qquad \mu \in [-1, 1],

    the simplest member of the Lorentz-boosted family used for bremsstrahlung
    angles (Salvat et al., PENELOPE-2018, sec. 3.3.4, doi:10.1787/32da5043-en; the
    doubly differential forms are in Koch & Motz (1959),
    doi:10.1103/RevModPhys.31.920). Its characteristic angle is :math:`1/\gamma`:
    the median is exactly :math:`\mu = \beta`. Inverted in closed form,

    .. math::

        1 - \mu = \frac{2 (1 - \beta)(1 - u)}{1 - \beta + 2 \beta u},

    with :math:`1 - \beta = (m_e c^2 / E)^2 / (1 + \beta)` evaluated without
    cancellation. Returned as :math:`1 - \mu` because that is the quantity single
    precision can hold: at 300 MeV the typical value is :math:`1.4 \times 10^{-6}`,
    which a float32 *cosine* resolves to only a few percent (measured: the float32
    cosine fails the distribution test at 300 MeV, this form passes), while
    :math:`\sin\theta = \sqrt{(1 - \mu)(1 + \mu)}` from this value is exact.

    Stated approximations: the dependence of the angular distribution on the photon
    energy and on Z, and the dipole anisotropy in the rest frame, are neglected.

    Parameters
    ----------
    energy
        Electron kinetic energy T in MeV at emission.
    rng_state
        Per-history RNG state; consumes one uniform.
    """
    total = energy + ELECTRON_MASS_MEV
    beta = math.sqrt(energy * (energy + 2.0 * ELECTRON_MASS_MEV)) / total
    inverse_gamma = ELECTRON_MASS_MEV / total
    one_minus_beta = inverse_gamma * inverse_gamma / (1.0 + beta)
    u = uniform(rng_state)
    return 2.0 * one_minus_beta * (1.0 - u) / (one_minus_beta + 2.0 * beta * u)


def sample_bremsstrahlung_cos_theta(energy: float, rng_state: RNGState) -> float:
    """Sample the photon polar-angle cosine relative to the electron direction.

    ``1 - x`` with ``x`` from :func:`sample_bremsstrahlung_one_minus_cos_theta`
    (angular law, references and approximations there); one uniform. Adequate in
    double precision at any energy and in single precision up to a few tens of
    MeV; above that a float32 kernel should take the ``1 - cos`` form.

    Parameters
    ----------
    energy
        Electron kinetic energy T in MeV at emission.
    rng_state
        Per-history RNG state; consumes one uniform.
    """
    return 1.0 - sample_bremsstrahlung_one_minus_cos_theta(energy, rng_state)
