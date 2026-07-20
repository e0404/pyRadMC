r"""Goudsmit-Saunderson multiple-scattering moments.

Host-side table and moment construction for the exact multiple-scattering angular
distribution. The sampling routine that *consumes* this lives in
:mod:`pyRadMC.physics.gs`; this module is data-layer (AGENTS.md section 3), so it
may allocate and use NumPy.

The Goudsmit-Saunderson result: for a path carrying ``Lambda`` elastic mean free
paths whose single-scattering distribution has normalized Legendre moments

.. math::

    G_\ell = \int_{-1}^{1} \bigl(1 - P_\ell(\mu)\bigr)\,
             \frac{1}{\sigma}\frac{d\sigma}{d\mu}\, d\mu ,

the accumulated deflection has moments and density

.. math::

    \langle P_\ell \rangle = e^{-\Lambda G_\ell}, \qquad
    f(\mu; \Lambda) = \sum_\ell \frac{2\ell + 1}{2}\, e^{-\Lambda G_\ell} P_\ell(\mu).

Goudsmit & Saunderson, Phys. Rev. 57, 24 (1940), doi:10.1103/PhysRev.57.24; see
also Kawrakow & Bielajew, NIM B 134, 325 (1998), doi:10.1016/S0168-583X(97)00723-4
for the electron-step algorithm this feeds.

**Why this is exactly compatible with the Gaussian hinge it replaces.** The first
transport moment fixes the mass angular scattering power,
``T = 2 (N_A/M) sigma_el G_1``, which is precisely the quantity
:meth:`~pyRadMC.data.interface.CrossSectionSource.scattering_power` already
reports. Anchoring ``sigma_el`` to that value makes the GS second moment reduce to
the Fermi-Eyges ``<theta^2> = T rho s`` in the small-step limit *identically*, so
GS is a refinement of the present model rather than a different one: it adds the
large-angle single-scattering tail and the saturation at long steps, and changes
nothing else. Test-pinned in ``tests/unit/test_gs_moments.py``.

**Stated approximation (the screened-Rutherford default).** With no elastic
differential cross section on hand, the single-scattering shape is taken as
screened Rutherford with a single Moliere screening parameter ``eta``, the model
underlying Moliere and PRESTA-II theory. It is exact in the Coulomb tail and
approximate near the screening angle; a source carrying real differential data
(the EEDL-compiled backend, MF=26/MT=525) overrides the moments with the
measured shape. Moliere, Z. Naturforsch. 3a, 78 (1948).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "first_transport_moment",
    "gs_cumulative",
    "gs_inverse_cdf",
    "mean_square_angle",
    "moliere_screening",
    "screened_rutherford_cos_theta",
    "screened_rutherford_moments",
]

_ELECTRON_MASS_MEV: float = 0.51099895
_HBAR_C_MEV_FM: float = 197.3269804
_BOHR_RADIUS_FM: float = 5.29177210903e4
_FINE_STRUCTURE: float = 7.2973525693e-3

_THOMAS_FERMI_MEV: float = _HBAR_C_MEV_FM / (0.885 * _BOHR_RADIUS_FM)
"""``hbar c / (0.885 a_0)`` in MeV: the momentum scale of the screening angle.

The Thomas-Fermi screening radius is ``a = 0.885 a_0 Z^{-1/3}``, so the
characteristic screening angle is ``chi_0 = (hbar c / pc) Z^{1/3} / (0.885 a_0)``.
Folding the constants once here keeps the unit conversion — the step where a
Bohr radius in the wrong unit moves ``eta`` by many decades — in a single place.
"""

_SERIES_CUTOFF: float = 1.0e-12
"""Damping factor ``exp(-Lambda G_l)`` below which series terms are dropped.

The series is truncated where its own terms are negligible rather than at a
fixed order, because the number of significant terms varies by orders of
magnitude across the step sizes the transport takes: the damping is set by
``Lambda G_l``, so a long step needs a handful of terms and a short one needs
hundreds. Truncating too early shows up as Gibbs ringing at the forward peak,
which :func:`gs_cumulative` would then hand to the inverter as a non-monotone
CDF — the failure this cutoff exists to prevent.
"""

_QUADRATURE_POINTS: int = 1 << 16
"""Nodes for the moment quadrature.

Placed uniformly in ``t = ln((xi + eta)/eta)`` rather than in the CDF variable
``xi`` itself. The reason is that the single-scattering weight is exactly

.. math:: 1 - \\mu(\\xi) = \\frac{2\\eta(1-\\xi)}{\\xi + \\eta},

which is ``~1/xi`` over the whole forward peak and contributes the familiar
``2 eta ln(1/eta)`` logarithmically, from ``xi ~ eta`` upward. A uniform ``xi``
grid puts its first node at ``1/(2N)`` and therefore silently discards every
decade between ``eta`` and that node — 8 percent of ``G_1`` at ``eta = 1e-6``,
which the closed-form test caught. The log substitution resolves down to
``xi ~ eta`` for any screening parameter, and reproduces the closed-form ``G_1``
to better than 1e-9 relative across the full range.
"""


def moliere_screening(composition: tuple[tuple[int, float], ...], energy: float) -> float:
    r"""Moliere screening parameter ``eta`` for a material at kinetic ``energy``.

    Per element, the screening angle carries Moliere's relativistic and
    spin/exchange correction,

    .. math::

        \chi_a^2 = \chi_0^2 \left(1.13 + 3.76 \left(\frac{\alpha Z}{\beta}\right)^2\right),
        \qquad \chi_0 = \frac{\hbar c}{pc}\,\frac{Z^{1/3}}{0.885 a_0},

    and ``eta = chi_a^2 / 4`` is the parameter of the screened-Rutherford
    (Wentzel) law in :func:`screened_rutherford_cos_theta`. Moliere,
    Z. Naturforsch. 2a, 133 (1947) and 3a, 78 (1948); the correction factor is
    Bethe, Phys. Rev. 89, 1256 (1953), doi:10.1103/PhysRev.89.1256.

    For a compound the elemental screening angles combine as a **geometric mean
    weighted by ``w_i Z_i (Z_i + 1) / A_i``** — the per-unit-mass elastic
    scattering strength, the same ``Z(Z+1)/A`` weighting the compiled scattering
    powers are validated against. This is Bethe's compound prescription, and it
    is exact for a single element (test-pinned).

    **Stated approximation.** A single effective screening parameter cannot
    represent a mixture whose elements screen at genuinely different angles; the
    geometric mean is the standard first moment of ``ln chi_a^2`` and is what
    Moliere theory for compounds assumes. For the low-Z media this engine
    transports it is well inside the other approximations in the multiple-
    scattering model. A source carrying real elastic differential data can
    override the moments outright rather than route through ``eta``.

    Parameters
    ----------
    composition
        ``((Z, mass_fraction), ...)`` as carried by
        :class:`~pyRadMC.data.materials.MaterialData`.
    energy
        Electron kinetic energy in MeV.
    """
    from pyRadMC.data.materials import STANDARD_ATOMIC_WEIGHT

    if not composition:
        raise ValueError("empty composition")
    if energy <= 0.0:
        raise ValueError(f"non-positive electron kinetic energy {energy} MeV")

    total_energy = energy + _ELECTRON_MASS_MEV
    momentum = math.sqrt(energy * (energy + 2.0 * _ELECTRON_MASS_MEV))  # pc, MeV
    beta = momentum / total_energy

    weight_sum = 0.0
    log_sum = 0.0
    for z, fraction in composition:
        # Elastic strength per unit mass: Rutherford scales as Z(Z+1) per atom.
        weight = fraction * z * (z + 1) / STANDARD_ATOMIC_WEIGHT[z]
        chi_0 = _THOMAS_FERMI_MEV * z ** (1.0 / 3.0) / momentum
        correction = 1.13 + 3.76 * (_FINE_STRUCTURE * z / beta) ** 2
        weight_sum += weight
        log_sum += weight * math.log(0.25 * chi_0 * chi_0 * correction)
    return math.exp(log_sum / weight_sum)


def screened_rutherford_cos_theta(xi: float, eta: float) -> float:
    r"""Single elastic scattering cosine from the screened-Rutherford CDF.

    Exact inversion of

    .. math::

        \frac{1}{\sigma}\frac{d\sigma}{d\mu} =
            \frac{2\eta(1+\eta)}{(1 - \mu + 2\eta)^2},

    the Wentzel/screened-Rutherford form normalized on ``mu in [-1, 1]``. The
    screening parameter ``eta`` is half the squared screening angle in the
    small-angle limit; it removes the Coulomb divergence at ``mu -> 1``.

    Parameters
    ----------
    xi
        Uniform variate on ``[0, 1)``.
    eta
        Moliere screening parameter, dimensionless and positive.
    """
    inverse = xi / (2.0 * eta * (1.0 + eta)) + 1.0 / (2.0 + 2.0 * eta)
    return 1.0 + 2.0 * eta - 1.0 / inverse


def first_transport_moment(eta: float) -> float:
    r"""Closed-form ``G_1`` for the screened-Rutherford distribution.

    .. math::

        G_1 = \langle 1 - \mu \rangle
            = 2\eta(1+\eta)\ln\!\frac{1+\eta}{\eta} - 2\eta .

    This is the quantity the mass scattering power is built from,
    ``T = 2 (N_A/M) sigma_el G_1``, and therefore the anchor between GS and the
    Fermi-Eyges hinge. For small ``eta`` it reduces to the familiar
    ``2 eta (ln(1/eta) - 1)``.
    """
    return float(2.0 * eta * (1.0 + eta) * np.log((1.0 + eta) / eta) - 2.0 * eta)


def screened_rutherford_moments(eta: float, l_max: int) -> np.ndarray:
    r"""Legendre moments ``G_l``, ``l = 0 .. l_max``, of one elastic scatter.

    Integrated in the CDF variable ``xi``, under which the single-scattering
    measure is uniform:

    .. math::

        G_\ell = \int_0^1 \bigl(1 - P_\ell(\mu(\xi))\bigr)\, d\xi ,

    evaluated on a grid uniform in ``t = ln((xi + eta)/eta)`` so that the
    logarithmic forward peak is resolved; see :data:`_QUADRATURE_POINTS`.

    ``G_0`` is identically zero (``P_0 = 1``) and is returned as such, since a
    nonzero value would damp the entire GS series and break normalization.

    Returns
    -------
    ndarray
        Shape ``(l_max + 1,)``, ascending in ``l``, non-negative and increasing
        in ``l`` for a forward-peaked distribution.
    """
    if eta <= 0.0:
        raise ValueError(f"screening parameter must be positive, got {eta}")
    if l_max < 0:
        raise ValueError(f"l_max must be non-negative, got {l_max}")

    # Midpoint rule in t = ln((xi + eta)/eta), which maps [0, 1] in xi onto
    # [0, ln((1+eta)/eta)] and spreads nodes evenly across the decades of the
    # forward peak. The Jacobian dxi = eta e^t dt carries the measure.
    t_max = np.log((1.0 + eta) / eta)
    t = (np.arange(_QUADRATURE_POINTS) + 0.5) * (t_max / _QUADRATURE_POINTS)
    exp_t = np.exp(t)
    xi = eta * (exp_t - 1.0)
    weight = eta * exp_t * (t_max / _QUADRATURE_POINTS)
    # Exact, and free of the cancellation in 1 + 2 eta - 1/inv near mu -> 1.
    mu = 1.0 - 2.0 * eta * (1.0 - xi) / (xi + eta)

    moments = np.empty(l_max + 1)
    moments[0] = 0.0
    total = weight.sum()
    # Upward recurrence (n+1) P_{n+1} = (2n+1) mu P_n - n P_{n-1}: the whole
    # ladder in one pass, rather than re-deriving it per order.
    p_prev = np.ones_like(mu)  # P_0
    p_curr = mu.copy()  # P_1
    for ell in range(1, l_max + 1):
        moments[ell] = float(total - np.dot(p_curr, weight))
        p_prev, p_curr = (
            p_curr,
            ((2 * ell + 1) * mu * p_curr - ell * p_prev) / (ell + 1),
        )
    return moments


def gs_cumulative(lam: float, moments: np.ndarray, mu: np.ndarray) -> np.ndarray:
    r"""Cumulative GS angular distribution ``F(mu)`` over a path of ``lam`` MFPs.

    Integrating the GS series term by term with
    ``\int_{-1}^{x} P_\ell = (P_{\ell+1}(x) - P_{\ell-1}(x))/(2\ell+1)`` collapses
    the ``(2\ell+1)/2`` weights exactly:

    .. math::

        F(x) = \frac{1 + x}{2}
             + \frac{1}{2} \sum_{\ell \ge 1} e^{-\Lambda G_\ell}
               \bigl(P_{\ell+1}(x) - P_{\ell-1}(x)\bigr).

    The ``l = 0`` term is the isotropic ``(1 + x)/2`` and is kept in closed form.

    Returns a CDF clamped to ``[0, 1]`` and forced non-decreasing: a truncated
    Legendre series rings slightly at the forward peak, and an inverter handed a
    non-monotone CDF produces a silently wrong angular distribution rather than
    an error. The clamp is a numerical guard on a converged series, not a
    physics approximation — the residual it removes is at the truncation level
    (:data:`_SERIES_CUTOFF`), which the monotonicity test pins.
    """
    damping = np.exp(-lam * np.asarray(moments, dtype=np.float64))
    significant = np.flatnonzero(damping > _SERIES_CUTOFF)
    l_max = int(significant[-1]) if significant.size else 0

    mu = np.asarray(mu, dtype=np.float64)
    cdf = 0.5 * (1.0 + mu)
    if l_max >= 1:
        # P_{l-1} and P_{l+1} by upward recurrence: eval_legendre per order would
        # re-derive the whole ladder each time.
        p_prev = np.ones_like(mu)  # P_0
        p_curr = mu.copy()  # P_1
        p_next = 1.5 * mu * p_curr - 0.5 * p_prev  # P_2
        for ell in range(1, l_max + 1):
            cdf += 0.5 * damping[ell] * (p_next - p_prev)
            p_prev, p_curr = p_curr, p_next
            n = ell + 1
            p_next = ((2 * n + 1) * mu * p_curr - n * p_prev) / (n + 1)

    np.clip(cdf, 0.0, 1.0, out=cdf)
    np.maximum.accumulate(cdf, out=cdf)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    return cdf


def gs_inverse_cdf(lam: float, moments: np.ndarray, n_nodes: int = 4096) -> np.ndarray:
    r"""Tabulate ``mu(u)`` on ``n_nodes`` uniform nodes of ``u`` in ``[0, 1]``.

    The sampling face of :func:`gs_cumulative`: one uniform variate indexes this
    table and interpolates. The forward CDF is built on a grid **logarithmic in
    ``1 - mu``**, because the distribution spans many decades of angle and a
    uniform ``mu`` grid would put essentially every node in the isotropic tail
    while leaving the peak — where nearly all the probability sits — unresolved.

    Returns
    -------
    ndarray
        Shape ``(n_nodes,)``, ascending, ``mu(0) = -1`` to ``mu(1) = 1``.
    """
    # Log grid in 1 - mu, densest at the forward peak, plus the exact endpoints.
    one_minus_mu = np.geomspace(1.0e-12, 2.0, 1 << 15)
    mu_grid = np.clip(1.0 - one_minus_mu, -1.0, 1.0)[::-1]
    mu_grid[0], mu_grid[-1] = -1.0, 1.0

    cdf = gs_cumulative(lam, moments, mu_grid)
    # Strictly increasing abscissae are required by np.interp; ties at the
    # saturated ends carry no probability, so dropping them is exact.
    keep = np.concatenate(([True], np.diff(cdf) > 0.0))
    u = np.linspace(0.0, 1.0, n_nodes)
    return np.interp(u, cdf[keep], mu_grid[keep])


def mean_square_angle(lam: float, g1: float) -> float:
    r"""GS mean-square deflection ``<theta^2>`` over a path of ``lam`` mean free paths.

    Uses the small-angle identification ``<theta^2> = 2 <1 - cos theta>`` with the
    exact GS first moment ``<cos theta> = exp(-Lambda G_1)``:

    .. math::

        \langle \theta^2 \rangle = 2\bigl(1 - e^{-\Lambda G_1}\bigr).

    In the small-step limit this is ``2 Lambda G_1 = T rho s``, the Fermi-Eyges
    value the Gaussian hinge uses — the anchoring described in the module
    docstring. At long steps it **saturates** toward 2 (isotropy) instead of
    growing without bound, which is the physical behaviour the Gaussian model
    lacks and the reason its step had to be capped.

    Evaluated with ``expm1`` rather than ``1 - exp(-x)``: the small-step limit is
    exactly where the transport spends its time and where that subtraction
    cancels, losing about six digits at the step sizes this engine actually
    takes.
    """
    return float(-2.0 * np.expm1(-lam * g1))
