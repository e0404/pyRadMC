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
    "cached_moments",
    "first_transport_moment",
    "gs_cumulative",
    "gs_scaled_deflection_table",
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

    **The unscattered delta is removed before the series is summed.** A fraction
    ``e^{-Lambda}`` of electrons cross the step without a single elastic
    collision, so the distribution carries a delta at ``mu = 1`` of exactly that
    weight. A Legendre series cannot represent a delta: truncating one that
    contains it does not converge, it rings, and the reconstructed density is
    wrong by an amount that grows as the step shortens — measured at 11 times the
    intended second moment at ``Lambda`` of order one, which real
    boundary-truncated substeps reach. Since ``delta(mu - 1)`` has coefficients
    ``(2l+1)/2``, subtracting it leaves smooth coefficients
    ``e^{-Lambda G_l} - e^{-Lambda}`` that genuinely decay (``G_l -> 1``, so the
    difference vanishes). The delta is added back as the jump to 1 at ``mu = 1``,
    where inversion turns it into the flat forward region of the sampling table —
    so it costs no branch and no extra variate at sampling time.

    Returns a CDF clamped to ``[0, 1]`` and forced non-decreasing: a truncated
    Legendre series rings slightly at the forward peak, and an inverter handed a
    non-monotone CDF produces a silently wrong angular distribution rather than
    an error. The clamp is a numerical guard on a converged series, not a
    physics approximation — the residual it removes is at the truncation level
    (:data:`_SERIES_CUTOFF`), which the monotonicity test pins.
    """
    unscattered = math.exp(-lam)
    # Truncate on the *smooth* coefficient, which is the one being summed.
    coefficients = np.exp(-lam * np.asarray(moments, dtype=np.float64)) - unscattered
    significant = np.flatnonzero(np.abs(coefficients) > _SERIES_CUTOFF)
    l_max = int(significant[-1]) if significant.size else 0

    mu = np.asarray(mu, dtype=np.float64)
    cdf = (1.0 - unscattered) * 0.5 * (1.0 + mu)
    if l_max >= 1:
        # P_{l-1} and P_{l+1} by upward recurrence: eval_legendre per order would
        # re-derive the whole ladder each time.
        p_prev = np.ones_like(mu)  # P_0
        p_curr = mu.copy()  # P_1
        p_next = 1.5 * mu * p_curr - 0.5 * p_prev  # P_2
        for ell in range(1, l_max + 1):
            cdf += 0.5 * coefficients[ell] * (p_next - p_prev)
            p_prev, p_curr = p_curr, p_next
            n = ell + 1
            p_next = ((2 * n + 1) * mu * p_curr - n * p_prev) / (n + 1)

    np.clip(cdf, 0.0, 1.0, out=cdf)
    np.maximum.accumulate(cdf, out=cdf)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    return cdf


_SERIES_MIN_LAMBDA: float = 300.0
"""Elastic path count below which the table is built by composition, not the series.

The Goudsmit-Saunderson *moments* are exact at every ``Lambda``, but reconstructing
the *density* from them is a Legendre expansion, and that expansion cannot resolve
the near-forward structure of a lightly-scattered distribution. Measured departure
of the reconstructed ``2(1 - <cos>)`` from its exact value ``2(1 - e^{-Lambda G_1})``:
1.005 at ``Lambda = 300``, **1.80 at 20, and 14.6 at 1.4** — and raising the
Legendre order from 1024 to 4096 does not move it, so this is a representation
limit, not truncation. Real substeps reach that regime routinely: measured over
100842 substeps of a slab transport, 20 percent had ``Lambda < 50`` and 5 percent
``Lambda < 20``, because a substep truncated at a voxel face is far shorter than
the energy-limited one.
"""

_MC_TABLE_SAMPLES: int = 1 << 18
"""Composed deflections per Monte-Carlo-built table."""

_MC_TABLE_SEED: int = 20260720
"""Fixed seed for Monte-Carlo table construction.

The tables are *data*, not samples: a given (eta, Lambda) bin must yield the same
table on every run, or the transport stops being reproducible for a given seed
(AGENTS.md 2.3). Build-time statistical noise is therefore a fixed, testable
property of the table rather than a source of run-to-run variation."""


def _compose(cos_a: np.ndarray, cos_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r"""Compose two independent deflections into a net polar cosine.

    Only the *polar* angle is tracked. By azimuthal symmetry the second
    deflection's azimuth relative to the first is uniform, so the net angle
    follows the spherical law of cosines

    .. math::

        \cos\theta = \cos\theta_a \cos\theta_b
                   + \sin\theta_a \sin\theta_b \cos\varphi ,

    which is exact and needs no direction vector.
    """
    sin_a = np.sqrt(np.maximum(0.0, 1.0 - cos_a * cos_a))
    sin_b = np.sqrt(np.maximum(0.0, 1.0 - cos_b * cos_b))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=cos_a.size)
    composed: np.ndarray = np.clip(cos_a * cos_b + sin_a * sin_b * np.cos(phi), -1.0, 1.0)
    return composed


def _mc_gs_cosines(eta: float, lam: float, n: int, rng: np.random.Generator) -> np.ndarray:
    r"""Sample net deflection cosines by explicit composition of single scatters.

    This is the *definition* of the Goudsmit-Saunderson distribution — a Poisson
    number of elastic collisions, composed — so it is exact wherever the
    Legendre reconstruction is not, and needs no series to converge.

    Composition is done by **repeated self-convolution** rather than one scatter
    at a time. A Poisson process splits, ``Poisson(2 Lambda) = Poisson(Lambda) +
    Poisson(Lambda)``, so a sample for ``Lambda`` composed with an independent
    sample for ``Lambda`` is a sample for ``2 Lambda``. Starting from
    ``Lambda_0 <= 1`` and doubling costs ``log2(Lambda)`` passes instead of
    ``Lambda`` — about ten rather than eight hundred at the step lengths this
    engine takes. The independent partner is a random permutation of the same
    sample: a random matching, exact in the marginals, and the residual pairing
    correlation is checked by validating this construction against the series in
    the regime where the series is trustworthy.
    """
    doublings = max(0, math.ceil(math.log2(lam)) if lam > 1.0 else 0)
    lam_0 = lam / (2.0**doublings)

    # Base: a genuinely Poisson number of scatters, mean <= 1.
    counts = rng.poisson(lam_0, size=n)
    cos_acc = np.ones(n)
    for step in range(int(counts.max()) if counts.size else 0):
        active = counts > step
        k = int(active.sum())
        if k == 0:
            break
        # Vectorized over the active lanes; the scalar form is the same algebra.
        xi = rng.random(k)
        inverse = xi / (2.0 * eta * (1.0 + eta)) + 1.0 / (2.0 + 2.0 * eta)
        single = 1.0 + 2.0 * eta - 1.0 / inverse
        cos_acc[active] = _compose(cos_acc[active], single, rng)

    for _ in range(doublings):
        cos_acc = _compose(cos_acc, rng.permutation(cos_acc), rng)
    return cos_acc


_MOMENT_L_MAX: int = 1024
"""Legendre orders computed per screening parameter.

Generous rather than adaptive because the moments are cached per ``eta`` while
the series truncation is chosen per *step* from the damping: the shortest steps
the engine takes need ~200 orders (measured), and the margin costs one cached
array per screening bin.
"""

_moment_cache: dict[float, np.ndarray] = {}


def cached_moments(eta: float) -> np.ndarray:
    """Legendre moments for ``eta``, memoized across table builds.

    The moments depend only on the screening parameter, while the tables that
    consume them are keyed on screening *and* step length; caching here keeps
    the expensive quadrature from being repeated across every step-length bin of
    the same material and energy.
    """
    moments = _moment_cache.get(eta)
    if moments is None:
        moments = screened_rutherford_moments(eta, l_max=_MOMENT_L_MAX)
        _moment_cache[eta] = moments
    return moments


def gs_scaled_deflection_table(
    eta: float, theta2: float, n_u: int = 512, n_mu: int = 4096
) -> np.ndarray:
    r"""Tabulate the scaled deflection ``w = (1 - mu) / <1 - mu>`` in ``n_u`` bins.

    The sampling face of the Goudsmit-Saunderson distribution, consumed by
    :func:`pyRadMC.physics.gs.sample_gs_cos_theta_bilinear`. The step length enters
    through the anchoring ``Lambda G_1 = <theta^2>/2``, so the caller specifies
    the step by the same Fermi-Eyges mean-square angle the Gaussian hinge takes.

    **Nodes are bin averages, not quantiles.** Entry ``i`` is the conditional mean
    of ``1 - mu`` over the ``i``-th equal-probability bin of the distribution, and
    the sampler draws it piecewise-constant. This is what makes the table's
    moments exact rather than approximate. Storing *quantiles* and interpolating
    between them cannot represent a heavy tail: the outermost node is then an
    extreme order statistic — around the ``1/N``-th quantile of the build sample —
    while carrying ``1/n_u`` of the sampling weight, which over-weighted wide
    angles by a factor of roughly 500 and inflated ``<1 - cos>`` by up to 35x at
    small ``Lambda``. Bin averaging gives every bin its correct probability *and*
    its correct mean contribution, at the cost of the angular spread within a
    bin — negligible where bins are narrow, and in the outermost bin a
    moment-preserving stand-in for a spread of rare wide angles.

    **Normalized to ``<w> = 1``**, so the caller rescales by its own exact
    ``<1 - cos theta> = 1 - e^{-<theta^2>/2}``. The first moment is then exact
    however coarsely ``eta`` and ``<theta^2>`` were binned, and only the shape
    carries interpolation error. ``<1 - cos>`` is the anchor rather than
    ``<theta^2>`` because it is the moment Goudsmit-Saunderson theory pins
    exactly (``<cos> = e^{-Lambda G_1}``) and the one
    :meth:`~pyRadMC.data.interface.CrossSectionSource.scattering_power` encodes;
    ``<theta^2>`` is tail-dominated for a heavy-tailed law, so anchoring on it
    amplifies precisely the least well represented part of the distribution.

    Returns
    -------
    ndarray
        Shape ``(n_u,)``, non-negative, mean exactly 1, ascending ``u`` ordered
        from the widest deflection to the most forward.
    """
    raw = _gs_raw_cosines(eta, theta2, n_mu)
    # Equal-probability bins; entry i is the conditional mean of 1 - mu over bin i.
    one_minus_mu = (1.0 - raw).reshape(n_u, raw.size // n_u).mean(axis=1)
    mean = float(one_minus_mu.mean())
    if mean <= 0.0:
        # Degenerate bin: Lambda so small that no build sample scattered at all
        # (near the floored key, Lambda ~ 1e-9, the all-zero Poisson outcome is
        # a near certainty). The correct limit is "no deflection": w = 0 makes
        # the sampler return mu = 1 - 0 * anchor = 1 exactly. Dividing by the
        # zero mean instead yields a NaN row, and max(-1, nan) is -1 under both
        # Python and CUDA fmaxf semantics — silent certain backscatter, not a
        # crash. The bias of the forward table against the true distribution is
        # the anchor itself, 1 - e^{-theta2/2} <= ~1e-12 for any theta2 that
        # can key this bin. Test-pinned in tests/unit/test_gs_sampler.py.
        return np.zeros(n_u)
    normalized: np.ndarray = one_minus_mu / mean
    return normalized


_RAW_SAMPLES: int = 1 << 19
"""Equally weighted deflection cosines a table is condensed from.

A multiple of any ``n_u`` the sampler uses, so the bin averages in
:func:`gs_scaled_deflection_table` are over exactly equal-probability groups.
"""


def _gs_raw_cosines(eta: float, theta2: float, n_mu: int) -> np.ndarray:
    """Equally weighted deflection cosines for one ``(eta, <theta^2>)`` pair.

    Ascending, one per equal-probability slot, so the caller can condense them
    into bin averages by a plain reshape. Two constructions feed it, split at
    :data:`_SERIES_MIN_LAMBDA`: explicit composition where the Legendre density
    reconstruction cannot represent a lightly scattered distribution, and the
    series elsewhere, where it is accurate and far cheaper.
    """
    g1 = first_transport_moment(eta)
    lam = 0.5 * theta2 / g1

    if lam < _SERIES_MIN_LAMBDA:
        rng = np.random.default_rng(_MC_TABLE_SEED)
        return np.sort(_mc_gs_cosines(eta, lam, _RAW_SAMPLES, rng))

    # Log grid in 1 - mu: the deflection spans decades and a uniform mu grid
    # would leave the peak — where the probability is — unresolved.
    one_minus_mu = np.geomspace(1.0e-12, 2.0, n_mu)
    mu_grid = np.clip(1.0 - one_minus_mu, -1.0, 1.0)[::-1]
    mu_grid[0], mu_grid[-1] = -1.0, 1.0
    cdf = gs_cumulative(lam, cached_moments(eta), mu_grid)
    # Midpoint sampling of the inverse CDF gives equally weighted representatives
    # without an extreme order statistic at either end.
    u = (np.arange(_RAW_SAMPLES) + 0.5) / _RAW_SAMPLES
    keep_lo = int(np.flatnonzero(cdf <= 0.0)[-1]) if np.any(cdf <= 0.0) else 0
    keep_hi = int(np.flatnonzero(cdf >= 1.0)[0]) if np.any(cdf >= 1.0) else cdf.size - 1
    if keep_hi <= keep_lo:
        keep_lo, keep_hi = 0, cdf.size - 1
    sub_cdf, sub_mu = cdf[keep_lo : keep_hi + 1], mu_grid[keep_lo : keep_hi + 1]
    keep = np.concatenate(([True], np.diff(sub_cdf) > 0.0))
    representatives: np.ndarray = np.interp(u, sub_cdf[keep], sub_mu[keep])
    return representatives


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
