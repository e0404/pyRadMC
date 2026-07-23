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

**Stated approximation (the screened-Rutherford default).** The
single-scattering shape is taken as screened Rutherford with a single Moliere
screening parameter ``eta``, the model underlying Moliere and PRESTA-II theory.
**Validated against the EEDL MF=26/MT=525 tabulated shape** on the quantity the
anchoring actually consumes — the normalized transport-moment ladder
``G_l/G_1`` (absolute moments cancel when ``sigma_el`` is back-derived from the
scattering power): the Moliere ladder tracks EEDL within ~3-7 percent through
``l = 32`` for H, O and Ca at 0.26 and 10 MeV, and the large-angle tail per
unit strength within 1-13 percent, the worst cases sitting where EEDL's own
``mu = 0.999999`` forward truncation makes the reference ambiguous at a
similar level. Re-fitting ``eta`` to the EEDL ladder was measured *not* to
improve on Moliere (the residual is the screened-Rutherford family, not the
parameter), and the residual bounds the eta-related dose effect at
~0.03 percent of dose. Pinned in ``tests/validation/test_gs_eta_ladder.py``.
Moliere, Z. Naturforsch. 3a, 78 (1948).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from pyRadMC.data.tables import CrossSectionTables

__all__ = [
    "GSGridTables",
    "build_gs_grid",
    "cached_moments",
    "first_transport_moment",
    "gs_cumulative",
    "gs_scaled_deflection_table",
    "gs_window_from_tables",
    "mean_square_angle",
    "moliere_screening",
    "screened_rutherford_cos_theta",
    "screened_rutherford_moments",
]

_log = logging.getLogger(__name__)

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


_node_cache: dict[tuple[float, float, int, int, int], np.ndarray] = {}
"""Process-wide memo of built node tables, keyed on the node parameters and the
build sample count. A node's value is a pure function of that key (fixed build
seed), so sharing across sources is exact — without this, every
``CrossSectionSource`` instance in a test session rebuilt identical tables into
its own lazy cache, multiplying the ~0.1 s per node across dozens of sources
once Goudsmit-Saunderson became the default. Deliberately unlocked: the
reference loop is single-threaded, and a concurrent duplicate build writes the
identical deterministic array. Cached arrays are frozen read-only, since one
array is now shared by every consumer."""


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
        from the widest deflection to the most forward. **Read-only and shared**
        (see :data:`_node_cache`); a consumer that needs to mutate must copy.
    """
    key = (eta, theta2, n_u, n_mu, _RAW_SAMPLES)
    cached = _node_cache.get(key)
    if cached is not None:
        return cached

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
        normalized: np.ndarray = np.zeros(n_u)
    else:
        normalized = one_minus_mu / mean
    normalized.flags.writeable = False
    _node_cache[key] = normalized
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


# -- the eager node grid (device backends) ----------------------------------------


@dataclass(frozen=True)
class GSGridTables:
    """The full rectangle of scaled-deflection tables, precomputed for upload.

    The reference backend memoizes tables lazily per ``(eta, <theta^2>)`` node
    key; a device backend precomputes the whole reachable window before the
    first launch instead — a mid-transport host build was measured at 86 s, and
    a lazily-growing dict is not shareable across ``devices=[...]`` shard
    threads. Every entry is built by :func:`gs_scaled_deflection_table` at the
    identical node parameters and fixed build seed the lazy cache uses, so the
    two backends sample bit-identical tables (float64, before any device cast).

    Attributes
    ----------
    values
        ``(n_eta, n_theta2, n_u)`` float64; ``values[i, j]`` is the table at
        integer node indices ``(ix0 + i, iy0 + j)``, i.e. at
        ``eta = exp((ix0 + i) / bins_per_log)`` and
        ``<theta^2> = exp((iy0 + j) / bins_per_log)``. The u-bin axis is
        fastest so one table is contiguous.
    ix0, iy0
        Integer node index of the first eta column and first theta2 row — the
        offsets a sampler subtracts from ``floor(log(key) * bins_per_log)``.
    n_u
        Equal-probability bins per table.
    bins_per_log
        Grid nodes per natural log in both axes.
    """

    values: npt.NDArray[np.float64]
    ix0: int
    iy0: int
    n_u: int
    bins_per_log: float


_grid_cache: dict[tuple[int, int, int, int, int, float, int], GSGridTables] = {}
_grid_cache_lock = threading.Lock()


_GS_GRID_CACHE_VERSION: int = 1
"""Version of the table construction, for the persistent grid cache key.

**Bump this with any commit that changes what a node's table contains**: the
fixed build seed or sample count, the series/composition split
(:data:`_SERIES_MIN_LAMBDA`), the moment quadrature or Legendre order, the
bin-average condensation, or the screened-Rutherford single-scattering model
itself. Nothing else invalidates the cache — node values are pure functions of
the node coordinates and this algorithm, so materials, geometries, cutoffs,
sources and even the screening/scattering-power *data* only move the requested
window, which the cache handles by extension, not rebuild.
"""

_GS_GRID_CACHE_DIR: Path = Path.home() / ".cache" / "pyRadMC" / "gs-grid"
"""Default home of the persistent grid, beside the EPICS library cache."""


def _grid_cache_key() -> tuple[float | int | str, ...]:
    """Return the construction identity the persistent cache is keyed on.

    Includes numpy's feature version: ``Generator`` distribution streams
    (``poisson``, ``permutation``) are not guaranteed stable across numpy
    feature releases, and a stale grid would silently drift bit-wise from what
    the reference backend builds in-process — one clean rebuild per numpy
    upgrade is the safe trade.
    """
    from pyRadMC.data.interface import _GS_BINS_PER_LOG, _GS_TABLE_NODES, _GS_THETA2_MIN

    numpy_feature = ".".join(np.__version__.split(".")[:2])
    return (
        _GS_GRID_CACHE_VERSION,
        _GS_BINS_PER_LOG,
        _GS_TABLE_NODES,
        _GS_THETA2_MIN,
        _RAW_SAMPLES,
        _MC_TABLE_SEED,
        _SERIES_MIN_LAMBDA,
        _MOMENT_L_MAX,
        _QUADRATURE_POINTS,
        _SERIES_CUTOFF,
        numpy_feature,
    )


def _grid_cache_file(cache_dir: Path) -> Path:
    """One file per construction identity; distinct identities never collide."""
    digest = hashlib.sha256(repr(_grid_cache_key()).encode()).hexdigest()[:16]
    return cache_dir / f"gs-grid-{digest}.npz"


def _load_grid_cache(path: Path) -> GSGridTables | None:
    """Read a stored grid, or None on absence, corruption, or key mismatch."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["key"]) != repr(_grid_cache_key()):
                return None
            from pyRadMC.data.interface import _GS_BINS_PER_LOG, _GS_TABLE_NODES

            return GSGridTables(
                values=np.ascontiguousarray(stored["values"]),
                ix0=int(stored["ix0"]),
                iy0=int(stored["iy0"]),
                n_u=_GS_TABLE_NODES,
                bins_per_log=_GS_BINS_PER_LOG,
            )
    except Exception:  # a damaged cache entry must never take the run down
        _log.warning("unreadable GS grid cache at %s; rebuilding", path)
        return None


def _save_grid_cache(path: Path, grid: GSGridTables) -> None:
    """Write atomically (temp + replace), the EPICS-cache convention.

    An interrupted save never leaves a truncated file masquerading as a cache
    entry; a concurrent process at worst rebuilds the same deterministic values
    and the last replace wins.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".partial-{os.getpid()}")
    np.savez(
        temporary,
        values=grid.values,
        ix0=grid.ix0,
        iy0=grid.iy0,
        key=repr(_grid_cache_key()),
    )
    # savez appends .npz to the handed name.
    temporary.with_suffix(temporary.suffix + ".npz").replace(path)


_PARALLEL_BUILD_THRESHOLD: int = 256
"""Window size (tables) below which the build stays serial: thread-pool spin-up
is pure overhead on the small windows tests construct."""

_BUILD_WORKER_CAP: int = 16
"""Upper bound on build threads. The table construction releases the GIL in its
large-array NumPy phases (measured ~3x at 4 workers) but not everywhere, so
unbounded oversubscription past the physical cores buys nothing."""


def build_gs_grid(
    log_eta_min: float,
    log_eta_max: float,
    theta2_max: float,
    *,
    theta2_min: float | None = None,
    n_u: int | None = None,
    bins_per_log: float | None = None,
    max_workers: int | None = None,
    cache_dir: Path | str | None = None,
) -> GSGridTables:
    """Precompute every deflection table inside a reachable-key window.

    The window brackets: any query with ``log_eta`` in
    ``[log_eta_min, log_eta_max]`` and ``<theta^2>`` in
    ``[theta2_min, theta2_max]`` has all four bilinear bracketing nodes in the
    grid, because both axes extend one node past the floor of their upper edge.
    Keys outside (which a correctly derived window never produces — see
    :func:`gs_window_from_tables`) are the caller's to clamp; clamping is
    anchor-safe since table resolution never moves the first moment.

    The defaults for ``theta2_min``, ``n_u`` and ``bins_per_log`` are the
    reference sampler's grid constants
    (:mod:`pyRadMC.data.interface`), so the eager grid and the lazy cache key
    the same nodes by construction; they are overridable only so tests can
    build small windows cheaply.

    **Memoized process-wide** on the integer node window (plus the build
    statistics), because every node value is a pure function of its node
    parameters and the fixed build seed — two engines asking for the same
    window get one immutable build, whatever thread asks first. The frozen
    result must never be mutated by callers.

    **Built column-parallel** (one eta per task) above a small-window
    threshold: the per-node work is dominated by large-array NumPy phases that
    release the GIL, and the fixed per-node seed makes the result independent
    of scheduling — worker count is a wall-clock knob, never a value change
    (test-pinned bit-equal against the serial build). ``max_workers`` exists
    for that pin and for constrained environments; ``None`` auto-sizes.

    **Persisted to disk** (default-constants builds only; ``cache_dir`` is for
    tests, ``None`` resolving beside the EPICS cache): node values are pure
    functions of the node coordinates and the construction algorithm, so the
    stored grid is keyed on the construction identity alone
    (:data:`_GS_GRID_CACHE_VERSION` states the — deliberately short —
    invalidation rule) and **extended in place** when a request needs a wider
    window: new phantoms, materials, cutoffs, sources, and even changed
    screening or scattering-power data reuse every stored node bit-for-bit and
    build only the genuinely new ones. A fresh process therefore loads the
    grid in tens of milliseconds instead of paying the ~30 s cold build once
    per run. Builds with overridden grid constants are test instruments and
    stay in-memory.
    """
    from pyRadMC.data.interface import _GS_BINS_PER_LOG, _GS_TABLE_NODES, _GS_THETA2_MIN

    bins = _GS_BINS_PER_LOG if bins_per_log is None else bins_per_log
    nodes = _GS_TABLE_NODES if n_u is None else n_u
    floor = _GS_THETA2_MIN if theta2_min is None else theta2_min
    if log_eta_max < log_eta_min:
        raise ValueError(f"empty eta window: [{log_eta_min}, {log_eta_max}]")
    if not 0.0 < floor <= theta2_max:
        raise ValueError(f"invalid theta2 window: [{floor}, {theta2_max}]")

    ix0 = math.floor(log_eta_min * bins)
    ix_top = math.floor(log_eta_max * bins) + 1
    iy0 = math.floor(math.log(floor) * bins)
    iy_top = math.floor(math.log(theta2_max) * bins) + 1

    use_disk = theta2_min is None and n_u is None and bins_per_log is None
    key = (ix0, ix_top, iy0, iy_top, nodes, bins, _RAW_SAMPLES)
    with _grid_cache_lock:
        cached = _grid_cache.get(key)
        if cached is not None:
            return cached

        stored = None
        cache_file = None
        if use_disk:
            directory = _GS_GRID_CACHE_DIR if cache_dir is None else Path(cache_dir)
            cache_file = _grid_cache_file(directory)
            started = time.perf_counter()
            stored = _load_grid_cache(cache_file)
            if stored is not None:
                s_ix_top = stored.ix0 + stored.values.shape[0] - 1
                s_iy_top = stored.iy0 + stored.values.shape[1] - 1
                covered = (
                    stored.ix0 <= ix0
                    and ix_top <= s_ix_top
                    and stored.iy0 <= iy0
                    and iy_top <= s_iy_top
                )
                if covered:
                    _log.info(
                        "loaded the Goudsmit-Saunderson grid from %s in %.0f ms",
                        cache_file,
                        1e3 * (time.perf_counter() - started),
                    )
                    _grid_cache[key] = stored
                    return stored
                # Partial coverage: extend to the union, so every stored node is
                # reused bit-for-bit and the file only ever grows.
                ix0 = min(ix0, stored.ix0)
                iy0 = min(iy0, stored.iy0)
                ix_top = max(ix_top, s_ix_top)
                iy_top = max(iy_top, s_iy_top)

        n_columns = ix_top - ix0 + 1
        n_rows = iy_top - iy0 + 1

        def build_column(column: int) -> npt.NDArray[np.float64]:
            i = ix0 + column
            eta = math.exp(i / bins)
            result = np.empty((n_rows, nodes))
            for row in range(n_rows):
                j = iy0 + row
                if (
                    stored is not None
                    and 0 <= i - stored.ix0 < stored.values.shape[0]
                    and 0 <= j - stored.iy0 < stored.values.shape[1]
                ):
                    result[row] = stored.values[i - stored.ix0, j - stored.iy0]
                else:
                    result[row] = gs_scaled_deflection_table(eta, math.exp(j / bins), n_u=nodes)
            return result

        new_cells = n_columns * n_rows
        if stored is not None:
            new_cells -= stored.values.shape[0] * stored.values.shape[1]
        workers = max_workers
        if workers is None:
            small = new_cells < _PARALLEL_BUILD_THRESHOLD
            workers = 1 if small else min(n_columns, os.cpu_count() or 1, _BUILD_WORKER_CAP)

        started = time.perf_counter()
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                columns = list(pool.map(build_column, range(n_columns)))
        else:
            columns = [build_column(column) for column in range(n_columns)]
        values = np.stack(columns)
        _log.info(
            "built the Goudsmit-Saunderson grid: %d x %d tables of %d bins "
            "(%d new, %d reused) in %.1f s (%d workers)",
            n_columns,
            n_rows,
            nodes,
            new_cells,
            n_columns * n_rows - new_cells,
            time.perf_counter() - started,
            workers,
        )
        grid = GSGridTables(values=values, ix0=ix0, iy0=iy0, n_u=nodes, bins_per_log=bins)
        if cache_file is not None:
            _save_grid_cache(cache_file, grid)
        _grid_cache[key] = grid
        return grid


def gs_window_from_tables(
    tables: CrossSectionTables, materials: Sequence[int]
) -> tuple[float, float, float]:
    r"""Return the ``(log_eta_min, log_eta_max, theta2_max)`` window transport can reach.

    Derived from the flattened tables themselves, so the bounds are exact for
    the kernel that consumes them rather than estimated:

    - **eta**: the kernel reads ``eta`` through the log-linear lookup of
      ``tables.log_eta``, which returns a convex combination of row nodes and
      clamps flat outside the grid — so the row minimum and maximum over the
      transported materials *are* the reachable extremes, exactly.
    - **theta2**: the hinge deflection is ``T(E_hinge) rho s``. The range-out
      clamp bounds every substep by ``s <= restricted_range(E_start) / rho``
      independently of ``step_energy_fraction``, and the scattering power is a
      lookup bounded by its row maximum, so the product of per-material row
      maxima is a hard ceiling. Generous by construction (the two maxima sit at
      opposite ends of the energy grid); the excess buys a few extra rows in
      the long-step regime, where the series construction is cheapest.

    Parameters
    ----------
    tables
        Flattened tables carrying ``log_eta``, ``scattering_power`` and
        ``restricted_range``.
    materials
        Material rows actually present in the transport grid's voxel map —
        electrons only ever step inside grid voxels, so absent registry
        materials cannot key a lookup.
    """
    rows = np.asarray(sorted(set(int(m) for m in materials)), dtype=int)
    if rows.size == 0:
        raise ValueError("no materials given; the window would be empty")
    log_eta = tables.log_eta[rows]
    ceiling = float(
        np.max(
            tables.scattering_power[rows].max(axis=1) * tables.restricted_range[rows].max(axis=1)
        )
    )
    return float(log_eta.min()), float(log_eta.max()), ceiling
