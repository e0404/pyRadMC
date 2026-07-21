r"""Goudsmit-Saunderson multiple-scattering deflection: angular sampling.

Pure scalar functions (AGENTS.md section 2.5). The distribution this samples is
the exact multiple-scattering law for an *arbitrary* path length, in contrast to
the small-angle Gaussian of :mod:`pyRadMC.physics.msc`, whose validity is what
forces :data:`~pyRadMC.transport.electron.STEP_HINGE_THETA2_MAX` to cap the
substep. The construction of the table lives in
:mod:`pyRadMC.data.goudsmit_saunderson`; the data that selects it is reached
through :meth:`~pyRadMC.data.interface.CrossSectionSource.sample_gs_cos_theta`,
the same split the coherent form-factor sampler uses (AGENTS.md section 2.6).

**The table is in a scaled deflection** ``w = (1 - mu) / <1 - mu>``, held as bin
averages over equal-probability bins and normalized so that ``<w> = 1``. That is
not a storage convenience, it is what keeps the model anchored: the table is
looked up at a *binned* screening parameter and step length, and rescaling by the
caller's exact ``<1 - cos theta> = 1 - e^{-<theta^2>/2}`` makes the first moment
come out exact regardless of how coarse the binning is. ``<1 - cos>`` is the
anchor rather than ``<theta^2>`` because it is the moment Goudsmit-Saunderson
theory pins exactly (``<cos> = e^{-Lambda G_1}``) and the one the mass scattering
power encodes; ``<theta^2>`` is tail-dominated for a heavy-tailed law. Table
resolution therefore perturbs only the *shape* of the deflection — never its
strength — so a transport-level comparison against the Gaussian hinge measures
the large-angle tail that Goudsmit-Saunderson adds, and not a residual
mis-scaling of the table.

Goudsmit & Saunderson, Phys. Rev. 57, 24 (1940), doi:10.1103/PhysRev.57.24.
"""

from __future__ import annotations

import math

from pyRadMC.data.handles import Table1D
from pyRadMC.rng import RNGState, uniform

__all__ = ["sample_gs_cos_theta_bilinear"]


def sample_gs_cos_theta_bilinear(
    row_00: Table1D,
    row_10: Table1D,
    row_01: Table1D,
    row_11: Table1D,
    tx: float,
    ty: float,
    n_u: int,
    mean_square_angle: float,
    rng_state: RNGState,
) -> float:
    r"""Sample a deflection cosine, interpolating between four neighbouring tables.

    The table is irreducibly two-dimensional in ``(eta, <theta^2>)`` — measured,
    the shape varies by 0.11 to 1.04 in probability-weighted relative terms
    across the ranges real transport spans, and it does not collapse onto a
    single parameter. Resolving that by *refining* the grid is quadratic in cost:
    the resolution the shape test demands under nearest-bin lookup runs to tens
    of megabytes, against the kilobyte-scale L1 residency that makes this
    engine's other tables fast.

    Interpolating between coarse tables attacks the same error for four loads and
    a blend, and is what texture hardware would do natively if the tables later
    move to the GPU. Only the *sampled node* is blended, never a whole table, so
    the cost is per-sample constant.

    Safe for the anchoring by construction: each row is normalized to unit mean,
    and the bilinear weights sum to one, so the blend also has unit mean. The
    deflection's strength therefore stays exactly the caller's
    ``<1 - cos theta>`` regardless of where between the grid nodes this lands —
    interpolation moves the distribution's *shape* only.

    Parameters
    ----------
    row_00, row_10, row_01, row_11
        Scaled-deflection tables at the four bracketing grid corners, ordered
        ``(eta_lo, theta2_lo)``, ``(eta_hi, theta2_lo)``, ``(eta_lo, theta2_hi)``,
        ``(eta_hi, theta2_hi)``.
    tx, ty
        Fractional position between the bracketing nodes in ``eta`` and
        ``<theta^2>`` respectively, each in ``[0, 1]``.
    n_u
        Equal-probability bins per row.
    mean_square_angle
        The substep's ``<theta^2> = T rho s`` in rad^2.
    rng_state
        Per-history RNG state; one uniform, as the Gaussian hinge takes.
    """
    if mean_square_angle <= 0.0:
        return 1.0
    u = uniform(rng_state) * float(n_u)
    index = int(u)
    if index > n_u - 1:
        index = n_u - 1

    w_lo = float(row_00[index]) * (1.0 - tx) + float(row_10[index]) * tx
    w_hi = float(row_01[index]) * (1.0 - tx) + float(row_11[index]) * tx
    w = w_lo * (1.0 - ty) + w_hi * ty

    one_minus_cos = -math.expm1(-0.5 * mean_square_angle)
    return max(-1.0, 1.0 - w * one_minus_cos)
