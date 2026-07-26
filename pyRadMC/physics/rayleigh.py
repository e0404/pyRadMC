r"""Coherent (Rayleigh) scattering: angular sampling.

Pure scalar functions (AGENTS.md section 2.5). The transport loops are
channel-complete (AGENTS.md section 2.10): they sample this channel whenever the data
source reports a nonzero coherent cross-section. Every current source keeps that
column at zero — **no Rayleigh by default** is a statement of data, and with it this
sampler is unreachable; it exists so that enabling the channel is purely a
data change.

Angular model, named here where it is implemented: the Thomson differential
cross-section

.. math::

    \frac{d\sigma}{d\Omega} \propto \frac{1 + \cos^2\theta}{2},

the zero-momentum-transfer limit :math:`F(q, Z) \to Z` of the coherent form-factor
DCS (Hubbell et al., J. Phys. Chem. Ref. Data 4, 471 (1975), doi:10.1063/1.555523;
Salvat et al., PENELOPE-2018, sec. 2.1, doi:10.1787/32da5043-en). **Stated
approximation:** the atomic form factor is data, not physics. The tabulated
source must bring form-factor angular sampling in the same change that turns the
coherent channel on — together with the pair-channel recalibration that docs/decisions.md
already flags for whoever enables Rayleigh.
"""

from __future__ import annotations

from pyRadMC import RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV
from pyRadMC.data.handles import Table1D
from pyRadMC.rng import RNGState, uniform

__all__ = [
    "interp_increasing",
    "sample_coherent_cos_theta_form_factor",
    "sample_rayleigh_cos_theta",
]


def sample_coherent_cos_theta_form_factor(
    cumulative: Table1D, x_grid: Table1D, n: int, energy: float, rng_state: RNGState
) -> float:
    r"""Sample the coherent polar cosine from an atomic form factor, in cm^2 units.

    The coherent differential cross-section is
    :math:`d\sigma/d\Omega \propto (1+\mu^2)\,F^2(x)` with momentum transfer
    :math:`x = C\,E\,\sin(\theta/2)` (:data:`~pyRadMC.RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV`),
    so :math:`x^2 = (CE)^2 (1-\mu)/2`. Sampling factorises (Salvat et al., PENELOPE-2018,
    sec. 2.1): draw :math:`x` from the form-factor part :math:`F^2(x)\,x\,dx` by inverting
    its cumulative, then accept with the Thomson polarisation factor :math:`(1+\mu^2)/2`.

    Parameters
    ----------
    cumulative, x_grid, n
        ``cumulative[i] = \int_0^{x_grid[i]} F^2(x')\,x'\,dx'`` — the form-factor
        cumulative, monotone increasing from zero — and its abscissae ``x_grid`` (inverse
        angstroms, ascending), length ``n``. A *flat* ``F^2`` gives ``cumulative`` ∝
        ``x^2`` and reduces this exactly to the Thomson sampler.
    energy
        Photon energy in MeV; sets the maximum momentum transfer ``x_max = C * energy``.
    rng_state
        Per-history RNG state.
    """
    x_max = RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV * energy
    a_max = interp_increasing(x_max, x_grid, cumulative, n)
    while True:
        target = uniform(rng_state) * a_max
        x = interp_increasing(target, cumulative, x_grid, n)
        ratio = x / x_max
        mu = 1.0 - 2.0 * ratio * ratio
        if 2.0 * uniform(rng_state) <= 1.0 + mu * mu:
            return mu


def interp_increasing(query: float, xs: Table1D, ys: Table1D, n: int) -> float:
    """Linear interpolation of ``ys(xs)`` at ``query`` for ascending ``xs``; flat outside.

    Bisection rather than a hashed index because it is used *inverted* too (querying the
    cumulative to recover ``x``), where the abscissae are not uniformly spaced. Single
    source for both host (NumPy) and, when the coherent channel is ported, the Warp
    kernel; hence the scalar, allocation-free body.
    """
    if query <= xs[0]:
        return float(ys[0])
    if query >= xs[n - 1]:
        return float(ys[n - 1])
    # int(...) declares these as Warp dynamic (loop-mutable) variables; without it the
    # Warp codegen rejects mutating a compile-time constant inside the while loop below.
    lo = int(0)  # noqa: UP018, RUF046
    hi = int(n - 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= query:
            lo = mid
        else:
            hi = mid
    x_lo = float(xs[lo])
    x_hi = float(xs[hi])
    t = (query - x_lo) / (x_hi - x_lo)
    return float(ys[lo]) * (1.0 - t) + float(ys[hi]) * t


def sample_rayleigh_cos_theta(rng_state: RNGState) -> float:
    r"""Sample the coherent polar scattering cosine from the Thomson distribution.

    Rejection from a uniform envelope on [-1, 1]: a candidate :math:`\mu` is
    accepted with probability :math:`(1 + \mu^2)/2` (the pdf's maximum is at the
    endpoints), giving a mean acceptance of 2/3. Exact for the Thomson form; see
    the module docstring for what "Thomson" approximates.

    Parameters
    ----------
    rng_state
        Per-history RNG state; two uniforms per rejection round.
    """
    while True:
        mu = 1.0 - 2.0 * uniform(rng_state)
        if 2.0 * uniform(rng_state) <= 1.0 + mu * mu:
            return mu
