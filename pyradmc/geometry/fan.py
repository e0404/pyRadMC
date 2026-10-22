"""Single-source scalar samplers: the divergent spectral fan and a discrete CDF draw.

Pure, scalar, allocation-free functions of scalars, array handles and the RNG state
(AGENTS.md 2.5): they run under NumPy on the host and the Warp physics loader
compiles them to ``@wp.func`` for the in-kernel generators, so every generator of
the spectral fan — open field, transmission-masked, beamlet fan, segmented fan and
its plan view — draws its primaries through one definition instead of a copy per
kernel.

- :func:`fan_primary` is the in-kernel form of
  :meth:`pyradmc.geometry.source._DivergentFan.emit_through` (with
  :meth:`~pyradmc.geometry.spectrum.Spectrum.sample_energy`): four uniforms in the
  host order — spectrum bin, position within the bin, then the in-aperture ``u``
  and ``v`` offsets — and the unit ray from the focal point through the aperture.
- :func:`select_by_cdf` draws a discrete index from cumulative shares: the first
  ``i`` with ``cdf[i] > u`` (``searchsorted`` ``side="right"``), so an index of
  zero probability is never returned, even for ``u`` on a share boundary.

Integer loop variables are declared with ``int(0)`` and floats with ``float(0.0)``
because Warp needs the cast to declare a variable mutated inside a dynamic loop.
"""

from __future__ import annotations

import math

from pyradmc.data.handles import Table1D
from pyradmc.rng import RNGState, uniform

__all__ = ["cdf_lower_bound", "fan_primary", "select_by_cdf"]


def cdf_lower_bound(cdf: Table1D, n: int, u: float) -> int:
    """Return the first index ``i < n`` with ``cdf[i] >= u``, or ``n`` if there is none.

    Binary search; ``numpy.searchsorted(cdf[:n], u)`` (``side="left"``), and the
    index ``wp.lower_bound`` returns.
    """
    lo = int(0)  # noqa: UP018, RUF046 - Warp dynamic loop var
    hi = int(n)  # a fresh loop-mutated var for Warp, not the argument
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    return lo


def select_by_cdf(cdf: Table1D, n: int, u: float) -> int:
    """Draw an index from cumulative shares: the first ``i`` with ``cdf[i] > u``.

    ``cdf`` holds ``n`` non-decreasing cumulative shares ending at 1 and ``u`` is
    uniform in ``[0, 1)``, so ``P(i) = cdf[i] - cdf[i - 1]`` and a zero-share index
    (``cdf[i] == cdf[i - 1]``) is never returned. Clamped to ``n - 1`` against a
    rounded final share below ``u``.
    """
    lo = int(0)  # noqa: UP018, RUF046 - Warp dynamic loop var
    hi = int(n)  # a fresh loop-mutated var for Warp, not the argument
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] <= u:
            lo = mid + 1
        else:
            hi = mid
    return min(lo, n - 1)


def fan_primary(
    rng_state: RNGState,
    edges: Table1D,
    cdf: Table1D,
    n_bins: int,
    fx: float,
    fy: float,
    fz: float,
    ax: float,
    ay: float,
    az: float,
    uxa: float,
    uya: float,
    uza: float,
    vxa: float,
    vya: float,
    vza: float,
    width_u: float,
    width_v: float,
) -> tuple[float, float, float, float]:
    """Sample one fan photon: ``(energy, ux, uy, uz)``; consumes exactly four uniforms.

    The energy by CDF inversion of the histogram ``edges``/``cdf`` (``n_bins``
    bins), the direction from the focal point ``f`` toward the point
    ``a + du * u_axis + dv * v_axis`` of the ``width_u x width_v`` aperture centred
    at ``a``, with ``du``, ``dv`` uniform. The primary starts at ``f``; uniform
    aperture sampling from a point source carries the 1/r^2 divergence.
    """
    u_bin = uniform(rng_state)
    u_within = uniform(rng_state)
    du = (uniform(rng_state) - 0.5) * width_u
    dv = (uniform(rng_state) - 0.5) * width_v
    k = min(cdf_lower_bound(cdf, n_bins, u_bin), n_bins - 1)
    lo = edges[k]
    energy = lo + (edges[k + 1] - lo) * u_within
    dx = ax + du * uxa + dv * vxa - fx
    dy = ay + du * uya + dv * vya - fy
    dz = az + du * uza + dv * vza - fz
    inv_norm = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
    return energy, dx * inv_norm, dy * inv_norm, dz * inv_norm
