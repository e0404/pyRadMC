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
- :func:`spectrum_energy` is the spectrum inversion alone, for a generator that draws
  its uniforms itself (the fan edge-gradient rays, :mod:`pyradmc.geometry.edge_rays`).
- :func:`select_by_cdf` draws a discrete index from cumulative shares: the first
  ``i`` with ``cdf[i] > u`` (``searchsorted`` ``side="right"``), so an index of
  zero probability is never returned, even for ``u`` on a share boundary.
- :func:`fan_primary_aperture` is the fan photon under **aperture importance
  sampling** (:class:`~pyradmc.geometry.segments.SegmentedFanSource`'s
  ``aperture_sampling``): the in-aperture point is drawn from a per-segment
  defensive mixture instead of uniformly, and the importance weight that keeps the
  estimate unbiased is returned with the ray. Five uniforms. Its table layout is
  documented there (``aperture_table``); :func:`aperture_contains`
  is its region test and :func:`upper_bound_at` / :func:`select_by_cdf_at` are the
  offset forms of the CDF searches (a kernel cannot slice an array).

Integer loop variables are declared with ``int(0)`` and floats with ``float(0.0)``
because Warp needs the cast to declare a variable mutated inside a dynamic loop.
"""

from __future__ import annotations

import math

from pyradmc.data.handles import Table1D
from pyradmc.rng import RNGState, uniform

__all__ = [
    "aperture_contains",
    "cdf_lower_bound",
    "fan_primary",
    "fan_primary_aperture",
    "select_by_cdf",
    "select_by_cdf_at",
    "spectrum_energy",
    "upper_bound_at",
]

# The per-segment aperture table read by fan_primary_aperture and aperture_contains
# (offsets written out as literals below: Warp compiles these bodies from source).
# With ``n`` bands, a segment's ``4 n + 4`` floats starting at ``base`` hold
#   base + 0                       the weight outside the region, 1 / delta
#   base + 1                       the weight inside it, 1 / (delta + (1 - delta) |F| / |A|)
#   base + 2 .. base + 2 + n       the n + 1 cumulative component shares (entry 0: delta)
#   base + 3 + n .. base + 3 + 2n  the n + 1 band edges along v (non-decreasing)
#   base + 4 + 2n .. base + 3 + 3n the n band lower u bounds
#   base + 4 + 3n .. base + 3 + 4n the n band upper u bounds
# Built by pyradmc.geometry.segments.SegmentedFanSource (its aperture_table).


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


def spectrum_energy(
    edges: Table1D, cdf: Table1D, n_bins: int, u_bin: float, u_within: float
) -> float:
    """Invert a histogram spectrum at two uniforms: ``u_bin`` the bin, ``u_within`` the energy.

    :meth:`~pyradmc.geometry.spectrum.Spectrum.sample_energy`'s inversion (the bin is
    the first ``cdf`` entry ``>= u_bin``, clamped to the top bin against a rounded
    ``cdf[n_bins - 1] < 1``; the energy uniform within it), for generators that draw
    their uniforms themselves. :func:`fan_primary` performs the same two steps inline.
    """
    k = min(cdf_lower_bound(cdf, n_bins, u_bin), n_bins - 1)
    lo = float(edges[k])
    return lo + (float(edges[k + 1]) - lo) * u_within


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


def upper_bound_at(table: Table1D, offset: int, n: int, x: float) -> int:
    """Return the first ``i < n`` with ``table[offset + i] > x``, or ``n`` if there is none.

    ``numpy.searchsorted(table[offset:offset + n], x, side="right")`` over a
    non-decreasing run of ``table``; the offset form, since a kernel cannot slice.
    """
    lo = int(0)  # noqa: UP018, RUF046 - Warp dynamic loop var
    hi = int(n)  # a fresh loop-mutated var for Warp, not the argument
    while lo < hi:
        mid = (lo + hi) // 2
        if table[offset + mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def select_by_cdf_at(table: Table1D, offset: int, n: int, u: float) -> int:
    """:func:`select_by_cdf` over the ``n`` shares stored at ``table[offset:]``."""
    return min(upper_bound_at(table, offset, n, u), n - 1)


def aperture_contains(table: Table1D, base: int, n_bands: int, du: float, dv: float) -> int:
    """Return 1 if the aperture offset ``(du, dv)`` lies in a segment's region, else 0.

    The region is a stack of ``n_bands`` bands along ``v``: band ``b`` is
    ``v_edges[b] <= dv < v_edges[b + 1]`` and holds the half-open interval
    ``u_lo[b] <= du < u_hi[b]`` (empty when ``u_lo == u_hi``). Table layout at
    ``base``: the module comment above.
    """
    i = upper_bound_at(table, base + 3 + n_bands, n_bands + 1, dv)  # first edge above dv
    inside = int(0)  # noqa: UP018, RUF046 - Warp var assigned in a branch
    if i >= 1 and i <= n_bands:
        b = i - 1
        if du >= table[base + 4 + 2 * n_bands + b] and du < table[base + 4 + 3 * n_bands + b]:
            inside = 1
    return inside


def fan_primary_aperture(
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
    table: Table1D,
    base: int,
    n_bands: int,
) -> tuple[float, float, float, float, float]:
    """Sample one fan photon by aperture importance sampling; consumes exactly five uniforms.

    Returns ``(energy, ux, uy, uz, factor)``. The fan is :func:`fan_primary`'s (same
    energy inversion, same ray from ``f`` through ``a + du * u_axis + dv *
    v_axis``); only the in-aperture point ``(du, dv)`` is drawn differently. Its
    target density is uniform on the ``width_u x width_v`` field ``F``,
    ``p = 1 / |F|``; it is drawn instead from the defensive mixture (Hesterberg,
    Technometrics 37, 185 (1995), doi:10.1080/00401706.1995.10484303)

        q(x) = delta / |F| + (1 - delta) 1_A(x) / |A|,

    uniform on ``F`` with probability ``delta`` and uniform on the region ``A`` (the
    table's bands) otherwise, and the returned ``factor`` is the likelihood ratio

        p(x) / q(x) = 1 / (delta + (1 - delta) |F| / |A| 1_A(x)),

    so ``E_q[factor * g(x)] = E_p[g(x)]`` for every ``g``: multiplying a history's
    weight by ``factor`` leaves every expectation — a Dij column, a plan dose —
    unchanged, whatever ``A`` is, as long as ``A`` lies inside ``F``. The weight
    is bounded by ``1 / delta``. A table with ``factor_out = factor_in = 1`` and
    ``cdf = (1, ..., 1)`` is plain uniform sampling (the empty-region fallback).

    Stream contract — five uniforms in this order, whatever component is drawn:
    the spectrum bin, the position within the bin, the component (``select_by_cdf``
    over the ``n_bands + 1`` cumulative shares: index 0 is the whole field, index
    ``b + 1`` band ``b``, chosen in proportion to its area), then the ``u`` and
    ``v`` positions inside the chosen rectangle. Under correlated sampling every
    segment therefore replays the same five uniforms through its own table. Table
    layout at ``base``: the module comment above, built by
    :attr:`~pyradmc.geometry.segments.SegmentedFanSource.aperture_table`.
    """
    u_bin = uniform(rng_state)
    u_within = uniform(rng_state)
    u_pick = uniform(rng_state)
    u_a = uniform(rng_state)
    u_b = uniform(rng_state)
    k = select_by_cdf_at(table, base + 2, n_bands + 1, u_pick)
    du = (u_a - 0.5) * width_u
    dv = (u_b - 0.5) * width_v
    factor = table[base + 1]
    if k > 0:
        b = k - 1
        v_lo = table[base + 3 + n_bands + b]
        v_hi = table[base + 4 + n_bands + b]
        u_lo = table[base + 4 + 2 * n_bands + b]
        u_hi = table[base + 4 + 3 * n_bands + b]
        du = u_lo + (u_hi - u_lo) * u_a
        dv = v_lo + (v_hi - v_lo) * u_b
    elif aperture_contains(table, base, n_bands, du, dv) == 0:
        factor = table[base]
    j = min(cdf_lower_bound(cdf, n_bins, u_bin), n_bins - 1)
    lo = edges[j]
    energy = lo + (edges[j + 1] - lo) * u_within
    dx = ax + du * uxa + dv * vxa - fx
    dy = ay + du * uya + dv * vya - fy
    dz = az + du * uza + dv * vza - fz
    inv_norm = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
    return energy, dx * inv_norm, dy * inv_norm, dz * inv_norm, factor
