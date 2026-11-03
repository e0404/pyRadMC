"""The single-source fan edge-ray scalars (:mod:`pyradmc.geometry.edge_rays`) on the host.

The Warp edge generators compile these functions through the physics loader and the
host route of :class:`FanEdgeDerivativeSource` calls them. Pinned here in float64:

- the shared pieces compute exactly what the host route's vectorized arithmetic
  computed before they existed (``np.array_equal``), so the reference stays put;
- the pieces only the kernel takes — the near-tangent closed form, the device's
  proposal construction, the focused jaw's slant chord — agree with the host's exact
  geometry and proposal in float64, so what the device evaluates in float32 is the
  host's quantity and not a different one.
"""

from __future__ import annotations

import math

import numpy as np

from pyradmc.geometry import edge_rays
from pyradmc.geometry.aperture_gradient import EdgeParameter
from pyradmc.geometry.fan import spectrum_energy
from tests.unit.test_segmented_edge_source import (
    SPECTRUM,
    _expected_parameters,
    _per_edge,
    _segments,
)


def _proposal_table(n: int = 33, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A host-style proposal (s grid, cdf, cell density) with one zero-mass cell."""
    rng = np.random.default_rng(seed)
    mass = rng.random(n - 1)
    mass[5] = 0.0
    s = np.concatenate([[0.0], np.sort(rng.random(n - 1))])
    cdf = np.cumsum(mass) / mass.sum()
    return s, cdf, mass / mass.sum() / np.diff(s)


def test_proposal_point_is_the_host_inversion() -> None:
    """``searchsorted`` side right, clamped residual, ``q = q0 + sign s^2``: bit for bit.

    Uniforms include every cdf entry (a draw on a cell boundary, and on the zero-mass
    cell's), and the table is read at an offset as a kernel reads its row.
    """
    s_grid, cdf, cell = _proposal_table()
    u = np.concatenate([np.random.default_rng(4).random(4_000), cdf[:-1], [0.0]])
    q0, sign = 0.7, -1.0
    k = np.minimum(np.searchsorted(cdf, u, side="right"), cdf.size - 1)
    lower = np.where(k > 0, cdf[np.maximum(k - 1, 0)], 0.0)
    residual = np.clip((u - lower) / (cdf[k] - lower), 0.0, 1.0)
    s = s_grid[k] + (s_grid[k + 1] - s_grid[k]) * residual
    want = np.stack([s, q0 + sign * s**2, cell[k] / np.maximum(2.0 * s, 1.0e-300)], axis=1)
    pad = np.full(7, np.nan)
    tables = [np.concatenate([pad, t]) for t in (cdf, cell, s_grid)]
    got = np.array(
        [edge_rays.edge_proposal_point(*tables, 7, 7, cdf.size, x, q0, sign, 1.0e-300) for x in u]
    )
    np.testing.assert_array_equal(got, want)


def test_ray_point_of_a_jump_edge_sits_on_the_edge_line() -> None:
    nothing = np.zeros(1)
    s, q, r, density = edge_rays.edge_ray_point(
        2, 0.37, 0.1, 1.0, -1.0, 3.0, nothing, nothing, nothing, 0, 0, 0, 1.0e-300, 0.25, 0.9
    )
    assert (s, q, r, density) == (0.0, 0.37, 0.0, 1.0)


def test_ray_crossing_is_the_host_normalization() -> None:
    """The crossing's norm: the host's ``np.linalg.norm`` division, bit for bit.

    And the kernel's direction is the same norm's reciprocal times each component.
    """
    rng = np.random.default_rng(5)
    q = rng.normal(0.0, 4.0, 3_000)
    r = rng.normal(0.0, 2.0, 3_000)
    z_mid = 51.0
    for q_axis in (0, 1):
        u, v = (q, r) if q_axis == 0 else (r, q)
        local = np.stack([u, v, np.full_like(u, z_mid)], axis=1)
        norm = np.linalg.norm(local, axis=1, keepdims=True)
        pairs = list(zip(q, r, strict=True))
        got = np.array([edge_rays.edge_ray_crossing(a, b, q_axis, z_mid) for a, b in pairs])
        np.testing.assert_array_equal(got[:, :2], np.stack([u, v], axis=1))
        np.testing.assert_array_equal(local / got[:, 2:], local / norm)
        kernel = np.array([edge_rays.edge_ray_direction(a, b, q_axis, z_mid) for a, b in pairs])
        np.testing.assert_array_equal(kernel[:, :2], got[:, :2])
        np.testing.assert_array_equal(kernel[:, 2:], local * (1.0 / got[:, 2:]))


def test_in_field_is_the_closed_rectangle() -> None:
    rng = np.random.default_rng(6)
    u, v = rng.uniform(-2.0, 2.0, (2, 2_000))
    u[:4], v[:4] = (-1.0, 1.5, 0.0, 0.0), (0.0, 0.0, -0.5, 0.5)  # on the boundary
    want = (u >= -1.0) & (u <= 1.5) & (v >= -0.5) & (v <= 0.5)
    got = [edge_rays.edge_in_field(a, b, -1.0, 1.5, -0.5, 0.5) for a, b in zip(u, v, strict=True)]
    np.testing.assert_array_equal(np.array(got, dtype=bool), want)


def test_spectrum_energy_is_the_spectrum_inversion() -> None:
    rng = np.random.default_rng(7)
    u_bin, u_within = rng.random((2, 2_000))
    edges, cdf = SPECTRUM.edges, SPECTRUM.cdf
    got = [
        spectrum_energy(edges, cdf, cdf.size, a, b) for a, b in zip(u_bin, u_within, strict=True)
    ]
    np.testing.assert_array_equal(got, SPECTRUM.sample_energies(u_bin, u_within))


# -- the kernel-only pieces against the host's exact geometry, in float64 ---------------


def _tip_edges():
    """The fixture's rounded tips whose rays come within the near-tangent band."""
    edges = _per_edge(_segments(), _expected_parameters(4))
    return [
        e
        for e in edges
        if not e._jump and e._q_tangent is not None and e._sign * (e._q0 - e._q_tangent) < 1.9e-3
    ]


def _block_terms(edge, s: float, near_tangent_s2: float) -> tuple[float, float, float]:
    """:func:`edge_rays.edge_block_terms` of ``edge`` on its central ray at node ``s``.

    Along the host's direction of that ray (the norm divided out), so the twins see
    exactly the ray the host's chord and rate are evaluated on.
    """
    block = edge._block
    q = edge_rays.edge_q(edge._q0, edge._sign, s)
    local, _ = edge._directions(np.array([q]), np.array([edge._r_c]))
    du, dv, dw = local[0]
    return edge_rays.edge_block_terms(
        0, edge._sign, edge._x, edge._z_mid, block.z_top, block.z_bottom, block.tip_radius,
        block.leaf_edges_v[0], block.leaf_edges_v[1], 0, edge._q_axis, edge._q_tangent,
        edge._sign * (edge._q0 - edge._q_tangent), near_tangent_s2, q, s, du, dv, dw,
    )  # fmt: skip


def test_block_terms_without_the_closed_form_are_the_twins() -> None:
    """Below a negative threshold the kernel's chord and rate are the host's twin values."""
    for edge in _tip_edges():
        s = np.sqrt(np.geomspace(1.0e-4, edge._span, 20))
        q = edge._q0 + edge._sign * s**2
        local, _ = edge._directions(q, np.full(s.size, edge._r_c))
        rates = edge._chord_rates(local)
        for k, sk in enumerate(s):
            l_block, rate, l_rest = _block_terms(edge, float(sk), -1.0)
            assert l_block == l_rest
            assert rate == rates[k]


# The closed form is exact (the disc chord, clipped at the slab faces), so in float64
# it meets the host chord and rate to the rounding of the host's own ray-circle test.
# That test cancels like the float32 twin of tests/dij/test_segmented_edge_source_warp.py
# (relative error ~ k eps t_c^2 / (2 h^2)), here at eps = 2^-53: 3e-10 at k = 8 at the
# band's edge s^2 = 1e-4 on this fixture. Measured 2.0e-10 (chord) and 1.2e-10 (rate),
# over 640 nodes of 16 tips. An expansion in place of the closed form would be off by
# O(s_t^2) >= 1e-4, a wrong term by O(1): both far above this.
CLOSED_FORM_RTOL = 1.0e-8


def test_near_tangent_closed_form_is_the_exact_chord_in_float64() -> None:
    """Every near-tangent node of every fixture tip, whole and face-clipped chords alike."""
    n_closed = 0
    for edge in _tip_edges():
        t_off = edge._sign * (edge._q0 - edge._q_tangent)
        s = np.sqrt(np.geomspace(1.0e-4, 2.0e-3 - t_off - 1.0e-6, 40))
        q = edge._q0 + edge._sign * s**2
        r = np.full(s.size, edge._r_c)
        local, _ = edge._directions(q, r)
        terms = np.array([_block_terms(edge, float(sk), 2.0e-3) for sk in s])
        np.testing.assert_allclose(terms[:, 0], edge._block_chords(q, r), rtol=CLOSED_FORM_RTOL)
        np.testing.assert_allclose(terms[:, 1], edge._chord_rates(local), rtol=CLOSED_FORM_RTOL)
        n_closed += s.size
    assert n_closed >= 200


def test_device_proposal_is_the_host_proposal() -> None:
    """The device construction (sequential sum, no renormalization) against ``_proposal``.

    Same node densities, cell masses and defensive mixture; the two normalize the
    cumulative sum differently (the host divides by the pairwise total again), so they
    agree to rounding and not bit for bit (measured 1.2e-14 in the cdf, 1e-15 in the
    cell densities). Any consistent proposal is unbiased; this pins that the device's
    is the host's.
    """
    for edge in _per_edge(_segments(), _expected_parameters(4)):
        if edge._jump:
            continue
        s, cdf, cell = edge._proposal()
        q = edge_rays.edge_q(edge._q0, edge._sign, s)
        e_ref = np.full(s.size, edge._beam.spectrum.mean_energy)
        mu, transmission, rates = edge._derivative_terms(q, np.full(s.size, edge._r_c), e_ref)
        f = [
            edge_rays.edge_node_density(*terms)
            for terms in zip(s, mu, transmission, rates, strict=True)
        ]
        masses = np.zeros(s.size - 1)
        total = 0.0
        for c in range(s.size - 1):
            masses[c] = edge_rays.proposal_cell_mass(f[c], f[c + 1], s[c], s[c + 1])
            total += masses[c]
        device_cdf = np.zeros(s.size - 1)
        edge_rays.edge_proposal_tables(
            masses, device_cdf, 0, s, 0, s.size - 1, total, edge._defensive
        )
        assert device_cdf[-1] == 1.0
        assert np.all(np.diff(device_cdf) >= 0.0)
        np.testing.assert_allclose(device_cdf, cdf, rtol=1.0e-12, atol=1.0e-15)
        np.testing.assert_allclose(masses, cell, rtol=1.0e-12)


def test_focused_slab_chord_is_the_full_slant_thickness() -> None:
    """A focused jaw's jump weight: the closed-form slant chord against the host model."""
    edges = _per_edge(_segments(), [(s, EdgeParameter(0, side)) for s in range(3)
                                     for side in ("neg", "pos")])  # fmt: skip
    for edge in edges:
        assert edge._jump
        r = np.linspace(edge._r_range[0], edge._r_range[1], 9)
        q = np.full(r.size, edge._x + edge._sign * 1.0e-7)  # just inside, as the host
        local, _ = edge._directions(q, r)
        block = edge._block
        got = [edge_rays.focused_slab_chord(block.z_top, block.z_bottom, dw) for dw in local[:, 2]]
        np.testing.assert_allclose(got, edge._block_chords(q, r), rtol=1.0e-12)  # 4e-16
        assert math.isfinite(got[0])


def test_stratified_position_puts_one_history_in_each_stratum() -> None:
    """An edge step's batch of ``n`` histories: history ``i`` lands in ``[i/n, (i+1)/n)``."""
    n = 97
    u = np.random.default_rng(8).random(n)
    got = np.array([edge_rays.stratified_position(i, n, u[i]) for i in range(n)])
    np.testing.assert_array_equal(got, (np.arange(n) + u) / n)
    np.testing.assert_array_equal(np.floor(got * n), np.arange(n))
