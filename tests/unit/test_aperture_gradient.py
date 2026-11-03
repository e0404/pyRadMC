"""Aperture-gradient estimators: the source-side identities, without transport.

Every estimator here is a :class:`~pyradmc.geometry.source.Source` whose expected
weight-times-response equals the derivative of the forward collimated source's
expected response with respect to one edge coordinate (a leaf tip or jaw edge).
Transport is independent of that coordinate (the stack only rescales weights), so
the identity can be checked on a known response ``g(q, r)`` instead of a dose:
the reference value is the derivative of a 1-D quadrature of the forward
transmission, which owes nothing to the sampling being tested.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.aperture_gradient import (
    BixelChainRuleSource,
    EdgeDerivativeSource,
    EdgeParameter,
    ProfileChainRuleSource,
    RayFiniteDifferenceSource,
    edge_value,
    with_edge,
)
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.source import ParallelBeamSource
from pyradmc.rng.host import HostRNG, uniform
from tests.conftest import SEED

ENERGY = 2.0
# Water at a heavy-metal-like density keeps the fast tier on the analytic backend
# (no EPDL data) while giving partial transmission through 6 cm (T_min ~ 0.23).
SURROGATE_DENSITY = 5.0
FIELD_X = (0.5, 9.5)
FIELD_Y = (0.0, 4.0)
N_SAMPLES = 20_000
XS = AnalyticCrossSections()
_MU_OVER_RHO = XS.mu_over_rho_total(ENERGY, WATER)

MLC_POS = EdgeParameter(device=1, side="pos", pair=0)
MLC_NEG = EdgeParameter(device=1, side="neg", pair=0)
JAW_POS = EdgeParameter(device=0, side="pos")
JAW_NEG = EdgeParameter(device=0, side="neg")


def _stack() -> BeamLimitingStack:
    """An upstream u-jaw pair over a two-pair rounded-tip MLC (local = engine x, y).

    The jaw's +u edge (5.6) sits inside the pair-0 +u tip's penumbra (5.25 to
    5.83), so each parameter's derivative is weighted by a non-trivial
    transmission through the *other* device.
    """
    return BeamLimitingStack(
        frame=BeamFrame(origin=(0.0, 0.0, -100.0)),
        devices=(
            JawPair(
                axis="u",
                z_top=30.0,
                z_bottom=36.0,
                edge_neg=1.5,
                edge_pos=5.6,
                material=WATER,
                density=SURROGATE_DENSITY,
            ),
            MLC(
                z_top=50.0,
                z_bottom=56.0,
                leaf_edges_v=(0.0, 2.0, 4.0),
                tips_neg=(3.25, 2.0),
                tips_pos=(5.25, 7.0),
                tip_radius=8.0,
                material=WATER,
                density=SURROGATE_DENSITY,
            ),
        ),
    )


def _beam() -> ParallelBeamSource:
    return ParallelBeamSource(ENERGY, -1.0, FIELD_X, FIELD_Y)


def _transmission(stack: BeamLimitingStack, q: float, r: float) -> float:
    """Narrow-beam transmission of the vertical ray at local (u, v) = (q, r)."""
    lengths = stack.path_lengths(np.array([[q, r, 0.0]]), np.array([[0.0, 0.0, 1.0]]))[0]
    tau = sum(
        _MU_OVER_RHO * device.density * length
        for device, length in zip(stack.devices, lengths, strict=True)
    )
    return math.exp(-tau)


def _g_q(q: float, centre: float) -> float:
    return math.exp(-(((q - centre) / 0.4) ** 2))


def _forward_response(stack: BeamLimitingStack, centre: float, breaks: list[float]) -> float:
    """``E_forward[T g] = (1/A) int int T(q, r) g_q(q) g_r(r) dq dr`` by quadrature.

    T depends on r only through the MLC strip (parallel-sided leaves, vertical
    rays), so each strip is one 1-D quadrature in q times the exact r-integral.
    """
    area = (FIELD_X[1] - FIELD_X[0]) * (FIELD_Y[1] - FIELD_Y[0])
    total = 0.0
    for r_lo, r_hi in ((0.0, 2.0), (2.0, 4.0)):
        r_mid = 0.5 * (r_lo + r_hi)
        g_r_integral = (r_hi - r_lo) + 0.05 * (r_hi**2 - r_lo**2)
        value, _ = quad(
            lambda q, r=r_mid: _transmission(stack, q, r) * _g_q(q, centre),
            FIELD_X[0],
            FIELD_X[1],
            points=sorted(breaks),
            limit=400,
            epsabs=1e-10,
            epsrel=1e-9,
        )
        total += g_r_integral * value
    return total / area


def _quadrature_derivative(parameter: EdgeParameter, step: float) -> float:
    """Central difference of the forward quadrature at step ``step``."""
    stack = _stack()
    x = edge_value(stack, parameter)
    plus = with_edge(stack, parameter, x + step)
    minus = with_edge(stack, parameter, x - step)
    breaks = [1.5, 5.6, 3.25, 5.25, 2.0, 7.0, x + step, x - step]
    return (_forward_response(plus, x, breaks) - _forward_response(minus, x, breaks)) / (2 * step)


def _mc_mean(source, centre: float, n: int = N_SAMPLES) -> tuple[float, float]:
    """Mean and standard error of ``weight * g`` over the source's batch route.

    :func:`test_emit_and_batch_routes_agree` pins the batch route to ``emit``.
    """
    columns = source.sample_batch(SEED, 0, n)
    x = columns["x"].astype(np.float64)
    y = columns["y"].astype(np.float64)
    values = columns["weight"].astype(np.float64) * np.exp(-(((x - centre) / 0.4) ** 2))
    values *= 1.0 + 0.1 * y
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n))


# --- parameter plumbing ----------------------------------------------------------


def test_with_edge_moves_only_the_named_coordinate() -> None:
    stack = _stack()
    moved = with_edge(stack, MLC_POS, 6.0)
    assert moved.devices[1].tips_pos == (6.0, 7.0)
    assert moved.devices[1].tips_neg == stack.devices[1].tips_neg
    assert moved.devices[0] == stack.devices[0]
    assert with_edge(stack, JAW_NEG, 1.0).devices[0].edge_neg == 1.0
    for parameter in (MLC_POS, MLC_NEG, JAW_POS, JAW_NEG):
        assert edge_value(with_edge(stack, parameter, 4.0), parameter) == 4.0


@pytest.mark.parametrize(
    "parameter",
    [
        EdgeParameter(device=1, side="pos"),  # MLC needs a pair
        EdgeParameter(device=0, side="pos", pair=0),  # a jaw has none
        EdgeParameter(device=1, side="up", pair=0),
        EdgeParameter(device=2, side="pos"),
    ],
)
def test_malformed_parameters_are_refused(parameter: EdgeParameter) -> None:
    with pytest.raises(ValueError):
        edge_value(_stack(), parameter)


def test_focused_geometry_is_refused() -> None:
    """Edge translation equals ray translation only for unfocused faces."""
    stack = _stack()
    focused = replace(stack, devices=(replace(stack.devices[0], focused=True), stack.devices[1]))
    with pytest.raises(ValueError, match="focused"):
        EdgeDerivativeSource(_beam(), focused, XS, JAW_POS)


@pytest.mark.parametrize(
    "make",
    [
        lambda: EdgeDerivativeSource(_beam(), _stack(), XS, MLC_POS),
        lambda: BixelChainRuleSource(_beam(), _stack(), XS, JAW_NEG, bixel_pitch=0.5),
    ],
)
def test_edge_sampled_routes_share_one_arithmetic(make) -> None:
    """``emit`` is the batch arithmetic applied to the history's own two uniforms."""
    source = make()
    rng = HostRNG()
    for i in range(20):
        draws = rng.init_state(SEED, i)
        u_r, u_q = uniform(draws), uniform(draws)
        points, weights = source._rays(np.array([u_r]), np.array([u_q]))
        p = source.emit(rng.init_state(SEED, i))
        assert (p.x, p.y, p.z) == tuple(points[0])
        assert p.weight == weights[0]


def test_edge_sampled_batches_are_chunk_invariant() -> None:
    """History h draws the same rays however the batch route is chunked."""
    source = EdgeDerivativeSource(_beam(), _stack(), XS, MLC_POS)
    whole = source.sample_batch(SEED, 0, 100)
    tail = source.sample_batch(SEED, 60, 40)
    for name, column in whole.items():
        np.testing.assert_array_equal(column[60:], tail[name])


def test_finite_difference_routes_agree() -> None:
    """The reweighting wrapper follows its beam's stream on both routes."""
    source = RayFiniteDifferenceSource(_beam(), _stack(), XS, MLC_NEG, step=0.05)
    rng = HostRNG()
    columns = source.sample_batch(SEED, 7, 50)
    for i in range(50):
        p = source.emit(rng.init_state(SEED, 7 + i))
        for name in ("energy", "x", "y", "z", "ux", "uy", "uz"):
            assert columns[name][i] == np.float32(getattr(p, name)), name
        # The wrapper reweights the beam's float32 columns, emit its float64
        # primary: the positions differ by one float32 rounding, the weights by
        # the profile slope times that (the collimation wrappers behave the same).
        assert columns["weight"][i] == pytest.approx(p.weight, rel=1e-4, abs=1e-6)


# --- the exact (importance-sampled) estimator -------------------------------------


def test_rounded_tip_sampler_inverts_the_tip_transmission() -> None:
    """``edge_coordinate(t)`` lands where the isolated tip transmits ``T_min + t(1-T_min)``."""
    source = EdgeDerivativeSource(_beam(), _stack(), XS, MLC_POS)
    tip_only = with_edge(_stack(), JAW_POS, 1.0e4)  # retract the jaw block
    t_min = _transmission(tip_only, 20.0, 1.0)
    for t in np.linspace(0.0, 1.0, 11):
        q = source.edge_coordinate(float(t))
        assert 5.25 <= q <= 5.25 + 8.0
        assert _transmission(tip_only, q, 1.0) == pytest.approx(t_min + t * (1 - t_min), abs=1e-6)


def test_straight_edge_samples_sit_on_the_edge() -> None:
    """A straight jaw face is a jump in T: the derivative measure is a point mass."""
    source = EdgeDerivativeSource(_beam(), _stack(), XS, JAW_POS)
    rng = HostRNG()
    xs = {source.emit(rng.init_state(SEED, i)).x for i in range(200)}
    assert all(abs(x - 5.6) < 1e-9 for x in xs)


@pytest.mark.parametrize("parameter", [MLC_POS, MLC_NEG, JAW_POS, JAW_NEG])
def test_edge_derivative_source_is_unbiased(parameter: EdgeParameter) -> None:
    """``E[w g]`` equals dE_forward[T g]/dx from quadrature, for both sides and devices."""
    source = EdgeDerivativeSource(_beam(), _stack(), XS, parameter)
    centre = edge_value(_stack(), parameter)
    reference = _quadrature_derivative(parameter, 1.0e-3)
    mean, sem = _mc_mean(source, centre)
    assert reference != 0.0
    assert abs(mean - reference) < 4.0 * sem, (mean, sem, reference)


def test_opening_and_closing_sides_have_opposite_signs() -> None:
    """Moving a +u tip up opens the field (w >= 0); a -u tip up closes it (w <= 0)."""
    rng = HostRNG()
    for parameter, sign in ((MLC_POS, 1.0), (MLC_NEG, -1.0)):
        source = EdgeDerivativeSource(_beam(), _stack(), XS, parameter)
        weights = [source.emit(rng.init_state(SEED, i)).weight for i in range(100)]
        assert all(sign * w >= 0.0 for w in weights)
        assert any(w != 0.0 for w in weights)


# --- the finite-difference oracle -------------------------------------------------


@pytest.mark.parametrize("parameter", [MLC_POS, JAW_POS])
def test_ray_finite_difference_is_unbiased_for_its_step(parameter: EdgeParameter) -> None:
    """Common-random-number FD reweights the forward rays: unbiased for FD_h itself."""
    step = 0.05
    source = RayFiniteDifferenceSource(_beam(), _stack(), XS, parameter, step=step)
    centre = edge_value(_stack(), parameter)
    mean, sem = _mc_mean(source, centre)
    assert abs(mean - _quadrature_derivative(parameter, step)) < 4.0 * sem


# --- the bixel chain rule (what DAO through a bixel Dij computes) ------------------


def test_bixel_chain_rule_spreads_the_edge_over_its_bixel() -> None:
    """Rays fill the bixel holding the tip uniformly; the weight matches the exact source."""
    pitch = 0.5
    source = BixelChainRuleSource(_beam(), _stack(), XS, MLC_POS, bixel_pitch=pitch)
    exact = EdgeDerivativeSource(_beam(), _stack(), XS, MLC_POS)
    rng = HostRNG()
    xs = np.array([source.emit(rng.init_state(SEED, i)).x for i in range(2000)])
    assert xs.min() >= 5.0 and xs.max() <= 5.5  # the bixel [5.0, 5.5) holds the tip 5.25
    assert abs(xs.mean() - 5.25) < 4.0 * (pitch / math.sqrt(12.0)) / math.sqrt(xs.size)
    # Both carry the same total derivative measure; only where it is placed differs.
    assert source.weight_scale == pytest.approx(exact.weight_scale)


# --- the profile-aware chain rule -------------------------------------------------


def _profile_masses(parameter: EdgeParameter, pitch: float) -> dict[int, float]:
    """Exact per-bixel share of ``|dT/dx|``: transmission drops across each bixel.

    Independent of the source: both jaw blocks and the opposing tip are retracted,
    and the lattice is walked boundary by boundary along the leaf's profile with
    the test's own transmission helper (vertical rays in strip 0).
    """
    stack = with_edge(with_edge(_stack(), JAW_POS, 1.0e4), JAW_NEG, -1.0e4)
    pos = parameter.side == "pos"
    stack = with_edge(stack, MLC_NEG if pos else MLC_POS, -1.0e4 if pos else 1.0e4)
    x = edge_value(stack, parameter)
    first = math.floor(x / pitch) if pos else math.ceil(x / pitch) - 1
    masses: dict[int, float] = {}
    previous = 1.0  # just outside the block
    for j in range(int(15.0 / pitch) + 2):
        k = first + j if pos else first - j
        boundary = (k + 1) * pitch if pos else k * pitch
        t = _transmission(stack, boundary, 1.0)
        masses[k] = previous - t
        previous = t
    return masses


@pytest.mark.parametrize("parameter", [MLC_POS, MLC_NEG])
def test_profile_chain_rule_weights_bixels_by_their_profile_share(parameter) -> None:
    """Bixel k is drawn with probability m_k / (1 - T_min), uniformly inside it."""
    pitch = 0.1
    source = ProfileChainRuleSource(_beam(), _stack(), XS, parameter, bixel_pitch=pitch)
    columns = source.sample_batch(SEED, 0, 40_000)
    x = columns["x"].astype(np.float64)
    k = np.floor(x / pitch).astype(int)
    expected = _profile_masses(parameter, pitch)
    total = sum(expected.values())
    for bixel, mass in expected.items():
        p = mass / total
        if p < 1e-3:
            continue
        observed = np.mean(k == bixel)
        assert abs(observed - p) < 4.0 * math.sqrt(p * (1 - p) / x.size) + 1e-12, bixel
    # Uniform within the bixel: the in-bixel offset has mean 1/2, variance 1/12.
    offset = x / pitch - k
    assert abs(offset.mean() - 0.5) < 4.0 * math.sqrt(1 / 12 / x.size)
    exact = EdgeDerivativeSource(_beam(), _stack(), XS, parameter)
    assert source.weight_scale == pytest.approx(exact.weight_scale)


def test_profile_chain_rule_puts_a_straight_edge_in_one_bixel() -> None:
    """A jump has all its mass at the edge: the bixel holding the edge gets all of it."""
    source = ProfileChainRuleSource(_beam(), _stack(), XS, JAW_POS, bixel_pitch=0.5)
    x = source.sample_batch(SEED, 0, 2000)["x"]
    assert x.min() >= 5.5 and x.max() <= 6.0  # the edge 5.6 sits in [5.5, 6.0)


def test_profile_chain_rule_converges_to_the_exact_gradient() -> None:
    """At a pitch fine against the penumbra it is unbiased for dE[T g]/dx."""
    source = ProfileChainRuleSource(_beam(), _stack(), XS, MLC_POS, bixel_pitch=0.002)
    mean, sem = _mc_mean(source, edge_value(_stack(), MLC_POS))
    reference = _quadrature_derivative(MLC_POS, 1.0e-3)
    assert abs(mean - reference) < 4.0 * sem, (mean, sem, reference)


@pytest.mark.parametrize("parameter", [MLC_POS, MLC_NEG, JAW_POS])
def test_strip_restricted_rest_of_stack_is_bit_identical(parameter: EdgeParameter) -> None:
    """Under a parallel beam a ray in strip i cuts no other leaf pair: dropping them is exact.

    The edge sources evaluate the rest-of-stack transmission on the parameter's strip
    only (the host cost scales with the number of leaf pairs otherwise); every
    skipped pair contributes an exact +0.0 to the chord sum, so the weights must be
    bit-identical to the full stack with the parameter's block retracted.
    """
    from pyradmc.geometry.collimation import _StackTransmission

    source = EdgeDerivativeSource(_beam(), _stack(), XS, parameter)
    far = 1.0e4 if parameter.side == "pos" else -1.0e4
    full = _StackTransmission(with_edge(_stack(), parameter, far), XS, ENERGY, 0.010, 512)
    u = np.random.default_rng(SEED).random((2000, 2))
    points, weights = source._rays(u[:, 0], u[:, 1])
    directions = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    expected = source.weight_scale * full.factors(np.full(len(points), ENERGY), points, directions)
    np.testing.assert_array_equal(weights, expected)


def test_moved_edges_reuse_cached_tables_bit_identically() -> None:
    """Rebuilding a source after the leaves move reuses the profile and mu tables.

    Under a parallel beam the edge profile is translation-invariant along the leaf
    travel and the mu(E) tables depend only on the materials, so an optimizer
    re-targeting hundreds of edges per iteration need not rebuild either; the
    reused source must be bit-identical to one built from scratch.
    """
    from pyradmc.geometry import aperture_gradient

    moved = with_edge(_stack(), MLC_POS, 5.4)
    warm = EdgeDerivativeSource(_beam(), _stack(), XS, MLC_POS)  # fills the caches
    reused = EdgeDerivativeSource(_beam(), moved, XS, MLC_POS)
    aperture_gradient.clear_caches()
    fresh = EdgeDerivativeSource(_beam(), moved, XS, MLC_POS)
    u = np.random.default_rng(SEED).random((500, 2))
    for a, b in zip(reused._rays(u[:, 0], u[:, 1]), fresh._rays(u[:, 0], u[:, 1]), strict=True):
        np.testing.assert_array_equal(a, b)
    assert reused.weight_scale == fresh.weight_scale
    assert warm.edge_coordinate(0.3) + 0.15 == pytest.approx(reused.edge_coordinate(0.3))


def test_transmission_table_reuse_refuses_other_materials() -> None:
    from pyradmc.geometry.collimation import _StackTransmission

    table = _StackTransmission(_stack(), XS, ENERGY, 0.010, 512)
    other = replace(
        _stack(), devices=(replace(_stack().devices[0], density=1.0), _stack().devices[1])
    )
    with pytest.raises(ValueError, match="materials"):
        table.for_stack(other)
    assert table.for_stack(with_edge(_stack(), MLC_POS, 6.0))._stack.devices[1].tips_pos[0] == 6.0


def test_edge_gradient_beamlets_delegate_to_their_edges() -> None:
    """Beamlet j of the edge set emits exactly edge j's rays, signed weights kept."""
    from pyradmc.geometry.aperture_gradient import EdgeGradientBeamletSource

    edges = [
        EdgeDerivativeSource(_beam(), _stack(), XS, MLC_POS),
        EdgeDerivativeSource(_beam(), _stack(), XS, MLC_NEG),
    ]
    source = EdgeGradientBeamletSource(edges)
    assert source.n_beamlets == 2
    assert source.max_energy == ENERGY
    rng = HostRNG()
    for j, edge in enumerate(edges):
        assert source.emit(j, rng.init_state(SEED, 3)) == edge.emit(rng.init_state(SEED, 3))
        batch = source.sample_beamlet_batch(SEED, 5, 40, j)
        direct = edge.sample_batch(SEED, 5, 40)
        for name, column in direct.items():
            np.testing.assert_array_equal(batch[name], column)
    assert np.all(source.sample_beamlet_batch(SEED, 0, 40, 1)["weight"] <= 0.0)
