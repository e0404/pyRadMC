"""The fan-beam aperture-gradient estimator: source-side identities, no transport.

Under a divergent fan an edge move is not a ray translation, so
:class:`FanEdgeDerivativeSource` samples rays from a tabulated proposal along the
edge and weights each by its exact ``dT/dx`` (the chord edge-derivative twins) over
the proposal density, with the focused-jaw jump sampled on the edge line. Checked,
as for the parallel estimator, on a known response ``g``: the reference is the
derivative of a fine quadrature of the forward fan response over the aperture, the
strips and the energy bin, which owes nothing to the sampling under test.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.aperture_gradient import (
    EdgeParameter,
    FanEdgeDerivativeSource,
    edge_value,
    with_edge,
)
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

XS = AnalyticCrossSections()
DENSITY = 5.0  # water at a heavy-metal-like density: partial transmission (T_min ~ 0.2)
SPECTRUM = Spectrum(edges=[1.8, 2.2], weights=[1.0])
Z_REF = 100.0
APERTURE_U = (-1.0, 7.0)  # at the reference plane
APERTURE_V = (-1.0, 3.0)
JAW_MID = 33.9
MLC_MID = 53.0
N_SAMPLES = 20_000

MLC_POS = EdgeParameter(device=1, side="pos", pair=0)
MLC_NEG = EdgeParameter(device=1, side="neg", pair=0)
JAW_POS = EdgeParameter(device=0, side="pos")


def _beam() -> SpectralBeamSource:
    centre = (0.5 * sum(APERTURE_U), 0.5 * sum(APERTURE_V), Z_REF)
    return SpectralBeamSource(
        SPECTRUM,
        focal_point=(0.0, 0.0, 0.0),
        center=centre,
        width_u=APERTURE_U[1] - APERTURE_U[0],
        width_v=APERTURE_V[1] - APERTURE_V[0],
    )


def _stack(focused_jaw: bool = False) -> BeamLimitingStack:
    """A u-jaw over a focused-sides MLC, frame at the focal spot (local = engine).

    The jaw's +u edge (2.0 at its mid-plane) projects to 3.13 at the MLC mid-plane,
    inside the pair-0 +u tip's penumbra (from 2.6): each parameter is weighted by
    a non-trivial transmission through the other device.
    """
    return BeamLimitingStack(
        frame=BeamFrame(origin=(0.0, 0.0, 0.0)),
        devices=(
            JawPair(
                axis="u",
                z_top=30.0,
                z_bottom=37.8,
                edge_neg=-0.4,
                edge_pos=2.0,
                material=WATER,
                density=DENSITY,
                focused=focused_jaw,
            ),
            MLC(
                z_top=50.0,
                z_bottom=56.0,
                leaf_edges_v=(0.0, 1.0, 2.0),
                tips_neg=(0.8, 1.0),
                tips_pos=(2.6, 3.0),
                tip_radius=8.0,
                material=WATER,
                density=DENSITY,
                focused_sides=True,
            ),
        ),
    )


def _transmission(stack: BeamLimitingStack, u_ref: np.ndarray, v_ref: float, e: float):
    """Forward transmission of the focal rays through reference-plane points."""
    d = np.column_stack([u_ref, np.full_like(u_ref, v_ref), np.full_like(u_ref, Z_REF)])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    lengths = stack.path_lengths(np.zeros_like(d), d)
    mu = XS.mu_over_rho_total(e, WATER) * DENSITY
    return np.exp(-mu * lengths.sum(axis=1))


def _g(u_ref: np.ndarray, v_ref, z_mid: float, centre: float) -> np.ndarray:
    """The test response, in the parameter device's mid-plane coordinates."""
    u = u_ref * z_mid / Z_REF
    v = v_ref * z_mid / Z_REF
    return np.exp(-(((u - centre) / 0.4) ** 2)) * (1.0 + 0.1 * v)


def _fd_reference(stack, parameter, z_mid: float, step: float = 1.0e-3) -> float:
    """d/dx (1/A) int int dE T g by central difference of a fine quadrature.

    Integrates only the difference T(x+h) - T(x-h), which vanishes away from the
    edge, on fine trapezoid grids split at every kink and jump (strip edges in v,
    tip apexes and jaw faces in u); Gauss-Legendre over v within each strip and
    over the energy bin, where the integrand is smooth.
    """
    x = edge_value(stack, parameter)
    plus = with_edge(stack, parameter, x + step)
    minus = with_edge(stack, parameter, x - step)
    to_ref = Z_REF / z_mid
    window = (max(APERTURE_U[0], (x - 1.5) * to_ref), min(APERTURE_U[1], (x + 1.5) * to_ref))
    breaks = sorted(
        {
            *window,
            *(b * Z_REF / MLC_MID for b in (0.8, 1.0, 2.6, 3.0)),
            *(b * Z_REF / JAW_MID for b in (-0.4, 2.0)),
            (x + step) * to_ref,
            (x - step) * to_ref,
        }
    )
    breaks = [b for b in breaks if window[0] <= b <= window[1]]
    u_grid = np.concatenate(
        [np.linspace(a, b, 4001)[:-1] for a, b in pairwise(breaks)] + [np.array([breaks[-1]])]
    )
    v_cuts = sorted({APERTURE_V[0], APERTURE_V[1], *(e * Z_REF / MLC_MID for e in (0.0, 1.0, 2.0))})
    v_cuts = [v for v in v_cuts if APERTURE_V[0] <= v <= APERTURE_V[1]]
    gl_v, gw_v = np.polynomial.legendre.leggauss(6)
    gl_e, gw_e = np.polynomial.legendre.leggauss(3)
    e_lo, e_hi = SPECTRUM.edges[0], SPECTRUM.edges[-1]
    total = 0.0
    for v0, v1 in pairwise(v_cuts):
        for gv, wv in zip(gl_v, gw_v, strict=True):
            v = 0.5 * (v0 + v1) + 0.5 * (v1 - v0) * gv
            for ge, we in zip(gl_e, gw_e, strict=True):
                e = 0.5 * (e_lo + e_hi) + 0.5 * (e_hi - e_lo) * ge
                diff = _transmission(plus, u_grid, v, e) - _transmission(minus, u_grid, v, e)
                integrand = diff * _g(u_grid, v, z_mid, x)
                total += 0.25 * (v1 - v0) * wv * we * float(np.trapezoid(integrand, u_grid))
    area = (APERTURE_U[1] - APERTURE_U[0]) * (APERTURE_V[1] - APERTURE_V[0])
    return total / (2.0 * step) / area


def _mc(source, z_mid: float, centre: float, n: int = N_SAMPLES) -> tuple[float, float]:
    """Mean and standard error of ``weight * g`` over the source's batch route."""
    columns = source.sample_batch(SEED, 0, n)
    d = np.column_stack([columns["ux"], columns["uy"], columns["uz"]]).astype(np.float64)
    u_ref = d[:, 0] / d[:, 2] * Z_REF
    v_ref = d[:, 1] / d[:, 2] * Z_REF
    values = columns["weight"].astype(np.float64) * _g(u_ref, v_ref, z_mid, centre)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n))


@pytest.mark.parametrize(
    ("parameter", "focused_jaw", "z_mid"),
    [
        (MLC_POS, False, MLC_MID),
        (MLC_NEG, False, MLC_MID),
        (JAW_POS, False, JAW_MID),  # straight face under a fan: smooth
        (JAW_POS, True, JAW_MID),  # focused face: a pure jump on the edge line
    ],
    ids=["tip-pos", "tip-neg", "jaw-straight", "jaw-focused"],
)
def test_fan_edge_derivative_is_unbiased(parameter, focused_jaw: bool, z_mid: float) -> None:
    stack = _stack(focused_jaw)
    source = FanEdgeDerivativeSource(_beam(), stack, XS, parameter)
    reference = _fd_reference(stack, parameter, z_mid)
    mean, sem = _mc(source, z_mid, edge_value(stack, parameter))
    assert reference != 0.0
    assert abs(mean - reference) < 4.0 * sem, (mean, sem, reference)


def test_fan_rays_start_at_the_focal_spot_and_consume_four_uniforms() -> None:
    source = FanEdgeDerivativeSource(_beam(), _stack(), XS, MLC_POS)
    rng = HostRNG()
    for i in range(50):
        state = rng.init_state(SEED, i)
        p = source.emit(state)
        assert (p.x, p.y, p.z) == (0.0, 0.0, 0.0)
        assert math.isclose(p.ux**2 + p.uy**2 + p.uz**2, 1.0, rel_tol=1e-12)
        assert 1.8 <= p.energy <= 2.2
        # The fifth draw of the history's stream is the one emit did not consume.
        replay = rng.init_state(SEED, i)
        for _ in range(4):
            replay.random()
        assert state.random() == replay.random()


def test_fan_estimator_refuses_parallel_sided_leaves() -> None:
    """A fan ray crosses parallel strip sides obliquely: T then depends on v at the edge."""
    stack = _stack()
    parallel = replace(
        stack, devices=(stack.devices[0], replace(stack.devices[1], focused_sides=False))
    )
    with pytest.raises(ValueError, match="focused"):
        FanEdgeDerivativeSource(_beam(), parallel, XS, MLC_POS)


def test_fan_estimator_needs_the_focal_spot_at_the_frame_origin() -> None:
    stack = replace(_stack(), frame=BeamFrame(origin=(0.0, 0.0, -1.0)))
    with pytest.raises(ValueError, match="focal"):
        FanEdgeDerivativeSource(_beam(), stack, XS, MLC_POS)


@pytest.mark.parametrize(
    ("parameter", "focused_jaw"),
    [(MLC_POS, False), (MLC_NEG, False), (JAW_POS, False)],
    ids=["tip-pos", "tip-neg", "jaw-straight"],
)
def test_support_start_closed_form_matches_the_chord_bisection(parameter, focused_jaw) -> None:
    """The tangent / corner ray in closed form is where the isolated block's chord begins."""
    from pyradmc.geometry.aperture_gradient import fan_support_start

    stack = _stack(focused_jaw)
    source = FanEdgeDerivativeSource(_beam(), stack, XS, parameter)
    q0 = fan_support_start(source.device_record())
    r_c = 0.5 * sum(source.device_record().r_range)
    sign = 1.0 if parameter.side == "pos" else -1.0
    outside = source._block_chords(np.array([q0 - sign * 1e-9]), np.array([r_c]))[0]
    inside = source._block_chords(np.array([q0 + sign * 1e-6]), np.array([r_c]))[0]
    assert outside == 0.0
    assert inside > 0.0
