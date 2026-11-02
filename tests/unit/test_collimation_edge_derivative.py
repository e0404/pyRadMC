"""Edge derivatives of the collimation chords: the pathwise half of a fan gradient.

Under a divergent fan, moving a leaf tip or jaw edge is no longer a translation of
the rays, so the aperture-gradient estimator needs each ray's own ``d(chord)/dx``.
The scalar twins compute it analytically: every chord endpoint lies on a surface,
and a surface moving with velocity ``V`` moves its intersection parameter by
``(n . V) / (n . d)`` (Hadamard). Pinned here against a float64 central difference
of the chord twins themselves, on random oblique rays, away from the measure-zero
rays where the chord has a kink.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc.geometry.collimation import (
    jaw_edge_derivative,
    jaw_path_length,
    mlc_path_length,
    mlc_tip_edge_derivative,
)
from tests.conftest import SEED

STEP = 1.0e-6
Z_TOP, Z_BOTTOM = 50.0, 56.0
TIP_RADIUS = 8.0


def _rays(n: int, rng: np.random.Generator, spread: float) -> list[tuple[float, ...]]:
    """Rays from above the slab through points near the edges, local frame."""
    rays = []
    for _ in range(n):
        target = (rng.uniform(4.0, 7.0), rng.uniform(0.1, 1.9), rng.uniform(Z_TOP, Z_BOTTOM))
        origin = (
            target[0] - spread * rng.normal(),
            target[1] - 0.2 * rng.normal(),
            0.0,
        )
        d = np.subtract(target, origin)
        d /= np.linalg.norm(d)
        rays.append((*origin, *d))
    return rays


def _jaw(p_q, d_q, p_w, d_w, edge_neg, edge_pos, focused) -> float:
    return jaw_path_length(p_q, d_q, p_w, d_w, Z_TOP, Z_BOTTOM, edge_neg, edge_pos, focused, 0)


@pytest.mark.parametrize("focused", [0, 1])
@pytest.mark.parametrize("side_pos", [True, False])
def test_jaw_edge_derivative_matches_finite_difference(focused: int, side_pos: bool) -> None:
    rng = np.random.default_rng(SEED)
    checked = 0
    for p_u, _, p_w, d_u, _, d_w in _rays(3000, rng, spread=6.0):
        neg, pos = (1.0, 5.5) if side_pos else (5.5, 9.0)

        ray = (p_u, d_u, p_w, d_w)

        def chord(edge: float, ray=ray, neg=neg, pos=pos) -> float:
            return _jaw(*ray, *((neg, edge) if side_pos else (edge, pos)), focused)

        x = pos if side_pos else neg
        fd = (chord(x + STEP) - chord(x - STEP)) / (2 * STEP)
        fd_half = (chord(x + STEP / 2) - chord(x - STEP / 2)) / STEP
        if abs(fd - fd_half) > 1e-5 * (1.0 + abs(fd)):
            continue  # straddles a kink of the chord
        derivative = jaw_edge_derivative(
            p_u, d_u, p_w, d_w, Z_TOP, Z_BOTTOM, neg, pos, focused, 1 if side_pos else 0
        )
        assert derivative == pytest.approx(fd, rel=1e-5, abs=1e-6)
        checked += fd != 0.0
    assert checked > 200


def test_focused_jaw_is_end_on_from_its_focal_spot() -> None:
    """Every focal ray sees full thickness or nothing: the chord is a step in x."""
    rng = np.random.default_rng(SEED)
    for _ in range(500):
        u_mid = rng.uniform(4.0, 7.0)
        d = np.array([u_mid, 0.0, 0.5 * (Z_TOP + Z_BOTTOM)])
        d /= np.linalg.norm(d)
        value = jaw_edge_derivative(0.0, d[0], 0.0, d[2], Z_TOP, Z_BOTTOM, 1.0, 5.5, 1, 1)
        assert value == 0.0


@pytest.mark.parametrize("focused_sides", [0, 1])
@pytest.mark.parametrize("above", [True, False])
def test_mlc_tip_edge_derivative_matches_finite_difference(focused_sides: int, above: bool) -> None:
    rng = np.random.default_rng(SEED + 1)
    edges = np.array([0.0, 2.0])
    checked = 0
    for p_u, p_v, p_w, d_u, d_v, d_w in _rays(3000, rng, spread=3.0):
        tip = 5.25

        ray = (p_u, p_v, p_w, d_u, d_v, d_w)

        def chord(value: float, ray=ray) -> float:
            neg = np.array([value if not above else -1.0e4])
            pos = np.array([value if above else 1.0e4])
            geometry = (Z_TOP, Z_BOTTOM, TIP_RADIUS, 1, 0, 0, edges, neg, pos, focused_sides, 0)
            return mlc_path_length(*ray, *geometry)

        fd = (chord(tip + STEP) - chord(tip - STEP)) / (2 * STEP)
        fd_half = (chord(tip + STEP / 2) - chord(tip - STEP / 2)) / STEP
        if abs(fd - fd_half) > 1e-5 * (1.0 + abs(fd)) or not math.isfinite(fd):
            continue
        derivative = mlc_tip_edge_derivative(
            p_u, p_v, p_w, d_u, d_v, d_w, Z_TOP, Z_BOTTOM, TIP_RADIUS,
            0.0, 2.0, tip, focused_sides, 1 if above else 0,
        )  # fmt: skip
        assert derivative == pytest.approx(fd, rel=1e-4, abs=1e-5)
        checked += fd != 0.0
    assert checked > 200


@pytest.mark.parametrize("focused_sides", [0, 1])
@pytest.mark.parametrize("above", [True, False])
def test_single_leaf_chord_equals_the_one_pair_mlc_chord(focused_sides: int, above: bool) -> None:
    """mlc_leaf_path_length is mlc_path_length of one pair with the other tip retracted."""
    from pyradmc.geometry.collimation import mlc_leaf_path_length

    rng = np.random.default_rng(SEED + 2)
    edges = np.array([0.0, 2.0])
    tip = 5.25
    neg = np.array([-1.0e4 if above else tip])
    pos = np.array([tip if above else 1.0e4])
    for p_u, p_v, p_w, d_u, d_v, d_w in _rays(2000, rng, spread=3.0):
        ray = (p_u, p_v, p_w, d_u, d_v, d_w)
        full = mlc_path_length(
            *ray, Z_TOP, Z_BOTTOM, TIP_RADIUS, 1, 0, 0, edges, neg, pos, focused_sides, 0
        )
        leaf = mlc_leaf_path_length(
            *ray, Z_TOP, Z_BOTTOM, TIP_RADIUS, 0.0, 2.0, tip, focused_sides, 1 if above else 0
        )
        assert leaf == pytest.approx(full, rel=1e-12, abs=1e-12)
