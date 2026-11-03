"""The pathwise aperture gradient: Warp's tape on the collimation geometry.

The collimation chords are the only aperture-dependent step of a collimated run
(the stack rescales weights; transport never sees the edge), so taping them is
the whole of what autodiff of the simulation can contribute. These tests pin the
taped chord derivative against host finite differences and then show what the
pathwise estimator built on it does: unbiased at a rounded tip, blind to a
straight edge.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyradmc.geometry.aperture_gradient import (
    EdgeParameter,
    PathwiseDerivativeSource,
    edge_value,
    isolated_block,
    isolated_parameter,
    with_edge,
)
from tests.conftest import SEED
from tests.unit.test_aperture_gradient import (
    JAW_NEG,
    JAW_POS,
    MLC_NEG,
    MLC_POS,
    XS,
    _beam,
    _mc_mean,
    _quadrature_derivative,
    _stack,
)

pytestmark = pytest.mark.warp


def _random_rays(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Oblique local-frame rays from above the stack aimed through the tip regions."""
    target = np.column_stack(
        [rng.uniform(0.0, 9.0, n), rng.uniform(0.05, 1.95, n), np.full(n, 53.0)]
    )
    origin = np.column_stack(
        [
            target[:, 0] + rng.normal(0.0, 3.0, n),
            target[:, 1] + rng.normal(0.0, 0.5, n),
            np.zeros(n),
        ]
    )
    direction = target - origin
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return origin, direction


@pytest.mark.parametrize("parameter", [MLC_POS, MLC_NEG, JAW_POS, JAW_NEG])
def test_taped_chord_gradient_matches_host_finite_difference(parameter: EdgeParameter) -> None:
    """d(chord)/d(edge) from the tape equals the float64 host central difference.

    Rays whose finite difference is not converged (they straddle a kink of the
    chord, where the derivative jumps) are excluded: the tape returns one
    one-sided derivative there and neither side is "the" answer.
    """
    from pyradmc.backends.warp.aperture_ad import edge_chord_gradients

    stack = _stack()
    x = edge_value(stack, parameter)
    alone = isolated_block(stack, parameter)
    origin, direction = _random_rays(4000, np.random.default_rng(SEED))
    chords, gradients = edge_chord_gradients(alone.devices[0], parameter.side, origin, direction)

    def host(value: float) -> np.ndarray:
        moved = with_edge(alone, isolated_parameter(parameter), value)
        return moved.devices[0].path_lengths(origin, direction)

    assert chords == pytest.approx(host(x), abs=2e-4)
    fd = (host(x + 1e-5) - host(x - 1e-5)) / 2e-5
    fd_half = (host(x + 5e-6) - host(x - 5e-6)) / 1e-5
    smooth = np.abs(fd - fd_half) < 1e-4 * (1.0 + np.abs(fd))
    assert smooth.sum() > 3000
    hit = smooth & (np.abs(fd) > 0.0)
    assert hit.sum() > 50  # the comparison is not vacuous
    np.testing.assert_allclose(gradients[smooth], fd[smooth], rtol=2e-3, atol=2e-3)


def test_tape_is_blind_to_a_straight_edge_under_a_parallel_beam() -> None:
    """Vertical rays and a straight jaw face: the chord is a step, the tape says zero."""
    from pyradmc.backends.warp.aperture_ad import edge_chord_gradients

    alone = isolated_block(_stack(), JAW_POS)
    q = np.linspace(4.0, 7.0, 3001)
    origin = np.column_stack([q, np.ones_like(q), np.zeros_like(q)])
    direction = np.tile([0.0, 0.0, 1.0], (q.size, 1))
    chords, gradients = edge_chord_gradients(alone.devices[0], "pos", origin, direction)
    assert chords.min() == 0.0 and chords.max() == pytest.approx(6.0)
    assert np.all(gradients == 0.0)


def test_pathwise_source_is_unbiased_at_a_rounded_tip() -> None:
    """The smooth part is all of dT/dx at a rounded tip: the mean is right."""
    source = PathwiseDerivativeSource(_beam(), _stack(), XS, MLC_POS)
    mean, sem = _mc_mean(source, edge_value(_stack(), MLC_POS), n=40_000)
    reference = _quadrature_derivative(MLC_POS, 1.0e-3)
    assert abs(mean - reference) < 4.0 * sem, (mean, sem, reference)


def test_pathwise_source_misses_the_straight_edge_entirely() -> None:
    """The boundary term is the whole derivative here, and the tape cannot see it."""
    source = PathwiseDerivativeSource(_beam(), _stack(), XS, JAW_POS)
    columns = source.sample_batch(SEED, 0, 5000)
    assert np.all(columns["weight"] == 0.0)
    assert _quadrature_derivative(JAW_POS, 1.0e-3) > 0.01
