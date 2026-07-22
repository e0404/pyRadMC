r"""The voxel-boundary distance is non-negative — including under float32.

Regression pin for a measured GPU transport stall (2026-07-22): in float32, a
position can land in the sub-ulp band around a voxel face where
``point_axis_index`` rounds *up* to the voxel whose recomputed lower face sits
half an ulp above the position. Moving toward that face, the exit distance
``(face - coord) / u`` then comes out **negative**, and a grazing direction
amplifies the half-ulp face error by ``1/|u|`` — measured -6e-5 at
``|uz| = 0.0088``, overwhelming the +1e-4 boundary nudge. The electron loop's
substep became negative: zero energy loss (the clamped range map), a frozen
position (sub-ulp moves), and — under the GS model, whose zero-step guard
returns exactly forward — a frozen direction: a self-sustaining fixed point
that held single GPU lanes for ~1e6+ iterations (one 191-block launch measured
at 126.9 s against a 10.8 ms median). The Gaussian hinge escaped the same state
only by accident: ``sqrt`` of its negative mean-square angle is NaN, which
fails ``point_inside`` and terminates the lane.

The distance to the exit face of the voxel a point is assigned to is
non-negative *by definition*; a negative return is float rounding, clamped at
the primitive. These tests drive the **Warp-compiled** primitive (float32, the
arithmetic that bites) with the dumped pathological state verbatim, plus a fuzz
of grazing near-face states, on each device.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

# The thorax grid of the measured stall: origin 0, spacing 3 mm, (128, 64, 80).
SPACING = 0.3
NX, NY, NZ = 128, 64, 80

# The trapped lane's state, dumped from the stalled kernel (float32 verbatim).
TRAPPED_POSITION = (17.48038, 11.789452, 13.5)
TRAPPED_DIRECTION = (-0.20822741, 0.9780409, -0.008783912)


@pytest.fixture(scope="module", params=DEVICES)
def boundary_distances(request):
    """The compiled primitive over the trapped state and a grazing near-face fuzz."""
    device = request.param
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    from pyRadMC.backends.warp.physics import warp_physics

    distance_fn = warp_physics().distance_to_voxel_boundary

    @wp.kernel
    def probe(
        px: wp.array(dtype=float),
        py: wp.array(dtype=float),
        pz: wp.array(dtype=float),
        dx: wp.array(dtype=float),
        dy: wp.array(dtype=float),
        dz: wp.array(dtype=float),
        out: wp.array(dtype=float),
    ):
        tid = wp.tid()
        out[tid] = distance_fn(
            px[tid],
            py[tid],
            pz[tid],
            dx[tid],
            dy[tid],
            dz[tid],
            0.0,
            0.0,
            0.0,
            SPACING,
            SPACING,
            SPACING,
            NX,
            NY,
            NZ,
        )

    rng = np.random.default_rng(SEED)
    n_fuzz = 200_000
    # Positions in the sub-ulp band around interior faces of every axis, paired
    # with grazing directions toward the face: the amplification regime.
    face_index = rng.integers(1, 60, size=(n_fuzz, 3))
    faces = face_index.astype(np.float64) * SPACING
    offsets = rng.uniform(-2e-6, 2e-6, size=(n_fuzz, 3))
    positions = (faces + offsets).astype(np.float32)
    directions = rng.normal(size=(n_fuzz, 3))
    # Make one random axis grazing (|u| in 1e-4..2e-2) toward its face.
    axis = rng.integers(0, 3, size=n_fuzz)
    grazing = rng.uniform(1e-4, 2e-2, size=n_fuzz) * np.where(rng.random(n_fuzz) < 0.5, -1, 1)
    directions[np.arange(n_fuzz), axis] = grazing
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions.astype(np.float32)

    positions = np.vstack([np.array([TRAPPED_POSITION], dtype=np.float32), positions])
    directions = np.vstack([np.array([TRAPPED_DIRECTION], dtype=np.float32), directions])

    arrays = [
        wp.array(column.copy(), dtype=float, device=device)
        for column in (*positions.T, *directions.T)
    ]
    out = wp.zeros(positions.shape[0], dtype=float, device=device)
    wp.launch(probe, dim=positions.shape[0], inputs=[*arrays, out], device=device)
    wp.synchronize_device(device)
    return out.numpy()


def test_trapped_state_gets_a_nonnegative_distance(boundary_distances: np.ndarray) -> None:
    """The exact dumped lane state: was -6e-5, must be >= 0."""
    assert boundary_distances[0] >= 0.0, (
        f"the measured trap state still yields a negative boundary distance "
        f"({boundary_distances[0]!r})"
    )


def test_distance_is_never_negative_near_faces(boundary_distances: np.ndarray) -> None:
    """Grazing near-face fuzz in float32: no state may see a negative distance.

    A negative here is not an accuracy nit: beyond the 1e-4 boundary nudge it
    makes the electron substep length negative, which is the fixed-point stall.
    """
    worst = float(boundary_distances.min())
    assert worst >= 0.0, f"negative boundary distance {worst!r} in the near-face fuzz"
