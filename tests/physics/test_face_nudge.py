"""A particle that has reached a voxel face is carried across along the face normal.

The electron loop ends a chord on the face it reaches and then steps the particle a
micron into the neighbour. Taking that micron along the *direction* fails in float32
at grazing incidence: its normal component is ``1e-4 * |u_n|``, below one ulp of a
coordinate near 10 cm once ``|u_n|`` is a few 1e-3, so the coordinate does not change
and the particle is still on the face. With no in-voxel path left the next substep has
zero-length chords, a bent chord under a magnetic field is cut at the face again, and
the substep ends without the hinge or a rotation that could turn it away: a fixed
point. Measured 2026-10-18 on the water / air slab at 1.5 T: 7.6x slower on the GPU
(single lanes charging one carry micron per iteration until their energy was gone),
no slowdown at all in the float64 reference loop. Taking the micron along the normal
of the face the chord reached (``voxel_face_axis``, ``nudge_across_face``) is always
resolvable and changes the geometry by the same micron.

These tests pin the helpers on the host and as Warp-compiled float32 primitives, and
the float32 arithmetic fact that motivates them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc.geometry.grid import (
    distance_to_voxel_boundary,
    nudge_across_face,
    point_axis_index,
    voxel_face_axis,
)
from pyradmc.transport.electron import BOUNDARY_NUDGE_CM
from tests.conftest import SEED

GRID = (0.0, 0.0, 0.0, 0.5, 0.5, 0.1, 40, 40, 100)  # the slab of the measured stall


def _unit(rng: np.random.Generator) -> tuple[float, float, float]:
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    return float(v[0]), float(v[1]), float(v[2])


def test_face_axis_names_the_face_the_distance_measures_to() -> None:
    """The axis is the one whose exit distance is the minimum (first axis on ties)."""
    rng = np.random.default_rng(SEED)
    x_lo, y_lo, z_lo, sx, sy, sz = GRID[:6]
    for _ in range(2000):
        x, y, z = rng.uniform(0.0, 20.0), rng.uniform(0.0, 20.0), rng.uniform(0.0, 10.0)
        ux, uy, uz = _unit(rng)
        d = distance_to_voxel_boundary(x, y, z, ux, uy, uz, *GRID)
        axis = voxel_face_axis(x, y, z, ux, uy, uz, *GRID)
        p = [x + ux * d, y + uy * d, z + uz * d]
        lo, sp = (x_lo, y_lo, z_lo)[axis], (sx, sy, sz)[axis]
        # The landing coordinate on that axis is on a face plane (to rounding).
        k = (p[axis] - lo) / sp
        assert abs(k - round(k)) < 1e-9, (axis, k)


def test_face_axis_breaks_ties_like_the_distance() -> None:
    # Toward the corner of a voxel: all three distances equal; the first axis wins.
    assert voxel_face_axis(0.25, 0.25, 0.05, 1.0, 1.0, 0.2, *GRID) == 0
    # A direction with no x component never names x.
    assert voxel_face_axis(0.25, 0.25, 0.05, 0.0, 0.3, 0.95, *GRID) == 2
    assert voxel_face_axis(0.25, 0.25, 0.05, 0.0, 0.95, 0.001, *GRID) == 1


def test_nudge_is_along_the_normal_in_the_sense_of_travel() -> None:
    assert nudge_across_face(0, 0.3, 0.9, 0.1, 1e-4) == (1e-4, 0.0, 0.0)
    assert nudge_across_face(0, -0.3, 0.9, 0.1, 1e-4) == (-1e-4, 0.0, 0.0)
    assert nudge_across_face(1, 0.3, -0.9, 0.1, 1e-4) == (0.0, -1e-4, 0.0)
    assert nudge_across_face(2, 0.3, 0.9, 0.1, 1e-4) == (0.0, 0.0, 1e-4)
    # A negative length steps back behind the face (the terminal pull-back).
    assert nudge_across_face(2, 0.3, 0.9, -0.1, -1e-4) == (0.0, 0.0, 1e-4)


def test_normal_nudge_crosses_the_face_where_the_direction_nudge_cannot_in_float32() -> None:
    """The arithmetic fact behind the fix: at grazing incidence the direction-aligned
    micron is below a float32 ulp of the coordinate; the normal micron never is."""
    z_face = np.float32(4.0)  # a face of the slab (40 x 0.1 cm)
    u_n = 2.0e-3  # grazing: |u_z| of a gyrating electron near its tangent point
    along_direction = np.float32(z_face + np.float32(BOUNDARY_NUDGE_CM * u_n))
    assert along_direction == z_face, "the direction-aligned nudge must fail to move"
    dz = nudge_across_face(2, 0.7, 0.7, u_n, BOUNDARY_NUDGE_CM)[2]
    along_normal = np.float32(z_face + np.float32(dz))
    assert along_normal > z_face
    # ... and the point is then assigned to the upper voxel, as a crossing must be.
    assert point_axis_index(float(along_normal), 0.0, 0.1, 100) == 40
    assert point_axis_index(float(along_direction), 0.0, 0.1, 100) == 40  # on-face = upper


@pytest.mark.warp
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_compiled_helpers_match_the_host(device: str) -> None:
    """The Warp-compiled float32 twins name the same face and nudge the same way."""
    wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.physics import warp_physics

    p = warp_physics()
    axis_fn, nudge_fn = p.voxel_face_axis, p.nudge_across_face
    x_lo, y_lo, z_lo, sx, sy, sz, nx, ny, nz = GRID

    @wp.kernel
    def probe(
        pos: wp.array(dtype=wp.vec3),
        dirs: wp.array(dtype=wp.vec3),
        axes: wp.array(dtype=int),
        moved: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        q = pos[i]
        u = dirs[i]
        a = axis_fn(q[0], q[1], q[2], u[0], u[1], u[2], x_lo, y_lo, z_lo, sx, sy, sz, nx, ny, nz)
        axes[i] = a
        dx, dy, dz = nudge_fn(a, u[0], u[1], u[2], BOUNDARY_NUDGE_CM)
        moved[i] = wp.vec3(q[0] + dx, q[1] + dy, q[2] + dz)

    rng = np.random.default_rng(SEED)
    n = 4096
    # Interior points with their directions, as the loop asks the question (the
    # chord's start, before the move); a point *on* a face is the one place float32
    # cannot name the face unambiguously, and the loop never asks it there.
    pos = np.column_stack(
        [rng.uniform(0.0, 20.0, n), rng.uniform(0.0, 20.0, n), rng.uniform(0.0, 10.0, n)]
    )
    dirs = np.array([_unit(rng) for _ in range(n)])
    pos32 = pos.astype(np.float32)
    dirs32 = dirs.astype(np.float32)
    host_axes = np.array(
        [
            voxel_face_axis(*pos32[i].astype(np.float64), *dirs32[i].astype(np.float64), *GRID)
            for i in range(n)
        ]
    )
    axes = wp.zeros(n, dtype=int, device=device)
    moved = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(
        probe,
        dim=n,
        inputs=[
            wp.array(pos32, dtype=wp.vec3, device=device),
            wp.array(dirs32, dtype=wp.vec3, device=device),
            axes,
            moved,
        ],
        device=device,
    )
    wp.synchronize()
    compiled_axes = axes.numpy()
    # float32 may re-rank two nearly tied faces; away from ties the answers agree.
    tied = np.zeros(n, dtype=bool)
    for i in range(n):
        ds = []
        for ax in range(3):
            u = float(dirs32[i][ax])
            if u == 0.0:
                ds.append(math.inf)
                continue
            lo, sp, cnt = (x_lo, y_lo, z_lo)[ax], (sx, sy, sz)[ax], (nx, ny, nz)[ax]
            k = point_axis_index(float(pos32[i][ax]), lo, sp, cnt)
            face = lo + (k + 1) * sp if u > 0.0 else lo + k * sp
            ds.append((face - float(pos32[i][ax])) / u)
        ds = sorted(ds)
        tied[i] = ds[1] - ds[0] < 1e-4 * max(ds[0], 1e-3)
    agree = compiled_axes[~tied] == host_axes[~tied]
    assert agree.all(), f"{(~agree).sum()} disagreements away from ties"
    # The nudge moved each point by one micron along its face's normal and nothing else.
    m = moved.numpy().astype(np.float64)
    delta = m - pos32.astype(np.float64)
    for i in range(n):
        ax = compiled_axes[i]
        expect = np.zeros(3)
        expect[ax] = BOUNDARY_NUDGE_CM if dirs32[i][ax] > 0.0 else -BOUNDARY_NUDGE_CM
        assert np.allclose(delta[i], expect, atol=2e-6), (i, ax, delta[i], expect)
