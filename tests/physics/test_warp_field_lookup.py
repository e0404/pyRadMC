"""The magnetic-field lookup, compiled under Warp, against its host values.

``field_at`` is one source for the reference loop (NumPy, float64) and the Warp
kernels (``@wp.func``, float32, ``wp.array3d(dtype=wp.vec3)``). These tests drive
the **compiled** lookup through a throwaway kernel on each device and compare with
the host function evaluated on the same float32-rounded samples: a same-formula
precision check, not the cross-target dose equality AGENTS.md 2.3 forbids.

Covered per device: all three modes; the flat clamp beyond the last centre and
outside the map; a degenerate (``n == 1``) axis; cell-centre equality. Points sit at
fixed fractions of a cell, away from faces, so float32 index rounding cannot flip a
nearest-mode cell between host and device (the trilinear lerp is continuous there).
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

# float32-exact geometry so the host and device index arithmetic see the same box.
SPACING = (0.5, 0.25, 1.0)
ORIGIN = (-1.0, 2.0, 0.0)
FRACTIONS = (0.1, 0.37, 0.5, 0.73, 0.9)
RTOL = 1e-5
ATOL = 1e-6


def _fields():
    from pyradmc.geometry.field import MagneticField
    from pyradmc.geometry.grid import VoxelGrid

    rng = np.random.default_rng(SEED)
    grid = VoxelGrid.uniform_water(shape=(4, 6, 5), spacing=SPACING, origin=ORIGIN)
    values = rng.normal(0.0, 1.0, size=(*grid.shape, 3))
    return {
        "uniform": MagneticField.uniform((0.1, -0.2, 1.5)),
        "nearest": MagneticField.on_grid(grid, values),
        "trilinear": MagneticField(values=values, spacing=SPACING, origin=ORIGIN),
        "degenerate": MagneticField(
            values=rng.normal(size=(1, 6, 1, 3)), spacing=(2.0, 0.5, 3.0), origin=(0.0, -1.0, 0.0)
        ),
    }


def _points(field) -> np.ndarray:
    """Interior points at fixed cell fractions, every cell centre, and far-outside points."""
    lo = np.asarray(field.origin)
    sp = np.asarray(field.spacing)
    n = np.asarray(field.shape)
    rng = np.random.default_rng(SEED + 1)
    cells = rng.integers(0, n, size=(200, 3))
    fracs = rng.choice(FRACTIONS, size=(200, 3))
    interior = lo + (cells + fracs) * sp
    ii, jj, kk = np.meshgrid(*(np.arange(m) for m in n), indexing="ij")
    centres = lo + (np.stack([ii, jj, kk], axis=-1).reshape(-1, 3) + 0.5) * sp
    hi = lo + n * sp
    outside = np.array(
        [
            lo - 3.0 * sp,
            hi + 3.0 * sp,
            [lo[0] - 1.0, lo[1] + 0.37 * sp[1], hi[2] + 2.0],
            [hi[0] + 5.0, lo[1] - 2.0, lo[2] + 0.73 * sp[2]],
            lo + (n - 0.1) * sp,  # between the last centre and the upper face
            lo + 0.1 * sp,  # between the lower face and the first centre
        ]
    )
    return np.concatenate([interior, centres, outside]).astype(np.float64)


def _host_expected(field, points: np.ndarray) -> np.ndarray:
    """``field_at`` on the host over the float32-rounded samples the device holds."""
    from pyradmc.geometry.field import field_at

    values = field.values.astype(np.float32).astype(np.float64)
    b0 = values[0, 0, 0]
    out = np.empty_like(points)
    for row, (x, y, z) in enumerate(points):
        out[row] = field_at(
            values,
            field.mode,
            float(b0[0]),
            float(b0[1]),
            float(b0[2]),
            *field.origin,
            *field.spacing,
            *field.shape,
            x,
            y,
            z,
        )
    return out


@pytest.fixture(scope="module", params=DEVICES)
def device_lookup(request):
    """The compiled ``field_at`` launched over each field's point set on one device."""
    device = request.param
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    from pyradmc.backends.warp.physics import warp_physics

    lookup = warp_physics().field_at

    @wp.kernel
    def probe(
        values: wp.array3d(dtype=wp.vec3),
        mode: int,
        b0x: float,
        b0y: float,
        b0z: float,
        ox: float,
        oy: float,
        oz: float,
        sx: float,
        sy: float,
        sz: float,
        nx: int,
        ny: int,
        nz: int,
        px: wp.array(dtype=float),
        py: wp.array(dtype=float),
        pz: wp.array(dtype=float),
        out: wp.array2d(dtype=float),
    ):
        tid = wp.tid()
        bx, by, bz = lookup(
            values,
            mode,
            b0x,
            b0y,
            b0z,
            ox,
            oy,
            oz,
            sx,
            sy,
            sz,
            nx,
            ny,
            nz,
            px[tid],
            py[tid],
            pz[tid],
        )
        out[tid, 0] = bx
        out[tid, 1] = by
        out[tid, 2] = bz

    results = {}
    for name, field in _fields().items():
        points = _points(field)
        if name == "uniform":
            # The array must not be read: poison it.
            host_values = np.full((1, 1, 1, 3), np.nan, dtype=np.float32)
        else:
            host_values = field.values.astype(np.float32)
        values = wp.array(host_values, dtype=wp.vec3, device=device)
        b0 = field.values[0, 0, 0]
        coords = [
            wp.array(points[:, a].astype(np.float32), dtype=float, device=device) for a in range(3)
        ]
        out = wp.zeros((len(points), 3), dtype=float, device=device)
        wp.launch(
            probe,
            dim=len(points),
            inputs=[
                values,
                int(field.mode),
                float(b0[0]),
                float(b0[1]),
                float(b0[2]),
                *(float(v) for v in field.origin),
                *(float(v) for v in field.spacing),
                *(int(v) for v in field.shape),
                *coords,
            ],
            outputs=[out],
            device=device,
        )
        wp.synchronize_device(device)
        results[name] = (field, points, out.numpy().astype(np.float64))
    return results


@pytest.mark.parametrize("name", ["uniform", "nearest", "trilinear", "degenerate"])
def test_compiled_lookup_matches_host(device_lookup, name: str) -> None:
    field, points, got = device_lookup[name]
    want = _host_expected(field, points)
    assert np.all(np.isfinite(got)), "device lookup produced non-finite values"
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


def test_uniform_mode_ignores_the_array(device_lookup) -> None:
    field, _points, got = device_lookup["uniform"]
    np.testing.assert_array_equal(got, np.tile(np.float32(field.values[0, 0, 0]), (len(got), 1)))


def test_nearest_mode_reads_whole_samples(device_lookup) -> None:
    """Every nearest-mode result is one of the uploaded vec3 samples, bit for bit."""
    field, _points, got = device_lookup["nearest"]
    samples = field.values.astype(np.float32).astype(np.float64).reshape(-1, 3)
    for row in got:
        assert np.any(np.all(samples == row, axis=1))


def test_clamp_outside_the_map_returns_edge_cells(device_lookup) -> None:
    field, points, got = device_lookup["trilinear"]
    values = field.values.astype(np.float32).astype(np.float64)
    lo = np.asarray(field.origin)
    sp = np.asarray(field.spacing)
    hi = lo + np.asarray(field.shape) * sp
    below = np.all(points < lo, axis=1)
    above = np.all(points >= hi, axis=1)
    assert below.sum() >= 1 and above.sum() >= 1
    np.testing.assert_allclose(
        got[below], np.broadcast_to(values[0, 0, 0], got[below].shape), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        got[above], np.broadcast_to(values[-1, -1, -1], got[above].shape), rtol=RTOL, atol=ATOL
    )


def test_degenerate_axes_vary_along_the_long_axis_only(device_lookup) -> None:
    field, points, got = device_lookup["degenerate"]
    values = field.values.astype(np.float32).astype(np.float64)
    cy = field.origin[1] + (np.arange(field.shape[1]) + 0.5) * field.spacing[1]
    want = np.stack([np.interp(points[:, 1], cy, values[0, :, 0, c]) for c in range(3)], axis=1)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
