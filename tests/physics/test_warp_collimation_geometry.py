"""The scalar collimation geometry twins, compiled under Warp, against the host.

The device pre-solve traces one ray per thread through the beam-limiting devices,
so it calls the ``@wp.func``-compiled ``jaw_path_length`` / ``mlc_path_length``
(the single-ray twins of :meth:`JawPair.path_lengths` / :meth:`MLC.path_lengths`).
This is a **same-formula float32 precision check** against the float64 host values
(the twins are already pinned equal to the vectorized host path in
``tests/unit/test_collimation.py``), not the cross-target dose equality AGENTS.md
2.3 forbids: a codegen or shim defect shows up as a chord that no longer matches
its own host evaluation. Infinite (parallel-trapped) chords must land on exactly
the same rays.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.geometry.collimation import jaw_path_length, mlc_path_length

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

N = 4000
# float32 on the geometry's divisions; the relative error is a uniform ~6e-5 across
# all chord magnitudes (measured), so one relative tolerance covers every ray.
RTOL = 2.0e-3
ATOL = 2.0e-3

Z_TOP, Z_BOTTOM = 40.0, 47.0
EDGE_NEG, EDGE_POS = -2.0, 2.0
MLC_Z_TOP, MLC_Z_BOTTOM = 48.0, 52.0
TIP_RADIUS = 10.0
LEAF_EDGES = np.array([-3.0, -1.0, 1.0, 3.0])
TIPS_NEG = np.array([-1.0, 0.5, -1.0])
TIPS_POS = np.array([1.0, 0.5, 1.0])


def _random_rays(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    origins = rng.uniform([-8.0, -8.0, 0.0], [8.0, 8.0, 60.0], size=(N, 3))
    directions = rng.normal(size=(N, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origins, directions


def _require_device(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")


def _compare(device: str, got: np.ndarray, host: np.ndarray) -> None:
    np.testing.assert_array_equal(
        np.isinf(got), np.isinf(host), err_msg=f"{device}: infinite chords on different rays"
    )
    finite = np.isfinite(host)
    np.testing.assert_allclose(got[finite], host[finite], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("focused", [0, 1])
def test_jaw_twin_matches_host_on_device(device: str, focused: int) -> None:
    _require_device(device)
    from pyRadMC.backends.warp.physics import warp_physics

    jaw = warp_physics().jaw_path_length

    @wp.kernel
    def probe(
        p_q: wp.array(dtype=float),
        d_q: wp.array(dtype=float),
        p_w: wp.array(dtype=float),
        d_w: wp.array(dtype=float),
        focused_flag: int,
        out: wp.array(dtype=float),
    ) -> None:
        tid = wp.tid()
        out[tid] = jaw(
            p_q[tid],
            d_q[tid],
            p_w[tid],
            d_w[tid],
            Z_TOP,
            Z_BOTTOM,
            EDGE_NEG,
            EDGE_POS,
            focused_flag,
            0,
        )

    origins, directions = _random_rays(seed=101)
    host = np.array(
        [
            jaw_path_length(
                origins[k, 0],
                directions[k, 0],
                origins[k, 2],
                directions[k, 2],
                Z_TOP,
                Z_BOTTOM,
                EDGE_NEG,
                EDGE_POS,
                focused,
                0,
            )
            for k in range(N)
        ]
    )
    cols = [
        wp.array(origins[:, 0].astype(np.float32), dtype=float, device=device),
        wp.array(directions[:, 0].astype(np.float32), dtype=float, device=device),
        wp.array(origins[:, 2].astype(np.float32), dtype=float, device=device),
        wp.array(directions[:, 2].astype(np.float32), dtype=float, device=device),
    ]
    out = wp.zeros(N, dtype=float, device=device)
    wp.launch(probe, dim=N, inputs=[*cols, focused], outputs=[out], device=device)
    wp.synchronize_device(device)
    _compare(device, out.numpy(), host)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("focused_sides", [0, 1])
def test_mlc_twin_matches_host_on_device(device: str, focused_sides: int) -> None:
    _require_device(device)
    from pyRadMC.backends.warp.physics import warp_physics

    mlc = warp_physics().mlc_path_length
    n_pairs = len(LEAF_EDGES) - 1

    @wp.kernel
    def probe(
        p_u: wp.array(dtype=float),
        p_v: wp.array(dtype=float),
        p_w: wp.array(dtype=float),
        d_u: wp.array(dtype=float),
        d_v: wp.array(dtype=float),
        d_w: wp.array(dtype=float),
        edges: wp.array(dtype=float),
        tips_neg: wp.array(dtype=float),
        tips_pos: wp.array(dtype=float),
        pairs: int,
        focused_flag: int,
        out: wp.array(dtype=float),
    ) -> None:
        tid = wp.tid()
        out[tid] = mlc(
            p_u[tid],
            p_v[tid],
            p_w[tid],
            d_u[tid],
            d_v[tid],
            d_w[tid],
            MLC_Z_TOP,
            MLC_Z_BOTTOM,
            TIP_RADIUS,
            pairs,
            0,
            0,
            edges,
            tips_neg,
            tips_pos,
            focused_flag,
            0,
        )

    origins, directions = _random_rays(seed=202)
    host = np.array(
        [
            mlc_path_length(
                origins[k, 0],
                origins[k, 1],
                origins[k, 2],
                directions[k, 0],
                directions[k, 1],
                directions[k, 2],
                MLC_Z_TOP,
                MLC_Z_BOTTOM,
                TIP_RADIUS,
                n_pairs,
                0,
                0,
                LEAF_EDGES,
                TIPS_NEG,
                TIPS_POS,
                focused_sides,
                0,
            )
            for k in range(N)
        ]
    )
    cols = [
        wp.array(origins[:, 0].astype(np.float32), dtype=float, device=device),
        wp.array(origins[:, 1].astype(np.float32), dtype=float, device=device),
        wp.array(origins[:, 2].astype(np.float32), dtype=float, device=device),
        wp.array(directions[:, 0].astype(np.float32), dtype=float, device=device),
        wp.array(directions[:, 1].astype(np.float32), dtype=float, device=device),
        wp.array(directions[:, 2].astype(np.float32), dtype=float, device=device),
    ]
    edges = wp.array(LEAF_EDGES.astype(np.float32), dtype=float, device=device)
    tips_neg = wp.array(TIPS_NEG.astype(np.float32), dtype=float, device=device)
    tips_pos = wp.array(TIPS_POS.astype(np.float32), dtype=float, device=device)
    out = wp.zeros(N, dtype=float, device=device)
    wp.launch(
        probe,
        dim=N,
        inputs=[*cols, edges, tips_neg, tips_pos, n_pairs, focused_sides],
        outputs=[out],
        device=device,
    )
    wp.synchronize_device(device)
    _compare(device, out.numpy(), host)
