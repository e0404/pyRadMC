"""A flattened, uploaded ``BeamLimitingStack`` reproduces ``path_lengths`` on device.

Slice between the per-device geometry twins (``test_warp_collimation_geometry``)
and the device pre-solve: it checks that the frame transform, the per-device
jaw/MLC dispatch, and the shared MLC leaf-array addressing all agree with the host
:meth:`BeamLimitingStack.path_lengths` — through a **rotated** frame and a stack
mixing a straight jaw, a focused jaw and a staircase MLC, so every branch of
``device_path_length`` is exercised. Same-formula float32 precision (AGENTS.md 2.3),
not dose equality; infinite chords must fall on the same rays.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyRadMC.data.materials import TUNGSTEN
from pyRadMC.geometry.collimation import (
    MLC,
    BeamFrame,
    BeamLimitingStack,
    JawPair,
)

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

N = 4000
RTOL = 2.0e-3
ATOL = 2.0e-3


def _rotated_stack() -> BeamLimitingStack:
    s = 1.0 / math.sqrt(2.0)
    frame = BeamFrame(
        origin=(2.0, -1.0, 3.0),
        u_axis=(s, s, 0.0),
        v_axis=(-s, s, 0.0),
        w_axis=(0.0, 0.0, 1.0),
    )
    return BeamLimitingStack(
        frame=frame,
        devices=(
            JawPair(
                axis="u",
                z_top=28.0,
                z_bottom=35.0,
                edge_neg=-2.0,
                edge_pos=2.0,
                material=TUNGSTEN,
                density=18.0,
            ),
            JawPair(
                axis="v",
                z_top=36.0,
                z_bottom=43.0,
                edge_neg=-2.5,
                edge_pos=2.5,
                material=TUNGSTEN,
                density=17.5,
                focused=True,
            ),
            MLC(
                z_top=48.0,
                z_bottom=52.0,
                leaf_edges_v=(-3.0, -1.0, 1.0, 3.0),
                tips_neg=(-1.0, 0.5, -1.0),
                tips_pos=(1.0, 0.5, 1.0),
                tip_radius=10.0,
                material=TUNGSTEN,
                density=18.0,
            ),
        ),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("from_origin", [0, 1])
def test_uploaded_stack_matches_host_path_lengths(device: str, from_origin: int) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyRadMC.backends.warp.presolve import stack_path_lengths_kernel, upload_stack

    stack = _rotated_stack()
    rng = np.random.default_rng(7)
    # Engine-frame rays spanning the head, mostly forward-going toward the phantom.
    origins = rng.uniform([-6.0, -6.0, -2.0], [10.0, 6.0, 60.0], size=(N, 3))
    directions = rng.normal([0.0, 0.0, 3.0], 1.0, size=(N, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    host = stack.path_lengths(origins, directions, from_origin=bool(from_origin))

    uploaded = upload_stack(stack.flatten(), device)
    cols = [
        wp.array(origins[:, 0].astype(np.float32), dtype=float, device=device),
        wp.array(origins[:, 1].astype(np.float32), dtype=float, device=device),
        wp.array(origins[:, 2].astype(np.float32), dtype=float, device=device),
        wp.array(directions[:, 0].astype(np.float32), dtype=float, device=device),
        wp.array(directions[:, 1].astype(np.float32), dtype=float, device=device),
        wp.array(directions[:, 2].astype(np.float32), dtype=float, device=device),
    ]
    out = wp.zeros((N, stack.flatten().n_devices), dtype=float, device=device)
    wp.launch(
        stack_path_lengths_kernel,
        dim=N,
        inputs=[uploaded, *cols, from_origin],
        outputs=[out],
        device=device,
    )
    wp.synchronize_device(device)
    got = out.numpy()

    np.testing.assert_array_equal(
        np.isinf(got), np.isinf(host), err_msg=f"{device}: infinite chords on different rays"
    )
    finite = np.isfinite(host)
    np.testing.assert_allclose(got[finite], host[finite], rtol=RTOL, atol=ATOL)
