"""The device segment transmission reproduces the host one, segment by segment.

:func:`~pyradmc.backends.warp.segments.segment_transmission` is the weight the
segmented-fan generator gives each primary: one uploaded stack layout plus the
per-segment jaw-edge and leaf-tip columns, through the pre-solve's collimation
twins and attenuation table. Checked against
:meth:`~pyradmc.geometry.segments.SegmentedFanSource.transmission` (host, float64)
on random rays through a **rotated** frame with a straight jaw, a focused jaw and a
focused-side MLC whose settings differ per segment — so the per-segment indexing of
both setting kinds is exercised. Same-formula float32 precision (AGENTS.md 2.3), the
tolerance of the pre-solve's attenuation pin; dense "water" devices keep the
attenuation strong without EPDL data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

N = 3000
RTOL = 2.0e-3
ATOL = 1.0e-6
FOCAL = (2.0, -1.0, 3.0)


def _source() -> SegmentedFanSource:
    s = 1.0 / math.sqrt(2.0)
    frame = BeamFrame(origin=FOCAL, u_axis=(s, s, 0.0), v_axis=(-s, s, 0.0))
    template = BeamLimitingStack(
        frame=frame,
        devices=(
            JawPair("u", 28.0, 35.0, -2.0, 2.0, material=WATER, density=8.0),
            JawPair("v", 36.0, 43.0, -2.5, 2.5, material=WATER, density=8.0, focused=True),
            MLC(
                z_top=48.0,
                z_bottom=54.0,
                leaf_edges_v=(-3.0, -1.0, 1.0, 3.0),
                tips_neg=(-1.0, 0.5, -1.0),
                tips_pos=(1.0, 0.5, 1.0),
                tip_radius=10.0,
                material=WATER,
                density=10.0,
                focused_sides=True,
            ),
        ),
    )
    fan = SpectralBeamSource(
        Spectrum((0.5, 1.0, 2.0, 6.0), (1.0, 2.0, 1.0)),
        focal_point=FOCAL,
        center=(2.0, -1.0, 103.0),
        width_u=10.0,
        width_v=10.0,
        u_axis=frame.u_axis,
        v_axis=frame.v_axis,
    )
    return SegmentedFanSource(
        fan,
        template,
        AnalyticCrossSections(),
        jaw_neg=[[-2.0, -2.5], [-0.5, -1.0], [-3.0, 0.0], [0.0, -2.0]],
        jaw_pos=[[2.0, 2.5], [1.5, 2.0], [-1.0, 3.0], [0.0, 2.0]],
        tips_neg=[[-1.0, 0.5, -1.0], [-2.0, -2.0, -2.0], [0.0, -0.5, -1.5], [-3.0, 0.0, 0.0]],
        tips_pos=[[1.0, 0.5, 1.0], [2.0, 2.0, 2.0], [0.3, 0.5, 1.5], [3.0, 0.0, 0.2]],
    )


@pytest.mark.parametrize("device", DEVICES)
def test_device_transmission_matches_host_per_segment(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.segments import segment_transmission_kernel, upload_segmented_fan

    source = _source()
    rng = np.random.default_rng(11)
    # Fan rays from the focal spot through the field: the generator's own geometry.
    batch = source.fan.sample_batch(int(rng.integers(1 << 30)), 0, N)
    energies = batch["energy"].astype(np.float64)
    origins = np.stack([batch["x"], batch["y"], batch["z"]], axis=1).astype(np.float64)
    directions = np.stack([batch["ux"], batch["uy"], batch["uz"]], axis=1).astype(np.float64)
    host = np.stack(
        [source.transmission(s, energies, origins, directions) for s in range(source.n_segments)],
        axis=1,
    )
    assert host.min() < 0.05 and host.max() > 0.9  # blocked and open rays both present

    upload = upload_segmented_fan(source, device)

    def col(values: np.ndarray) -> wp.array:
        return wp.array(values.astype(np.float32), dtype=float, device=device)

    out = wp.zeros((N, source.n_segments), dtype=float, device=device)
    wp.launch(
        segment_transmission_kernel,
        dim=(N, source.n_segments),
        inputs=[
            upload.stack,
            upload.segments,
            upload.mu,
            col(energies),
            *(col(origins[:, k]) for k in range(3)),
            *(col(directions[:, k]) for k in range(3)),
        ],
        outputs=[out],
        device=device,
    )
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), host, rtol=RTOL, atol=ATOL)
