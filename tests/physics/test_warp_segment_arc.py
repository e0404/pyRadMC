"""The segmented-fan generators with one fan and one frame per segment: deterministic pins.

A segment of an arc is its own fan through the stack in its own frame. Read back from
the queue, on each device:

- the device transmission of every segment, on rays of that segment's own fan,
  equals the host's (float64, the segment's own rotated stack) at the float32 chord
  tolerance of ``test_warp_segment_transmission``;
- ``generate_segmented_fan`` over an arc writes, for every row of segment ``s``,
  **bit for bit** what it writes for a single-fan source of ``fans[s]`` through
  ``stacks[s]`` alone under the same stream key — the one-launch arc is the
  per-angle launches it replaces, history for history — with the column tag
  ``t // per_batch`` at ``group_start > 0``, each row's energy and direction the
  host sampler of *its segment's* fan on the replayed uniforms, and its weight the
  host transmission of that segment;
- a 90-degree gantry segment attenuates through its *own* rotated leaves: with one
  leaf bank closed, a ray's weight follows the side of its own frame's ``u`` axis
  it passes (a kernel reading another segment's frame would see the slab edge-on
  and give every ray weight 1);
- ``generate_segmented_plan`` draws the segment from the first uniform, then that
  segment's fan from the next four, at that segment's transmission.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import MLC, BeamLimitingStack, JawPair
from pyradmc.geometry.segments import SegmentedFanSource
from tests.conftest import SEED
from tests.physics.test_warp_segment_generators import _assert_weights, _read, _Replay
from tests.unit.test_segmented_arc_source import arc_fan, arc_frame

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

# (gantry, collimator, couch) per segment, degrees. Segment 0 (the reference index a
# mis-indexed read would fall back to) is plain; segment 1 a 90-degree gantry; segment 2
# collimator-rotated (u, v turned about w); segment 3 collimator-rotated and
# non-coplanar (its focal spot leaves the gantry plane: focal-y, frame origin-y and
# every axis component differ from segment 0's), with its own field: off the
# isocentre and narrower. So every component of every per-segment read (frame
# origin, rotation, focal point, field centre, aperture axes, widths) differs from
# segment 0's somewhere, and a read of segment 0's value for any one of them is seen.
ORIENTATIONS = ((0.0, 0.0, 0.0), (90.0, 0.0, 0.0), (200.0, 45.0, 0.0), (315.0, 90.0, 30.0))
FIELDS = {3: dict(width_u=7.0, width_v=6.0, offset=(0.8, -0.6))}
PER_BATCH = 1_500
N_PER = 6_000
GROUP_START = 1
GROUP = 3  # segments 1..3 of 4
BATCH = 2


def _require(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")


def _stack(
    orientation: tuple[float, float, float],
    jaw,
    tips_neg,
    tips_pos,
    density: float = 8.0,
    mlc_density: float = 10.0,
) -> BeamLimitingStack:
    return BeamLimitingStack(
        frame=arc_frame(*orientation),
        devices=(
            JawPair("u", 28.0, 35.0, jaw[0], jaw[1], material=WATER, density=density),
            JawPair("v", 36.0, 43.0, -2.5, 2.5, material=WATER, density=density, focused=True),
            MLC(
                z_top=48.0,
                z_bottom=54.0,
                leaf_edges_v=(-3.0, -1.0, 1.0, 3.0),
                tips_neg=tuple(tips_neg),
                tips_pos=tuple(tips_pos),
                tip_radius=10.0,
                material=WATER,
                density=mlc_density,
                focused_sides=True,
            ),
        ),
    )


SETTINGS = (
    ((-2.0, 2.0), (-1.0, 0.5, -1.0), (1.0, 0.5, 1.0)),
    ((-0.5, 1.5), (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)),
    ((-3.0, 0.5), (0.0, -0.5, -1.5), (0.3, 0.5, 1.5)),
    ((-1.5, 2.5), (-3.0, 0.0, 0.0), (3.0, 0.0, 0.2)),
)


def _stacks() -> list[BeamLimitingStack]:
    return [_stack(o, *s) for o, s in zip(ORIENTATIONS, SETTINGS, strict=True)]


def _fan(orientation: tuple[float, float, float], field: dict | None = None):
    gantry, collimator, couch = orientation
    return arc_fan(gantry, collimator=collimator, couch=couch, **(field or {}))


def _arc() -> SegmentedFanSource:
    return SegmentedFanSource.from_stacks(
        [_fan(o, FIELDS.get(s)) for s, o in enumerate(ORIENTATIONS)],
        _stacks(),
        AnalyticCrossSections(),
    )


def _launch_fan(source, device: str, group_start: int, group: int, correlated: int):
    from pyradmc.backends.warp.engine import _upload_queue
    from pyradmc.backends.warp.segments import generate_segmented_fan, upload_segmented_fan

    upload = upload_segmented_fan(source, device)
    n = group * PER_BATCH
    slots = wp.zeros(n, dtype=wp.uint32, device=device)
    q = _upload_queue(n, device)
    emitted = wp.zeros(1, dtype=wp.int64, device=device)
    wp.launch(
        generate_segmented_fan,
        dim=n,
        inputs=[
            SEED,
            group_start,
            N_PER,
            PER_BATCH,
            BATCH,
            0,
            correlated,
            upload.edges,
            upload.cdf,
            int(upload.cdf.shape[0]),
            upload.stack,
            upload.segments,
            upload.mu,
            slots,
            emitted,
            q,
        ],
        device=device,
    )
    return _read(q, n), float(emitted.numpy()[0]) * 1.0e-9


@wp.kernel
def _uniforms(seed: int, keys: wp.array(dtype=int), out: wp.array2d(dtype=float)):
    """Replay the first five uniforms of stream ``keys[tid]``."""
    tid = wp.tid()
    state = wp.rand_init(seed, keys[tid])
    for k in range(5):
        out[tid, k] = wp.randf(state)


def _replayed(device: str, keys: np.ndarray) -> np.ndarray:
    out = wp.zeros((keys.size, 5), dtype=float, device=device)
    k = wp.array(keys.astype(np.int32), dtype=int, device=device)
    wp.launch(_uniforms, dim=keys.size, inputs=[SEED, k, out], device=device)
    return out.numpy().astype(np.float64)


def _fan_scalars(fan) -> list[float]:
    return [*fan.focal_point, *fan.center, *fan.u_axis, *fan.v_axis, fan.width_u, fan.width_v]


@pytest.mark.parametrize("device", DEVICES)
def test_device_transmission_matches_host_per_arc_segment(device: str) -> None:
    _require(device)
    from pyradmc.backends.warp.segments import segment_transmission_kernel, upload_segmented_fan

    source = _arc()
    upload = upload_segmented_fan(source, device)
    n = 2_000
    for s in range(source.n_segments):
        batch = source.fans[s].sample_batch(SEED + s, 0, n)
        energies = batch["energy"].astype(np.float64)
        origins = np.stack([batch["x"], batch["y"], batch["z"]], axis=1).astype(np.float64)
        directions = np.stack([batch["ux"], batch["uy"], batch["uz"]], axis=1).astype(np.float64)

        def col(values: np.ndarray) -> wp.array:
            return wp.array(values.astype(np.float32), dtype=float, device=device)

        out = wp.zeros((n, source.n_segments), dtype=float, device=device)
        wp.launch(
            segment_transmission_kernel,
            dim=(n, source.n_segments),
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
        got = out.numpy()[:, s].astype(np.float64)
        host = source.transmission(s, energies, origins, directions)
        assert host.max() > 0.9
        assert host.min() < 0.05
        np.testing.assert_allclose(got, host, rtol=2.0e-3, atol=1.0e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlated", [0, 1])
def test_arc_generator_is_the_per_angle_generators(device: str, correlated: int) -> None:
    """Row for row, segment s of the arc is the single-fan source of fans[s] alone."""
    _require(device)
    arc = _arc()
    got, book = _launch_fan(arc, device, GROUP_START, GROUP, correlated)
    np.testing.assert_array_equal(got["beamlet"], np.arange(GROUP * PER_BATCH) // PER_BATCH)
    segment = GROUP_START + got["beamlet"]
    for s in range(GROUP_START, GROUP_START + GROUP):
        alone = SegmentedFanSource.from_stacks(
            arc.fans[s], [_stacks()[s]] * (s + 1), AnalyticCrossSections()
        )  # segment s of a single-fan source: the same stream key in both modes
        want, _ = _launch_fan(alone, device, s, 1, correlated)
        rows = segment == s
        for name in ("kind", "primary", "energy", "weight", "x", "y", "z", "ux", "uy", "uz", "rng"):
            np.testing.assert_array_equal(got[name][rows], want[name], err_msg=f"{s} {name}")
    _assert_weights(arc, segment, got)
    expected = float(np.sum(got["weight"].astype(np.float64) * got["energy"]))
    assert book == pytest.approx(expected, rel=1e-6)

    # Each row's ray is its own segment's fan on the replayed uniforms.
    t = np.arange(GROUP * PER_BATCH)
    r = BATCH * PER_BATCH + (t - got["beamlet"] * PER_BATCH)
    keys = r if correlated else segment * N_PER + r
    u = _replayed(device, keys)
    spectrum = arc.spectrum
    from pyradmc.geometry.fan import fan_primary

    for i in range(0, t.size, 7):
        fan = arc.fans[int(segment[i])]
        host = fan_primary(
            _Replay(u[i, :4]), spectrum.edges, spectrum.cdf, spectrum.cdf.size, *_fan_scalars(fan)
        )
        assert got["energy"][i] == pytest.approx(host[0], rel=1e-6)
        np.testing.assert_allclose(
            [got["ux"][i], got["uy"][i], got["uz"][i]], host[1:], rtol=0.0, atol=1e-5
        )
        np.testing.assert_allclose(
            [got["x"][i], got["y"][i], got["z"][i]], fan.focal_point, rtol=1e-6, atol=1e-5
        )


@pytest.mark.parametrize("device", DEVICES)
def test_a_90_degree_segment_attenuates_through_its_own_rotated_leaves(device: str) -> None:
    """One leaf bank closed: the weight follows the side of the segment's own u axis."""
    _require(device)
    angles = (0.0, 90.0)
    stacks = [
        _stack((a, 0.0, 0.0), (-4.0, 4.0), (0.0, 0.0, 0.0), (3.0, 3.0, 3.0), mlc_density=40.0)
        for a in angles
    ]
    source = SegmentedFanSource.from_stacks(
        [arc_fan(a) for a in angles], stacks, AnalyticCrossSections()
    )
    got, _ = _launch_fan(source, device, 0, 2, 1)
    segment = got["beamlet"]
    directions = np.stack([got["ux"], got["uy"], got["uz"]], axis=1).astype(np.float64)
    for s in (0, 1):
        rows = segment == s
        frame = source.frames[s]
        _, d = frame.to_local(np.zeros((int(rows.sum()), 3)), directions[rows])
        u_mid = 51.0 * d[:, 0] / d[:, 2]  # leaf-travel coordinate at the MLC mid-plane
        v_mid = 51.0 * d[:, 1] / d[:, 2]  # leaf-width coordinate there
        clear = np.abs(np.abs(v_mid) - 1.0) > 0.15  # clear of the leaf-pair boundaries
        blocked = clear & (u_mid < -0.5)
        open_ = clear & (u_mid > 0.5)
        assert blocked.sum() > 100 and open_.sum() > 100
        w = got["weight"][rows]
        assert np.all(w[blocked] < 0.05), f"segment {s}: a closed-bank ray came through"
        assert np.all(w[open_] > 0.5), f"segment {s}: an open ray was blocked"


@pytest.mark.parametrize("device", DEVICES)
def test_arc_plan_generator_draws_the_segment_then_its_fan(device: str) -> None:
    _require(device)
    from pyradmc.backends.warp.engine import _upload_queue
    from pyradmc.backends.warp.segments import generate_segmented_plan, upload_segmented_fan
    from pyradmc.geometry.fan import fan_primary

    arc = _arc()
    plan = arc.plan([1.0, 2.0, 0.0, 3.0])
    upload = upload_segmented_fan(arc, device)
    selection = wp.array(plan.selection_cdf.astype(np.float32), dtype=float, device=device)
    n = 4_000
    offset = 700
    slots = wp.zeros(n, dtype=wp.uint32, device=device)
    q = _upload_queue(n, device)
    emitted = wp.zeros(1, dtype=wp.int64, device=device)
    wp.launch(
        generate_segmented_plan,
        dim=n,
        inputs=[
            SEED,
            offset,
            selection,
            arc.n_segments,
            upload.edges,
            upload.cdf,
            int(upload.cdf.shape[0]),
            upload.stack,
            upload.segments,
            upload.mu,
            slots,
            emitted,
            q,
        ],
        device=device,
    )
    got = _read(q, n)
    u = _replayed(device, offset + np.arange(n))
    cdf32 = plan.selection_cdf.astype(np.float32)
    segment = np.minimum(np.searchsorted(cdf32, u[:, 0].astype(np.float32), side="right"), 3)
    assert 2 not in set(segment.tolist())
    np.testing.assert_array_equal(got["beamlet"], 0)
    spectrum = arc.spectrum
    for i in range(n):
        fan = arc.fans[int(segment[i])]
        host = fan_primary(
            _Replay(u[i, 1:]), spectrum.edges, spectrum.cdf, spectrum.cdf.size, *_fan_scalars(fan)
        )
        assert got["energy"][i] == pytest.approx(host[0], rel=1e-6)
        np.testing.assert_allclose(
            [got["ux"][i], got["uy"][i], got["uz"][i]], host[1:], rtol=0.0, atol=1e-5
        )
        np.testing.assert_allclose(
            [got["x"][i], got["y"][i], got["z"][i]], fan.focal_point, rtol=1e-6, atol=1e-5
        )
    _assert_weights(arc, segment, got)
