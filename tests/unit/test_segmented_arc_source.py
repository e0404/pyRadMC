"""Segments with their own fan and beam frame: the control points of an arc as beamlets.

:class:`~pyradmc.geometry.segments.SegmentedFanSource` given one fan and one frame
per segment (a gantry angle each) instead of one shared fan: segment ``s`` is the
fan ``fans[s]`` through the stack in frame ``frames[s]`` at that segment's settings.
The independent oracle is again
:class:`~pyradmc.geometry.collimation.CollimatedSource` of that segment's own fan
and stack. Correlated replay means *the same uniforms drive every segment's own
fan*: one energy and one in-aperture offset, hence one beam's-eye-view ray in each
segment's frame. Malformed arcs — another spectrum, a frame off its fan's focal
spot, a fan not aligned with its frame, counts that disagree — are refused at
construction, naming the segment.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import (
    MLC,
    BeamFrame,
    BeamLimitingStack,
    CollimatedSource,
    JawPair,
)
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
ISO = (8.0, 8.0, 8.0)
SAD = 100.0
ANGLES = (0.0, 90.0, 225.0)
EDGES = (-3.0, -1.0, 1.0, 3.0)
# Per-segment settings: (jaw edge_neg, jaw edge_pos, tips_neg, tips_pos), mid-plane cm.
SETTINGS = (
    (-2.0, 2.0, (-1.5, -0.5, -1.5), (1.5, 0.5, 1.5)),
    (-1.0, 2.5, (-0.5, 0.0, -2.0), (0.5, 0.0, 0.2)),
    (-2.5, 0.5, (0.0, -1.0, -0.3), (0.0, 1.0, 0.3)),
)


Axes = tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]


def _rotate(vector: tuple[float, ...], axis: tuple[float, ...], deg: float) -> tuple[float, ...]:
    """Rodrigues rotation of ``vector`` about the unit ``axis`` by ``deg`` degrees."""
    a = math.radians(deg)
    k = np.asarray(axis, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    out = v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * float(k @ v) * (1 - math.cos(a))
    return tuple(float(c) for c in out)


def beam_axes(angle_deg: float, collimator: float = 0.0, couch: float = 0.0) -> Axes:
    """``(u, v, w)`` of a beam: ``w`` focal spot -> isocentre, ``u x v = w``.

    Gantry ``angle_deg`` about +y (``v`` = +y), then the collimator about ``w`` (turns
    ``u`` and ``v`` in the aperture plane), then a couch-like rotation of the whole
    triad about +z (tilts ``w`` out of the x-z plane: a non-coplanar beam whose focal
    spot leaves the gantry plane).
    """
    a = math.radians(angle_deg)
    u: tuple[float, ...] = (math.cos(a), 0.0, -math.sin(a))
    v: tuple[float, ...] = (0.0, 1.0, 0.0)
    w: tuple[float, ...] = (math.sin(a), 0.0, math.cos(a))
    if collimator:
        u, v = _rotate(u, w, collimator), _rotate(v, w, collimator)
    if couch:
        z = (0.0, 0.0, 1.0)
        u, v, w = _rotate(u, z, couch), _rotate(v, z, couch), _rotate(w, z, couch)
    return u, v, w


def focal(angle_deg: float, collimator: float = 0.0, couch: float = 0.0) -> tuple[float, ...]:
    w = beam_axes(angle_deg, collimator, couch)[2]
    return (ISO[0] - SAD * w[0], ISO[1] - SAD * w[1], ISO[2] - SAD * w[2])


def arc_fan(
    angle_deg: float,
    spectrum: Spectrum = SPECTRUM,
    *,
    collimator: float = 0.0,
    couch: float = 0.0,
    width_u: float = 8.0,
    width_v: float = 8.0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> SpectralBeamSource:
    """The fan of one beam; ``offset`` moves the field centre along its ``u``/``v``."""
    u, v, _ = beam_axes(angle_deg, collimator, couch)
    centre = tuple(c + offset[0] * a + offset[1] * b for c, a, b in zip(ISO, u, v, strict=True))
    return SpectralBeamSource(
        spectrum,
        focal(angle_deg, collimator, couch),
        centre,
        width_u=width_u,
        width_v=width_v,
        u_axis=u,
        v_axis=v,
    )


def arc_frame(angle_deg: float, collimator: float = 0.0, couch: float = 0.0) -> BeamFrame:
    u, v, w = beam_axes(angle_deg, collimator, couch)
    return BeamFrame(origin=focal(angle_deg, collimator, couch), u_axis=u, v_axis=v, w_axis=w)


def arc_stack(angle_deg: float, jaw_neg, jaw_pos, tips_neg, tips_pos) -> BeamLimitingStack:
    return BeamLimitingStack(
        frame=arc_frame(angle_deg),
        devices=(
            JawPair("u", 30.0, 36.0, jaw_neg, jaw_pos, material=WATER, density=1.0),
            MLC(
                z_top=40.0,
                z_bottom=46.0,
                leaf_edges_v=EDGES,
                tips_neg=tuple(tips_neg),
                tips_pos=tuple(tips_pos),
                tip_radius=8.0,
                material=WATER,
                density=1.0,
                focused_sides=True,
            ),
        ),
    )


def _stacks() -> list[BeamLimitingStack]:
    return [arc_stack(a, *s) for a, s in zip(ANGLES, SETTINGS, strict=True)]


def _fans() -> list[SpectralBeamSource]:
    return [arc_fan(a) for a in ANGLES]


def _source() -> SegmentedFanSource:
    return SegmentedFanSource.from_stacks(_fans(), _stacks(), AnalyticCrossSections())


def _arrays():
    jaws = np.array([[s[0], s[1]] for s in SETTINGS])
    return dict(
        jaw_neg=jaws[:, :1],
        jaw_pos=jaws[:, 1:],
        tips_neg=np.array([s[2] for s in SETTINGS]),
        tips_pos=np.array([s[3] for s in SETTINGS]),
    )


def test_an_arc_has_one_fan_and_one_frame_per_segment() -> None:
    source = _source()
    assert source.n_segments == source.n_beamlets == len(ANGLES)
    assert len(source.fans) == len(ANGLES)
    assert source.frames == tuple(arc_frame(a) for a in ANGLES)
    assert source.spectrum is SPECTRUM
    assert source.max_energy == SPECTRUM.max_energy
    for s, angle in enumerate(ANGLES):
        assert source.fans[s].focal_point == focal(angle)
        assert source.segment_stack(s) == _stacks()[s]
    with pytest.raises(ValueError, match="one fan per segment"):
        _ = source.fan


def test_a_single_fan_source_reports_its_fan_and_frame_per_segment() -> None:
    """The single-fan path: every segment shares the fan and the template frame."""
    fan = arc_fan(0.0)
    stacks = [arc_stack(0.0, *s) for s in SETTINGS]
    source = SegmentedFanSource.from_stacks(fan, stacks, AnalyticCrossSections())
    assert source.fan is fan
    assert source.fans == (fan, fan, fan)
    assert source.frames == (stacks[0].frame,) * 3
    assert source.spectrum is SPECTRUM


@pytest.mark.parametrize("segment", range(len(ANGLES)))
def test_emit_is_the_collimated_fan_of_that_segment(segment: int) -> None:
    """Segment s emits its own fan's ray, weighted by its own rotated stack."""
    source = _source()
    oracle = CollimatedSource(_fans()[segment], _stacks()[segment], AnalyticCrossSections())
    rng = HostRNG()
    for history in range(200):
        got = source.emit(segment, rng.init_state(SEED, history))
        want = oracle.emit(rng.init_state(SEED, history))
        assert got[:7] == pytest.approx(want[:7], rel=0.0, abs=0.0)
        assert got.weight == pytest.approx(want.weight, rel=1e-9)


@pytest.mark.parametrize("segment", range(len(ANGLES)))
def test_vectorized_batch_is_the_collimated_fan_batch(segment: int) -> None:
    source = _source()
    oracle = CollimatedSource(_fans()[segment], _stacks()[segment], AnalyticCrossSections())
    got = source.sample_beamlet_batch(SEED, 1_000, 4_000, segment)
    want = oracle.sample_batch(SEED, 1_000, 4_000)
    assert set(got) == set(want)
    for name in ("particle_type", "energy", "x", "y", "z", "ux", "uy", "uz"):
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)
    np.testing.assert_allclose(got["weight"], want["weight"], rtol=1e-6)
    assert 0.0 < want["weight"].min() < 0.9 and want["weight"].max() > 0.9


def test_correlated_replay_is_one_beams_eye_view_ray_in_every_frame() -> None:
    """The same uniforms drive each segment's own fan: one energy, one local ray."""
    source = _source()
    rng = HostRNG()
    for history in range(100):
        primaries = [source.emit(s, rng.init_state(SEED, history)) for s in range(len(ANGLES))]
        local = []
        for s, p in enumerate(primaries):
            frame = source.frames[s]
            _, d = frame.to_local(np.array([[p.x, p.y, p.z]]), np.array([[p.ux, p.uy, p.uz]]))
            local.append(d[0])
            assert p.energy == primaries[0].energy
        for d in local[1:]:
            np.testing.assert_allclose(d, local[0], rtol=0.0, atol=1e-12)


def test_vectorized_batch_is_chunk_invariant() -> None:
    source = _source()
    whole = source.sample_beamlet_batch(SEED, 0, 3_000, 1)
    parts = [source.sample_beamlet_batch(SEED, start, 1_000, 1) for start in (0, 1_000, 2_000)]
    for name, column in whole.items():
        np.testing.assert_array_equal(column, np.concatenate([p[name] for p in parts]))


def test_array_constructor_with_frames_describes_the_same_arc() -> None:
    """The cheap path (arrays, one template, frames) equals the stack-list path."""
    from_arrays = SegmentedFanSource(
        _fans(),
        _stacks()[0],
        AnalyticCrossSections(),
        frames=[arc_frame(a) for a in ANGLES],
        **_arrays(),
    )
    from_list = _source()
    for s in range(len(ANGLES)):
        assert from_arrays.segment_stack(s) == _stacks()[s]
        a = from_arrays.sample_beamlet_batch(SEED, 0, 2_000, s)
        b = from_list.sample_beamlet_batch(SEED, 0, 2_000, s)
        for name, column in a.items():
            np.testing.assert_array_equal(column, b[name], err_msg=name)


def test_per_segment_fans_fix_the_count_without_settings() -> None:
    """Fans and frames alone fix n_segments; the settings default to the template's."""
    template = _stacks()[1]
    source = SegmentedFanSource(
        _fans(), template, AnalyticCrossSections(), frames=[arc_frame(a) for a in ANGLES]
    )
    assert source.n_segments == len(ANGLES)
    for s, angle in enumerate(ANGLES):
        assert source.segment_stack(s) == BeamLimitingStack(
            frame=arc_frame(angle), devices=template.devices
        )


def test_with_settings_keeps_the_fans_and_frames() -> None:
    source = _source()
    tips_neg = np.array([s[2] for s in SETTINGS]) - 0.1
    moved = source.with_settings(tips_neg=tips_neg)
    assert moved.fans == source.fans
    assert moved.frames == source.frames
    assert moved.attenuation_tables is source.attenuation_tables
    for s in range(len(ANGLES)):
        jaw, mlc = moved.segment_stack(s).devices
        assert moved.segment_stack(s).frame == arc_frame(ANGLES[s])
        assert jaw == _stacks()[s].devices[0]
        np.testing.assert_array_equal(mlc.tips_neg, tips_neg[s])
    with pytest.raises(ValueError, match="segments"):
        source.with_settings(tips_neg=tips_neg[:2], tips_pos=np.array([s[3] for s in SETTINGS])[:2])


def test_plan_emits_the_chosen_segments_own_fan() -> None:
    source = _source()
    weights = np.array([1.0, 3.0, 2.0])
    plan = source.plan(weights)
    rng = HostRNG()
    chosen_set = set()
    for history in range(200):
        state = rng.init_state(SEED, history)
        u = state.random()
        chosen = int(np.searchsorted(np.cumsum(weights) / weights.sum(), u, side="right"))
        chosen_set.add(chosen)
        want = source.emit(chosen, state)
        assert plan.emit(rng.init_state(SEED, history)) == want
        assert (want.x, want.y, want.z) == pytest.approx(focal(ANGLES[chosen]), abs=1e-12)
    assert chosen_set == {0, 1, 2}


def test_plan_batch_rows_are_their_segments_batch() -> None:
    source = _source()
    plan = source.plan([1.0, 3.0, 6.0])
    whole = plan.sample_batch(SEED, 0, 12_000)
    segment = plan.segment_of(SEED, 0, 12_000)
    for s in range(len(ANGLES)):
        rows = segment == s
        want = source.sample_beamlet_batch(SEED, 0, 12_000, s)
        for name, column in whole.items():
            np.testing.assert_array_equal(column[rows], want[name][rows], err_msg=name)


# --- validation ----------------------------------------------------------------------


def test_another_spectrum_is_refused() -> None:
    other = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.5))
    fans = _fans()
    fans[2] = arc_fan(ANGLES[2], other)
    with pytest.raises(ValueError, match=r"segment 2.*spectrum"):
        SegmentedFanSource.from_stacks(fans, _stacks(), AnalyticCrossSections())


def test_an_equal_spectrum_object_is_accepted() -> None:
    fans = _fans()
    fans[1] = arc_fan(ANGLES[1], Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0)))
    source = SegmentedFanSource.from_stacks(fans, _stacks(), AnalyticCrossSections())
    assert source.spectrum is SPECTRUM


def test_a_frame_off_its_fans_focal_point_is_refused() -> None:
    frames = [arc_frame(a) for a in ANGLES]
    frames[1] = arc_frame(ANGLES[2])  # the 90-degree fan with the 225-degree frame
    with pytest.raises(ValueError, match=r"segment 1.*focal point"):
        SegmentedFanSource(_fans(), _stacks()[0], AnalyticCrossSections(), frames=frames)


def test_a_fan_not_aligned_with_its_frame_is_refused() -> None:
    """A rotated focal spot and frame origin with unrotated axes: the missing rotation."""
    frames = [arc_frame(a) for a in ANGLES]
    frames[2] = BeamFrame(origin=focal(ANGLES[2]))  # identity axes
    with pytest.raises(ValueError, match=r"segment 2"):
        SegmentedFanSource(_fans(), _stacks()[0], AnalyticCrossSections(), frames=frames)
    u, v, _ = beam_axes(30.0)
    fans = _fans()
    fans[0] = SpectralBeamSource(SPECTRUM, focal(0.0), ISO, 8.0, 8.0, u_axis=u, v_axis=v)
    frames = [arc_frame(a) for a in ANGLES]
    with pytest.raises(ValueError, match=r"segment 0.*axis"):
        SegmentedFanSource(fans, _stacks()[0], AnalyticCrossSections(), frames=frames)


def test_a_field_upstream_of_its_frame_is_refused() -> None:
    fans = _fans()
    u, v, w = beam_axes(90.0)
    behind = tuple(f - 10.0 * a for f, a in zip(focal(90.0), w, strict=True))
    fans[1] = SpectralBeamSource(SPECTRUM, focal(90.0), behind, 8.0, 8.0, u_axis=u, v_axis=v)
    with pytest.raises(ValueError, match=r"segment 1.*downstream"):
        SegmentedFanSource.from_stacks(fans, _stacks(), AnalyticCrossSections())


def test_counts_must_agree() -> None:
    xs = AnalyticCrossSections()
    frames = [arc_frame(a) for a in ANGLES]
    with pytest.raises(ValueError, match="frames"):
        SegmentedFanSource(_fans(), _stacks()[0], xs, frames=frames[:2])
    with pytest.raises(ValueError, match="segments"):
        SegmentedFanSource(
            _fans()[:2], _stacks()[0], xs, frames=frames[:2], tips_neg=np.zeros((3, 3)),
            tips_pos=np.zeros((3, 3)),
        )  # fmt: skip
    with pytest.raises(ValueError, match="stacks"):
        SegmentedFanSource.from_stacks(_fans()[:2], _stacks(), xs)
    with pytest.raises(ValueError, match="at least one"):
        SegmentedFanSource([], _stacks()[0], xs, frames=[])


def test_per_segment_fans_without_frames_need_a_shared_focal_point() -> None:
    """No frames: every fan must sit on the template frame's origin (a collimator-only arc)."""
    with pytest.raises(ValueError, match=r"segment 1.*focal point"):
        SegmentedFanSource(_fans(), _stacks()[0], AnalyticCrossSections(), **_arrays())


def test_a_structurally_different_arc_segment_is_refused() -> None:
    stacks = _stacks()
    jaw, mlc = stacks[1].devices
    stacks[1] = BeamLimitingStack(
        frame=stacks[1].frame, devices=(jaw, MLC(**{**mlc.__dict__, "tip_radius": 9.0}))
    )
    with pytest.raises(ValueError, match=r"segment 1.*device 1"):
        SegmentedFanSource.from_stacks(_fans(), stacks, AnalyticCrossSections())
