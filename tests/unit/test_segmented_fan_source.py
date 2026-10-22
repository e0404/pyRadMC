"""The segments of one collimated fan as the beamlets of a source.

:class:`~pyradmc.geometry.segments.SegmentedFanSource` emits the fan of a
:class:`~pyradmc.geometry.source.SpectralBeamSource` through one beam-limiting
stack per segment: beamlet ``s`` is segment ``s``, and a history's primary weight
is the stack transmission ``exp(-tau_s)`` along its ray at its energy — the
semantics of the head pre-solve's ``mode="attenuation"`` and of
:class:`~pyradmc.geometry.collimation.CollimatedSource`, which is the independent
oracle here (its own table, its own wrapper code).

The fan draws are segment-blind (correlated Dij replay: only the weight differs
between columns), the stack-list and array constructors describe the same
segments, and malformed segment sets are refused at construction. Analytic water
"collimators" keep the tier EPDL-free.
"""

from __future__ import annotations

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
FOCAL = (8.0, 8.0, -100.0)
FRAME = BeamFrame(origin=FOCAL)
EDGES = (-3.0, -1.0, 1.0, 3.0)
# Per-segment settings: (jaw edge_neg, jaw edge_pos, tips_neg, tips_pos), mid-plane cm.
SETTINGS = (
    (-2.0, 2.0, (-1.5, -0.5, -1.5), (1.5, 0.5, 1.5)),
    (-1.0, 2.5, (-0.5, 0.0, -2.0), (0.5, 0.0, 0.2)),
    (-2.5, 0.5, (0.0, -1.0, -0.3), (0.0, 1.0, 0.3)),
)


def _fan() -> SpectralBeamSource:
    return SpectralBeamSource(
        SPECTRUM, focal_point=FOCAL, center=(8.0, 8.0, 0.0), width_u=8.0, width_v=8.0
    )


def _stack(jaw_neg: float, jaw_pos: float, tips_neg, tips_pos, frame=FRAME) -> BeamLimitingStack:
    return BeamLimitingStack(
        frame=frame,
        devices=(
            JawPair(
                axis="u",
                z_top=30.0,
                z_bottom=36.0,
                edge_neg=jaw_neg,
                edge_pos=jaw_pos,
                material=WATER,
                density=1.0,
            ),
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
    return [_stack(*setting) for setting in SETTINGS]


def _source() -> SegmentedFanSource:
    return SegmentedFanSource.from_stacks(_fan(), _stacks(), AnalyticCrossSections())


def test_segments_are_the_beamlets() -> None:
    source = _source()
    assert source.n_beamlets == len(SETTINGS)
    assert source.n_segments == len(SETTINGS)
    assert source.max_energy == SPECTRUM.max_energy


@pytest.mark.parametrize("segment", range(len(SETTINGS)))
def test_emit_is_the_collimated_fan_of_that_segment(segment: int) -> None:
    """Same draws, same ray, weight = the segment's stack transmission."""
    source = _source()
    oracle = CollimatedSource(_fan(), _stacks()[segment], AnalyticCrossSections())
    rng = HostRNG()
    for history in range(200):
        got = source.emit(segment, rng.init_state(SEED, history))
        want = oracle.emit(rng.init_state(SEED, history))
        assert got[:7] == pytest.approx(want[:7], rel=0.0, abs=0.0)
        assert got.kind == want.kind
        assert got.weight == pytest.approx(want.weight, rel=1e-9)


def test_emit_is_segment_blind_but_the_weights_differ() -> None:
    """Correlated replay: one stream gives one ray in every column, weighted per segment."""
    source = _source()
    rng = HostRNG()
    weights = []
    for history in range(100):
        primaries = [source.emit(s, rng.init_state(SEED, history)) for s in range(3)]
        for p in primaries[1:]:
            assert p[:7] == primaries[0][:7]
        weights.append([p.weight for p in primaries])
    weights = np.asarray(weights)
    assert np.all((weights > 0.0) & (weights <= 1.0))
    assert not np.allclose(weights[:, 0], weights[:, 1])
    assert not np.allclose(weights[:, 0], weights[:, 2])


@pytest.mark.parametrize("segment", range(len(SETTINGS)))
def test_vectorized_batch_is_the_collimated_fan_batch(segment: int) -> None:
    source = _source()
    oracle = CollimatedSource(_fan(), _stacks()[segment], AnalyticCrossSections())
    got = source.sample_beamlet_batch(SEED, 1_000, 4_000, segment)
    want = oracle.sample_batch(SEED, 1_000, 4_000)
    assert set(got) == set(want)
    for name in ("particle_type", "energy", "x", "y", "z", "ux", "uy", "uz"):
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)
    np.testing.assert_allclose(got["weight"], want["weight"], rtol=1e-6)


def test_vectorized_batch_is_chunk_invariant() -> None:
    source = _source()
    whole = source.sample_beamlet_batch(SEED, 0, 3_000, 1)
    parts = [source.sample_beamlet_batch(SEED, start, 1_000, 1) for start in (0, 1_000, 2_000)]
    for name, column in whole.items():
        np.testing.assert_array_equal(column, np.concatenate([p[name] for p in parts]))


def test_array_constructor_describes_the_same_segments() -> None:
    """The cheap path (tip arrays, no MLC objects) equals the stack-list path."""
    stacks = _stacks()
    jaws = np.array([[s[0], s[1]] for s in SETTINGS])
    tips_neg = np.array([s[2] for s in SETTINGS])
    tips_pos = np.array([s[3] for s in SETTINGS])
    from_arrays = SegmentedFanSource(
        _fan(),
        stacks[0],
        AnalyticCrossSections(),
        jaw_neg=jaws[:, :1],
        jaw_pos=jaws[:, 1:],
        tips_neg=tips_neg,
        tips_pos=tips_pos,
    )
    from_list = _source()
    for s in range(3):
        assert from_arrays.segment_stack(s) == stacks[s]
        a = from_arrays.sample_beamlet_batch(SEED, 0, 2_000, s)
        b = from_list.sample_beamlet_batch(SEED, 0, 2_000, s)
        np.testing.assert_array_equal(a["weight"], b["weight"])


def test_with_settings_replaces_the_segments_and_keeps_the_rest() -> None:
    source = _source()
    tips_neg = np.array([[-0.2, -0.2, -0.2]])
    tips_pos = np.array([[0.2, 0.2, 0.2]])
    narrow = source.with_settings(tips_neg=tips_neg, tips_pos=tips_pos)
    assert narrow.n_segments == 1
    # The jaws default to the template's (segment 0 of the original).
    assert narrow.segment_stack(0).devices[0] == _stacks()[0].devices[0]
    assert narrow.segment_stack(0).devices[1].tips_neg == (-0.2, -0.2, -0.2)
    assert narrow.attenuation_tables is source.attenuation_tables


def test_unspecified_settings_default_to_the_template() -> None:
    stacks = _stacks()
    source = SegmentedFanSource(
        _fan(),
        stacks[1],
        AnalyticCrossSections(),
        tips_neg=np.array([SETTINGS[1][2], SETTINGS[1][2]]),
        tips_pos=np.array([SETTINGS[1][3], SETTINGS[1][3]]),
    )
    assert source.n_segments == 2
    assert source.segment_stack(0) == stacks[1]
    assert source.segment_stack(1) == stacks[1]


def test_device_layout_is_one_stack_and_per_segment_columns() -> None:
    """Jaw edges are (segment, device) with MLC rows unused; tips flatten over MLCs."""
    source = _source()
    assert source.jaw_neg.shape == (3, 2)
    assert source.tips_neg.shape == (3, 3)
    np.testing.assert_array_equal(source.jaw_pos[:, 0], [s[1] for s in SETTINGS])
    np.testing.assert_array_equal(source.tips_pos, [s[3] for s in SETTINGS])
    compiled = source.template.flatten()
    assert compiled.tips_neg.size == source.tips_neg.shape[1]


# --- validation ----------------------------------------------------------------------


def test_beamlet_out_of_range_is_refused() -> None:
    source = _source()
    with pytest.raises(IndexError):
        source.emit(3, HostRNG().init_state(SEED, 0))
    with pytest.raises(IndexError):
        source.sample_beamlet_batch(SEED, 0, 10, -1)


def test_no_segments_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        SegmentedFanSource.from_stacks(_fan(), [], AnalyticCrossSections())
    with pytest.raises(ValueError, match="per-segment"):
        SegmentedFanSource(_fan(), _stacks()[0], AnalyticCrossSections())


def test_a_different_frame_is_refused() -> None:
    other = BeamFrame(origin=FOCAL, u_axis=(0.0, 1.0, 0.0), v_axis=(-1.0, 0.0, 0.0))
    stacks = [*_stacks()[:2], _stack(*SETTINGS[2], frame=other)]
    with pytest.raises(ValueError, match=r"segment 2.*frame"):
        SegmentedFanSource.from_stacks(_fan(), stacks, AnalyticCrossSections())


def test_a_frame_away_from_the_focal_point_is_refused() -> None:
    frame = BeamFrame(origin=(8.0, 8.0, -90.0))
    stacks = [_stack(*s, frame=frame) for s in SETTINGS]
    with pytest.raises(ValueError, match="focal point"):
        SegmentedFanSource.from_stacks(_fan(), stacks, AnalyticCrossSections())


def test_a_fan_aimed_away_from_the_stack_is_refused() -> None:
    upward = SpectralBeamSource(
        SPECTRUM, focal_point=FOCAL, center=(8.0, 8.0, -200.0), width_u=8.0, width_v=8.0
    )
    with pytest.raises(ValueError, match="downstream"):
        SegmentedFanSource.from_stacks(upward, _stacks(), AnalyticCrossSections())


@pytest.mark.parametrize(
    "change",
    [
        lambda d: MLC(**{**d.__dict__, "leaf_edges_v": (-3.0, -1.5, 1.0, 3.0)}),
        lambda d: MLC(**{**d.__dict__, "tip_radius": 9.0}),
        lambda d: MLC(**{**d.__dict__, "density": 1.1}),
        lambda d: MLC(**{**d.__dict__, "focused_sides": False}),
        lambda d: MLC(**{**d.__dict__, "z_bottom": 47.0}),
    ],
    ids=["leaf-edges", "tip-radius", "density", "focused-sides", "slab"],
)
def test_a_structurally_different_segment_is_refused(change) -> None:
    stacks = _stacks()
    jaw, mlc = stacks[1].devices
    stacks[1] = BeamLimitingStack(frame=FRAME, devices=(jaw, change(mlc)))
    with pytest.raises(ValueError, match=r"segment 1.*device 1"):
        SegmentedFanSource.from_stacks(_fan(), stacks, AnalyticCrossSections())


def test_a_segment_with_another_device_list_is_refused() -> None:
    stacks = _stacks()
    stacks[2] = BeamLimitingStack(frame=FRAME, devices=(stacks[2].devices[1],))
    with pytest.raises(ValueError, match="segment 2"):
        SegmentedFanSource.from_stacks(_fan(), stacks, AnalyticCrossSections())


def test_malformed_setting_arrays_are_refused() -> None:
    template = _stacks()[0]
    xs = AnalyticCrossSections()
    good = np.zeros((2, 3))
    with pytest.raises(ValueError, match="shape"):
        SegmentedFanSource(_fan(), template, xs, tips_neg=np.zeros((2, 4)), tips_pos=good)
    with pytest.raises(ValueError, match="segments"):
        SegmentedFanSource(_fan(), template, xs, tips_neg=np.zeros((3, 3)), tips_pos=good)
    with pytest.raises(ValueError, match="cross"):
        SegmentedFanSource(_fan(), template, xs, tips_neg=good + 0.1, tips_pos=good)
    with pytest.raises(ValueError, match="finite"):
        SegmentedFanSource(_fan(), template, xs, tips_neg=good * np.nan, tips_pos=good)
    with pytest.raises(ValueError, match="cross"):
        SegmentedFanSource(
            _fan(), template, xs, jaw_neg=np.array([[1.0]]), jaw_pos=np.array([[0.0]])
        )
