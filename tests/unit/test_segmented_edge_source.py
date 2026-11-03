"""The edges of a segmented fan as beamlets, built without one Python object per edge.

:class:`SegmentedEdgeSource` lists every leaf-tip and jaw edge of every segment of a
:class:`SegmentedFanSource` and builds the plain-data columns an in-kernel generator
reads (:class:`FanEdgeColumns`) with array arithmetic. The oracle is the per-edge path:
one :class:`FanEdgeDerivativeSource` per edge on the segment's own stack, its
:meth:`~FanEdgeDerivativeSource.device_record`, and :func:`fan_edge_columns` over those
records. Pinned here, exactly (``np.array_equal`` in float64, before any float32
cast): every column, over segments with open, nearly closed and closed pairs, tips on
the field limit, straight and focused jaws on both axes, and rounded tips whose
tangent point leaves the slab (the bisection fallback). The host route of the source
delegates to the per-edge sources and is pinned equal to them draw for draw.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.aperture_gradient import (
    EdgeParameter,
    FanEdgeColumns,
    FanEdgeDerivativeSource,
    _support_start,
    fan_edge_columns,
)
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.segment_edges import SegmentedEdgeSource
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

XS = AnalyticCrossSections()
SPECTRUM = Spectrum(edges=[1.0, 1.8, 2.2], weights=[0.3, 0.7])
FOCAL = (1.0, -2.0, -100.0)
LEAF_EDGES = (-1.5, -0.5, 0.5, 1.5)


def _fan() -> SpectralBeamSource:
    """Off axis in u, so a 2 cm slab's rounded tips need the bisection beyond u ~ 2.6."""
    return SpectralBeamSource(
        SPECTRUM,
        focal_point=FOCAL,
        center=(FOCAL[0] + 6.0, FOCAL[1], FOCAL[2] + 100.0),
        width_u=8.0,
        width_v=6.0,
    )


def _template() -> BeamLimitingStack:
    """A focused v-jaw (jump edges), a straight u-jaw and a thin-slab 3-pair MLC."""
    return BeamLimitingStack(
        frame=BeamFrame(origin=FOCAL),
        devices=(
            JawPair(
                axis="v", z_top=20.0, z_bottom=27.8, edge_neg=-1.0, edge_pos=1.0,
                material=WATER, density=5.0, focused=True,
            ),
            JawPair(
                axis="u", z_top=30.0, z_bottom=37.8, edge_neg=0.5, edge_pos=3.0,
                material=WATER, density=5.0,
            ),
            MLC(
                z_top=50.0, z_bottom=52.0, leaf_edges_v=LEAF_EDGES, tips_neg=(1.0,) * 3,
                tips_pos=(4.0,) * 3, tip_radius=20.0, material=WATER, density=5.0,
                focused_sides=True,
            ),
        ),
    )  # fmt: skip


# The MLC field at its mid-plane (51 cm): u in [1.02, 5.1]; the closed-form support
# start holds for tips up to u ~ 2.6 and the bisection takes over beyond.
TIPS_NEG = np.array(
    [
        [1.5, 2.0, 1.2],  # open
        [3.0, 2.5, 1.4],  # pair 0 nearly closed, pair 1 closed
        [1.02, 1.02, 2.7],  # on the field's low limit
        [3.1, 4.4, 2.6],  # deep in the bisection range
    ]
)
TIPS_POS = np.array(
    [
        [4.0, 4.5, 2.2],
        [3.0 + 1.0e-6, 2.5, 4.9],
        [5.1, 3.3, 2.7],
        [5.0, 4.6, 3.9],
    ]
)
JAW_NEG = np.array([[-1.0, 0.5], [-0.2, 0.9], [-0.7, 0.6], [0.0, 1.6]])
JAW_POS = np.array([[1.0, 3.0], [0.4, 2.2], [0.7, 3.4], [0.0, 1.6]])  # last: closed jaws


def _segments() -> SegmentedFanSource:
    return SegmentedFanSource(
        _fan(),
        _template(),
        XS,
        jaw_neg=JAW_NEG,
        jaw_pos=JAW_POS,
        tips_neg=TIPS_NEG,
        tips_pos=TIPS_POS,
    )


def _expected_parameters(n_segments: int, devices=(0, 1, 2)) -> list[tuple[int, EdgeParameter]]:
    """Segment-major, then device in the given order, then leaf pair, then side."""
    out = []
    for s in range(n_segments):
        for j in devices:
            if j == 2:
                for i in range(len(LEAF_EDGES) - 1):
                    out += [(s, EdgeParameter(j, "neg", i)), (s, EdgeParameter(j, "pos", i))]
            else:
                out += [(s, EdgeParameter(j, "neg")), (s, EdgeParameter(j, "pos"))]
    return out


def _per_edge(segments: SegmentedFanSource, parameters, **kwargs) -> list[FanEdgeDerivativeSource]:
    """The oracle: one per-edge source per edge, on the segment's own fan and stack."""
    return [
        FanEdgeDerivativeSource(
            segments.fans[s], segments.segment_stack(s), segments.cross_sections, p, **kwargs
        )
        for s, p in parameters
    ]


# -- an arc: one fan and one frame per segment ---------------------------------------------

# (gantry, collimator, couch, field width_u, width_v, field offset along u, v): a plain
# beam, a turned collimator with its own off-axis field, a non-coplanar beam with
# another, so every field-dependent column differs between segments.
ARC = (
    (0.0, 0.0, 0.0, 8.0, 8.0, (0.0, 0.0)),
    (90.0, 30.0, 0.0, 7.0, 9.0, (0.5, -0.3)),
    (225.0, 0.0, 25.0, 9.0, 7.0, (-0.4, 0.4)),
)


def _arc_segments(xs=XS) -> SegmentedFanSource:
    """Three control points of an arc through a straight u-jaw and a 3-pair MLC."""
    from tests.unit.test_segmented_arc_source import _arrays, arc_fan, arc_frame, arc_stack

    fans = [
        arc_fan(g, SPECTRUM, collimator=c, couch=k, width_u=wu, width_v=wv, offset=o)
        for g, c, k, wu, wv, o in ARC
    ]
    frames = [arc_frame(g, c, k) for g, c, k, *_ in ARC]
    template = arc_stack(0.0, -2.0, 2.0, (-1.5,) * 3, (1.5,) * 3)
    return SegmentedFanSource(fans, template, xs, frames=frames, **_arrays())


def _arc_parameters() -> list[tuple[int, EdgeParameter]]:
    """Segment-major: the u-jaw's two edges, then the MLC pair by pair, neg then pos."""
    return [
        (s, p)
        for s in range(len(ARC))
        for p in (
            EdgeParameter(0, "neg"),
            EdgeParameter(0, "pos"),
            *(EdgeParameter(1, side, i) for i in range(3) for side in ("neg", "pos")),
        )
    ]


def s_grids(columns: FanEdgeColumns) -> np.ndarray:
    """Each smooth edge's ``s`` grid, gathered through ``s_row`` (rows may be shared)."""
    smooth = columns.s_row >= 0
    if not smooth.any():  # no grids: ``s`` is a one-element placeholder
        return np.zeros((0, columns.n_table))
    return columns.s.reshape(-1, columns.n_table)[columns.s_row[smooth]]


def assert_columns_equal(got: FanEdgeColumns, want: FanEdgeColumns) -> None:
    """Every column exactly equal (float64, int32), the stack and data by value/identity.

    ``s``/``s_row`` are compared as what an edge reads: its own grid, row by row.
    """
    np.testing.assert_array_equal(got.s_row >= 0, want.s_row >= 0)
    np.testing.assert_array_equal(s_grids(got), s_grids(want))
    for field in dataclasses.fields(FanEdgeColumns):
        a, b = getattr(got, field.name), getattr(want, field.name)
        if field.name in ("s", "s_row"):
            continue
        if field.name == "stack":
            for sub in dataclasses.fields(a):
                x, y = getattr(a, sub.name), getattr(b, sub.name)
                np.testing.assert_array_equal(x, y, err_msg=f"stack.{sub.name}")
                assert np.asarray(x).dtype == np.asarray(y).dtype, sub.name
        elif field.name == "cross_sections":
            assert a is b
        elif field.name == "spectrum":
            np.testing.assert_array_equal(a.edges, b.edges)
            np.testing.assert_array_equal(a.cdf, b.cdf)
        elif isinstance(b, np.ndarray):
            assert a.dtype == b.dtype, (field.name, a.dtype, b.dtype)
            np.testing.assert_array_equal(a, b, err_msg=field.name)
        else:
            assert a == b, field.name


def test_beamlets_are_segment_major_then_device_pair_side() -> None:
    source = SegmentedEdgeSource(_segments())
    assert list(source.parameters) == _expected_parameters(4)
    assert source.n_beamlets == 4 * (2 + 2 + 6)
    assert source.max_energy == _fan().max_energy


def test_device_selection_keeps_the_given_order() -> None:
    source = SegmentedEdgeSource(_segments(), devices=(2, 0))
    assert list(source.parameters) == _expected_parameters(4, devices=(2, 0))


def test_edge_values_are_the_segment_settings() -> None:
    source = SegmentedEdgeSource(_segments())
    segments = source.segments
    want = [
        float(
            segments.segment_stack(s).devices[p.device].tips_pos[p.pair]
            if p.pair is not None and p.side == "pos"
            else segments.segment_stack(s).devices[p.device].tips_neg[p.pair]
            if p.pair is not None
            else getattr(segments.segment_stack(s).devices[p.device], f"edge_{p.side}")
        )
        for s, p in source.parameters
    ]
    np.testing.assert_array_equal(source.values, np.array(want))


def test_the_geometry_exercises_every_support_branch() -> None:
    """Guard on the fixture: jumps, straight faces, closed-form tips and bisected tips."""
    records = [e.device_record() for e in _per_edge(_segments(), _expected_parameters(4))]
    kinds = {r.kind for r in records}
    assert kinds == {0, 1, 2}
    tips = [r for r in records if r.kind == 0]
    bisected = [_support_start(r.block, r.x, r.sign, r.z_mid) is None for r in tips]
    assert any(bisected) and not all(bisected), sum(bisected)


@pytest.mark.parametrize(("n_table", "defensive"), [(513, 0.05), (65, 0.2)])
def test_device_columns_equal_the_per_edge_records(n_table: int, defensive: float) -> None:
    segments = _segments()
    source = SegmentedEdgeSource(segments, n_table=n_table, defensive=defensive)
    edges = _per_edge(segments, _expected_parameters(4), n_table=n_table, defensive=defensive)
    assert_columns_equal(
        source.device_columns(), fan_edge_columns([e.device_record() for e in edges])
    )


def test_device_columns_follow_with_settings() -> None:
    """The optimizer's path: new tips through ``with_settings``, a new edge source."""
    moved = _segments().with_settings(tips_neg=TIPS_NEG - 0.05, tips_pos=TIPS_POS + 0.05)
    source = SegmentedEdgeSource(moved, devices=(2,))
    edges = _per_edge(moved, _expected_parameters(4, devices=(2,)))
    assert_columns_equal(
        source.device_columns(), fan_edge_columns([e.device_record() for e in edges])
    )


def test_a_jaw_only_selection_has_no_smooth_table_quirks() -> None:
    """Only focused-jaw (jump) edges: no proposal tables, the per-edge path's n_table 2."""
    segments = _segments()
    source = SegmentedEdgeSource(segments, devices=(0,))
    edges = _per_edge(segments, _expected_parameters(4, devices=(0,)))
    assert_columns_equal(
        source.device_columns(), fan_edge_columns([e.device_record() for e in edges])
    )
    assert source.device_columns().n_table == 2


def test_host_route_is_the_per_edge_route_draw_for_draw() -> None:
    segments = _segments()
    source = SegmentedEdgeSource(segments)
    edges = _per_edge(segments, _expected_parameters(4))
    rng = HostRNG()
    for k in (0, 3, 9, 17, 25, 39):
        for i in range(5):
            assert source.emit(k, rng.init_state(SEED, i)) == edges[k].emit(rng.init_state(SEED, i))
        got = source.sample_beamlet_batch(SEED, 7, 50, k)
        want = edges[k].sample_batch(SEED, 7, 50)
        assert got.keys() == want.keys()
        for key in want:
            np.testing.assert_array_equal(got[key], want[key])


def test_refuses_what_the_per_edge_source_refuses() -> None:
    segments = _segments()
    with pytest.raises(ValueError, match="defensive"):
        SegmentedEdgeSource(segments, defensive=1.0)
    with pytest.raises(ValueError, match="device"):
        SegmentedEdgeSource(segments, devices=(3,))
    with pytest.raises(ValueError, match="device"):
        SegmentedEdgeSource(segments, devices=())
    template = _template()
    mlc = dataclasses.replace(template.devices[2], focused_sides=False)
    parallel = SegmentedFanSource(
        _fan(),
        dataclasses.replace(template, devices=(*template.devices[:2], mlc)),
        XS,
        tips_neg=TIPS_NEG,
        tips_pos=TIPS_POS,
    )
    with pytest.raises(ValueError, match="focused"):
        SegmentedEdgeSource(parallel)
    # The segmented fan does not compare the aperture axes with the frame's; the edge
    # estimator needs them equal (a turned collimator is a turned frame *and* fan).
    turned = dataclasses.replace(
        template, frame=BeamFrame(origin=FOCAL, u_axis=(0.0, 1.0, 0.0), v_axis=(-1.0, 0.0, 0.0))
    )
    with pytest.raises(ValueError, match="axes"):
        SegmentedEdgeSource(
            SegmentedFanSource(_fan(), turned, XS, tips_neg=TIPS_NEG, tips_pos=TIPS_POS)
        )


def test_refuses_a_leaf_pair_outside_the_field() -> None:
    """The per-edge source refuses an edge with no extent in the field; so does the set."""
    narrow = SpectralBeamSource(
        SPECTRUM, FOCAL, (FOCAL[0] + 6.0, FOCAL[1] + 2.5, 0.0), 8.0, 1.0
    )  # v field at the MLC mid-plane: [1.02, 1.53], clear of pair 0
    segments = SegmentedFanSource(narrow, _template(), XS, tips_neg=TIPS_NEG, tips_pos=TIPS_POS)
    with pytest.raises(ValueError, match="outside the beam's field"):
        FanEdgeDerivativeSource(narrow, segments.segment_stack(0), XS, EdgeParameter(2, "pos", 0))
    with pytest.raises(ValueError, match="outside the beam's field"):
        SegmentedEdgeSource(segments)
    SegmentedEdgeSource(segments, devices=(0, 1))  # the jaws still see the field


def test_vectorized_support_starts_equal_the_scalar_definition() -> None:
    """Random tips, plus slabs whose faces sit exactly on the tangent point.

    ``np.hypot`` and ``math.hypot`` differ by an ulp on a sixth of inputs, which can
    flip a slab decision taken on the boundary; those are re-decided by the scalar
    definition, so the flags and every closed-form ``q0`` are exactly the scalar's.
    """
    import math

    from pyradmc.geometry.aperture_gradient import _tip_support_start, tip_support_starts

    rng = np.random.default_rng(SEED)
    n = 4_000
    x = rng.uniform(-12.0, 12.0, n)
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    z = rng.uniform(30.0, 60.0, n)
    radius = rng.uniform(3.0, 16.0, n)
    z_top = z - rng.uniform(0.5, 4.0, n)
    z_bottom = z + rng.uniform(0.5, 4.0, n)
    # A third of the slabs get a face exactly on the scalar tangent point.
    for k in range(0, n, 3):
        c = x[k] + sign[k] * radius[k]
        root = math.sqrt(z[k] * z[k] + c * c - radius[k] * radius[k])
        q0 = (c * z[k] * z[k] - sign[k] * z[k] * radius[k] * root) / (
            z[k] * z[k] - radius[k] * radius[k]
        )
        norm = math.hypot(q0, z[k])
        w = (c * q0 + z[k] * z[k]) / norm * z[k] / norm
        if k % 2:
            z_top[k] = w
        else:
            z_bottom[k] = w
    got, closed = tip_support_starts(x, sign, z, radius, z_top, z_bottom)
    want = [
        _tip_support_start(*(float(a[k]) for a in (x, sign, z, radius, z_top, z_bottom)))
        for k in range(n)
    ]
    assert closed.tolist() == [w is not None for w in want]
    assert 0 < closed.sum() < n
    np.testing.assert_array_equal(got[closed], np.array([w for w in want if w is not None]))


def test_layout_cache_keeps_each_sources_cross_sections() -> None:
    """Two segmented fans sharing fan, template and segment count, different data.

    The position-free layout is memoised by fan and template; the cross-section
    source is not part of it and must come from each source's own segments.
    """
    first = _segments()
    other = AnalyticCrossSections()
    second = SegmentedFanSource(
        first.fan,
        first.template,
        other,
        jaw_neg=JAW_NEG,
        jaw_pos=JAW_POS,
        tips_neg=TIPS_NEG,
        tips_pos=TIPS_POS,
    )
    assert SegmentedEdgeSource(first).device_columns().cross_sections is XS
    assert SegmentedEdgeSource(second).device_columns().cross_sections is other


def test_the_arc_fields_differ_per_segment() -> None:
    """Guard on the fixture: each segment's edges see their own field and frame."""
    records = [e.device_record() for e in _per_edge(_arc_segments(), _arc_parameters())]
    per_segment = len(records) // len(ARC)
    for name in ("field", "r_range", "scale", "r_c"):
        rows = [tuple(getattr(r, name) for r in records[s::per_segment]) for s in range(3)]
        assert all(len(set(row)) == len(ARC) for row in rows), name
    origins = {tuple(r.stack.frame.origin) for r in records}
    assert len(origins) == len(ARC)


def test_arc_beamlets_are_segment_major() -> None:
    source = SegmentedEdgeSource(_arc_segments())
    assert list(source.parameters) == _arc_parameters()
    assert source.max_energy == SPECTRUM.edges[-1]


@pytest.mark.parametrize(("n_table", "defensive"), [(513, 0.05), (65, 0.2)])
def test_arc_device_columns_equal_the_per_edge_records(n_table: int, defensive: float) -> None:
    """Each segment's edges on its own fan and frame: the per-edge columns exactly."""
    segments = _arc_segments()
    source = SegmentedEdgeSource(segments, n_table=n_table, defensive=defensive)
    edges = _per_edge(segments, _arc_parameters(), n_table=n_table, defensive=defensive)
    assert_columns_equal(
        source.device_columns(), fan_edge_columns([e.device_record() for e in edges])
    )


def test_arc_device_columns_follow_with_settings() -> None:
    moved = _arc_segments().with_settings(
        tips_neg=_arc_segments().tips_neg - 0.1, tips_pos=_arc_segments().tips_pos + 0.1
    )
    source = SegmentedEdgeSource(moved)
    edges = _per_edge(moved, _arc_parameters())
    assert_columns_equal(
        source.device_columns(), fan_edge_columns([e.device_record() for e in edges])
    )


def test_arcs_sharing_a_template_do_not_share_a_layout() -> None:
    """Same template, segment count and spectrum, other frames: each its own geometry."""
    first = _arc_segments()
    turned = SegmentedFanSource(
        first.fans[::-1],
        first.template,
        XS,
        frames=first.frames[::-1],
        tips_neg=first.tips_neg,
        tips_pos=first.tips_pos,
        jaw_neg=first.jaw_neg[:, :1],
        jaw_pos=first.jaw_pos[:, :1],
    )
    SegmentedEdgeSource(first).device_columns()  # fills the layout cache first
    edges = _per_edge(turned, _arc_parameters())
    assert_columns_equal(
        SegmentedEdgeSource(turned).device_columns(),
        fan_edge_columns([e.device_record() for e in edges]),
    )


def test_arc_host_route_is_the_per_edge_route_draw_for_draw() -> None:
    segments = _arc_segments()
    source = SegmentedEdgeSource(segments)
    edges = _per_edge(segments, _arc_parameters())
    rng = HostRNG()
    for k in (0, 5, 9, 14, 18, 23):
        for i in range(5):
            assert source.emit(k, rng.init_state(SEED, i)) == edges[k].emit(rng.init_state(SEED, i))
        got = source.sample_beamlet_batch(SEED, 7, 50, k)
        want = edges[k].sample_batch(SEED, 7, 50)
        for key in want:
            np.testing.assert_array_equal(got[key], want[key])


def test_an_arc_segment_with_a_pair_outside_its_field_is_refused() -> None:
    """The segment's own field decides; the error names the segment."""
    from tests.unit.test_segmented_arc_source import arc_fan

    segments = _arc_segments()
    fans = list(segments.fans)
    fans[2] = arc_fan(225.0, SPECTRUM, couch=25.0, width_v=1.0, offset=(0.0, 4.0))
    narrow = SegmentedFanSource(
        fans, segments.template, XS, frames=segments.frames,
        tips_neg=segments.tips_neg, tips_pos=segments.tips_pos,
        jaw_neg=segments.jaw_neg[:, :1], jaw_pos=segments.jaw_pos[:, :1],
    )  # fmt: skip
    with pytest.raises(ValueError, match=r"segment 2.*outside the beam's field"):
        SegmentedEdgeSource(narrow)
