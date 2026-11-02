"""Aperture importance sampling of a segmented fan, on the host.

With ``aperture_sampling=delta``, segment ``s``'s in-aperture point is drawn from
the defensive mixture ``q_s = delta * uniform(F) + (1 - delta) * uniform(A_s)``
over the fan's field ``F``, where ``A_s`` is the segment's opening mapped onto the
fan's aperture plane, and the history weight is multiplied by
``p / q_s = 1 / (delta + (1 - delta) |F| / |A_s| 1_{A_s})``. These tests pin, on
the host and without transport:

- the default (off) is today's source: four uniforms, the fan's own draws;
- the region ``A_s``: leaf openings widened by the margin, clipped by the jaws and
  the field, mapped through each segment's own frame (a single fan and an arc);
  closed pairs keep their margin band (their touching rounded tips leak); an empty
  region falls back to uniform sampling;
- the weight identity ``factor * q_s(x) = 1 / |F|`` at sampled points (float64);
- unbiasedness at the source level, ``E_q[factor * 1_R] = |R| / |F|`` over a grid
  of test cells, by a chi-squared with the full sample covariance — and its power:
  a 3 % weight error and a wrong component share are both caught at the same seed;
- one sampler: the vectorized batch and the scalar ``emit`` map the same five
  uniforms to the same primary, and the scalar sampler's uniform component is
  ``fan_primary``;
- ``with_settings`` carries the option and rebuilds the tables; invalid options
  are refused.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.fan import fan_primary, fan_primary_aperture
from pyradmc.geometry.segments import SegmentedFanSource, _aperture_offsets
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED
from tests.unit.test_segmented_arc_source import arc_fan, arc_frame

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
FRAME = BeamFrame(origin=FOCAL)
EDGES = (-3.0, -1.0, 1.0, 3.0)
JAW_Z = (30.0, 36.0)  # mid-plane 33
MLC_Z = (40.0, 46.0)  # mid-plane 43
TIP_RADIUS = 8.0
# Per-segment settings at the device mid-planes: (jaw neg, jaw pos, tips_neg, tips_pos).
SETTINGS = (
    # 0: a small opening in the middle row, the outer rows closed at the axis
    (-2.0, 2.0, (0.0, -0.3, 0.0), (0.0, 0.3, 0.0)),
    # 1: jaws narrower than the leaf opening, one row closed off-axis
    (-0.5, 0.5, (-1.5, -1.5, 0.8), (1.5, 1.5, 0.8)),
    # 2: closed jaws: an empty region (uniform fallback)
    (0.0, 0.0, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    # 3: an off-axis, asymmetric opening
    (-2.0, 1.0, (0.2, 0.4, -0.6), (1.2, 0.9, 0.1)),
)
DELTA = 0.2


def _fan() -> SpectralBeamSource:
    return SpectralBeamSource(
        SPECTRUM, focal_point=FOCAL, center=(8.0, 8.0, 0.0), width_u=8.0, width_v=8.0
    )


def _stack(jaw_neg, jaw_pos, tips_neg, tips_pos, frame=FRAME, density=1.0) -> BeamLimitingStack:
    return BeamLimitingStack(
        frame=frame,
        devices=(
            JawPair("u", *JAW_Z, jaw_neg, jaw_pos, material=WATER, density=density),
            MLC(
                z_top=MLC_Z[0],
                z_bottom=MLC_Z[1],
                leaf_edges_v=EDGES,
                tips_neg=tuple(tips_neg),
                tips_pos=tuple(tips_pos),
                tip_radius=TIP_RADIUS,
                material=WATER,
                density=density,
                focused_sides=True,
            ),
        ),
    )


def _source(delta=DELTA, margin=None, density=1.0, cls=SegmentedFanSource) -> SegmentedFanSource:
    return cls.from_stacks(
        _fan(),
        [_stack(*s, density=density) for s in SETTINGS],
        AnalyticCrossSections(),
        aperture_sampling=delta,
        aperture_margin=margin,
    )


# An arc: three gantry angles (the second collimator-rotated, the third off-axis and
# non-coplanar), each fan in its own frame.
ARC = ((0.0, 0.0, 0.0), (90.0, 30.0, 0.0), (225.0, 0.0, 25.0))
ARC_SETTINGS = (SETTINGS[0], SETTINGS[1], SETTINGS[3])


def _arc(delta=DELTA, margin=None, density=1.0) -> SegmentedFanSource:
    fans, stacks = [], []
    for (angle, collimator, couch), setting in zip(ARC, ARC_SETTINGS, strict=True):
        offset = (0.5, -0.4) if couch else (0.0, 0.0)
        fans.append(
            arc_fan(angle, SPECTRUM, collimator=collimator, couch=couch, width_u=9.0, offset=offset)
        )
        frame = arc_frame(angle, collimator, couch)
        stacks.append(_stack(*setting, frame=frame, density=density))
    return SegmentedFanSource.from_stacks(
        fans, stacks, AnalyticCrossSections(), aperture_sampling=delta, aperture_margin=margin
    )


class _Counting:
    """A host RNG state that counts its draws (``uniform`` calls ``random``)."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._rng.random())


class _Replay:
    """A host RNG state handing out recorded uniforms."""

    def __init__(self, values) -> None:
        self._values = iter(np.asarray(values, dtype=np.float64).tolist())

    def random(self) -> float:
        return next(self._values)


def _fan_args(fan: SpectralBeamSource) -> list[float]:
    return [*fan.focal_point, *fan.center, *fan.u_axis, *fan.v_axis, fan.width_u, fan.width_v]


def _scalar(source: SegmentedFanSource, segment: int, uniforms) -> tuple[float, ...]:
    fan = source.fans[segment]
    table = source.aperture_table
    return fan_primary_aperture(
        _Replay(uniforms),
        SPECTRUM.edges,
        SPECTRUM.cdf,
        SPECTRUM.cdf.size,
        *_fan_args(fan),
        table.reshape(-1),
        segment * table.shape[1],
        source.aperture_bands,
    )


def _offsets(source: SegmentedFanSource, segment: int, u: np.ndarray, table=None):
    """Vectorized (du, dv, factor) of segment ``segment`` from (n, 3) uniforms."""
    fan = source.fans[segment]
    table = source.aperture_table if table is None else table
    return _aperture_offsets(table[segment], source.aperture_bands, fan.width_u, fan.width_v, u)


# --- default off -----------------------------------------------------------------------


def test_default_is_off_and_draws_the_fans_four_uniforms() -> None:
    plain = SegmentedFanSource.from_stacks(
        _fan(), [_stack(*s) for s in SETTINGS], AnalyticCrossSections()
    )
    assert plain.aperture_sampling is None
    assert plain.aperture_table is None and plain.aperture_fraction is None
    off = _source(delta=None)
    for s in range(len(SETTINGS)):
        state = _Counting(SEED + s)
        got = off.emit(s, state)
        assert state.calls == 4
        want = plain.emit(s, HostRNG().init_state(SEED, s))
        again = off.emit(s, HostRNG().init_state(SEED, s))
        assert again == want
        assert got.weight <= 1.0
        a = off.sample_beamlet_batch(SEED, 17, 300, s)
        b = plain.sample_beamlet_batch(SEED, 17, 300, s)
        for key in b:
            np.testing.assert_array_equal(a[key], b[key])


def test_aperture_sampling_draws_five_uniforms_per_history() -> None:
    source = _source()
    for s in range(len(SETTINGS)):
        state = _Counting(SEED + s)
        source.emit(s, state)
        assert state.calls == 5
    plan = source.plan([1.0, 1.0, 1.0, 1.0])
    state = _Counting(SEED)
    plan.emit(state)
    assert state.calls == 6  # the segment choice, then the segment's five


# --- the region A_s ----------------------------------------------------------------


def test_default_margin_is_the_rounded_tips_partial_band() -> None:
    """Beyond ``R - sqrt(R^2 - (H/2)^2)`` from the apex a vertical ray cuts the full leaf."""
    height = MLC_Z[1] - MLC_Z[0]
    want = TIP_RADIUS - math.sqrt(TIP_RADIUS**2 - (height / 2) ** 2)
    assert _source().aperture_margins == pytest.approx((want,))
    assert _source(margin=0.25).aperture_margins == (0.25,)


def test_region_is_the_leaf_openings_plus_margin_in_the_fan_plane() -> None:
    """Segment 0 by hand: one row open at +-0.3, two closed rows at 0, margin 0.1.

    The fan is centred on the frame's axis with its aperture plane at w = 100, so a
    mid-plane coordinate t of the MLC (w = 43) is the aperture offset t * 100 / 43;
    the strip edges +-1 land at +-2.33, the outer ones beyond the 8 cm field.
    """
    source = _source(margin=0.1)
    k = 100.0 / 43.0
    v_edges, u_lo, u_hi = source.aperture_region(0)
    occupied = u_hi > u_lo
    np.testing.assert_allclose(v_edges[0], -4.0)
    np.testing.assert_allclose(v_edges[-1], 4.0)
    centre = (v_edges[:-1] + v_edges[1:]) / 2
    middle = occupied & (np.abs(centre) < k)
    outer = occupied & (np.abs(centre) > k)
    np.testing.assert_allclose(u_lo[middle], -0.4 * k)
    np.testing.assert_allclose(u_hi[middle], 0.4 * k)
    np.testing.assert_allclose(u_lo[outer], -0.1 * k)
    np.testing.assert_allclose(u_hi[outer], 0.1 * k)
    area = 2 * k * 0.8 * k + 2 * (4.0 - k) * 0.2 * k
    assert source.aperture_fraction[0] == pytest.approx(area / 64.0, rel=1e-12)


def test_jaws_narrower_than_the_leaves_clip_the_region() -> None:
    """Segment 1: jaws +-0.5 at w = 33 (+-1.52 in the fan plane) inside leaves at +-1.5."""
    source = _source(margin=0.1)
    _, u_lo, u_hi = source.aperture_region(1)
    jaw = 0.5 * 100.0 / 33.0
    occupied = u_hi > u_lo
    assert u_lo[occupied].min() == pytest.approx(-jaw)
    assert u_hi[occupied].max() == pytest.approx(jaw)
    # The closed row (tips at 0.8, margin 0.1 -> 0.7..0.9 at w = 43) lies beyond the jaw:
    # the bands above the middle strip's edge (100 / 43 in the fan plane) are empty.
    assert jaw < 0.7 * 100.0 / 43.0
    v_edges, _, _ = source.aperture_region(1)
    assert not np.any(occupied & (v_edges[:-1] >= 100.0 / 43.0 - 1e-12))


def test_an_empty_region_falls_back_to_uniform_sampling() -> None:
    source = _source()
    assert source.aperture_fraction[2] == 0.0
    row = source.aperture_table[2]
    assert row[0] == row[1] == 1.0  # factor 1 in and out
    n = source.aperture_bands
    np.testing.assert_array_equal(row[2 : 3 + n], 1.0)  # always the uniform component
    u = np.random.default_rng(SEED).random((2_000, 3))
    du, dv, factor = _offsets(source, 2, u)
    np.testing.assert_array_equal(factor, 1.0)
    np.testing.assert_array_equal(du, (u[:, 1] - 0.5) * 8.0)
    np.testing.assert_array_equal(dv, (u[:, 2] - 0.5) * 8.0)


def test_every_open_ray_lies_in_the_region() -> None:
    """Coverage: with dense leaves, any uniform ray through an opening is in ``A_s``.

    Not needed for unbiasedness (an uncovered opening costs efficiency only), but a
    mapping error — a wrong frame, scale or axis — leaves open rays outside.
    """
    for source in (_source(density=10.0), _arc(density=10.0)):
        fan_widths = [(f.width_u, f.width_v) for f in source.fans]
        u = np.random.default_rng(SEED).random((20_000, 2))
        for s in range(source.n_segments):
            if source.aperture_fraction[s] == 0.0:
                continue
            width_u, width_v = fan_widths[s]
            du, dv = (u[:, 0] - 0.5) * width_u, (u[:, 1] - 0.5) * width_v
            transmission = _transmission_at(source, s, du, dv)
            inside = source.proposal_density(s, du, dv) > DELTA / (width_u * width_v) * 1.0001
            open_rays = transmission > 0.6
            assert open_rays.sum() > 100
            assert np.all(inside[open_rays]), f"segment {s}: open rays outside the region"


def _transmission_at(source: SegmentedFanSource, segment: int, du, dv) -> np.ndarray:
    fan = source.fans[segment]
    target = (
        np.asarray(fan.center)
        + du[:, None] * np.asarray(fan.u_axis)
        + dv[:, None] * np.asarray(fan.v_axis)
    )
    direction = target - np.asarray(fan.focal_point)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    origins = np.broadcast_to(np.asarray(fan.focal_point), direction.shape)
    return source.transmission(segment, np.full(du.size, 2.0), origins, direction)


@pytest.mark.parametrize("which", ["fan", "arc"])
def test_region_samples_cross_the_mlc_inside_the_widened_opening(which: str) -> None:
    """Rays drawn from the region pass the MLC mid-plane inside ``[tip - m, tip + m]``.

    Checked in each segment's own frame, through the engine-frame ray — the mapping
    from device mid-plane to fan plane, per segment, end to end.
    """
    margin = 0.1
    source = _source(margin=margin) if which == "fan" else _arc(margin=margin)
    mid = 0.5 * sum(MLC_Z)
    jaw_mid = 0.5 * sum(JAW_Z)
    for s in range(source.n_segments):
        if source.aperture_fraction[s] == 0.0:
            continue
        rng = np.random.default_rng(SEED + s)
        u = np.column_stack(
            [rng.random(5_000) * 0.999 * (1 - DELTA) + DELTA, rng.random((5_000, 2))]
        )
        du, dv, _ = _offsets(source, s, u)
        fan, frame = source.fans[s], source.frames[s]
        target = (
            np.asarray(fan.center)
            + du[:, None] * np.asarray(fan.u_axis)
            + dv[:, None] * np.asarray(fan.v_axis)
        )
        _, d_local = frame.to_local(np.zeros((du.size, 3)), target - np.asarray(fan.focal_point))
        t_u = d_local[:, 0] / d_local[:, 2] * mid
        t_v = d_local[:, 1] / d_local[:, 2] * mid
        row = np.searchsorted(EDGES, t_v, side="right") - 1
        assert np.all((row >= 0) & (row < 3))
        lo = source.tips_neg[s][row] - margin
        hi = source.tips_pos[s][row] + margin
        assert np.all((t_u >= lo - 1e-9) & (t_u <= hi + 1e-9))
        jaw_u = d_local[:, 0] / d_local[:, 2] * jaw_mid
        assert np.all(jaw_u >= source.jaw_neg[s, 0] - 1e-9)
        assert np.all(jaw_u <= source.jaw_pos[s, 0] + 1e-9)


# --- the weight ------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["fan", "arc"])
def test_weight_times_proposal_density_is_the_uniform_density(which: str) -> None:
    """``factor(x) * q_s(x) = 1 / |F|`` at sampled points, inside and outside ``A_s``."""
    source = _source() if which == "fan" else _arc()
    u = np.random.default_rng(SEED).random((20_000, 3))
    for s in range(source.n_segments):
        fan = source.fans[s]
        du, dv, factor = _offsets(source, s, u)
        q = source.proposal_density(s, du, dv)
        np.testing.assert_allclose(factor * q, 1.0 / (fan.width_u * fan.width_v), rtol=1e-12)
        if source.aperture_fraction[s] > 0.0:
            assert np.any(np.isclose(factor, 1.0 / DELTA))  # outside draws present
            assert np.any(factor < 1.0)  # inside draws present
        # And the scalar sampler, on the host's per-history stream.
        rng = HostRNG()
        for h in range(200):
            *_, f = _scalar(source, s, rng.init_state(SEED, h).random(5))
            assert f in (source.aperture_table[s, 0], source.aperture_table[s, 1])


def _cell_chi2(source, segment, n, seed, table=None) -> float:
    """p-value of ``E_q[factor * 1_R] = |R| / |F|`` over a 10 x 9 grid of cells ``R``.

    The grid is not aligned with the region's edges, so it straddles them. One
    statistic over the cells with the full sample covariance (each draw lands in
    exactly one cell, so the cell estimators are anti-correlated); the last cell is
    left out, because under uniform sampling (factor 1) the cells sum to exactly 1.
    """
    fan = source.fans[segment]
    u = np.random.default_rng(seed).random((n, 3))
    du, dv, factor = _offsets(source, segment, u, table)
    nu, nv = 10, 9
    iu = np.clip(((du / fan.width_u + 0.5) * nu).astype(int), 0, nu - 1)
    iv = np.clip(((dv / fan.width_v + 0.5) * nv).astype(int), 0, nv - 1)
    cell = iu * nv + iv
    sums = np.bincount(cell, weights=factor, minlength=nu * nv)
    squares = np.bincount(cell, weights=factor**2, minlength=nu * nv)
    mean = (sums / n)[:-1]
    covariance = (np.diag(squares[:-1] / n) - np.outer(mean, mean)) / n
    d = mean - 1.0 / (nu * nv)
    chi2 = float(d @ np.linalg.solve(covariance, d))
    return float(stats.chi2.sf(chi2, d.size))


@pytest.mark.parametrize("which", ["fan", "arc"])
def test_the_proposal_is_unbiased_over_the_field(which: str) -> None:
    source = _source() if which == "fan" else _arc()
    for s in range(source.n_segments):
        assert _cell_chi2(source, s, 400_000, SEED + s) > 0.01, f"segment {s}"


def test_the_unbiasedness_check_has_power() -> None:
    """The same check, same seeds, catches a 3 % inside-weight error and a wrong share."""
    source = _source()
    s = 3
    assert _cell_chi2(source, s, 400_000, SEED + s) > 0.01
    heavy = source.aperture_table.copy()
    heavy[s, 1] *= 1.03
    assert _cell_chi2(source, s, 400_000, SEED + s, heavy) < 1e-6
    shifted = source.aperture_table.copy()
    n = source.aperture_bands
    cdf = shifted[s, 2 : 3 + n]
    occupied = np.flatnonzero(np.diff(cdf) > 0) + 1  # cdf entries ending an occupied band
    first = occupied[0]
    cdf[first:-1] = np.minimum(cdf[first:-1] + 0.05 * (cdf[first] - cdf[first - 1]), 1.0)
    assert _cell_chi2(source, s, 400_000, SEED + s, shifted) < 1e-6


# --- one sampler -----------------------------------------------------------------------


def test_the_uniform_component_is_fan_primary() -> None:
    """On the empty-region table the aperture sampler is ``fan_primary`` minus one draw."""
    source = _source()
    fan = source.fans[2]
    for u in np.random.default_rng(SEED).random((300, 5)):
        got = _scalar(source, 2, u)
        want = fan_primary(
            _Replay(u[[0, 1, 3, 4]]),
            SPECTRUM.edges,
            SPECTRUM.cdf,
            SPECTRUM.cdf.size,
            *_fan_args(fan),
        )
        assert got[:4] == want
        assert got[4] == 1.0


@pytest.mark.parametrize("which", ["fan", "arc"])
def test_vectorized_batch_is_the_scalar_sampler(which: str) -> None:
    """Row ``h`` of the batch is the scalar sampler on uniforms ``5h .. 5h + 4``."""
    source = _source() if which == "fan" else _arc()
    offset, n = 40, 400
    bitgen = np.random.PCG64(SEED)
    bitgen.advance(5 * offset)
    u = np.random.Generator(bitgen).random((n, 5))
    for s in range(source.n_segments):
        batch = source.sample_beamlet_batch(SEED, offset, n, s)
        fan = source.fans[s]
        for h in range(0, n, 7):
            energy, ux, uy, uz, factor = _scalar(source, s, u[h])
            assert batch["energy"][h] == np.float32(energy)
            np.testing.assert_array_equal(
                [batch["ux"][h], batch["uy"][h], batch["uz"][h]],
                np.array([ux, uy, uz], dtype=np.float32),
            )
            # The weight is the transmission of the ray as transported (float32).
            origin = np.array([fan.focal_point], dtype=np.float32).astype(np.float64)
            ray = np.array([[ux, uy, uz]], dtype=np.float32).astype(np.float64)
            energy32 = np.array([energy], dtype=np.float32).astype(np.float64)
            weight = factor * source.transmission(s, energy32, origin, ray)
            assert batch["weight"][h] == np.float32(weight[0])
        for key in ("x", "y", "z"):
            np.testing.assert_array_equal(batch[key], np.float32(fan.focal_point["xyz".index(key)]))


def test_emit_is_the_scalar_sampler_and_its_transmission() -> None:
    source = _source()
    rng = HostRNG()
    for s in range(source.n_segments):
        fan = source.fans[s]
        for h in range(50):
            got = source.emit(s, rng.init_state(SEED, h))
            energy, ux, uy, uz, factor = _scalar(source, s, rng.init_state(SEED, h).random(5))
            assert (got.energy, got.ux, got.uy, got.uz) == (energy, ux, uy, uz)
            assert (got.x, got.y, got.z) == fan.focal_point
            transmission = source.transmission(
                s, np.array([energy]), np.array([fan.focal_point]), np.array([[ux, uy, uz]])
            )
            assert got.weight == pytest.approx(factor * transmission[0], rel=1e-15)


def test_batch_is_chunk_invariant_and_segment_blind() -> None:
    source = _source()
    whole = source.sample_beamlet_batch(SEED, 0, 900, 0)
    parts = [source.sample_beamlet_batch(SEED, o, 300, 0) for o in (0, 300, 600)]
    for key in whole:
        np.testing.assert_array_equal(whole[key], np.concatenate([p[key] for p in parts]))
    # Every segment draws the same uniforms: the same energies, its own rays.
    energies = [source.sample_beamlet_batch(SEED, 0, 900, s)["energy"] for s in range(4)]
    for e in energies[1:]:
        np.testing.assert_array_equal(e, energies[0])


def test_identical_segments_replay_identical_primaries() -> None:
    stack = _stack(*SETTINGS[3])
    source = SegmentedFanSource.from_stacks(
        _fan(), [stack, stack], AnalyticCrossSections(), aperture_sampling=DELTA
    )
    a = source.sample_beamlet_batch(SEED, 0, 500, 0)
    b = source.sample_beamlet_batch(SEED, 0, 500, 1)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_plan_batch_rows_are_their_segments_aperture_batch() -> None:
    source = _source()
    plan = source.plan([1.0, 2.0, 0.5, 3.0])
    got = plan.sample_batch(SEED, 11, 800)
    segment = plan.segment_of(SEED, 11, 800)
    for s in range(4):
        want = source.sample_beamlet_batch(SEED, 11, 800, s)
        rows = segment == s
        for key in want:
            np.testing.assert_array_equal(got[key][rows], want[key][rows])


# --- with_settings and validation ---------------------------------------------------


def test_with_settings_carries_the_option_and_rebuilds_the_tables() -> None:
    source = _source(margin=0.15)
    tips_neg = source.tips_neg.copy()
    tips_pos = source.tips_pos.copy()
    tips_neg[0, 1], tips_pos[0, 1] = -0.9, 0.6
    moved = source.with_settings(tips_neg=tips_neg, tips_pos=tips_pos)
    assert moved.aperture_sampling == DELTA
    assert moved.aperture_margins == (0.15,)
    assert moved.aperture_bands == source.aperture_bands
    fresh = SegmentedFanSource(
        _fan(),
        source.template,
        AnalyticCrossSections(),
        jaw_neg=source.jaw_neg[:, :1],
        jaw_pos=source.jaw_pos[:, :1],
        tips_neg=tips_neg,
        tips_pos=tips_pos,
        aperture_sampling=DELTA,
        aperture_margin=0.15,
    )
    np.testing.assert_array_equal(moved.aperture_table, fresh.aperture_table)
    assert not np.array_equal(moved.aperture_table[0], source.aperture_table[0])
    np.testing.assert_array_equal(moved.aperture_table[1:], source.aperture_table[1:])
    assert not moved.aperture_table.flags.writeable
    arc = _arc()
    np.testing.assert_array_equal(arc.with_settings().aperture_table, arc.aperture_table)
    assert source.with_settings().aperture_sampling == DELTA
    assert _source(delta=None).with_settings().aperture_sampling is None


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_a_defensive_share_outside_the_unit_interval_is_refused(delta: float) -> None:
    with pytest.raises(ValueError, match="aperture_sampling"):
        _source(delta=delta)


@pytest.mark.parametrize(
    "delta", [1e-17, 2.0**-54, 1e-12, 1e-8, 2.0**-24, 1e-6, math.nextafter(2.0**-10, 0.0)]
)
def test_a_defensive_share_too_small_to_realize_is_refused(delta: float) -> None:
    """Below ``2**-10`` the share is refused instead of silently biasing the dose.

    At ``1e-17`` the float64 table stores a share ``1 - (1 - delta) = 0`` and a weight
    ``1 / 0``: the whole field is never drawn and the blocked region's transmission
    is lost. At ``1e-8`` the float64 table is fine, but the Warp generators pick the
    component with a 24-bit uniform, which lands below the float32 share with
    probability ``2**-24``, not ``1e-8``, at weight ``1e8``: the blocked region's
    contribution comes out 6x too large.
    """
    with pytest.raises(ValueError, match=r"aperture_sampling.*2\*\*-10"):
        _source(delta=delta)


def _realized_share(table32: np.ndarray) -> float:
    """The whole field's share as a 24-bit uniform realizes it: ``P(u < cdf32[0])``."""
    return math.ceil(float(table32[2]) * 2**24) / 2**24


@pytest.mark.parametrize("delta", [2.0**-10, 1e-3, 0.05, 0.2, 0.5, 1.0 - 2.0**-10])
def test_an_accepted_share_keeps_the_expectation_in_float32(delta: float) -> None:
    """Every accepted share survives both tables: the weight times the drawn share is 1.

    Host (float64 uniforms): ``factor_out * cdf[0] = 1`` to rounding. Device (the
    float32 cast the Warp upload makes, 24-bit uniforms): the share realized by the
    component draw times the float32 ``factor_out`` is 1 within ``2**-24 / delta <=
    2**-14``, the bound the refusal guarantees.
    """
    source = _source(delta=delta)
    table = source.aperture_table
    table32 = table.astype(np.float32)
    for s in range(source.n_segments):
        if source.aperture_fraction[s] == 0.0:
            continue
        assert np.all(np.isfinite(table[s])) and np.all(np.isfinite(table32[s]))
        assert table[s, 0] * table[s, 2] == pytest.approx(1.0, rel=1e-15)
        realized = _realized_share(table32[s]) * float(table32[s, 0])
        assert abs(realized - 1.0) <= 2.0**-24 / delta + 1e-7


@pytest.mark.parametrize("margin", [-0.1, float("inf"), float("nan")])
def test_a_bad_margin_is_refused(margin: float) -> None:
    with pytest.raises(ValueError, match="aperture_margin"):
        _source(margin=margin)


def test_a_margin_without_aperture_sampling_is_refused() -> None:
    with pytest.raises(ValueError, match="aperture_margin"):
        _source(delta=None, margin=0.1)


def test_a_single_fan_not_aligned_with_its_frame_is_refused_only_when_sampling() -> None:
    s = 1.0 / math.sqrt(2.0)
    fan = SpectralBeamSource(
        SPECTRUM, FOCAL, (8.0, 8.0, 0.0), 8.0, 8.0, u_axis=(s, s, 0.0), v_axis=(-s, s, 0.0)
    )
    stacks = [_stack(*SETTINGS[0])]
    SegmentedFanSource.from_stacks(fan, stacks, AnalyticCrossSections())  # accepted, off
    with pytest.raises(ValueError, match="u_axis"):
        SegmentedFanSource.from_stacks(
            fan, stacks, AnalyticCrossSections(), aperture_sampling=DELTA
        )


def test_a_fan_subclass_is_refused_when_sampling() -> None:
    class Halved(SpectralBeamSource):
        pass

    fan = Halved(SPECTRUM, focal_point=FOCAL, center=(8.0, 8.0, 0.0), width_u=8.0, width_v=8.0)
    stacks = [_stack(*SETTINGS[0])]
    SegmentedFanSource.from_stacks(fan, stacks, AnalyticCrossSections())  # accepted, off
    with pytest.raises(ValueError, match="SpectralBeamSource"):
        SegmentedFanSource.from_stacks(
            fan, stacks, AnalyticCrossSections(), aperture_sampling=DELTA
        )


class _LegacySubclass(SegmentedFanSource):
    """A user subclass written against the constructor before aperture sampling."""

    def __init__(
        self,
        fan,
        template,
        cross_sections,
        *,
        jaw_neg=None,
        jaw_pos=None,
        tips_neg=None,
        tips_pos=None,
        e_min=0.010,
        n_table=512,
        attenuation_tables=None,
        frames=None,
    ) -> None:
        super().__init__(
            fan,
            template,
            cross_sections,
            jaw_neg=jaw_neg,
            jaw_pos=jaw_pos,
            tips_neg=tips_neg,
            tips_pos=tips_pos,
            e_min=e_min,
            n_table=n_table,
            attenuation_tables=attenuation_tables,
            frames=frames,
        )


def test_a_subclass_without_the_option_still_builds_and_resets() -> None:
    source = _LegacySubclass.from_stacks(
        _fan(), [_stack(*s) for s in SETTINGS], AnalyticCrossSections()
    )
    moved = source.with_settings(tips_neg=source.tips_neg - 0.1)
    assert type(moved) is _LegacySubclass
    np.testing.assert_array_equal(moved.tips_neg, source.tips_neg - 0.1)
