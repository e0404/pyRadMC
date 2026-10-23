"""An arc's control points as the segments of one source: one launch for every angle.

``SegmentedFanSource`` with one fan and one frame per segment — a plain beam at gantry
0, a 90-degree gantry with the collimator at 30 degrees (``u``/``v`` turned about
``w``), and a non-coplanar 225-degree beam (couch-like 25-degree turn about +z, so its
focal spot, frame origin and every axis leave the gantry plane) — against the
launches it replaces, one single-fan source per angle, and against the reference
engine. The leaf tips are asymmetric in ``v`` in every segment, so a frame read with
the wrong ``v`` row is not hidden by the phantom's mirror symmetry:

- a correlated arc column is *bit-identical* to the single-fan Dij column of that
  angle's fan and stack on the same device (same stream keys, fixed-point books);
- with independent seeds, the arc's columns agree statistically with the per-angle
  sources and with the reference engine (host ``emit``), and the plan view's
  ``total_weight * run(plan)`` with the MU-weighted sum of per-angle plan runs at
  equal total histories (``total_weight`` restored on both sides);
- the scheduling contract (group size bit-inert) and exact-type routing hold.

Statistical power, not convenience: scored on 2 x 4 x 2 cm bins, so in-field bins
collect tens of KERMA events per batch, masked on the pooled dose, equal histories
per side (the batched chi-squared is miscalibrated at unequal counts and powerless on
one-deposit-per-batch voxels; docs/decisions.md). Its power against a wrong segment
frame, swapped fans, a missing rotation and mis-indexed single components of the
frame and fan reads is recorded in docs/decisions.md ("Arcs in one launch").
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import MLC, BeamLimitingStack, JawPair
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from pyradmc.scoring.grid import ScoringGrid
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.unit.test_segmented_arc_source import ISO, arc_frame, beam_axes, focal

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
# (gantry, collimator, couch) per control point, degrees.
ORIENTATIONS = ((0.0, 0.0, 0.0), (90.0, 30.0, 0.0), (225.0, 0.0, 25.0))
ANGLES = tuple(o[0] for o in ORIENTATIONS)
# (width_u, width_v, centre offset along u, along v) where a fan differs from 10 x 8.
FIELDS = {ORIENTATIONS[2]: (9.0, 7.0, 0.5, -0.4)}
# Mid-plane settings (jaw at w = 33, MLC at w = 43): open, off-axis, narrow; the three
# leaf pairs differ in every segment (v-asymmetric apertures).
SETTINGS = (
    (-2.0, 2.0, (-1.5, -1.0, -0.5), (1.5, 1.0, 0.5)),
    (-0.5, 2.5, (-0.5, 0.0, -1.5), (2.0, 2.0, 0.5)),
    (-2.5, 0.5, (-0.3, -0.6, -1.0), (0.3, 0.6, 0.0)),
)
N_REFERENCE = 12_000  # histories per column, reference comparison
N_SOURCES = 40_000  # histories per column, per-angle comparison
PLAN_WEIGHTS = (1.0, 2.0, 3.0)
N_PLAN = 60_000  # histories per side of the plan comparison (10k/20k/30k per angle)
N_BATCHES = 8


def arc_fan(orientation: tuple[float, float, float]) -> SpectralBeamSource:
    """Each control point's fan; the non-coplanar one has its own, off-axis field."""
    u, v, _ = beam_axes(*orientation)
    width_u, width_v, du, dv = FIELDS.get(orientation, (10.0, 8.0, 0.0, 0.0))
    centre = tuple(c + du * a + dv * b for c, a, b in zip(ISO, u, v, strict=True))
    return SpectralBeamSource(
        SPECTRUM, focal(*orientation), centre, width_u, width_v, u_axis=u, v_axis=v
    )


def arc_stack(
    orientation: tuple[float, float, float], jaw_neg, jaw_pos, tips_neg, tips_pos
) -> BeamLimitingStack:
    # Thick, dense water stands in for tungsten: ~2-4 mean free paths when closed.
    return BeamLimitingStack(
        frame=arc_frame(*orientation),
        devices=(
            JawPair("u", 30.0, 36.0, jaw_neg, jaw_pos, material=WATER, density=4.0),
            MLC(
                z_top=40.0,
                z_bottom=46.0,
                leaf_edges_v=(-2.0, -0.6, 0.6, 2.0),
                tips_neg=tuple(tips_neg),
                tips_pos=tuple(tips_pos),
                tip_radius=8.0,
                material=WATER,
                density=4.0,
                focused_sides=True,
            ),
        ),
    )


def _stacks() -> list[BeamLimitingStack]:
    return [arc_stack(o, *s) for o, s in zip(ORIENTATIONS, SETTINGS, strict=True)]


def _arc(cls=SegmentedFanSource) -> SegmentedFanSource:
    return cls.from_stacks([arc_fan(o) for o in ORIENTATIONS], _stacks(), AnalyticCrossSections())


def _single(s: int) -> SegmentedFanSource:
    """The per-angle source the arc replaces: angle ``s``'s fan through its stack."""
    return SegmentedFanSource.from_stacks(
        arc_fan(ORIENTATIONS[s]), [_stacks()[s]], AnalyticCrossSections()
    )


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _warp(grid: VoxelGrid, device: str, **kwargs):
    from pyradmc.backends.warp.engine import WarpEngine

    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device, **kwargs)


def _require(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")


def _bins(grid: VoxelGrid) -> ScoringGrid:
    """2 cm along x and z (both in the gantry plane), 4 cm along y (the gantry axis)."""
    return ScoringGrid.rebin(grid, shape=(8, 4, 8), spacing=(2.0, 4.0, 2.0), origin=grid.origin)


def _mask(a_dose, a_sigma, b_dose, b_sigma):
    pooled = 0.5 * (a_dose + b_dose)
    return (pooled > 0.1 * pooled.max()) & (a_sigma > 0.0) & (b_sigma > 0.0)


def _assert_consistent(a_dose, a_sigma, b_dose, b_sigma, label: str) -> None:
    mask = _mask(a_dose, a_sigma, b_dose, b_sigma)
    assert mask.sum() >= 20, f"{label}: too few bins in the compared region"
    assert_chi2_consistent_batched(
        a_dose, a_sigma, N_BATCHES, b_dose, b_sigma, N_BATCHES, mask=mask
    )


def _common(grid: VoxelGrid) -> dict:
    return dict(n_batches=N_BATCHES, transport_electrons=False, scoring_grid=_bins(grid))


@pytest.mark.parametrize("device", DEVICES)
def test_a_correlated_arc_column_is_the_per_angle_column_bit_for_bit(device: str) -> None:
    """Same device, same seed, correlated keys: the arc launch is the per-angle launches."""
    _require(device)
    grid = _grid()
    engine = _warp(grid, device)
    kwargs = dict(n_histories_per_beamlet=3_000, n_batches=3, seed=SEED, truncation=0.0)
    arc = engine.run_dij(_arc(), **kwargs)
    for s in range(len(ANGLES)):
        alone = engine.run_dij(_single(s), **kwargs)
        np.testing.assert_array_equal(arc.column_dense(s), alone.column_dense(0))
        np.testing.assert_array_equal(arc.sigma_dense(s), alone.sigma_dense(0))
        assert arc.column_dense(s).max() > 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_arc_columns_match_the_per_angle_sources(device: str) -> None:
    _require(device)
    grid = _grid()
    engine = _warp(grid, device)
    common = _common(grid)
    arc = engine.run_dij(
        _arc(), n_histories_per_beamlet=N_SOURCES, seed=SEED, truncation=0.0, **common
    )
    for s in range(len(ANGLES)):
        alone = engine.run_dij(
            _single(s), n_histories_per_beamlet=N_SOURCES, seed=SEED + 30 + s, truncation=0.0,
            **common,
        )  # fmt: skip
        _assert_consistent(
            arc.column_dense(s),
            arc.sigma_dense(s),
            alone.column_dense(0),
            alone.sigma_dense(0),
            f"angle {ANGLES[s]}",
        )


@pytest.fixture(scope="module")
def reference_dij():
    grid = _grid()
    return ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run_dij(
        _arc(),
        n_histories_per_beamlet=N_REFERENCE,
        seed=SEED + 1,
        correlated=False,
        truncation=0.0,
        **_common(grid),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_arc_columns_match_the_reference(device: str, reference_dij) -> None:
    _require(device)
    grid = _grid()
    got = _warp(grid, device).run_dij(
        _arc(),
        n_histories_per_beamlet=N_REFERENCE,
        seed=SEED + 2,
        correlated=False,
        truncation=0.0,
        **_common(grid),
    )
    for s in range(len(ANGLES)):
        _assert_consistent(
            reference_dij.column_dense(s),
            reference_dij.sigma_dense(s),
            got.column_dense(s),
            got.sigma_dense(s),
            f"angle {ANGLES[s]}",
        )


@pytest.mark.parametrize("device", DEVICES)
def test_arc_plan_matches_the_weighted_per_angle_runs(device: str) -> None:
    """``W * run(arc.plan(w))`` against ``sum_s w_s * run(single_s.plan([1]))``.

    Equal total histories per side: the per-angle runs get ``N * w_s / W`` each,
    the allocation the plan view samples multinomially; the per-angle sum's sigma
    is the exact quadrature of independent runs.
    """
    _require(device)
    grid = _grid()
    engine = _warp(grid, device)
    common = _common(grid)
    plan = _arc().plan(PLAN_WEIGHTS)
    got = engine.run(plan, n_histories=N_PLAN, seed=SEED + 3, **common)
    total = plan.total_weight
    want = np.zeros_like(got.dose)
    var = np.zeros_like(got.dose)
    for s, w in enumerate(PLAN_WEIGHTS):
        n_s = int(N_PLAN * w / total)
        one = engine.run(_single(s).plan([1.0]), n_histories=n_s, seed=SEED + 40 + s, **common)
        want += w * one.dose
        var += (w * one.dose_sigma) ** 2
    _assert_consistent(total * got.dose, total * got.dose_sigma, want, np.sqrt(var), "plan")


def test_arc_plan_matches_the_reference() -> None:
    grid = _grid()
    common = _common(grid)
    plan = _arc().plan(PLAN_WEIGHTS)
    ref = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run(
        plan, n_histories=N_REFERENCE, seed=SEED + 4, **common
    )
    got = _warp(grid, "cpu").run(plan, n_histories=N_REFERENCE, seed=SEED + 5, **common)
    _assert_consistent(ref.dose, ref.dose_sigma, got.dose, got.dose_sigma, "plan vs reference")


def test_arc_beamlet_group_size_is_bit_inert() -> None:
    engine = _warp(_grid(), "cpu", chunk_size=1_000)
    kwargs = dict(n_histories_per_beamlet=2_000, n_batches=4, seed=SEED, transport_electrons=False)
    one = engine.run_dij(_arc(), beamlet_group_size=1, **kwargs)
    two = engine.run_dij(_arc(), beamlet_group_size=2, **kwargs)
    np.testing.assert_array_equal(one.indices, two.indices)
    np.testing.assert_array_equal(one.dose, two.dose)
    np.testing.assert_array_equal(one.sigma, two.sigma)
    assert one.energy_emitted == two.energy_emitted


def test_the_exact_arc_never_calls_the_host_samplers(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the exact type must be generated in-kernel")

    monkeypatch.setattr(SegmentedFanSource, "sample_beamlet_batch", refuse)
    monkeypatch.setattr(SegmentedFanSource, "emit", refuse)
    monkeypatch.setattr(SpectralBeamSource, "sample_batch", refuse)
    engine = _warp(_grid(), "cpu")
    kwargs = dict(n_batches=2, seed=SEED, transport_electrons=False)
    assert engine.run_dij(_arc(), n_histories_per_beamlet=400, **kwargs).energy_emitted > 0.0
    assert engine.run(_arc().plan(PLAN_WEIGHTS), n_histories=400, **kwargs).dose.max() > 0.0


class _HalfWeightFan(SpectralBeamSource):
    """A fan subclass whose primaries carry weight 0.5 before collimation."""

    def sample_batch(self, seed: int, history_offset: int, n: int):
        columns = super().sample_batch(seed, history_offset, n)
        columns["weight"] = np.full(n, 0.5, dtype=np.float32)
        return columns


def test_one_subclassed_fan_sends_the_arc_to_the_host_route() -> None:
    """An override in *any* segment's fan must be honoured, so the whole arc is host-sampled."""
    u, v, _ = beam_axes(*ORIENTATIONS[1])
    fans = [arc_fan(o) for o in ORIENTATIONS]
    fans[1] = _HalfWeightFan(SPECTRUM, focal(*ORIENTATIONS[1]), ISO, 10.0, 8.0, u_axis=u, v_axis=v)
    source = SegmentedFanSource.from_stacks(fans, _stacks(), AnalyticCrossSections())
    n = 600
    host = sum(
        float(
            np.sum(
                source.sample_beamlet_batch(SEED, b * 300, 300, s)["weight"].astype(np.float64)
                * fans[s].sample_batch(SEED, b * 300, 300)["energy"].astype(np.float64)
            )
        )
        for s in range(len(ANGLES))
        for b in range(2)
    )
    dij = _warp(_grid(), "cpu").run_dij(
        source, n_histories_per_beamlet=n, n_batches=2, seed=SEED, transport_electrons=False
    )
    assert dij.energy_emitted == pytest.approx(host, rel=1e-6)


def test_a_duck_typed_single_fan_keeps_the_host_route() -> None:
    """A non-spectral single fan (accepted before arcs existed) still runs, host-sampled."""
    from tests.unit.test_segmented_arc_source import DelegatingFan

    fan = DelegatingFan(arc_fan(ORIENTATIONS[0]))
    source = SegmentedFanSource.from_stacks(fan, [_stacks()[0]] * 2, AnalyticCrossSections())
    host = sum(
        float(
            np.sum(
                source.sample_beamlet_batch(SEED, b * 300, 300, s)["weight"].astype(np.float64)
                * fan.sample_batch(SEED, b * 300, 300)["energy"].astype(np.float64)
            )
        )
        for s in range(2)
        for b in range(2)
    )
    dij = _warp(_grid(), "cpu").run_dij(
        source, n_histories_per_beamlet=600, n_batches=2, seed=SEED, transport_electrons=False
    )
    assert dij.energy_emitted == pytest.approx(host, rel=1e-6)
