"""Aperture-sampled segment doses are the uniform-sampled doses, in expectation.

``SegmentedFanSource(..., aperture_sampling=delta)`` draws each segment's
histories mostly inside its opening and corrects the weight by the likelihood
ratio, so every Dij column — and the plan dose — must be unchanged in expectation.
Compared per column at **equal histories**, on coarse profile bins scored by the
engines (1 cm along the leaf travel, 4 cm across, two depth halves: tens of KERMA
events per bin and batch, true batch sigmas), masked on the *pooled* dose:

- Warp in-kernel, aperture-sampled vs uniform, a single fan and an arc;
- the host pre-sampling route (a subclass) aperture-sampled vs in-kernel uniform;
- the reference engine (``emit``) aperture-sampled vs Warp aperture-sampled;
- the plan view, aperture-sampled vs uniform.

The segments: a small opening with closed rows beside it, jaws narrower than the
leaf opening (one row closed), closed jaws (an empty region: the uniform
fallback), and an off-axis asymmetric opening. **Power**, at the same seeds: the
fan comparison fails for an inside weight 3 % too large and for a wrong component
share (the sampling density moved without its weight). Plus: identical segments
replay identical columns when correlated, the energy ledger closes, and the small
opening's in-field sigma drops at equal histories (the point of the exercise).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from pyradmc.scoring.grid import ScoringGrid
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.unit.test_segmented_arc_source import arc_fan, arc_frame

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
FRAME = BeamFrame(origin=FOCAL)
DELTA = 0.2
# Mid-plane settings (jaw at w = 33, MLC at w = 43; rows split at v = -0.6, 0.6).
SETTINGS = (
    (-2.0, 2.0, (0.0, -0.4, 0.0), (0.0, 0.4, 0.0)),  # small, closed rows beside it
    (-0.6, 0.6, (-1.5, -1.5, 0.8), (1.5, 1.5, 0.8)),  # jaws inside the leaves
    (0.0, 0.0, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),  # closed jaws: empty region
    (-2.0, 1.0, (0.2, 0.4, -0.6), (1.2, 0.9, 0.1)),  # off-axis, asymmetric
)
ARC = ((0.0, 0.0, 0.0), (90.0, 30.0, 0.0), (225.0, 0.0, 25.0))
ARC_SETTINGS = (SETTINGS[0], SETTINGS[1], SETTINGS[3])
N_BATCHES = 8
DENSITY = 12.0


def _stack(jaw_neg, jaw_pos, tips_neg, tips_pos, frame=FRAME) -> BeamLimitingStack:
    # Dense water stands in for tungsten: ~3.5 mean free paths at 2 MeV when closed.
    return BeamLimitingStack(
        frame=frame,
        devices=(
            JawPair("u", 30.0, 36.0, jaw_neg, jaw_pos, material=WATER, density=DENSITY),
            MLC(
                z_top=40.0,
                z_bottom=46.0,
                leaf_edges_v=(-2.0, -0.6, 0.6, 2.0),
                tips_neg=tuple(tips_neg),
                tips_pos=tuple(tips_pos),
                tip_radius=8.0,
                material=WATER,
                density=DENSITY,
                focused_sides=True,
            ),
        ),
    )


def _fan_source(delta=DELTA, cls=SegmentedFanSource) -> SegmentedFanSource:
    fan = SpectralBeamSource(
        SPECTRUM, focal_point=FOCAL, center=(8.0, 8.0, 0.0), width_u=10.0, width_v=8.0
    )
    return cls.from_stacks(
        fan, [_stack(*s) for s in SETTINGS], AnalyticCrossSections(), aperture_sampling=delta
    )


def _arc_source(delta=DELTA) -> SegmentedFanSource:
    fans, stacks = [], []
    for (angle, collimator, couch), setting in zip(ARC, ARC_SETTINGS, strict=True):
        offset = (0.5, -0.4) if couch else (0.0, 0.0)
        fans.append(
            arc_fan(
                angle, SPECTRUM, collimator=collimator, couch=couch, width_u=10.0, offset=offset
            )
        )
        stacks.append(_stack(*setting, frame=arc_frame(angle, collimator, couch)))
    return SegmentedFanSource.from_stacks(
        fans, stacks, AnalyticCrossSections(), aperture_sampling=delta
    )


class _HostRouted(SegmentedFanSource):
    """A subclass: the Warp engine host pre-samples it (``sample_beamlet_batch``)."""


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _warp(grid: VoxelGrid, device: str):
    from pyradmc.backends.warp.engine import WarpEngine

    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device)


def _require(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")


def _profiles(grid: VoxelGrid) -> ScoringGrid:
    """1 cm along u (leaf travel), 4 cm along v, two depth halves (see the segmented
    fan Dij tests: per-voxel KERMA bins have no chi-squared power)."""
    return ScoringGrid.rebin(grid, shape=(16, 4, 2), spacing=(1.0, 4.0, 8.0), origin=grid.origin)


def _kwargs(grid: VoxelGrid, n: int, seed: int) -> dict:
    return dict(
        n_histories_per_beamlet=n,
        n_batches=N_BATCHES,
        seed=seed,
        transport_electrons=False,
        truncation=0.0,
        scoring_grid=_profiles(grid),
    )


def _assert_column(a, b, s: int) -> None:
    """Batched chi-squared of column ``s`` of two Dij results, on the pooled-dose mask."""
    a_dose, a_sigma = a.column_dense(s), a.sigma_dense(s)
    b_dose, b_sigma = b.column_dense(s), b.sigma_dense(s)
    pooled = 0.5 * (a_dose + b_dose)
    mask = (pooled > 0.1 * pooled.max()) & (a_sigma > 0.0) & (b_sigma > 0.0)
    assert mask.sum() >= 12, f"segment {s}: too few bins in the compared region"
    assert_chi2_consistent_batched(
        a_dose, a_sigma, a.n_batches, b_dose, b_sigma, b.n_batches, mask=mask
    )


def _assert_columns(a, b) -> None:
    for s in range(a.n_beamlets):
        _assert_column(a, b, s)


def _mutant(source: SegmentedFanSource, kind: str) -> SegmentedFanSource:
    """Tamper with the uploaded tables: what a wrong weight or density would do.

    ``"weight"``: every inside factor 3 % too large. ``"share"``: the first occupied
    band of each region drawn 5 % more often (the next bands' cumulative shares
    shifted with it), its weights left alone — a sampling density that no longer
    matches the density the weight assumes.
    """
    table = source.aperture_table.copy()
    n = source.aperture_bands
    if kind == "weight":
        table[:, 1] *= 1.03
    else:
        for s in range(table.shape[0]):
            cdf = table[s, 2 : 3 + n]
            occupied = np.flatnonzero(np.diff(cdf) > 0.0) + 1
            if occupied.size > 1:
                first = occupied[0]
                step = 0.05 * (cdf[first] - cdf[first - 1])
                cdf[first:-1] = np.minimum(cdf[first:-1] + step, 1.0)
    table.setflags(write=False)
    source._aperture_table = table
    source._aperture_flat = table.reshape(-1)
    return source


# Histories per column of the Warp comparisons. At 1M the uniform side's in-field
# bins reach about 1 % (the aperture side's are smaller), so 3 % weight errors are
# visible. Calibrated over 30 reseeded pairs per source on cuda: fan 0-1 of 30 per
# column below 0.01 (1-4 below 0.05), arc 0 of 30 (0-2 below 0.05).
N_WARP = 1_000_000


@pytest.fixture(scope="module")
def uniform_fan():
    """The uniform-sampled fan Dij on Warp cpu (cpu and cuda draw the same streams)."""
    grid = _grid()
    return _warp(grid, "cpu").run_dij(_fan_source(delta=None), **_kwargs(grid, N_WARP, SEED))


@pytest.fixture(scope="module")
def aperture_fan():
    grid = _grid()
    return _warp(grid, "cpu").run_dij(_fan_source(), **_kwargs(grid, N_WARP, SEED + 1))


@pytest.mark.parametrize("device", DEVICES)
def test_aperture_columns_match_the_uniform_columns(device: str, uniform_fan) -> None:
    _require(device)
    grid = _grid()
    got = _warp(grid, device).run_dij(_fan_source(), **_kwargs(grid, N_WARP, SEED + 1))
    _assert_columns(uniform_fan, got)


@pytest.mark.parametrize("kind", ["weight", "share"])
def test_the_column_comparison_catches_a_mutant(kind: str, uniform_fan) -> None:
    """Power, same seeds: the comparison above fails for each tampered table."""
    grid = _grid()
    source = _mutant(_fan_source(), kind)
    got = _warp(grid, "cpu").run_dij(source, **_kwargs(grid, N_WARP, SEED + 1))
    with pytest.raises(AssertionError, match="not statistically consistent"):
        _assert_columns(uniform_fan, got)


@pytest.mark.parametrize("device", DEVICES)
def test_aperture_arc_columns_match_the_uniform_columns(device: str) -> None:
    _require(device)
    grid = _grid()
    engine = _warp(grid, device)
    uniform = engine.run_dij(_arc_source(delta=None), **_kwargs(grid, N_WARP, SEED))
    got = engine.run_dij(_arc_source(), **_kwargs(grid, N_WARP, SEED + 1))
    _assert_columns(uniform, got)
    if device == "cpu":  # power, same seeds (cpu and cuda are one statistical trial)
        mutant = engine.run_dij(_mutant(_arc_source(), "weight"), **_kwargs(grid, N_WARP, SEED + 1))
        with pytest.raises(AssertionError, match="not statistically consistent"):
            _assert_columns(uniform, mutant)


def test_host_route_aperture_columns_match_the_uniform_columns(uniform_fan) -> None:
    """The vectorized host batch (a subclass is host pre-sampled), with its power."""
    grid = _grid()
    engine = _warp(grid, "cpu")
    got = engine.run_dij(_fan_source(cls=_HostRouted), **_kwargs(grid, N_WARP, SEED + 2))
    _assert_columns(uniform_fan, got)
    mutant = _mutant(_fan_source(cls=_HostRouted), "weight")
    tampered = engine.run_dij(mutant, **_kwargs(grid, N_WARP, SEED + 2))
    with pytest.raises(AssertionError, match="not statistically consistent"):
        _assert_columns(uniform_fan, tampered)


def test_host_route_is_taken_by_the_subclass(monkeypatch) -> None:
    calls = []
    original = SegmentedFanSource.sample_beamlet_batch

    def spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_HostRouted, "sample_beamlet_batch", spy)
    _warp(_grid(), "cpu").run_dij(
        _fan_source(cls=_HostRouted), n_histories_per_beamlet=400, n_batches=2, seed=SEED
    )
    assert calls


# The reference engine is the oracle (its emit, per history): 12k per column, the
# segmented fan tests' reference count; KERMA, independent columns.
N_REFERENCE = 12_000


@pytest.fixture(scope="module")
def reference_aperture():
    grid = _grid()
    engine = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())
    return engine.run_dij(_fan_source(), correlated=False, **_kwargs(grid, N_REFERENCE, SEED + 3))


@pytest.mark.parametrize("device", DEVICES)
def test_reference_and_warp_aperture_columns_agree(device: str, reference_aperture) -> None:
    _require(device)
    grid = _grid()
    got = _warp(grid, device).run_dij(
        _fan_source(), correlated=False, **_kwargs(grid, N_REFERENCE, SEED + 4)
    )
    _assert_columns(reference_aperture, got)


@pytest.mark.parametrize("device", DEVICES)
def test_aperture_plan_matches_the_uniform_plan(device: str) -> None:
    _require(device)
    grid = _grid()
    engine = _warp(grid, device)
    weights = [1.0, 2.0, 0.5, 3.0]
    common = dict(
        n_histories=N_WARP,
        n_batches=N_BATCHES,
        transport_electrons=False,
        scoring_grid=_profiles(grid),
    )
    uniform = engine.run(_fan_source(delta=None).plan(weights), seed=SEED, **common)
    got = engine.run(_fan_source().plan(weights), seed=SEED + 1, **common)
    pooled = 0.5 * (uniform.dose + got.dose)
    mask = (pooled > 0.1 * pooled.max()) & (uniform.dose_sigma > 0) & (got.dose_sigma > 0)
    assert mask.sum() >= 12
    assert_chi2_consistent_batched(
        uniform.dose, uniform.dose_sigma, N_BATCHES, got.dose, got.dose_sigma, N_BATCHES, mask=mask
    )


def test_identical_segments_replay_one_column_when_correlated() -> None:
    stack = _stack(*SETTINGS[3])
    fan = _fan_source().fans[0]
    source = SegmentedFanSource.from_stacks(
        fan, [stack, stack, stack], AnalyticCrossSections(), aperture_sampling=DELTA
    )
    engine = _warp(_grid(), "cpu")
    kwargs = dict(n_histories_per_beamlet=4_000, n_batches=2, seed=SEED, beamlet_group_size=2)
    correlated = engine.run_dij(source, **kwargs)
    for s in (1, 2):  # within a group and across the group boundary
        np.testing.assert_array_equal(correlated.column_dense(0), correlated.column_dense(s))
    independent = engine.run_dij(source, correlated=False, **kwargs)
    assert not np.array_equal(independent.column_dense(0), independent.column_dense(1))


@pytest.mark.parametrize("device", DEVICES)
def test_aperture_ledger_closes(device: str) -> None:
    _require(device)
    dij = _warp(_grid(), device).run_dij(
        _fan_source(), n_histories_per_beamlet=20_000, n_batches=4, seed=SEED
    )
    total = dij.energy_deposited + dij.energy_escaped + dij.energy_unscored
    assert dij.energy_emitted == pytest.approx(total, rel=1e-6)


def test_aperture_sampling_lowers_the_in_field_sigma(uniform_fan, aperture_fan) -> None:
    """The point: at equal histories the in-field sigma of the small openings drops.

    Not a statistical oracle — a regression guard on the efficiency, at fixed seeds.
    Segment 2 (closed jaws) falls back to uniform sampling and is left out.
    """
    for s in (0, 1, 3):
        dose = uniform_fan.column_dense(s)
        field = dose > 0.5 * dose.max()
        ratio = np.median(aperture_fan.sigma_dense(s)[field] / uniform_fan.sigma_dense(s)[field])
        assert ratio < 0.85, f"segment {s}: sigma ratio {ratio:.2f}"
