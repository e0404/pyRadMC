"""A segmented fan's Dij: one column per collimator setting, from one engine launch.

Each column of ``run_dij(SegmentedFanSource)`` estimates the dose per fan history
through that segment's stack under ``mode="attenuation"`` physics. Three anchors:

- the reference engine (host ``emit``) against the Warp in-kernel generator, column
  for column — two unbiased estimators of one quantity;
- the route it replaces — ``presolve_head_device(mode="attenuation")`` plus a
  ``run()`` per segment — against the in-kernel column of the same segment;
- routing: the exact type is generated in-kernel (its host samplers are never
  called), while a subclass, or a subclass *fan*, keeps host pre-sampling so an
  override is honoured.

Plus the scheduling contract every Dij route keeps on one device: the beamlet group
size is bit-inert. Analytic water "collimators" keep the tier EPDL-free; electrons
are off (KERMA) where the reference engine runs, to keep the oracle fast.
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
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
FRAME = BeamFrame(origin=FOCAL)
# Mid-plane settings at w = 43 (0.43 x the surface scale): open, off-axis, narrow.
SETTINGS = (
    (-2.0, 2.0, (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)),
    (-0.5, 2.5, (-0.5, 0.0, -0.5), (2.0, 2.0, 2.0)),
    (-2.5, 0.5, (-0.3, -0.6, -0.3), (0.3, 0.6, 0.3)),
)


def _fan() -> SpectralBeamSource:
    return SpectralBeamSource(
        SPECTRUM, focal_point=FOCAL, center=(8.0, 8.0, 0.0), width_u=10.0, width_v=8.0
    )


def _stack(jaw_neg: float, jaw_pos: float, tips_neg, tips_pos) -> BeamLimitingStack:
    # Thick, dense water stands in for tungsten: ~2-4 mean free paths when closed.
    return BeamLimitingStack(
        frame=FRAME,
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
    return [_stack(*setting) for setting in SETTINGS]


def _source() -> SegmentedFanSource:
    return SegmentedFanSource.from_stacks(_fan(), _stacks(), AnalyticCrossSections())


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


def _assert_columns_consistent(a, b, columns) -> None:
    for s in columns:
        a_dose, a_sigma = a.column_dense(s), a.sigma_dense(s)
        b_dose, b_sigma = b.column_dense(s), b.sigma_dense(s)
        mask = (a_dose > 0.1 * a_dose.max()) & (a_sigma > 0.0) & (b_sigma > 0.0)
        assert mask.sum() > 50, f"segment {s}: too few voxels in the compared region"
        assert_chi2_consistent_batched(
            a_dose, a_sigma, a.n_batches, b_dose, b_sigma, b.n_batches, mask=mask
        )


@pytest.mark.parametrize("device", DEVICES)
def test_inkernel_columns_match_the_reference(device: str) -> None:
    _require(device)
    grid = _grid()
    source = _source()
    kwargs = dict(
        n_histories_per_beamlet=2_400,
        n_batches=8,
        seed=SEED,
        transport_electrons=False,
        correlated=False,
        truncation=0.0,
    )
    ref = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run_dij(
        source, **kwargs
    )
    got = _warp(grid, device).run_dij(source, **kwargs)
    _assert_columns_consistent(ref, got, range(source.n_segments))


@pytest.mark.parametrize("device", DEVICES)
def test_inkernel_columns_match_presolve_plus_run(device: str) -> None:
    """Each column equals the per-segment route it replaces: pre-solve, then run."""
    _require(device)
    from pyradmc.backends.warp.presolve import presolve_head_device

    grid = _grid()
    engine = _warp(grid, device)
    source = _source()
    n = 40_000
    kerma = dict(transport_electrons=False)
    dij = engine.run_dij(
        source, n_histories_per_beamlet=n, n_batches=8, seed=SEED, truncation=0.0, **kerma
    )
    assert dij.correlated  # the shipped sampling configuration
    for s, stack in enumerate(_stacks()):
        phase_space = presolve_head_device(
            _fan(),
            cross_sections=AnalyticCrossSections(),
            n_histories=n,
            seed=SEED + 10 + s,
            exit_z=60.0,
            stack=stack,
            mode="attenuation",
            device=device,
            return_device_source=True,
        )
        single = engine.run(phase_space, n_histories=n, n_batches=8, seed=SEED + 20 + s, **kerma)
        column, sigma = dij.column_dense(s), dij.sigma_dense(s)
        mask = (single.dose > 0.1 * single.dose.max()) & (single.dose_sigma > 0.0)
        mask &= sigma > 0.0
        assert mask.sum() > 200
        assert_chi2_consistent_batched(
            single.dose, single.dose_sigma, 8, column, sigma, dij.n_batches, mask=mask
        )


@pytest.mark.parametrize("device", DEVICES)
def test_inkernel_ledger_closes(device: str) -> None:
    _require(device)
    dij = _warp(_grid(), device).run_dij(
        _source(), n_histories_per_beamlet=4_000, n_batches=4, seed=SEED
    )
    total = dij.energy_deposited + dij.energy_escaped + dij.energy_unscored
    assert dij.energy_emitted == pytest.approx(total, rel=1e-6)
    # Attenuated: well below the open fan's book, above the all-blocked one.
    open_book = 3 * 4_000 * SPECTRUM.mean_energy
    assert 0.2 < dij.energy_emitted / open_book < 0.9


def test_beamlet_group_size_is_bit_inert() -> None:
    grid = _grid()
    engine = _warp(grid, "cpu", chunk_size=1_000)
    kwargs = dict(n_histories_per_beamlet=2_000, n_batches=4, seed=SEED, transport_electrons=False)
    one = engine.run_dij(_source(), beamlet_group_size=1, **kwargs)
    all_ = engine.run_dij(_source(), beamlet_group_size=3, **kwargs)
    np.testing.assert_array_equal(one.indptr, all_.indptr)
    np.testing.assert_array_equal(one.indices, all_.indices)
    np.testing.assert_array_equal(one.dose, all_.dose)
    np.testing.assert_array_equal(one.sigma, all_.sigma)
    assert one.energy_emitted == all_.energy_emitted


def test_exact_type_never_calls_the_host_samplers(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the exact type must be generated in-kernel")

    monkeypatch.setattr(SegmentedFanSource, "sample_beamlet_batch", refuse)
    monkeypatch.setattr(SegmentedFanSource, "emit", refuse)
    monkeypatch.setattr(SpectralBeamSource, "sample_batch", refuse)
    dij = _warp(_grid(), "cpu").run_dij(
        _source(), n_histories_per_beamlet=400, n_batches=2, seed=SEED, transport_electrons=False
    )
    assert dij.energy_emitted > 0.0


class _HalfWeightSegments(SegmentedFanSource):
    """Every primary at weight 0.5: honoured only by the host pre-sampling route."""

    def sample_beamlet_batch(self, seed: int, history_offset: int, n: int, beamlet: int):
        columns = super().sample_beamlet_batch(seed, history_offset, n, beamlet)
        columns["weight"] = np.full(n, 0.5, dtype=np.float32)
        return columns


class _HalfWeightFan(SpectralBeamSource):
    """A fan subclass whose primaries carry weight 0.5 before collimation."""

    def sample_batch(self, seed: int, history_offset: int, n: int):
        columns = super().sample_batch(seed, history_offset, n)
        columns["weight"] = np.full(n, 0.5, dtype=np.float32)
        return columns


def test_a_subclass_keeps_the_host_route() -> None:
    source = _HalfWeightSegments.from_stacks(_fan(), _stacks(), AnalyticCrossSections())
    n = 600
    dij = _warp(_grid(), "cpu").run_dij(
        source, n_histories_per_beamlet=n, n_batches=2, seed=SEED, transport_electrons=False
    )
    # Correlated: every segment's batch b replays the fan's histories [b*300, b*300+300).
    energies = _fan().sample_batch(SEED, 0, n)["energy"].astype(np.float64)
    assert dij.energy_emitted == pytest.approx(0.5 * 3 * energies.sum(), rel=1e-6)


def test_a_fan_subclass_keeps_the_host_route() -> None:
    fan = _HalfWeightFan(
        SPECTRUM, focal_point=FOCAL, center=(8.0, 8.0, 0.0), width_u=10.0, width_v=8.0
    )
    source = SegmentedFanSource.from_stacks(fan, _stacks(), AnalyticCrossSections())
    n = 600
    host = sum(
        float(
            np.sum(
                source.sample_beamlet_batch(SEED, b * 300, 300, s)["weight"].astype(np.float64)
                * fan.sample_batch(SEED, b * 300, 300)["energy"].astype(np.float64)
            )
        )
        for s in range(3)
        for b in range(2)
    )
    dij = _warp(_grid(), "cpu").run_dij(
        source, n_histories_per_beamlet=n, n_batches=2, seed=SEED, transport_electrons=False
    )
    assert dij.energy_emitted == pytest.approx(host, rel=1e-6)


@pytest.mark.gpu
def test_concurrent_batches_are_bit_inert() -> None:
    _require("cuda:0")
    engine = _warp(_grid(), "cuda:0")
    kwargs = dict(n_histories_per_beamlet=8_000, n_batches=4, seed=SEED)
    sequential = engine.run_dij(_source(), **kwargs)
    lanes = engine.run_dij(_source(), concurrent_batches=2, **kwargs)
    np.testing.assert_array_equal(sequential.indices, lanes.indices)
    np.testing.assert_array_equal(sequential.dose, lanes.dose)
    np.testing.assert_array_equal(sequential.sigma, lanes.sigma)
    assert sequential.energy_emitted == lanes.energy_emitted
