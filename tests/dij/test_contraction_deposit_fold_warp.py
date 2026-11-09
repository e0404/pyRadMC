"""The deposit fold: a beamlet contraction scored at the deposit, not over a dense map.

``WarpEngine.run_beamlet_contraction(fold="deposit")`` (the default) adds each
energy deposit, weighted by its scoring voxel's ``a_v / m_v``, straight into its
beamlet's int64 accumulator, so its cost follows the histories rather than
``beamlets x scoring voxels``. ``fold="dense"`` is the verification instrument: the
same histories folded through the dense per-batch quanta map, as before.

The two folds see the same transport (same routes, seeds, history mapping) and the
same per-deposit energy quanta; they differ only in where the weighting happens.
The deposit fold rounds each weighted deposit to its call's weighted quantum
``1 / K`` (``K`` a power of two, :func:`pyradmc.backends.warp.engine._fold_scale`),
an error of at most half a weighted quantum per deposit, which on a per-history
value is ``0.5 * ENERGY_QUANTUM_MEV / K`` per deposit. Bounding the deposits by
``DEPOSITS_PER_HISTORY`` per history gives the agreement tolerance used here
(:func:`_resolution`); float64 ordering of the dense contraction adds the
``1e-12`` relative term. The energy books are the same integers on both paths and
are compared exactly.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource
from pyradmc.scoring.grid import ScoringGrid
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyradmc.backends.warp import engine as warp_engine  # noqa: E402  (needs warp)
from pyradmc.backends.warp.kernels import ENERGY_QUANTUM_MEV  # noqa: E402
from pyradmc.data.analytic import AnalyticCrossSections  # noqa: E402

pytestmark = pytest.mark.warp

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]

DEPOSITS_PER_HISTORY = 1000
"""Upper bound on the scored deposits of one history in these phantoms. A 6 MeV
photon's electrons take some tens of substeps each at the shipped step fraction;
this is an order of magnitude above that, so the bound is safe, not tuned."""

FLOAT_RTOL = 1e-12
"""float64 rounding of the dense path's per-voxel products and chunked sums."""


def _engine(device: str, grid: VoxelGrid | None = None, xs=None):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    grid = grid or VoxelGrid.uniform_water(shape=(10, 10, 8), spacing=(1.0, 1.0, 1.0))
    xs = xs or AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return warp_engine.WarpEngine(grid=grid, cross_sections=xs, device=device)


def _source() -> BeamletGridSource:
    return BeamletGridSource(2.0, -1.0, (2.0, 8.0), (2.0, 8.0), 3, 3)


def _weights(shape: tuple[int, int, int]) -> np.ndarray:
    """Signed, so the contraction cancels in part, as an objective gradient does."""
    x, y, z = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return np.sin(0.7 * x) + 0.3 * y - 0.1 * z


def _scale(a: np.ndarray, scoring: ScoringGrid, per_batch: int, max_energy: float) -> float:
    mass = scoring.voxel_mass
    covered = mass > 0.0
    ratio = float(np.max(np.abs(a[covered]) / mass[covered])) if np.any(covered) else 0.0
    return warp_engine._fold_scale(ratio, per_batch, max_energy)


def _resolution(scale: float) -> float:
    """Largest deposit-fold error of one per-history value: half a quantum per deposit."""
    return 0.5 * DEPOSITS_PER_HISTORY * ENERGY_QUANTUM_MEV / scale


def _both(engine, source, a, **kwargs):
    deposit, dense = (
        engine.run_beamlet_contraction(source, a, fold=fold, **kwargs)
        for fold in ("deposit", "dense")
    )
    return deposit, dense


def _assert_agree(deposit, dense, scale: float) -> None:
    tol = _resolution(scale) + FLOAT_RTOL * np.max(np.abs(dense.values))
    np.testing.assert_allclose(deposit.values, dense.values, rtol=0.0, atol=tol)
    # sigma is the spread of batch values each carrying the same error bound; relative
    # to a batch spread of a few percent that is far below this.
    np.testing.assert_allclose(deposit.sigma, dense.sigma, rtol=1e-6, atol=tol)
    assert deposit.energy_emitted == dense.energy_emitted
    assert deposit.energy_deposited == dense.energy_deposited
    assert deposit.energy_escaped == dense.energy_escaped
    assert deposit.energy_unscored == dense.energy_unscored


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("electrons", [True, False], ids=["electrons", "kerma"])
def test_deposit_fold_agrees_with_the_dense_fold(device: str, electrons: bool) -> None:
    engine = _engine(device)
    a = _weights(engine.grid.shape)
    kwargs = dict(
        n_histories_per_beamlet=800, n_batches=8, seed=SEED, transport_electrons=electrons
    )
    deposit, dense = _both(engine, _source(), a, **kwargs)
    scoring = ScoringGrid.for_grid(engine.grid)
    _assert_agree(deposit, dense, _scale(a, scoring, 100, _source().max_energy))
    assert np.all(deposit.sigma > 0.0)
    assert not np.array_equal(deposit.values, np.zeros_like(deposit.values))


@pytest.mark.parametrize("device", DEVICES)
def test_deposit_fold_weights_dose_to_water_like_the_dense_fold(device: str) -> None:
    """A non-water medium: the tally quanta carry the SPR factor on both paths."""
    from tests.integration.test_warp_dose_to_water import _medium_engine

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    engine = _medium_engine(device)
    a = _weights(engine.grid.shape)
    source = BeamletGridSource(6.0, -1.0, (2.0, 14.0), (2.0, 14.0), 2, 2)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED, scoring_mode="dose_to_water")
    deposit, dense = _both(engine, source, a, **kwargs)
    _assert_agree(deposit, dense, _scale(a, ScoringGrid.for_grid(engine.grid), 100, 6.0))


@pytest.mark.parametrize("device", DEVICES)
def test_deposit_fold_routes_into_the_scoring_grid_like_the_dense_fold(device: str) -> None:
    """A coarser, shifted scoring grid that leaves part of the phantom unscored."""
    engine = _engine(device)
    scoring = ScoringGrid.rebin(engine.grid, (4, 3, 3), (2.0, 2.5, 2.0), (1.5, 1.0, 1.0))
    a = _weights(scoring.shape) + 0.5
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED, scoring_grid=scoring)
    deposit, dense = _both(engine, _source(), a, **kwargs)
    assert dense.energy_unscored > 0.0  # the grid really leaves deposits out
    _assert_agree(deposit, dense, _scale(a, scoring, 100, _source().max_energy))


@pytest.mark.parametrize("device", DEVICES)
def test_signed_edge_weights_fold_like_the_dense_fold(device: str) -> None:
    """Leaf-edge columns: negative-weight primaries, the study's gradient source."""
    from pyradmc.geometry.segment_edges import SegmentedEdgeSource
    from tests.unit.test_segmented_edge_source import XS, _segments

    grid = VoxelGrid.uniform_water(
        shape=(16, 3, 8), spacing=(0.5, 2.0, 1.0), origin=(3.0, -5.0, 0.0)
    )
    engine = _engine(device, grid, XS)
    source = SegmentedEdgeSource(_segments())
    a = _weights(grid.shape)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=2, seed=SEED)
    deposit, dense = _both(engine, source, a, **kwargs)
    assert np.any(dense.values < 0.0) and np.any(dense.values > 0.0)
    # The weights scale each deposit's quanta, not the number of deposits, so the
    # half-quantum-per-deposit bound holds unchanged for signed, unbounded weights.
    _assert_agree(deposit, dense, _scale(a, ScoringGrid.for_grid(grid), 200, source.max_energy))


@pytest.mark.parametrize("device", DEVICES)
def test_deposit_fold_is_bitwise_inert_to_scheduling(device: str, monkeypatch) -> None:
    """Integer accumulation: group size, sub-accumulators and lanes change no bit."""
    engine = _engine(device)
    a = _weights(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED)
    base = engine.run_beamlet_contraction(_source(), a, **kwargs)
    variants = [
        engine.run_beamlet_contraction(_source(), a, beamlet_group_size=g, **kwargs) for g in (1, 4)
    ]
    if device.startswith("cuda"):
        variants.append(
            engine.run_beamlet_contraction(_source(), a, concurrent_batches=3, **kwargs)
        )
    monkeypatch.setattr(warp_engine, "_FOLD_SUBSLOT_TARGET", 1)  # one sub-accumulator
    variants.append(engine.run_beamlet_contraction(_source(), a, **kwargs))
    for other in variants:
        np.testing.assert_array_equal(other.values, base.values)
        np.testing.assert_array_equal(other.sigma, base.sigma)
        assert other.energy_deposited == base.energy_deposited


@pytest.mark.parametrize("device", DEVICES)
def test_deposit_fold_rescales_when_the_int64_headroom_would_be_exceeded(
    device: str, monkeypatch, caplog
) -> None:
    """A scale far past the int64 budget is caught by the guard and the call rerun.

    ``_FOLD_SCALE_BITS = 66`` puts the a-priori budget above ``2**63``; the per-batch
    absolute tally then proves the accumulators may have wrapped, and the call is
    repeated (same seeds, same transport) at the largest safe power-of-two scale,
    which is finer than the default one here.
    """
    engine = _engine(device)
    a = _weights(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED)
    dense = engine.run_beamlet_contraction(_source(), a, fold="dense", **kwargs)
    default_scale = _scale(a, ScoringGrid.for_grid(engine.grid), 100, _source().max_energy)
    monkeypatch.setattr(warp_engine, "_FOLD_SCALE_BITS", 66)
    with caplog.at_level(logging.INFO, logger=warp_engine.__name__):
        deposit = engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert any("rescal" in record.getMessage() for record in caplog.records)
    _assert_agree(deposit, dense, default_scale)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_weights_contract_to_exactly_zero(device: str) -> None:
    engine = _engine(device)
    a = np.zeros(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    deposit, dense = _both(engine, _source(), a, **kwargs)
    np.testing.assert_array_equal(deposit.values, 0.0)
    np.testing.assert_array_equal(deposit.sigma, 0.0)
    assert deposit.energy_deposited == dense.energy_deposited > 0.0


def test_an_unknown_fold_is_refused() -> None:
    engine = _engine("cpu")
    with pytest.raises(ValueError, match="fold"):
        engine.run_beamlet_contraction(
            _source(),
            np.ones(engine.grid.shape),
            n_histories_per_beamlet=8,
            n_batches=2,
            seed=SEED,
            fold="sparse",
        )


class TestFoldScale:
    """The weighted quantum: a power of two putting one batch's budget in its octave."""

    @pytest.mark.parametrize("ratio", [1e-6, 0.37, 1.0, 3.0, 12345.0])
    @pytest.mark.parametrize(("per_batch", "energy"), [(1, 0.5), (500, 6.0), (10**6, 20.0)])
    def test_budget_sits_in_its_octave(self, ratio: float, per_batch: int, energy: float) -> None:
        scale = warp_engine._fold_scale(ratio, per_batch, energy)
        mantissa, _ = np.frexp(scale)
        assert mantissa == 0.5  # an exact power of two: a_v / m_v * K is exact
        budget = ratio * per_batch * energy / ENERGY_QUANTUM_MEV * scale
        bits = warp_engine._FOLD_SCALE_BITS
        assert 2.0 ** (bits - 1) <= budget < 2.0**bits

    def test_zero_weights_take_unit_scale(self) -> None:
        assert warp_engine._fold_scale(0.0, 500, 6.0) == 1.0


class TestFoldGroupSize:
    """No dense map: the group is bounded by the sub-accumulators, not the voxels."""

    def test_unknown_memory_takes_every_beamlet(self) -> None:
        assert warp_engine._auto_fold_group_size(0, 7020, 1) == 7020

    def test_ample_memory_is_not_capped_at_the_dense_cap(self) -> None:
        group = warp_engine._auto_fold_group_size(4 * 2**30, 7020, 1)
        assert group == 7020 > warp_engine._GROUP_SIZE_CAP

    def test_budget_arithmetic(self) -> None:
        """The narrow-group sub-accumulators are charged up front, then per beamlet."""
        per_beamlet = warp_engine._fold_beamlet_bytes(2)
        extra = 2 * warp_engine._FOLD_SUBSLOT_TARGET * warp_engine._FOLD_SLOT_STRIDE * 8
        free = 2 * (extra + 1000 * per_beamlet)  # half is budgeted
        assert warp_engine._auto_fold_group_size(free, 7020, 2) == 1000

    def test_the_accumulators_fit_what_was_budgeted(self) -> None:
        """Whatever group comes out, its lanes' sub-accumulator blocks fit the budget."""
        stride = warp_engine._FOLD_SLOT_STRIDE
        for free in (10**5, 10**6, 10**7, 10**9):
            for lanes in (1, 3):
                group = warp_engine._auto_fold_group_size(free, 10**6, lanes)
                n_sub = warp_engine._fold_subslots(group)
                used = lanes * group * n_sub * stride * 8 + 16 * group
                assert group == 1 or used <= free * warp_engine._GROUP_MEMORY_FRACTION

    def test_tiny_memory_floors_at_one(self) -> None:
        assert warp_engine._auto_fold_group_size(1, 7020, 1) == 1


class TestFoldSubslots:
    """Few beamlets spread over many sub-accumulators, many beamlets over one each."""

    def test_counts(self) -> None:
        target = warp_engine._FOLD_SUBSLOT_TARGET
        cap = warp_engine._FOLD_SUBSLOTS_MAX
        assert warp_engine._fold_subslots(1) == min(cap, target)
        assert warp_engine._fold_subslots(target) == 1
        assert warp_engine._fold_subslots(7020) == 1
        for group in (1, 3, 100, 1000, 5000):
            n_sub = warp_engine._fold_subslots(group)
            assert n_sub & (n_sub - 1) == 0  # a power of two: the kernels mask with it
            assert n_sub == 1 or n_sub * group <= target
