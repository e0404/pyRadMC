"""The deposit fold: a beamlet contraction scored at the deposit, not over a dense map.

``WarpEngine.run_beamlet_contraction(fold="deposit")`` (the default) adds each
energy deposit, weighted by its scoring voxel's ``a_v / m_v``, straight into its
beamlet's int64 accumulator, so its cost follows the histories rather than
``beamlets x scoring voxels``. ``fold="dense"`` is the verification instrument: the
same histories folded through the dense per-batch quanta map, as before.

The two folds see the same transport (same routes, seeds, history mapping) and the
same per-deposit energy quanta; they differ only in where the weighting happens.
The deposit fold rounds each weighted deposit to its octave band's weighted quantum
``1 / K_b`` (powers of two, :func:`pyradmc.backends.warp.engine._fold_bands`), an
error of at most half a weighted quantum per deposit, which on a per-history value
is at most ``0.5 * ENERGY_QUANTUM_MEV / K_0`` per deposit (band 0's quantum is the
coarsest). Bounding the deposits by ``DEPOSITS_PER_HISTORY`` per history gives the
agreement tolerance used here (:func:`_resolution`); float64 ordering of the dense
contraction adds the ``1e-12`` relative term. Where band 0 is set by a voxel the
histories do not reach, the per-voxel guarantee is the tighter bound
(:func:`_guaranteed`). The energy books are the same integers on both paths and are
compared exactly.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
from fractions import Fraction

import numpy as np
import pytest

from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource
from pyradmc.scoring.grid import ScoringGrid
from tests.conftest import SEED
from tests.contraction_fixtures import engine as _engine
from tests.contraction_fixtures import source as _source
from tests.contraction_fixtures import weights as _weights

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


def _scale(a: np.ndarray, scoring: ScoringGrid, per_batch: int, max_energy: float) -> float:
    """The engine's ``K``: the budget rule at the floor of the exact octave of max|a/m|."""
    e = _exact_octave(a, scoring.voxel_mass)
    return warp_engine._fold_scale(2.0 ** (e - 1), per_batch, max_energy)


def _resolution(scale: float) -> float:
    """Largest deposit-fold error of one per-history value: half a quantum per deposit."""
    return 0.5 * DEPOSITS_PER_HISTORY * ENERGY_QUANTUM_MEV / scale


def _both(engine, source, a, **kwargs):
    deposit, dense = (
        engine.run_beamlet_contraction(source, a, fold=fold, **kwargs)
        for fold in ("deposit", "dense")
    )
    return deposit, dense


def _guaranteed(a: np.ndarray, mass: np.ndarray, reached: np.ndarray) -> float:
    """Largest per-history error under the per-voxel resolution guarantee.

    Half a weighted quantum per deposit is at most ``2**(R-1)`` energy quanta times
    the ``|a/m|`` of the deposit's voxel; ``reached`` are the voxels deposits can land
    in.
    """
    sel = reached & (mass > 0.0) & (a != 0.0)
    bits = warp_engine._FOLD_RESOLUTION_BITS
    ratio = float(np.max(np.abs(a[sel]) / mass[sel]))
    return DEPOSITS_PER_HISTORY * 2.0 ** (bits - 1) * ENERGY_QUANTUM_MEV * ratio


def _assert_agree(deposit, dense, scale: float) -> None:
    _assert_within(deposit, dense, _resolution(scale))


def _assert_within(deposit, dense, bound: float) -> None:
    tol = bound + FLOAT_RTOL * np.max(np.abs(dense.values))
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

    ``_FOLD_SCALE_BITS = 74`` makes the accumulators really wrap: with the guard off
    the result is wrong, which proves the test has power. With it on, the per-batch
    absolute tally proves the accumulators may have wrapped, and the call is repeated
    (same seeds, same transport) at the largest safe power-of-two scale, which is
    finer than the default one here.
    """
    engine = _engine(device)
    a = _weights(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED)
    dense = engine.run_beamlet_contraction(_source(), a, fold="dense", **kwargs)
    default_scale = _scale(a, ScoringGrid.for_grid(engine.grid), 100, _source().max_energy)
    monkeypatch.setattr(warp_engine, "_FOLD_SCALE_BITS", 74)
    with monkeypatch.context() as unguarded:
        unguarded.setattr(warp_engine, "_FOLD_GUARD_BITS", 1000)  # never fires
        wrapped = engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert np.max(np.abs(wrapped.values - dense.values)) > 0.1 * np.max(np.abs(dense.values))
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("magnitude", [1e-290, 1e-300, 1e-302, 1e-305, 1e-310])
def test_tiny_weights_fold_like_the_dense_fold(device: str, magnitude: float) -> None:
    """Tiny finite weights: no exception and no silent zero, the dense fold's values.

    Tiny weights need ``K`` near or past the float64 range: ``2**1031`` at 1e-305
    (``math.ldexp`` raised), and near 1e-302 ``K * per_batch`` overflowed, the batch
    scale became zero and every value with it; at 1e-300 the batch scale was a
    subnormal with a few significant bits. Where ``K``, the band scales and the
    batch scales cannot all stay normal floats, the call runs the dense fold.
    """
    engine = _engine(device)
    a = np.full(engine.grid.shape, magnitude)
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED)
    deposit, dense = _both(engine, _source(), a, **kwargs)
    assert np.all(dense.values != 0.0)
    np.testing.assert_allclose(deposit.values, dense.values, rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(deposit.sigma, dense.sigma, rtol=1e-6, atol=0.0)


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


def _exact_octave(a: np.ndarray, mass: np.ndarray, extreme=max) -> int:
    """``e`` with ``max |a_v| / m_v`` in ``[2**(e-1), 2**e)``, in exact rational arithmetic.

    Over the voxels with mass and a nonzero weight; ``extreme=min`` for the minimum.
    """
    covered = (mass > 0.0) & (a != 0.0)
    pairs = zip(a[covered], mass[covered], strict=True)
    ratio = extreme(Fraction(abs(float(x))) / Fraction(float(m)) for x, m in pairs)
    e = 0
    while ratio >= Fraction(2) ** e:
        e += 1
    while ratio < Fraction(2) ** (e - 1):
        e -= 1
    return e


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("power_of_two", [False, True], ids=["general", "power-of-two-max"])
def test_fold_scale_follows_the_exact_ratio_octave(device: str, power_of_two: bool) -> None:
    """``K`` comes from the exact octave of ``max|a/m|``, not from a fast_math quotient.

    The fast_math module divides approximately, so a quotient one ulp short of a power
    of two would put ``K`` an octave off on one target and not on another. ``K`` is the
    budget rule (:func:`_fold_scale`) applied to the power of two at the bottom of the
    exact octave, which no rounding of a quotient can move.
    """
    engine = _engine(device)
    scoring = ScoringGrid.for_grid(engine.grid)
    a = _weights(engine.grid.shape)
    if power_of_two:
        a = a / (2.0 * np.max(np.abs(a)))
        a[3, 4, 5] = -4.0 * scoring.voxel_mass[3, 4, 5]  # max |a/m| is exactly 4
    e = _exact_octave(a, scoring.voxel_mass)
    spec = engine._fold_spec(a, scoring, 100, _source())
    assert spec.scale == warp_engine._fold_scale(2.0 ** (e - 1), 100, _source().max_energy)
    assert spec.ratio_bound == 2.0**e


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("factor", [1.0 - 2.0**-52, 1.0 + 2.0**-52, 0.125, 8.0, 0.0, math.inf])
def test_fold_scale_does_not_depend_on_the_quotient_seed(
    device: str, factor: float, monkeypatch
) -> None:
    """Any target's approximate quotient (here: perturbed, even absurdly) gives the same ``K``.

    The seed only says where the exact search starts; host weights and device weights
    on any device therefore get the same ``K`` and bit-identical results.
    """
    engine = _engine(device)
    scoring = ScoringGrid.for_grid(engine.grid)
    a = _weights(engine.grid.shape)
    a = a / (2.0 * np.max(np.abs(a)))
    a[3, 4, 5] = 4.0 * scoring.voxel_mass[3, 4, 5]  # on an octave boundary
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    expected_spec = engine._fold_spec(a, scoring, 100, _source())
    expected = engine.run_beamlet_contraction(_source(), a, **kwargs)
    seed = warp_engine._fold_ratio_seed

    def perturbed(*args, **kw):
        s = seed(*args, **kw)
        return dataclasses.replace(s, ratio_max=s.ratio_max * factor)

    monkeypatch.setattr(warp_engine, "_fold_ratio_seed", perturbed)
    spec = engine._fold_spec(a, scoring, 100, _source())
    assert spec.scale == expected_spec.scale
    assert spec.ratio_bound == expected_spec.ratio_bound == 8.0
    result = engine.run_beamlet_contraction(_source(), a, **kwargs)
    np.testing.assert_array_equal(result.values, expected.values)
    np.testing.assert_array_equal(result.sigma, expected.sigma)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("factor", [1.0, 1.0 - 2.0**-52, 1.0 + 2.0**-52, 0.125, 8.0, 0.0])
def test_smallest_ratio_octave_is_exact_and_seed_independent(
    device: str, factor: float, monkeypatch
) -> None:
    """The resolution check reads the exact octave of ``min |a/m|`` over the weighted voxels.

    Zero weights are not in it (they score nothing whatever their mass); the minimum
    here sits exactly on a power of two, where a one-ulp quotient would misplace it.
    """
    engine = _engine(device)
    scoring = ScoringGrid.for_grid(engine.grid)
    a = _weights(engine.grid.shape) + 5.0  # |a| in [~2, ~9]
    a[0, 0, :] = 0.0
    a[2, 3, 4] = 0.25 * scoring.voxel_mass[2, 3, 4]  # min |a/m| is exactly 1/4
    seed = warp_engine._fold_ratio_seed

    def perturbed(*args, **kw):
        s = seed(*args, **kw)
        return dataclasses.replace(s, ratio_min=s.ratio_min * factor)

    monkeypatch.setattr(warp_engine, "_fold_ratio_seed", perturbed)
    spec = engine._fold_spec(a, scoring, 100, _source())
    assert spec.min_exponent == _exact_octave(a, scoring.voxel_mass, min) == -1
    assert spec.ratio_exponent == _exact_octave(a, scoring.voxel_mass)


SLIVER_ORIGIN_X = -15.983
"""A transport-grid x origin (0.9 cm voxels) whose scoring grid, padded by one voxel,
keeps an outer layer of float-overlap mass ~1.8e-15 g: the rebin's overlap of the
padding voxel with the grid is a rounding residue, not zero (found by a search over
origins; the test asserts the sliver is there)."""


def _sliver_setup(device: str):
    n, s = 12, 0.9
    grid = VoxelGrid.uniform_water(
        shape=(n, 10, 8), spacing=(s, 1.0, 1.0), origin=(SLIVER_ORIGIN_X, 0.0, 0.0)
    )
    engine = _engine(device, grid)
    padded = ScoringGrid.rebin(grid, (n + 2, 10, 8), (s, 1.0, 1.0), (SLIVER_ORIGIN_X - s, 0.0, 0.0))
    mass = padded.voxel_mass
    assert 0.0 < mass[0].max() < 1e-12 * mass.max()  # the sliver this is about
    x0 = SLIVER_ORIGIN_X + 2.0
    source = BeamletGridSource(2.0, -1.0, (x0, x0 + 6.0), (2.0, 8.0), 3, 3)
    return engine, padded, source


def _fallbacks(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "dense fold for this call" in r.getMessage()]


@pytest.mark.parametrize("device", DEVICES)
def test_a_sliver_voxel_does_not_zero_the_deposit_fold(device: str, caplog) -> None:
    """One near-massless weighted voxel must not collapse ``K`` for every other voxel.

    The sliver's ``a/m`` is ~5e14 times the phantom's; a single ``K`` sized for it
    weighted every real deposit to well below half a quantum, which rounded to an
    exact zero for all nine beamlets. The phantom now sits in a lower octave band
    with its own scale, so every weighted voxel keeps the per-voxel resolution, and
    the call stays on the deposit fold.
    """
    engine, padded, source = _sliver_setup(device)
    a = np.ones(padded.shape)
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED, scoring_grid=padded)
    with caplog.at_level(logging.WARNING, logger=warp_engine.__name__):
        deposit, dense = _both(engine, source, a, **kwargs)
    assert not _fallbacks(caplog)
    assert engine._fold_spec(a, padded, 100, source).n_bands >= 2
    # No deposit reached the sliver (one would put ~1e9 into a value): the phantom's
    # voxels are where the bound applies.
    assert np.all((dense.values > 0.1) & (dense.values < 1.0))
    mass = padded.voxel_mass
    _assert_within(deposit, dense, _guaranteed(a, mass, mass > 1e-6 * mass.max()))


@pytest.mark.parametrize("device", DEVICES)
def test_zero_weights_on_a_sliver_keep_one_band(device: str, caplog) -> None:
    """A zero weight scores nothing at any mass, so it does not count against the range."""
    engine, padded, source = _sliver_setup(device)
    a = np.where(padded.voxel_mass > 1e-6 * padded.voxel_mass.max(), 1.0, 0.0)
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED, scoring_grid=padded)
    with caplog.at_level(logging.WARNING, logger=warp_engine.__name__):
        deposit, dense = _both(engine, source, a, **kwargs)
    assert not _fallbacks(caplog)
    assert engine._fold_spec(a, padded, 100, source).n_bands == 1
    assert np.all(dense.values > 0.1)
    _assert_agree(deposit, dense, _scale(a, padded, 100, source.max_energy))


@pytest.mark.parametrize("device", DEVICES)
def test_signed_weights_through_zero_fold_in_bands(device: str) -> None:
    """The tests' own signed weights cancel to 5.6e-17 in one voxel: a 2**56 range.

    One ``K`` resolves 29 octaves at this budget, so that takes two bands, and the
    result agrees with the dense fold to band 0's resolution (the coarsest), as
    every test here asserts.
    """
    engine = _engine(device)
    scoring = ScoringGrid.for_grid(engine.grid)
    a = _weights(engine.grid.shape)
    assert np.min(np.abs(a[a != 0.0])) < 1e-15
    spec = engine._fold_spec(a, scoring, 100, _source())
    assert (spec.band_width, spec.n_bands) == (29, 2)
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED)
    deposit, dense = _both(engine, _source(), a, **kwargs)
    _assert_agree(deposit, dense, _scale(a, scoring, 100, _source().max_energy))


@pytest.mark.parametrize("device", DEVICES)
def test_a_range_beyond_the_bands_runs_the_dense_fold(device: str, caplog) -> None:
    """Past ``_FOLD_BANDS_MAX`` bands the call is the dense fold itself, and says so."""
    engine = _engine(device)
    a = np.ones(engine.grid.shape)
    a[5, 5, 2] = 2.0**-400  # in the field, far below every band one call can hold
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    dense = engine.run_beamlet_contraction(_source(), a, fold="dense", **kwargs)
    with caplog.at_level(logging.WARNING, logger=warp_engine.__name__):
        deposit = engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert len(_fallbacks(caplog)) == 1
    np.testing.assert_array_equal(deposit.values, dense.values)
    np.testing.assert_array_equal(deposit.sigma, dense.sigma)


@pytest.mark.parametrize("device", DEVICES)
def test_a_rescale_rebands_the_fold(device: str, monkeypatch, caplog) -> None:
    """The overflow guard's smaller ``K_0`` narrows the bands; the rerun takes more of them.

    ``_FOLD_SCALE_BITS = 74`` forces the guard (as in the rescale test above); the
    weights span 2**40 in ``a/m``, one band at the a-priori ``K_0`` and more at the
    rescaled one. The rerun keeps the per-voxel resolution, so it agrees with the
    dense fold to the guarantee (``K_0`` itself is coarser than the default here).
    """
    engine = _engine(device)
    a = np.ones(engine.grid.shape)
    a[5, 5, 2] = 2.0**-40  # in the field: it does receive deposits
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED)
    dense = engine.run_beamlet_contraction(_source(), a, fold="dense", **kwargs)
    monkeypatch.setattr(warp_engine, "_FOLD_SCALE_BITS", 74)
    assert engine._fold_spec(a, ScoringGrid.for_grid(engine.grid), 100, _source()).n_bands == 1
    with caplog.at_level(logging.INFO, logger=warp_engine.__name__):
        deposit = engine.run_beamlet_contraction(_source(), a, **kwargs)
    rescales = [r.getMessage() for r in caplog.records if "rescaling" in r.getMessage()]
    assert len(rescales) == 1 and "(1 octave bands)" not in rescales[0]
    assert not _fallbacks(caplog)
    mass = ScoringGrid.for_grid(engine.grid).voxel_mass
    _assert_within(deposit, dense, _guaranteed(a, mass, mass > 0.0))


@pytest.mark.parametrize("device", DEVICES)
def test_band_assignment_is_exact_on_the_floors(device: str) -> None:
    """Each voxel's band and coefficient, against exact host arithmetic.

    Ratios placed exactly on band floors (where a one-ulp quotient would pick the
    wrong band) and just below them; the band is decided by exact comparisons.
    """
    engine = _engine(device)
    scoring = ScoringGrid.for_grid(engine.grid)
    mass = scoring.voxel_mass
    a = np.ones(engine.grid.shape)
    a[5, 5, 2] = 2.0**-100  # spans several bands
    spec = engine._fold_spec(a, scoring, 100, _source())
    width, n, e = spec.band_width, spec.n_bands, spec.ratio_exponent
    assert n >= 3
    floors = [2.0 ** (e - (b + 1) * width) for b in range(n - 1)]
    flat = a.reshape(-1).copy()
    m = mass.reshape(-1)
    for i, floor in enumerate(floors):
        flat[10 + 2 * i] = floor * m[10 + 2 * i]  # exactly on the floor: band i
        flat[11 + 2 * i] = -np.nextafter(floor, 0.0) * m[11 + 2 * i]  # just below: band i + 1
    a = flat.reshape(a.shape)
    spec = engine._fold_spec(a, scoring, 100, _source())
    assert (spec.band_width, spec.n_bands) == (width, n)
    weights = wp.array(flat, dtype=wp.float64, device=device)
    _, voxel_mass = engine._upload_scoring(scoring, device)
    coef, band = warp_engine._fold_coefficients(spec, weights, voxel_mass, device)
    expected_band = np.array(
        [
            sum(Fraction(abs(float(x))) < Fraction(t) * Fraction(float(mv)) for t in floors)
            for x, mv in zip(flat, m, strict=True)
        ]
    )
    np.testing.assert_array_equal(band.numpy(), expected_band)
    for i in range(len(floors)):
        assert band.numpy()[10 + 2 * i] == i and band.numpy()[11 + 2 * i] == i + 1
    k_b = spec.scale * 2.0 ** (width * expected_band.astype(float))
    np.testing.assert_allclose(coef.numpy(), flat / m * k_b, rtol=4e-16, atol=0.0)
    # every weighted voxel keeps the resolution, and no band top exceeds the budget
    ratio = np.abs(flat) / m
    assert np.all(k_b * ratio >= 2.0**-warp_engine._FOLD_RESOLUTION_BITS)
    assert np.all(k_b * ratio < spec.scale * spec.ratio_bound)


class TestFoldBands:
    """One weighted quantum is worth at most ``2**R`` energy quanta in every weighted voxel."""

    @pytest.mark.parametrize("k", [-40, 0, 17, 300])
    @pytest.mark.parametrize("e", [-3, 0, 5])
    def test_every_band_keeps_the_resolution_and_the_budget(self, k: int, e: int) -> None:
        r = warp_engine._FOLD_RESOLUTION_BITS
        for f in range(e, e - 200, -7):
            bands = warp_engine._fold_bands(k, e, f)
            if bands is None:
                assert k + e + r < 1 or -(-(e - f + 1) // (k + e + r)) > 6
                continue
            width, n = bands
            for b in range(n):
                k_b = k + b * width
                assert k_b + e - b * width == k + e  # each band's top: the same budget
                assert k_b + e - (b + 1) * width >= -r  # each band's floor: resolved
            assert e - n * width <= f - 1  # the last band reaches the smallest ratio
            assert n == 1 or e - (n - 1) * width > f - 1  # and no band is spare

    def test_zero_weights_take_one_band(self) -> None:
        assert warp_engine._fold_bands(0, None, None) == (1, 1)

    def test_a_single_octave_is_one_band_at_the_budget_rule(self) -> None:
        for per_batch, energy in ((1, 0.5), (500, 6.0), (10**6, 20.0)):
            for e in (-20, 0, 3):
                k = warp_engine._fold_scale_exponent(e, per_batch, energy)
                assert warp_engine._fold_bands(k, e, e) == (k + e + 10, 1)

    def test_the_stride_grows_past_two_bands(self) -> None:
        assert warp_engine._fold_stride(1) == warp_engine._fold_stride(2) == 4
        assert warp_engine._fold_stride(3) == warp_engine._fold_stride(6) == 8

    def test_the_stride_holds_the_kernels_slot_layout(self) -> None:
        """The host's strides and the kernels' slot offsets describe one layout.

        A sub-accumulator keeps the tally quanta at offsets 1 and 2 and band ``b``'s
        weighted sum at ``kernels._fold_band_slot(b)`` (called here from Python): for
        every band count the fold may use, those offsets are distinct and inside the
        stride the host allocates and indexes with.
        """
        from pyradmc.backends.warp import kernels

        for n_bands in range(1, warp_engine._FOLD_BANDS_MAX + 1):
            offsets = [1, 2] + [kernels._fold_band_slot(b) for b in range(n_bands)]
            assert len(set(offsets)) == len(offsets)
            assert min(offsets) >= 0
            assert max(offsets) < warp_engine._fold_stride(n_bands)


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

    @pytest.mark.parametrize("ratio_exponent", [-30, -1, 0, 1, 2, 14])
    @pytest.mark.parametrize(("per_batch", "energy"), [(1, 0.5), (500, 6.0), (10**6, 20.0)])
    def test_exponent_form_is_the_budget_rule(
        self, ratio_exponent: int, per_batch: int, energy: float
    ) -> None:
        """Integer form of the rule at the octave's floor ``2**(e-1)``: no float to overflow."""
        k = warp_engine._fold_scale_exponent(ratio_exponent, per_batch, energy)
        floor = math.ldexp(1.0, ratio_exponent - 1)
        assert math.ldexp(1.0, k) == warp_engine._fold_scale(floor, per_batch, energy)

    def test_rescale_keeps_the_guard_bound_below_half_the_range(self) -> None:
        for ratio_exponent in (-5, 0, 3):
            for q in (1, 2**20 + 1, 2**40, 2**59):
                k = warp_engine._fold_rescale_exponent(ratio_exponent, q)
                bound = (math.ldexp(1.0, k + ratio_exponent) + 0.5) * q
                assert bound < 2.0 ** (warp_engine._FOLD_GUARD_BITS - 1) * (1 + 2**-40)
                # and it is the largest such power of two
                assert (math.ldexp(1.0, k + 1 + ratio_exponent) + 0.5) * q > 2.0 ** (
                    warp_engine._FOLD_GUARD_BITS - 1
                )


STRIDE = warp_engine._FOLD_SLOT_STRIDE
"""A one- or two-band fold's sub-accumulator stride, the narrow one."""


class TestFoldGroupSize:
    """No dense map: the group is bounded by the sub-accumulators, not the voxels."""

    def test_unknown_memory_takes_every_beamlet(self) -> None:
        assert warp_engine._auto_fold_group_size(0, 7020, 1, 0, STRIDE) == 7020

    def test_ample_memory_is_not_capped_at_the_dense_cap(self) -> None:
        group = warp_engine._auto_fold_group_size(4 * 2**30, 7020, 1, 0, STRIDE)
        assert group == 7020 > warp_engine._GROUP_SIZE_CAP

    def test_budget_arithmetic(self) -> None:
        """The narrow-group sub-accumulators are charged up front, then per beamlet."""
        per_beamlet = warp_engine._fold_beamlet_bytes(2, STRIDE)
        extra = 2 * warp_engine._FOLD_SUBSLOT_TARGET * STRIDE * 8
        free = 2 * (extra + 1000 * per_beamlet)  # half is budgeted
        assert warp_engine._auto_fold_group_size(free, 7020, 2, 0, STRIDE) == 1000

    @pytest.mark.parametrize(
        "stride", [warp_engine._FOLD_SLOT_STRIDE, warp_engine._FOLD_SLOT_STRIDE_WIDE]
    )
    def test_the_accumulators_fit_what_was_budgeted(self, stride: int) -> None:
        """Whatever group comes out, its lanes' sub-accumulator blocks fit the budget."""
        for free in (10**5, 10**6, 10**7, 10**9):
            for lanes in (1, 3):
                group = warp_engine._auto_fold_group_size(free, 10**6, lanes, 0, stride)
                n_sub = warp_engine._fold_subslots(group)
                used = lanes * group * n_sub * stride * 8 + 16 * group
                assert group == 1 or used <= free * warp_engine._GROUP_MEMORY_FRACTION

    def test_tiny_memory_floors_at_one(self) -> None:
        assert warp_engine._auto_fold_group_size(1, 7020, 1, 0, STRIDE) == 1


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


class TestFoldShardGroups:
    """Several groups per device on a multi-device call; one device keeps every beamlet."""

    def test_one_device_keeps_the_group(self) -> None:
        assert warp_engine._shard_group_size(7020, 7020, 1) == 7020

    def test_a_smaller_group_is_kept(self) -> None:
        assert warp_engine._shard_group_size(3, 7020, 2) == 3

    @pytest.mark.parametrize("n_devices", [2, 3, 4])
    def test_every_device_has_a_group_to_pull(self, n_devices: int) -> None:
        per_device = warp_engine._FOLD_GROUPS_PER_DEVICE
        for n in range(1, 400):
            group = warp_engine._shard_group_size(n, n, n_devices)
            groups = -(-n // group)
            assert group >= 1
            assert groups >= min(n, n_devices)
            if n >= per_device * n_devices:
                assert groups >= 2 * n_devices


class TestBookSafeGroups:
    """A pass's unit-weight energy, group x histories x E_max, stays inside the books' budget."""

    @pytest.mark.parametrize(
        ("n_per", "energy"), [(1000, 6.0), (10**6, 6.0), (10**6, 20.0), (10**8, 6.0)]
    )
    def test_largest_group_within_the_budget(self, n_per: int, energy: float) -> None:
        budget = 2.0**warp_engine._BOOK_BUDGET_BITS
        group = warp_engine._book_safe_group_size(10**7, n_per, energy)
        per_beamlet = n_per * energy / ENERGY_QUANTUM_MEV
        assert group * per_beamlet <= budget < (group + 1) * per_beamlet

    def test_the_study_case_stays_one_group(self) -> None:
        """TG-119, 7020 leaf edges of 1000 histories at 6 MeV: the launch pattern is kept."""
        assert warp_engine._book_safe_group_size(7020, 1000, 6.0) == 7020

    def test_a_call_at_the_int64_limit_is_split(self) -> None:
        """7020 edges of 1e6 histories absorbing ~1.3 MeV each book ~9.1e18 quanta."""
        group = warp_engine._book_safe_group_size(7020, 10**6, 6.0)
        assert 1 <= group < 7020
        assert group * 10**6 * 6.0 / ENERGY_QUANTUM_MEV <= 2.0**warp_engine._BOOK_BUDGET_BITS

    def test_one_beamlet_is_the_floor(self) -> None:
        assert warp_engine._book_safe_group_size(5, 2**31 - 1, 20.0) == 1
        assert 2.0**31 * 20.0 / ENERGY_QUANTUM_MEV > 2.0**warp_engine._BOOK_BUDGET_BITS


def _plan(engine, contraction, devices, n_beamlets: int, n_per: int = 8) -> int:
    return engine._plan_group_size(
        None,
        device_list=devices,
        n_beamlets=n_beamlets,
        n_histories_per_beamlet=n_per,
        n_batches=2,
        concurrent_batches=1,
        n_voxels=ScoringGrid.for_grid(engine.grid).n_voxels,
        reduction=warp_engine._reduction_for(contraction, truncation=0.0, device_blocks=False),
        max_energy=_source().max_energy,
    )


def test_the_planner_shards_the_fold_over_every_device() -> None:
    """The fold's one whole-call group is cut for several devices; the dense plan is not."""
    engine = _engine("cpu")
    a = _weights(engine.grid.shape)
    spec = engine._fold_spec(a, ScoringGrid.for_grid(engine.grid), 4, _source())
    assert _plan(engine, spec, ["cpu"], 400) == 400  # one device: one group, as before
    group = _plan(engine, spec, ["cpu", "cpu"], 400)
    assert -(-400 // group) >= 2 * warp_engine._FOLD_GROUPS_PER_DEVICE
    dense = _plan(engine, a, ["cpu"], 400)
    assert _plan(engine, a, ["cpu", "cpu"], 400) == dense


def test_the_planner_bounds_the_fold_by_the_books() -> None:
    engine = _engine("cpu")
    a = _weights(engine.grid.shape)
    spec = engine._fold_spec(a, ScoringGrid.for_grid(engine.grid), 4, _source())
    group = _plan(engine, spec, ["cpu"], 7020, n_per=10**6)
    assert group == warp_engine._book_safe_group_size(7020, 10**6, _source().max_energy) < 7020


def _record_pulls(monkeypatch, barrier=None) -> list[tuple[str, int]]:
    """Record every group a shard pulls, as ``(device, group_start)``.

    With ``barrier``, each shard's first pull waits there after pulling, so no shard
    can take every group before the others have pulled once.
    """
    pulls: list[tuple[str, int]] = []
    original = warp_engine.WarpEngine._run_dij_shard

    def spy(self, device, next_group, **kwargs):
        first = [barrier is not None]

        def pull():
            group_start = next_group()
            if group_start is not None:
                pulls.append((device, group_start))
            if first[0]:
                first[0] = False
                barrier.wait()
            return group_start

        return original(self, device, pull, **kwargs)

    monkeypatch.setattr(warp_engine.WarpEngine, "_run_dij_shard", spy)
    return pulls


def test_a_book_bound_split_changes_no_bit(monkeypatch) -> None:
    """Groups the books force are scheduling only: values, sigma and books are unchanged."""
    engine = _engine("cpu")
    a = _weights(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED)
    whole = engine.run_beamlet_contraction(_source(), a, **kwargs)
    per_beamlet = 800 * _source().max_energy / ENERGY_QUANTUM_MEV
    budget_bits = math.ceil(math.log2(2 * per_beamlet))
    assert 2 * per_beamlet <= 2.0**budget_bits < 3 * per_beamlet  # two beamlets fit, not three
    monkeypatch.setattr(warp_engine, "_BOOK_BUDGET_BITS", budget_bits)
    pulls = _record_pulls(monkeypatch)
    split = engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert [start for _, start in pulls] == [0, 2, 4, 6, 8]
    np.testing.assert_array_equal(split.values, whole.values)
    np.testing.assert_array_equal(split.sigma, whole.sigma)
    assert split.energy_deposited == whole.energy_deposited
    assert split.energy_escaped == whole.energy_escaped
    assert split.energy_unscored == whole.energy_unscored
    assert split.energy_emitted == whole.energy_emitted


SHARD_DEVICES = ["cuda:0", "cpu"]


@pytest.mark.gpu
def test_a_multi_device_fold_gives_every_device_work(monkeypatch) -> None:
    """With ``devices=[cuda, cpu]`` both shards pull groups, each column its device's bits.

    With the fold's former single whole-call group the second shard's pull found
    nothing and that device idled. Ownership is timing-dependent, so each value must
    bit-match the whole run of *some* device (``test_dij_multi_device``'s contract).
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    import threading

    engine = _engine("cuda:0")
    a = _weights(engine.grid.shape)
    source = BeamletGridSource(2.0, -1.0, (2.0, 8.0), (2.0, 8.0), 4, 4)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    whole = [
        engine.run_beamlet_contraction(source, a, devices=[d], **kwargs) for d in SHARD_DEVICES
    ]
    pulls = _record_pulls(monkeypatch, threading.Barrier(2, timeout=120))
    sharded = engine.run_beamlet_contraction(source, a, devices=SHARD_DEVICES, **kwargs)
    assert {d for d, _ in pulls} == set(SHARD_DEVICES)
    for j in range(source.n_beamlets):
        assert any(
            sharded.values[j] == w.values[j] and sharded.sigma[j] == w.sigma[j] for w in whole
        ), f"beamlet {j} matches neither device's whole run"


def _launches(monkeypatch) -> set[tuple[str, str]]:
    """Record ``(kernel key, device)`` of every kernel launch (a stream names its device)."""
    seen: set[tuple[str, str]] = set()
    real = wp.launch

    def spy(kernel, *args, **kwargs):
        stream = kwargs.get("stream")
        device = stream.device if stream is not None else kwargs.get("device")
        seen.add((kernel.key, str(wp.get_device(device))))
        return real(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", spy)
    return seen


def test_fold_setup_stays_on_the_listed_devices(monkeypatch) -> None:
    """An engine on cuda:0 called with ``devices=['cpu']`` touches only the cpu.

    The fold's set-up (weights upload, scoring masses, the ratio and band kernels)
    used to run on the engine's own device even when ``devices`` excluded it.
    Building the engine touches no device, so this runs without CUDA too (where the
    old set-up raised).
    """
    grid = VoxelGrid.uniform_water(shape=(10, 10, 8), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine = warp_engine.WarpEngine(grid=grid, cross_sections=xs, device="cuda:0")
    a = _weights(grid.shape)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    expected = _engine("cpu").run_beamlet_contraction(_source(), a, **kwargs)
    seen = _launches(monkeypatch)
    result = engine.run_beamlet_contraction(_source(), a, devices=["cpu"], **kwargs)
    assert {device for _, device in seen} == {"cpu"}
    assert set(engine._resident) == {"cpu"}
    np.testing.assert_array_equal(result.values, expected.values)
    np.testing.assert_array_equal(result.sigma, expected.sigma)


@pytest.mark.gpu
def test_cuda_weights_on_a_cpu_call_set_up_on_the_cpu(monkeypatch) -> None:
    """Weights on a device the call does not run on are copied to one it does.

    The weights are validated where they lie (``device_voxel_weights``, one small
    launch, on either fold); beyond that the deposit fold launches nothing on cuda:0
    that the dense fold does not, where it used to run its whole set-up there.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    engine = _engine("cpu")
    a = _weights(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    expected = engine.run_beamlet_contraction(_source(), a, **kwargs)
    weights = wp.array(a, dtype=wp.float64, device="cuda:0")
    with monkeypatch.context() as patch:
        dense_launches = _launches(patch)
        engine.run_beamlet_contraction(_source(), weights, fold="dense", **kwargs)
    seen = _launches(monkeypatch)
    result = engine.run_beamlet_contraction(_source(), weights, **kwargs)
    on_cuda = {key for key, device in seen if device != "cpu"}
    assert on_cuda <= {key for key, device in dense_launches if device != "cpu"}
    np.testing.assert_array_equal(result.values, expected.values)
    np.testing.assert_array_equal(result.sigma, expected.sigma)


# -- the overflow guard checks every pass it returns ---------------------------------


def _record_passes(monkeypatch, quanta_factor=lambda n: 1) -> list[int | None]:
    """Record each transport pass of a contraction: its ``log2 K_0``, ``None`` if dense.

    ``quanta_factor(n)`` multiplies the ``sum |q|`` the guard reads after pass ``n``
    (counted over every call on the engine): what a source with larger weights, or
    one whose rerun deposits more than its first pass (the device pre-solve is
    order-nondeterministic at a fixed seed), shows the guard. Only the guard's input
    changes; the values are the real transport's.
    """
    passes: list[int | None] = []
    original = warp_engine.WarpEngine._compute_dij

    def spy(self, source, **kwargs):
        run = original(self, source, **kwargs)
        contraction = kwargs.get("contraction")
        fold = isinstance(contraction, warp_engine._FoldSpec)
        if fold:
            run.fold_abs_quanta_max *= quanta_factor(len(passes))
        passes.append(contraction.scale_exponent if fold else None)
        return run

    monkeypatch.setattr(warp_engine.WarpEngine, "_compute_dij", spy)
    return passes


def _assert_reaches_the_total_once(events, total: int) -> None:
    done = [e.histories_done for e in events]
    assert done and all(x < y for x, y in itertools.pairwise(done))
    assert done[-1] == total
    assert all(e.histories_total == total for e in events)


def test_a_rerun_past_the_guard_is_rescaled_again(monkeypatch, caplog) -> None:
    """The rerun's own ``sum |q|`` is checked: a rerun past the bound is never returned.

    ``_FOLD_SCALE_BITS = 74`` makes the first pass really wrap. The second pass's
    bound is raised 2**4, a rerun that deposits more than the pass it was scaled for,
    so its accumulators may have wrapped too: it must be rescaled and run again rather
    than trusted on the claim "same histories". The third pass holds and is returned.
    """
    engine = _engine("cpu")
    a = np.ones(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    dense = engine.run_beamlet_contraction(_source(), a, fold="dense", **kwargs)
    monkeypatch.setattr(warp_engine, "_FOLD_SCALE_BITS", 74)
    passes = _record_passes(monkeypatch, lambda n: 2**4 if n == 1 else 1)
    events: list = []
    with caplog.at_level(logging.INFO, logger=warp_engine.__name__):
        deposit = engine.run_beamlet_contraction(_source(), a, progress=events.append, **kwargs)
    assert len(passes) == 3 and None not in passes
    assert passes[0] > passes[1] > passes[2]
    assert not _fallbacks(caplog)
    mass = ScoringGrid.for_grid(engine.grid).voxel_mass
    _assert_within(deposit, dense, _guaranteed(a, mass, mass > 0.0))
    _assert_reaches_the_total_once(events, _source().n_beamlets * 400)


def test_a_fold_that_never_holds_runs_the_dense_fold(monkeypatch, caplog) -> None:
    """Passes that keep exceeding their guard end, bounded, in the dense fold: no wrap.

    Each pass's bound is raised 2**8 over the last, so no rescale ever holds. After
    ``_FOLD_PASSES_MAX`` deposit-fold passes the call runs the dense fold, logged,
    and returns exactly what ``fold="dense"`` returns.
    """
    engine = _engine("cpu")
    a = np.ones(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    dense = engine.run_beamlet_contraction(_source(), a, fold="dense", **kwargs)
    monkeypatch.setattr(warp_engine, "_FOLD_SCALE_BITS", 74)
    passes = _record_passes(monkeypatch, lambda n: 2 ** (8 * n))
    events: list = []
    with caplog.at_level(logging.INFO, logger=warp_engine.__name__):
        deposit = engine.run_beamlet_contraction(_source(), a, progress=events.append, **kwargs)
    assert passes[-1] is None
    n = warp_engine._FOLD_PASSES_MAX
    assert len(passes) == n + 1 and None not in passes[:n]
    assert len(_fallbacks(caplog)) == 1
    np.testing.assert_array_equal(deposit.values, dense.values)
    np.testing.assert_array_equal(deposit.sigma, dense.sigma)
    _assert_reaches_the_total_once(events, _source().n_beamlets * 400)


# -- the settled scale carries to the engine's next call -----------------------------

HEAVY = 2**12
"""``sum |q|`` factor of a source whose weights trip the guard at the a-priori scale."""


def _heavy(monkeypatch) -> tuple[list[int | None], dict[str, int]]:
    """Every pass's guard sees ``factor['q']`` times its ``sum |q|`` (``HEAVY`` at first)."""
    factor = {"q": HEAVY}
    return _record_passes(monkeypatch, lambda n: factor["q"]), factor


def _fold_logs(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == warp_engine.__name__ and r.levelno >= logging.INFO
    ]


def test_a_settled_rescale_carries_to_the_next_call(monkeypatch, caplog) -> None:
    """An optimizer loop's next call starts where the last one settled: one transport.

    A source whose weights trip the guard at the a-priori scale used to transport
    every call twice and log each time, because each call started again from the
    a-priori scale. The engine now remembers how far below the a-priori scale the
    last call settled; the next call starts there and transports once. It settles
    where a fresh engine would, so its bits are the first call's; and the memory is
    relative to the a-priori scale, so new weights (twice as large: one octave up)
    start one octave lower, again where they settle.
    """
    engine = _engine("cpu")
    a = _weights(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    passes, _ = _heavy(monkeypatch)
    first = engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert len(passes) == 2 and passes[0] > passes[1]
    settled = passes[1]
    with caplog.at_level(logging.INFO, logger=warp_engine.__name__):
        second = engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert passes[2:] == [settled]
    assert not _fold_logs(caplog)
    np.testing.assert_array_equal(second.values, first.values)
    np.testing.assert_array_equal(second.sigma, first.sigma)
    doubled = engine.run_beamlet_contraction(_source(), 2.0 * a, **kwargs)
    assert passes[3:] == [settled - 1]
    np.testing.assert_array_equal(doubled.values, 2.0 * first.values)


def test_a_remembered_scale_never_changes_a_result(monkeypatch) -> None:
    """The memory is a guess each call verifies, never a scale it is held to.

    After a tripping call, the same source with weights that no longer trip settles
    at the a-priori scale, as a fresh engine does: the remembered start is checked
    against the scale the call's own ``sum |q|`` gives and rerun there, so the result
    is a fresh engine's bit for bit, not the coarser remembered one. One outlier call
    therefore costs at most one extra transport and no resolution, and the memory is
    then cleared: the call after transports once, at the a-priori scale.
    """
    a = _weights((10, 10, 8))
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    fresh = _engine("cpu").run_beamlet_contraction(_source(), a, **kwargs)
    engine = _engine("cpu")
    passes, factor = _heavy(monkeypatch)
    engine.run_beamlet_contraction(_source(), a, **kwargs)
    a_priori, settled = passes
    factor["q"] = 1
    light = engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert passes[2:] == [settled, a_priori]
    np.testing.assert_array_equal(light.values, fresh.values)
    np.testing.assert_array_equal(light.sigma, fresh.sigma)
    engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert passes[4:] == [a_priori]


def test_a_remembered_scale_stays_with_its_source(monkeypatch) -> None:
    """Another source, or the same one in KERMA mode, starts from the a-priori scale.

    The memory is kept per source type, beamlet count, scoring mode and transport
    mode, so a loop alternating two contractions keeps one start for each.
    """
    engine = _engine("cpu")
    a = _weights(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    passes, _ = _heavy(monkeypatch)
    engine.run_beamlet_contraction(_source(), a, **kwargs)
    a_priori, settled = passes
    other = BeamletGridSource(2.0, -1.0, (2.0, 8.0), (2.0, 8.0), 2, 2)
    engine.run_beamlet_contraction(other, a, **kwargs)
    assert passes[2] == a_priori and len(passes) == 4
    engine.run_beamlet_contraction(_source(), a, transport_electrons=False, **kwargs)
    assert passes[4] == a_priori and len(passes) == 6
    engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert passes[6:] == [settled]


# -- the provenance says which fold, at which scale, produced the result -------------

FOLD_FIELDS = ("fold", "fold_requested", "fold_scale_exponent", "fold_bands", "fold_band_width")


def _without_fold(provenance) -> dict:
    """Every provenance field that is not the fold's."""
    record = dataclasses.asdict(provenance)
    for name in FOLD_FIELDS:
        record.pop(name)
    return record


def _assert_dense_record(provenance, requested: str) -> None:
    assert provenance.fold == "dense" and provenance.fold_requested == requested
    assert provenance.fold_scale_exponent is None
    assert provenance.fold_bands is None and provenance.fold_band_width is None


def test_provenance_tells_the_deposit_fold_from_the_dense_one() -> None:
    """Two folds differ at ~1e-10, so two results from them cannot share a record.

    Everything else in the record (seed, device, cuts) is identical; the fold, and for
    the deposit fold the scale and bands that fix its rounding, are what tell the two
    results apart.
    """
    engine = _engine("cpu")
    a = _weights(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    deposit, dense = _both(engine, _source(), a, **kwargs)
    scoring = ScoringGrid.for_grid(engine.grid)
    spec = engine._fold_spec(a, scoring, 100, _source())
    assert deposit.provenance is not None and dense.provenance is not None
    assert deposit.provenance.fold == "deposit" == deposit.provenance.fold_requested
    assert deposit.provenance.fold_scale_exponent == spec.scale_exponent
    assert deposit.provenance.fold_bands == spec.n_bands
    assert deposit.provenance.fold_band_width == spec.band_width
    _assert_dense_record(dense.provenance, "dense")
    assert deposit.provenance != dense.provenance
    assert _without_fold(deposit.provenance) == _without_fold(dense.provenance)
    assert "deposit" in deposit.provenance.summary()
    assert "dense" in dense.provenance.summary()


def test_provenance_of_a_dij_has_no_fold() -> None:
    """A Dij has no fold: the fields stay at their not-applicable ``None``."""
    engine = _engine("cpu")
    dij = engine.run_dij(_source(), n_histories_per_beamlet=100, n_batches=2, seed=SEED)
    assert dij.provenance is not None
    for name in FOLD_FIELDS:
        assert getattr(dij.provenance, name) is None


def test_a_fallback_to_the_dense_fold_is_recorded_as_dense(caplog) -> None:
    """A deposit fold that ran dense says so, and still says it was asked for deposit."""
    engine = _engine("cpu")
    a = np.ones(engine.grid.shape)
    a[5, 5, 2] = 2.0**-400  # beyond every band one call can hold
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    with caplog.at_level(logging.WARNING, logger=warp_engine.__name__):
        fell_back = engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert len(_fallbacks(caplog)) == 1
    asked_dense = engine.run_beamlet_contraction(_source(), a, fold="dense", **kwargs)
    assert fell_back.provenance is not None and asked_dense.provenance is not None
    _assert_dense_record(fell_back.provenance, "deposit")
    _assert_dense_record(asked_dense.provenance, "dense")
    assert fell_back.provenance != asked_dense.provenance  # same bits, different request
    np.testing.assert_array_equal(fell_back.values, asked_dense.values)


def test_a_fold_that_never_holds_is_recorded_as_dense(monkeypatch, caplog) -> None:
    """The guard's own dense fallback, after every pass failed, is recorded as dense."""
    engine = _engine("cpu")
    a = np.ones(engine.grid.shape)
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    monkeypatch.setattr(warp_engine, "_FOLD_SCALE_BITS", 74)
    passes = _record_passes(monkeypatch, lambda n: 2 ** (8 * n))
    with caplog.at_level(logging.INFO, logger=warp_engine.__name__):
        result = engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert passes[-1] is None
    assert result.provenance is not None
    _assert_dense_record(result.provenance, "deposit")


@pytest.mark.parametrize("device", DEVICES)
def test_a_guard_rescale_shows_in_the_recorded_scale(device: str, monkeypatch, caplog) -> None:
    """The scale the returned pass ran at is recorded, not the a-priori one.

    A different ``K_0`` gives different bits, so a result the overflow guard rescaled
    must not carry the record of the unscaled run. ``_FOLD_SCALE_BITS = 74`` trips the
    guard; the call settles at the rescale, and its record carries that exponent and
    the band width it implies, which differ from the record of the unforced run.
    """
    engine = _engine(device)
    a = np.ones(engine.grid.shape)
    a[5, 5, 2] = 2.0**-40  # re-banded at the rescaled scale
    kwargs = dict(n_histories_per_beamlet=800, n_batches=8, seed=SEED)
    plain = engine.run_beamlet_contraction(_source(), a, **kwargs)
    monkeypatch.setattr(warp_engine, "_FOLD_SCALE_BITS", 74)
    passes = _record_passes(monkeypatch)
    with caplog.at_level(logging.INFO, logger=warp_engine.__name__):
        rescaled = engine.run_beamlet_contraction(_source(), a, **kwargs)
    assert len(passes) == 2 and None not in passes and passes[1] != passes[0]
    assert plain.provenance is not None and rescaled.provenance is not None
    assert plain.provenance.fold == rescaled.provenance.fold == "deposit"
    assert rescaled.provenance.fold_scale_exponent == passes[1]
    assert rescaled.provenance.fold_scale_exponent != plain.provenance.fold_scale_exponent
    assert rescaled.provenance.fold_band_width != plain.provenance.fold_band_width
    assert rescaled.provenance != plain.provenance
