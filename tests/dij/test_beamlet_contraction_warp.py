"""Beamlet contraction on Warp: the Dij group machinery with a contraction fold.

``WarpEngine.run_beamlet_contraction`` transports exactly the histories
``run_dij`` transports (same routes and mapping) and replaces only the scoring: by
default each deposit is weighted and folded into its beamlet's int64 accumulator
(the deposit fold, pinned against the dense fold in
``test_contraction_deposit_fold_warp``), so no dense dose map, truncation or
compaction exists. Pinned against its own untruncated Dij contracted afterwards
(same target, float64 rounding and the deposit fold's weighted quantum apart, both
orders below the tolerance), against itself run to run (bitwise), across batch
lanes (bitwise), and statistically against the reference contraction.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.contraction_fixtures import engine as _engine
from tests.contraction_fixtures import phantom_grid as _grid
from tests.contraction_fixtures import phantom_xs as _xs
from tests.contraction_fixtures import source as _source
from tests.contraction_fixtures import weights as _weights

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
KWARGS = dict(n_batches=8, transport_electrons=False)


@pytest.mark.parametrize("device", DEVICES)
def test_contraction_equals_its_own_contracted_dij(device: str) -> None:
    engine = _engine(device)
    a = _weights(engine.grid.shape)
    dij = engine.run_dij(
        _source(), n_histories_per_beamlet=800, seed=SEED, truncation=0.0, **KWARGS
    )
    result = engine.run_beamlet_contraction(
        _source(), a, n_histories_per_beamlet=800, seed=SEED, **KWARGS
    )
    expected = np.array([np.sum(a * dij.column_dense(j)) for j in range(dij.n_beamlets)])
    np.testing.assert_allclose(result.values, expected, rtol=1e-9, atol=1e-14)
    assert result.energy_emitted == dij.energy_emitted
    assert result.energy_deposited == dij.energy_deposited
    assert result.energy_escaped == dij.energy_escaped
    assert np.all(result.sigma > 0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_contraction_is_bit_reproducible(device: str) -> None:
    engine = _engine(device)
    a = _weights(engine.grid.shape)
    runs = [
        engine.run_beamlet_contraction(
            _source(), a, n_histories_per_beamlet=800, seed=SEED, beamlet_group_size=g, **KWARGS
        )
        for g in (9, 9, 4)
    ]
    np.testing.assert_array_equal(runs[0].values, runs[1].values)
    np.testing.assert_array_equal(runs[0].sigma, runs[1].sigma)
    # The group size is scheduling only: each column is computed whole in one group.
    np.testing.assert_array_equal(runs[0].values, runs[2].values)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("flat", [False, True])
def test_device_weights_give_the_numpy_weights_result(device: str, flat: bool) -> None:
    """A float64 Warp array of the weights is used where it lies, with the same result."""
    engine = _engine(device)
    a = _weights(engine.grid.shape)
    on_device = wp.array(a.ravel() if flat else a, dtype=wp.float64, device=device)
    host, dev = (
        engine.run_beamlet_contraction(
            _source(), w, n_histories_per_beamlet=800, seed=SEED, beamlet_group_size=4, **KWARGS
        )
        for w in (a, on_device)
    )
    np.testing.assert_array_equal(host.values, dev.values)
    np.testing.assert_array_equal(host.sigma, dev.sigma)
    assert host.energy_deposited == dev.energy_deposited
    np.testing.assert_array_equal(on_device.numpy().reshape(a.shape), a)  # left untouched


@pytest.mark.gpu
def test_device_weights_on_another_device_are_copied_there() -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    engine = _engine("cpu")
    a = _weights(engine.grid.shape)
    run = dict(n_histories_per_beamlet=800, seed=SEED, **KWARGS)
    host, dev = (
        engine.run_beamlet_contraction(_source(), w, **run)
        for w in (a, wp.array(a, dtype=wp.float64, device="cuda:0"))
    )
    np.testing.assert_array_equal(host.values, dev.values)


@pytest.mark.parametrize("device", DEVICES)
def test_device_weights_are_validated_like_numpy_weights(device: str) -> None:
    engine = _engine(device)
    a = _weights(engine.grid.shape)
    run = dict(n_histories_per_beamlet=800, seed=SEED, **KWARGS)
    with pytest.raises(ValueError, match="shape"):
        engine.run_beamlet_contraction(
            _source(), wp.array(a[:-1], dtype=wp.float64, device=device), **run
        )
    with pytest.raises(ValueError, match="float64"):
        engine.run_beamlet_contraction(
            _source(), wp.array(a, dtype=wp.float32, device=device), **run
        )
    bad = a.copy()
    bad[3, 4, 5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        engine.run_beamlet_contraction(
            _source(), wp.array(bad, dtype=wp.float64, device=device), **run
        )


@pytest.mark.gpu
def test_contraction_is_inert_to_batch_lanes() -> None:
    engine = _engine("cuda:0")
    a = _weights(engine.grid.shape)
    one, four = (
        engine.run_beamlet_contraction(
            _source(), a, n_histories_per_beamlet=800, seed=SEED, concurrent_batches=lanes, **KWARGS
        )
        for lanes in (1, 4)
    )
    np.testing.assert_array_equal(one.values, four.values)
    np.testing.assert_array_equal(one.sigma, four.sigma)


def test_contraction_agrees_with_the_reference_engine() -> None:
    """Independent streams: correlated columns share fluctuations (AGENTS.md 8)."""
    independent = dict(KWARGS, correlated=False)
    warp_result = _engine("cpu").run_beamlet_contraction(
        _source(), _weights(_grid().shape), n_histories_per_beamlet=1_600, seed=SEED, **independent
    )
    ref = ReferenceEngine(grid=_grid(), cross_sections=_xs(_grid()), rng=HostRNG())
    ref_result = ref.run_beamlet_contraction(
        _source(), _weights(_grid().shape), n_histories_per_beamlet=400, seed=SEED, **independent
    )
    assert_chi2_consistent_batched(
        ref_result.values,
        ref_result.sigma,
        ref_result.n_batches,
        warp_result.values,
        warp_result.sigma,
        warp_result.n_batches,
    )
