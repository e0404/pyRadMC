"""A segmented fan's plan dose in one ``run()``: segments summed with MU weights.

:meth:`SegmentedFanSource.plan` turns the segments into one open-field source whose
histories pick a segment with probability ``w_s / W`` and carry ``W`` times its
transmission, so ``run`` estimates ``sum_s w_s * column_s`` of the segment Dij
without building it. Anchors: that weighted column sum (from an *independent* Dij,
whose quadrature sigma is exact, AGENTS.md section 8), the reference engine, the
ledger, and the exact-type routing (a subclass keeps host pre-sampling).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.segments import SegmentedFanSource, SegmentedPlanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.dij.test_segmented_fan_dij import SPECTRUM, _fan, _grid, _source, _stacks, _warp, _xs

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

WEIGHTS = np.array([2.0, 1.0, 3.0])


def _require(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")


@pytest.mark.parametrize("device", DEVICES)
def test_plan_run_is_the_weighted_column_sum(device: str) -> None:
    _require(device)
    engine = _warp(_grid(), device)
    source = _source()
    n = 30_000
    kerma = dict(transport_electrons=False)
    plan = engine.run(source.plan(WEIGHTS), n_histories=n, n_batches=8, seed=SEED + 1, **kerma)
    # Independent columns: the quadrature sigma of the weighted sum is exact.
    dij = engine.run_dij(
        source,
        n_histories_per_beamlet=n,
        n_batches=8,
        seed=SEED,
        correlated=False,
        truncation=0.0,
        **kerma,
    )
    total = sum(w * dij.column_dense(s) for s, w in enumerate(WEIGHTS))
    sigma = np.sqrt(sum((w * dij.sigma_dense(s)) ** 2 for s, w in enumerate(WEIGHTS)))
    mask = (plan.dose > 0.1 * plan.dose.max()) & (plan.dose_sigma > 0.0) & (sigma > 0.0)
    assert mask.sum() > 200
    assert_chi2_consistent_batched(plan.dose, plan.dose_sigma, 8, total, sigma, 8, mask=mask)
    assert plan.energy_emitted == pytest.approx(
        plan.energy_deposited + plan.energy_escaped + plan.energy_unscored, rel=1e-6
    )


def test_plan_run_matches_the_reference() -> None:
    grid = _grid()
    plan = _source().plan(WEIGHTS)
    kwargs = dict(n_histories=8_000, n_batches=8, seed=SEED, transport_electrons=False)
    ref = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run(plan, **kwargs)
    got = _warp(grid, "cpu").run(plan, **kwargs)
    mask = (ref.dose > 0.1 * ref.dose.max()) & (ref.dose_sigma > 0.0) & (got.dose_sigma > 0.0)
    assert mask.sum() > 100
    assert_chi2_consistent_batched(
        ref.dose, ref.dose_sigma, 8, got.dose, got.dose_sigma, 8, mask=mask
    )


def test_exact_plan_type_never_calls_the_host_samplers(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the exact type must be generated in-kernel")

    monkeypatch.setattr(SegmentedPlanSource, "sample_batch", refuse)
    monkeypatch.setattr(SegmentedPlanSource, "emit", refuse)
    monkeypatch.setattr(SegmentedFanSource, "transmission", refuse)
    monkeypatch.setattr(SpectralBeamSource, "sample_batch", refuse)
    result = _warp(_grid(), "cpu").run(
        _source().plan(WEIGHTS), n_histories=400, n_batches=2, seed=SEED, transport_electrons=False
    )
    assert result.energy_emitted > 0.0


class _HalfWeightPlan(SegmentedPlanSource):
    def sample_batch(self, seed: int, history_offset: int, n: int):
        columns = super().sample_batch(seed, history_offset, n)
        columns["weight"] = np.full(n, 0.5, dtype=np.float32)
        return columns


def test_a_plan_subclass_keeps_the_host_route() -> None:
    plan = _HalfWeightPlan(_source(), WEIGHTS)
    result = _warp(_grid(), "cpu").run(
        plan, n_histories=600, n_batches=2, seed=SEED, transport_electrons=False
    )
    energies = _fan().sample_batch(SEED, 0, 600)["energy"].astype(np.float64)
    assert result.energy_emitted == pytest.approx(0.5 * energies.sum(), rel=1e-6)


def test_a_plan_over_segment_subclasses_keeps_the_host_route(monkeypatch) -> None:
    class Segments(SegmentedFanSource):
        pass

    segments = Segments.from_stacks(_fan(), _stacks(), AnalyticCrossSections())
    called = []
    original = SegmentedPlanSource.sample_batch

    def spy(self, *args, **kwargs):
        called.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SegmentedPlanSource, "sample_batch", spy)
    _warp(_grid(), "cpu").run(
        segments.plan(WEIGHTS), n_histories=400, n_batches=2, seed=SEED, transport_electrons=False
    )
    assert called


@pytest.mark.gpu
def test_plan_concurrent_batches_are_bit_inert() -> None:
    _require("cuda:0")
    engine = _warp(_grid(), "cuda:0")
    plan = _source().plan(WEIGHTS)
    kwargs = dict(n_histories=16_000, n_batches=4, seed=SEED)
    sequential = engine.run(plan, **kwargs)
    lanes = engine.run(plan, concurrent_batches=2, **kwargs)
    np.testing.assert_array_equal(sequential.dose, lanes.dose)
    np.testing.assert_array_equal(sequential.dose_sigma, lanes.dose_sigma)
    assert sequential.energy_emitted == lanes.energy_emitted


def test_spectrum_is_shared() -> None:
    assert _source().plan(WEIGHTS).max_energy == SPECTRUM.max_energy
