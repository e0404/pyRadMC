"""Stochastic Dij truncation on the Warp engine.

The device applies the reference engine's roulette rule (``roulette_dij_entry``, in
float64) with its own counter-based uniform ``keyed_uniform(seed ^ DIJ_ROULETTE_SALT,
global column, voxel)`` (:mod:`pyradmc.backends.warp.sparsify`). Pinned here:

- **the oracle**: stochastic against untruncated, same seed and device, so the dense
  columns are bit-identical before sparsification and only the roulette's known
  variance ``sum m (t - m)`` remains — at ``N`` and ``4 N``, with the hard rule's
  growing bias alongside, as on the reference engine
  (``test_dij_stochastic_truncation_ref.py``, same geometry and counts);
- **within-device determinism**: repeat runs, ``beamlet_group_size``, ``chunk_size``,
  the truncation chunk layout, ``concurrent_batches`` and sharding leave every bit
  in place, and the device-output path equals the host-returning one;
- **what the engine does**: its stochastic matrix is exactly the device kernels'
  roulette of its own untruncated matrix with the salted seed and global columns;
- **reference against Warp**, statistically only: the fraction of the column sum
  the mode keeps agrees between the backends within the two roulette sigmas;
- **DVH endpoints** (AGENTS.md 2.8), on the plan of the Warp truncation DVH test.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest

from pyradmc import DIJ_TRUNCATION_RELATIVE
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource
from pyradmc.rng.host import HostRNG
from pyradmc.scoring.dij import DijResult
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from tests.dij.test_dij_device_output import _assert_identical, _case  # noqa: E402
from tests.dij.test_dij_device_output import _engine as _output_engine  # noqa: E402
from tests.integration.test_warp_stochastic_truncation import _sweep  # noqa: E402

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

N = 2_000
# Two-sided normal false-alarm probability of 6.3e-5 per z-score assertion.
K_SIGMA = 4.0
ARRAYS = ("indptr", "indices", "dose", "sigma")


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _warp(device: str, **kwargs):
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device, **kwargs)


def _beamlet(n_x: int = 1, n_y: int = 1) -> BeamletGridSource:
    """1 cm beamlets centred on the cube: the reference test's source."""
    return BeamletGridSource(
        energy=6.0,
        z=-1.0,
        x_range=(8.0 - 0.5 * n_x, 8.0 + 0.5 * n_x),
        y_range=(8.0 - 0.5 * n_y, 8.0 + 0.5 * n_y),
        n_x=n_x,
        n_y=n_y,
    )


def _run(engine, n: int, source: BeamletGridSource | None = None, **kwargs) -> DijResult:
    return engine.run_dij(
        _beamlet() if source is None else source,
        n_histories_per_beamlet=n,
        n_batches=4,
        seed=SEED,
        transport_electrons=False,
        **kwargs,
    )


def _roulette_variance(full: DijResult, j: int) -> tuple[float, int]:
    m = full.column_dense(j).ravel()
    t = DIJ_TRUNCATION_RELATIVE * m.max()
    tail = m[(m > 0.0) & (m < t)]
    return float(np.sum(tail * (t - tail))), int(tail.size)


def _assert_bitwise(a: DijResult, b: DijResult) -> None:
    for name in ARRAYS:
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype, name
        assert np.array_equal(x, y), name


@pytest.fixture(scope="module")
def single(device: str) -> dict[tuple[int, str], DijResult]:
    engine = _warp(device)
    out: dict[tuple[int, str], DijResult] = {}
    for n in (N, 4 * N):
        out[n, "full"] = _run(engine, n, truncation=0.0)
        out[n, "stochastic"] = _run(engine, n, truncation_mode="stochastic")
        out[n, "hard"] = _run(engine, n)
    return out


class TestColumnSum:
    @pytest.mark.parametrize("n", [N, 4 * N])
    def test_stochastic_column_sum_is_the_untruncated_one(
        self, single: dict[tuple[int, str], DijResult], n: int
    ) -> None:
        full, sto = single[n, "full"], single[n, "stochastic"]
        assert sto.truncation_mode == "stochastic"
        variance, n_tail = _roulette_variance(full, 0)
        assert n_tail > 50, "the tail must hold sub-threshold entries"
        assert sto.indices.size < full.indices.size, "the mode must sparsify"
        z = (sto.dose.sum() - full.dose.sum()) / math.sqrt(variance)
        assert abs(z) < K_SIGMA, f"stochastic column sum off by z = {z:.2f} at N = {n}"

    def test_hard_mode_removes_more_than_the_roulette_sigma(
        self, single: dict[tuple[int, str], DijResult]
    ) -> None:
        """Characterization of the defect on this engine, at 4N."""
        full, hard = single[4 * N, "full"], single[4 * N, "hard"]
        variance, _ = _roulette_variance(full, 0)
        removed = full.dose.sum() - hard.dose.sum()
        assert removed > K_SIGMA * math.sqrt(variance)

    def test_kept_fraction_agrees_with_the_reference_engine(
        self, single: dict[tuple[int, str], DijResult]
    ) -> None:
        """Ratio of stochastic to untruncated column sum, per backend, at 4N.

        Within a backend the Monte Carlo noise cancels, so each ratio is 1 up to
        its roulette sigma; the two backends' ratios must agree within the
        quadrature of both (they draw independently, so nothing tighter holds).
        """
        ref = ReferenceEngine(grid=_grid(), cross_sections=_xs(_grid()), rng=HostRNG())
        ref_full = _run(ref, 4 * N, truncation=0.0)
        ref_sto = _run(ref, 4 * N, truncation_mode="stochastic")
        ratios, sigmas = [], []
        for full, sto in (
            (ref_full, ref_sto),
            (single[4 * N, "full"], single[4 * N, "stochastic"]),
        ):
            variance, _ = _roulette_variance(full, 0)
            ratios.append(sto.dose.sum() / full.dose.sum())
            sigmas.append(math.sqrt(variance) / full.dose.sum())
        z = (ratios[1] - ratios[0]) / math.hypot(*sigmas)
        assert abs(z) < K_SIGMA, (ratios, sigmas)

    def test_the_engine_roulettes_its_own_untruncated_matrix(
        self, device: str, single: dict[tuple[int, str], DijResult]
    ) -> None:
        """Stochastic == the sparsify kernels on the untruncated column, same device.

        Keyed on ``seed ^ DIJ_ROULETTE_SALT`` and the global column: a missing salt,
        a block-local key, or a pass that disagrees with the other breaks it.
        """
        from pyradmc.scoring.dij import DIJ_ROULETTE_SALT

        full, sto = single[4 * N, "full"], single[4 * N, "stochastic"]
        mean = full.column_dense(0).reshape(1, -1)
        sigma = full.sigma_dense(0).reshape(1, -1)
        counts, idx, val, sig = _sweep(
            device, mean, sigma, DIJ_TRUNCATION_RELATIVE, roulette_seed=SEED ^ DIJ_ROULETTE_SALT
        )
        assert np.array_equal(sto.indptr, [0, counts[0]])
        assert np.array_equal(sto.indices, idx)
        assert np.array_equal(sto.dose, val)
        assert np.array_equal(sto.sigma, sig)


class TestDeterminism:
    """Within one device, the stochastic matrix is as scheduling-inert as the hard one."""

    KW: ClassVar[dict[str, object]] = dict(
        n_histories_per_beamlet=800, n_batches=4, seed=SEED, truncation_mode="stochastic"
    )

    def _lattice_run(self, device: str, engine_kw=None, **kw) -> DijResult:
        return _warp(device, **(engine_kw or {})).run_dij(_beamlet(2, 2), **self.KW, **kw)

    def test_repeat_runs_are_bit_identical(self, device: str) -> None:
        _assert_bitwise(self._lattice_run(device), self._lattice_run(device))

    def test_beamlet_group_size_is_inert(self, device: str) -> None:
        """Group 1 puts every column at local index 0: a local key would collide."""
        want = self._lattice_run(device, beamlet_group_size=4)
        for group in (1, 3):
            _assert_bitwise(self._lattice_run(device, beamlet_group_size=group), want)

    def test_chunk_size_is_inert(self, device: str) -> None:
        want = self._lattice_run(device, beamlet_group_size=4)
        got = self._lattice_run(device, engine_kw=dict(chunk_size=300), beamlet_group_size=4)
        _assert_bitwise(got, want)

    def test_truncation_chunk_layout_is_inert(self, device: str, monkeypatch) -> None:
        from pyradmc.backends.warp import kernels

        want = self._lattice_run(device)
        for target in (1, 10_000_000):
            monkeypatch.setattr(kernels, "TRUNCATION_TARGET_THREADS", target)
            _assert_bitwise(self._lattice_run(device), want)

    def test_zero_truncation_equals_the_hard_mode(self, device: str) -> None:
        engine = _warp(device)
        kw = dict(self.KW, truncation=0.0)
        sto = engine.run_dij(_beamlet(2, 2), **kw)
        kw["truncation_mode"] = "hard"
        _assert_bitwise(sto, engine.run_dij(_beamlet(2, 2), **kw))

    def test_unknown_mode_is_refused_before_transport(self, device: str) -> None:
        engine = _warp(device)
        for method in (engine.run_dij, engine.run_dij_device):
            with pytest.raises(ValueError, match="truncation_mode"):
                method(_beamlet(), 4, 1, SEED, truncation_mode="soft")


@pytest.mark.gpu
@pytest.mark.parametrize("lanes", [2, 3])
def test_concurrent_batches_are_inert(lanes: int) -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    kw = dict(
        n_histories_per_beamlet=600,
        n_batches=6,
        seed=SEED,
        beamlet_group_size=2,
        truncation_mode="stochastic",
    )
    want = _warp("cuda:0").run_dij(_beamlet(2, 2), **kw)
    _assert_bitwise(_warp("cuda:0").run_dij(_beamlet(2, 2), concurrent_batches=lanes, **kw), want)


@pytest.mark.gpu
def test_sharded_columns_are_bit_identical_to_their_own_device() -> None:
    """Sharded over cuda + cpu, every column equals its owner's whole run (global key)."""
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    kw = dict(
        n_histories_per_beamlet=400,
        n_batches=4,
        seed=SEED,
        beamlet_group_size=1,
        truncation_mode="stochastic",
    )
    source = _beamlet(2, 2)
    sharded = _warp("cuda:0").run_dij(source, devices=["cuda:0", "cpu"], **kw)
    whole = {d: _warp(d).run_dij(source, **kw) for d in ("cuda:0", "cpu")}
    for j in range(source.n_beamlets):
        lo, hi = int(sharded.indptr[j]), int(sharded.indptr[j + 1])
        col = (sharded.indices[lo:hi], sharded.dose[lo:hi], sharded.sigma[lo:hi])
        matches = []
        for w in whole.values():
            a, b = int(w.indptr[j]), int(w.indptr[j + 1])
            want = (w.indices[a:b], w.dose[a:b], w.sigma[a:b])
            matches.append(all(np.array_equal(x, y) for x, y in zip(col, want, strict=True)))
        assert any(matches), f"column {j} bit-matches neither device's whole run"


@pytest.mark.parametrize("kind", ["lattice", "segmented"])
def test_device_output_equals_the_host_returning_path(device: str, kind: str) -> None:
    """Both paths run the same kernels: ``to_host()`` equals ``run_dij`` field for field."""
    grid, source, kwargs = _case(kind)
    engine = _output_engine(grid, device)
    common = dict(seed=SEED, beamlet_group_size=2, truncation_mode="stochastic", **kwargs)
    want = engine.run_dij(source, **common)
    got = engine.run_dij_device(source, **common)
    assert got.truncation_mode == "stochastic"
    _assert_identical(got.to_host(), want)
    assert want.truncation_mode == "stochastic"


@pytest.fixture(scope="module")
def plan(device: str):
    from pyradmc.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(16, 16, 24), spacing=(1.0, 1.0, 0.5))
    engine = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device)
    source = BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(2.0, 14.0), y_range=(2.0, 14.0), n_x=3, n_y=3
    )
    kwargs = dict(n_histories_per_beamlet=24_000, n_batches=4, seed=SEED)
    full = engine.run_dij(source, truncation=0.0, **kwargs)
    sto = engine.run_dij(source, truncation_mode="stochastic", **kwargs)
    hard = engine.run_dij(source, **kwargs)
    w = np.ones(source.n_beamlets)
    variance = np.zeros(full.n_voxels)
    reach = np.zeros(full.n_voxels)
    for j in range(source.n_beamlets):
        m = full.column_dense(j).ravel()
        t = DIJ_TRUNCATION_RELATIVE * m.max()
        tail = (m > 0.0) & (m < t)
        variance += np.where(tail, w[j] ** 2 * m * (t - m), 0.0)
        reach += np.where(tail, w[j] * np.maximum(m, t - m), 0.0)
    return dict(
        full=full.dose_for_weights(w).ravel(),
        sto=sto.dose_for_weights(w).ravel(),
        hard=hard.dose_for_weights(w).ravel(),
        variance=variance,
        reach=reach,
        nnz=(full.indices.size, sto.indices.size),
    )


class TestStochasticTruncationAgainstDVH:
    """AGENTS.md 2.8 on the plan of ``test_dij_warp.py::TestTruncationAgainstDVH``.

    3 x 3 lattice, 24000 histories per beamlet, full transport, stochastic against
    untruncated from the same seed on one device. Assertions as on the reference
    engine: the ROI mean dose is linear in the matrix, so its shift is a z-score
    against the roulette's variance ``sqrt(sum_i sigma_i^2) / n`` with
    ``sigma_i^2 = sum_j w_j^2 m_ij (t_j - m_ij)``; a DVH percentile moves by at most
    the roulette's deterministic reach ``max_i sum_j w_j max(m_ij, t_j - m_ij)``
    (no false alarm), asserted in the high-dose ROI, where that reach is below
    0.5 % of the plan maximum.
    """

    @staticmethod
    def _rois(full: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "high": full > 0.1 * full.max(),
            "low": (full > 0.0) & (full < 0.01 * full.max()),
        }

    def test_mean_dose_is_unbiased(self, plan) -> None:
        n_full, n_sto = plan["nnz"]
        assert n_sto < 0.95 * n_full, "non-vacuity: the mode must sparsify"
        for name, roi in self._rois(plan["full"]).items():
            n = int(np.count_nonzero(roi))
            assert n > 200, name
            sigma_mean = math.sqrt(float(plan["variance"][roi].sum())) / n
            shift = float(plan["sto"][roi].mean() - plan["full"][roi].mean())
            if sigma_mean == 0.0:
                assert shift == 0.0, name
                continue
            assert abs(shift / sigma_mean) < K_SIGMA, (name, shift / sigma_mean)

    def test_hard_rule_biases_the_low_dose_mean(self, plan) -> None:
        roi = self._rois(plan["full"])["low"]
        n = int(np.count_nonzero(roi))
        sigma_mean = math.sqrt(float(plan["variance"][roi].sum())) / n
        shift = float(plan["hard"][roi].mean() - plan["full"][roi].mean())
        assert shift < -K_SIGMA * sigma_mean, (shift, sigma_mean)

    @pytest.mark.parametrize("volume_percent", [2.0, 50.0, 98.0])
    def test_high_dose_dvh_endpoints_within_the_roulette_reach(
        self, plan, volume_percent: float
    ) -> None:
        roi = self._rois(plan["full"])["high"]
        reach = float(plan["reach"][roi].max())
        d_full = float(np.percentile(plan["full"][roi], 100.0 - volume_percent))
        d_sto = float(np.percentile(plan["sto"][roi], 100.0 - volume_percent))
        # Non-vacuity on the clinical scale (dose differences against the maximum):
        # the 0.1-max boundary of this ROI puts D98 near 10 % of the maximum, where
        # the reach is ~2.5 % of the endpoint itself but 0.25 % of the maximum.
        assert reach < 5.0e-3 * float(plan["full"].max()), "non-vacuity: the bound must be tight"
        assert abs(d_sto - d_full) <= reach, (volume_percent, d_sto - d_full, reach)
