"""Warp Dij against the reference oracle and its own scheduling invariances.

The design claim: because streams are pure functions of (seed, history)
and scoring is associative int64, *how* the engine schedules beamlets — one per
launch, grouped, any chunk size — cannot change the Dij by a single bit on one
device. That claim is pinned here. Statistical equivalence with the reference
Dij runs under full coupled transport, because that is what exercises beamlet
tag inheritance through the photon-electron queue ping-pong. Cross-target
comparisons are chi-squared only (AGENTS.md 2.3 and 4).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC import DIJ_TRUNCATION_RELATIVE
from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletGridSource, ParallelBeamSource
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

ENERGY_BALANCE_RTOL = 1.0e-4  # float32 transport + 1e-9 MeV scoring quanta

FIELD_X = (2.0, 14.0)
FIELD_Y = (2.0, 14.0)
ENERGY = 6.0
Z0 = -1.0


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(2.0, 2.0, 0.5))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _lattice(n_x: int, n_y: int) -> BeamletGridSource:
    return BeamletGridSource(
        energy=ENERGY, z=Z0, x_range=FIELD_X, y_range=FIELD_Y, n_x=n_x, n_y=n_y
    )


def _warp_engine(device: str, **kwargs):
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device, **kwargs)


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


class TestSchedulingIsBitInert:
    """The design's central claim: scheduling freedom, bit-identical columns."""

    def test_beamlet_group_size_does_not_change_the_dij(self, device: str) -> None:
        results = [
            _warp_engine(device).run_dij(
                _lattice(2, 2),
                n_histories_per_beamlet=800,
                n_batches=4,
                seed=SEED,
                beamlet_group_size=group,
            )
            for group in (1, 3, 4)
        ]
        for other in results[1:]:
            np.testing.assert_array_equal(results[0].indptr, other.indptr)
            np.testing.assert_array_equal(results[0].indices, other.indices)
            np.testing.assert_array_equal(results[0].dose, other.dose)
            np.testing.assert_array_equal(results[0].sigma, other.sigma)
            assert results[0].energy_deposited == other.energy_deposited
            assert results[0].energy_escaped == other.energy_escaped

    def test_chunk_size_does_not_change_the_dij(self, device: str) -> None:
        results = [
            _warp_engine(device, chunk_size=chunk).run_dij(
                _lattice(2, 2),
                n_histories_per_beamlet=800,
                n_batches=4,
                seed=SEED,
                beamlet_group_size=4,
            )
            for chunk in (300, 8_192)
        ]
        np.testing.assert_array_equal(results[0].dose, results[1].dose)
        np.testing.assert_array_equal(results[0].indices, results[1].indices)

    def test_repeat_runs_are_bit_identical(self, device: str) -> None:
        runs = [
            _warp_engine(device).run_dij(
                _lattice(2, 1), n_histories_per_beamlet=600, n_batches=3, seed=SEED
            )
            for _ in range(2)
        ]
        np.testing.assert_array_equal(runs[0].dose, runs[1].dose)
        np.testing.assert_array_equal(runs[0].sigma, runs[1].sigma)
        assert runs[0].energy_deposited == runs[1].energy_deposited


class TestSingleBeamletIsTheOpenField:
    def test_column_equals_open_field_run_bitwise(self, device: str) -> None:
        """1x1 lattice, untruncated: run_dij degenerates to run(), bit for bit.

        The lattice generator must consume the same draws and do the same float32
        position arithmetic as the parallel-beam generator for this to hold.
        """
        engine = _warp_engine(device)
        dij = engine.run_dij(
            _lattice(1, 1),
            n_histories_per_beamlet=2_000,
            n_batches=4,
            seed=SEED,
            truncation=0.0,
        )
        run = engine.run(
            ParallelBeamSource(energy=ENERGY, z=Z0, x_range=FIELD_X, y_range=FIELD_Y),
            n_histories=2_000,
            n_batches=4,
            seed=SEED,
        )
        np.testing.assert_array_equal(dij.column_dense(0), run.dose)
        np.testing.assert_array_equal(dij.sigma_dense(0), run.dose_sigma)
        # The dose is bitwise; the energy books cross two float summation paths
        # (run() sums float64 per batch, run_dij tallies exact quanta), so they
        # agree to accumulation roundoff only.
        assert dij.energy_deposited == pytest.approx(run.energy_deposited, rel=1.0e-12)
        assert dij.energy_escaped == pytest.approx(run.energy_escaped, rel=1.0e-12)


class TestStatisticalEquivalenceWithReference:
    def test_columns_match_reference_dij(self, device: str) -> None:
        """Per-column chi-squared, full coupled transport: tags survive the queues."""
        grid = _grid()
        ref_engine = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())
        ref = ref_engine.run_dij(
            _lattice(2, 2),
            n_histories_per_beamlet=400,
            n_batches=8,
            seed=SEED,
            truncation=0.0,
        )
        result = _warp_engine(device).run_dij(
            _lattice(2, 2),
            n_histories_per_beamlet=1_600,
            n_batches=8,
            seed=SEED,
            truncation=0.0,
        )
        for j in range(4):
            ref_dose = ref.column_dense(j)
            ref_sigma = ref.sigma_dense(j)
            warp_dose = result.column_dense(j)
            warp_sigma = result.sigma_dense(j)
            mask = ref_dose > 0.1 * ref_dose.max()
            mask &= (ref_sigma > 0.0) & (warp_sigma > 0.0)
            assert np.count_nonzero(mask) > 50, f"column {j}: mask too small"
            assert_chi2_consistent_batched(
                ref_dose,
                ref_sigma,
                ref.n_batches,
                warp_dose,
                warp_sigma,
                result.n_batches,
                mask=mask,
            )


class TestBookkeeping:
    def test_energy_conservation(self, device: str) -> None:
        dij = _warp_engine(device).run_dij(
            _lattice(2, 2), n_histories_per_beamlet=500, n_batches=2, seed=SEED
        )
        balance = dij.energy_deposited + dij.energy_escaped
        assert balance == pytest.approx(dij.energy_emitted, rel=ENERGY_BALANCE_RTOL)
        assert dij.energy_emitted == pytest.approx(4 * 500 * ENERGY, rel=1.0e-12)

    def test_group_size_must_be_positive(self, device: str) -> None:
        with pytest.raises(ValueError):
            _warp_engine(device).run_dij(
                _lattice(1, 1),
                n_histories_per_beamlet=100,
                n_batches=1,
                seed=SEED,
                beamlet_group_size=0,
            )


class TestCorrelatedSampling:
    """The stream remap on the kernel path.

    The mapping specification lives in ``ReferenceEngine.run_dij`` and is
    pinned in ``test_dij_correlated.py``; here only what the kernel adds needs
    pinning: the remap keys slots on the within-beamlet index, scheduling
    stays bit-inert under it, and the correlated columns remain statistically
    equivalent to the reference's correlated columns. Electronless where
    statistics suffice — the remap changes stream keys only, and coupled
    transport is exercised by ``TestStatisticalEquivalenceWithReference``.
    """

    def test_default_is_correlated_and_beamlet_zero_is_mode_invariant(self, device: str) -> None:
        """Default run_dij is correlated; for beamlet 0 the remap is the identity.

        The shipped configuration keys on the within-beamlet index;
        ``correlated=False`` is the test instrument. On beamlet 0 the two mappings
        coincide (rw == h), and column 1 must actually move.
        """
        engine = _warp_engine(device)
        kwargs = dict(n_histories_per_beamlet=800, n_batches=4, seed=SEED, truncation=0.0)
        corr = engine.run_dij(_lattice(2, 1), **kwargs)  # default
        ind = engine.run_dij(_lattice(2, 1), correlated=False, **kwargs)
        assert corr.correlated is True
        assert ind.correlated is False
        np.testing.assert_array_equal(corr.column_dense(0), ind.column_dense(0))
        np.testing.assert_array_equal(corr.sigma_dense(0), ind.sigma_dense(0))
        # Non-vacuity: column 1 draws different streams under the remap, so it
        # must actually move — a kernel that ignores the flag would pass above.
        assert not np.array_equal(corr.column_dense(1), ind.column_dense(1))

    def test_scheduling_stays_bit_inert_under_correlation(self, device: str) -> None:
        """Grouping touches neither h nor the within-beamlet index; pin it."""
        results = [
            _warp_engine(device).run_dij(
                _lattice(2, 2),
                n_histories_per_beamlet=800,
                n_batches=4,
                seed=SEED,
                beamlet_group_size=group,
                correlated=True,
            )
            for group in (1, 3, 4)
        ]
        for other in results[1:]:
            np.testing.assert_array_equal(results[0].indptr, other.indptr)
            np.testing.assert_array_equal(results[0].indices, other.indices)
            np.testing.assert_array_equal(results[0].dose, other.dose)
            np.testing.assert_array_equal(results[0].sigma, other.sigma)

    def test_correlated_columns_match_reference_correlated_columns(self, device: str) -> None:
        """Cross-target, per-column chi-squared: the remap keys the same histories."""
        grid = _grid()
        ref_engine = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())
        kwargs = dict(
            n_batches=8,
            seed=SEED,
            transport_electrons=False,
            truncation=0.0,
            correlated=True,
        )
        ref = ref_engine.run_dij(_lattice(2, 1), n_histories_per_beamlet=800, **kwargs)
        result = _warp_engine(device).run_dij(
            _lattice(2, 1), n_histories_per_beamlet=1_600, **kwargs
        )
        # Geometric beamlet footprints on the 2 cm voxels: x 2..8 and 8..14 cm.
        footprints = {0: (slice(1, 4)), 1: (slice(4, 7))}
        for j in range(2):
            ref_dose, ref_sigma = ref.column_dense(j), ref.sigma_dense(j)
            warp_dose, warp_sigma = result.column_dense(j), result.sigma_dense(j)
            mask = np.zeros(ref_dose.shape, dtype=bool)
            mask[footprints[j], 1:7, :] = True
            mask &= (ref_sigma > 0.0) & (warp_sigma > 0.0)
            assert np.count_nonzero(mask) > 50, f"column {j}: mask too small"
            assert_chi2_consistent_batched(
                ref_dose,
                ref_sigma,
                ref.n_batches,
                warp_dose,
                warp_sigma,
                result.n_batches,
                mask=mask,
            )


class TestTruncationAgainstDVH:
    """AGENTS.md 2.8: the truncation threshold is tested on DVH endpoints.

    Truncated and untruncated Dij come from the same seed on the same device, so
    the columns are bit-identical before sparsification and the endpoint shifts
    below are the *deterministic* effect of truncation alone — no statistical
    flake is possible here.
    """

    @staticmethod
    def _dvh_d(dose_in_roi: np.ndarray, volume_percent: float) -> float:
        """Dose received by at least ``volume_percent`` of the ROI volume."""
        return float(np.percentile(dose_in_roi, 100.0 - volume_percent))

    def test_default_truncation_moves_dvh_endpoints_less_than_half_a_percent(
        self, device: str
    ) -> None:
        grid = VoxelGrid.uniform_water(shape=(16, 16, 24), spacing=(1.0, 1.0, 0.5))
        from pyRadMC.backends.warp.engine import WarpEngine

        engine = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device)
        source = _lattice(3, 3)
        # Statistics chosen to populate the sub-threshold tail: Russian roulette
        # thins exactly the photons that seed it, so the non-vacuity guard below
        # needs enough histories for truncation to have real work to do.
        kwargs = dict(n_histories_per_beamlet=24_000, n_batches=4, seed=SEED)
        full = engine.run_dij(source, truncation=0.0, **kwargs)
        cut = engine.run_dij(source, **kwargs)  # default DIJ_TRUNCATION_RELATIVE
        assert cut.truncation == DIJ_TRUNCATION_RELATIVE

        # Non-vacuity: the threshold must actually remove entries, or this test
        # certifies nothing about it.
        assert cut.indices.size < 0.95 * full.indices.size

        weights = np.ones(source.n_beamlets)
        dose_full = full.dose_for_weights(weights)
        dose_cut = cut.dose_for_weights(weights)
        roi = dose_full > 0.1 * dose_full.max()
        assert np.count_nonzero(roi) > 500

        for volume_percent in (2.0, 50.0, 98.0):
            d_full = self._dvh_d(dose_full[roi], volume_percent)
            d_cut = self._dvh_d(dose_cut[roi], volume_percent)
            assert abs(d_cut - d_full) <= 5.0e-3 * d_full, (
                f"D{volume_percent:g} moved by {abs(d_cut - d_full) / d_full:.2e} "
                "relative under default truncation"
            )
        # The tail bias is one-sided: truncation only removes dose.
        assert np.all(dose_cut <= dose_full + 1e-30)
