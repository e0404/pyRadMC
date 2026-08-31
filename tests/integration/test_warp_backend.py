"""Warp backend against the reference oracle, on every available device.

The comparisons follow AGENTS.md 2.3 and 4 exactly: *statistical* equivalence via the
chi-squared detection oracle over the high-dose region, never bit equality across
targets. Within one device, one seed, bit reproducibility is exact and asserted —
the Warp backend scores in fixed-point int64, so even atomic scheduling cannot
reorder its sums.

Energy conservation for this backend is float32 transport arithmetic plus int64
quantization (1e-9 MeV per quantum), so the balance is asserted at 1e-4 relative —
documented in :mod:`pyradmc.backends.warp.engine`, not a tunable.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

ENERGY_BALANCE_RTOL = 1.0e-4  # float32 transport + 1e-9 MeV scoring quanta


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 30), spacing=(2.0, 2.0, 0.2))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _source() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0))


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


@pytest.fixture(scope="module")
def ref_results() -> dict:
    """Reference runs, shared across devices: KERMA and full coupled transport."""
    grid = _grid()
    engine = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())
    return {
        "kerma": engine.run(
            _source(), n_histories=6_000, n_batches=12, seed=SEED, transport_electrons=False
        ),
        "full": engine.run(
            _source(), n_histories=4_000, n_batches=8, seed=SEED, transport_electrons=True
        ),
    }


def _warp_engine(device: str):
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device)


def _mask(ref, warp_result) -> np.ndarray:
    """High-dose region where both estimates carry a defined uncertainty."""
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp_result.dose_sigma > 0.0)
    assert np.count_nonzero(mask) > 300, "mask too small to detect anything"
    return mask


class TestStatisticalEquivalenceWithReference:
    """The chi-squared detection oracle (AGENTS.md 4), per transport mode."""

    def test_kerma_dose_matches_ref(self, device: str, ref_results: dict) -> None:
        result = _warp_engine(device).run(
            _source(), n_histories=24_000, n_batches=12, seed=SEED, transport_electrons=False
        )
        ref = ref_results["kerma"]
        assert_chi2_consistent_batched(
            ref.dose,
            ref.dose_sigma,
            ref.n_batches,
            result.dose,
            result.dose_sigma,
            result.n_batches,
            mask=_mask(ref, result),
        )

    def test_full_transport_dose_matches_ref(self, device: str, ref_results: dict) -> None:
        result = _warp_engine(device).run(
            _source(), n_histories=16_000, n_batches=8, seed=SEED, transport_electrons=True
        )
        ref = ref_results["full"]
        assert_chi2_consistent_batched(
            ref.dose,
            ref.dose_sigma,
            ref.n_batches,
            result.dose,
            result.dose_sigma,
            result.n_batches,
            mask=_mask(ref, result),
        )

    def test_electron_beam_dose_matches_ref(self, device: str) -> None:
        """Electron primaries: vacuum entry, condensed history, positron-free."""
        grid = VoxelGrid.uniform_water(shape=(8, 8, 40), spacing=(2.0, 2.0, 0.1))
        source = ParallelBeamSource(energy=5.0, z=-0.5, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
        ref_engine = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())
        ref = ref_engine.run(
            source, n_histories=1_200, n_batches=8, seed=SEED, primary_kind="electron"
        )

        from pyradmc.backends.warp.engine import WarpEngine

        engine = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device)
        result = engine.run(
            source, n_histories=4_800, n_batches=8, seed=SEED, primary_kind="electron"
        )
        assert_chi2_consistent_batched(
            ref.dose,
            ref.dose_sigma,
            ref.n_batches,
            result.dose,
            result.dose_sigma,
            result.n_batches,
            mask=_mask(ref, result),
        )


class TestDeterminism:
    def test_bit_reproducible_within_device(self, device: str) -> None:
        """Same device, same seed: bit-identical dose, sigma, and energy books.

        Fixed-point scoring makes the accumulation order irrelevant, so this holds
        on CUDA despite atomic scheduling. This is the *within*-target guarantee of
        AGENTS.md 2.3; nothing here compares across targets.
        """
        runs = [
            _warp_engine(device).run(
                _source(), n_histories=4_000, n_batches=4, seed=SEED, transport_electrons=True
            )
            for _ in range(2)
        ]
        assert np.array_equal(runs[0].dose, runs[1].dose)
        assert np.array_equal(runs[0].dose_sigma, runs[1].dose_sigma)
        assert runs[0].energy_deposited == runs[1].energy_deposited
        assert runs[0].energy_escaped == runs[1].energy_escaped

    def test_chunking_does_not_change_the_result(self, device: str) -> None:
        """Streams are per history and scoring is order-free, so chunk size is inert."""
        from pyradmc.backends.warp.engine import WarpEngine

        grid = _grid()
        results = [
            WarpEngine(grid=grid, cross_sections=_xs(grid), device=device, chunk_size=chunk).run(
                _source(), n_histories=2_000, n_batches=4, seed=SEED
            )
            for chunk in (500, 2_000)
        ]
        assert np.array_equal(results[0].dose, results[1].dose)


class TestEnergyConservation:
    @pytest.mark.parametrize("transport_electrons", [False, True])
    def test_emitted_equals_deposited_plus_escaped(
        self, device: str, transport_electrons: bool
    ) -> None:
        result = _warp_engine(device).run(
            _source(),
            n_histories=4_000,
            n_batches=4,
            seed=SEED,
            transport_electrons=transport_electrons,
        )
        balance = result.energy_deposited + result.energy_escaped
        assert balance == pytest.approx(result.energy_emitted, rel=ENERGY_BALANCE_RTOL)

    def test_sigma_is_scored(self, device: str) -> None:
        """Dose-squared scoring is required infrastructure (AGENTS.md 2.4)."""
        result = _warp_engine(device).run(_source(), n_histories=4_000, n_batches=4, seed=SEED)
        high = result.dose > 0.5 * result.dose.max()
        assert np.all(result.dose_sigma[high] > 0.0)
