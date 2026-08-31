"""The planar Gaussian-spot source transports consistently on both backends.

The reference engine runs ``emit`` (per-history streams); Warp takes the
vectorized pre-sampling batch (its own PCG64 stream) — independent draws, so the
comparison is the batched chi-squared, never bitwise (spectral-source precedent).
"""

from __future__ import annotations

import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import GaussianSpotBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

pytestmark = pytest.mark.warp


def _source() -> GaussianSpotBeamSource:
    return GaussianSpotBeamSource(
        spectrum=Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0)),
        focal_point=(8.0, 8.0, -80.0),
        center=(8.0, 8.0, -2.0),
        width_u=6.0,
        width_v=6.0,
        sigma_u=0.2,
        sigma_v=0.2,
    )


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def test_gaussian_spot_agrees_across_backends() -> None:
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    kwargs = dict(n_batches=8, seed=SEED, transport_electrons=False)
    ref = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
        _source(), n_histories=6_000, **kwargs
    )
    warp_result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        _source(), n_histories=24_000, **kwargs
    )
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp_result.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        warp_result.dose,
        warp_result.dose_sigma,
        warp_result.n_batches,
        mask=mask,
    )


def test_ledgers_close_on_both_backends() -> None:
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    ref = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
        _source(), n_histories=500, n_batches=2, seed=SEED
    )
    assert ref.energy_emitted == pytest.approx(ref.energy_deposited + ref.energy_escaped, rel=1e-9)
    warp_result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        _source(), n_histories=2_000, n_batches=2, seed=SEED
    )
    assert warp_result.energy_emitted == pytest.approx(
        warp_result.energy_deposited + warp_result.energy_escaped, rel=1e-4
    )
