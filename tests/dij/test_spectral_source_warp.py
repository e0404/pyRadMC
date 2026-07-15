"""The spectral beamlet source drives the Warp Dij via pre-sampling (Phase 5).

:class:`~pyRadMC.geometry.source.SpectralBeamletSource` carries no
``warp_beamlet_sampler``, so it reaches the device Dij through the simple route:
per beamlet, the default ``sample_beamlet_batch`` host-samples energy, position and
direction with the correlated-sampling history key and the columns are uploaded
(``generate_from_upload``). These tests pin the cross-backend agreement (per-column
chi-squared against the reference, on every available device) and the exactness of
the polyenergetic energy ledger on the Warp side.
"""

from __future__ import annotations

import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import SpectralBeamletSource
from pyRadMC.geometry.spectrum import Spectrum
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

ENERGY_BALANCE_RTOL = 1.0e-4  # float32 transport + 1e-9 MeV scoring quanta

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -80.0)
CENTERS = ((5.0, 8.0, 0.0), (8.0, 8.0, 0.0), (11.0, 8.0, 0.0))


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _source() -> SpectralBeamletSource:
    return SpectralBeamletSource(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        centers=CENTERS,
        width_u=3.0,
        width_v=3.0,
    )


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def test_spectral_dij_matches_reference(device: str) -> None:
    """Each pre-sampled Warp column is statistically consistent with the reference."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    source = _source()
    kwargs = dict(n_histories_per_beamlet=2_400, n_batches=8, seed=SEED, transport_electrons=False)

    ref = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run_dij(
        source, **kwargs
    )
    warp = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device).run_dij(
        source, beamlet_group_size=2, **kwargs
    )
    for j in range(source.n_beamlets):
        ref_dose, ref_sigma = ref.column_dense(j), ref.sigma_dense(j)
        warp_dose, warp_sigma = warp.column_dense(j), warp.sigma_dense(j)
        mask = ref_dose > 0.1 * ref_dose.max()
        mask &= (ref_sigma > 0.0) & (warp_sigma > 0.0)
        assert_chi2_consistent_batched(
            ref_dose, ref_sigma, ref.n_batches, warp_dose, warp_sigma, warp.n_batches, mask=mask
        )


def test_spectral_dij_ledger_closes_on_warp(device: str) -> None:
    """emitted == deposited + escaped for the polyenergetic pre-sampled Dij."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    dij = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device).run_dij(
        _source(), n_histories_per_beamlet=600, n_batches=3, seed=SEED
    )
    assert dij.energy_emitted == pytest.approx(
        dij.energy_deposited + dij.energy_escaped, rel=ENERGY_BALANCE_RTOL
    )
    n = 3 * 600
    assert n * 0.5 < dij.energy_emitted < n * 6.0
