"""The spectral beamlet source generates in-kernel on the Warp Dij.

:class:`~pyradmc.geometry.source.SpectralBeamletSource` reaches the device Dij
through a built-in generator kernel — the same block mapping as the beamlet
lattice (group-local column tag, correlated key ``r``) with the spectrum CDF
inversion and the divergent-fan geometry inline, its four uniforms drawn from the
per-history slot stream. The host pre-sampling route was measured wall-dominant on
CT-grade Dij runs (2026-07-19). These tests pin: the route is taken (host sampling
never runs), cross-backend agreement (per-column chi-squared against the reference,
on every available device), the polyenergetic energy ledger, the bit-inertness of
``beamlet_group_size`` on the new route, and the fallback — a *subclass* keeps the
pre-sampling route, since the engine cannot know what an override changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import SpectralBeamletSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
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
    """Each in-kernel Warp column is statistically consistent with the reference."""
    from pyradmc.backends.warp.engine import WarpEngine

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
    """emitted == deposited + escaped for the polyenergetic in-kernel Dij."""
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    dij = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device).run_dij(
        _source(), n_histories_per_beamlet=600, n_batches=3, seed=SEED
    )
    assert dij.energy_emitted == pytest.approx(
        dij.energy_deposited + dij.energy_escaped, rel=ENERGY_BALANCE_RTOL
    )
    n = 3 * 600
    assert n * 0.5 < dij.energy_emitted < n * 6.0


def test_warp_never_host_samples_the_spectral_beamlets(device: str) -> None:
    """The exact spectral beamlet type transports with host sampling off-limits."""
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    source = _source()
    source.sample_beamlet_batch = None  # type: ignore[method-assign] - any host call would raise
    dij = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device).run_dij(
        source, n_histories_per_beamlet=600, n_batches=3, seed=SEED
    )
    assert dij.column_dense(1).sum() > 0.0


def test_beamlet_group_size_is_bitwise_inert_on_the_spectral_route(device: str) -> None:
    """Grouping is scheduling: the in-kernel spectral Dij is identical for any group."""
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    columns = []
    for group in (1, 3):
        dij = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device).run_dij(
            _source(),
            n_histories_per_beamlet=800,
            n_batches=4,
            seed=SEED,
            beamlet_group_size=group,
        )
        columns.append(np.stack([dij.column_dense(j) for j in range(dij.n_beamlets)]))
    np.testing.assert_array_equal(columns[0], columns[1])


def test_subclass_keeps_the_presampling_dij_route(device: str) -> None:
    """A subclass may override emit/sample_beamlet_batch; it must not be bypassed."""
    from pyradmc.backends.warp.engine import WarpEngine

    class ShiftedBeamlets(SpectralBeamletSource):
        sampled = False

        def sample_beamlet_batch(self, seed: int, history_offset: int, n: int, beamlet: int):
            type(self).sampled = True
            return super().sample_beamlet_batch(seed, history_offset, n, beamlet)

    grid = _grid()
    source = ShiftedBeamlets(
        spectrum=SPECTRUM, focal_point=FOCAL, centers=CENTERS, width_u=3.0, width_v=3.0
    )
    dij = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device).run_dij(
        source, n_histories_per_beamlet=400, n_batches=2, seed=SEED
    )
    assert dij.column_dense(0).sum() > 0.0
    assert ShiftedBeamlets.sampled, "subclass must transport via its own sample_beamlet_batch"
