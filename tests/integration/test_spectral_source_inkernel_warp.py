"""The spectral open-field source generates in-kernel on the Warp backend.

:class:`~pyradmc.geometry.source.SpectralBeamSource` was measured host-bound on a
CT-grade workload (2026-07-19): per-chunk ``sample_batch`` cost ~42 ms per 262k
histories against ~10 ms of GPU transport, leaving the device idle. The Warp engine
therefore routes the exact spectral source type to a built-in generator kernel —
CDF inversion over the uploaded spectrum tables plus the divergent-fan geometry,
drawing four uniforms from the per-history slot stream — instead of host pre-sampling.

These tests pin: the route is taken (host sampling never runs), cross-backend
statistical agreement, the polyenergetic energy ledger, the sampled mean energy
against the spectrum's analytic mean, chunk-size bit-invariance (streams are keyed
by global history index), and the fallback — a *subclass* keeps the pre-sampling
route, since the engine cannot know what an override changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum, ali_rogers_mv
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
CENTER = (8.0, 8.0, 0.0)


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _source(spectrum: Spectrum = SPECTRUM) -> SpectralBeamSource:
    return SpectralBeamSource(
        spectrum=spectrum, focal_point=FOCAL, center=CENTER, width_u=6.0, width_v=6.0
    )


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _engine(grid: VoxelGrid, device: str):
    from pyradmc.backends.warp.engine import WarpEngine

    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device)


def test_warp_never_host_samples_the_spectral_source(device: str) -> None:
    """The exact spectral type transports even when host sampling is off-limits."""
    source = _source()
    source.sample_batch = None  # type: ignore[method-assign] - any host call would raise
    result = _engine(_grid(), device).run(source, n_histories=8_000, n_batches=8, seed=SEED)
    assert result.dose.sum() > 0.0


def test_inkernel_spectral_agrees_with_reference(device: str) -> None:
    """In-kernel generation matches the reference transport statistically."""
    grid = _grid()
    source = _source()
    ref = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run(
        source, n_histories=6_000, n_batches=12, seed=SEED, transport_electrons=False
    )
    warp_res = _engine(grid, device).run(
        source, n_histories=24_000, n_batches=12, seed=SEED, transport_electrons=False
    )
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp_res.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        warp_res.dose,
        warp_res.dose_sigma,
        warp_res.n_batches,
        mask=mask,
    )


def test_inkernel_energy_ledger_closes(device: str) -> None:
    """emitted == deposited + escaped, with emitted booked in-kernel per primary."""
    result = _engine(_grid(), device).run(_source(), n_histories=8_000, n_batches=8, seed=SEED)
    assert result.energy_emitted == pytest.approx(
        result.energy_deposited + result.energy_escaped, rel=ENERGY_BALANCE_RTOL
    )


def test_inkernel_mean_energy_matches_spectrum(device: str) -> None:
    """The booked emitted energy pins the CDF inversion against the analytic mean.

    A fine (100-bin) clinical spectrum exercises the in-kernel binary search well
    beyond the 4-bin fixture. 24k histories put the standard error of the mean near
    sigma_E / sqrt(n) ~ 0.008 MeV; the 0.05 MeV gate is ~6 sigma, loose enough to
    never flake at the pinned seed and tight enough to catch an off-by-one bin.
    """
    spectrum = ali_rogers_mv("varian-6mv")
    n = 24_000
    result = _engine(_grid(), device).run(_source(spectrum), n_histories=n, n_batches=8, seed=SEED)
    assert result.energy_emitted / n == pytest.approx(spectrum.mean_energy, abs=0.05)


def test_chunk_size_is_bitwise_inert(device: str) -> None:
    """Streams are keyed by global history index, so chunking cannot move a draw."""
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    doses = []
    for chunk in (512, 8_000):
        engine = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device, chunk_size=chunk)
        doses.append(engine.run(_source(), n_histories=8_000, n_batches=4, seed=SEED).dose)
    np.testing.assert_array_equal(doses[0], doses[1])


def test_subclass_keeps_the_presampling_route(device: str) -> None:
    """A subclass may override emit/sample_batch; the engine must not bypass it."""

    class ShiftedSpectral(SpectralBeamSource):
        sampled = False

        def sample_batch(self, seed: int, history_offset: int, n: int):
            type(self).sampled = True
            return super().sample_batch(seed, history_offset, n)

    source = ShiftedSpectral(
        spectrum=SPECTRUM, focal_point=FOCAL, center=CENTER, width_u=6.0, width_v=6.0
    )
    result = _engine(_grid(), device).run(source, n_histories=4_000, n_batches=4, seed=SEED)
    assert result.dose.sum() > 0.0
    assert ShiftedSpectral.sampled, "subclass must transport via its own sample_batch"
