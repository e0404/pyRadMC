"""Warp-native forward transport for a spectral beam through a transmission mask."""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.collimation import TransmissionMaskSource
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import SpectralBeamSource
from pyRadMC.geometry.spectrum import Spectrum
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _source() -> TransmissionMaskSource:
    inner = SpectralBeamSource(
        spectrum=Spectrum((1.0, 2.0, 4.0), (1.0, 2.0)),
        focal_point=(8.0, 8.0, -50.0),
        center=(8.0, 8.0, 0.0),
        width_u=10.0,
        width_v=10.0,
    )
    mask = np.ones((10, 10))
    mask[:5, :] = 0.0
    return TransmissionMaskSource(
        inner,
        mask=mask,
        plane_center=(8.0, 8.0, 0.0),
        width_u=10.0,
        width_v=10.0,
    )


def test_spectral_mask_is_generated_on_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The recognized composition must never call host ``sample_batch``."""
    from pyRadMC.backends.warp.engine import WarpEngine

    def fail_host_sampling(*args, **kwargs):
        raise AssertionError("spectral transmission mask fell back to host sampling")

    monkeypatch.setattr(TransmissionMaskSource, "sample_batch", fail_host_sampling)
    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    result = WarpEngine(grid=grid, cross_sections=xs, device="cpu", chunk_size=2048).run(
        _source(),
        n_histories=4_000,
        n_batches=4,
        seed=SEED,
        transport_electrons=False,
    )
    assert result.dose.sum() > 0.0
    assert result.energy_emitted == pytest.approx(
        result.energy_deposited + result.energy_escaped, rel=1.0e-4
    )


def test_device_mask_agrees_with_reference_statistically() -> None:
    """The device mask interpolation and source sampling preserve the reference dose."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = _source()
    ref = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
        source,
        n_histories=6_000,
        n_batches=12,
        seed=SEED,
        transport_electrons=False,
    )
    warp = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        source,
        n_histories=24_000,
        n_batches=12,
        seed=SEED,
        transport_electrons=False,
    )
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        warp.dose,
        warp.dose_sigma,
        warp.n_batches,
        mask=mask,
    )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)],
)
def test_device_mask_is_chunk_size_bitwise_inert(device: str) -> None:
    """Global history keys make mask compaction and chunking scheduling-only."""
    from pyRadMC.backends.warp.engine import WarpEngine

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    runs = [
        WarpEngine(grid=grid, cross_sections=xs, device=device, chunk_size=chunk).run(
            _source(),
            n_histories=8_000,
            n_batches=4,
            seed=SEED,
            transport_electrons=False,
        )
        for chunk in (512, 8_000)
    ]
    np.testing.assert_array_equal(runs[0].dose, runs[1].dose)
    np.testing.assert_array_equal(runs[0].dose_sigma, runs[1].dose_sigma)
    assert runs[0].energy_emitted == runs[1].energy_emitted
