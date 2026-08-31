"""Concurrent statistical-batch lanes for ``WarpEngine.run``.

Each CUDA lane owns queues, RNG slots, and a fixed-point dose map. Transport may
finish out of order, but batch dose maps are folded into the float64 sums in batch
order, keeping ``concurrent_batches`` a bit-inert scheduling knob. CPU has no CUDA
streams and deliberately collapses to the sequential path.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from pyradmc.backends.warp.engine import WarpEngine  # noqa: E402
from pyradmc.data.analytic import AnalyticCrossSections  # noqa: E402
from pyradmc.geometry.collimation import TransmissionMaskSource  # noqa: E402
from pyradmc.geometry.grid import VoxelGrid  # noqa: E402
from pyradmc.geometry.source import (  # noqa: E402
    ParallelBeamSource,
    Primary,
    Source,
    SpectralBeamSource,
)
from pyradmc.geometry.spectrum import Spectrum  # noqa: E402
from pyradmc.rng.host import uniform  # noqa: E402
from tests.conftest import SEED  # noqa: E402


def _engine(device: str) -> WarpEngine:
    grid = VoxelGrid.uniform_water(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device=device, chunk_size=2048)


def _source() -> TransmissionMaskSource:
    inner = SpectralBeamSource(
        Spectrum((1.0, 2.0, 4.0), (1.0, 2.0)),
        focal_point=(6.0, 6.0, -50.0),
        center=(6.0, 6.0, 0.0),
        width_u=8.0,
        width_v=8.0,
    )
    mask = np.ones((8, 8))
    mask[:3, :] = 0.0
    return TransmissionMaskSource(
        inner,
        mask=mask,
        plane_center=(6.0, 6.0, 0.0),
        width_u=8.0,
        width_v=8.0,
    )


class HostSampledSource(Source):
    """Small custom source that exercises the host pre-sampling lane route."""

    @property
    def max_energy(self) -> float:
        return 3.0

    def emit(self, rng_state: object) -> Primary:
        return Primary(
            3.0,
            3.0 + 6.0 * uniform(rng_state),
            3.0 + 6.0 * uniform(rng_state),
            -1.0,
            0.0,
            0.0,
            1.0,
        )


def _run(device: str, lanes: int):
    return _engine(device).run(
        _source(),
        n_histories=6_000,
        n_batches=6,
        seed=SEED,
        transport_electrons=False,
        concurrent_batches=lanes,
    )


def _assert_bit_identical(got, want) -> None:
    np.testing.assert_array_equal(got.dose, want.dose)
    np.testing.assert_array_equal(got.dose_sigma, want.dose_sigma)
    assert got.energy_deposited == want.energy_deposited
    assert got.energy_escaped == want.energy_escaped
    assert got.energy_unscored == want.energy_unscored
    assert got.energy_emitted == want.energy_emitted


@pytest.mark.gpu
@pytest.mark.parametrize("lanes", [2, 3])
def test_cuda_lanes_are_bit_identical_to_sequential(lanes: int) -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    _assert_bit_identical(_run("cuda:0", lanes), _run("cuda:0", 1))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "source",
    [
        ParallelBeamSource(3.0, -1.0, (3.0, 9.0), (3.0, 9.0)),
        HostSampledSource(),
    ],
    ids=["builtin-generator", "host-presampled"],
)
def test_other_forward_routes_are_bit_identical(source: Source) -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    def run(lanes: int):
        return _engine("cuda:0").run(
            source,
            n_histories=6_000,
            n_batches=6,
            seed=SEED,
            transport_electrons=False,
            concurrent_batches=lanes,
        )

    _assert_bit_identical(run(2), run(1))


def test_cpu_falls_back_to_the_sequential_path() -> None:
    _assert_bit_identical(_run("cpu", 4), _run("cpu", 1))


def test_nonpositive_lane_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="concurrent_batches"):
        _run("cpu", 0)
