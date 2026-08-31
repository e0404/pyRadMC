"""A user warp_sampler runs in-kernel.

A source that exposes a ``@wp.func`` ``warp_sampler`` is generated on-device: the engine
wraps the sampler into a generator kernel (``make_generator_kernel``) instead of
host-sampling. This pins that the in-kernel route is taken (a source whose ``emit``
raises still runs, proving the GPU never falls back to pre-sampling), agrees with the
reference transport of the equivalent host ``emit``, and keeps the energy ledger closed.
"""

from __future__ import annotations

import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import Primary, Source
from pyradmc.rng.host import HostRNG, uniform
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from pyradmc.rng.warp_shim import WarpRNGState  # noqa: E402
from pyradmc.rng.warp_shim import uniform as wp_uniform  # noqa: E402
from pyradmc.transport.particles import PHOTON  # noqa: E402


@wp.func
def _wobble_sampler(history_index: int, state: WarpRNGState):
    """6 MeV photon, uniform over a 10 cm square field, +z; mirror of the host emit."""
    x = 3.0 + 10.0 * wp_uniform(state)
    y = 3.0 + 10.0 * wp_uniform(state)
    return int(PHOTON), 6.0, x, y, -1.0, 0.0, 0.0, 1.0, 1.0


class WobbleWarpSource(Source):
    """Open-field source with both a host emit and a matching in-kernel warp_sampler."""

    energy = 6.0
    warp_sampler = _wobble_sampler

    @property
    def max_energy(self) -> float:
        return self.energy

    def emit(self, rng_state: object) -> Primary:
        x = 3.0 + 10.0 * uniform(rng_state)
        y = 3.0 + 10.0 * uniform(rng_state)
        return Primary(self.energy, x, y, -1.0, 0.0, 0.0, 1.0)


class SamplerOnlySource(WobbleWarpSource):
    """The warp_sampler is the only valid path; emit raises to prove it is not used."""

    def emit(self, rng_state: object) -> Primary:
        raise AssertionError("the warp route must use warp_sampler, never emit")


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def test_warp_uses_the_sampler_not_emit() -> None:
    """A source whose emit raises still transports on Warp via its warp_sampler."""
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        SamplerOnlySource(), n_histories=8_000, n_batches=8, seed=SEED
    )
    assert result.dose.sum() > 0.0


def test_in_kernel_source_agrees_with_reference() -> None:
    """In-kernel generation matches the reference transport of the equivalent emit."""
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = WobbleWarpSource()

    ref = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
        source, n_histories=6_000, n_batches=12, seed=SEED, transport_electrons=False
    )
    warp = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        source, n_histories=24_000, n_batches=12, seed=SEED, transport_electrons=False
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


def test_in_kernel_energy_ledger_closes() -> None:
    """emitted = deposited + escaped on Warp for the in-kernel source."""
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        WobbleWarpSource(), n_histories=8_000, n_batches=8, seed=SEED
    )
    assert result.energy_emitted == pytest.approx(
        result.energy_deposited + result.energy_escaped, rel=1e-4
    )
