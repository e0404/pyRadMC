"""A user warp_beamlet_sampler drives the Warp Dij in-kernel (Phase 5, advanced route).

A BeamletSource exposing a ``@wp.func`` ``warp_beamlet_sampler`` is generated on-device:
the engine wraps it into a beamlet generator kernel that bakes the correlated-sampling
history mapping (``h = j*n_per + r``) and the batch-resolved column tag, and calls the
sampler for each primary's position. This pins that the in-kernel Dij route is taken (a
source whose ``emit`` raises still runs), matches the reference Dij column for column,
and closes the energy ledger.
"""

from __future__ import annotations

import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletSource, Primary
from pyRadMC.rng.host import HostRNG, uniform
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from pyRadMC.rng.warp_shim import WarpRNGState  # noqa: E402
from pyRadMC.rng.warp_shim import uniform as wp_uniform  # noqa: E402

ENERGY = 6.0
Z0 = -1.0
N_STRIPS = 3
X_LO, X_SPAN = 2.0, 12.0
Y_LO, Y_SPAN = 2.0, 12.0


@wp.func
def _strip_beamlet_sampler(beamlet: int, within_index: int, state: WarpRNGState):
    """Position for one strip beamlet; mirror of the host emit (x then y)."""
    x_lo = X_LO + X_SPAN * float(beamlet) / float(N_STRIPS)
    x_hi = X_LO + X_SPAN * float(beamlet + 1) / float(N_STRIPS)
    x = x_lo + (x_hi - x_lo) * wp_uniform(state)
    y = Y_LO + Y_SPAN * wp_uniform(state)
    return 6.0, x, y, -1.0, 0.0, 0.0, 1.0, 1.0


class StripWarpBeamletSource(BeamletSource):
    """Strip-lattice beamlet source with an in-kernel warp_beamlet_sampler + host emit."""

    warp_beamlet_sampler = _strip_beamlet_sampler

    @property
    def max_energy(self) -> float:
        return ENERGY

    @property
    def n_beamlets(self) -> int:
        return N_STRIPS

    def emit(self, beamlet: int, rng_state: object) -> Primary:
        x_lo = X_LO + X_SPAN * beamlet / N_STRIPS
        x_hi = X_LO + X_SPAN * (beamlet + 1) / N_STRIPS
        x = x_lo + (x_hi - x_lo) * uniform(rng_state)
        y = Y_LO + Y_SPAN * uniform(rng_state)
        return Primary(ENERGY, x, y, Z0, 0.0, 0.0, 1.0)


class SamplerOnlyBeamletSource(StripWarpBeamletSource):
    """The warp_beamlet_sampler is the only valid path; emit raises to prove it."""

    def emit(self, beamlet: int, rng_state: object) -> Primary:
        raise AssertionError("the Warp Dij must use warp_beamlet_sampler, never emit")


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def test_warp_dij_uses_the_sampler_not_emit() -> None:
    """A source whose emit raises still assembles a Dij on Warp via its sampler."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    dij = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu").run_dij(
        SamplerOnlyBeamletSource(), n_histories_per_beamlet=600, n_batches=3, seed=SEED
    )
    assert dij.n_beamlets == N_STRIPS
    assert all(dij.column_dense(j).sum() > 0.0 for j in range(N_STRIPS))


def test_in_kernel_dij_matches_reference() -> None:
    """In-kernel Dij generation matches the reference column for column."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    source = StripWarpBeamletSource()
    kwargs = dict(n_histories_per_beamlet=2_400, n_batches=8, seed=SEED, transport_electrons=False)
    ref = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run_dij(
        source, **kwargs
    )
    warp = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu").run_dij(
        source, beamlet_group_size=2, **kwargs
    )
    for j in range(N_STRIPS):
        ref_dose, ref_sigma = ref.column_dense(j), ref.sigma_dense(j)
        warp_dose, warp_sigma = warp.column_dense(j), warp.sigma_dense(j)
        mask = ref_dose > 0.1 * ref_dose.max()
        mask &= (ref_sigma > 0.0) & (warp_sigma > 0.0)
        assert_chi2_consistent_batched(
            ref_dose, ref_sigma, ref.n_batches, warp_dose, warp_sigma, warp.n_batches, mask=mask
        )


def test_in_kernel_dij_conserves_energy() -> None:
    """emitted = deposited + escaped for the in-kernel Warp Dij."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    dij = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu").run_dij(
        StripWarpBeamletSource(), n_histories_per_beamlet=600, n_batches=3, seed=SEED
    )
    assert dij.energy_emitted == pytest.approx(dij.energy_deposited + dij.energy_escaped, rel=1e-4)
    assert dij.energy_emitted == pytest.approx(N_STRIPS * 600 * ENERGY, rel=1e-4)


@wp.func
def _half_weight_sampler(beamlet: int, within_index: int, state: WarpRNGState):
    """The strip sampler at statistical weight 0.5 (weighted-Dij contract pin)."""
    x_lo = X_LO + X_SPAN * float(beamlet) / float(N_STRIPS)
    x_hi = X_LO + X_SPAN * float(beamlet + 1) / float(N_STRIPS)
    x = x_lo + (x_hi - x_lo) * wp_uniform(state)
    y = Y_LO + Y_SPAN * wp_uniform(state)
    return 6.0, x, y, -1.0, 0.0, 0.0, 1.0, 0.5


class HalfWeightWarpBeamletSource(StripWarpBeamletSource):
    """The strip source emitting at weight 0.5 on both routes."""

    warp_beamlet_sampler = _half_weight_sampler

    def emit(self, beamlet: int, rng_state: object) -> Primary:
        return super().emit(beamlet, rng_state)._replace(weight=0.5)


def test_in_kernel_dij_books_the_sampler_weight() -> None:
    """The wrapped kernel books emitted weight*energy and transports the weight."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    dij = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu").run_dij(
        HalfWeightWarpBeamletSource(), n_histories_per_beamlet=600, n_batches=3, seed=SEED
    )
    assert dij.energy_emitted == pytest.approx(0.5 * N_STRIPS * 600 * ENERGY, rel=1e-9)
    assert dij.energy_emitted == pytest.approx(dij.energy_deposited + dij.energy_escaped, rel=1e-4)
