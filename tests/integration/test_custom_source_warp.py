"""A user Source runs on Warp via the pre-sampling route.

A custom :class:`~pyRadMC.geometry.source.Source` with no ``warp_sampler`` reaches the
GPU through the general pre-sampling path (``sample_batch`` -> ``generate_from_upload``),
the same route the phase-space source uses. Because the default ``sample_batch`` draws
the same per-history stream as ``emit``, the reference and Warp backends transport the
*identical* primaries, so they agree by the cross-backend chi-squared oracle (statistical,
never bit-wise; AGENTS.md 2.3) and the Warp energy ledger closes.
"""

from __future__ import annotations

import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import Primary, Source
from pyRadMC.rng.host import HostRNG, uniform
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp


class WobbleBeamSource(Source):
    """A 6 MeV photon beam, uniform over a square field, entering along +z.

    A user-space re-creation of a broad beam through the public interface, with no
    warp_sampler, so both backends must reach it through pre-sampling. Draws two
    uniforms per history (like the built-in parallel beam) so the emit stream is
    non-trivial.
    """

    energy = 6.0

    @property
    def max_energy(self) -> float:
        return self.energy

    def emit(self, rng_state: object) -> Primary:
        x = 3.0 + 10.0 * uniform(rng_state)
        y = 3.0 + 10.0 * uniform(rng_state)
        return Primary(self.energy, x, y, -1.0, 0.0, 0.0, 1.0)


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def test_custom_source_agrees_across_backends() -> None:
    """Ref and Warp transport of the pre-sampled custom source are consistent."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = WobbleBeamSource()

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


def test_warp_energy_ledger_closes_for_the_custom_source() -> None:
    """emitted = deposited + escaped on Warp for the pre-sampled source."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        WobbleBeamSource(), n_histories=8_000, n_batches=8, seed=SEED
    )
    assert result.energy_emitted == pytest.approx(
        result.energy_deposited + result.energy_escaped, rel=1e-4
    )
