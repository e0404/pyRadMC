"""A user BeamletSource drives the Warp Dij via pre-sampling.

A custom :class:`~pyRadMC.geometry.source.BeamletSource` with no ``warp_beamlet_sampler``
reaches the GPU Dij through host pre-sampling: per beamlet, ``sample_beamlet_batch``
draws the primaries and they are uploaded with the correlated-sampling history key
(``r`` correlated, ``h = j*n_per + r`` independent) and the batch-resolved beamlet tag,
so the pre-sampled Dij matches the reference column-for-column (statistically) and keeps
the same scheduling invariances (group-size independence, reproducibility) and energy
ledger as the built-in lattice.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletSource, Primary
from pyRadMC.rng.host import HostRNG, uniform
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

ENERGY = 6.0
Z0 = -1.0
FIELD_X = (2.0, 14.0)
FIELD_Y = (2.0, 14.0)


class StripBeamletSource(BeamletSource):
    """A user source: ``n`` vertical strips tiling a field, each a parallel sub-beam."""

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def max_energy(self) -> float:
        return ENERGY

    @property
    def n_beamlets(self) -> int:
        return self._n

    def emit(self, beamlet: int, rng_state: object) -> Primary:
        x_lo = FIELD_X[0] + (FIELD_X[1] - FIELD_X[0]) * beamlet / self._n
        x_hi = FIELD_X[0] + (FIELD_X[1] - FIELD_X[0]) * (beamlet + 1) / self._n
        x = x_lo + (x_hi - x_lo) * uniform(rng_state)
        y = FIELD_Y[0] + (FIELD_Y[1] - FIELD_Y[0]) * uniform(rng_state)
        return Primary(ENERGY, x, y, Z0, 0.0, 0.0, 1.0)


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def test_presampled_dij_matches_reference() -> None:
    """Each column of the pre-sampled Warp Dij is consistent with the reference."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    source = StripBeamletSource(3)
    kwargs = dict(n_histories_per_beamlet=2_400, n_batches=8, seed=SEED, transport_electrons=False)

    ref = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run_dij(
        source, **kwargs
    )
    warp = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu").run_dij(
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


def test_presampled_dij_is_group_size_invariant() -> None:
    """The pre-sampled Dij is bit-identical across beamlet group sizes."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    source = StripBeamletSource(4)
    engine = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu")
    kwargs = dict(n_histories_per_beamlet=800, n_batches=4, seed=SEED, transport_electrons=False)
    a = engine.run_dij(source, beamlet_group_size=1, **kwargs)
    b = engine.run_dij(source, beamlet_group_size=4, **kwargs)
    for j in range(source.n_beamlets):
        np.testing.assert_array_equal(a.column_dense(j), b.column_dense(j))


def test_presampled_dij_conserves_energy() -> None:
    """emitted = deposited + escaped for the pre-sampled Warp Dij."""
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    dij = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu").run_dij(
        StripBeamletSource(3), n_histories_per_beamlet=600, n_batches=3, seed=SEED
    )
    assert dij.energy_emitted == pytest.approx(dij.energy_deposited + dij.energy_escaped, rel=1e-4)
    assert dij.energy_emitted == pytest.approx(3 * 600 * ENERGY, rel=1e-4)
