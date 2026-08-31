"""Composite sources transport on both backends via pre-sampling (VSM use case).

A ``CompositeSource`` / ``CompositeBeamletSource`` carries no in-kernel generator, so it
reaches the GPU through the general pre-sampling route — no engine change. These pin
that a mixture of built-in sources (open field and beamlet/Dij) agrees with the
reference transport, so a virtual source model assembled from simple components produces
consistent dose and beamlet-resolved dose across backends.
"""

from __future__ import annotations

import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import (
    BeamletGridSource,
    CompositeBeamletSource,
    CompositeSource,
    ParallelBeamSource,
)
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def test_composite_open_field_agrees_across_backends() -> None:
    """A wide+narrow parallel-field mixture: ref and Warp (pre-sampled) agree."""
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    source = CompositeSource(
        [
            (ParallelBeamSource(6.0, -1.0, (3.0, 13.0), (3.0, 13.0)), 0.7),
            (ParallelBeamSource(6.0, -1.0, (6.0, 10.0), (6.0, 10.0)), 0.3),
        ]
    )
    ref = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run(
        source, n_histories=6_000, n_batches=12, seed=SEED, transport_electrons=False
    )
    warp = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu").run(
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


def test_composite_beamlet_dij_agrees_across_backends() -> None:
    """A per-beamlet mixture of two lattices: ref and Warp Dij agree per column."""
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    source = CompositeBeamletSource(
        [
            (BeamletGridSource(6.0, -1.0, (2.0, 14.0), (2.0, 14.0), 2, 2), 0.7),
            (BeamletGridSource(6.0, -1.0, (3.0, 13.0), (3.0, 13.0), 2, 2), 0.3),
        ]
    )
    kwargs = dict(n_histories_per_beamlet=2_400, n_batches=8, seed=SEED, transport_electrons=False)
    ref = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run_dij(
        source, **kwargs
    )
    warp = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu").run_dij(
        source, beamlet_group_size=2, **kwargs
    )
    assert ref.energy_emitted == pytest.approx(ref.energy_deposited + ref.energy_escaped, rel=1e-9)
    for j in range(source.n_beamlets):
        ref_dose, ref_sigma = ref.column_dense(j), ref.sigma_dense(j)
        warp_dose, warp_sigma = warp.column_dense(j), warp.sigma_dense(j)
        mask = ref_dose > 0.1 * ref_dose.max()
        mask &= (ref_sigma > 0.0) & (warp_sigma > 0.0)
        assert_chi2_consistent_batched(
            ref_dose, ref_sigma, ref.n_batches, warp_dose, warp_sigma, warp.n_batches, mask=mask
        )
