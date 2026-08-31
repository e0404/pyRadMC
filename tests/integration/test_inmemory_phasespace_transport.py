"""An in-memory phase space transports with exact books on both backends.

A handcrafted mixed-particle population (photon + electron + positron, non-unit
weights) pins the weighted emitted-energy ledger — including the positron's
``2 m_e c^2`` annihilation book — and a photon-only population pins ref-vs-warp
chi-squared through the pre-sampling route. Oversampling the tiny population is
the point here, so the latent-variance caveat is acknowledged per test.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.phasespace import InMemoryPhaseSpaceSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _mixed_source() -> InMemoryPhaseSpaceSource:
    """One of each supported kind, distinct energies and weights."""
    return InMemoryPhaseSpaceSource(
        particle_type=np.array([1, 2, 3], dtype=np.int32),
        energy=np.array([4.0, 2.0, 1.5]),
        x=np.array([8.0, 8.0, 8.0]),
        y=np.array([8.0, 8.0, 8.0]),
        z=np.array([-1.0, -1.0, -1.0]),
        ux=np.zeros(3),
        uy=np.zeros(3),
        uz=np.ones(3),
        weight=np.array([1.0, 0.5, 0.25]),
    )


@pytest.mark.filterwarnings("ignore:.*latent.*")
def test_weighted_mixed_ledger_closes_exactly() -> None:
    """emitted = sum over drawn records of w * (E + positron rest-mass book)."""
    source = _mixed_source()
    engine = ReferenceEngine(grid=_grid(), cross_sections=_xs(_grid()), rng=HostRNG())
    result = engine.run(source, n_histories=300, n_batches=3, seed=SEED)

    # Reconstruct the expected emitted book from the very same per-history draws.
    rng = HostRNG()
    expected = 0.0
    for i in range(300):
        p = source.emit(rng.init_state(SEED, i))
        latent = 2.0 * ELECTRON_MASS_MEV if p.kind == "positron" else 0.0
        expected += p.weight * (p.energy + latent)
    assert result.energy_emitted == pytest.approx(expected, rel=1e-12)
    assert result.energy_emitted == pytest.approx(
        result.energy_deposited + result.energy_escaped, rel=1e-9
    )


@pytest.mark.warp
@pytest.mark.filterwarnings("ignore:.*latent.*")
def test_photon_population_agrees_across_backends() -> None:
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp.engine import WarpEngine

    rng = np.random.default_rng(20260715)
    n = 256
    source = InMemoryPhaseSpaceSource(
        particle_type=np.ones(n, dtype=np.int32),
        energy=rng.uniform(1.0, 6.0, n),
        x=rng.uniform(3.0, 13.0, n),
        y=rng.uniform(3.0, 13.0, n),
        z=np.full(n, -1.0),
        ux=np.zeros(n),
        uy=np.zeros(n),
        uz=np.ones(n),
        weight=rng.uniform(0.2, 1.0, n),
    )
    grid = _grid()
    kwargs = dict(n_batches=8, seed=SEED, transport_electrons=False)
    ref = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()).run(
        source, n_histories=6_000, **kwargs
    )
    warp_result = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu").run(
        source, n_histories=24_000, **kwargs
    )
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp_result.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        warp_result.dose,
        warp_result.dose_sigma,
        warp_result.n_batches,
        mask=mask,
    )
