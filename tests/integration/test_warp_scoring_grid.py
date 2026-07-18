"""Decoupled scoring grid on the Warp backend, per device.

The scoring grid must not touch transport: streams, interaction sequences and the
escape ledger are invariant to it, and the int64 fixed-point deposits are
quantized per deposit *before* routing, so a subregion run's covered voxels carry
exactly the covering run's quanta. That makes the identities here exact on one
device (AGENTS.md 2.3: bit assertions never cross targets — the cross-backend
check is the chi-squared oracle at the end).

Geometry uses power-of-two spacings so the host-side overlap mass rebin is exact
binary arithmetic and dose comparisons inherit the quanta-level exactness.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.backends.warp.kernels import ENERGY_QUANTUM_MEV
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource
from pyRadMC.rng.host import HostRNG
from pyRadMC.scoring.grid import ScoringGrid
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

ENERGY_BALANCE_RTOL = 1.0e-4  # float32 transport + 1e-9 MeV scoring quanta


def _grid() -> VoxelGrid:
    shape = (8, 8, 24)
    rng = np.random.default_rng(20260714)
    density = rng.uniform(0.5, 1.5, size=shape)
    from pyRadMC.data.materials import WATER

    return VoxelGrid(
        shape=shape,
        spacing=(2.0, 2.0, 0.25),
        density=density,
        material=np.full(shape, WATER, dtype=np.int32),
    )


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _source() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(2.0, 14.0), y_range=(2.0, 14.0))


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _warp_engine(device: str):
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device)


def _run(engine, scoring_grid, n: int = 4_000, n_batches: int = 4):
    return engine.run(
        _source(), n_histories=n, n_batches=n_batches, seed=SEED, scoring_grid=scoring_grid
    )


def test_default_is_bit_identical_to_explicit_transport_grid_scoring(device: str) -> None:
    engine = _warp_engine(device)
    a = _run(engine, None)
    b = _run(engine, ScoringGrid.for_grid(engine.grid))
    np.testing.assert_array_equal(a.dose, b.dose)
    np.testing.assert_array_equal(a.dose_sigma, b.dose_sigma)
    assert a.energy_deposited == b.energy_deposited
    assert a.energy_escaped == b.energy_escaped
    assert a.energy_unscored == 0.0
    assert b.energy_unscored == 0.0


def test_aligned_coarsening_is_the_exact_energy_rebin(device: str) -> None:
    """Deposits are quantized before routing, and integer sums are associative:
    each coarse voxel's quanta equal the sum of its children's, exactly."""
    engine = _warp_engine(device)
    fine = _run(engine, None)
    grid = engine.grid
    coarse_sg = ScoringGrid.rebin(
        grid, shape=(4, 4, 12), spacing=(4.0, 4.0, 0.5), origin=grid.origin
    )
    coarse = _run(engine, coarse_sg)

    assert coarse.energy_escaped == fine.energy_escaped
    assert coarse.energy_unscored == 0.0
    assert coarse.energy_deposited == fine.energy_deposited  # same quanta, regrouped

    fine_mass = grid.density * grid.voxel_volume
    fine_energy = fine.dose * fine_mass
    child = fine_energy.reshape(4, 2, 4, 2, 12, 2).sum(axis=(1, 3, 5))
    child_mass = fine_mass.reshape(4, 2, 4, 2, 12, 2).sum(axis=(1, 3, 5))
    np.testing.assert_allclose(coarse.dose, child / child_mass, rtol=1e-12)


def test_subregion_books_unscored_and_never_clamps(device: str) -> None:
    engine = _warp_engine(device)
    full = _run(engine, None)
    grid = engine.grid
    # First 12 of 24 z-slabs: z in [0, 3), full x-y coverage.
    sub_sg = ScoringGrid.rebin(
        grid, shape=(8, 8, 12), spacing=(2.0, 2.0, 0.25), origin=(0.0, 0.0, 0.0)
    )
    sub = _run(engine, sub_sg)

    assert sub.energy_unscored > 0.0
    assert sub.energy_escaped == full.energy_escaped

    # The guarantee is exact in int64 *quanta* (quantization happens once, before
    # routing); comparing in MeV would ask float64 addition to associate — the
    # quanta sums are the invariant, so reconstruct and compare those.
    def _quanta(energy_mev: float) -> int:
        return round(energy_mev / ENERGY_QUANTUM_MEV)

    assert _quanta(sub.energy_deposited) + _quanta(sub.energy_unscored) == _quanta(
        full.energy_deposited
    )
    assert sub.energy_deposited + sub.energy_unscored + sub.energy_escaped == pytest.approx(
        sub.energy_emitted, rel=ENERGY_BALANCE_RTOL
    )
    # No clamping: identical quanta in the covered region, edge slab included.
    np.testing.assert_array_equal(sub.dose, full.dose[:, :, :12])
    np.testing.assert_array_equal(sub.dose_sigma, full.dose_sigma[:, :, :12])


def test_coarse_grid_dose_matches_reference_statistically(device: str) -> None:
    """Cross-backend on a shared coarse subregion grid: chi-squared, never bits."""
    grid = _grid()
    sg = ScoringGrid.rebin(grid, shape=(4, 4, 8), spacing=(4.0, 4.0, 0.5), origin=grid.origin)
    ref_engine = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())
    ref = ref_engine.run(_source(), n_histories=4_000, n_batches=8, seed=SEED, scoring_grid=sg)
    result = _warp_engine(device).run(
        _source(), n_histories=16_000, n_batches=8, seed=SEED, scoring_grid=sg
    )
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (result.dose_sigma > 0.0)
    assert np.count_nonzero(mask) > 30, "mask too small to detect anything"
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        result.dose,
        result.dose_sigma,
        result.n_batches,
        mask=mask,
    )
