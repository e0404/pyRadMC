"""A collimated source through a water phantom: ledger, shielding, cross-backend.

The wrapper only rescales primary weights, so the whole transport chain — weighted
deposits, the weighted emitted-energy book, batch sigma — must behave exactly as
the phase-space precedent established. Water jaws on the analytic backend keep the
fast tier EPDL-free; the field is half-blocked so the shielded and open regions of
one phantom pin the attenuation dosimetrically.
"""

from __future__ import annotations

import math

import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.geometry.collimation import (
    BeamFrame,
    BeamLimitingStack,
    CollimatedSource,
    JawPair,
)
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

ENERGY = 2.0
THICKNESS = 10.0  # cm of water-equivalent jaw: T = exp(-mu * 10) ~ 0.61 at 2 MeV


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _source() -> CollimatedSource:
    """A full-field parallel beam whose x > 8 half passes under a water block."""
    inner = ParallelBeamSource(ENERGY, -1.0, (2.0, 14.0), (2.0, 14.0))
    stack = BeamLimitingStack(
        frame=BeamFrame(origin=(0.0, 0.0, -100.0)),
        devices=(
            JawPair(
                axis="u",
                z_top=50.0,
                z_bottom=50.0 + THICKNESS,
                edge_neg=-1000.0,
                edge_pos=8.0,
                material=WATER,
                density=1.0,
            ),
        ),
    )
    return CollimatedSource(inner, stack, AnalyticCrossSections())


def test_reference_ledger_closes_exactly() -> None:
    """emitted = deposited + escaped with per-ray transmission weights."""
    engine = ReferenceEngine(grid=_grid(), cross_sections=AnalyticCrossSections(), rng=HostRNG())
    result = engine.run(_source(), n_histories=400, n_batches=2, seed=SEED)
    assert result.energy_emitted == pytest.approx(
        result.energy_deposited + result.energy_escaped, rel=1e-9
    )
    # Half the field is open (weight 1), half attenuated: the emitted book must
    # sit strictly between the fully-open and fully-blocked totals.
    transmission = math.exp(-AnalyticCrossSections().mu_over_rho_total(ENERGY, WATER) * THICKNESS)
    open_total = 400 * ENERGY
    assert transmission * open_total < result.energy_emitted < open_total


def test_shielded_half_receives_the_transmission_dose() -> None:
    """Entrance-layer dose under the block is ~T times the open half's.

    KERMA mode (photon-only) at the entrance layer keeps the comparison primary-
    dominated; in-phantom scatter still crosses the field edge, so the gate is a
    generous bracket around the Beer-Lambert ratio, not an equality.
    """
    engine = ReferenceEngine(grid=_grid(), cross_sections=AnalyticCrossSections(), rng=HostRNG())
    result = engine.run(
        _source(), n_histories=20_000, n_batches=4, seed=SEED, transport_electrons=False
    )
    entrance = result.dose[:, :, 0]
    open_half = float(entrance[3:6, 4:12].mean())
    shielded_half = float(entrance[10:13, 4:12].mean())
    transmission = math.exp(-AnalyticCrossSections().mu_over_rho_total(ENERGY, WATER) * THICKNESS)
    ratio = shielded_half / open_half
    assert 0.7 * transmission < ratio < 1.4 * transmission


@pytest.mark.warp
def test_collimated_source_agrees_across_backends() -> None:
    """Ref vs Warp cpu chi-squared for the half-blocked field."""
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    kwargs = dict(n_batches=8, seed=SEED, transport_electrons=False)
    ref = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
        _source(), n_histories=6_000, **kwargs
    )
    warp_result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        _source(), n_histories=24_000, **kwargs
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


@pytest.mark.warp
def test_warp_ledger_closes() -> None:
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        _source(), n_histories=2_000, n_batches=2, seed=SEED
    )
    assert result.energy_emitted == pytest.approx(
        result.energy_deposited + result.energy_escaped, rel=1e-4
    )
