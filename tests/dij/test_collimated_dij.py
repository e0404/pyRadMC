"""A collimated beamlet source assembles a Dij: per-column weights end to end.

Wraps the spectral beamlet source in water jaws that fully block one bixel's fan:
its column must collapse to the bulk-transmission level while the open columns are
untouched, the weighted per-column ledger must close (slice 4 machinery), and the
pre-sampled Warp route must agree with the reference column for column.
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
    CollimatedBeamletSource,
    JawPair,
)
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import SpectralBeamletSource
from pyRadMC.geometry.spectrum import Spectrum
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
CENTERS = ((5.0, 8.0, 0.0), (8.0, 8.0, 0.0), (11.0, 8.0, 0.0))
THICKNESS = 7.0
# The jaw edge at the mid-plane: bixel 2's whole fan (u in [2, 4] at the aperture
# plane, w = 100 local) projects to u >= 0.87 at the mid-plane w = 43.5; edge 0.5
# blocks it entirely while bixel 0 (u in [-4, -2]) and most of bixel 1 stay open.
EDGE = 0.5


def _source() -> CollimatedBeamletSource:
    inner = SpectralBeamletSource(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        centers=CENTERS,
        width_u=2.0,
        width_v=2.0,
    )
    stack = BeamLimitingStack(
        frame=BeamFrame(origin=FOCAL),
        devices=(
            JawPair(
                axis="u",
                z_top=40.0,
                z_bottom=40.0 + THICKNESS,
                edge_neg=-1000.0,
                edge_pos=EDGE,
                material=WATER,
                density=1.0,
            ),
        ),
    )
    return CollimatedBeamletSource(inner, stack, AnalyticCrossSections())


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _ref_engine(grid: VoxelGrid) -> ReferenceEngine:
    return ReferenceEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )


def test_weighted_column_ledger_closes() -> None:
    source = _source()
    dij = _ref_engine(_grid()).run_dij(source, n_histories_per_beamlet=300, n_batches=3, seed=SEED)
    assert dij.energy_emitted == pytest.approx(dij.energy_deposited + dij.energy_escaped, rel=1e-9)
    # Bixel 2 is fully blocked, so the total emitted energy must sit well below
    # the unattenuated polyenergetic book but above the all-blocked one.
    assert 0.5 < dij.energy_emitted / (3 * 300 * SPECTRUM.mean_energy) < 0.95


def test_blocked_column_collapses_to_the_transmission_level() -> None:
    """The blocked bixel's peak dose is ~T_bulk of an open bixel's peak.

    The slant path through the block exceeds the nominal thickness slightly and
    the spectrum spans 0.5-6 MeV, so the gate brackets the mono-2-MeV estimate
    generously rather than pinning it.
    """
    dij = _ref_engine(_grid()).run_dij(
        _source(),
        n_histories_per_beamlet=2_000,
        n_batches=4,
        seed=SEED,
        transport_electrons=False,
    )
    open_peak = dij.column_dense(0).max()
    blocked_peak = dij.column_dense(2).max()
    t_bulk = math.exp(-AnalyticCrossSections().mu_over_rho_total(2.0, WATER) * THICKNESS)
    assert blocked_peak < 1.6 * t_bulk * open_peak
    assert blocked_peak > 0.2 * t_bulk * open_peak


@pytest.mark.warp
def test_collimated_dij_matches_reference_per_column() -> None:
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = _source()
    kwargs = dict(n_histories_per_beamlet=2_400, n_batches=8, seed=SEED, transport_electrons=False)
    ref = _ref_engine(grid).run_dij(source, **kwargs)
    warp_result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run_dij(
        source, beamlet_group_size=2, **kwargs
    )
    for j in range(source.n_beamlets):
        ref_dose, ref_sigma = ref.column_dense(j), ref.sigma_dense(j)
        warp_dose, warp_sigma = warp_result.column_dense(j), warp_result.sigma_dense(j)
        mask = ref_dose > 0.1 * ref_dose.max()
        mask &= (ref_sigma > 0.0) & (warp_sigma > 0.0)
        assert_chi2_consistent_batched(
            ref_dose,
            ref_sigma,
            ref.n_batches,
            warp_dose,
            warp_sigma,
            warp_result.n_batches,
            mask=mask,
        )
