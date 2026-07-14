"""Phase 0 exit criterion: exponential attenuation against the analytic law.

The expectations are computed from the *same* CrossSectionSource the transport uses,
so these tests isolate the transport (Woodcock tracking, free-path sampling, channel
bookkeeping) from the data. The data itself is pinned against NIST in the unit tier.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from pyRadMC import PCUT_MEV
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import PencilBeamSource
from pyRadMC.rng.host import HostRNG
from pyRadMC.transport.photon import transport_photon
from tests.conftest import SEED

ENERGY_MEV = 2.0
N_HISTORIES = 4_000


class _FirstDepositDepth:
    """Deposit callback recording only the depth of the first energy deposit."""

    def __init__(self, dz: float) -> None:
        self.dz = dz
        self.depth: float | None = None

    def __call__(self, x: float, y: float, z: float, energy: float) -> None:
        if self.depth is None:
            # Deposits are position-keyed; quantize to the voxel-centre depth the
            # binned exponential comparison below was calibrated against.
            self.depth = (math.floor(z / self.dz) + 0.5) * self.dz


def _first_interaction_depths(
    grid: VoxelGrid, xs: AnalyticCrossSections, n: int
) -> tuple[np.ndarray, int]:
    """Depth of each history's first real interaction; count of clean traversals.

    The first deposit of a history is its first real interaction (delta scatters
    deposit nothing, and the pair vertex deposits before its annihilation photons
    move). Histories with no deposit at all traversed the column untouched.
    """
    source = PencilBeamSource(
        energy=ENERGY_MEV, position=(2.0, 2.0, -1.0), direction=(0.0, 0.0, 1.0)
    )
    rng = HostRNG()
    depths = []
    traversed = 0
    for history in range(n):
        state = rng.init_state(SEED, history)
        primary = source.emit(state)
        recorder = _FirstDepositDepth(grid.spacing[2])
        transport_photon(
            primary.energy,
            primary.x,
            primary.y,
            primary.z,
            primary.ux,
            primary.uy,
            primary.uz,
            grid,
            xs,
            state,
            recorder,
            PCUT_MEV,
        )
        if recorder.depth is not None:
            depths.append(recorder.depth)
        else:
            traversed += 1
    return np.array(depths), traversed


def test_first_interaction_depth_is_exponential_in_water() -> None:
    """Interaction depths follow mu exp(-mu z) and the survival count matches exp(-mu L).

    Chi-squared over depth bins plus the traversal channel — this is the detection
    oracle of AGENTS.md section 4 applied to the attenuation law. Voxel-center depth
    assignment quantizes z to 0.25 cm bins, so the test bins at 2 cm, coarse enough
    that quantization moves no event across a bin edge... except at the edges
    themselves, which the half-open binning below handles consistently.
    """
    grid = VoxelGrid.uniform_water(shape=(16, 16, 64), spacing=(0.25, 0.25, 0.25))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    mu = 1.0 * xs.mu_over_rho_total(ENERGY_MEV, WATER)  # 1/cm at unit density
    depth_total = 64 * 0.25

    depths, traversed = _first_interaction_depths(grid, xs, N_HISTORIES)
    assert len(depths) + traversed == N_HISTORIES

    edges = np.linspace(0.0, depth_total, 9)
    observed = np.append(np.histogram(depths, bins=edges)[0], traversed)
    cdf = 1.0 - np.exp(-mu * edges)
    expected = np.append(np.diff(cdf), np.exp(-mu * depth_total)) * N_HISTORIES
    assert np.all(expected > 20.0), "rebin: chi-squared approximation needs counts"

    chi2, p_value = stats.chisquare(observed, expected * observed.sum() / expected.sum())
    assert p_value > 0.01, (
        f"first-interaction depths not exponential: chi2={chi2:.1f} on "
        f"{len(observed) - 1} dof, p={p_value:.2e}"
    )


def test_survival_through_heterogeneous_slabs_multiplies() -> None:
    """Woodcock tracking through a density step: survival = exp(-mu_1 L_1 - mu_2 L_2).

    This is the test that catches a majorant/delta-scattering bookkeeping error, which
    a homogeneous phantom cannot see (there, delta scattering never fires at unit
    density). Front half water at 1.0 g/cm^3, back half water-like at 0.3 g/cm^3
    (lung-ish); binomial test on the traversal count.
    """
    grid = VoxelGrid.uniform_water(shape=(16, 16, 64), spacing=(0.25, 0.25, 0.25))
    grid.density[:, :, 32:] = 0.3
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    mu_over_rho = xs.mu_over_rho_total(ENERGY_MEV, WATER)
    p_survive = float(np.exp(-mu_over_rho * (1.0 * 8.0 + 0.3 * 8.0)))

    _, traversed = _first_interaction_depths(grid, xs, N_HISTORIES)

    p_value = stats.binomtest(traversed, N_HISTORIES, p_survive).pvalue
    assert p_value > 0.01, (
        f"{traversed}/{N_HISTORIES} traversals vs expected "
        f"{p_survive * N_HISTORIES:.0f}: binomial p={p_value:.2e}. Suspect the "
        "Woodcock delta-scattering bookkeeping, not noise."
    )
