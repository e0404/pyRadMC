"""broad-beam depth dose shows scatter buildup.

In the KERMA approximation there is no electron buildup, but *scatter* buildup is
fully present: with increasing depth the scattered-photon fluence adds dose that a
primary-only exponential cannot account for. Two analytic expectations follow, both
tested with margins matched to this tier (tiny phantom, low statistics):

1. D(z) * exp(+mu z) — the dose with primary attenuation divided out — must *grow*
   with depth. Without scatter transport it is flat.
2. The effective attenuation slope of ln D(z) must be shallower than the narrow-beam
   mu, but still positive and physical.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource
from tests.conftest import SEED

ENERGY_MEV = 2.0
N_HISTORIES = 10_000
N_BATCHES = 10


@pytest.fixture(scope="module")
def depth_dose() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """One engine run shared by the tests in this module (it is the expensive part)."""
    from pyRadMC.backends.ref.engine import ReferenceEngine
    from pyRadMC.rng.host import HostRNG

    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = ParallelBeamSource(energy=ENERGY_MEV, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    # KERMA mode, deliberately: this test's analytic expectation (D exp(mu z) growth
    # from *scatter* alone) is a photon-only statement, and docs/decisions.md keeps KERMA
    # as the explicit option for exactly this kind of test. Electron buildup has its
    # own test on a fine grid.
    result = engine.run(
        source, n_histories=N_HISTORIES, n_batches=N_BATCHES, seed=SEED, transport_electrons=False
    )

    mu = 1.0 * xs.mu_over_rho_total(ENERGY_MEV, WATER)
    depth = (np.arange(16) + 0.5) * 1.0
    # Slice dose: average over the transverse plane. Slice sigma from quadrature is an
    # under-estimate (in-slice correlations), so assertions below use wide margins.
    dose_z = result.dose.mean(axis=(0, 1))
    sigma_z = np.sqrt((result.dose_sigma**2).sum(axis=(0, 1))) / (16 * 16)
    return depth, dose_z, mu, sigma_z


def test_broad_beam_shows_scatter_buildup_and_reduced_effective_slope(
    depth_dose: tuple[np.ndarray, np.ndarray, float, np.ndarray],
) -> None:
    """Both analytic expectations on one (expensive) engine run."""
    depth, dose_z, mu, _sigma_z = depth_dose
    assert np.all(dose_z > 0.0), "empty depth-dose slices"

    # --- 1. Buildup: primary-corrected dose grows with depth ---------------------
    corrected = dose_z * np.exp(mu * depth)
    front = corrected[:3].mean()
    back = corrected[-5:].mean()
    # At 2 MeV over 16 cm (mu z ~ 0.77) published water buildup factors are ~1.3-1.7;
    # requiring >10 percent growth detects "no scatter" (flat) against slice noise of
    # a few percent, without pinning a number this tier cannot resolve.
    assert back > 1.10 * front, (
        f"no scatter buildup: D exp(mu z) front={front:.4g}, back={back:.4g}. "
        "Scattered photons are not contributing dose."
    )

    # --- 2. Effective slope shallower than narrow-beam mu ------------------------
    slope = -np.polyfit(depth, np.log(dose_z), 1)[0]
    assert slope < mu, (
        f"broad-beam effective slope {slope:.4f} >= narrow-beam mu {mu:.4f} 1/cm: "
        "scatter is missing or double-attenuated"
    )
    assert slope > 0.3 * mu, (
        f"effective slope {slope:.4f} implausibly shallow vs mu {mu:.4f} 1/cm: "
        "suspect over-scattering or an attenuation bookkeeping error"
    )


def test_sigma_estimates_are_available_and_positive_in_the_beam(
    depth_dose: tuple[np.ndarray, np.ndarray, float, np.ndarray],
) -> None:
    """Batched sigma is required infrastructure (AGENTS.md section 2.4).

    Every voxel the beam traverses must carry a nonzero uncertainty estimate.
    """
    _, dose_z, _, sigma_z = depth_dose
    assert np.all(sigma_z > 0.0)
    # Relative slice uncertainty should be planning-grade-ish at this history count.
    assert np.all(sigma_z / dose_z < 0.2)
