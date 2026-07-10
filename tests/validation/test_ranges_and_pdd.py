"""Validation tier: ESTAR-anchored range gates and physics-derived PDD characteristics.

Runs only with ``-m validation`` (nightly / tagged). Phase 1 content per the
maintainer-approved plan: gates anchored to NIST ESTAR ranges and to physics-derived
depth-dose expectations, plus a documented slot awaiting maintainer-supplied benchmark
PDD curves — the gamma-index comparison (the field convention, and welcome *only* in
this tier per AGENTS.md section 4) lands together with that data.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine, TransportResult
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED


def _electron_beam(
    energy: float, depth_cm: float, n_z: int
) -> tuple[TransportResult, np.ndarray, float]:
    dz = depth_cm / n_z
    grid = VoxelGrid.uniform_water(shape=(8, 8, n_z), spacing=(2.0, 2.0, dz))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = ParallelBeamSource(energy=energy, z=-0.5, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    result = engine.run(source, n_histories=4_000, n_batches=8, seed=SEED, primary_kind="electron")
    depth = (np.arange(n_z) + 0.5) * dz
    return result, depth, xs.csda_range(energy, WATER)


@pytest.mark.validation
@pytest.mark.parametrize("energy", [2.0, 5.0, 10.0])
def test_electron_r50_tracks_the_estar_csda_range(energy: float) -> None:
    """R50 / R_CSDA sits in the detour-factor window across the clinical range.

    For broad monoenergetic MeV electron beams in water the ratio is ~0.8 (the
    detour factor of multiple scattering); the window [0.72, 0.92] gates against
    systematic over- or under-ranging while leaving room for the Gaussian-hinge
    approximation. The CSDA range itself is ESTAR-pinned in the unit tier.
    """
    result, depth, r_csda = _electron_beam(energy, depth_cm=1.5 * r_csda_estimate(energy), n_z=60)
    dose_z = result.dose.mean(axis=(0, 1))
    above_half = np.nonzero(dose_z >= 0.5 * dose_z.max())[0]
    r50 = float(depth[above_half[-1]])
    assert 0.72 * r_csda <= r50 <= 0.92 * r_csda, (
        f"R50({energy} MeV) = {r50:.2f} cm vs R_CSDA = {r_csda:.2f} cm (ratio {r50 / r_csda:.2f})"
    )


def r_csda_estimate(energy: float) -> float:
    """Rough water CSDA range in cm for sizing the phantom; exactness irrelevant."""
    return float(AnalyticCrossSections().csda_range(energy, WATER))


@pytest.mark.validation
@pytest.mark.parametrize("energy", [5.0, 10.0])
def test_only_bremsstrahlung_survives_beyond_the_practical_range(energy: float) -> None:
    """Beyond 1.15 R_CSDA the dose is the brems tail: under 2 percent of Dmax."""
    result, depth, r_csda = _electron_beam(energy, depth_cm=1.6 * r_csda_estimate(energy), n_z=64)
    dose_z = result.dose.mean(axis=(0, 1))
    tail = dose_z[depth > 1.15 * r_csda]
    assert tail.size >= 3
    assert np.all(tail < 0.02 * dose_z.max()), (
        f"dose beyond 1.15 R_CSDA reaches {tail.max() / dose_z.max():.1%} of Dmax"
    )


@pytest.mark.validation
def test_photon_buildup_depth_grows_with_energy() -> None:
    """Monoenergetic photon beams: the buildup depth ordering follows the energy.

    The buildup region ends roughly at the forward range of the most energetic
    secondaries, which grows monotonically with photon energy. Metric: first depth
    reaching 80 percent of the maximum (robust against plateau noise, unlike argmax).
    """
    buildup_depths = []
    for energy, depth_cm in [(2.0, 4.0), (6.0, 8.0), (15.0, 12.0)]:
        n_z = 40
        grid = VoxelGrid.uniform_water(shape=(8, 8, n_z), spacing=(2.0, 2.0, depth_cm / n_z))
        xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
        source = ParallelBeamSource(energy=energy, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
        engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
        result = engine.run(source, n_histories=10_000, n_batches=8, seed=SEED)
        dose_z = result.dose.mean(axis=(0, 1))
        depth = (np.arange(n_z) + 0.5) * depth_cm / n_z
        first_80 = float(depth[np.nonzero(dose_z >= 0.8 * dose_z.max())[0][0]])
        buildup_depths.append(first_80)

    d2, d6, d15 = buildup_depths
    assert d2 < d6 < d15, f"buildup depths not ordered: {buildup_depths}"


@pytest.mark.validation
def test_pdd_against_benchmark_curves_awaits_data() -> None:
    """The gamma-index PDD gate: waiting for maintainer-supplied benchmark curves.

    What is needed (maintainer decision of 2026-07-10, AGENTS.md 7.2): trusted
    depth-dose curves for **monoenergetic parallel photon beams in water** (and
    ideally broad monoenergetic electron beams), e.g. from an EGSnrc/DOSXYZnrc or
    TOPAS run, or a literature table verified against the original publication.
    Format: depth (cm), dose (relative), 1-sigma uncertainty per point.

    When the data lands in ``tests/validation/data/``, this test becomes: run the
    matching beam at high statistics, interpolate onto the benchmark grid, and gate
    with a gamma index (2 percent / 2 mm to start) — the field convention, used only
    in this tier (AGENTS.md section 4). Transcribing published curves from memory was
    considered and rejected: a mis-transcribed oracle is worse than none.
    """
    pytest.skip("waiting for maintainer-supplied benchmark PDD data (see docstring)")
