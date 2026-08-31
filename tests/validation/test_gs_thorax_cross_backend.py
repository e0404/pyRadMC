"""Cross-backend GS validation on the realistic case: tabulated 6 MV spectral thorax.

The integration tier pins the Warp GS port on an analytic slab; this is the
realistic-case gate the port must pass before any GPU wall number is trusted
(the ECUT lesson: analytic-mono-homogeneous under-measures electron-step
sensitivity by ~2x). Reference vs Warp CUDA, both at the validated
``msc_model="gs"``, ``step_energy_fraction=0.20`` configuration, on the same
synthetic thorax and Ali-Rogers 6 MV divergent beam the perf tier benchmarks.

Method notes, from the L1 record in the workstream notes:

- **Laterally-integrated** energy per z-slab is the compared quantity — never
  per-voxel maxima (a noise outlier at any realistic statistics), and energy
  rather than dose so near-vacuum voxels do not inject 1/mass speckle.
- Per-arm uncertainty is the **seed spread** over independent runs (the engine
  scores per-voxel sigma, but a profile sums correlated voxels, so quadrature
  over voxels would understate it). Six seeds a side satisfies the calibrated
  batched chi-squared helper's dof requirement.
- The compared region is defined on the **mean of both arms** (winner's-curse
  lesson: an ROI defined on one arm's realization biases against the other).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyradmc.adapters.ct import grid_from_hu  # noqa: E402
from pyradmc.backends.ref.engine import ReferenceEngine  # noqa: E402
from pyradmc.data.tabulated import build  # noqa: E402
from pyradmc.data.tabulated.precompile import compile_materials  # noqa: E402
from pyradmc.data.tabulated.source import TabulatedCrossSections  # noqa: E402
from pyradmc.geometry.source import SpectralBeamSource  # noqa: E402
from pyradmc.geometry.spectrum import ali_rogers_mv  # noqa: E402
from pyradmc.rng.host import HostRNG  # noqa: E402
from tests.perf.test_ct_throughput import SHAPE, SPACING, thorax_hu  # noqa: E402

pytestmark = [pytest.mark.validation, pytest.mark.warp]

N_SEEDS = 6
N_REF = 20_000
N_WARP = 80_000
GS_FRACTION = 0.20


@pytest.fixture(scope="module")
def thorax():
    """The perf tier's thorax grid with compiled EPICS media, or skip."""
    epdl, eedl = build.library_path("epdl", None), build.library_path("eedl", None)
    if not (epdl.is_file() and eedl.is_file()):
        pytest.skip("EPICS libraries not cached; run python -m pyradmc.data.tabulated.build")
    grid = grid_from_hu(thorax_hu(), SPACING)
    data = compile_materials(
        epdl.read_text(encoding="latin-1"), eedl.read_text(encoding="latin-1"), e_max=8.0
    )
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    return grid, xs


def _source() -> SpectralBeamSource:
    cx, cy = SHAPE[0] * SPACING[0] / 2.0, SHAPE[1] * SPACING[1] / 2.0
    return SpectralBeamSource(
        spectrum=ali_rogers_mv("varian-6mv"),
        focal_point=(cx, cy, -100.0),
        center=(cx, cy, 0.0),
        width_u=10.0,
        width_v=10.0,
    )


def _depth_energy_profile(dose: np.ndarray, grid) -> np.ndarray:
    """Energy deposited per z-slab: laterally integrated, mass-weighted."""
    voxel_volume = float(np.prod(grid.spacing))
    return np.sum(dose * grid.density * voxel_volume, axis=(0, 1))


@pytest.mark.gpu
def test_gs_thorax_depth_profile_agrees_across_backends(thorax) -> None:
    """Ref and Warp CUDA agree in laterally-integrated depth dose under GS f=0.20.

    The calibrated chi-squared runs over every z-slab carrying more than 1
    percent of the peak of the two arms' mean profile — the fair region, on
    which a port defect anywhere in the GS path (window, cast, keying, blend,
    cap conditional) shows up as depth-profile drift. A scalar check on total
    deposited energy backs it up with an interpretable magnitude.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    grid, xs = thorax
    source = _source()

    ref_profiles = []
    ref_totals = []
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    for s in range(N_SEEDS):
        result = engine.run(
            source,
            n_histories=N_REF,
            n_batches=1,
            seed=SEED + s,
            msc_model="gs",
            step_energy_fraction=GS_FRACTION,
        )
        ref_profiles.append(_depth_energy_profile(result.dose, grid))
        ref_totals.append(result.energy_deposited / result.n_histories)

    warp_engine = WarpEngine(grid=grid, cross_sections=xs, device="cuda:0")
    warp_profiles = []
    warp_totals = []
    for s in range(N_SEEDS):
        result = warp_engine.run(
            source,
            n_histories=N_WARP,
            n_batches=1,
            seed=SEED + 100 + s,
            msc_model="gs",
            step_energy_fraction=GS_FRACTION,
        )
        warp_profiles.append(_depth_energy_profile(result.dose, grid))
        warp_totals.append(result.energy_deposited / result.n_histories)

    a = np.mean(ref_profiles, axis=0)
    sem_a = np.std(ref_profiles, axis=0, ddof=1) / np.sqrt(N_SEEDS)
    b = np.mean(warp_profiles, axis=0)
    sem_b = np.std(warp_profiles, axis=0, ddof=1) / np.sqrt(N_SEEDS)

    mean_profile = 0.5 * (a + b)
    mask = mean_profile > 0.01 * mean_profile.max()
    mask &= (sem_a > 0.0) & (sem_b > 0.0)
    assert int(mask.sum()) >= 20, "thorax profile mask unexpectedly small"

    assert_chi2_consistent_batched(a, sem_a, N_SEEDS, b, sem_b, N_SEEDS, mask=mask)

    # Interpretable scalar: mean deposited energy per history, Welch z.
    ta, tb = np.asarray(ref_totals), np.asarray(warp_totals)
    combined = np.hypot(ta.std(ddof=1) / np.sqrt(N_SEEDS), tb.std(ddof=1) / np.sqrt(N_SEEDS))
    z = abs(ta.mean() - tb.mean()) / combined
    assert z < 4.0, (
        f"total deposited energy differs: ref {ta.mean():.5e} vs warp {tb.mean():.5e} "
        f"MeV/history ({z:.1f} combined sigma)"
    )
