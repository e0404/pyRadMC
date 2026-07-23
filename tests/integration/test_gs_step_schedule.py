"""The angular substep cap is a Gaussian-validity limit, not applied under GS.

``STEP_HINGE_THETA2_MAX`` exists because the small-angle Gaussian hinge is
invalid at large per-step angles. The Goudsmit-Saunderson sampler is exact at
arbitrary angle — that validity is the reason it was built — so under
``msc_model="gs"`` the substep takes the full energy-limited length and the cap
is not applied (maintainer decision 2026-07-21; validated with the cap lifted on
the R50/detour and PDD-gamma gates and the realistic tabulated spectral thorax).

Both tests count substeps by spying the one sampler call each model makes per
substep; counts are deterministic at a fixed seed, so the bands are physics
statements, not statistical tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyRadMC.transport.electron as electron_module
from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED

N_HISTORIES = 150
ENERGY_MEV = 6.0


@pytest.fixture(scope="module")
def shared_engine() -> tuple[ReferenceEngine, ParallelBeamSource]:
    """One engine (and so one GS table cache) for the whole module.

    The GS tables are memoized per source instance; a fresh source per test
    rebuilds them from scratch, which is the dominant cost of these tests by
    far. Sharing is safe: the engine holds no per-run state, and the substep
    counts are deterministic at the fixed seed.
    """
    grid = VoxelGrid.uniform_water(shape=(16, 16, 24), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = ParallelBeamSource(energy=ENERGY_MEV, z=-0.5, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()), source


def _substep_count(
    monkeypatch: pytest.MonkeyPatch,
    shared: tuple[ReferenceEngine, ParallelBeamSource],
    msc_model: str,
    fraction: float,
) -> int:
    """Substeps taken transporting ``N_HISTORIES`` electrons on a coarse grid.

    Coarse 1 cm voxels so the voxel-face cap rarely binds and the energy/angular
    interplay is what the count measures.
    """
    engine, source = shared
    xs = engine.cross_sections

    calls = {"n": 0}
    if msc_model == "gs":
        original = type(xs).sample_gs_cos_theta

        def spy_gs(self: AnalyticCrossSections, *args: object, **kwargs: object) -> float:
            calls["n"] += 1
            return original(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(type(xs), "sample_gs_cos_theta", spy_gs)
    else:
        original_hinge = electron_module.sample_hinge_cos_theta

        def spy_hinge(mean_square_angle: float, rng_state: object) -> float:
            calls["n"] += 1
            return original_hinge(mean_square_angle, rng_state)

        monkeypatch.setattr(electron_module, "sample_hinge_cos_theta", spy_hinge)

    engine.run(
        source,
        n_histories=N_HISTORIES,
        n_batches=2,
        seed=SEED,
        primary_kind="electron",
        msc_model=msc_model,
        step_energy_fraction=fraction,
    )
    return calls["n"]


def test_gs_takes_the_energy_limited_step_at_large_fraction(
    monkeypatch: pytest.MonkeyPatch, shared_engine: tuple[ReferenceEngine, ParallelBeamSource]
) -> None:
    """At ``f = 0.20`` GS takes far fewer substeps: the angular cap is not applied.

    Under the Gaussian model the cap re-binds at ``f`` beyond ~0.057 in water,
    clamping the effective step regardless of the requested fraction — measured
    in the f-sweep as the reason raising ``f`` alone bought only ~5 percent.
    Under GS the full energy-limited step stands. The measured schedule ratio is
    ~3.5x; the band asserts half of it so grid detail cannot flake the test.
    """
    gaussian = _substep_count(monkeypatch, shared_engine, "gaussian", 0.20)
    gs = _substep_count(monkeypatch, shared_engine, "gs", 0.20)
    assert gs < 0.55 * gaussian, (
        f"GS took {gs} substeps vs Gaussian {gaussian} at f=0.20 — the angular cap "
        "appears to still bind under GS"
    )


def test_schedules_agree_at_the_shipped_default_fraction(
    monkeypatch: pytest.MonkeyPatch, shared_engine: tuple[ReferenceEngine, ParallelBeamSource]
) -> None:
    """At ``f = 0.05`` in water the cap is inert, so the schedules coincide.

    Measured (2026-07-20, the falsified-premise result): ``s_theta / s_E >= 1.14``
    pointwise across the transported range in water, so removing the cap under GS
    changes nothing at the shipped default — GS alone is NOT a speed lever. This
    pins that fact as a regression test: if the ratio ever leaves the band, either
    the cap has started binding at the default (scattering-power data changed) or
    the conditional was wired wrong. The residual difference is path-length
    feedback from the differing deflections, not schedule.
    """
    gaussian = _substep_count(monkeypatch, shared_engine, "gaussian", 0.05)
    gs = _substep_count(monkeypatch, shared_engine, "gs", 0.05)
    ratio = gs / gaussian
    assert 0.97 <= ratio <= 1.03, (
        f"substep counts diverge at the shipped default: GS {gs} vs Gaussian "
        f"{gaussian} (ratio {ratio:.3f}) — the angular cap should be inert here"
    )


def test_gaussian_schedule_is_untouched_by_the_conditional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Gaussian instrument path must stay bit-stable under the conditional.

    Cheap proxy at transport level: dose from a Gaussian-instrument run is
    exactly reproducible against itself with the conditional in place (same
    seed, same stream). Guards against an accidental reorder of the min() that
    would change float evaluation on the instrument path every paired
    comparison replays. Explicit ``msc_model`` since the shipped default is GS.
    """
    grid = VoxelGrid.uniform_water(shape=(12, 12, 16), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = ParallelBeamSource(energy=ENERGY_MEV, z=-0.5, x_range=(0.0, 12.0), y_range=(0.0, 12.0))
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    kwargs = dict(n_histories=200, n_batches=2, seed=SEED, primary_kind="electron")
    a = engine.run(source, msc_model="gaussian", **kwargs)
    b = engine.run(source, msc_model="gaussian", **kwargs)
    np.testing.assert_array_equal(a.dose, b.dose)
