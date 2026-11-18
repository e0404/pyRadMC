"""Annihilation in flight can make a photon harder than anything the source emits.

Every other secondary has less energy than its parent, which is why the engines
size their tables to ``source.max_energy`` and check the data's energy domain
there. A positron annihilating in flight is the exception: its photons share the
kinetic energy *and* both rest masses, so the harder one can carry up to

    k_max = (T + 2 m) / 2 + sqrt(T (T + 2 m)) / 2,

between 0.51 and 0.77 MeV above the positron's kinetic energy T. A positron
born of a photon has at least 2 m less than that photon, so its annihilation
photons stay below the source energy; only a positron *the source itself starts*
can exceed it. Under ``"bhabha-in-flight"`` with positron primaries the run's
energy ceiling is therefore ``k_max(source.max_energy)``, and that is what the
tables are built to and what the data's domain is checked against (AGENTS.md 2.6:
fail closed, never clamp).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc import ELECTRON_MASS_MEV, PhysicsOptions
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.collimation import TransmissionMaskSource
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import CompositeSource, PencilBeamSource, Primary, Source
from pyradmc.physics.annihilation import sample_annihilation_photon_energy
from pyradmc.physics_options import CLINICAL_PHYSICS, ResolvedPhysics, run_energy_ceiling
from pyradmc.rng import RNGState
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

IN_FLIGHT = ResolvedPhysics(
    bremsstrahlung="thin-target", pair="uniform-split", positron="bhabha-in-flight"
)
TWO_M = 2.0 * ELECTRON_MASS_MEV


class _PositronPencil(Source):
    particle_kinds = frozenset({"positron"})

    def __init__(self, energy: float) -> None:
        self._energy = energy

    @property
    def max_energy(self) -> float:
        return self._energy

    def emit(self, rng_state: RNGState) -> Primary:
        return Primary(self._energy, 4.0, 4.0, -1.0, 0.0, 0.0, 1.0, kind="positron")


def _pencil(energy: float) -> PencilBeamSource:
    return PencilBeamSource(energy=energy, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))


def _mask(inner: Source) -> TransmissionMaskSource:
    return TransmissionMaskSource(
        inner, mask=np.ones((2, 2)), plane_center=(4.0, 4.0, 0.0), width_u=4.0, width_v=4.0
    )


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


class TestCeiling:
    @pytest.mark.parametrize("energy", [0.3, 1.0, 12.0, 200.0])
    def test_positron_primaries_under_the_model_reach_the_hard_photon(self, energy: float) -> None:
        ceiling = run_energy_ceiling(energy, {"positron"}, IN_FLIGHT)
        expected = 0.5 * (energy + TWO_M) + 0.5 * math.sqrt(energy * (energy + TWO_M))
        assert ceiling == pytest.approx(expected, rel=1e-14)
        assert energy + ELECTRON_MASS_MEV < ceiling < energy + 1.5 * ELECTRON_MASS_MEV

    @pytest.mark.parametrize("energy", [1.0, 12.0, 200.0])
    def test_no_sampled_photon_exceeds_it(self, energy: float) -> None:
        ceiling = run_energy_ceiling(energy, {"positron"}, IN_FLIGHT)
        state = HostRNG().init_state(SEED, 7)
        hardest = 0.0
        for _ in range(20_000):
            k = sample_annihilation_photon_energy(energy, state)
            hardest = max(hardest, k, energy + TWO_M - k)
        assert energy < hardest <= ceiling * (1.0 + 1e-12)

    @pytest.mark.parametrize(
        ("kinds", "physics"),
        [
            ({"photon"}, IN_FLIGHT),  # pair-born positrons stay below the photon
            ({"photon", "electron"}, IN_FLIGHT),
            ({"positron"}, CLINICAL_PHYSICS),  # at rest: 511 keV photons only
            ({"photon", "electron", "positron"}, CLINICAL_PHYSICS),
        ],
    )
    def test_every_other_run_is_bounded_by_its_source(
        self, kinds: set[str], physics: ResolvedPhysics
    ) -> None:
        """Exactly the source energy: the default path's table key does not move."""
        assert run_energy_ceiling(12.0, kinds, physics) == 12.0


@pytest.mark.parametrize("backend", ["ref", "warp"])
class TestTheDomainIsCheckedAtTheCeiling:
    def _engine(self, backend: str):  # type: ignore[no-untyped-def]
        grid = _grid()
        if backend == "ref":
            return ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())
        pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
        from pyradmc.backends.warp.engine import WarpEngine

        return WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu")

    def test_a_positron_source_at_the_domain_edge_is_refused_under_the_model(
        self, backend: str
    ) -> None:
        """19.8 MeV positrons on data that ends at 20 MeV: photons would reach 20.56."""
        engine = self._engine(backend)
        common = dict(n_histories=8, n_batches=2, seed=SEED, msc_model="gaussian")
        with pytest.raises(ValueError, match="energy domain"):
            engine.run(
                _PositronPencil(19.8), **common, physics=PhysicsOptions(positron="bhabha-in-flight")
            )
        # Annihilating at rest, the same source stays inside the domain and runs.
        clinical = engine.run(_PositronPencil(19.8), **common)
        assert clinical.dose.max() > 0.0

    @pytest.mark.parametrize("wrap", [False, True], ids=["mixture", "wrapped-mixture"])
    def test_a_mixture_starting_positrons_is_refused_under_the_model(
        self, backend: str, wrap: bool
    ) -> None:
        """A positron component keeps its ceiling when mixed with a beam.

        The beam leaves its kind to the engine, so the mixture cannot declare its
        kinds itself; the positrons it starts still reach 20.56 MeV photons.
        """
        engine = self._engine(backend)
        common = dict(n_histories=8, n_batches=2, seed=SEED, msc_model="gaussian")

        def mixture() -> Source:
            mixed = CompositeSource([(_PositronPencil(19.8), 0.5), (_pencil(19.8), 0.5)])
            return _mask(mixed) if wrap else mixed

        with pytest.raises(ValueError, match="energy domain"):
            engine.run(mixture(), **common, physics=PhysicsOptions(positron="bhabha-in-flight"))
        clinical = engine.run(mixture(), **common)
        assert clinical.dose.max() > 0.0

    def test_a_photon_source_at_the_domain_edge_runs_under_the_model(self, backend: str) -> None:
        engine = self._engine(backend)
        beam = PencilBeamSource(energy=20.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
        result = engine.run(
            beam,
            n_histories=200,
            n_batches=2,
            seed=SEED,
            msc_model="gaussian",
            physics=PhysicsOptions(positron="bhabha-in-flight"),
        )
        assert result.dose.max() > 0.0


@pytest.mark.warp
def test_warp_tables_are_built_to_the_ceiling() -> None:
    """The photon grid's top node covers the hardest annihilation photon."""
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp import kernels
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    engine = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu")
    for physics in (PhysicsOptions(positron="bhabha-in-flight"), PhysicsOptions()):
        engine.run(_PositronPencil(12.0), n_histories=400, n_batches=2, seed=SEED, physics=physics)
    tops = {}
    for tab in engine._state("cpu").tables.values():
        tops[int(tab.positron_model)] = float(
            np.exp(tab.p_log_e_min + (tab.n_points - 1) / tab.p_inv_dlog)
        )
    ceiling = run_energy_ceiling(12.0, {"positron"}, IN_FLIGHT)
    assert tops[kernels.POSITRON_BHABHA_IN_FLIGHT] >= ceiling
    assert tops[kernels.POSITRON_MOLLER_AT_REST] == pytest.approx(12.0, rel=1e-5)


@pytest.mark.warp
def test_warp_tables_of_a_mixture_are_built_to_the_ceiling() -> None:
    """A positron component mixed with a beam sizes the photon grid like it alone."""
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp import kernels
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    engine = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu")
    mixture = CompositeSource([(_PositronPencil(12.0), 0.5), (_pencil(12.0), 0.5)])
    engine.run(
        mixture,
        n_histories=400,
        n_batches=2,
        seed=SEED,
        physics=PhysicsOptions(positron="bhabha-in-flight"),
    )
    (tab,) = engine._state("cpu").tables.values()
    assert int(tab.positron_model) == kernels.POSITRON_BHABHA_IN_FLIGHT
    top = float(np.exp(tab.p_log_e_min + (tab.n_points - 1) / tab.p_inv_dlog))
    assert top >= run_energy_ceiling(12.0, {"positron"}, IN_FLIGHT)
