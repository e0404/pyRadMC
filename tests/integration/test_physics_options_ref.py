"""The ``physics`` argument on the reference engine (AGENTS.md 2.10).

With one registered model per process the selector cannot change a result, and
that is the property under test: naming today's models explicitly transports the
same histories, bit for bit, as leaving every process on ``"auto"`` — on ``run``
and on ``run_dij``. The beam is 10 MeV so the pair channel, bremsstrahlung and
positrons are all actually sampled; a low-energy beam would pass vacuously.

The remaining tests pin the plumbing a second model will rely on: the resolved
choice is logged once per run and recorded in the provenance, a bad option is
refused before any history starts, and a model that is registered but not
implemented in a loop raises instead of running as another.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

from pyradmc import PhysicsOptions, physics_options
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.phasespace import InMemoryPhaseSpaceSource
from pyradmc.geometry.source import BeamletGridSource, PencilBeamSource
from pyradmc.physics_options import CLINICAL_PHYSICS, ProcessModels, ResolvedPhysics
from pyradmc.rng.host import HostRNG
from pyradmc.transport.electron import electron_steps
from pyradmc.transport.particles import StackEntry
from pyradmc.transport.photon import photon_steps
from tests.conftest import SEED

EXPLICIT = PhysicsOptions(
    bremsstrahlung="thin-target", pair="uniform-split", positron="moller-at-rest"
)
ENERGY_MEV = 10.0


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def engine() -> ReferenceEngine:
    grid = _grid()
    return ReferenceEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )


def _beam() -> PencilBeamSource:
    return PencilBeamSource(energy=ENERGY_MEV, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))


def _beamlets() -> BeamletGridSource:
    return BeamletGridSource(
        energy=ENERGY_MEV, z=-1.0, x_range=(2.0, 6.0), y_range=(3.0, 5.0), n_x=2, n_y=1
    )


@pytest.fixture(scope="module")
def default_run(engine: ReferenceEngine):
    return engine.run(_beam(), n_histories=80, n_batches=4, seed=SEED)


class TestByteIdentity:
    @pytest.mark.parametrize("physics", ["auto", PhysicsOptions(), EXPLICIT])
    def test_run_matches_the_default(self, engine: ReferenceEngine, default_run, physics) -> None:
        result = engine.run(_beam(), n_histories=80, n_batches=4, seed=SEED, physics=physics)

        assert default_run.dose.max() > 0.0
        assert result.dose.tobytes() == default_run.dose.tobytes()
        assert result.dose_sigma.tobytes() == default_run.dose_sigma.tobytes()
        assert result.energy_deposited == default_run.energy_deposited
        assert result.energy_escaped == default_run.energy_escaped

    def test_run_dij_matches_the_default(self, engine: ReferenceEngine) -> None:
        common = dict(n_histories_per_beamlet=40, n_batches=4, seed=SEED)
        default = engine.run_dij(_beamlets(), **common)
        explicit = engine.run_dij(_beamlets(), **common, physics=EXPLICIT)

        a, b = default.dose_csc(), explicit.dose_csc()
        assert a.nnz > 0
        assert a.data.tobytes() == b.data.tobytes()
        assert np.array_equal(a.indices, b.indices)
        assert np.array_equal(a.indptr, b.indptr)
        assert default.energy_escaped == explicit.energy_escaped


class TestResolutionIsRecorded:
    def test_provenance_carries_the_resolved_models(self, default_run) -> None:
        assert default_run.provenance is not None
        assert default_run.provenance.physics == CLINICAL_PHYSICS
        assert "pair=uniform-split" in default_run.provenance.summary()

    def test_dij_provenance_carries_the_resolved_models(self, engine: ReferenceEngine) -> None:
        result = engine.run_dij(_beamlets(), n_histories_per_beamlet=4, n_batches=2, seed=SEED)
        assert result.provenance is not None
        assert result.provenance.physics == CLINICAL_PHYSICS

    def test_run_logs_the_choice_and_reason_once(
        self, engine: ReferenceEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="pyradmc.backends.ref.engine"):
            engine.run(_beam(), n_histories=8, n_batches=2, seed=SEED)

        records = [r for r in caplog.records if "physics" in r.getMessage()]
        assert len(records) == 1
        message = records[0].getMessage()
        assert records[0].levelno == logging.INFO
        assert "bremsstrahlung=thin-target" in message
        assert "photon primaries up to 10 MeV" in message
        assert "at or below the 30 MeV" in message

    def test_run_dij_logs_once(
        self, engine: ReferenceEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="pyradmc.backends.ref.engine"):
            engine.run_dij(_beamlets(), n_histories_per_beamlet=4, n_batches=2, seed=SEED)

        assert sum("physics" in r.getMessage() for r in caplog.records) == 1

    def test_log_names_the_fallback_primary_kind(
        self, engine: ReferenceEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="pyradmc.backends.ref.engine"):
            engine.run(_beam(), n_histories=4, n_batches=2, seed=SEED, primary_kind="electron")

        assert "electron primaries" in caplog.text

    def test_log_names_the_kinds_a_phase_space_declares(
        self, engine: ReferenceEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        n = 4
        source = InMemoryPhaseSpaceSource(
            particle_type=np.array([1, 3, 1, 3], dtype=np.int32),
            energy=np.full(n, 2.0),
            x=np.full(n, 4.0),
            y=np.full(n, 4.0),
            z=np.full(n, -0.5),
            ux=np.zeros(n),
            uy=np.zeros(n),
            uz=np.ones(n),
            weight=np.ones(n),
        )
        with caplog.at_level(logging.INFO, logger="pyradmc.backends.ref.engine"):
            engine.run(source, n_histories=4, n_batches=2, seed=SEED)

        assert "photon/positron primaries" in caplog.text


class TestRefusals:
    def test_unknown_preset_is_refused_before_transport(self, engine: ReferenceEngine) -> None:
        with pytest.raises(ValueError, match="unknown physics preset"):
            engine.run(_beam(), n_histories=4, n_batches=2, seed=SEED, physics="fancy")

    def test_unknown_preset_is_refused_by_run_dij(self, engine: ReferenceEngine) -> None:
        with pytest.raises(ValueError, match="unknown physics preset"):
            engine.run_dij(
                _beamlets(), n_histories_per_beamlet=4, n_batches=2, seed=SEED, physics="fancy"
            )


@pytest.fixture
def unimplemented(monkeypatch: pytest.MonkeyPatch) -> ResolvedPhysics:
    """A second model per process, registered in the table but implemented nowhere."""
    table = {
        process: ProcessModels(
            models=(*row.models, "unimplemented"),
            auto_clinical=row.auto_clinical,
            auto_high_energy=row.auto_high_energy,
        )
        for process, row in physics_options.PHYSICS_MODELS.items()
    }
    monkeypatch.setattr(physics_options, "PHYSICS_MODELS", table)
    return ResolvedPhysics(
        bremsstrahlung="unimplemented", pair="unimplemented", positron="unimplemented"
    )


class TestALoopRefusesAModelItDoesNotImplement:
    """Registering a name without implementing it must fail loudly, never silently."""

    def _loop_args(self, engine: ReferenceEngine) -> tuple:
        stack: list[StackEntry] = []

        def deposit(x: float, y: float, z: float, energy: float, weighted: float) -> None:
            pass

        state = engine.rng.init_state(SEED, 0)
        return (engine.grid, engine.cross_sections, state, deposit, stack.append)

    def test_photon_loop_refuses_an_unimplemented_pair_model(
        self, engine: ReferenceEngine, unimplemented: ResolvedPhysics
    ) -> None:
        with pytest.raises(NotImplementedError, match="pair model 'unimplemented'"):
            photon_steps(
                ENERGY_MEV,
                1.0,
                4.0,
                4.0,
                0.5,
                0.0,
                0.0,
                1.0,
                *self._loop_args(engine),
                pcut=0.05,
                ecut=0.2,
                transport_electrons=True,
                is_primary=True,
                physics=unimplemented,
            )

    @pytest.mark.parametrize(
        ("is_positron", "process"), [(False, "bremsstrahlung"), (True, "positron")]
    )
    def test_electron_loop_refuses_an_unimplemented_model(
        self,
        engine: ReferenceEngine,
        unimplemented: ResolvedPhysics,
        is_positron: bool,
        process: str,
    ) -> None:
        physics = dataclasses.replace(CLINICAL_PHYSICS, **{process: "unimplemented"})
        assert physics != unimplemented  # only the process under test is unimplemented
        with pytest.raises(NotImplementedError, match=f"{process} model 'unimplemented'"):
            electron_steps(
                is_positron,
                10.0,
                1.0,
                4.0,
                4.0,
                0.5,
                0.0,
                0.0,
                1.0,
                *self._loop_args(engine),
                pcut=0.05,
                ecut=0.2,
                physics=physics,
            )

    def test_an_electron_is_indifferent_to_the_positron_model(
        self, engine: ReferenceEngine, unimplemented: ResolvedPhysics
    ) -> None:
        """Only a positron's transport reads the positron model."""
        physics = dataclasses.replace(unimplemented, bremsstrahlung="thin-target")
        electron_steps(
            False, 1.0, 1.0, 4.0, 4.0, 0.5, 0.0, 0.0, 1.0,
            *self._loop_args(engine), pcut=0.05, ecut=0.2, physics=physics,
        )  # fmt: skip
