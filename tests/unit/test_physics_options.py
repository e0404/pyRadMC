"""Selectable physics models: validation, the registration table, and ``"auto"``.

AGENTS.md 2.10: the physics a run uses is chosen per process through
:class:`~pyradmc.physics_options.PhysicsOptions`; ``"auto"`` is resolved from the
run configuration by one pure function reading one table. These tests pin that
function and that table: the shipped rows (bremsstrahlung switches to the screened
Bethe-Heitler model above the high-energy threshold; pair production and the
positron have one model each), and the threshold logic for *every* process against
a patched table that registers a second model per row.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import pytest

import pyradmc
from pyradmc import physics_options
from pyradmc.physics_options import (
    CLINICAL_PHYSICS,
    PHYSICS_MODELS,
    PhysicsOptions,
    ProcessModels,
    ResolvedPhysics,
    require_implemented,
    resolve_physics,
    run_primary_kinds,
)

PROCESSES = ("bremsstrahlung", "pair", "positron")
PHOTONS = frozenset({"photon"})


class TestRegistrationTable:
    def test_the_three_processes_are_registered(self) -> None:
        assert tuple(PHYSICS_MODELS) == PROCESSES

    def test_the_registered_models(self) -> None:
        """The shipped table: two bremsstrahlung models, one for the other processes."""
        assert PHYSICS_MODELS["bremsstrahlung"].models == ("thin-target", "bethe-heitler")
        assert PHYSICS_MODELS["pair"].models == ("uniform-split",)
        assert PHYSICS_MODELS["positron"].models == ("moller-at-rest",)

    def test_bremsstrahlung_switches_model_at_the_threshold(self) -> None:
        """Accuracy-defining (AGENTS.md 2.10): what "auto" names on each side."""
        row = PHYSICS_MODELS["bremsstrahlung"]
        assert row.auto_clinical == "thin-target"
        assert row.auto_high_energy == "bethe-heitler"

    @pytest.mark.parametrize("process", ["pair", "positron"])
    def test_single_model_rows_name_it_on_both_sides(self, process: str) -> None:
        row = PHYSICS_MODELS[process]
        assert row.auto_clinical == row.auto_high_energy == row.models[0]

    @pytest.mark.parametrize("process", PROCESSES)
    def test_auto_choices_are_registered_models(self, process: str) -> None:
        row = PHYSICS_MODELS[process]
        assert row.auto_clinical in row.models
        assert row.auto_high_energy in row.models

    def test_a_row_refuses_an_unregistered_auto_choice(self) -> None:
        with pytest.raises(ValueError, match="not a registered model"):
            ProcessModels(models=("a",), auto_clinical="a", auto_high_energy="b")

    def test_a_row_refuses_auto_as_a_model_name(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            ProcessModels(models=("auto",), auto_clinical="auto", auto_high_energy="auto")

    def test_the_table_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            PHYSICS_MODELS["pair"] = PHYSICS_MODELS["pair"]  # type: ignore[index]

    def test_the_threshold_is_the_public_provisional_constant(self) -> None:
        assert pyradmc.HIGH_ENERGY_PHYSICS_THRESHOLD_MEV == 30.0


class TestPhysicsOptions:
    def test_defaults_are_auto(self) -> None:
        assert PhysicsOptions() == PhysicsOptions("auto", "auto", "auto")

    def test_is_frozen_and_hashable(self) -> None:
        options = PhysicsOptions()
        with pytest.raises(dataclasses.FrozenInstanceError):
            options.pair = "uniform-split"  # type: ignore[misc]
        assert hash(options) == hash(PhysicsOptions())

    def test_accepts_every_registered_model(self) -> None:
        options = PhysicsOptions(
            bremsstrahlung="thin-target", pair="uniform-split", positron="moller-at-rest"
        )
        assert options.bremsstrahlung == "thin-target"
        assert PhysicsOptions(bremsstrahlung="bethe-heitler").bremsstrahlung == "bethe-heitler"

    @pytest.mark.parametrize("process", PROCESSES)
    def test_unknown_model_lists_the_valid_ones(self, process: str) -> None:
        valid = PHYSICS_MODELS[process].models[0]
        with pytest.raises(ValueError, match=f"unknown {process} model 'nope'.*'auto'.*{valid}"):
            PhysicsOptions(**{process: "nope"})

    def test_a_model_of_another_process_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown pair model 'thin-target'"):
            PhysicsOptions(pair="thin-target")

    def test_is_public(self) -> None:
        assert pyradmc.PhysicsOptions is PhysicsOptions


class TestResolvedPhysics:
    def test_refuses_auto(self) -> None:
        with pytest.raises(ValueError, match="unknown pair model 'auto'"):
            ResolvedPhysics(bremsstrahlung="thin-target", pair="auto", positron="moller-at-rest")

    def test_reason_does_not_enter_identity(self) -> None:
        """The models are the configuration; why they were chosen is commentary."""
        a = dataclasses.replace(CLINICAL_PHYSICS, reason="one")
        b = dataclasses.replace(CLINICAL_PHYSICS, reason="another")
        assert a == b
        assert hash(a) == hash(b)

    def test_clinical_physics_is_the_below_threshold_row(self) -> None:
        for process in PROCESSES:
            assert getattr(CLINICAL_PHYSICS, process) == PHYSICS_MODELS[process].auto_clinical

    def test_summary_names_every_process(self) -> None:
        summary = CLINICAL_PHYSICS.summary()
        assert "bremsstrahlung=thin-target" in summary
        assert "pair=uniform-split" in summary
        assert "positron=moller-at-rest" in summary


class TestResolveToday:
    """What "auto" resolves to with the shipped table."""

    @pytest.mark.parametrize("energy", [0.5, 6.0, 20.0, 30.0])
    def test_auto_is_the_clinical_set_up_to_the_threshold(self, energy: float) -> None:
        resolved = resolve_physics("auto", max_energy_mev=energy, primary_kinds=PHOTONS)
        assert resolved == CLINICAL_PHYSICS
        assert resolved.bremsstrahlung == "thin-target"

    @pytest.mark.parametrize("energy", [30.000001, 50.0, 200.0])
    def test_auto_takes_bethe_heitler_above_the_threshold(self, energy: float) -> None:
        """Strictly above: a source at exactly the threshold stays clinical."""
        resolved = resolve_physics("auto", max_energy_mev=energy, primary_kinds=PHOTONS)
        assert resolved == dataclasses.replace(CLINICAL_PHYSICS, bremsstrahlung="bethe-heitler")
        assert "above the 30 MeV" in resolved.reason

    @pytest.mark.parametrize("kinds", [{"photon"}, {"electron"}, {"photon", "positron"}])
    def test_the_switch_does_not_depend_on_the_primary_kind(self, kinds: set[str]) -> None:
        below = resolve_physics("auto", max_energy_mev=18.0, primary_kinds=kinds)
        above = resolve_physics("auto", max_energy_mev=40.0, primary_kinds=kinds)
        assert below.bremsstrahlung == "thin-target"
        assert above.bremsstrahlung == "bethe-heitler"

    @pytest.mark.parametrize("energy", [6.0, 40.0])
    @pytest.mark.parametrize("model", ["thin-target", "bethe-heitler"])
    def test_an_explicit_model_overrides_the_threshold(self, energy: float, model: str) -> None:
        resolved = resolve_physics(
            PhysicsOptions(bremsstrahlung=model), max_energy_mev=energy, primary_kinds=PHOTONS
        )
        assert resolved.bremsstrahlung == model
        assert f"bremsstrahlung={model} set explicitly" in resolved.reason

    def test_default_argument_is_auto(self) -> None:
        assert resolve_physics(max_energy_mev=6.0, primary_kinds=PHOTONS) == resolve_physics(
            PhysicsOptions(), max_energy_mev=6.0, primary_kinds=PHOTONS
        )

    def test_explicit_current_models_equal_auto(self) -> None:
        explicit = PhysicsOptions(
            bremsstrahlung="thin-target", pair="uniform-split", positron="moller-at-rest"
        )
        assert resolve_physics(
            explicit, max_energy_mev=6.0, primary_kinds=PHOTONS
        ) == resolve_physics("auto", max_energy_mev=6.0, primary_kinds=PHOTONS)

    def test_result_is_hashable_and_fully_resolved(self) -> None:
        resolved = resolve_physics("auto", max_energy_mev=6.0, primary_kinds=PHOTONS)
        assert isinstance(hash(resolved), int)
        for process in PROCESSES:
            assert getattr(resolved, process) in PHYSICS_MODELS[process].models

    def test_is_pure(self) -> None:
        first = resolve_physics("auto", max_energy_mev=6.0, primary_kinds=PHOTONS)
        second = resolve_physics("auto", max_energy_mev=6.0, primary_kinds=PHOTONS)
        assert first == second
        assert first.reason == second.reason


class TestResolveValidation:
    def test_unknown_preset_string_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"unknown physics preset 'fancy'.*'auto'"):
            resolve_physics("fancy", max_energy_mev=6.0, primary_kinds=PHOTONS)

    def test_a_model_name_is_not_a_preset(self) -> None:
        """A bare model name is ambiguous across processes; it needs PhysicsOptions."""
        with pytest.raises(ValueError, match="PhysicsOptions"):
            resolve_physics("thin-target", max_energy_mev=6.0, primary_kinds=PHOTONS)

    def test_wrong_type_is_refused(self) -> None:
        with pytest.raises(TypeError, match="PhysicsOptions"):
            resolve_physics(None, max_energy_mev=6.0, primary_kinds=PHOTONS)  # type: ignore[arg-type]

    @pytest.mark.parametrize("energy", [0.0, -1.0, math.inf, math.nan])
    def test_non_physical_energy_is_refused(self, energy: float) -> None:
        with pytest.raises(ValueError, match="max_energy_mev"):
            resolve_physics("auto", max_energy_mev=energy, primary_kinds=PHOTONS)

    def test_empty_kinds_are_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one primary kind"):
            resolve_physics("auto", max_energy_mev=6.0, primary_kinds=frozenset())

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"unknown primary kind.*'proton'"):
            resolve_physics("auto", max_energy_mev=6.0, primary_kinds={"photon", "proton"})


@pytest.fixture
def two_model_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a second model per process, chosen above the threshold.

    Exactly the one-table edit the next branches make. Patched rather than shipped:
    a registered name must have an implementation on both backends (AGENTS.md 2.10).
    """
    table = {
        "bremsstrahlung": ProcessModels(
            models=("thin-target", "brems-hi"),
            auto_clinical="thin-target",
            auto_high_energy="brems-hi",
        ),
        "pair": ProcessModels(
            models=("uniform-split", "pair-hi"),
            auto_clinical="uniform-split",
            auto_high_energy="pair-hi",
        ),
        "positron": ProcessModels(
            models=("moller-at-rest", "positron-hi"),
            auto_clinical="moller-at-rest",
            auto_high_energy="positron-hi",
        ),
    }
    monkeypatch.setattr(physics_options, "PHYSICS_MODELS", table)


@pytest.mark.usefixtures("two_model_table")
class TestThresholdLogic:
    """The resolution rule itself, against a table with something to choose."""

    def test_at_or_below_threshold_is_clinical(self) -> None:
        for energy in (6.0, pyradmc.HIGH_ENERGY_PHYSICS_THRESHOLD_MEV):
            resolved = resolve_physics("auto", max_energy_mev=energy, primary_kinds=PHOTONS)
            assert (resolved.bremsstrahlung, resolved.pair, resolved.positron) == (
                "thin-target",
                "uniform-split",
                "moller-at-rest",
            )
            assert "at or below" in resolved.reason

    def test_above_threshold_is_high_energy(self) -> None:
        energy = math.nextafter(pyradmc.HIGH_ENERGY_PHYSICS_THRESHOLD_MEV, math.inf)
        resolved = resolve_physics("auto", max_energy_mev=energy, primary_kinds={"electron"})
        assert (resolved.bremsstrahlung, resolved.pair, resolved.positron) == (
            "brems-hi",
            "pair-hi",
            "positron-hi",
        )
        assert "above" in resolved.reason
        assert "electron" in resolved.reason

    def test_explicit_field_overrides_auto_per_process(self) -> None:
        """An override pins its own process and leaves the others on auto."""
        resolved = resolve_physics(
            PhysicsOptions(pair="uniform-split"), max_energy_mev=200.0, primary_kinds=PHOTONS
        )
        assert resolved.bremsstrahlung == "brems-hi"
        assert resolved.pair == "uniform-split"
        assert resolved.positron == "positron-hi"
        assert "pair=uniform-split set explicitly" in resolved.reason

    def test_explicit_high_energy_model_below_threshold(self) -> None:
        resolved = resolve_physics(
            PhysicsOptions(bremsstrahlung="brems-hi"), max_energy_mev=6.0, primary_kinds=PHOTONS
        )
        assert resolved.bremsstrahlung == "brems-hi"
        assert resolved.pair == "uniform-split"

    def test_threshold_is_read_at_call_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(physics_options, "HIGH_ENERGY_PHYSICS_THRESHOLD_MEV", 10.0)
        resolved = resolve_physics("auto", max_energy_mev=15.0, primary_kinds=PHOTONS)
        assert resolved.bremsstrahlung == "brems-hi"


class TestRunPrimaryKinds:
    def test_an_undeclared_source_takes_the_engine_fallback(self) -> None:
        assert run_primary_kinds(object(), "electron") == frozenset({"electron"})

    def test_a_declaring_source_wins_over_the_fallback(self) -> None:
        class Declares:
            particle_kinds = frozenset({"photon", "positron"})

        assert run_primary_kinds(Declares(), "electron") == frozenset({"photon", "positron"})

    def test_an_explicit_none_declaration_takes_the_fallback(self) -> None:
        class Defers:
            particle_kinds = None

        assert run_primary_kinds(Defers(), "photon") == PHOTONS


class TestImplementationGuard:
    def test_an_implemented_model_passes(self) -> None:
        require_implemented("pair", "uniform-split", ("uniform-split",), "a loop")

    def test_an_unimplemented_model_names_itself_and_the_site(self) -> None:
        with pytest.raises(NotImplementedError, match=r"pair model 'x'.*in a loop.*'uniform-s"):
            require_implemented("pair", "x", ("uniform-split",), "a loop")

    def test_reference_loops_implement_exactly_the_registered_models(self) -> None:
        """Registering a name without implementing it in the oracle fails here."""
        from pyradmc.transport import electron, photon

        assert PHYSICS_MODELS["bremsstrahlung"].models == electron.BREMSSTRAHLUNG_MODELS
        assert PHYSICS_MODELS["pair"].models == photon.PAIR_MODELS
        assert PHYSICS_MODELS["positron"].models == electron.POSITRON_MODELS


def test_resolution_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    """The selector is pure: logging the choice is the engine's job, once per run."""
    with caplog.at_level(logging.DEBUG, logger="pyradmc"):
        resolve_physics("auto", max_energy_mev=6.0, primary_kinds=PHOTONS)
    assert caplog.records == []
