"""``Source.particle_kinds``: what a run starts, readable without sampling it.

The ``"auto"`` physics selection reads the run configuration before any history
starts (:mod:`pyradmc.physics_options`). A beam source leaves the kind to the
engine's ``primary_kind`` and declares ``None``; a source that sets
``Primary.kind`` per record declares the kinds it holds, and wrappers pass the
declaration through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyradmc.geometry.collimation import TransmissionMaskSource
from pyradmc.geometry.phasespace import InMemoryPhaseSpaceSource, PhaseSpaceSource
from pyradmc.geometry.source import CompositeSource, PencilBeamSource, Primary, Source
from pyradmc.physics_options import run_primary_kinds
from tests.phsp_fixtures import Rec, write_phsp


def _pencil() -> PencilBeamSource:
    return PencilBeamSource(6.0, (4.0, 4.0, -1.0), (0.0, 0.0, 1.0))


def _in_memory(codes: list[int]) -> InMemoryPhaseSpaceSource:
    n = len(codes)
    return InMemoryPhaseSpaceSource(
        particle_type=np.array(codes, dtype=np.int32),
        energy=np.full(n, 2.0),
        x=np.full(n, 4.0),
        y=np.full(n, 4.0),
        z=np.zeros(n),
        ux=np.zeros(n),
        uy=np.zeros(n),
        uz=np.ones(n),
        weight=np.ones(n),
    )


class _UserSource(Source):
    """A user subclass that predates the declaration and implements only the ABC."""

    @property
    def max_energy(self) -> float:
        return 5.0

    def emit(self, rng_state: object) -> Primary:
        return Primary(5.0, 4.0, 4.0, -1.0, 0.0, 0.0, 1.0, kind="electron")


def test_beam_sources_leave_the_kind_to_the_engine() -> None:
    assert _pencil().particle_kinds is None
    assert run_primary_kinds(_pencil(), "electron") == frozenset({"electron"})


def test_declaring_is_optional_for_a_user_source() -> None:
    """The property is not abstract: existing subclasses keep working unchanged."""
    assert _UserSource().particle_kinds is None


def test_in_memory_phase_space_declares_the_kinds_it_holds() -> None:
    assert _in_memory([1, 1, 2]).particle_kinds == frozenset({"photon", "electron"})
    assert _in_memory([3]).particle_kinds == frozenset({"positron"})


def test_phase_space_file_declares_its_transportable_kinds(tmp_path: Path) -> None:
    write_phsp(tmp_path / "s", [Rec(1, 6.0), Rec(3, 1.0), Rec(4, 2.0)])
    with pytest.warns(UserWarning, match="unsupported particle types"):
        source = PhaseSpaceSource(tmp_path / "s")
    # The neutron is dropped from the sampled population, so it is not declared.
    assert source.particle_kinds == frozenset({"photon", "positron"})


def test_composite_declares_the_union() -> None:
    mixture = CompositeSource([(_in_memory([1]), 0.5), (_in_memory([2, 3]), 0.5)])
    assert mixture.particle_kinds == frozenset({"photon", "electron", "positron"})


def test_composite_with_an_undeclared_component_declares_nothing() -> None:
    """One engine-decided component makes the mixture's kinds engine-dependent."""
    mixture = CompositeSource([(_in_memory([2]), 0.5), (_pencil(), 0.5)])
    assert mixture.particle_kinds is None


def _wrapped(inner: Source) -> TransmissionMaskSource:
    return TransmissionMaskSource(
        inner, mask=np.ones((2, 2)), plane_center=(4.0, 4.0, 0.0), width_u=4.0, width_v=4.0
    )


def test_weight_wrapper_passes_the_declaration_through() -> None:
    assert _wrapped(_in_memory([1, 2])).particle_kinds == frozenset({"photon", "electron"})
    assert _wrapped(_pencil()).particle_kinds is None


class TestRunKindsOfAMixture:
    """A mixture's run kinds: declared components as declared, the rest the fallback.

    ``particle_kinds`` of a mixture with an engine-decided component is ``None``,
    because the source cannot know ``primary_kind``. The kinds the run starts are
    still knowable once the engine supplies it, and must keep what the declaring
    components hold: a positron phase space mixed with a beam starts positrons.
    """

    def test_a_declared_positron_survives_an_undeclared_beam(self) -> None:
        mixture = CompositeSource([(_in_memory([3]), 0.5), (_pencil(), 0.5)])
        assert run_primary_kinds(mixture, "photon") == frozenset({"photon", "positron"})
        assert run_primary_kinds(mixture, "electron") == frozenset({"electron", "positron"})

    def test_the_source_resolves_its_own_fallback(self) -> None:
        mixture = CompositeSource([(_in_memory([2, 3]), 0.5), (_pencil(), 0.5)])
        assert mixture.primary_kinds("photon") == frozenset({"photon", "electron", "positron"})

    def test_a_single_source_resolves_like_run_primary_kinds(self) -> None:
        assert _pencil().primary_kinds("electron") == frozenset({"electron"})
        assert _UserSource().primary_kinds("photon") == frozenset({"photon"})
        assert _in_memory([3]).primary_kinds("photon") == frozenset({"positron"})

    def test_nested_mixtures_resolve_recursively(self) -> None:
        inner = CompositeSource([(_in_memory([3]), 0.5), (_pencil(), 0.5)])
        outer = CompositeSource([(inner, 0.5), (_in_memory([2]), 0.5)])
        assert outer.particle_kinds is None
        assert run_primary_kinds(outer, "photon") == frozenset({"photon", "electron", "positron"})

    def test_a_wrapped_mixture_resolves_through_the_wrapper(self) -> None:
        mixture = CompositeSource([(_in_memory([3]), 0.5), (_pencil(), 0.5)])
        assert run_primary_kinds(_wrapped(mixture), "photon") == frozenset({"photon", "positron"})
        inside = CompositeSource([(_wrapped(mixture), 0.5), (_pencil(), 0.5)])
        assert run_primary_kinds(inside, "electron") == frozenset({"electron", "positron"})

    def test_a_fully_declared_mixture_ignores_the_fallback(self) -> None:
        mixture = CompositeSource([(_in_memory([1]), 0.5), (_in_memory([3]), 0.5)])
        assert run_primary_kinds(mixture, "electron") == frozenset({"photon", "positron"})

    def test_an_undeclared_mixture_starts_the_fallback_only(self) -> None:
        """The clinical case: beams only, the engine's ``primary_kind`` alone."""
        mixture = CompositeSource([(_pencil(), 0.5), (_wrapped(_pencil()), 0.5)])
        assert run_primary_kinds(mixture, "photon") == frozenset({"photon"})
        assert run_primary_kinds(_wrapped(mixture), "electron") == frozenset({"electron"})
