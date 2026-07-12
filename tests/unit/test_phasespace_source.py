"""PhaseSpaceSource: sampling primaries from an IAEA phase-space file.

Phase 5, phase-space source, Slice 2. The source loads phsp records once and
:meth:`emit` draws one per history from the history's RNG stream (random sampling
with replacement), returning a :class:`Primary` that carries the record's particle
*kind* and statistical *weight* — unlike the monoenergetic beam sources, a phsp is
a mix of photons / electrons / positrons at non-unit weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyRadMC.geometry.phasespace import PhaseSpaceSource
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED
from tests.phsp_fixtures import Rec, write_phsp


def _rng_state(seed: int = SEED, history: int = 0):
    return HostRNG().init_state(seed, history)


class TestLoading:
    def test_len_counts_records(self, tmp_path: Path) -> None:
        write_phsp(tmp_path / "s", [Rec(1, 6.0), Rec(2, 1.0), Rec(1, 3.0)])
        src = PhaseSpaceSource(tmp_path / "s")
        assert len(src) == 3

    def test_unsupported_particle_skipped_by_default(self, tmp_path: Path) -> None:
        """A neutron/proton is dropped (with a warning), not fatal: real files carry them.

        The supported population is the two photons; the neutron between them is
        excluded, and sampling never returns it.
        """
        write_phsp(tmp_path / "s", [Rec(1, 6.0), Rec(4, 2.0), Rec(1, 3.0)])
        with pytest.warns(UserWarning, match="unsupported particle types"):
            src = PhaseSpaceSource(tmp_path / "s")
        assert len(src) == 2  # the neutron is excluded
        kinds = {src.emit(_rng_state(SEED, h)).kind for h in range(30)}
        assert kinds == {"photon"}  # never the neutron

    def test_unsupported_particle_strict_raises(self, tmp_path: Path) -> None:
        """skip_unsupported=False rejects a file with any untransportable particle."""
        write_phsp(tmp_path / "s", [Rec(1, 6.0), Rec(4, 2.0)])
        with pytest.raises(ValueError, match="unsupported particle type 4"):
            PhaseSpaceSource(tmp_path / "s", skip_unsupported=False)

    def test_skipped_records_are_never_sampled_and_energies_are_correct(
        self, tmp_path: Path
    ) -> None:
        """Streaming random access decodes the right record around skipped ones."""
        # Photons at energies 1..5 with a proton wedged at index 2.
        recs = [Rec(1, 1.0), Rec(1, 2.0), Rec(5, 9.0), Rec(1, 4.0), Rec(1, 5.0)]
        write_phsp(tmp_path / "s", recs)
        with pytest.warns(UserWarning):
            src = PhaseSpaceSource(tmp_path / "s")
        assert len(src) == 4
        energies = {round(src.emit(_rng_state(SEED, h)).energy, 3) for h in range(200)}
        assert energies == {1.0, 2.0, 4.0, 5.0}  # 9.0 (the proton) never appears


class TestEmit:
    def test_emit_carries_kind_and_weight(self, tmp_path: Path) -> None:
        """A single-record file emits that record's kind, weight, energy, geometry."""
        write_phsp(
            tmp_path / "s",
            [Rec(2, 1.5, x=1.0, y=2.0, z=3.0, u=0.6, v=0.0, weight=0.25)],
        )
        src = PhaseSpaceSource(tmp_path / "s")
        p = src.emit(_rng_state())
        assert p.kind == "electron"
        assert p.energy == pytest.approx(1.5)
        assert p.weight == pytest.approx(0.25)
        assert (p.x, p.y, p.z) == pytest.approx((1.0, 2.0, 3.0))
        assert p.uz == pytest.approx(0.8)  # w = +sqrt(1-0.36)

    def test_positron_maps_to_positron_kind(self, tmp_path: Path) -> None:
        write_phsp(tmp_path / "s", [Rec(3, 0.7)])
        p = PhaseSpaceSource(tmp_path / "s").emit(_rng_state())
        assert p.kind == "positron"

    def test_emit_is_a_pure_function_of_the_stream(self, tmp_path: Path) -> None:
        """Same stream -> same sampled record; the source draws from the RNG state."""
        write_phsp(tmp_path / "s", [Rec(1, e) for e in (1.0, 2.0, 3.0, 4.0, 5.0)])
        src = PhaseSpaceSource(tmp_path / "s")
        a = src.emit(_rng_state(SEED, 7))
        b = src.emit(_rng_state(SEED, 7))
        assert a == b

    def test_emit_samples_across_records(self, tmp_path: Path) -> None:
        """Different histories reach different records (sampling actually varies)."""
        write_phsp(tmp_path / "s", [Rec(1, float(i)) for i in range(1, 21)])
        src = PhaseSpaceSource(tmp_path / "s")
        energies = {src.emit(_rng_state(SEED, h)).energy for h in range(50)}
        assert len(energies) > 1
