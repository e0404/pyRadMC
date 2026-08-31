"""PhaseSpaceSource: sampling primaries from an IAEA phase-space file.

The source loads phsp records once and
:meth:`emit` draws one per history from the history's RNG stream (random sampling
with replacement), returning a :class:`Primary` that carries the record's particle
*kind* and statistical *weight* — unlike the monoenergetic beam sources, a phsp is
a mix of photons / electrons / positrons at non-unit weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyradmc.geometry.phasespace import PhaseSpaceSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED
from tests.phsp_fixtures import Rec, write_phsp

# These tests deliberately oversample tiny synthetic phase spaces; the finite-
# reuse latent-variance caveat is expected and acknowledged (pinned explicitly
# in the tripwire tests).
pytestmark = pytest.mark.filterwarnings("ignore:.*latent variance:UserWarning")


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


class TestSampleBatch:
    """The vectorized batch sampler used by the warp backend."""

    def _src(self, tmp_path: Path) -> PhaseSpaceSource:
        recs = [
            Rec(t, float(i + 1), x=float(i), y=-float(i), z=1.0, u=0.6, v=0.0, weight=0.5 + 0.1 * i)
            for i, t in enumerate([1, 2, 3, 1, 2, 1, 1, 3, 1, 2])
        ]
        write_phsp(tmp_path / "s", recs)
        return PhaseSpaceSource(tmp_path / "s")

    def test_chunk_invariant(self, tmp_path: Path) -> None:
        """History h draws the same record regardless of chunk boundaries."""
        src = self._src(tmp_path)
        whole = src.sample_batch(SEED, 0, 10)
        first = src.sample_batch(SEED, 0, 4)
        rest = src.sample_batch(SEED, 4, 6)
        for key in whole:
            joined = np.concatenate([first[key], rest[key]])
            np.testing.assert_array_equal(whole[key], joined)

    def test_decodes_valid_supported_records(self, tmp_path: Path) -> None:
        """Every sampled primary is a real, supported record with a unit direction."""
        src = self._src(tmp_path)
        b = src.sample_batch(SEED, 0, 200)
        assert set(np.unique(b["particle_type"])).issubset({1, 2, 3})
        norm = b["ux"] ** 2 + b["uy"] ** 2 + b["uz"] ** 2
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)
        # energies belong to the file's set {1..10}
        assert set(np.round(b["energy"]).astype(int)).issubset(set(range(1, 11)))

    def test_skips_unsupported_records(self, tmp_path: Path) -> None:
        """A proton in the file is never returned by the batch sampler."""
        write_phsp(tmp_path / "s", [Rec(1, 1.0), Rec(5, 9.0), Rec(1, 2.0), Rec(1, 3.0)])
        with pytest.warns(UserWarning):
            src = PhaseSpaceSource(tmp_path / "s")
        b = src.sample_batch(SEED, 0, 300)
        assert 9.0 not in set(np.round(b["energy"]))
        assert set(np.unique(b["particle_type"])) == {1}


class TestLatentVarianceTripwire:
    """The file-backed source carries the same finite-reuse caveat as the in-memory one."""

    def test_oversampling_the_file_warns_once(self, tmp_path: Path) -> None:
        recs = [Rec(1, 2.0, 1.0, 1.0, 0.0), Rec(1, 4.0, 2.0, 2.0, 0.0)]
        src = PhaseSpaceSource(write_phsp(tmp_path / "s", recs))
        with pytest.warns(UserWarning, match="latent variance"):
            src.sample_batch(SEED, 0, 3)
        # Once per source: later chunks stay silent.
        import warnings as warnings_module

        with warnings_module.catch_warnings():
            warnings_module.simplefilter("error")
            src.sample_batch(SEED, 3, 4)
