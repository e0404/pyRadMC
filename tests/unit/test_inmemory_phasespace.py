"""The in-memory phase-space source: column arrays instead of an IAEA file.

The treatment-head pre-solve's output container (BLD workstream): the same
sampling contract as :class:`PhaseSpaceSource` — one uniform per ``emit``,
chunk-invariant vectorized ``sample_batch`` on its own PCG64 stream, per-particle
``kind`` and ``weight`` — but backed by arrays built in memory, no file.

Both phase-space classes also gain the finite-reuse **result caveat**: sampling
with replacement from N stored particles, the per-batch sigmas never see the
latent variance of the finite phase space; once the histories drawn exceed N the
reuse is certain, so a single ``warnings.warn`` fires there (a pragmatic
tripwire, not a claim that fewer histories are latent-variance-free — the
docstring says so).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.geometry.phasespace import InMemoryPhaseSpaceSource
from pyRadMC.rng.host import HostRNG, uniform
from tests.conftest import SEED


def _columns(n: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260715)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return {
        "particle_type": np.tile([1, 2], n // 2).astype(np.int32),
        "energy": rng.uniform(0.5, 6.0, n),
        "x": rng.uniform(0.0, 16.0, n),
        "y": rng.uniform(0.0, 16.0, n),
        "z": np.zeros(n),
        "ux": directions[:, 0],
        "uy": directions[:, 1],
        "uz": directions[:, 2],
        "weight": rng.uniform(0.1, 1.0, n),
    }


def _source(n: int = 8, **overrides: np.ndarray) -> InMemoryPhaseSpaceSource:
    columns = _columns(n)
    columns.update(overrides)
    return InMemoryPhaseSpaceSource(**columns)


class TestConstruction:
    def test_len_and_max_energy(self) -> None:
        columns = _columns()
        source = _source()
        assert len(source) == 8
        assert source.max_energy == columns["energy"].max()

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="length"):
            _source(energy=np.ones(3))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="particle"):
            _source(0)

    def test_unsupported_codes_raise(self) -> None:
        with pytest.raises(ValueError, match="IAEA"):
            _source(particle_type=np.full(8, 4, dtype=np.int32))  # neutrons

    def test_non_positive_energy_raises(self) -> None:
        bad = _columns()["energy"]
        bad[3] = 0.0
        with pytest.raises(ValueError, match="energy"):
            _source(energy=bad)

    def test_non_unit_direction_raises(self) -> None:
        bad = _columns()["ux"].copy()
        bad[2] += 0.1
        with pytest.raises(ValueError, match="unit"):
            _source(ux=bad)

    def test_negative_weight_raises(self) -> None:
        bad = _columns()["weight"].copy()
        bad[1] = -0.5
        with pytest.raises(ValueError, match="weight"):
            _source(weight=bad)


class TestEmit:
    def test_emit_consumes_exactly_one_uniform(self) -> None:
        rng = HostRNG()
        source = _source()
        state = rng.init_state(SEED, 4)
        source.emit(state)
        reference = rng.init_state(SEED, 4)
        uniform(reference)
        assert uniform(state) == uniform(reference)

    def test_emit_returns_the_stored_particle(self) -> None:
        """The emitted primary is one of the stored rows, kind mapped from IAEA."""
        columns = _columns(64)
        source = _source(64)
        rng = HostRNG()
        for i in range(32):
            p = source.emit(rng.init_state(SEED, i))
            k = int(np.flatnonzero(columns["energy"] == p.energy)[0])
            assert p.kind == ("photon" if columns["particle_type"][k] == 1 else "electron")
            assert p.weight == columns["weight"][k]
            assert (p.x, p.y, p.z) == (columns["x"][k], columns["y"][k], 0.0)


class TestSampleBatch:
    def test_chunk_invariance(self) -> None:
        source = _source(64)
        whole = source.sample_batch(SEED, 0, 48)
        parts = [source.sample_batch(SEED, 0, 16), source.sample_batch(SEED, 16, 32)]
        for name in whole:
            np.testing.assert_array_equal(
                whole[name], np.concatenate([p[name] for p in parts]), err_msg=name
            )

    def test_batch_rows_are_stored_rows(self) -> None:
        columns = _columns(64)
        batch = _source(64).sample_batch(SEED, 0, 64)
        stored_energies = columns["energy"].astype(np.float32)
        assert np.all(np.isin(batch["energy"], stored_energies))
        assert set(np.unique(batch["particle_type"])) <= {1, 2}


class TestReuseWarning:
    def test_sampling_past_the_population_warns_once(self) -> None:
        source = _source(8)
        with pytest.warns(UserWarning, match="latent"):
            source.sample_batch(SEED, 0, 9)
        # Only once per source: the caveat is per run, not per chunk.
        import warnings as warnings_module

        with warnings_module.catch_warnings():
            warnings_module.simplefilter("error")
            source.sample_batch(SEED, 9, 8)

    def test_emit_counts_histories_toward_the_tripwire(self) -> None:
        source = _source(8)
        rng = HostRNG()
        with pytest.warns(UserWarning, match="latent"):
            for i in range(9):
                source.emit(rng.init_state(SEED, i))

    def test_within_population_is_silent(self) -> None:
        import warnings as warnings_module

        source = _source(8)
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("error")
            source.sample_batch(SEED, 0, 8)
