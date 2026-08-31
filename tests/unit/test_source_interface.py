"""The public Source / BeamletSource interface.

A user can define a source by subclassing ``Source`` (open field) or ``BeamletSource``
(Dij) and implementing ``emit`` + ``max_energy``; the reference backend runs it through
``emit``, and the default ``sample_batch`` gives any such source the host pre-sampling
columns a device backend uploads (the "simple" GPU route). This slice pins the interface
and the reference path; the Warp routes land in later slices.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.phasespace import PhaseSpaceSource
from pyradmc.geometry.source import (
    BeamletGridSource,
    BeamletSource,
    ParallelBeamSource,
    PencilBeamSource,
    Primary,
    Source,
)
from pyradmc.rng.host import HostRNG, uniform
from tests.conftest import SEED


class JitterElectronSource(Source):
    """Minimal user source: a 5 MeV electron with a one-draw lateral jitter in x.

    Draws exactly one uniform so the default ``sample_batch`` has a stream to match,
    and emits an electron so the particle-type mapping is exercised.
    """

    energy = 5.0

    @property
    def max_energy(self) -> float:
        return self.energy

    def emit(self, rng_state: np.random.Generator) -> Primary:
        x = 4.0 + uniform(rng_state)
        return Primary(self.energy, x, 4.0, -1.0, 0.0, 0.0, 1.0, kind="electron")


class TestBuiltinsImplementInterface:
    def test_open_field_sources_are_sources(self) -> None:
        assert issubclass(PencilBeamSource, Source)
        assert issubclass(ParallelBeamSource, Source)
        assert issubclass(PhaseSpaceSource, Source)

    def test_beamlet_source_is_a_beamlet_source(self) -> None:
        assert issubclass(BeamletGridSource, BeamletSource)
        # A Dij source is not an open-field Source (its emit takes a beamlet index).
        assert not issubclass(BeamletGridSource, Source)

    def test_max_energy_is_the_beam_energy_for_mono_sources(self) -> None:
        assert PencilBeamSource(6.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)).max_energy == 6.0
        assert ParallelBeamSource(6.0, -1.0, (0.0, 1.0), (0.0, 1.0)).max_energy == 6.0
        assert BeamletGridSource(6.0, -1.0, (0.0, 2.0), (0.0, 2.0), 2, 2).max_energy == 6.0


class TestDefaultSampleBatch:
    def test_columns_match_emit_history_for_history(self) -> None:
        """The default sample_batch reproduces emit on the same per-history streams."""
        source = JitterElectronSource()
        n = 64
        batch = source.sample_batch(SEED, 0, n)
        rng = HostRNG()
        for i in range(n):
            expected = source.emit(rng.init_state(SEED, i))
            assert batch["x"][i] == pytest.approx(expected.x, rel=1e-6)
            assert batch["energy"][i] == pytest.approx(expected.energy)
            assert batch["weight"][i] == pytest.approx(expected.weight)

    def test_offset_indexes_the_history_stream(self) -> None:
        source = JitterElectronSource()
        rng = HostRNG()
        batch = source.sample_batch(SEED, 100, 8)
        assert batch["x"][0] == pytest.approx(source.emit(rng.init_state(SEED, 100)).x, rel=1e-6)

    def test_particle_type_follows_the_primary_kind(self) -> None:
        """kind='electron' -> IAEA code 2; the beam sources' None -> photon (1)."""
        assert np.all(JitterElectronSource().sample_batch(SEED, 0, 16)["particle_type"] == 2)
        parallel = ParallelBeamSource(6.0, -1.0, (0.0, 1.0), (0.0, 1.0))
        assert np.all(parallel.sample_batch(SEED, 0, 16)["particle_type"] == 1)

    def test_batch_has_all_upload_columns_as_float32(self) -> None:
        batch = JitterElectronSource().sample_batch(SEED, 0, 4)
        assert set(batch) == {
            "particle_type",
            "energy",
            "x",
            "y",
            "z",
            "ux",
            "uy",
            "uz",
            "weight",
        }
        for name in ("energy", "x", "y", "z", "ux", "uy", "uz", "weight"):
            assert batch[name].dtype == np.float32
        assert batch["particle_type"].dtype.kind == "i"


def test_custom_source_runs_on_the_reference_engine() -> None:
    """A user Source transports on ref via emit, with the energy ledger closing."""
    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(0.5, 0.5, 0.5))
    xs = AnalyticCrossSections()
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    result = engine.run(JitterElectronSource(), n_histories=500, n_batches=5, seed=SEED)
    assert result.dose.sum() > 0.0
    assert result.energy_emitted == pytest.approx(
        result.energy_deposited + result.energy_escaped, rel=1e-9
    )
