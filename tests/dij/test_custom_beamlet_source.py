"""A user BeamletSource drives the reference Dij (Phase 5 source interface).

The Dij engine asks a :class:`~pyRadMC.geometry.source.BeamletSource` only for
``n_beamlets`` and per-beamlet ``emit`` (beamlet *geometry* is the source's own
business), so a user can define an arbitrary beamlet layout. This pins that a custom
strip-lattice source assembles a correct Dij on the reference backend: energy
conservation is exact, and each beamlet's dose localizes under its strip. The default
``sample_beamlet_batch`` (used by the Warp Dij route in a later slice) is checked to
match ``emit``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletSource, Primary
from pyRadMC.rng.host import HostRNG, uniform
from tests.conftest import SEED

ENERGY = 6.0
Z0 = -1.0
FIELD_X = (2.0, 14.0)
FIELD_Y = (2.0, 14.0)


class StripBeamletSource(BeamletSource):
    """A user source: ``n`` vertical strips tiling a field, each a parallel sub-beam."""

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def max_energy(self) -> float:
        return ENERGY

    @property
    def n_beamlets(self) -> int:
        return self._n

    def emit(self, beamlet: int, rng_state: object) -> Primary:
        x_lo = FIELD_X[0] + (FIELD_X[1] - FIELD_X[0]) * beamlet / self._n
        x_hi = FIELD_X[0] + (FIELD_X[1] - FIELD_X[0]) * (beamlet + 1) / self._n
        x = x_lo + (x_hi - x_lo) * uniform(rng_state)
        y = FIELD_Y[0] + (FIELD_Y[1] - FIELD_Y[0]) * uniform(rng_state)
        return Primary(ENERGY, x, y, Z0, 0.0, 0.0, 1.0)


def _engine() -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    return ReferenceEngine(grid=grid, cross_sections=AnalyticCrossSections(), rng=HostRNG())


def test_custom_beamlet_source_conserves_energy() -> None:
    """The Dij ledger closes and books the source's total emitted energy exactly."""
    source = StripBeamletSource(4)
    dij = _engine().run_dij(source, n_histories_per_beamlet=150, n_batches=3, seed=SEED)
    assert dij.n_beamlets == 4
    assert dij.energy_emitted == pytest.approx(dij.energy_deposited + dij.energy_escaped, rel=1e-12)
    assert dij.energy_emitted == pytest.approx(4 * 150 * ENERGY, rel=1e-12)


def test_custom_beamlet_columns_localize_under_their_strip() -> None:
    """Each beamlet's dose sits mostly under its own x-strip, not a neighbour's."""
    source = StripBeamletSource(4)
    dij = _engine().run_dij(
        source, n_histories_per_beamlet=400, n_batches=4, seed=SEED, transport_electrons=False
    )
    nx = 16
    for beamlet in range(4):
        by_x = dij.column_dense(beamlet).sum(axis=(1, 2))
        strip = slice(beamlet * nx // 4, (beamlet + 1) * nx // 4)
        assert by_x[strip].sum() > 0.6 * by_x.sum(), f"beamlet {beamlet} not localized"


def test_default_sample_beamlet_batch_matches_emit() -> None:
    """The default pre-sampling columns reproduce emit for a given beamlet."""
    source = StripBeamletSource(3)
    rng = HostRNG()
    batch = source.sample_beamlet_batch(SEED, 0, 32, beamlet=2)
    for i in range(32):
        expected = source.emit(2, rng.init_state(SEED, i))
        assert batch["x"][i] == pytest.approx(expected.x, rel=1e-6)
        assert batch["y"][i] == pytest.approx(expected.y, rel=1e-6)
    assert np.all(batch["particle_type"] == 1)  # photons
