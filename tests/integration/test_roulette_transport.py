"""Russian roulette in the transport loops: unbiased, active, exactly booked.

Roulette is always on (one configuration, AGENTS.md 2.10); the roulette-free
comparison runs exist only as a test instrument, produced by pointing the
transport modules' threshold constant at zero via monkeypatching. The oracle for
unbiasedness is the batched chi-squared detection test of AGENTS.md 4, on the
reference backend, where any bias would be a physics bug and not a float32
artifact.

The energy books must stay *exact*, not exact-in-expectation: every roulette
game books its weight-energy change (killed energy in, survivor boost out)
through the escaped-energy ledger, so emitted = deposited + escaped holds per
run to float64 arithmetic, roulette or not.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyradmc.transport.electron as electron_mod
import pyradmc.transport.photon as photon_mod
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched


def _engine() -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(2.0, 2.0, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())


def _source() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(2.0, 14.0), y_range=(2.0, 14.0))


def _disable_roulette(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test instrument: a zero threshold means no photon ever plays."""
    monkeypatch.setattr(photon_mod, "PHOTON_ROULETTE_MEV", 0.0)
    monkeypatch.setattr(electron_mod, "PHOTON_ROULETTE_MEV", 0.0)


class TestUnbiasedness:
    def test_dose_matches_roulette_free_transport(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Roulette on vs off, independent seeds: chi-squared consistent."""
        engine = _engine()
        with_rr = engine.run(_source(), n_histories=2_400, n_batches=8, seed=SEED)
        _disable_roulette(monkeypatch)
        without_rr = engine.run(_source(), n_histories=2_400, n_batches=8, seed=SEED + 1)
        mask = without_rr.dose > 0.1 * without_rr.dose.max()
        mask &= (with_rr.dose_sigma > 0.0) & (without_rr.dose_sigma > 0.0)
        assert np.count_nonzero(mask) > 200
        assert_chi2_consistent_batched(
            with_rr.dose,
            with_rr.dose_sigma,
            with_rr.n_batches,
            without_rr.dose,
            without_rr.dose_sigma,
            without_rr.n_batches,
            mask=mask,
        )

    def test_roulette_actually_fires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same seed, roulette on vs off, must diverge — otherwise the test above
        certifies nothing."""
        engine = _engine()
        with_rr = engine.run(_source(), n_histories=400, n_batches=1, seed=SEED)
        _disable_roulette(monkeypatch)
        without_rr = engine.run(_source(), n_histories=400, n_batches=1, seed=SEED)
        assert not np.array_equal(with_rr.dose, without_rr.dose)


class TestExactBooks:
    @pytest.mark.parametrize("transport_electrons", [False, True])
    def test_emitted_equals_deposited_plus_escaped(self, transport_electrons: bool) -> None:
        """The ledger identity survives roulette exactly (float64 reference)."""
        result = _engine().run(
            _source(),
            n_histories=600,
            n_batches=2,
            seed=SEED,
            transport_electrons=transport_electrons,
        )
        assert result.energy_deposited + result.energy_escaped == pytest.approx(
            result.energy_emitted, rel=1.0e-12
        )
