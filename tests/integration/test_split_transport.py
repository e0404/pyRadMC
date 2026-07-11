"""Compton splitting in the transport loop: unbiased, active, exactly booked.

Splitting samples a *primary* photon's first Compton final state
``PHOTON_SPLIT_N`` times, each copy (scattered photon + recoil electron)
carrying weight ``1 / PHOTON_SPLIT_N``. It is always on (one configuration,
AGENTS.md 2.10); the split-free comparison runs exist only as a test
instrument, produced by pointing the transport module's multiplicity constant
at one via monkeypatching. The oracle for unbiasedness is the batched
chi-squared detection test of AGENTS.md 4, on the reference backend, where any
bias would be a physics bug and not a float32 artifact.

Energy is conserved *per realization*, not just in expectation: each copy
carries ``w/N`` of both the scattered photon and its recoil electron, which sum
to the incident weight-energy exactly regardless of the sampled ratios, so
emitted = deposited + escaped holds per run to float64 arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyRadMC.transport.photon as photon_mod
from pyRadMC import PCUT_MEV, PHOTON_SPLIT_N
from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource
from pyRadMC.rng.host import HostRNG
from pyRadMC.transport.photon import photon_steps
from tests.conftest import SEED, assert_chi2_consistent_batched


def _engine() -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(2.0, 2.0, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())


def _source() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(2.0, 14.0), y_range=(2.0, 14.0))


def _disable_splitting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test instrument: multiplicity one means no photon ever splits."""
    monkeypatch.setattr(photon_mod, "PHOTON_SPLIT_N", 1)


class TestUnbiasedness:
    def test_dose_matches_split_free_transport(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Splitting on vs off, independent seeds: chi-squared consistent."""
        engine = _engine()
        with_split = engine.run(_source(), n_histories=2_400, n_batches=8, seed=SEED)
        _disable_splitting(monkeypatch)
        without_split = engine.run(_source(), n_histories=2_400, n_batches=8, seed=SEED + 1)
        mask = without_split.dose > 0.1 * without_split.dose.max()
        mask &= (with_split.dose_sigma > 0.0) & (without_split.dose_sigma > 0.0)
        assert np.count_nonzero(mask) > 200
        assert_chi2_consistent_batched(
            with_split.dose,
            with_split.dose_sigma,
            with_split.n_batches,
            without_split.dose,
            without_split.dose_sigma,
            without_split.n_batches,
            mask=mask,
        )

    def test_splitting_actually_fires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same seed, splitting on vs off, must diverge — else the test above
        certifies nothing."""
        engine = _engine()
        with_split = engine.run(_source(), n_histories=400, n_batches=1, seed=SEED)
        _disable_splitting(monkeypatch)
        without_split = engine.run(_source(), n_histories=400, n_batches=1, seed=SEED)
        assert not np.array_equal(with_split.dose, without_split.dose)


class TestVarianceReduction:
    """Splitting must actually *reduce* variance, not merely stay unbiased.

    Unbiasedness (above) only says splitting does no harm; this says it does
    the good it exists for. At a matched primary count the split copies are N
    independent samples of the scattered final state, so the batch-estimated
    variance falls, most where scattered radiation contributes — the high-dose
    region. Both runs are on the reference at one seed, so the comparison is
    deterministic; the measured ratio is well clear of the asserted bound.
    """

    def test_high_dose_variance_falls_at_matched_histories(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _engine()
        split = engine.run(_source(), n_histories=2_400, n_batches=8, seed=SEED)
        _disable_splitting(monkeypatch)
        unsplit = engine.run(_source(), n_histories=2_400, n_batches=8, seed=SEED)
        high = unsplit.dose > 0.3 * unsplit.dose.max()
        assert np.count_nonzero(high) > 200
        var_ratio = (split.dose_sigma[high] ** 2).mean() / (unsplit.dose_sigma[high] ** 2).mean()
        # Measured ~0.69 at N=2 over this region; the bound leaves margin for the
        # variance estimate's own noise while still failing a no-op split.
        assert var_ratio < 0.85, (
            f"splitting did not reduce high-dose variance: ratio {var_ratio:.3f}"
        )


class TestExactBooks:
    @pytest.mark.parametrize("transport_electrons", [False, True])
    def test_emitted_equals_deposited_plus_escaped(self, transport_electrons: bool) -> None:
        """The ledger identity survives splitting exactly (float64 reference)."""
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


class TestFairCopies:
    """A primary Compton produces exactly N copies, each of weight w/N.

    Driven at the ``photon_steps`` level in KERMA mode so recoil electrons
    deposit and only scattered photons are spawned — the copies are then
    directly countable. The seed is one where a 1 MeV primary's first
    interaction is a Compton scatter leaving every copy above PCUT (found by
    the probe below; any such seed pins the same structural invariant).
    """

    def _transport_one(self, seed: int, is_primary: bool):
        grid = VoxelGrid.uniform_water(shape=(4, 4, 40), spacing=(2.0, 2.0, 0.5))
        xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
        state = HostRNG().init_state(4242, seed)
        spawned: list = []
        deposits: list = []
        escaped = photon_steps(
            1.0,
            1.0,
            4.0,
            4.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            grid,
            xs,
            state,
            lambda ix, iy, iz, en: deposits.append(en),
            spawned.append,
            PCUT_MEV,
            0.2,
            False,
            is_primary,
        )
        return spawned, deposits, escaped

    def test_primary_splits_into_n_fair_copies_conserving_energy(self) -> None:
        seed = _PRIMARY_COMPTON_SEED
        spawned, deposits, escaped = self._transport_one(seed, is_primary=True)
        photons = [s for s in spawned if s[0] == photon_mod.PHOTON]
        assert len(photons) == PHOTON_SPLIT_N
        for copy in photons:
            assert copy[2] == pytest.approx(1.0 / PHOTON_SPLIT_N, rel=1.0e-12)
        total = sum(p[1] * p[2] for p in spawned) + sum(deposits) + escaped
        assert total == pytest.approx(1.0, rel=1.0e-12)

    def test_same_photon_as_non_primary_does_not_split(self) -> None:
        """The identical stream, marked non-primary, yields one scattered photon."""
        spawned, _, _ = self._transport_one(_PRIMARY_COMPTON_SEED, is_primary=False)
        photons = [s for s in spawned if s[0] == photon_mod.PHOTON]
        assert len(photons) == 1
        assert photons[0][2] == pytest.approx(1.0, rel=1.0e-12)


# A seed whose 1 MeV primary Comptons first with all copies above PCUT (found by
# sweeping (4242, seed) for the first clean N-fold split; any such seed pins the
# same structural invariant).
_PRIMARY_COMPTON_SEED = 13
