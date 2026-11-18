"""The ``"bethe-heitler"`` pair model in the reference photon loop.

The samplers are pinned on their own (``tests/physics/test_pair_sampling.py``) and
so is the screening data (``tests/unit/test_pair_screening.py``); these tests pin
the *wiring* of the model into :func:`pyradmc.transport.photon.photon_steps`, and
what distinguishes it from the uniform split at the level of the pair itself.

The instrument is one photon at a time in an effectively infinite block of water,
driven through the loop directly so the leptons it spawns can be tallied. The loop
returns at the photon's first interaction that is not coherent, and a pair is the
only one that spawns a positron. Leptons are spawned whatever their energy
(``ecut=0`` on the loop, a test instrument: the production threshold would hide
the soft end of the sharing), except where the threshold itself is the subject.

Every expectation is an interface quantity: the pair fraction from
``mu_over_rho``, the sharing from :func:`pyradmc.data.pair.pair_sharing_dcs` with
the source's ``pair_screening``, the angle from the lepton's own speed. Seeds are
fixed.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy import integrate, stats

from pyradmc import (
    ECUT_MEV,
    ELECTRON_MASS_MEV,
    PCUT_MEV,
    PHOTON_ROULETTE_WEIGHT_CAP,
    PhysicsOptions,
)
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.interface import PhotonProcess
from pyradmc.data.materials import WATER
from pyradmc.data.pair import pair_sharing_dcs
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource, Primary, Source
from pyradmc.physics_options import CLINICAL_PHYSICS, ResolvedPhysics
from pyradmc.rng import RNGState
from pyradmc.rng.host import HostRNG
from pyradmc.transport.particles import ELECTRON, PHOTON, POSITRON, StackEntry
from pyradmc.transport.photon import photon_steps
from tests.conftest import SEED
from tests.integration.test_bethe_heitler_ref import _WideAnalyticWater

BETHE_HEITLER = dataclasses.replace(CLINICAL_PHYSICS, pair="bethe-heitler")
ENERGY_MEV = 18.0
N_PHOTONS = 40_000
BLOCK_CM = 1.0e5  # nothing leaves


@dataclass(frozen=True)
class _PairTally:
    """The pairs made by ``n_photons`` photons at their first real interaction."""

    n_photons: int
    electron_energy: np.ndarray  # T-, MeV
    positron_energy: np.ndarray  # T+, MeV
    electron_cos: np.ndarray  # polar cosine about the photon direction
    positron_cos: np.ndarray
    relative_azimuth: np.ndarray  # phi+ - phi-, in [0, 2 pi)
    balance_error: float

    def __len__(self) -> int:
        return int(self.electron_energy.size)


def _first_interactions(
    physics: ResolvedPhysics,
    stream: int,
    energy: float = ENERGY_MEV,
    n: int = N_PHOTONS,
    ecut: float = 0.0,
    weight: float = 1.0,
) -> _PairTally:
    grid = VoxelGrid.uniform_water(shape=(1, 1, 1), spacing=(BLOCK_CM, BLOCK_CM, BLOCK_CM))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    rng = HostRNG()
    half = 0.5 * BLOCK_CM

    t_minus: list[float] = []
    t_plus: list[float] = []
    cos_minus: list[float] = []
    cos_plus: list[float] = []
    azimuth: list[float] = []
    worst = 0.0
    for i in range(n):
        spawned: list[StackEntry] = []
        deposited = [0.0]

        def deposit(x: float, y: float, z: float, amount: float, weighted: float) -> None:
            deposited[0] += amount  # noqa: B023 - consumed within this iteration

        escaped = photon_steps(
            energy, weight, half, half, half, 0.0, 0.0, 1.0,
            grid, xs, rng.init_state(SEED + stream, i), deposit, spawned.append,
            pcut=PCUT_MEV, ecut=ecut, transport_electrons=True, is_primary=True,
            physics=physics,
        )  # fmt: skip

        carried = 0.0
        leptons: dict[int, StackEntry] = {}
        for entry in spawned:
            kind, e, w = entry[:3]
            carried += w * (e + (2.0 * ELECTRON_MASS_MEV if kind == POSITRON else 0.0))
            if kind != PHOTON:
                leptons[kind] = entry
        worst = max(
            worst, abs(weight * energy - (deposited[0] + escaped + carried)) / (weight * energy)
        )
        if POSITRON not in leptons or ELECTRON not in leptons:
            continue
        electron, positron = leptons[ELECTRON], leptons[POSITRON]
        t_minus.append(electron[1])
        t_plus.append(positron[1])
        cos_minus.append(electron[8])
        cos_plus.append(positron[8])
        delta = math.atan2(positron[7], positron[6]) - math.atan2(electron[7], electron[6])
        azimuth.append(delta % (2.0 * math.pi))

    return _PairTally(
        n_photons=n,
        electron_energy=np.array(t_minus),
        positron_energy=np.array(t_plus),
        electron_cos=np.array(cos_minus),
        positron_cos=np.array(cos_plus),
        relative_azimuth=np.array(azimuth),
        balance_error=worst,
    )


@pytest.fixture(scope="module")
def bethe_heitler() -> _PairTally:
    return _first_interactions(BETHE_HEITLER, stream=1)


@pytest.fixture(scope="module")
def uniform_split() -> _PairTally:
    return _first_interactions(CLINICAL_PHYSICS, stream=2)


# -- expectations from the interface ------------------------------------------------


def _pair_fraction(energy: float) -> float:
    """Probability that a photon's first non-coherent interaction is a pair."""
    xs = AnalyticCrossSections()
    pair = xs.mu_over_rho(energy, WATER, PhotonProcess.PAIR)
    coherent = xs.mu_over_rho(energy, WATER, PhotonProcess.RAYLEIGH)
    return pair / (xs.mu_over_rho_total(energy, WATER) - coherent)


def _sharing_p_value(electron_energy: np.ndarray, energy: float, model: str) -> float:
    """Chi-squared p-value of the electron's kinetic energy against a sharing law.

    ``model`` is ``"bethe-heitler"`` (the screened DCS at the source's water
    screening) or ``"uniform"``. Twenty equal bins in kinetic energy.
    """
    kinetic = energy - 2.0 * ELECTRON_MASS_MEV
    edges = np.linspace(0.0, kinetic, 21)
    observed, _ = np.histogram(electron_energy, bins=edges)
    if model == "uniform":
        expected = np.full(20, observed.sum() / 20.0)
    else:
        screening = AnalyticCrossSections().pair_screening(WATER)
        expected = np.array(
            [
                integrate.quad(
                    lambda t: pair_sharing_dcs(energy, (t + ELECTRON_MASS_MEV) / energy, screening),
                    a,
                    b,
                )[0]
                for a, b in itertools.pairwise(edges)
            ]
        )
        expected *= observed.sum() / expected.sum()
    assert np.all(expected > 20.0), "bin layout leaves low-count bins; rebin"
    return float(stats.chisquare(observed, expected).pvalue)


def _angle_quantile(cos_theta: np.ndarray, kinetic: np.ndarray) -> np.ndarray:
    """Map each polar cosine through the CDF of ``(1 - beta mu)^-2`` at its own beta.

    Uniform on (0, 1) if and only if every lepton's angle follows the leading-order
    law for *its* energy: ``2 xi - 1 = (mu - beta) / (1 - beta mu)``.
    """
    tau = kinetic / ELECTRON_MASS_MEV
    beta = np.sqrt(tau * (tau + 2.0)) / (tau + 1.0)
    return 0.5 * (1.0 + (cos_theta - beta) / (1.0 - beta * cos_theta))


# -- what the two models share --------------------------------------------------------


class TestPairRateIsUnchanged:
    """The model distributes a pair's energy; it does not decide how often one is made."""

    @pytest.mark.parametrize("name", ["bethe_heitler", "uniform_split"])
    def test_pair_fraction_is_the_cross_section_ratio(
        self, name: str, request: pytest.FixtureRequest
    ) -> None:
        tally: _PairTally = request.getfixturevalue(name)
        expected = _pair_fraction(ENERGY_MEV)
        assert expected > 0.2  # pair is a major channel at 18 MeV in water (0.41)
        assert stats.binomtest(len(tally), tally.n_photons, expected).pvalue > 0.01

    def test_the_models_make_pairs_at_the_same_rate(
        self, bethe_heitler: _PairTally, uniform_split: _PairTally
    ) -> None:
        table = [
            [len(bethe_heitler), bethe_heitler.n_photons - len(bethe_heitler)],
            [len(uniform_split), uniform_split.n_photons - len(uniform_split)],
        ]
        assert stats.chi2_contingency(table).pvalue > 0.01


class TestEnergyIsConserved:
    @pytest.mark.parametrize("name", ["bethe_heitler", "uniform_split"])
    def test_the_pair_carries_the_photon_energy_less_two_masses(
        self, name: str, request: pytest.FixtureRequest
    ) -> None:
        tally: _PairTally = request.getfixturevalue(name)
        total = tally.electron_energy + tally.positron_energy
        np.testing.assert_allclose(total, ENERGY_MEV - 2.0 * ELECTRON_MASS_MEV, rtol=1.0e-13)

    @pytest.mark.parametrize("name", ["bethe_heitler", "uniform_split"])
    def test_every_history_balances(self, name: str, request: pytest.FixtureRequest) -> None:
        """in == deposited + escaped + carried (a positron carries its 2 m c^2)."""
        tally: _PairTally = request.getfixturevalue(name)
        assert tally.balance_error < 1.0e-13


# -- where they differ ----------------------------------------------------------------


class TestEnergySharing:
    def test_bethe_heitler_follows_the_screened_dcs(self, bethe_heitler: _PairTally) -> None:
        assert _sharing_p_value(bethe_heitler.electron_energy, ENERGY_MEV, "bethe-heitler") > 0.01

    def test_uniform_split_is_uniform(self, uniform_split: _PairTally) -> None:
        assert _sharing_p_value(uniform_split.electron_energy, ENERGY_MEV, "uniform") > 0.01

    def test_each_oracle_rejects_the_other_model(
        self, bethe_heitler: _PairTally, uniform_split: _PairTally
    ) -> None:
        """Power: the two sharing laws are told apart by the same test."""
        assert _sharing_p_value(bethe_heitler.electron_energy, ENERGY_MEV, "uniform") < 1.0e-6
        assert _sharing_p_value(uniform_split.electron_energy, ENERGY_MEV, "bethe-heitler") < 1.0e-6

    def test_the_sharing_is_symmetric_between_the_charges(self, bethe_heitler: _PairTally) -> None:
        """Born approximation: the positron's share has the electron's distribution."""
        assert _sharing_p_value(bethe_heitler.positron_energy, ENERGY_MEV, "bethe-heitler") > 0.01

    def test_below_two_mev_the_model_shares_uniformly(self) -> None:
        """The model's stated near-threshold approximation, reached through the loop."""
        tally = _first_interactions(BETHE_HEITLER, stream=3, energy=1.9, n=400_000)
        assert len(tally) > 400
        kinetic = 1.9 - 2.0 * ELECTRON_MASS_MEV
        assert stats.kstest(tally.electron_energy / kinetic, "uniform").pvalue > 0.01
        # ... while the directions are still the model's, not forward.
        assert np.all(tally.electron_cos < 1.0)


class TestLeptonAngles:
    def test_uniform_split_emits_both_leptons_forward(self, uniform_split: _PairTally) -> None:
        assert np.all(uniform_split.electron_cos == 1.0)
        assert np.all(uniform_split.positron_cos == 1.0)

    @pytest.mark.parametrize("lepton", ["electron", "positron"])
    def test_each_lepton_follows_the_leading_order_law_at_its_own_energy(
        self, bethe_heitler: _PairTally, lepton: str
    ) -> None:
        cos_theta = getattr(bethe_heitler, f"{lepton}_cos")
        kinetic = getattr(bethe_heitler, f"{lepton}_energy")
        assert stats.kstest(_angle_quantile(cos_theta, kinetic), "uniform").pvalue > 0.01

    def test_the_law_at_the_other_leptons_energy_is_rejected(
        self, bethe_heitler: _PairTally
    ) -> None:
        """Power: an arm that sampled both angles at one energy would fail above."""
        swapped = _angle_quantile(bethe_heitler.electron_cos, bethe_heitler.positron_energy)
        assert stats.kstest(swapped, "uniform").pvalue < 1.0e-6

    def test_the_azimuths_are_independent(self, bethe_heitler: _PairTally) -> None:
        """Stated approximation: uniform relative azimuth, not back to back."""
        relative = bethe_heitler.relative_azimuth / (2.0 * math.pi)
        assert stats.kstest(relative, "uniform").pvalue > 0.01

    def test_the_polar_angles_are_independent(self, bethe_heitler: _PairTally) -> None:
        """Stated approximation: no correlation between the two leptons' quantiles."""
        a = _angle_quantile(bethe_heitler.electron_cos, bethe_heitler.electron_energy)
        b = _angle_quantile(bethe_heitler.positron_cos, bethe_heitler.positron_energy)
        assert stats.spearmanr(a, b).pvalue > 0.01


# -- the production threshold ---------------------------------------------------------


class TestProductionThreshold:
    """A lepton born at or below ECUT is not transported, under either model."""

    @pytest.mark.parametrize("physics", [BETHE_HEITLER, CLINICAL_PHYSICS])
    def test_a_sub_threshold_positron_annihilates_where_it_is_made(
        self, physics: ResolvedPhysics
    ) -> None:
        """At 1.3 MeV the pair has 0.278 MeV: at most one lepton clears 0.2 MeV.

        The books then hold the positron's kinetic energy as a local deposit and
        its rest mass as two 511 keV photons, with no positron spawned.
        """
        grid = VoxelGrid.uniform_water(shape=(1, 1, 1), spacing=(BLOCK_CM, BLOCK_CM, BLOCK_CM))
        xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
        rng = HostRNG()
        half = 0.5 * BLOCK_CM
        seen = 0
        for i in range(60_000):
            spawned: list[StackEntry] = []
            deposited = [0.0]

            def deposit(x: float, y: float, z: float, amount: float, weighted: float) -> None:
                deposited[0] += amount  # noqa: B023 - consumed within this iteration

            escaped = photon_steps(
                1.3, 1.0, half, half, half, 0.0, 0.0, 1.0,
                grid, xs, rng.init_state(SEED + 7, i), deposit, spawned.append,
                pcut=PCUT_MEV, ecut=ECUT_MEV, transport_electrons=True, is_primary=True,
                physics=physics,
            )  # fmt: skip
            annihilation = [s for s in spawned if s[0] == PHOTON and s[1] == ELECTRON_MASS_MEV]
            if len(annihilation) != 2:
                continue
            seen += 1
            assert all(s[0] != POSITRON for s in spawned)
            carried = sum(s[2] * s[1] for s in spawned)
            assert deposited[0] + escaped + carried == pytest.approx(1.3, rel=1.0e-13)
            assert deposited[0] > 0.0
        assert seen > 10


# -- through the engine ---------------------------------------------------------------


def _engine(xs_type: type[AnalyticCrossSections] = AnalyticCrossSections) -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))
    return ReferenceEngine(
        grid=grid,
        cross_sections=xs_type(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )


class TestThroughTheEngine:
    def test_the_engine_ledger_closes(self) -> None:
        result = _engine().run(
            PencilBeamSource(energy=18.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0)),
            n_histories=200,
            n_batches=4,
            seed=SEED,
            msc_model="gaussian",
            physics=PhysicsOptions(pair="bethe-heitler"),
        )
        assert result.provenance is not None
        assert result.provenance.physics.pair == "bethe-heitler"
        assert result.provenance.physics.positron == "moller-at-rest"
        balance = result.energy_deposited + result.energy_escaped
        assert balance == pytest.approx(result.energy_emitted, rel=1.0e-9)

    def test_the_model_changes_the_transported_histories(self) -> None:
        beam = PencilBeamSource(energy=18.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
        common = dict(n_histories=200, n_batches=4, seed=SEED, msc_model="gaussian")
        engine = _engine()
        split = engine.run(beam, **common, physics=PhysicsOptions(pair="uniform-split"))
        default = engine.run(beam, **common)
        shared = engine.run(beam, **common, physics=PhysicsOptions(pair="bethe-heitler"))
        assert split.dose.tobytes() == default.dose.tobytes()  # "auto" is the uniform split
        assert shared.dose.tobytes() != default.dose.tobytes()

    def test_auto_selects_the_model_above_the_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 40 MeV photon source on the wide-domain instrument: mechanics only."""
        beam = PencilBeamSource(energy=40.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
        with caplog.at_level(logging.INFO, logger="pyradmc.backends.ref.engine"):
            result = _engine(_WideAnalyticWater).run(
                beam, n_histories=80, n_batches=4, seed=SEED, msc_model="gaussian"
            )
        assert "pair=bethe-heitler" in caplog.text
        assert result.provenance is not None
        assert result.provenance.physics.pair == "bethe-heitler"
        balance = result.energy_deposited + result.energy_escaped
        assert balance == pytest.approx(result.energy_emitted, rel=1.0e-9)

    def test_kerma_mode_does_not_read_the_model(self) -> None:
        """With charged secondaries deposited locally the sharing has no observable."""
        beam = PencilBeamSource(energy=18.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
        common = dict(
            n_histories=120, n_batches=4, seed=SEED, msc_model="gaussian", transport_electrons=False
        )
        engine = _engine()
        split = engine.run(beam, **common, physics=PhysicsOptions(pair="uniform-split"))
        shared = engine.run(beam, **common, physics=PhysicsOptions(pair="bethe-heitler"))
        assert split.dose.tobytes() == shared.dose.tobytes()


class _SignedPhotonPencil(Source):
    """An 18 MeV photon pencil along +z, every primary at one signed weight."""

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def max_energy(self) -> float:
        return 18.0

    def emit(self, rng_state: RNGState) -> Primary:
        return Primary(18.0, 4.0, 4.0, -1.0, 0.0, 0.0, 1.0, weight=self._weight)


@pytest.mark.parametrize("weight", [0.5, PHOTON_ROULETTE_WEIGHT_CAP])
def test_a_negative_weight_history_mirrors_the_positive_one_exactly(weight: float) -> None:
    """Weight -w against +w on one stream: the dose negates bit for bit, books close."""
    common = dict(
        n_histories=240,
        n_batches=4,
        seed=SEED,
        msc_model="gaussian",
        physics=PhysicsOptions(pair="bethe-heitler"),
    )
    engine = _engine()
    plus = engine.run(_SignedPhotonPencil(weight), **common)
    minus = engine.run(_SignedPhotonPencil(-weight), **common)

    assert np.any(plus.dose > 0.0)
    np.testing.assert_array_equal(minus.dose, -plus.dose)
    assert minus.energy_deposited == -plus.energy_deposited
    assert minus.energy_escaped == -plus.energy_escaped
    assert minus.energy_deposited + minus.energy_escaped == pytest.approx(
        minus.energy_emitted, rel=1.0e-9
    )
