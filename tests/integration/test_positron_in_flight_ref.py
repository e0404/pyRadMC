"""The ``"bhabha-in-flight"`` positron model in the reference electron loop.

The samplers are pinned on their own (``tests/physics/test_positron_sampling.py``)
and so is the data layer (``tests/unit/test_positron_data.py``); these tests pin
the *wiring* of the model into :func:`pyradmc.transport.electron.electron_steps`,
and what distinguishes it from ``"moller-at-rest"``.

Positrons are driven through the loop one at a time so that everything they spawn
can be tallied, on three instruments, each with an expectation that is a path
integral of interface quantities:

- **A thin slab** (0.5 cm of water that does not scatter, 10 MeV): delta rays per
  positron against ``int sigma_Bhabha rho dz``, and their spectrum against the
  Bhabha DCS. The clinical model's deltas follow Moller instead, at a rate 12
  percent higher, and fail the Bhabha spectrum.
- **A stopping block with continuous loss only** (no delta rays, no radiation, so
  the energy along the path is the positron range map and nothing else): the
  fraction annihilating in flight against
  ``1 - exp(-int sigma_ann / S+ dE)``, exactly. The annihilation photons of the
  same events give the Heitler spectrum and the momentum balance.
- **A finite slab with everything on**: every positron ends in exactly one of
  three ways (in flight, at rest, out of the grid) and its 2 m_e c^2 is booked
  exactly once whichever it is.

Seeds are fixed; every statistical comparison is a z-test against the sample's own
standard error or a goodness-of-fit p-value.
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
from pyradmc.data.materials import WATER
from pyradmc.data.positron import annihilation_dcs_per_electron, bhabha_dcs_per_electron
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource, Primary, Source
from pyradmc.physics_options import CLINICAL_PHYSICS, ResolvedPhysics
from pyradmc.rng import RNGState
from pyradmc.rng.host import HostRNG
from pyradmc.transport.electron import electron_steps
from pyradmc.transport.particles import ELECTRON, PHOTON, StackEntry
from tests.conftest import SEED
from tests.integration.test_bethe_heitler_ref import _StraightLineWater, _WideAnalyticWater

IN_FLIGHT = dataclasses.replace(CLINICAL_PHYSICS, positron="bhabha-in-flight")
TWO_M = 2.0 * ELECTRON_MASS_MEV
BIG_CM = 1.0e4

IN_FLIGHT_END = "in flight"
AT_REST_END = "at rest"
ESCAPED_END = "escaped"


class _ContinuousLossWater(AnalyticCrossSections):
    """Test instrument: water in which a positron only slows down or annihilates.

    No delta rays and no radiation, so a positron's energy along its path is the
    positron range map and nothing else, and the probability of annihilating in
    flight before reaching the cutoff is a one-dimensional integral. (The energy
    books still close: the restricted stopping power simply is the whole loss.)
    """

    def bhabha_cross_section(self, energy: float, material: int, delta_cut: float) -> float:
        return 0.0

    def radiative_stopping_power(self, energy: float, material: int) -> float:
        return 0.0


class _NoAnnihilationWater(_StraightLineWater):
    """Test instrument: water whose positrons neither annihilate in flight nor scatter.

    Isolates what the model's *collision* physics does to where a positron stops:
    without multiple scattering the depth it stops at is its path length less the
    detour of its own delta-ray recoils, so the spread is the energy straggling
    alone.
    """

    def annihilation_cross_section(self, energy: float, material: int) -> float:
        return 0.0


@dataclass(frozen=True)
class _History:
    """One positron's fate, as seen from outside the loop."""

    end: str
    photons: tuple[StackEntry, ...]  # annihilation photons (empty if escaped)
    deltas: tuple[float, ...]  # delta-ray kinetic energies, MeV
    stop_depth: float  # z of the annihilation, cm (nan if escaped)
    balance_error: float


def _transport(
    xs: AnalyticCrossSections,
    grid: VoxelGrid,
    physics: ResolvedPhysics,
    energy: float,
    n: int,
    stream: int,
    weight: float = PHOTON_ROULETTE_WEIGHT_CAP,
    start_depth: float = 1.0e-6,
) -> list[_History]:
    """Run ``n`` positrons along +z from ``start_depth`` (default: the entrance face).

    At the roulette weight cap no photon plays Russian roulette, so every tally is
    analog. Bremsstrahlung photons are told from annihilation photons by order:
    an annihilation is the last thing a positron does.
    """
    rng = HostRNG()
    x0 = 0.5 * grid.spacing[0] * grid.shape[0]
    y0 = 0.5 * grid.spacing[1] * grid.shape[1]
    histories = []
    for i in range(n):
        spawned: list[StackEntry] = []
        deposited = [0.0]

        def deposit(x: float, y: float, z: float, amount: float, weighted: float) -> None:
            deposited[0] += amount  # noqa: B023 - consumed within this iteration

        escaped = electron_steps(
            True, energy, weight, x0, y0, start_depth, 0.0, 0.0, 1.0,
            grid, xs, rng.init_state(SEED + stream, i), deposit, spawned.append,
            pcut=PCUT_MEV, ecut=ECUT_MEV, msc_model="gaussian", physics=physics,
        )  # fmt: skip

        carried = sum(entry[2] * entry[1] for entry in spawned)
        emitted = weight * (energy + TWO_M)
        balance = abs(emitted - (deposited[0] + escaped + carried)) / emitted
        deltas = tuple(entry[1] for entry in spawned if entry[0] == ELECTRON)
        last_two = tuple(spawned[-2:])
        pair = len(last_two) == 2 and all(entry[0] == PHOTON for entry in last_two)
        total = sum(entry[1] for entry in last_two) if pair else 0.0
        same_place = pair and last_two[0][3:6] == last_two[1][3:6]
        if pair and same_place and last_two[0][1] == last_two[1][1] == ELECTRON_MASS_MEV:
            end = AT_REST_END
        elif pair and same_place and total > TWO_M and escaped < weight * TWO_M:
            end = IN_FLIGHT_END
        else:
            end = ESCAPED_END
            last_two = ()
        histories.append(
            _History(
                end=end,
                photons=last_two,
                deltas=deltas,
                stop_depth=last_two[0][5] - start_depth if last_two else math.nan,
                balance_error=balance,
            )
        )
    return histories


def _slab(thickness: float) -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(1, 1, 1), spacing=(BIG_CM, BIG_CM, thickness))


def _block() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(1, 1, 1), spacing=(BIG_CM, BIG_CM, BIG_CM))


def _xs(xs_type: type[AnalyticCrossSections], grid: VoxelGrid) -> AnalyticCrossSections:
    return xs_type(geometry_densities=grid.max_density_by_material())


# -- delta rays: Bhabha, not Moller ---------------------------------------------------

SLAB_ENERGY_MEV = 10.0
SLAB_CM = 0.5
N_SLAB = 12_000


@pytest.fixture(scope="module")
def slab_in_flight() -> list[_History]:
    grid = _slab(SLAB_CM)
    return _transport(_xs(_StraightLineWater, grid), grid, IN_FLIGHT, SLAB_ENERGY_MEV, N_SLAB, 1)


@pytest.fixture(scope="module")
def slab_moller() -> list[_History]:
    grid = _slab(SLAB_CM)
    return _transport(
        _xs(_StraightLineWater, grid), grid, CLINICAL_PHYSICS, SLAB_ENERGY_MEV, N_SLAB, 2
    )


def _slab_integral(per_cm, positron: bool) -> float:  # type: ignore[no-untyped-def]
    """Integrate a per-path-length quantity over the slab along the continuous loss."""
    xs = AnalyticCrossSections()

    def energy_at(depth: float) -> float:
        if positron:
            return xs.positron_energy_after_mass_path(SLAB_ENERGY_MEV, WATER, ECUT_MEV, depth)
        return xs.energy_after_mass_path(SLAB_ENERGY_MEV, WATER, ECUT_MEV, depth)

    value, _ = integrate.quad(lambda z: per_cm(xs, energy_at(z)), 0.0, SLAB_CM)
    return float(value)


def _delta_rate(histories: list[_History]) -> tuple[float, float]:
    counts = np.array([len(h.deltas) for h in histories], dtype=float)
    return float(counts.mean()), float(counts.std(ddof=1) / math.sqrt(counts.size))


def _bhabha_spectrum_p_value(histories: list[_History]) -> float:
    """Chi-squared of the delta-ray energies against the Bhabha DCS at mid-slab energy.

    Twelve logarithmic bins from ECUT to 8 MeV (below the exit energy of every
    positron that has thrown no hard delta); the Moller DCS ends at half the
    kinetic energy and has a different shape below it.
    """
    xs = AnalyticCrossSections()
    mid = xs.positron_energy_after_mass_path(SLAB_ENERGY_MEV, WATER, ECUT_MEV, 0.5 * SLAB_CM)
    energies = np.array([w for h in histories for w in h.deltas])
    edges = np.geomspace(ECUT_MEV, 8.0, 13)
    observed, _ = np.histogram(energies, bins=edges)
    expected = np.array(
        [
            integrate.quad(lambda w: bhabha_dcs_per_electron(mid, w), a, b)[0]
            for a, b in itertools.pairwise(edges)
        ]
    )
    expected *= observed.sum() / expected.sum()
    assert np.all(expected > 10.0), "bin layout leaves low-count bins; rebin"
    return float(stats.chisquare(observed, expected).pvalue)


class TestDeltaRays:
    def test_rate_is_the_bhabha_cross_section(self, slab_in_flight: list[_History]) -> None:
        expected = _slab_integral(
            lambda xs, e: xs.bhabha_cross_section(e, WATER, ECUT_MEV), positron=True
        )
        mean, error = _delta_rate(slab_in_flight)
        assert abs(mean - expected) / error < 3.0

    def test_clinical_rate_is_the_moller_cross_section(self, slab_moller: list[_History]) -> None:
        expected = _slab_integral(
            lambda xs, e: xs.moller_cross_section(e, WATER, ECUT_MEV), positron=False
        )
        mean, error = _delta_rate(slab_moller)
        assert abs(mean - expected) / error < 3.0

    def test_the_two_rates_are_different_quantities(self) -> None:
        """Bhabha throws fewer delta rays above 200 keV than Moller: 11 percent at 10 MeV."""
        bhabha = _slab_integral(
            lambda xs, e: xs.bhabha_cross_section(e, WATER, ECUT_MEV), positron=True
        )
        moller = _slab_integral(
            lambda xs, e: xs.moller_cross_section(e, WATER, ECUT_MEV), positron=False
        )
        assert 0.85 < bhabha / moller < 0.92

    def test_spectrum_is_bhabha(self, slab_in_flight: list[_History]) -> None:
        assert _bhabha_spectrum_p_value(slab_in_flight) > 0.01

    def test_the_clinical_models_spectrum_is_not(self, slab_moller: list[_History]) -> None:
        """Power: Moller deltas fail the Bhabha spectrum."""
        assert _bhabha_spectrum_p_value(slab_moller) < 1.0e-6

    def test_a_delta_may_exceed_half_the_kinetic_energy(
        self, slab_in_flight: list[_History], slab_moller: list[_History]
    ) -> None:
        """Distinguishable particles: Bhabha transfers run to T, Moller's to T/2."""
        assert max(w for h in slab_in_flight for w in h.deltas) > 0.5 * SLAB_ENERGY_MEV
        assert max(w for h in slab_moller for w in h.deltas) <= 0.5 * SLAB_ENERGY_MEV

    def test_the_clinical_model_never_annihilates_in_flight(
        self, slab_moller: list[_History]
    ) -> None:
        assert all(h.end != IN_FLIGHT_END for h in slab_moller)


# -- annihilation in flight -----------------------------------------------------------

N_STOP = 5_000


def _expected_in_flight_fraction(energy: float) -> float:
    """``1 - exp(-int_ECUT^T sigma_ann / S+ dE)`` in water: continuous loss only."""
    xs = AnalyticCrossSections()
    optical, _ = integrate.quad(
        lambda e: (
            xs.annihilation_cross_section(e, WATER)
            / xs.positron_restricted_stopping_power(e, WATER, ECUT_MEV)
        ),
        ECUT_MEV,
        energy,
        limit=200,
    )
    return 1.0 - math.exp(-optical)


@pytest.fixture(scope="module", params=[1.0, 5.0])
def stopped(request: pytest.FixtureRequest) -> tuple[float, list[_History]]:
    energy = float(request.param)
    grid = _block()
    return energy, _transport(
        _xs(_ContinuousLossWater, grid),
        grid,
        IN_FLIGHT,
        energy,
        N_STOP,
        10 + int(energy),
        start_depth=0.5 * BIG_CM,
    )


class TestAnnihilationInFlight:
    def test_fraction_is_the_path_integral_of_the_cross_section(
        self, stopped: tuple[float, list[_History]]
    ) -> None:
        energy, histories = stopped
        expected = _expected_in_flight_fraction(energy)
        in_flight = sum(h.end == IN_FLIGHT_END for h in histories)
        assert all(h.end != ESCAPED_END for h in histories)
        assert stats.binomtest(in_flight, len(histories), expected).pvalue > 0.01

    def test_a_dropped_channel_would_be_seen(self, stopped: tuple[float, list[_History]]) -> None:
        """Power: the measured fraction is many sigma from zero and from twice itself."""
        energy, histories = stopped
        in_flight = sum(h.end == IN_FLIGHT_END for h in histories)
        expected = _expected_in_flight_fraction(energy)
        assert stats.binomtest(in_flight, len(histories), 2.0 * expected).pvalue < 1.0e-6
        assert in_flight > 100

    def test_photon_energies_follow_the_heitler_spectrum(
        self, stopped: tuple[float, list[_History]]
    ) -> None:
        """Each event's labelled photon, through the CDF of the DCS at its own energy."""
        _, histories = stopped
        quantiles = []
        for h in histories:
            if h.end != IN_FLIGHT_END:
                continue
            k_first, k_second = h.photons[0][1], h.photons[1][1]
            kinetic = k_first + k_second - TWO_M
            half_width = 0.5 * math.sqrt(kinetic / (kinetic + TWO_M))
            k_min = (kinetic + TWO_M) * (0.5 - half_width)
            k_max = (kinetic + TWO_M) * (0.5 + half_width)

            def dcs(k: float, kinetic: float = kinetic) -> float:
                return annihilation_dcs_per_electron(kinetic, k)

            below = integrate.quad(dcs, k_min, min(max(k_first, k_min), k_max))[0]
            quantiles.append(below / integrate.quad(dcs, k_min, k_max)[0])
        assert stats.kstest(quantiles, "uniform").pvalue > 0.01

    def test_the_photons_carry_the_positrons_momentum(
        self, stopped: tuple[float, list[_History]]
    ) -> None:
        """|k1 d1 + k2 d2| = sqrt(T (T + 2 m)): polar angles and back-to-back azimuths."""
        _, histories = stopped
        for h in histories:
            if h.end != IN_FLIGHT_END:
                continue
            first, second = h.photons
            momentum = np.array(first[6:9]) * first[1] + np.array(second[6:9]) * second[1]
            kinetic = first[1] + second[1] - TWO_M
            assert np.linalg.norm(momentum) == pytest.approx(
                math.sqrt(kinetic * (kinetic + TWO_M)), rel=1.0e-9
            )

    def test_every_history_balances(self, stopped: tuple[float, list[_History]]) -> None:
        _, histories = stopped
        assert max(h.balance_error for h in histories) < 1.0e-12


# -- one end per positron, one rest mass per history ----------------------------------


@pytest.fixture(scope="module", params=["thin-target", "bethe-heitler"])
def histories(request: pytest.FixtureRequest) -> list[_History]:
    """6 MeV positrons into 1.5 cm of water, all physics on, under either brems model."""
    physics = dataclasses.replace(IN_FLIGHT, bremsstrahlung=request.param)
    grid = _slab(1.5)
    return _transport(_xs(AnalyticCrossSections, grid), grid, physics, 6.0, 2_500, 20)


class TestRestMassIsBookedOnce:
    """Three ways to end, one rest mass each."""

    def test_all_three_ends_occur(self, histories: list[_History]) -> None:
        ends = [h.end for h in histories]
        assert ends.count(IN_FLIGHT_END) > 20
        assert ends.count(AT_REST_END) > 20
        assert ends.count(ESCAPED_END) > 20

    def test_in_equals_out_whichever_way_it_ends(self, histories: list[_History]) -> None:
        """T + 2 m in; deposits, escapes and secondaries out; to rounding, per history.

        An in-flight annihilation that also booked the latent 2 m c^2 (or a rest
        annihilation after one) would miss by 1.022 MeV in 7.022.
        """
        for end in (IN_FLIGHT_END, AT_REST_END, ESCAPED_END):
            worst = max(h.balance_error for h in histories if h.end == end)
            assert worst < 1.0e-12, end

    def test_annihilation_photons_carry_at_least_the_rest_masses(
        self, histories: list[_History]
    ) -> None:
        for h in histories:
            if h.end == IN_FLIGHT_END:
                assert sum(p[1] for p in h.photons) > TWO_M + ECUT_MEV
                assert min(p[1] for p in h.photons) > 0.5 * ELECTRON_MASS_MEV
            elif h.end == AT_REST_END:
                assert sum(p[1] for p in h.photons) == TWO_M


class TestAnAnnihilationPhotonBelowPcut:
    """A PCUT above the softest in-flight photon (about m c^2 / 2) deposits that photon.

    On the continuous-loss instrument a positron spawns nothing but its
    annihilation photons, so a history that spawns exactly one ended in flight
    with its partner below PCUT, deposited where the positron annihilated (a rest
    annihilation's two 511 keV photons both clear this PCUT).
    """

    PCUT_MEV = 0.4
    ENERGY_MEV = 6.0
    N = 1_000

    def test_the_partner_is_deposited_and_every_history_balances(self) -> None:
        grid = _block()
        xs = _xs(_ContinuousLossWater, grid)
        rng = HostRNG()
        weight = PHOTON_ROULETTE_WEIGHT_CAP  # analog: no photon plays roulette
        emitted = weight * (self.ENERGY_MEV + TWO_M)
        singles = []
        for i in range(self.N):
            spawned: list[StackEntry] = []
            deposited = [0.0]

            def deposit(x: float, y: float, z: float, amount: float, weighted: float) -> None:
                deposited[0] += amount  # noqa: B023 - consumed within this iteration

            escaped = electron_steps(
                True, self.ENERGY_MEV, weight, 0.5 * BIG_CM, 0.5 * BIG_CM, 0.5 * BIG_CM,
                0.0, 0.0, 1.0, grid, xs, rng.init_state(SEED + 40, i), deposit,
                spawned.append, pcut=self.PCUT_MEV, ecut=ECUT_MEV, msc_model="gaussian",
                physics=IN_FLIGHT,
            )  # fmt: skip
            assert all(entry[0] == PHOTON for entry in spawned)
            assert len(spawned) in (1, 2)
            carried = sum(entry[2] * entry[1] for entry in spawned)
            assert abs(emitted - (deposited[0] + escaped + carried)) / emitted < 1.0e-12
            if len(spawned) == 1:
                singles.append(spawned[0][1])
        assert len(singles) > 10
        # The spawned photon is the harder one: the partner took less than PCUT.
        assert min(singles) > TWO_M + ECUT_MEV - self.PCUT_MEV


# -- where a positron stops -----------------------------------------------------------


def _csda_range_ratio(energy: float) -> float:
    """Positron over electron CSDA range from the two unrestricted stopping powers."""
    xs = AnalyticCrossSections()

    def inverse_total(e: float, positron: bool) -> float:
        collision = (
            xs.positron_restricted_stopping_power(e, WATER, e)
            if positron
            else xs.restricted_stopping_power(e, WATER, e)
        )
        return 1.0 / (collision + xs.radiative_stopping_power(e, WATER))

    r_plus, _ = integrate.quad(inverse_total, ECUT_MEV, energy, args=(True,), limit=200)
    r_minus, _ = integrate.quad(inverse_total, ECUT_MEV, energy, args=(False,), limit=200)
    return float(r_plus / r_minus)


class TestStoppingDepth:
    """The positron's own stopping power moves where it stops, as its CSDA range says."""

    ENERGY_MEV = 10.0
    N = 3_000

    def _depths(self, physics: ResolvedPhysics, stream: int) -> np.ndarray:
        grid = _block()
        histories = _transport(
            _xs(_NoAnnihilationWater, grid),
            grid,
            physics,
            self.ENERGY_MEV,
            self.N,
            stream,
            start_depth=0.5 * BIG_CM,
        )
        assert all(h.end == AT_REST_END for h in histories)
        return np.array([h.stop_depth for h in histories])

    def test_positrons_stop_deeper_by_the_csda_range_ratio(self) -> None:
        expected = _csda_range_ratio(self.ENERGY_MEV)
        assert 1.015 < expected < 1.03  # the positron loses 2 to 3 percent less per cm

        own = self._depths(IN_FLIGHT, 30)
        electron_like = self._depths(CLINICAL_PHYSICS, 31)
        ratio = own.mean() / electron_like.mean()
        error = (
            ratio
            * math.hypot(
                own.std(ddof=1) / own.mean(), electron_like.std(ddof=1) / electron_like.mean()
            )
            / math.sqrt(self.N)
        )
        assert (ratio - 1.0) / error > 3.0  # deeper, resolved
        assert abs(ratio - expected) / error < 3.0


# -- through the engine ---------------------------------------------------------------


def _engine(xs_type: type[AnalyticCrossSections] = AnalyticCrossSections) -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))
    return ReferenceEngine(
        grid=grid,
        cross_sections=xs_type(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )


class TestThroughTheEngine:
    @pytest.mark.parametrize("primaries", ["photon", "positron"])
    def test_the_engine_ledger_closes(self, primaries: str) -> None:
        """Positrons from pair production, and positrons the source starts."""
        source: Source = (
            PencilBeamSource(energy=18.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
            if primaries == "photon"
            else _SignedPositronPencil(1.0)
        )
        result = _engine().run(
            source,
            n_histories=200,
            n_batches=4,
            seed=SEED,
            msc_model="gaussian",
            physics=PhysicsOptions(positron="bhabha-in-flight"),
        )
        assert result.provenance is not None
        assert result.provenance.physics.positron == "bhabha-in-flight"
        balance = result.energy_deposited + result.energy_escaped
        assert balance == pytest.approx(result.energy_emitted, rel=1.0e-9)

    def test_the_model_changes_the_transported_histories(self) -> None:
        beam = PencilBeamSource(energy=18.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
        common = dict(n_histories=200, n_batches=4, seed=SEED, msc_model="gaussian")
        engine = _engine()
        default = engine.run(beam, **common)
        named = engine.run(beam, **common, physics=PhysicsOptions(positron="moller-at-rest"))
        own = engine.run(beam, **common, physics=PhysicsOptions(positron="bhabha-in-flight"))
        assert named.dose.tobytes() == default.dose.tobytes()  # "auto" is moller-at-rest
        assert own.dose.tobytes() != default.dose.tobytes()

    def test_an_electron_beam_without_pairs_does_not_read_the_model(self) -> None:
        """Below the pair threshold there is no positron for the model to act on."""
        beam = PencilBeamSource(energy=1.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
        common = dict(
            n_histories=80, n_batches=4, seed=SEED, msc_model="gaussian", primary_kind="electron"
        )
        engine = _engine()
        default = engine.run(beam, **common)
        own = engine.run(beam, **common, physics=PhysicsOptions(positron="bhabha-in-flight"))
        assert own.dose.tobytes() == default.dose.tobytes()

    def test_auto_selects_the_model_above_the_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 40 MeV photon source on the wide-domain instrument: mechanics only."""
        beam = PencilBeamSource(energy=40.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
        with caplog.at_level(logging.INFO, logger="pyradmc.backends.ref.engine"):
            result = _engine(_WideAnalyticWater).run(
                beam, n_histories=80, n_batches=4, seed=SEED, msc_model="gaussian"
            )
        assert "positron=bhabha-in-flight" in caplog.text
        assert result.provenance is not None
        assert result.provenance.physics.positron == "bhabha-in-flight"
        balance = result.energy_deposited + result.energy_escaped
        assert balance == pytest.approx(result.energy_emitted, rel=1.0e-9)


class _SignedPositronPencil(Source):
    """A 12 MeV positron pencil along +z, every primary at one signed weight."""

    particle_kinds = frozenset({"positron"})

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def max_energy(self) -> float:
        return 12.0

    def emit(self, rng_state: RNGState) -> Primary:
        return Primary(12.0, 4.0, 4.0, -1.0, 0.0, 0.0, 1.0, weight=self._weight, kind="positron")


@pytest.mark.parametrize("weight", [0.5, PHOTON_ROULETTE_WEIGHT_CAP])
@pytest.mark.parametrize("bremsstrahlung", ["thin-target", "bethe-heitler"])
def test_a_negative_weight_history_mirrors_the_positive_one_exactly(
    weight: float, bremsstrahlung: str
) -> None:
    """Weight -w against +w on one stream: the dose negates bit for bit, books close.

    The annihilation arm makes the same sign-blind decisions as the bremsstrahlung
    arms (the deposit guard on the unweighted energy, a photon spawned for any
    nonzero weight, the roulette cap on ``abs(w)``).
    """
    common = dict(
        n_histories=160,
        n_batches=4,
        seed=SEED,
        msc_model="gaussian",
        physics=PhysicsOptions(positron="bhabha-in-flight", bremsstrahlung=bremsstrahlung),
    )
    engine = _engine()
    plus = engine.run(_SignedPositronPencil(weight), **common)
    minus = engine.run(_SignedPositronPencil(-weight), **common)

    assert np.any(plus.dose > 0.0)
    np.testing.assert_array_equal(minus.dose, -plus.dose)
    assert minus.energy_deposited == -plus.energy_deposited
    assert minus.energy_escaped == -plus.energy_escaped
    assert minus.energy_emitted < 0.0
    assert minus.energy_deposited + minus.energy_escaped == pytest.approx(
        minus.energy_emitted, rel=1.0e-9
    )
