"""The ``"bethe-heitler"`` bremsstrahlung model in the reference electron loop.

The sampling routines are pinned on their own (``tests/physics``) and so is the
data layer (``tests/unit``); these tests pin the *wiring* of the model into
:func:`pyradmc.transport.electron.electron_steps`, and what distinguishes it from
the thin-target model at the level a dose calculation sees.

The instrument is a thin water slab crossed by 15 MeV electrons, driven through
the loop directly so the photons it spawns can be tallied one by one. In a slab
the analytic expectation is a path integral of interface quantities, with no
transport model of its own:

- photons per electron: the discrete cross-section, ``int sigma rho dz`` —
  distance sampling and channel selection;
- energy radiated above PCUT per electron: the radiative stopping power less its
  sub-PCUT share — the normalization both models share, so the two must agree;
- mean photon energy, and the second moment ``sum k^2`` per electron, which is
  the variance the radiative loss adds to the electron's energy (compound
  Poisson): the spectrum and the straggling, where the models must differ.

The electrons carry the roulette weight cap, so no photon plays Russian roulette
and every tally is analog. Every comparison is a z-test against the sample's own
standard error; seeds are fixed.
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
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource, PencilBeamSource, Primary, Source
from pyradmc.physics.brems_bh import bethe_heitler_shape
from pyradmc.physics_options import CLINICAL_PHYSICS, ResolvedPhysics
from pyradmc.rng import RNGState
from pyradmc.rng.host import HostRNG
from pyradmc.transport.electron import electron_steps
from pyradmc.transport.particles import ELECTRON, PHOTON, StackEntry
from tests.conftest import SEED

BETHE_HEITLER = dataclasses.replace(CLINICAL_PHYSICS, bremsstrahlung="bethe-heitler")
ENERGY_MEV = 15.0
SLAB_CM = 0.5
N_ELECTRONS = 12_000


class _WideAnalyticWater(AnalyticCrossSections):
    """Test instrument: analytic water declaring a 250 MeV energy domain.

    The analytic source ends at 20 MeV (its pair and radiative calibration), so no
    shipped analytic run reaches the 30 MeV threshold of ``"auto"``. This instrument
    declares the wider domain the way ``tests/unit/test_energy_domain.py`` does, to
    exercise the *mechanics* above the threshold without the EPICS libraries: the
    restricted-range and bremsstrahlung grids extend to the declared domain, the
    collision stopping power is Berger-Seltzer and the radiative stopping power
    follows ESTAR to 1 GeV, both valid there. Its photon data is the analytic fit
    extrapolated, so nothing run with it is a validated dose. The validated route
    is a tabulated compile to the energy in question
    (``tests/validation/test_bethe_heitler_high_energy.py``).
    """

    @property
    def energy_max(self) -> float:
        return 250.0


class _StraightLineWater(AnalyticCrossSections):
    """Test instrument: water whose electrons do not scatter.

    With the scattering power switched off the Gaussian hinge deflects by nothing,
    so an electron that has thrown no delta ray still travels along the beam axis
    when it radiates, and a photon's angle to the axis *is* its emission angle.
    """

    def scattering_power(self, energy: float, material: int, delta_cut: float) -> float:
        return 1.0e-14


@dataclass(frozen=True)
class _SlabTally:
    """Photons spawned by ``N_ELECTRONS`` electrons crossing the slab, per electron."""

    count: np.ndarray  # photons
    radiated: np.ndarray  # sum of k, MeV
    second_moment: np.ndarray  # sum of k^2, MeV^2
    local: np.ndarray  # continuous loss deposited in the slab, MeV
    photon_energy: np.ndarray  # every photon, MeV
    axis_one_minus_cos: np.ndarray  # 1 - u_z of photons emitted before any delta ray
    balance_error: float  # worst |in - (deposited + escaped + spawned)| / in

    def mean_and_error(self, per_electron: np.ndarray) -> tuple[float, float]:
        return float(per_electron.mean()), float(per_electron.std(ddof=1) / math.sqrt(len(self)))

    def __len__(self) -> int:
        return int(self.count.size)


def _cross_slab(physics: ResolvedPhysics, stream: int) -> _SlabTally:
    grid = VoxelGrid.uniform_water(shape=(1, 1, 1), spacing=(400.0, 400.0, SLAB_CM))
    xs = _StraightLineWater(geometry_densities=grid.max_density_by_material())
    rng = HostRNG()
    weight = PHOTON_ROULETTE_WEIGHT_CAP  # at the cap: bremsstrahlung photons stay analog

    count = np.zeros(N_ELECTRONS)
    radiated = np.zeros(N_ELECTRONS)
    second_moment = np.zeros(N_ELECTRONS)
    local = np.zeros(N_ELECTRONS)
    energies: list[float] = []
    on_axis: list[float] = []
    worst = 0.0
    for i in range(N_ELECTRONS):
        spawned: list[StackEntry] = []
        deposited = [0.0]

        def deposit(x: float, y: float, z: float, energy: float, weighted: float) -> None:
            deposited[0] += energy  # noqa: B023 - consumed within this iteration

        escaped = electron_steps(
            False, ENERGY_MEV, weight, 200.0, 200.0, 1.0e-6, 0.0, 0.0, 1.0,
            grid, xs, rng.init_state(SEED + stream, i), deposit, spawned.append,
            pcut=PCUT_MEV, ecut=ECUT_MEV, msc_model="gaussian", physics=physics,
        )  # fmt: skip

        delta_thrown = False
        carried = 0.0
        for kind, energy, w, *_position, _ux, _uy, uz in spawned:
            carried += w * energy
            if kind == ELECTRON:
                delta_thrown = True
                continue
            assert kind == PHOTON
            assert w == weight  # analog: no roulette at the weight cap
            assert PCUT_MEV < energy < ENERGY_MEV
            count[i] += 1.0
            radiated[i] += energy
            second_moment[i] += energy * energy
            energies.append(energy)
            if not delta_thrown:
                on_axis.append(1.0 - uz)
        local[i] = deposited[0] / weight
        emitted = weight * ENERGY_MEV
        worst = max(worst, abs(emitted - (deposited[0] + escaped + carried)) / emitted)

    return _SlabTally(
        count=count,
        radiated=radiated,
        second_moment=second_moment,
        local=local,
        photon_energy=np.array(energies),
        axis_one_minus_cos=np.array(on_axis),
        balance_error=worst,
    )


@pytest.fixture(scope="module")
def bethe_heitler() -> _SlabTally:
    return _cross_slab(BETHE_HEITLER, stream=1)


@pytest.fixture(scope="module")
def thin_target() -> _SlabTally:
    return _cross_slab(CLINICAL_PHYSICS, stream=2)


# -- analytic expectations along the slab ------------------------------------------


def _energy_at_depth(xs: AnalyticCrossSections, depth: float) -> float:
    """Electron energy after ``depth`` cm of continuous collision loss in water."""
    return xs.energy_after_mass_path(ENERGY_MEV, WATER, ECUT_MEV, depth)


def _slab_integral(per_cm) -> float:  # type: ignore[no-untyped-def]
    """Integrate a per-path-length quantity over the slab along the mean energy."""
    xs = AnalyticCrossSections()
    value, _ = integrate.quad(lambda z: per_cm(xs, _energy_at_depth(xs, z)), 0.0, SLAB_CM)
    return float(value)


def _shape_moment(energy: float, power: int) -> float:
    """``int_PCUT^T G(k) k^(power - 1) dk`` of the water shape, integrated in ln k."""
    xs = AnalyticCrossSections()
    args = xs.bremsstrahlung_screening(WATER)[:5]
    value, _ = integrate.quad(
        lambda t: bethe_heitler_shape(energy, math.exp(t), *args) * math.exp(power * t),
        math.log(PCUT_MEV),
        math.log(energy),
        epsrel=1.0e-10,
        limit=200,
    )
    return float(value)


def _bh_rate(xs: AnalyticCrossSections, energy: float) -> float:
    return xs.bremsstrahlung_cross_section(energy, WATER, PCUT_MEV)


def _bh_radiated(xs: AnalyticCrossSections, energy: float) -> float:
    total = xs.radiative_stopping_power(energy, WATER)
    return total - xs.restricted_radiative_stopping_power(energy, WATER, PCUT_MEV)


def _bh_second_moment(xs: AnalyticCrossSections, energy: float) -> float:
    # sigma <k^2> = sigma * int G k dk / int (G / k) dk
    return _bh_rate(xs, energy) * _shape_moment(energy, 2) / _shape_moment(energy, 0)


def _thin_radiated(xs: AnalyticCrossSections, energy: float) -> float:
    return xs.radiative_stopping_power(energy, WATER) * (1.0 - PCUT_MEV / energy)


def _thin_rate(xs: AnalyticCrossSections, energy: float) -> float:
    mean_k = (energy - PCUT_MEV) / math.log(energy / PCUT_MEV)
    return _thin_radiated(xs, energy) / mean_k


def _thin_second_moment(xs: AnalyticCrossSections, energy: float) -> float:
    # 1/k spectrum: <k^2> / <k> = (T + PCUT) / 2
    return _thin_radiated(xs, energy) * 0.5 * (energy + PCUT_MEV)


def _local_radiative_excess(xs: AnalyticCrossSections, energy: float) -> float:
    """Sub-PCUT radiative stopping power: screened shape minus the 1/k share."""
    return xs.restricted_radiative_stopping_power(
        energy, WATER, PCUT_MEV
    ) - xs.radiative_stopping_power(energy, WATER) * (PCUT_MEV / energy)


def _total_radiative(xs: AnalyticCrossSections, energy: float) -> float:
    return xs.radiative_stopping_power(energy, WATER)


def _screened_shape_p_value(photon_energies: np.ndarray) -> float:
    """Chi-squared p-value of a photon sample against G(k)/k at the mid-slab energy.

    Binned uniformly in ln k, where the 1/k envelope is flat and the expected
    count of a bin is the integral of G alone. The last bin closes at 12 MeV,
    below the exit energy, so every bin is populated from the whole slab.
    """
    xs = AnalyticCrossSections()
    args = xs.bremsstrahlung_screening(WATER)[:5]
    mid = _energy_at_depth(xs, 0.5 * SLAB_CM)
    edges = np.geomspace(PCUT_MEV, 12.0, 13)
    observed, _ = np.histogram(photon_energies, bins=edges)
    expected = np.array(
        [
            integrate.quad(
                lambda t: bethe_heitler_shape(mid, math.exp(t), *args), math.log(a), math.log(b)
            )[0]
            for a, b in itertools.pairwise(edges)
        ]
    )
    expected *= observed.sum() / expected.sum()
    assert np.all(expected > 20.0), "bin layout leaves low-count bins; rebin"
    return float(stats.chisquare(observed, expected).pvalue)


def _z(measured: tuple[float, float], expected: float) -> float:
    mean, error = measured
    return (mean - expected) / error


def _z_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) / math.hypot(a[1], b[1])


# -- the loop conserves energy -----------------------------------------------------


class TestEnergyIsConserved:
    def test_every_history_balances(self, bethe_heitler: _SlabTally) -> None:
        """in == deposited + escaped + carried off by secondaries, history by history."""
        assert bethe_heitler.balance_error < 1.0e-12

    @pytest.mark.parametrize("primary_kind", ["photon", "electron"])
    def test_the_engine_ledger_closes(self, primary_kind: str) -> None:
        """emitted == deposited + escaped through the engine, roulette included.

        Under the Gaussian hinge: the balance does not depend on the angular law,
        and the reference backend's lazily built Goudsmit-Saunderson tables would
        cost a minute at 18 MeV.
        """
        grid = VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))
        engine = ReferenceEngine(
            grid=grid,
            cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
            rng=HostRNG(),
        )
        result = engine.run(
            PencilBeamSource(energy=18.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0)),
            n_histories=120,
            n_batches=4,
            seed=SEED,
            primary_kind=primary_kind,
            msc_model="gaussian",
            physics=PhysicsOptions(bremsstrahlung="bethe-heitler"),
        )
        assert result.provenance is not None
        assert result.provenance.physics.bremsstrahlung == "bethe-heitler"
        balance = result.energy_deposited + result.energy_escaped
        assert balance == pytest.approx(result.energy_emitted, rel=1.0e-9)


# -- what the two models share -----------------------------------------------------


class TestRadiatedEnergy:
    """Both models are normalized to the same radiative stopping power."""

    def test_bethe_heitler_radiates_the_stopping_power(self, bethe_heitler: _SlabTally) -> None:
        measured = bethe_heitler.mean_and_error(bethe_heitler.radiated)
        assert abs(_z(measured, _slab_integral(_bh_radiated))) < 3.0

    def test_thin_target_radiates_the_stopping_power(self, thin_target: _SlabTally) -> None:
        measured = thin_target.mean_and_error(thin_target.radiated)
        assert abs(_z(measured, _slab_integral(_thin_radiated))) < 3.0

    def test_the_models_radiate_the_same_energy(
        self, bethe_heitler: _SlabTally, thin_target: _SlabTally
    ) -> None:
        """Equal within the statistics; the expectations differ by 0.3 percent.

        The two models share the total and split it at PCUT differently: the 1/k
        spectrum leaves the fraction PCUT/T below the cut, the screened shape
        slightly more. That difference is far below what this sample resolves
        (each mean carries about 6 percent, the difference about 8), and is
        asserted here so the equality tested below it is not an accident.
        """
        expected_bh = _slab_integral(_bh_radiated)
        expected_thin = _slab_integral(_thin_radiated)
        assert expected_bh == pytest.approx(expected_thin, rel=5.0e-3)

        z = _z_between(
            bethe_heitler.mean_and_error(bethe_heitler.radiated),
            thin_target.mean_and_error(thin_target.radiated),
        )
        assert abs(z) < 3.0


class TestContinuousRadiativeLoss:
    """What the loop charges continuously for radiation: the sub-PCUT share only.

    The energy an electron deposits *in the slab* is its restricted collision loss
    plus the radiative loss the model keeps local. The collision part is the same
    in both models, and so is everything systematic about the slab (the boundary
    nudge lengthens every path by 0.02 percent), so the difference of the two
    means isolates the difference of the two local radiative shares — a quantity
    of 4e-4 MeV on a deposit of 0.84 MeV, resolved here at twelve sigma because
    the continuous deposit hardly fluctuates.
    """

    def test_local_share_is_the_restricted_stopping_power(
        self, bethe_heitler: _SlabTally, thin_target: _SlabTally
    ) -> None:
        measured = _z_between(
            bethe_heitler.mean_and_error(bethe_heitler.local),
            thin_target.mean_and_error(thin_target.local),
        )
        a, a_err = bethe_heitler.mean_and_error(bethe_heitler.local)
        b, b_err = thin_target.mean_and_error(thin_target.local)
        difference = (a - b, math.hypot(a_err, b_err))

        assert measured > 5.0  # the screened shape leaves more below PCUT, visibly
        assert abs(_z(difference, _slab_integral(_local_radiative_excess))) < 3.0

    def test_the_total_is_not_charged_on_top_of_the_photons(
        self, bethe_heitler: _SlabTally, thin_target: _SlabTally
    ) -> None:
        """Power: a loop that also charged the total S_rad would be off by 300 times this.

        The double-counting error the split invites would add the whole radiative
        stopping power to the continuous loss; the measured difference is a few
        thousandths of that.
        """
        difference = float(bethe_heitler.local.mean() - thin_target.local.mean())
        assert _slab_integral(_total_radiative) > 100.0 * _slab_integral(_local_radiative_excess)
        assert difference < 0.01 * _slab_integral(_total_radiative)


# -- where the two models differ ---------------------------------------------------


class TestPhotonRate:
    def test_rate_is_the_discrete_cross_section(self, bethe_heitler: _SlabTally) -> None:
        """Photons per electron == int sigma rho dz: distance sampling and channel choice."""
        measured = bethe_heitler.mean_and_error(bethe_heitler.count)
        assert abs(_z(measured, _slab_integral(_bh_rate))) < 3.0

    def test_thin_target_rate_is_its_bernoulli_probability(self, thin_target: _SlabTally) -> None:
        measured = thin_target.mean_and_error(thin_target.count)
        assert abs(_z(measured, _slab_integral(_thin_rate))) < 3.0

    def test_bethe_heitler_emits_more_photons(
        self, bethe_heitler: _SlabTally, thin_target: _SlabTally
    ) -> None:
        """Same energy in softer photons: more of them, far outside the noise."""
        assert _slab_integral(_bh_rate) > 1.2 * _slab_integral(_thin_rate)
        z = _z_between(
            bethe_heitler.mean_and_error(bethe_heitler.count),
            thin_target.mean_and_error(thin_target.count),
        )
        assert z > 5.0


class TestPhotonSpectrum:
    def test_spectrum_is_the_screened_shape(self, bethe_heitler: _SlabTally) -> None:
        """Photon energies follow G(k)/k at the mid-slab energy (chi-squared in ln k).

        The electron loses 7 percent of its energy across the slab, which moves
        only the tip of the spectrum, above the binned range.
        """
        assert _screened_shape_p_value(bethe_heitler.photon_energy) > 0.01

    def test_thin_target_spectrum_is_not_the_screened_shape(self, thin_target: _SlabTally) -> None:
        """The same oracle rejects the 1/k spectrum: the test above has power."""
        assert _screened_shape_p_value(thin_target.photon_energy) < 1.0e-6

    def test_bethe_heitler_photons_are_softer(
        self, bethe_heitler: _SlabTally, thin_target: _SlabTally
    ) -> None:
        """Mean photon energy: screened shape below the 1/k spectrum's."""

        def mean_k(tally: _SlabTally) -> tuple[float, float]:
            k = tally.photon_energy
            return float(k.mean()), float(k.std(ddof=1) / math.sqrt(k.size))

        assert _z_between(mean_k(thin_target), mean_k(bethe_heitler)) > 5.0


class TestRadiativeStraggling:
    """``sum k^2`` per electron: the variance radiation adds to the electron energy."""

    def test_bethe_heitler_second_moment(self, bethe_heitler: _SlabTally) -> None:
        measured = bethe_heitler.mean_and_error(bethe_heitler.second_moment)
        assert abs(_z(measured, _slab_integral(_bh_second_moment))) < 3.0

    def test_thin_target_second_moment(self, thin_target: _SlabTally) -> None:
        measured = thin_target.mean_and_error(thin_target.second_moment)
        assert abs(_z(measured, _slab_integral(_thin_second_moment))) < 3.0

    def test_bethe_heitler_straggles_less(self) -> None:
        """The screened shape puts less of the radiated energy into hard photons.

        Energy radiated per unit photon energy is flat for the 1/k spectrum and
        falls towards the tip for the screened shape, so at equal radiated energy
        the second moment — the straggling — is smaller. Asserted on the analytic
        expectations the two tests above tie the loop to: the sample itself
        resolves each to a few percent, not their difference to five sigma.
        """
        assert _slab_integral(_bh_second_moment) < 0.95 * _slab_integral(_thin_second_moment)


# -- emission angle ----------------------------------------------------------------


class TestEmissionAngle:
    def test_half_the_photons_leave_inside_the_one_over_gamma_cone(
        self, bethe_heitler: _SlabTally
    ) -> None:
        """The median of the angular law is cos(theta) = beta, exactly.

        Photons of electrons that have thrown no delta ray, in water that does not
        scatter: the angle to the beam axis is the emission angle. The cone is
        evaluated at the mid-slab energy; across the slab it moves by 7 percent,
        which shifts the inside fraction by under one percent.
        """
        xs = AnalyticCrossSections()
        total = _energy_at_depth(xs, 0.5 * SLAB_CM) + ELECTRON_MASS_MEV
        one_minus_beta = (ELECTRON_MASS_MEV / total) ** 2 / (
            1.0 + math.sqrt(1.0 - (ELECTRON_MASS_MEV / total) ** 2)
        )
        x = bethe_heitler.axis_one_minus_cos
        assert x.size > 500
        inside = int(np.count_nonzero(x < one_minus_beta))
        assert stats.binomtest(inside, x.size, 0.5).pvalue > 0.01

    def test_thin_target_photons_are_collinear(self, thin_target: _SlabTally) -> None:
        """The thin-target model emits along the electron: nothing leaves the axis."""
        assert thin_target.axis_one_minus_cos.size > 500
        assert np.all(thin_target.axis_one_minus_cos < 1.0e-9)


# -- the model is a different calculation, not an alias ----------------------------


def test_the_model_changes_the_transported_histories() -> None:
    """Same seed, different model: a different dose (and a recorded choice)."""
    grid = VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))
    engine = ReferenceEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )
    beam = ParallelBeamSource(energy=15.0, z=-0.5, x_range=(3.0, 5.0), y_range=(3.0, 5.0))
    common = dict(
        n_histories=40, n_batches=4, seed=SEED, primary_kind="electron", msc_model="gaussian"
    )
    thin = engine.run(beam, **common, physics=PhysicsOptions(bremsstrahlung="thin-target"))
    screened = engine.run(beam, **common, physics=PhysicsOptions(bremsstrahlung="bethe-heitler"))

    assert thin.provenance is not None
    assert screened.provenance is not None
    assert thin.provenance.physics.bremsstrahlung == "thin-target"
    assert screened.provenance.physics.bremsstrahlung == "bethe-heitler"
    assert thin.dose.tobytes() != screened.dose.tobytes()


def test_auto_selects_the_model_above_the_threshold_end_to_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 40 MeV source: ``"auto"`` resolves to the model, the run completes, the books close.

    Mechanics only, on the wide-domain instrument: the shipped analytic source
    refuses 40 MeV (pinned below), and a dose there needs a tabulated compile.
    """
    grid = VoxelGrid.uniform_water(shape=(1, 1, 40), spacing=(20.0, 20.0, 0.5))
    engine = ReferenceEngine(
        grid=grid,
        cross_sections=_WideAnalyticWater(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )
    beam = PencilBeamSource(energy=40.0, position=(10.0, 10.0, -0.5), direction=(0, 0, 1.0))
    with caplog.at_level(logging.INFO, logger="pyradmc.backends.ref.engine"):
        result = engine.run(
            beam,
            n_histories=80,
            n_batches=4,
            seed=SEED,
            primary_kind="electron",
            msc_model="gaussian",
        )

    assert "bremsstrahlung=bethe-heitler" in caplog.text
    assert "above the 30 MeV" in caplog.text
    assert result.provenance is not None
    assert result.provenance.physics == BETHE_HEITLER
    assert result.dose.max() > 0.0
    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=1.0e-9)


@pytest.mark.parametrize("energy", [40.0, 80.0, 200.0])
def test_the_analytic_source_refuses_a_run_above_its_domain(energy: float) -> None:
    """Fail closed: no model choice makes analytic data answer above 20 MeV."""
    grid = VoxelGrid.uniform_water(shape=(1, 1, 8), spacing=(20.0, 20.0, 0.5))
    engine = ReferenceEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )
    beam = PencilBeamSource(energy=energy, position=(10.0, 10.0, -0.5), direction=(0, 0, 1.0))
    for physics in ("auto", PhysicsOptions(bremsstrahlung="bethe-heitler")):
        with pytest.raises(ValueError, match="energy domain"):
            engine.run(
                beam,
                n_histories=4,
                n_batches=2,
                seed=SEED,
                primary_kind="electron",
                msc_model="gaussian",
                physics=physics,
            )


class _SignedElectronPencil(Source):
    """An 18 MeV pencil along +z, every primary at one signed statistical weight."""

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def max_energy(self) -> float:
        return 18.0

    def emit(self, rng_state: RNGState) -> Primary:
        return Primary(18.0, 4.0, 4.0, -1.0, 0.0, 0.0, 1.0, weight=self._weight)


@pytest.mark.parametrize("weight", [0.5, PHOTON_ROULETTE_WEIGHT_CAP])
def test_a_negative_weight_history_mirrors_the_positive_one_exactly(weight: float) -> None:
    """Weight -w against +w on one stream: the dose negates bit for bit, books close.

    The model's arm makes the same three sign-blind decisions the thin-target arm
    does (the deposit guard on the unweighted energy, a photon spawned for any
    nonzero weight, the roulette cap on ``abs(w)``); a ``w > 0`` test at any of
    them drops the negative history's photons or deposits and breaks both
    assertions. At the cap every roulette site is met.
    """
    grid = VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))
    engine = ReferenceEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )
    common = dict(
        n_histories=160,
        n_batches=4,
        seed=SEED,
        primary_kind="electron",
        msc_model="gaussian",
        physics=PhysicsOptions(bremsstrahlung="bethe-heitler"),
    )
    plus = engine.run(_SignedElectronPencil(weight), **common)
    minus = engine.run(_SignedElectronPencil(-weight), **common)

    assert np.any(plus.dose > 0.0)
    np.testing.assert_array_equal(minus.dose, -plus.dose)
    assert minus.energy_deposited == -plus.energy_deposited
    assert minus.energy_escaped == -plus.energy_escaped
    assert minus.energy_emitted < 0.0
    assert minus.energy_deposited + minus.energy_escaped == pytest.approx(
        minus.energy_emitted, rel=1.0e-9
    )
