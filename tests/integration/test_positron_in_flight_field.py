"""The ``"bhabha-in-flight"`` positron in a magnetic field and on long exterior-air steps.

The field code already turned a positron the other way; the model adds nothing
there. What it adds is two more discrete channels to every substep, including the
long exact-helix steps a charged particle takes in exterior air under a uniform
field (``pyradmc.geometry.exterior``), where a substep is tens of centimetres and
the whole of it is charged at its start voxel. Stated here because it is the place
to get it wrong:

- the flight to a discrete event is sampled from
  ``rho (sigma_Bhabha + sigma_annihilation [+ sigma_brems])`` at the start voxel
  and competes with the step's safety and energy caps;
- an annihilation happens where the step ends: the photons start at the chord
  end, inside the grid, and the positron's 2 m_e c^2 leaves with them, once;
- a positron that leaves the grid un-annihilated still carries its 2 m_e c^2 out.

Pinned as the bremsstrahlung model's long steps are
(``test_bethe_heitler_exterior_air.py``): the long-step path against the ordinary
stepping of the same loop in the same medium, which the slab tests tie to the
interface. The medium is water at 0.009 g/cm^3, just under the exterior-air
density bound; 15 MeV positrons in 0.35 T turn half a circle in it, 46 cm of
path, and leave through the face they entered by. Statistical comparisons, fixed
seeds.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

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
from pyradmc.geometry.exterior import EXTERIOR_AIR_DENSITY_MAX, ExteriorAirMap
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import Primary, Source
from pyradmc.physics_options import CLINICAL_PHYSICS
from pyradmc.rng import RNGState
from pyradmc.rng.host import HostRNG
from pyradmc.transport.electron import electron_steps
from pyradmc.transport.particles import ELECTRON, PHOTON, StackEntry
from tests.conftest import SEED

IN_FLIGHT = dataclasses.replace(CLINICAL_PHYSICS, positron="bhabha-in-flight")
DENSITY = 0.009
ENERGY_MEV = 15.0
TWO_M = 2.0 * ELECTRON_MASS_MEV
FIELD = MagneticField.uniform((0.35, 0.0, 0.0))
SHAPE = (30, 30, 30)
SPACING = (4.0, 4.0, 4.0)  # a 120 cm box
N_POSITRONS = 12_000


def _thin_medium() -> VoxelGrid:
    assert DENSITY < EXTERIOR_AIR_DENSITY_MAX
    return VoxelGrid(
        shape=SHAPE,
        spacing=SPACING,
        density=np.full(SHAPE, DENSITY),
        material=np.full(SHAPE, WATER, dtype=np.int32),
    )


@dataclasses.dataclass(frozen=True)
class _Tally:
    in_flight: np.ndarray  # 1 where the positron annihilated in flight
    deltas: np.ndarray  # delta rays per positron
    local: np.ndarray  # energy deposited in the medium, MeV
    substeps: float  # mean deposits per positron: how long the steps were
    photons_inside: bool  # every annihilation photon started inside the grid
    balance_error: float

    def mean_and_error(self, values: np.ndarray) -> tuple[float, float]:
        return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def _cross(long_steps: bool, stream: int) -> _Tally:
    grid = _thin_medium()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    safety = ExteriorAirMap.from_grid(grid).safety if long_steps else None
    if safety is not None:
        assert safety.max() > 20.0, "the medium was meant to be exterior"
    rng = HostRNG()
    weight = PHOTON_ROULETTE_WEIGHT_CAP  # at the cap: no photon plays roulette
    in_flight = np.zeros(N_POSITRONS)
    deltas = np.zeros(N_POSITRONS)
    local = np.zeros(N_POSITRONS)
    deposits = 0
    worst = 0.0
    inside = True
    for i in range(N_POSITRONS):
        spawned: list[StackEntry] = []
        deposited = [0.0, 0]

        def deposit(x: float, y: float, z: float, energy: float, weighted: float) -> None:
            deposited[0] += energy  # noqa: B023 - consumed within this iteration
            deposited[1] += 1  # noqa: B023

        escaped = electron_steps(
            True, ENERGY_MEV, weight, 60.0, 60.0, 1.0e-6, 0.0, 0.0, 1.0,
            grid, xs, rng.init_state(SEED + stream, i), deposit, spawned.append,
            pcut=PCUT_MEV, ecut=ECUT_MEV, msc_model="gaussian", physics=IN_FLIGHT,
            field=FIELD, exterior_safety=safety,
        )  # fmt: skip
        carried = sum(entry[2] * entry[1] for entry in spawned)
        deltas[i] = sum(1 for entry in spawned if entry[0] == ELECTRON)
        last_two = spawned[-2:]
        if (
            len(last_two) == 2
            and all(entry[0] == PHOTON for entry in last_two)
            and last_two[0][3:6] == last_two[1][3:6]
            and escaped < weight * TWO_M
        ):
            # Ended by an annihilation. At rest only after a Bhabha event that took
            # nearly everything (two photons of exactly m_e c^2); else in flight.
            inside &= grid.contains(*last_two[0][3:6])
            if not last_two[0][1] == last_two[1][1] == ELECTRON_MASS_MEV:
                assert sum(entry[1] for entry in last_two) > TWO_M + ECUT_MEV
                in_flight[i] = 1.0
        local[i] = deposited[0] / weight
        deposits += int(deposited[1])
        emitted = weight * (ENERGY_MEV + TWO_M)
        worst = max(worst, abs(emitted - (deposited[0] + escaped + carried)) / emitted)
    return _Tally(in_flight, deltas, local, deposits / N_POSITRONS, inside, worst)


@pytest.fixture(scope="module")
def long_steps() -> _Tally:
    return _cross(long_steps=True, stream=11)


@pytest.fixture(scope="module")
def ordinary() -> _Tally:
    return _cross(long_steps=False, stream=12)


def _z(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) / math.hypot(a[1], b[1])


def test_the_long_step_path_is_the_one_under_test(long_steps: _Tally, ordinary: _Tally) -> None:
    assert long_steps.substeps < 0.5 * ordinary.substeps
    assert long_steps.in_flight.sum() > 20  # annihilations at the end of a long step


def test_every_history_balances(long_steps: _Tally, ordinary: _Tally) -> None:
    """T + 2 m in, once out: whether the positron annihilated on the arc or left the grid."""
    assert long_steps.balance_error < 1.0e-12
    assert ordinary.balance_error < 1.0e-12


def test_annihilation_photons_start_inside_the_grid(long_steps: _Tally, ordinary: _Tally) -> None:
    assert long_steps.photons_inside
    assert ordinary.photons_inside


def test_annihilation_rate_is_that_of_ordinary_stepping(
    long_steps: _Tally, ordinary: _Tally
) -> None:
    """The channel's rate does not depend on how the path is cut."""
    z = _z(
        long_steps.mean_and_error(long_steps.in_flight),
        ordinary.mean_and_error(ordinary.in_flight),
    )
    assert abs(z) < 3.0


def test_annihilation_rate_is_the_interface_cross_section(long_steps: _Tally) -> None:
    """Annihilations per unit of energy lost locally: ``sigma_ann / S+_restricted``.

    Both are path integrals over the same arc, so their ratio needs no path
    length. The positron loses a few percent of its energy on the way, over which
    the ratio moves by less; the measured value must lie between its values at the
    entrance energy and 2 MeV below it.
    """
    xs = AnalyticCrossSections()

    def ratio(energy: float) -> float:
        return xs.annihilation_cross_section(energy, WATER) / (
            xs.positron_restricted_stopping_power(energy, WATER, ECUT_MEV)
            + xs.radiative_stopping_power(energy, WATER) * PCUT_MEV / energy
        )

    rate = long_steps.mean_and_error(long_steps.in_flight)
    # Per positron that completes the arc; the few that annihilate stop depositing,
    # a correction of the order of the rate itself (under one percent).
    local = float(long_steps.local.mean())
    bounds = sorted((ratio(ENERGY_MEV), ratio(ENERGY_MEV - 2.0)))
    assert bounds[1] < 1.15 * bounds[0]
    slack = 3.0 * rate[1] / local
    assert bounds[0] * 0.98 - slack < rate[0] / local < bounds[1] + slack


def test_delta_ray_rate_is_that_of_ordinary_stepping(long_steps: _Tally, ordinary: _Tally) -> None:
    z = _z(long_steps.mean_and_error(long_steps.deltas), ordinary.mean_and_error(ordinary.deltas))
    assert abs(z) < 3.0


def test_local_deposit_is_that_of_ordinary_stepping(long_steps: _Tally, ordinary: _Tally) -> None:
    """Continuous arm: the positron's own restricted collision loss, on either path."""
    z = _z(long_steps.mean_and_error(long_steps.local), ordinary.mean_and_error(ordinary.local))
    assert abs(z) < 3.0


class _PositronPencil(Source):
    particle_kinds = frozenset({"positron"})

    @property
    def max_energy(self) -> float:
        return ENERGY_MEV

    def emit(self, rng_state: RNGState) -> Primary:
        return Primary(ENERGY_MEV, 60.0, 60.0, -0.5, 0.0, 0.0, 1.0, kind="positron")


@pytest.mark.warp
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("bremsstrahlung", ["thin-target", "bethe-heitler"])
def test_warp_books_close_and_match_the_reference_through_long_steps(
    device: str, bremsstrahlung: str
) -> None:
    """Engine level, exterior-air stepping on by default: books, and the deposited share.

    The reference and the kernel each run eight replicates; the energy deposited
    in the medium per emitted energy must agree within their spread. The emitted
    energy counts the positron's 2 m_e c^2, so a rest mass booked twice or not at
    all on the long-step path opens the books by several percent.
    """
    wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _thin_medium()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    physics = PhysicsOptions(positron="bhabha-in-flight", bremsstrahlung=bremsstrahlung)
    with pytest.warns(UserWarning, match="exterior-air"):
        ref = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG(), magnetic_field=FIELD)
    with pytest.warns(UserWarning, match="exterior-air"):
        warp = WarpEngine(grid=grid, cross_sections=xs, device=device, magnetic_field=FIELD)

    def shares(engine, n: int, first_seed: int) -> np.ndarray:  # type: ignore[no-untyped-def]
        out = []
        for r in range(8):
            result = engine.run(
                _PositronPencil(),
                n_histories=n,
                n_batches=4,
                seed=first_seed + r,
                physics=physics,
                msc_model="gaussian",
            )
            assert result.provenance is not None
            assert result.provenance.exterior_air is not None
            books = result.energy_deposited + result.energy_escaped + result.energy_unscored
            assert books == pytest.approx(result.energy_emitted, rel=1.0e-6)
            out.append(result.energy_deposited / result.energy_emitted)
        return np.array(out)

    a = shares(ref, 400, SEED)
    b = shares(warp, 20_000, SEED + 100)
    error = math.hypot(a.std(ddof=1), b.std(ddof=1)) / math.sqrt(8)
    assert abs(a.mean() - b.mean()) < 3.0 * error, (a.mean(), b.mean(), error)
