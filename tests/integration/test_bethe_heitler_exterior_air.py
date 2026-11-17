"""The ``"bethe-heitler"`` model inside a long exterior-air step, and under a field.

Under a uniform field an electron in exterior air takes long exact-helix steps
(``pyradmc.geometry.exterior``): no gyration cap and no voxel face, the whole step
charged at its start voxel's density and material. What the model does there is
what it does in any substep, stated here because it is the place to get it wrong:

- the flight to a discrete event is sampled from ``rho (sigma_Moller + sigma_brems)``
  at the start voxel and competes with the step's safety and energy caps;
- only the radiative stopping power of photons *below* PCUT is charged continuously,
  over the step's path; photons above PCUT are the discrete events, so nothing is
  counted twice;
- a photon is emitted where the step ends, about the electron's direction there.

Pinned by comparing the long-step path against the ordinary stepping of the same
loop in the same medium, which the slab tests tie to the interface quantities: the
photons per electron, the energy they carry, and the energy deposited locally must
agree. The medium is water at 0.009 g/cm^3, just under the exterior-air density
bound, so that the path radiates measurably; 15 MeV electrons in 0.35 T
(gyroradius 15 cm) turn half a circle in it, 46 cm of path, and leave through the
face they entered by. The ordinary stepping takes that arc in gyration-capped
chords of 3 cm (69 deposits per electron), the long steps in 21. Measured
(6000 electrons a side, 2026-11-14): photons per electron 0.0775 +- 0.0036 against
0.0702 +- 0.0035, radiated energy 0.127 +- 0.010 against 0.123 +- 0.011 MeV, local
deposit 0.6971 +- 0.0019 against 0.6998 +- 0.0024 MeV. Statistical comparisons,
fixed seeds.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from pyradmc import ECUT_MEV, PCUT_MEV, PHOTON_ROULETTE_WEIGHT_CAP, PhysicsOptions
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.exterior import EXTERIOR_AIR_DENSITY_MAX, ExteriorAirMap
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource
from pyradmc.physics_options import CLINICAL_PHYSICS, ResolvedPhysics
from pyradmc.rng.host import HostRNG
from pyradmc.transport.electron import electron_steps
from pyradmc.transport.particles import PHOTON, StackEntry
from tests.conftest import SEED

BETHE_HEITLER = dataclasses.replace(CLINICAL_PHYSICS, bremsstrahlung="bethe-heitler")
DENSITY = 0.009
ENERGY_MEV = 15.0
FIELD = MagneticField.uniform((0.35, 0.0, 0.0))
SHAPE = (30, 30, 30)
SPACING = (4.0, 4.0, 4.0)  # a 120 cm box
N_ELECTRONS = 6_000


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
    count: np.ndarray  # photons above PCUT per electron
    radiated: np.ndarray  # their energy, MeV
    local: np.ndarray  # energy deposited in the medium, MeV
    substeps: float  # mean deposits per electron: how long the steps were
    balance_error: float

    def mean_and_error(self, values: np.ndarray) -> tuple[float, float]:
        return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def _cross(physics: ResolvedPhysics, long_steps: bool, stream: int) -> _Tally:
    grid = _thin_medium()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    safety = ExteriorAirMap.from_grid(grid).safety if long_steps else None
    if safety is not None:
        assert safety.max() > 20.0, "the medium was meant to be exterior"
    rng = HostRNG()
    weight = PHOTON_ROULETTE_WEIGHT_CAP  # at the cap: bremsstrahlung photons stay analog
    count = np.zeros(N_ELECTRONS)
    radiated = np.zeros(N_ELECTRONS)
    local = np.zeros(N_ELECTRONS)
    deposits = 0
    worst = 0.0
    for i in range(N_ELECTRONS):
        spawned: list[StackEntry] = []
        deposited = [0.0, 0]

        def deposit(x: float, y: float, z: float, energy: float, weighted: float) -> None:
            deposited[0] += energy  # noqa: B023 - consumed within this iteration
            deposited[1] += 1  # noqa: B023

        escaped = electron_steps(
            False, ENERGY_MEV, weight, 60.0, 60.0, 1.0e-6, 0.0, 0.0, 1.0,
            grid, xs, rng.init_state(SEED + stream, i), deposit, spawned.append,
            pcut=PCUT_MEV, ecut=ECUT_MEV, msc_model="gaussian", physics=physics,
            field=FIELD, exterior_safety=safety,
        )  # fmt: skip
        carried = 0.0
        for kind, energy, w, *_rest in spawned:
            carried += w * energy
            if kind == PHOTON:
                count[i] += 1.0
                radiated[i] += energy
        local[i] = deposited[0] / weight
        deposits += int(deposited[1])
        emitted = weight * ENERGY_MEV
        worst = max(worst, abs(emitted - (deposited[0] + escaped + carried)) / emitted)
    return _Tally(count, radiated, local, deposits / N_ELECTRONS, worst)


@pytest.fixture(scope="module")
def long_steps() -> _Tally:
    return _cross(BETHE_HEITLER, long_steps=True, stream=11)


@pytest.fixture(scope="module")
def ordinary() -> _Tally:
    return _cross(BETHE_HEITLER, long_steps=False, stream=12)


def _z(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) / math.hypot(a[1], b[1])


def test_the_long_step_path_is_the_one_under_test(long_steps: _Tally, ordinary: _Tally) -> None:
    """Far fewer, longer steps: the comparison is between two different step schemes."""
    assert long_steps.substeps < 0.5 * ordinary.substeps
    assert long_steps.count.sum() > 400  # enough photons for the comparisons below


def test_every_history_balances(long_steps: _Tally, ordinary: _Tally) -> None:
    assert long_steps.balance_error < 1.0e-12
    assert ordinary.balance_error < 1.0e-12


def test_photon_rate_is_that_of_ordinary_stepping(long_steps: _Tally, ordinary: _Tally) -> None:
    """Discrete channel: the event rate does not depend on how the path is cut."""
    z = _z(long_steps.mean_and_error(long_steps.count), ordinary.mean_and_error(ordinary.count))
    assert abs(z) < 3.0


def test_radiated_energy_is_that_of_ordinary_stepping(long_steps: _Tally, ordinary: _Tally) -> None:
    z = _z(
        long_steps.mean_and_error(long_steps.radiated),
        ordinary.mean_and_error(ordinary.radiated),
    )
    assert abs(z) < 3.0


def test_local_deposit_is_that_of_ordinary_stepping(long_steps: _Tally, ordinary: _Tally) -> None:
    """Continuous arm: collision loss plus the sub-PCUT radiative share, once.

    Charging the total radiative stopping power on top of the photons would raise
    the local deposit by the radiated energy, several times its statistical error.
    """
    a = long_steps.mean_and_error(long_steps.local)
    b = ordinary.mean_and_error(ordinary.local)
    assert abs(_z(a, b)) < 3.0
    radiated = float(ordinary.radiated.mean())
    assert radiated > 5.0 * math.hypot(a[1], b[1]), "no power against double counting"


def test_the_rate_is_the_interface_cross_section(long_steps: _Tally) -> None:
    """Photons per unit of energy lost locally, against the interface quantities.

    Both are path integrals over the same arc, so their ratio needs no path length:
    ``sigma_brems / (S_col + S_rad^<)``. The electron loses about 6 percent of its
    energy on the way, over which that ratio moves by a few percent; the measured
    value must lie between its values at the entrance and exit energies.
    """
    xs = AnalyticCrossSections()

    def ratio(energy: float) -> float:
        sigma, restricted = xs.bremsstrahlung_step_coefficients(energy, WATER, PCUT_MEV)
        # The local deposit is the restricted collision loss (Moller below ECUT
        # included) plus the sub-PCUT radiative share; delta rays are spawned.
        return sigma / (xs.restricted_stopping_power(energy, WATER, ECUT_MEV) + restricted)

    count = long_steps.mean_and_error(long_steps.count)
    local = float(long_steps.local.mean())
    exit_energy = ENERGY_MEV - local - float(long_steps.radiated.mean()) - 1.0
    bounds = sorted((ratio(ENERGY_MEV), ratio(exit_energy)))
    assert bounds[1] < 1.1 * bounds[0]
    slack = 3.0 * count[1] / local
    assert bounds[0] - slack < count[0] / local < bounds[1] + slack


@pytest.mark.warp
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_warp_books_close_and_match_the_reference_through_long_steps(device: str) -> None:
    """Engine level, exterior-air stepping on by default: books, and the deposited share.

    The reference and the kernel each run eight replicates; the energy deposited
    in the medium per emitted energy must agree within their spread.
    """
    wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _thin_medium()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    beam = PencilBeamSource(energy=ENERGY_MEV, position=(60.0, 60.0, -0.5), direction=(0, 0, 1.0))
    physics = PhysicsOptions(bremsstrahlung="bethe-heitler")
    with pytest.warns(UserWarning, match="exterior-air"):
        ref = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG(), magnetic_field=FIELD)
    with pytest.warns(UserWarning, match="exterior-air"):
        warp = WarpEngine(grid=grid, cross_sections=xs, device=device, magnetic_field=FIELD)

    def shares(engine, n: int, first_seed: int, **kwargs) -> np.ndarray:  # type: ignore[no-untyped-def]
        out = []
        for r in range(8):
            result = engine.run(
                beam,
                n_histories=n,
                n_batches=4,
                seed=first_seed + r,
                primary_kind="electron",
                physics=physics,
                **kwargs,
            )
            assert result.provenance is not None
            assert result.provenance.exterior_air is not None
            books = result.energy_deposited + result.energy_escaped + result.energy_unscored
            assert books == pytest.approx(result.energy_emitted, rel=1.0e-6)
            out.append(result.energy_deposited / result.energy_emitted)
        return np.array(out)

    a = shares(ref, 400, SEED, msc_model="gaussian")
    b = shares(warp, 20_000, SEED + 100, msc_model="gaussian")
    error = math.hypot(a.std(ddof=1), b.std(ddof=1)) / math.sqrt(8)
    assert abs(a.mean() - b.mean()) < 3.0 * error, (a.mean(), b.mean(), error)
