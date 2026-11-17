"""Pair production and positrons above the ``"auto"`` threshold, on tabulated data.

Above ``HIGH_ENERGY_PHYSICS_THRESHOLD_MEV`` (30 MeV) ``"auto"`` resolves all three
processes to their detailed models: screened Bethe-Heitler bremsstrahlung and pair
sharing, and the ``"bhabha-in-flight"`` positron. The analytic source ends at
20 MeV, so the only shipped route there is a tabulated compile to a higher
``e_max``; this file runs it at 40, 80 and 200 MeV in water compiled to 250 MeV
from the pinned EPICS 2023 libraries, on both backends:

- the pair cross-section the source reports up there is EPDL's total pair
  production (nuclear plus triplet, MT 516), not a fit: 0.0109, 0.0137 and
  0.0167 cm^2/g at 40, 80 and 200 MeV, 65, 81 and 92 percent of the total
  attenuation and rising toward the complete-screening limit ``(7/9) / X0``;
- a photon beam and a positron beam resolve to the three models and the energy
  books close, the positron's rest mass included;
- reference and Warp agree on the depth dose of a positron pencil beam, in the
  calibrated geometry of ``tests/integration/test_pair_positron_cross_backend.py``
  (one lateral column, depth bins above half the maximum, equal history counts).
  At these energies a positron radiates most of its energy and the photons make
  pairs, so all three arms are in the comparison.

What this certifies is the transport and the wiring of the models on data valid at
these energies. It is not a benchmark against an external code: none is on file
above 30 MeV.

Needs both ``PYRADMC_EPDL_PATH`` and ``PYRADMC_EEDL_PATH``; otherwise it skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyradmc import ELECTRON_MASS_MEV, HIGH_ENERGY_PHYSICS_THRESHOLD_MEV, PhysicsOptions
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.interface import PhotonProcess
from pyradmc.data.materials import WATER
from pyradmc.data.tabulated.model import TabulatedData
from pyradmc.data.tabulated.precompile import compile_water
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource, Primary, Source
from pyradmc.physics_options import ResolvedPhysics
from pyradmc.rng import RNGState
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.validation, pytest.mark.warp]

E_MAX_MEV = 250.0
WIDTH_CM = 60.0
N_Z = 60
HIGH_ENERGY = ResolvedPhysics(
    bremsstrahlung="bethe-heitler", pair="bethe-heitler", positron="bhabha-in-flight"
)
ENERGIES = [40.0, 80.0, 200.0]
RADIATION_LENGTH_WATER = 36.08  # g/cm^2 (Tsai 1974)


def _device() -> str:
    return "cuda:0" if wp.is_cuda_available() else "cpu"


@pytest.fixture(scope="module")
def water() -> TabulatedData:
    epdl = os.environ.get("PYRADMC_EPDL_PATH")
    eedl = os.environ.get("PYRADMC_EEDL_PATH")
    if not (epdl and Path(epdl).is_file() and eedl and Path(eedl).is_file()):
        pytest.skip("set PYRADMC_EPDL_PATH and PYRADMC_EEDL_PATH to run this validation")
    return compile_water(
        Path(epdl).read_text(encoding="latin-1"),
        Path(eedl).read_text(encoding="latin-1"),
        e_max=E_MAX_MEV,
    )


def _column(energy: float) -> VoxelGrid:
    """A laterally complete water column whose depth scales with the energy."""
    dz = 0.5 * energy / 40.0
    return VoxelGrid.uniform_water(shape=(1, 1, N_Z), spacing=(WIDTH_CM, WIDTH_CM, dz))


class _PositronPencil(Source):
    particle_kinds = frozenset({"positron"})

    def __init__(self, energy: float) -> None:
        self._energy = energy

    @property
    def max_energy(self) -> float:
        return self._energy

    def emit(self, rng_state: RNGState) -> Primary:
        centre = 0.5 * WIDTH_CM
        return Primary(self._energy, centre, centre, -0.5, 0.0, 0.0, 1.0, kind="positron")


def _photon_pencil(energy: float) -> PencilBeamSource:
    centre = 0.5 * WIDTH_CM
    return PencilBeamSource(energy=energy, position=(centre, centre, -0.5), direction=(0, 0, 1.0))


def _warp(grid: VoxelGrid, water: TabulatedData):  # type: ignore[no-untyped-def]
    from pyradmc.backends.warp.engine import WarpEngine

    return WarpEngine(grid=grid, cross_sections=TabulatedCrossSections(water), device=_device())


def _ref(grid: VoxelGrid, water: TabulatedData) -> ReferenceEngine:
    return ReferenceEngine(grid=grid, cross_sections=TabulatedCrossSections(water), rng=HostRNG())


def _books(result) -> float:  # type: ignore[no-untyped-def]
    total = result.energy_deposited + result.energy_escaped + result.energy_unscored
    return abs(total - result.energy_emitted) / abs(result.energy_emitted)


def test_the_pair_cross_section_is_the_librarys_up_to_the_domain(water: TabulatedData) -> None:
    """EPDL total pair production: the dominant channel, below the asymptote, rising."""
    xs = TabulatedCrossSections(water)
    assert xs.energy_max >= E_MAX_MEV
    asymptote = 7.0 / 9.0 / RADIATION_LENGTH_WATER
    previous = 0.0
    for energy, share in zip(ENERGIES, (0.65, 0.81, 0.92), strict=True):
        pair = xs.mu_over_rho(energy, WATER, PhotonProcess.PAIR)
        assert pair / xs.mu_over_rho_total(energy, WATER) == pytest.approx(share, abs=0.01)
        assert previous < pair < asymptote
        previous = pair
    assert xs.mu_over_rho(200.0, WATER, PhotonProcess.PAIR) / asymptote == pytest.approx(
        0.78, abs=0.01
    )


@pytest.mark.parametrize("energy", ENERGIES)
@pytest.mark.parametrize("primaries", ["photon", "positron"])
def test_auto_runs_the_three_models_and_the_books_close(
    water: TabulatedData, energy: float, primaries: str
) -> None:
    assert energy > HIGH_ENERGY_PHYSICS_THRESHOLD_MEV
    grid = _column(energy)
    source: Source = _photon_pencil(energy) if primaries == "photon" else _PositronPencil(energy)
    result = _warp(grid, water).run(source, n_histories=200_000, n_batches=10, seed=SEED)

    assert result.provenance is not None
    assert result.provenance.physics == HIGH_ENERGY
    assert _books(result) < 1.0e-6
    if primaries == "positron":
        per_history = result.energy_emitted / 200_000
        assert per_history == pytest.approx(energy + 2.0 * ELECTRON_MASS_MEV, rel=1.0e-6)

    reference = _ref(grid, water).run(
        source, n_histories=40, n_batches=4, seed=SEED, msc_model="gaussian"
    )
    assert reference.provenance is not None
    assert reference.provenance.physics == HIGH_ENERGY
    assert _books(reference) < 1.0e-9


@pytest.mark.parametrize("energy", ENERGIES)
def test_reference_and_warp_agree_on_a_positron_beam(water: TabulatedData, energy: float) -> None:
    """Equal history counts, Gaussian hinge on both (the reference GS tables are lazy).

    Measured at these seeds (2026-11-14), each against the null of its own geometry
    from 300 same-backend pairs: 0.74, 0.64 and 0.90 times the null mean at 40, 80
    and 200 MeV on 33, 30 and 24 bins (nominal p = 0.84, 0.91, 0.63; empirical
    p = 0.77, 0.85, 0.56). The nulls have means 0.99, 1.00 and 1.03 and put 5.3, 6.7
    and 9.3 percent of pairs below a nominal p of 0.01.
    """
    grid = _column(energy)
    common = dict(n_histories=2_000, n_batches=20, msc_model="gaussian")
    ref = _ref(grid, water).run(_PositronPencil(energy), seed=SEED, **common)
    other = _warp(grid, water).run(_PositronPencil(energy), seed=SEED + 1, **common)

    assert ref.provenance is not None
    assert ref.provenance.physics == HIGH_ENERGY
    mask = ref.dose > 0.5 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (other.dose_sigma > 0.0)
    assert np.count_nonzero(mask) >= 12
    assert np.all(ref.dose_sigma[mask] < 0.05 * ref.dose[mask])
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        other.dose,
        other.dose_sigma,
        other.n_batches,
        mask=mask,
    )


def test_the_switch_lowers_a_40_mev_photon_depth_dose(water: TabulatedData) -> None:
    """What ``"auto"`` selecting the pair and positron models does to a dose (AGENTS.md 2.10).

    40 MeV photons, ``"auto"`` against the same run with the clinical pair and
    positron models named (bremsstrahlung ``bethe-heitler`` on both sides), eight
    replicates each. Measured with 1e6 histories per replicate (2026-11-14), in
    percent of the maximum dose: -0.57 averaged over the build-up, -1.16 behind
    the maximum, at most -1.66 +- 0.12; of which the positron model alone is
    -1.40 and -1.32 and the pair model alone +0.75 and +0.07. At 80 MeV: -0.92,
    -1.05, at most -1.78 +- 0.09. The gate is the sign, the significance and the
    size class of the change behind the maximum.
    """
    grid = _column(40.0)
    engine = _warp(grid, water)
    doses = {}
    for m, physics in enumerate(
        (PhysicsOptions(), PhysicsOptions(pair="uniform-split", positron="moller-at-rest"))
    ):
        doses[m] = np.array(
            [
                engine.run(
                    _photon_pencil(40.0),
                    n_histories=400_000,
                    n_batches=4,
                    seed=SEED + 1_000 * m + r,
                    physics=physics,
                ).dose[0, 0]
                for r in range(8)
            ]
        )
    clinical = doses[1].mean(axis=0)
    behind = slice(int(np.argmax(clinical)), None)
    per_replicate = [
        100.0 * (d[:, behind] - clinical[behind]).mean(axis=1) / clinical.max()
        for d in doses.values()
    ]
    change = float(per_replicate[0].mean())
    error = float(np.hypot(per_replicate[0].std(ddof=1), per_replicate[1].std(ddof=1)) / np.sqrt(8))
    print(f"\n40 MeV photons, behind the maximum: {change:+.2f} +- {error:.2f} % of the maximum")
    assert change / error < -5.0
    assert -2.0 < change < -0.5
