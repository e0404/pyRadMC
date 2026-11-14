"""Bremsstrahlung above the ``"auto"`` threshold, on data whose domain reaches there.

The analytic source ends at 20 MeV, so the only shipped route to a source above
``HIGH_ENERGY_PHYSICS_THRESHOLD_MEV`` (30 MeV) is a tabulated compile to a higher
``e_max``: its grids are append-only above the 30 MeV reference span, and its
radiative stopping power follows ESTAR to 1 GeV. This file runs that route end to
end at 40, 80 and 200 MeV in water, compiled here to 250 MeV from the pinned EPICS
2023 libraries:

- ``"auto"`` resolves to ``"bethe-heitler"``, the run completes and the books close;
- the entrance bin holds a collision loss. With the restricted-range map clamped at
  64 MeV (before the energy domains) an 80 MeV electron deposited 16 MeV there;
- reference and Warp agree at 40 MeV under the model, in the calibrated geometry
  of ``tests/integration/test_bethe_heitler_warp.py`` (a pencil beam in one lateral
  column, depth bins above half the maximum, equal history counts);
- the dosimetric effect of what ``"auto"`` selects there (AGENTS.md 2.10): the
  depth of half-maximum dose under ``bethe-heitler`` against ``thin-target``.

What this certifies is the transport and the model's wiring on data that is valid
at these energies for electrons (Berger-Seltzer collision stopping, ESTAR radiative
stopping). It is not a benchmark against an external code: none is on file above
30 MeV, and the pair and positron models are still the clinical ones.

Needs both ``PYRADMC_EPDL_PATH`` and ``PYRADMC_EEDL_PATH``; otherwise it skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyradmc import ECUT_MEV, HIGH_ENERGY_PHYSICS_THRESHOLD_MEV, PhysicsOptions
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.materials import WATER
from pyradmc.data.tabulated.model import TabulatedData
from pyradmc.data.tabulated.precompile import compile_water
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.validation, pytest.mark.warp]

E_MAX_MEV = 250.0
WIDTH_CM = 60.0
N_Z = 60


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


def _column(energy: float) -> tuple[VoxelGrid, float]:
    """A laterally complete water column whose depth scales with the electron range."""
    dz = 0.5 * energy / 40.0
    return VoxelGrid.uniform_water(shape=(1, 1, N_Z), spacing=(WIDTH_CM, WIDTH_CM, dz)), dz


def _beam(energy: float) -> PencilBeamSource:
    centre = 0.5 * WIDTH_CM
    return PencilBeamSource(energy=energy, position=(centre, centre, -0.5), direction=(0, 0, 1.0))


def _warp(grid: VoxelGrid, water: TabulatedData):  # type: ignore[no-untyped-def]
    from pyradmc.backends.warp.engine import WarpEngine

    return WarpEngine(grid=grid, cross_sections=TabulatedCrossSections(water), device=_device())


@pytest.mark.parametrize("energy", [40.0, 80.0, 200.0])
def test_auto_runs_the_model_on_tabulated_data(water: TabulatedData, energy: float) -> None:
    assert energy > HIGH_ENERGY_PHYSICS_THRESHOLD_MEV
    grid, dz = _column(energy)
    xs = TabulatedCrossSections(water)
    assert xs.energy_max >= E_MAX_MEV
    result = _warp(grid, water).run(
        _beam(energy), n_histories=200_000, n_batches=10, seed=SEED, primary_kind="electron"
    )

    assert result.provenance is not None
    assert result.provenance.physics.bremsstrahlung == "bethe-heitler"
    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=1e-6)

    # Entrance bin: the restricted collision loss across it plus the local share of
    # the delta rays, measured 1.11 to 1.12 of S_restricted dz at all three
    # energies; an energy dump at a range clamp would be many times that.
    per_electron = float(result.dose[0, 0, 0]) * WIDTH_CM * WIDTH_CM * dz  # MeV/g -> MeV
    collision = xs.restricted_stopping_power(energy, WATER, ECUT_MEV) * dz
    assert 1.0 < per_electron / collision < 1.25, per_electron / collision


def test_reference_and_warp_agree_at_40_mev(water: TabulatedData) -> None:
    """Equal history counts, Gaussian hinge on both (the reference GS tables are lazy).

    Measured at these seeds: chi-squared 1.47 times its null mean on 33 bins,
    nominal p = 0.034, in an oracle whose null rejects about five times too often
    at this bin count (``tests/integration/test_bethe_heitler_warp.py``). Followed
    up with 40 independent replicates a side (120 000 histories each, per-bin
    errors from the replicate spread, 2026-11-14): chi-squared 44.7 on the same
    33 bins (p = 0.085), no bin beyond 0.5 % of the maximum, R50 16.371 +- 0.014
    against 16.354 +- 0.009 cm, deposited share 0.8969 +- 0.0004 against
    0.8973 +- 0.0004. The thin-target model in the same comparison: 22.7 (p = 0.91).
    """
    grid, _ = _column(40.0)
    common = dict(
        n_histories=3_000,
        n_batches=20,
        primary_kind="electron",
        msc_model="gaussian",
    )
    ref = ReferenceEngine(
        grid=grid, cross_sections=TabulatedCrossSections(water), rng=HostRNG()
    ).run(_beam(40.0), seed=SEED, **common)
    other = _warp(grid, water).run(_beam(40.0), seed=SEED + 1, **common)

    assert ref.provenance is not None
    assert ref.provenance.physics.bremsstrahlung == "bethe-heitler"
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


def _r50(dose: np.ndarray, dz: float) -> float:
    depth = (np.arange(dose.size) + 0.5) * dz
    peak = int(np.argmax(dose))
    half = 0.5 * dose[peak]
    below = peak + int(np.argmax(dose[peak:] < half))
    d0, d1 = dose[below - 1], dose[below]
    return float(depth[below - 1] + (d0 - half) / (d0 - d1) * dz)


def test_the_switch_moves_the_electron_range_shallower(water: TabulatedData) -> None:
    """R50 at 40 MeV: ``bethe-heitler`` against ``thin-target``, eight replicates each.

    Same radiated energy, less straggling: the screened spectrum spends it in
    more, softer photons, so the depth of half-maximum dose moves shallower.
    Measured here -2.83 +- 0.07 mm (2e5 histories per replicate, 2026-11-14). The
    gate is the sign, the significance and the size class.
    """
    grid, dz = _column(40.0)
    engine = _warp(grid, water)
    r50 = {}
    for m, model in enumerate(("thin-target", "bethe-heitler")):
        r50[model] = np.array(
            [
                _r50(
                    engine.run(
                        _beam(40.0),
                        n_histories=200_000,
                        n_batches=4,
                        seed=SEED + 1_000 * m + r,
                        primary_kind="electron",
                        physics=PhysicsOptions(bremsstrahlung=model),
                    ).dose[0, 0],
                    dz,
                )
                for r in range(8)
            ]
        )
    shift = float(r50["bethe-heitler"].mean() - r50["thin-target"].mean())
    error = float(
        np.hypot(r50["bethe-heitler"].std(ddof=1), r50["thin-target"].std(ddof=1)) / np.sqrt(8)
    )
    print(f"\nR50 shift at 40 MeV: {10 * shift:+.2f} +- {10 * error:.2f} mm")
    assert shift / error < -5.0
    assert -0.5 < shift < -0.1  # cm
