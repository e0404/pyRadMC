"""Validation tier: the compiled water tabulated backend, end to end.

Compiles liquid water from the real EPDL and EEDL libraries under the default
berger-seltzer strategy, then checks the whole path the engine will use: the assembled
:class:`TabulatedData` loads through :class:`TabulatedCrossSections`, its photon totals
reproduce NIST XCOM, its electron stopping quantities match the analytic ICRU-37 backend
they are compiled from, its scattering power tracks Highland, and it survives a
save/load round trip through the on-disk format and flattens through ``build_tables``.

Needs both ``PYRADMC_EPDL_PATH`` and ``PYRADMC_EEDL_PATH``; otherwise it skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.interface import PhotonProcess
from pyRadMC.data.materials import WATER
from pyRadMC.data.tabulated.format import load_tables, save_tables
from pyRadMC.data.tabulated.model import TabulatedData
from pyRadMC.data.tabulated.precompile import compile_water
from pyRadMC.data.tabulated.source import TabulatedCrossSections

NIST_WATER_MU_OVER_RHO = {1.0: 0.0707, 2.0: 0.0493, 6.0: 0.0277, 10.0: 0.0222, 15.0: 0.0194}
ECUT = 0.2


def _texts() -> tuple[str, str]:
    epdl = os.environ.get("PYRADMC_EPDL_PATH")
    eedl = os.environ.get("PYRADMC_EEDL_PATH")
    if not (epdl and Path(epdl).is_file() and eedl and Path(eedl).is_file()):
        pytest.skip("set PYRADMC_EPDL_PATH and PYRADMC_EEDL_PATH to run this validation")
    return Path(epdl).read_text(encoding="latin-1"), Path(eedl).read_text(encoding="latin-1")


@pytest.fixture(scope="module")
def water() -> TabulatedData:
    epdl_text, eedl_text = _texts()
    return compile_water(epdl_text, eedl_text, e_max=30.0)


@pytest.mark.validation
def test_compiled_structure(water: TabulatedData) -> None:
    """The compiled data is water on geometric grids with all four photon channels."""
    assert water.materials == ("water",)
    assert set(water.mu_over_rho) == {
        PhotonProcess.COMPTON,
        PhotonProcess.PHOTOELECTRIC,
        PhotonProcess.PAIR,
        PhotonProcess.RAYLEIGH,
    }
    for grid in (water.photon_energies, water.electron_energies):
        ratios = grid[1:] / grid[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)  # geometric


@pytest.mark.validation
def test_compiled_photons_match_nist(water: TabulatedData) -> None:
    """Loaded through the source, photon totals reproduce NIST XCOM to under 1 percent."""
    xs = TabulatedCrossSections(water)
    for energy, expected in NIST_WATER_MU_OVER_RHO.items():
        actual = xs.mu_over_rho_total(energy, WATER)
        assert abs(actual - expected) / expected < 0.01


@pytest.mark.validation
def test_compiled_electron_stopping_matches_analytic(water: TabulatedData) -> None:
    """Berger-seltzer electron quantities equal the analytic backend they come from."""
    xs = TabulatedCrossSections(water)
    analytic = AnalyticCrossSections()
    for energy in (0.5, 1.0, 5.0, 15.0):
        assert xs.restricted_stopping_power(energy, WATER, ECUT) == pytest.approx(
            analytic.restricted_stopping_power(energy, WATER, ECUT), rel=2e-3
        )
        assert xs.radiative_stopping_power(energy, WATER) == pytest.approx(
            analytic.radiative_stopping_power(energy, WATER), rel=2e-3
        )
        assert xs.csda_range(energy, WATER) == pytest.approx(
            analytic.csda_range(energy, WATER), rel=2e-3
        )


@pytest.mark.validation
def test_compiled_scattering_tracks_highland(water: TabulatedData) -> None:
    """Scattering power (EEDL) sits within a factor of two of analytic Highland."""
    xs = TabulatedCrossSections(water)
    analytic = AnalyticCrossSections()
    for energy in (1.0, 6.0, 20.0):
        ratio = xs.scattering_power(energy, WATER) / analytic.scattering_power(energy, WATER)
        assert 0.7 < ratio < 2.0


@pytest.mark.validation
def test_compiled_round_trips_and_flattens(water: TabulatedData, tmp_path: Path) -> None:
    """The compiled data survives the on-disk format and flattens through build_tables."""
    reloaded = load_tables(save_tables(water, tmp_path / "water.npz"))
    assert reloaded.materials == water.materials
    np.testing.assert_allclose(
        reloaded.mu_over_rho[PhotonProcess.COMPTON],
        water.mu_over_rho[PhotonProcess.COMPTON],
        rtol=1e-4,
    )
    tables = TabulatedCrossSections(reloaded).build_tables(ecut=ECUT, pcut=0.05, e_max=30.0)
    assert tables.mu_compton.shape[0] == 1
