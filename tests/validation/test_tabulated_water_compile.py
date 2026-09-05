"""Validation tier: the compiled water tabulated backend, end to end.

Compiles liquid water from the real EPDL and EEDL libraries under the default
berger-seltzer strategy, then checks the whole path the engine will use: the assembled
:class:`TabulatedData` loads through :class:`TabulatedCrossSections`, its photon totals
reproduce NIST XCOM, its electron stopping quantities match the analytic ICRU-37 backend
they are compiled from, its scattering power tracks the transport moment, and it survives a
save/load round trip through the on-disk format and flattens through ``build_tables``.

Needs both ``PYRADMC_EPDL_PATH`` and ``PYRADMC_EEDL_PATH``; otherwise it skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.interface import PhotonProcess
from pyradmc.data.materials import WATER
from pyradmc.data.tabulated.format import load_tables, save_tables
from pyradmc.data.tabulated.model import TabulatedData
from pyradmc.data.tabulated.precompile import ElectronStoppingStrategy, compile_water
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

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
def test_compiled_scattering_tracks_the_transport_moment(water: TabulatedData) -> None:
    """Scattering power (EEDL) sits within a factor of two of the analytic transport moment.

    EEDL supplies the tabulated source's nuclear moment and neglects an unresolved
    forward tail (``element_scattering_power``), reaching ~11 % at 20 MeV. Both
    sources add the same below-ECUT electron moment and exclude hard Moller events.
    Measured 2026-09-04, tabulated / analytic: 1.10 at 1 MeV, 1.12 at 6 MeV, 0.87 at
    20 MeV.
    """
    xs = TabulatedCrossSections(water)
    analytic = AnalyticCrossSections()
    for energy in (1.0, 6.0, 20.0):
        ratio = xs.scattering_power(energy, WATER, ECUT) / analytic.scattering_power(
            energy, WATER, ECUT
        )
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
    assert reloaded.coherent_x is not None and reloaded.coherent_cumulative is not None
    tables = TabulatedCrossSections(reloaded).build_tables(ecut=ECUT, pcut=0.05, e_max=30.0)
    assert tables.mu_compton.shape[0] == 1


@pytest.mark.validation
def test_compiled_coherent_scattering_is_forward_peaked(water: TabulatedData) -> None:
    """The EPDL form factor forward-peaks coherent scattering, increasingly with energy.

    Exercises the whole coherent chain end to end (EPDL MF=27 -> mix -> cumulative ->
    compile -> loader -> sampler): unlike the Thomson limit (mean cosine 0, mean angle
    90 deg), the atomic form factor concentrates coherent scattering forward, and more so
    as the accessible momentum transfer grows with energy. Measured mean scattering angle
    for water falls from ~35 deg at 30 keV to well under a degree at MeV energies.
    """
    xs = TabulatedCrossSections(water)
    state = HostRNG().init_state(SEED, 0)

    def mean_cos(energy: float, n: int = 20_000) -> float:
        samples = [xs.sample_coherent_cos_theta(energy, WATER, state) for _ in range(n)]
        return float(np.mean(samples))

    mu_soft = mean_cos(0.05)
    mu_hard = mean_cos(6.0)
    assert 0.0 < mu_soft < 0.99  # forward of Thomson (0), but not yet fully forward
    assert mu_hard > 0.999  # strongly forward at MeV energies
    assert mu_hard > mu_soft


@pytest.mark.validation
def test_eedl_strategy_collision_stopping_tracks_analytic() -> None:
    """The eedl strategy's restricted collision stopping is consistent with ICRU-37.

    Compiling with strategy='eedl' fills restricted collision and radiative stopping from
    EEDL (excitation MT=528 + ionization MT=534-572 integrated below the cut) instead of
    the analytic Berger-Seltzer form. EEDL's inelastic evaluation departs from ESTAR by a
    few percent (like its radiative one), so this gates consistency, not sub-percent
    accuracy; the berger-seltzer default remains the ESTAR-exact one.
    """
    epdl_text, eedl_text = _texts()
    data = compile_water(epdl_text, eedl_text, strategy=ElectronStoppingStrategy.EEDL, e_max=8.0)
    assert "eedl" in data.provenance and "EEDL restricted collision" in data.provenance
    xs = TabulatedCrossSections(data)
    analytic = AnalyticCrossSections()
    for energy in (0.5, 1.0, 3.0, 6.0):
        ratio = xs.restricted_stopping_power(energy, WATER, ECUT) / (
            analytic.restricted_stopping_power(energy, WATER, ECUT)
        )
        assert 0.93 < ratio < 1.07, f"eedl collision stopping ratio {ratio:.3f} at {energy} MeV"


@pytest.mark.validation
def test_tabulated_dose_agrees_across_backends(water: TabulatedData) -> None:
    """Ref and Warp transport of the compiled backend agree (chi-squared over high dose).

    The capstone for the coherent form-factor port: both backends now sample the same
    single-source form-factor coherent angle (Warp compiles it from the same physics
    file), so the whole tabulated transport is statistically equivalent across targets
    (AGENTS.md 2.3). Small phantom, KERMA mode to keep the reference cheap.
    """
    from pyradmc.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(2.0, 2.0, 1.0))
    xs = TabulatedCrossSections(water, geometry_densities=grid.max_density_by_material())
    source = ParallelBeamSource(energy=6.0, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0))

    ref = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
        source, n_histories=6_000, n_batches=12, seed=SEED, transport_electrons=False
    )
    warp = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        source, n_histories=24_000, n_batches=12, seed=SEED, transport_electrons=False
    )
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        warp.dose,
        warp.dose_sigma,
        warp.n_batches,
        mask=mask,
    )
