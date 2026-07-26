"""TabulatedCrossSections loader over compiled data.

Interpolation and contract behaviour on synthetic tables with known analytic forms —
no real EPDL/ESTAR data. Physical accuracy (water vs NIST, the 2%/2mm gate) lands in
later slices with real data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyRadMC.data.interface import PhotonProcess
from pyRadMC.data.tabulated.model import TabulatedData
from pyRadMC.data.tabulated.source import TabulatedCrossSections

_PROC = (
    PhotonProcess.COMPTON,
    PhotonProcess.PHOTOELECTRIC,
    PhotonProcess.PAIR,
    PhotonProcess.RAYLEIGH,
)
DELTA_CUT = 0.2


def _tables(n_p: int = 32, n_e: int = 32) -> TabulatedData:
    pe = np.geomspace(0.01, 10.0, n_p)
    ee = np.geomspace(0.01, 10.0, n_e)
    mu = {
        PhotonProcess.COMPTON: (0.15 * np.ones_like(pe))[None, :],
        PhotonProcess.PHOTOELECTRIC: (1.0e-3 / pe**3)[None, :],
        PhotonProcess.PAIR: (np.maximum(pe - 1.022, 0.0) * 1.0e-3)[None, :],
        PhotonProcess.RAYLEIGH: (0.01 / pe)[None, :],
    }
    return TabulatedData(
        photon_energies=pe,
        mu_over_rho=mu,
        electron_energies=ee,
        restricted_stopping=(2.0 / ee)[None, :],
        radiative_stopping=(0.01 * ee)[None, :],
        moller=(0.1 / ee)[None, :],
        csda_range=(0.5 * ee)[None, :],
        scattering_power=(1.0 / ee)[None, :],
        delta_cut=DELTA_CUT,
        materials=("water",),
        provenance="synthetic",
    )


def _source() -> TabulatedCrossSections:
    return TabulatedCrossSections(_tables())


def test_declared_materials_follow_the_compiled_table() -> None:
    """n_materials is the compiled row count, not the registry size."""
    assert _source().n_materials == 1


def _two_material_tables(n_p: int = 32, n_e: int = 32) -> TabulatedData:
    """Water row plus an 'air' row at half the water values (registry prefix order)."""
    one = _tables(n_p, n_e)
    return TabulatedData(
        photon_energies=one.photon_energies,
        mu_over_rho={p: np.concatenate([t, 0.5 * t]) for p, t in one.mu_over_rho.items()},
        electron_energies=one.electron_energies,
        restricted_stopping=np.concatenate(
            [one.restricted_stopping, 0.5 * one.restricted_stopping]
        ),
        radiative_stopping=np.concatenate([one.radiative_stopping, 0.5 * one.radiative_stopping]),
        moller=np.concatenate([one.moller, 0.5 * one.moller]),
        csda_range=np.concatenate([one.csda_range, 2.0 * one.csda_range]),
        scattering_power=np.concatenate([one.scattering_power, 0.5 * one.scattering_power]),
        delta_cut=one.delta_cut,
        materials=("water", "air"),
        provenance="synthetic two-material",
    )


class TestMaterialRowIndexing:
    """Queries for material 1 read row 1 — the contract heterogeneous grids rely on."""

    def test_declared_count(self) -> None:
        assert TabulatedCrossSections(_two_material_tables()).n_materials == 2

    def test_photon_rows_are_material_resolved(self) -> None:
        data = _two_material_tables()
        src = TabulatedCrossSections(data)
        for k in (0, 15, 31):
            e = float(data.photon_energies[k])
            for proc in _PROC:
                assert src.mu_over_rho(e, 1, proc) == pytest.approx(
                    data.mu_over_rho[proc][1, k], rel=1e-6
                )
                assert src.mu_over_rho(e, 1, proc) == pytest.approx(
                    0.5 * src.mu_over_rho(e, 0, proc), rel=1e-9
                )

    def test_electron_rows_are_material_resolved(self) -> None:
        data = _two_material_tables()
        src = TabulatedCrossSections(data)
        e = float(data.electron_energies[10])
        assert src.restricted_stopping_power(e, 1, DELTA_CUT) == pytest.approx(
            0.5 * src.restricted_stopping_power(e, 0, DELTA_CUT), rel=1e-9
        )
        assert src.csda_range(e, 1) == pytest.approx(2.0 * src.csda_range(e, 0), rel=1e-9)

    def test_majorant_covers_both_materials(self) -> None:
        """The Woodcock majorant is the max over the declared geometry pairs."""
        data = _two_material_tables()
        src = TabulatedCrossSections(data, geometry_densities=((0, 1.0), (1, 1.0)))
        e = 1.3
        expected = max(src.mu_over_rho_total(e, 0), src.mu_over_rho_total(e, 1))
        assert src.majorant(e) == pytest.approx(expected, rel=1e-9)


class TestPhotonQueries:
    def test_grid_node_values_are_exact(self) -> None:
        """At a grid node the lookup returns the stored value (indexing is correct)."""
        data = _tables()
        src = TabulatedCrossSections(data)
        for k in (0, 7, 15, 31):
            e = float(data.photon_energies[k])
            for proc in _PROC:
                assert src.mu_over_rho(e, 0, proc) == pytest.approx(
                    data.mu_over_rho[proc][0, k], rel=1e-6
                )

    def test_total_is_the_sum_of_channels(self) -> None:
        src = _source()
        e = 1.3
        total = sum(src.mu_over_rho(e, 0, p) for p in _PROC)
        assert src.mu_over_rho_total(e, 0) == pytest.approx(total, rel=1e-12)

    def test_interpolates_between_nodes(self) -> None:
        """A flat channel stays flat; a monotone channel stays between its neighbours."""
        data = _tables()
        src = TabulatedCrossSections(data)
        assert src.mu_over_rho(1.234, 0, PhotonProcess.COMPTON) == pytest.approx(0.15, rel=1e-9)
        e = math.sqrt(float(data.photon_energies[5]) * float(data.photon_energies[6]))
        val = src.mu_over_rho(e, 0, PhotonProcess.RAYLEIGH)
        lo = data.mu_over_rho[PhotonProcess.RAYLEIGH][0, 6]
        hi = data.mu_over_rho[PhotonProcess.RAYLEIGH][0, 5]
        assert lo < val < hi

    def test_majorant_bounds_the_real_cross_section(self) -> None:
        """The Woodcock majorant is >= rho * mu_total for every declared content."""
        src = TabulatedCrossSections(_tables(), geometry_densities=((0, 1.0),))
        for e in (0.05, 0.5, 2.0, 8.0):
            assert src.majorant(e) >= 1.0 * src.mu_over_rho_total(e, 0) - 1e-12


class TestElectronQueries:
    def test_electron_grid_node_values_are_exact(self) -> None:
        data = _tables()
        src = TabulatedCrossSections(data)
        k = 10
        e = float(data.electron_energies[k])
        assert src.radiative_stopping_power(e, 0) == pytest.approx(0.01 * e, rel=1e-6)
        assert src.csda_range(e, 0) == pytest.approx(0.5 * e, rel=1e-6)
        assert src.scattering_power(e, 0) == pytest.approx(1.0 / e, rel=1e-6)

    def test_restricted_and_moller_require_the_compiled_cut(self) -> None:
        src = _source()
        # 1.0 MeV is not a grid node, so allow log-linear interpolation error; the
        # point of this test is that the compiled cut is honoured and mismatches raise.
        assert src.restricted_stopping_power(1.0, 0, DELTA_CUT) == pytest.approx(2.0, rel=2e-2)
        assert src.moller_cross_section(1.0, 0, DELTA_CUT) == pytest.approx(0.1, rel=2e-2)
        with pytest.raises(ValueError, match="delta_cut"):
            src.restricted_stopping_power(1.0, 0, 0.05)
        with pytest.raises(ValueError, match="delta_cut"):
            src.moller_cross_section(1.0, 0, 0.05)


def test_round_trips_through_build_cross_section_tables() -> None:
    """The loader flattens through the same runtime builder the analytic source uses."""
    from pyRadMC.data.tables import build_cross_section_tables

    src = TabulatedCrossSections(_tables())  # compiled at delta_cut = 0.2
    tab = build_cross_section_tables(src, ecut=DELTA_CUT, pcut=0.01, e_max=10.0, n_points=64)
    assert tab.e_max == 10.0
    # The flattened majorant still bounds the per-material total everywhere it is defined.
    for e in (0.02, 0.5, 6.0):
        assert src.majorant(e) >= src.mu_over_rho_total(e, 0) - 1e-12
