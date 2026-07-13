"""Validation tier: multi-material tabulated compilation, end to end.

Compiles the whole registry (water, air, lung, adipose, cortical bone) from the real
EPDL and EEDL libraries and gates, per material: photon totals against NIST XCOM
mixtures (air and cortical bone — the extreme-Z media; water's gate lives in
``test_epdl_water_nist.py``), the loaded electron stopping against the same
Berger-Seltzer machinery the registry data feeds, physically-ordered scattering
powers, and the format round trip at five materials.

XCOM reference values were generated 2026-07-13 from the mixture interface
(https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html) with exactly the registry
mass fractions; totals *include* coherent scattering, matching the source's
channel-complete ``mu_over_rho_total``. Verify against XCOM before trusting them
further.

Needs both ``PYRADMC_EPDL_PATH`` and ``PYRADMC_EEDL_PATH``; otherwise it skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyRadMC.data import berger_seltzer
from pyRadMC.data.materials import ADIPOSE, AIR, CORTICAL_BONE, LUNG, MATERIALS, WATER
from pyRadMC.data.tabulated.format import load_tables, save_tables
from pyRadMC.data.tabulated.model import TabulatedData
from pyRadMC.data.tabulated.precompile import compile_materials
from pyRadMC.data.tabulated.source import TabulatedCrossSections

ECUT = 0.2

# Energy (MeV) -> total mass attenuation with coherent (cm^2/g), NIST XCOM mixtures.
XCOM_TOTAL_MU_OVER_RHO: dict[int, dict[float, float]] = {
    AIR: {1.0: 6.358e-2, 2.0: 4.447e-2, 6.0: 2.522e-2, 10.0: 2.045e-2, 15.0: 1.810e-2},
    CORTICAL_BONE: {1.0: 6.649e-2, 2.0: 4.663e-2, 6.0: 2.754e-2, 10.0: 2.320e-2, 15.0: 2.129e-2},
}

PHOTON_TOLERANCE = 0.01  # same gate as the water EPDL validation


def _texts() -> tuple[str, str]:
    epdl = os.environ.get("PYRADMC_EPDL_PATH")
    eedl = os.environ.get("PYRADMC_EEDL_PATH")
    if not (epdl and Path(epdl).is_file() and eedl and Path(eedl).is_file()):
        pytest.skip("set PYRADMC_EPDL_PATH and PYRADMC_EEDL_PATH to run this validation")
    return Path(epdl).read_text(encoding="latin-1"), Path(eedl).read_text(encoding="latin-1")


@pytest.fixture(scope="module")
def compiled() -> TabulatedData:
    epdl_text, eedl_text = _texts()
    return compile_materials(epdl_text, eedl_text, e_max=30.0)


@pytest.mark.validation
def test_compiled_rows_are_the_registry(compiled: TabulatedData) -> None:
    """Rows align with registry indices — the contract geometry material ids rely on."""
    assert compiled.materials == tuple(m.name for m in MATERIALS)
    for table in compiled.mu_over_rho.values():
        assert table.shape[0] == len(MATERIALS)
    assert compiled.coherent_cumulative is not None
    assert compiled.coherent_cumulative.shape[0] == len(MATERIALS)


@pytest.mark.validation
@pytest.mark.parametrize("material", sorted(XCOM_TOTAL_MU_OVER_RHO), ids=("air", "cortical_bone"))
def test_compiled_photons_match_xcom_mixtures(compiled: TabulatedData, material: int) -> None:
    """EPDL mixed by the registry mass fractions reproduces XCOM to under 1 percent.

    Air and cortical bone bracket the registry in effective Z; sub-percent agreement
    on both pins the mixing key (composition) and the per-element conversion at once.
    """
    xs = TabulatedCrossSections(compiled)
    for energy, expected in sorted(XCOM_TOTAL_MU_OVER_RHO[material].items()):
        actual = xs.mu_over_rho_total(energy, material)
        relative_error = abs(actual - expected) / expected
        assert relative_error < PHOTON_TOLERANCE, (
            f"mu/rho({MATERIALS[material].name}, {energy} MeV) = {actual:.5e}, "
            f"XCOM = {expected:.5e}, relative error {relative_error:.2%}"
        )


@pytest.mark.validation
def test_compiled_electron_stopping_matches_berger_seltzer(compiled: TabulatedData) -> None:
    """Loaded per-material stopping equals the machinery it was compiled from."""
    xs = TabulatedCrossSections(compiled)
    for index in (WATER, AIR, LUNG, ADIPOSE, CORTICAL_BONE):
        material = MATERIALS[index]
        coeffs = berger_seltzer.radiative_fit_coefficients(material.radiative_anchors)
        for energy in (0.5, 1.0, 5.0, 15.0):
            assert xs.restricted_stopping_power(energy, index, ECUT) == pytest.approx(
                berger_seltzer.restricted_collision_stopping(energy, material, ECUT), rel=2e-3
            )
            assert xs.radiative_stopping_power(energy, index) == pytest.approx(
                berger_seltzer.radiative_stopping(energy, coeffs), rel=2e-3
            )
            assert xs.moller_cross_section(energy, index, ECUT) == pytest.approx(
                berger_seltzer.restricted_moller_cross_section(energy, material, ECUT), rel=2e-3
            )


@pytest.mark.validation
def test_scattering_power_orders_physically(compiled: TabulatedData) -> None:
    """Per-material scattering powers sit at their Z(Z+1)/A ratios to water.

    The elastic transport moment scales to leading order with ``sum_i w_i Z_i(Z_i+1)/A_i``
    (the Rutherford prefactor; the per-element screening logarithm moves the ratio only
    weakly). From the registry compositions that predicts, relative to water (4.22):
    cortical bone 1.38, adipose 0.84, air 0.99, lung 1.00. Measured EEDL ratios
    (2026-07-13): bone 1.26-1.45, adipose 0.81-0.86, air 0.98-1.00, lung 0.98-0.99 over
    0.5-10 MeV. Bounds bracket both with margin; the *magnitude* convention is pinned by
    the water Highland gate. (Do not widen toward the radiation-length ratio ~2.2 for
    bone: X0 folds in screening much more strongly than the large-angle transport
    moment does, and 1.38 is the correct leading-order expectation here.)
    """
    xs = TabulatedCrossSections(compiled)
    for energy in (0.5, 2.0, 10.0):
        water = xs.scattering_power(energy, WATER)
        assert 1.15 < xs.scattering_power(energy, CORTICAL_BONE) / water < 1.65
        assert 0.70 < xs.scattering_power(energy, ADIPOSE) / water < 0.95
        assert 0.90 < xs.scattering_power(energy, AIR) / water < 1.10
        assert 0.90 < xs.scattering_power(energy, LUNG) / water < 1.10


@pytest.mark.validation
def test_five_material_round_trip(compiled: TabulatedData, tmp_path: Path) -> None:
    """The on-disk format and the flattening path carry five materials unchanged."""
    reloaded = load_tables(save_tables(compiled, tmp_path / "materials.npz"))
    assert reloaded.materials == compiled.materials
    np.testing.assert_allclose(
        reloaded.restricted_stopping, compiled.restricted_stopping, rtol=1e-4
    )
    tables = TabulatedCrossSections(reloaded).build_tables(ecut=ECUT, pcut=0.05, e_max=30.0)
    assert tables.mu_compton.shape[0] == len(MATERIALS)
    assert tables.coherent_cumulative.shape[0] == len(MATERIALS)
