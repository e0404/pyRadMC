"""Validation tier: EPDL-derived water reproduces NIST XCOM to better than 1 percent.

This is the accuracy claim the analytic backend's 5-percent test
(``tests/unit/test_cross_sections_water.py``) always deferred to Phase 5. It runs the
full tabulated photon path — parse ``EPDL2023.ALL``, convert units, mix H and O by
mass fraction onto the canonical geometric grid, and query through the shipping
:class:`~pyRadMC.data.tabulated.source.TabulatedCrossSections` loader — and gates the
total mass attenuation coefficient of liquid water against NIST XCOM (coherent
included) at 1, 2, 6, 10 and 15 MeV.

The measured agreement is ~0.3 percent; the gate is set at 1 percent because the NIST
reference values transcribed here carry only three significant figures, whose rounding
(up to ~0.2 percent) dominates the residual. Do not loosen this gate to accommodate a
data-layer regression (AGENTS.md 2.4 / the Phase 4 exit record's 2%/2mm carried item);
tighten the *references* instead if sub-percent resolution is ever needed.

The ~90 MB EPDL file is not committed. Point ``PYRADMC_EPDL_PATH`` at a local
``EPDL2023.ALL`` (www-nds.iaea.org/epics/ENDF2023/) to run this; otherwise it skips.
The download/cache script is tabulated slice C.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyRadMC.data.materials import WATER
from pyRadMC.data.tabulated.epdl import (
    element_photon_channels,
    mass_fractions_from_formula,
    material_mu_over_rho,
)
from pyRadMC.data.tabulated.model import TabulatedData
from pyRadMC.data.tabulated.source import TabulatedCrossSections

# NIST XCOM total mu/rho (cm^2/g), liquid water, coherent scattering included. The same
# three-figure transcription pinned in tests/unit/test_cross_sections_water.py; verify
# against https://physics.nist.gov/PhysRefData/Xcom/ before trusting further.
NIST_WATER_MU_OVER_RHO: dict[float, float] = {
    1.0: 0.0707,
    2.0: 0.0493,
    6.0: 0.0277,
    10.0: 0.0222,
    15.0: 0.0194,
}

TOLERANCE_RELATIVE = 0.01


def _epdl_path() -> Path:
    raw = os.environ.get("PYRADMC_EPDL_PATH")
    if not raw or not Path(raw).is_file():
        pytest.skip("set PYRADMC_EPDL_PATH to a local EPDL2023.ALL to run this validation")
    return Path(raw)


def _water_source(text: str) -> TabulatedCrossSections:
    """Build a water :class:`TabulatedCrossSections` from EPDL photon data.

    Electron tables are zero placeholders: this validation exercises only the photon
    channels, and the EEDL electron half of the tabulated compile is a later slice.
    """
    elements = element_photon_channels(text, elements={1, 8})
    fractions = mass_fractions_from_formula({1: 2, 8: 1})  # H2O
    photon_grid = np.geomspace(0.025, 20.0, 2048)  # [PCUT/2, e_max], as tables.py builds
    mixed = material_mu_over_rho(elements, fractions, photon_grid)

    electron_grid = np.geomspace(0.1, 20.0, 8)
    zeros = np.zeros((1, electron_grid.size))
    data = TabulatedData(
        photon_energies=photon_grid,
        mu_over_rho={process: values[np.newaxis, :] for process, values in mixed.items()},
        electron_energies=electron_grid,
        restricted_stopping=zeros.copy(),
        radiative_stopping=zeros.copy(),
        moller=zeros.copy(),
        csda_range=zeros.copy(),
        scattering_power=zeros.copy(),
        delta_cut=0.2,
        materials=("water",),
        provenance="EPDL2023 photons (electron tables placeholder); NIST-XCOM validation",
    )
    return TabulatedCrossSections(data)


@pytest.mark.validation
def test_epdl_water_matches_nist_within_one_percent() -> None:
    """Tabulated EPDL water mu/rho reproduces NIST XCOM to better than 1 percent."""
    xs = _water_source(_epdl_path().read_text(encoding="latin-1"))
    for energy, expected in NIST_WATER_MU_OVER_RHO.items():
        actual = xs.mu_over_rho_total(energy, WATER)
        relative_error = abs(actual - expected) / expected
        assert relative_error < TOLERANCE_RELATIVE, (
            f"mu/rho(water, {energy} MeV) = {actual:.5f} cm^2/g, NIST = {expected:.5f}, "
            f"relative error {relative_error:.2%}"
        )


@pytest.mark.validation
def test_epdl_water_has_a_nonzero_rayleigh_channel() -> None:
    """The tabulated backend carries coherent scattering the analytic backend omits.

    A data statement, not a flag (AGENTS.md 2.10): Rayleigh enters because the EPDL
    coherent cross section is nonzero, and the pair-refit warning (Phase 4 exit record)
    is moot here because every channel comes from one consistent EPDL decomposition.
    """
    from pyRadMC.data.interface import PhotonProcess

    xs = _water_source(_epdl_path().read_text(encoding="latin-1"))
    assert xs.mu_over_rho(1.0, WATER, PhotonProcess.RAYLEIGH) > 0.0
