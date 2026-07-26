"""Validation tier: EEDL scattering power for water, consistency vs analytic Highland.

Scattering power has no NIST reference table (unlike photon mu/rho vs XCOM or radiative
stopping vs ESTAR). This gate is therefore a *consistency* check: it confirms the
EEDL-derived mass angular scattering power for water agrees with the analytic
Rossi-Greisen/Highland scattering power already in the codebase to within a factor of
two across the transported range. Its real job is to pin the space-angle convention
(the mean-square-angle factor of two) and the order of magnitude -- a factor-2 error in
the moment definition would blow this gate. The two agree to ~0.9-1.7x in practice;
EEDL runs above Highland because the transport-cross-section moment exceeds the Gaussian
core and the analytic Highland omits its logarithmic step-length correction.

An *absolute* external gate (ICRU-35 water mass scattering powers) is intentionally left
as a documented slot: those values require the report and cannot be verified against a
live source here, and the true physical validator for this quantity is the slice-D
2%/2mm depth-dose gate (central-axis PDD is insensitive to scattering power; penumbra is
where it bites). See AGENTS.md section 7 for the same maintainer-supplied-data pattern.

Point ``PYRADMC_EEDL_PATH`` at a local ``EEDL2023.ALL`` to run this; otherwise it skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.data.tabulated.eedl import (
    element_scattering_power,
    material_scattering_power,
)
from pyRadMC.data.tabulated.epdl import mass_fractions_from_formula

# Consistency band (EEDL / Highland): wide enough that it certifies convention and
# magnitude, tight enough that a factor-2 moment error or a wrong mixing fails it.
RATIO_LOW, RATIO_HIGH = 0.7, 2.0
ENERGIES_MEV = [0.5, 1.0, 2.0, 6.0, 10.0, 20.0]


def _eedl_path() -> Path:
    raw = os.environ.get("PYRADMC_EEDL_PATH")
    if not raw or not Path(raw).is_file():
        pytest.skip("set PYRADMC_EEDL_PATH to a local EEDL2023.ALL to run this validation")
    return Path(raw)


@pytest.mark.validation
def test_eedl_water_scattering_power_tracks_highland() -> None:
    """EEDL water scattering power sits within a factor of two of analytic Highland."""
    text = _eedl_path().read_text(encoding="latin-1")
    elements = element_scattering_power(text, elements={1, 8})
    fractions = mass_fractions_from_formula({1: 2, 8: 1})  # H2O
    grid = np.geomspace(0.1, 20.0, 1024)
    scattering = material_scattering_power(elements, fractions, grid)

    analytic = AnalyticCrossSections()
    for energy in ENERGIES_MEV:
        eedl = float(np.interp(np.log(energy), np.log(grid), scattering))
        highland = analytic.scattering_power(energy, WATER)
        ratio = eedl / highland
        assert RATIO_LOW < ratio < RATIO_HIGH, (
            f"T/rho(water, {energy} MeV): EEDL = {eedl:.4g}, Highland = {highland:.4g}, "
            f"ratio {ratio:.2f} outside [{RATIO_LOW}, {RATIO_HIGH}]"
        )


@pytest.mark.validation
def test_eedl_water_scattering_power_is_monotone_decreasing() -> None:
    """Scattering power falls monotonically with energy (~1/(pv)^2); a smoothness check.

    Guards the sparse-shape-grid interpolation: the pre-fix bug interpolated the final
    product across EEDL's 0.256->10 MeV angular gap and produced a non-monotone bump.
    """
    text = _eedl_path().read_text(encoding="latin-1")
    elements = element_scattering_power(text, elements={1, 8})
    fractions = mass_fractions_from_formula({1: 2, 8: 1})
    grid = np.geomspace(0.5, 20.0, 256)
    scattering = material_scattering_power(elements, fractions, grid)
    assert np.all(np.diff(scattering) < 0.0), "scattering power is not monotone decreasing"
