"""Validation tier: EEDL scattering power for water, consistency vs the analytic anchor.

Scattering power has no NIST reference table (unlike photon mu/rho vs XCOM or radiative
stopping vs ESTAR). This gate is therefore a *consistency* check: it confirms the
EEDL-derived nuclear-elastic mass angular scattering power plus the restricted
soft-electron moment agrees with the analytic Class-II transport moment to within a
factor of two
across the transported range. Its real job is to pin the space-angle convention (the
mean-square-angle factor of two) and the order of magnitude -- a factor-2 error in the
moment definition would blow this gate. EEDL omits the unresolved forward tail
(~5 % at 10 MeV, ~11 % at 20 MeV, see ``element_scattering_power``), while its
2-4 MeV ``<1-mu>`` is interpolated across a 0.256->10 MeV angular-shape gap.
Both sides add the same below-ECUT electron term and exclude the hard Moller moment
whose deflection is transported separately. Measured 2026-09-04, tabulated / analytic:
0.95 at 0.2 MeV, 1.00 at 0.5, 1.10 at 1, 1.27 at 2, 1.12 at 6, 0.91 at 10, 0.87 at
20 MeV -- the 1-6 MeV excess is the gap interpolation, the 10-20 MeV shortfall the tail.

An *absolute* external gate (ICRU-35 water mass scattering powers, after accounting
for their electron-scattering convention and large-angle cutoff) is intentionally left as
a documented slot: those values require the report and cannot be verified against a
live source here. The existing EGSnrc PDD gamma gate and the EGSnrc/TOPAS pencil-kernel
comparisons exercise the transport effect; central-axis PDD beyond dmax is largely
insensitive to it.

Point ``PYRADMC_EEDL_PATH`` at a local ``EEDL2023.ALL`` to run this; otherwise it skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.goudsmit_saunderson import subthreshold_moller_scattering_power
from pyradmc.data.materials import MATERIALS, WATER
from pyradmc.data.tabulated.eedl import (
    element_scattering_power,
    material_scattering_power,
)
from pyradmc.data.tabulated.epdl import mass_fractions_from_formula

# Consistency band (EEDL / analytic): wide enough that it certifies convention and
# magnitude, tight enough that a factor-2 moment error or a wrong mixing fails it.
RATIO_LOW, RATIO_HIGH = 0.7, 2.0
ENERGIES_MEV = [0.5, 1.0, 2.0, 6.0, 10.0, 20.0]
ECUT = 0.2


def _eedl_path() -> Path:
    raw = os.environ.get("PYRADMC_EEDL_PATH")
    if not raw or not Path(raw).is_file():
        pytest.skip("set PYRADMC_EEDL_PATH to a local EEDL2023.ALL to run this validation")
    return Path(raw)


@pytest.mark.validation
def test_eedl_water_scattering_power_tracks_the_transport_moment() -> None:
    """EEDL water scattering power sits within a factor of two of the analytic anchor."""
    text = _eedl_path().read_text(encoding="latin-1")
    elements = element_scattering_power(text, elements={1, 8})
    fractions = mass_fractions_from_formula({1: 2, 8: 1})  # H2O
    grid = np.geomspace(0.1, 20.0, 1024)
    scattering = material_scattering_power(elements, fractions, grid)

    analytic = AnalyticCrossSections()
    for energy in ENERGIES_MEV:
        eedl = float(np.interp(np.log(energy), np.log(grid), scattering))
        eedl += subthreshold_moller_scattering_power(MATERIALS[WATER].composition, energy, ECUT)
        anchor = analytic.scattering_power(energy, WATER, ECUT)
        ratio = eedl / anchor
        assert RATIO_LOW < ratio < RATIO_HIGH, (
            f"T/rho(water, {energy} MeV): EEDL = {eedl:.4g}, analytic = {anchor:.4g}, "
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
    scattering += np.array(
        [
            subthreshold_moller_scattering_power(MATERIALS[WATER].composition, float(e), ECUT)
            for e in grid
        ]
    )
    assert np.all(np.diff(scattering) < 0.0), "scattering power is not monotone decreasing"
