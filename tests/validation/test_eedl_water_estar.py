"""Validation tier: EEDL radiative stopping for water vs NIST ESTAR.

Builds water's mass radiative (bremsstrahlung) stopping power from the real EEDL
library and gates it against NIST ESTAR at 1, 10 and 20 MeV. Unlike the photon totals
(sub-percent vs XCOM), EEDL's bremsstrahlung evaluation departs from ESTAR's
Seltzer-Berger data by a few percent in the MeV range (measured max ~4.2% at 10 MeV):
that is a genuine library difference, not a conversion error, and is exactly why the
default ``berger-seltzer`` electron-stopping strategy sources radiative stopping from
the analytic ESTAR fit instead. This gate certifies the ``eedl`` strategy's radiative
path at the 5% these two evaluations agree to; do not tighten it to sub-percent (that
would demand ESTAR data this path deliberately does not use) nor loosen it to hide a
regression.

The ~28 MB EEDL file is not committed. Point ``PYRADMC_EEDL_PATH`` at a local
``EEDL2023.ALL`` (www-nds.iaea.org/epics/ENDF2023/) to run this; otherwise it skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyRadMC.data.tabulated.eedl import (
    element_radiative_stopping,
    material_radiative_stopping,
)
from pyRadMC.data.tabulated.epdl import mass_fractions_from_formula

# NIST ESTAR radiative mass stopping power for liquid water (MeV cm^2/g); the same
# transcription pinned as the analytic backend's ESTAR anchors (pyRadMC/data/analytic.py).
ESTAR_WATER_RADIATIVE: dict[float, float] = {
    1.0: 0.0128,
    10.0: 0.1813,
    20.0: 0.4008,
}

TOLERANCE_RELATIVE = 0.05


def _eedl_path() -> Path:
    raw = os.environ.get("PYRADMC_EEDL_PATH")
    if not raw or not Path(raw).is_file():
        pytest.skip("set PYRADMC_EEDL_PATH to a local EEDL2023.ALL to run this validation")
    return Path(raw)


@pytest.mark.validation
def test_eedl_water_radiative_matches_estar_within_five_percent() -> None:
    """EEDL-derived water radiative stopping agrees with ESTAR to 5 percent."""
    text = _eedl_path().read_text(encoding="latin-1")
    elements = element_radiative_stopping(text, elements={1, 8})
    fractions = mass_fractions_from_formula({1: 2, 8: 1})  # H2O
    grid = np.geomspace(0.1, 20.0, 1024)
    s_rad = material_radiative_stopping(elements, fractions, grid)

    for energy, expected in ESTAR_WATER_RADIATIVE.items():
        actual = float(np.interp(np.log(energy), np.log(grid), s_rad))
        relative_error = abs(actual - expected) / expected
        assert relative_error < TOLERANCE_RELATIVE, (
            f"S_rad(water, {energy} MeV) = {actual:.5f} MeV cm^2/g, ESTAR = {expected:.5f}, "
            f"relative error {relative_error:.2%}"
        )
