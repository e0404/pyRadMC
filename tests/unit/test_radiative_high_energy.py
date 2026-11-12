"""Radiative stopping power above 20 MeV: gated against ESTAR up to 1 GeV.

The registry's three-anchor log-quadratic (1, 10, 20 MeV) is exact at its anchors
but extrapolates badly: against ESTAR it is +8 % (water) to +32 % (tungsten) high
at 200 MeV, and up to +110 % by 1 GeV. Above the top low anchor the model is now
the ratio ``g(E) = S_rad(E) / (E + m c^2)`` interpolated linearly in ``ln E``
between ESTAR anchors at 30, 50, 100, 200, 500 and 1000 MeV, and held at its
1 GeV value above that (the complete-screening limit, ``S_rad -> E_tot / X_0``).
At or below 20 MeV nothing changes, bit for bit.

Reference data: NIST ESTAR (Berger, Coursey, Zucker & Chang, doi:10.18434/T4NC7P),
radiative and CSDA columns of the default energy grid, retrieved 2026-11-12 for
materials 276 (water, liquid), 104 (air, dry), 190 (lung, ICRP), 103 (adipose
tissue, ICRP), 120 (bone, cortical, ICRP) and 074 (tungsten). The W95/Ni3.5/Cu1.5
alloy is the mass-fraction sum of ESTAR elements 074, 028 and 029: radiative
stopping is additive by mass fraction, and the same rule reproduces the alloy's
existing 1/10/20 MeV anchors exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data import berger_seltzer
from pyradmc.data.materials import (
    ADIPOSE,
    AIR,
    CORTICAL_BONE,
    LUNG,
    MATERIALS,
    TUNGSTEN,
    TUNGSTEN_ALLOY,
    WATER,
    MaterialData,
)

# ESTAR radiative mass stopping power (MeV cm^2/g) at kinetic energy (MeV), 25 MeV - 1 GeV.
ESTAR_RADIATIVE: dict[int, dict[float, float]] = {
    WATER: {
        25: 0.5277,
        30: 0.6489,
        35: 0.7716,
        40: 0.8955,
        45: 1.021,
        50: 1.146,
        55: 1.273,
        60: 1.4,
        70: 1.656,
        80: 1.914,
        90: 2.173,
        100: 2.434,
        125: 3.089,
        150: 3.749,
        175: 4.412,
        200: 5.078,
        250: 6.416,
        300: 7.76,
        350: 9.107,
        400: 10.46,
        450: 11.81,
        500: 13.17,
        550: 14.53,
        600: 15.89,
        700: 18.61,
        800: 21.33,
        900: 24.06,
        1000: 26.79,
    },
    AIR: {
        25: 0.5219,
        30: 0.6417,
        35: 0.763,
        40: 0.8855,
        45: 1.009,
        50: 1.133,
        55: 1.258,
        60: 1.384,
        70: 1.637,
        80: 1.892,
        90: 2.148,
        100: 2.405,
        125: 3.053,
        150: 3.705,
        175: 4.36,
        200: 5.018,
        250: 6.34,
        300: 7.667,
        350: 8.998,
        400: 10.33,
        450: 11.67,
        500: 13.01,
        550: 14.35,
        600: 15.69,
        700: 18.38,
        800: 21.07,
        900: 23.76,
        1000: 26.46,
    },
    LUNG: {
        25: 0.5215,
        30: 0.6413,
        35: 0.7626,
        40: 0.8851,
        45: 1.009,
        50: 1.133,
        55: 1.258,
        60: 1.384,
        70: 1.637,
        80: 1.892,
        90: 2.148,
        100: 2.406,
        125: 3.054,
        150: 3.706,
        175: 4.362,
        200: 5.02,
        250: 6.343,
        300: 7.672,
        350: 9.004,
        400: 10.34,
        450: 11.68,
        500: 13.02,
        550: 14.36,
        600: 15.71,
        700: 18.4,
        800: 21.09,
        900: 23.79,
        1000: 26.49,
    },
    ADIPOSE: {
        25: 0.4505,
        30: 0.5544,
        35: 0.6597,
        40: 0.7661,
        45: 0.8734,
        50: 0.9815,
        55: 1.09,
        60: 1.2,
        70: 1.42,
        80: 1.642,
        90: 1.865,
        100: 2.089,
        125: 2.653,
        150: 3.222,
        175: 3.794,
        200: 4.368,
        250: 5.522,
        300: 6.682,
        350: 7.845,
        400: 9.012,
        450: 10.18,
        500: 11.35,
        550: 12.53,
        600: 13.7,
        700: 16.05,
        800: 18.41,
        900: 20.77,
        1000: 23.13,
    },
    CORTICAL_BONE: {
        25: 0.7117,
        30: 0.8735,
        35: 1.037,
        40: 1.202,
        45: 1.369,
        50: 1.537,
        55: 1.705,
        60: 1.875,
        70: 2.215,
        80: 2.558,
        90: 2.903,
        100: 3.249,
        125: 4.12,
        150: 4.997,
        175: 5.878,
        200: 6.762,
        250: 8.537,
        300: 10.32,
        350: 12.11,
        400: 13.9,
        450: 15.69,
        500: 17.49,
        550: 19.29,
        600: 21.09,
        700: 24.7,
        800: 28.31,
        900: 31.92,
        1000: 35.53,
    },
    TUNGSTEN: {
        25: 3.065,
        30: 3.735,
        35: 4.412,
        40: 5.096,
        45: 5.784,
        50: 6.477,
        55: 7.174,
        60: 7.873,
        70: 9.28,
        80: 10.7,
        90: 12.12,
        100: 13.55,
        125: 17.13,
        150: 20.74,
        175: 24.36,
        200: 27.98,
        250: 35.25,
        300: 42.54,
        350: 49.84,
        400: 57.15,
        450: 64.46,
        500: 71.78,
        550: 79.11,
        600: 86.45,
        700: 101.1,
        800: 115.8,
        900: 130.5,
        1000: 145.2,
    },
    # 0.95 W + 0.035 Ni + 0.015 Cu by mass (see module docstring), unrounded.
    TUNGSTEN_ALLOY: {
        25: 2.99167,
        30: 3.645995,
        35: 4.307105,
        40: 4.975065,
        45: 5.64694,
        50: 6.32365,
        55: 7.00426,
        60: 7.686835,
        70: 9.060755,
        80: 10.44724,
        90: 11.833895,
        100: 13.23023,
        125: 16.726265,
        150: 20.25145,
        175: 23.7865,
        200: 27.32155,
        250: 34.42115,
        300: 41.54025,
        350: 48.6692,
        400: 55.8083,
        450: 62.94725,
        500: 70.09585,
        550: 77.2548,
        600: 84.4229,
        700: 98.7306,
        800: 113.0865,
        900: 127.44255,
        1000: 141.79875,
    },
}

# ESTAR CSDA range (g/cm^2).
ESTAR_CSDA_RANGE: dict[int, dict[float, float]] = {
    WATER: {30.0: 13.17, 50.0: 19.83, 100.0: 32.58, 200.0: 49.57},
    AIR: {30.0: 13.15, 50.0: 19.48, 100.0: 31.67, 200.0: 48.14},
    LUNG: {30.0: 13.32, 50.0: 20.06, 100.0: 32.95, 200.0: 50.14},
    ADIPOSE: {30.0: 13.24, 50.0: 20.13, 100.0: 33.66, 200.0: 52.19},
    CORTICAL_BONE: {30.0: 13.69, 50.0: 20.08, 100.0: 31.64, 200.0: 46.13},
    TUNGSTEN: {30.0: 11.9, 50.0: 15.06, 100.0: 19.6, 200.0: 24.28},
}

RADIATIVE_HIGH_ENERGY_TOLERANCE = 0.01
"""Measured worst case 0.25 % over 25 MeV - 1 GeV for every medium except water,
where 0.9 % at 25 MeV is inherited from the registry's 20 MeV water anchor
(0.4008 against ESTAR's 0.4086), the junction the interpolation starts from."""

RANGE_TOLERANCE = 0.03  # the existing per-medium CSDA gate (tests/unit/test_icrp_media.py)

ALL_MEDIA = [WATER, AIR, LUNG, ADIPOSE, CORTICAL_BONE, TUNGSTEN, TUNGSTEN_ALLOY]


@pytest.mark.parametrize("material", ALL_MEDIA, ids=lambda m: MATERIALS[m].name)
def test_radiative_matches_estar_up_to_a_gev(material: int) -> None:
    entry = MATERIALS[material]
    for energy, expected in sorted(ESTAR_RADIATIVE[material].items()):
        actual = berger_seltzer.radiative_stopping_power(float(energy), entry)
        assert actual == pytest.approx(expected, rel=RADIATIVE_HIGH_ENERGY_TOLERANCE), (
            f"{entry.name} at {energy} MeV: {actual:.4f} vs ESTAR {expected:.4f}"
        )


@pytest.mark.parametrize("material", ALL_MEDIA, ids=lambda m: MATERIALS[m].name)
def test_nothing_changes_at_or_below_the_top_low_anchor(material: int) -> None:
    """The 1-20 MeV model is the old log-quadratic, bit for bit."""
    entry = MATERIALS[material]
    coefficients = berger_seltzer.radiative_fit_coefficients(entry.radiative_anchors)
    for energy in np.geomspace(0.01, entry.radiative_anchors[-1][0], 53):
        assert berger_seltzer.radiative_stopping_power(
            float(energy), entry
        ) == berger_seltzer.radiative_stopping(float(energy), coefficients)


@pytest.mark.parametrize("material", ALL_MEDIA, ids=lambda m: MATERIALS[m].name)
def test_continuous_at_the_junction(material: int) -> None:
    entry = MATERIALS[material]
    top = entry.radiative_anchors[-1][0]
    below = berger_seltzer.radiative_stopping_power(top, entry)
    above = berger_seltzer.radiative_stopping_power(top * (1.0 + 1.0e-9), entry)
    assert above == pytest.approx(below, rel=1e-6)


def test_complete_screening_asymptote_above_the_last_anchor() -> None:
    """Above 1 GeV the radiative stopping is proportional to the total energy."""
    entry = MATERIALS[WATER]
    ratios = [
        berger_seltzer.radiative_stopping_power(e, entry) / (e + ELECTRON_MASS_MEV)
        for e in (1000.0, 2000.0, 5000.0)
    ]
    assert ratios[1] == ratios[0] and ratios[2] == ratios[0]


def test_a_material_without_high_anchors_refuses_above_its_fit() -> None:
    """Fail closed: no data above 20 MeV means no answer above 20 MeV."""
    bare = MaterialData(
        **{**MATERIALS[WATER].__dict__, "name": "bare", "radiative_high_anchors": ()}
    )
    assert berger_seltzer.radiative_stopping_power(20.0, bare) > 0.0
    with pytest.raises(ValueError, match="radiative"):
        berger_seltzer.radiative_stopping_power(25.0, bare)


def test_high_anchors_are_ordered_and_start_above_the_fit() -> None:
    for entry in MATERIALS:
        energies = [e for e, _ in entry.radiative_high_anchors]
        assert energies == sorted(energies)
        assert energies[0] > entry.radiative_anchors[-1][0]
        assert all(s > 0.0 for _, s in entry.radiative_high_anchors)


@pytest.mark.parametrize("material", list(ESTAR_CSDA_RANGE), ids=lambda m: MATERIALS[m].name)
def test_csda_range_table_reaches_vhee_energies(material: int) -> None:
    entry = MATERIALS[material]
    log_e, ranges = berger_seltzer.csda_range_table(entry, e_top=200.0)
    assert math.exp(log_e[-1]) >= 200.0
    for energy, expected in ESTAR_CSDA_RANGE[material].items():
        actual = float(np.interp(math.log(energy), log_e, ranges))
        assert actual == pytest.approx(expected, rel=RANGE_TOLERANCE), (
            f"{entry.name} CSDA at {energy} MeV: {actual:.3f} vs ESTAR {expected:.3f}"
        )


def test_csda_range_table_keeps_its_legacy_nodes() -> None:
    """Extending the range table appends nodes; the 1 keV - 30 MeV part is unchanged."""
    entry = MATERIALS[WATER]
    legacy_e, legacy_r = berger_seltzer.csda_range_table(entry)
    wide_e, wide_r = berger_seltzer.csda_range_table(entry, e_top=200.0)
    np.testing.assert_array_equal(wide_e[: legacy_e.size], legacy_e)
    np.testing.assert_array_equal(wide_r[: legacy_r.size], legacy_r)
    np.testing.assert_array_equal(legacy_e, np.log(np.geomspace(1.0e-3, 30.0, 600)))
