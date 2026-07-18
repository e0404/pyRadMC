"""Flattened restricted-range tables mirror the interface queries.

``build_cross_section_tables`` gains two electron-side tables for the
exact-energy-loss substep: ``restricted_range`` on the electron energy grid and
its inverse ``energy_of_restricted_range`` on a shared log-range grid (one grid
for all materials; per-material values clamp into their own [ecut, e_max]).
Parity pins: the flattened lookups reproduce the host queries, and the two
tables invert each other through the kernel-side lookup functions alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.tables import build_cross_section_tables, lookup_loglinear_2d

ECUT = 0.2
PCUT = 0.05
E_MAX = 20.0
WATER = 0


@pytest.fixture(scope="module")
def fixtures() -> tuple:
    xs = AnalyticCrossSections(geometry_densities=((WATER, 1.0),))
    tab = build_cross_section_tables(xs, ecut=ECUT, pcut=PCUT, e_max=E_MAX)
    return xs, tab


@pytest.mark.parametrize("energy", [0.3, 1.0, 6.0, 19.0])
def test_range_table_matches_interface(fixtures: tuple, energy: float) -> None:
    xs, tab = fixtures
    looked_up = lookup_loglinear_2d(
        tab.restricted_range,
        WATER,
        tab.electron_log_e_min,
        tab.electron_inv_dlog,
        tab.n_points,
        energy,
    )
    assert looked_up == pytest.approx(xs.restricted_range(energy, WATER, ECUT), rel=1e-4)


@pytest.mark.parametrize("energy", [0.5, 1.0, 6.0, 19.0])
def test_inverse_table_round_trips_through_lookups(fixtures: tuple, energy: float) -> None:
    # The kernel-side arithmetic: r = range(E); E' = energy_of_range(r - rho*s).
    xs, tab = fixtures
    r = lookup_loglinear_2d(
        tab.restricted_range,
        WATER,
        tab.electron_log_e_min,
        tab.electron_inv_dlog,
        tab.n_points,
        energy,
    )
    mass_path = 0.3 * r
    e_end = lookup_loglinear_2d(
        tab.energy_of_restricted_range,
        WATER,
        tab.range_log_r_min,
        tab.range_inv_dlog,
        tab.n_points,
        r - mass_path,
    )
    expected = xs.energy_after_mass_path(energy, WATER, ECUT, mass_path)
    assert e_end == pytest.approx(expected, rel=2e-4)


def test_inverse_table_is_monotone(fixtures: tuple) -> None:
    _, tab = fixtures
    values = tab.energy_of_restricted_range[WATER]
    assert np.all(np.diff(values) >= 0.0)
