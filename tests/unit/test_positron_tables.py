"""Flattened positron tables mirror the interface queries.

The kernel-side data of the ``"bhabha-in-flight"`` positron model
(:func:`pyradmc.data.positron.build_positron_tables`): the Bhabha and annihilation
cross-sections and the positron restricted range on the *electron* energy grid of
:class:`pyradmc.data.tables.CrossSectionTables` (so a kernel interpolates them
with the metadata it already carries), and the inverse range on a log-range grid
of its own. Parity pins, as for the electron range tables
(``test_range_tables.py``): the flattened lookups reproduce the host queries, and
the two range tables invert each other through the kernel-side lookup alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.positron import PositronTables, build_positron_tables
from pyradmc.data.tables import build_cross_section_tables, lookup_loglinear_2d

ECUT = 0.2
PCUT = 0.05
E_MAX = 20.0
WATER = 0


@pytest.fixture(scope="module")
def fixtures() -> tuple[AnalyticCrossSections, PositronTables]:
    xs = AnalyticCrossSections(geometry_densities=((WATER, 1.0),))
    return xs, build_positron_tables(xs, ecut=ECUT, e_max=E_MAX)


def _on_energy_grid(tab: PositronTables, values: np.ndarray, energy: float) -> float:
    return lookup_loglinear_2d(
        values, WATER, tab.electron_log_e_min, tab.electron_inv_dlog, tab.n_points, energy
    )


def test_the_energy_grid_is_the_electron_grid(fixtures: tuple) -> None:
    """One metadata pair serves both structs in the kernel."""
    xs, tab = fixtures
    base = build_cross_section_tables(xs, ecut=ECUT, pcut=PCUT, e_max=E_MAX)
    assert tab.n_points == base.n_points
    assert tab.electron_log_e_min == base.electron_log_e_min
    assert tab.electron_inv_dlog == base.electron_inv_dlog
    assert tab.bhabha.shape == base.moller.shape


@pytest.mark.parametrize("energy", [0.3, 0.41, 1.0, 6.0, 19.0])
def test_cross_sections_match_the_interface(fixtures: tuple, energy: float) -> None:
    xs, tab = fixtures
    assert _on_energy_grid(tab, tab.bhabha, energy) == pytest.approx(
        xs.bhabha_cross_section(energy, WATER, ECUT), rel=2e-3
    )
    assert _on_energy_grid(tab, tab.annihilation, energy) == pytest.approx(
        xs.annihilation_cross_section(energy, WATER), rel=1e-4
    )


def test_bhabha_opens_at_the_cut_not_at_twice_the_cut(fixtures: tuple) -> None:
    """The positron may hand over everything: nonzero between ECUT and 2 ECUT."""
    _, tab = fixtures
    assert _on_energy_grid(tab, tab.bhabha, 0.5 * ECUT * 1.5) == 0.0
    assert _on_energy_grid(tab, tab.bhabha, 1.5 * ECUT) > 0.0


@pytest.mark.parametrize("energy", [0.3, 1.0, 6.0, 19.0])
def test_range_table_matches_interface(fixtures: tuple, energy: float) -> None:
    xs, tab = fixtures
    assert _on_energy_grid(tab, tab.restricted_range, energy) == pytest.approx(
        xs.positron_restricted_range(energy, WATER, ECUT), rel=1e-4
    )


@pytest.mark.parametrize("energy", [0.5, 1.0, 6.0, 19.0])
def test_inverse_table_round_trips_through_lookups(fixtures: tuple, energy: float) -> None:
    # The kernel-side arithmetic: r = range(E); E' = energy_of_range(r - rho*s).
    xs, tab = fixtures
    r = _on_energy_grid(tab, tab.restricted_range, energy)
    mass_path = 0.3 * r
    e_end = lookup_loglinear_2d(
        tab.energy_of_restricted_range,
        WATER,
        tab.range_log_r_min,
        tab.range_inv_dlog,
        tab.n_points,
        r - mass_path,
    )
    expected = xs.positron_energy_after_mass_path(energy, WATER, ECUT, mass_path)
    assert e_end == pytest.approx(expected, rel=2e-4)


def test_inverse_table_is_monotone(fixtures: tuple) -> None:
    _, tab = fixtures
    assert np.all(np.diff(tab.energy_of_restricted_range[WATER]) >= 0.0)


def test_the_positron_range_is_not_the_electron_range(fixtures: tuple) -> None:
    """The tables carry the positron's own map (1.3 % longer near 1 MeV in water)."""
    xs, tab = fixtures
    positron = _on_energy_grid(tab, tab.restricted_range, 1.0)
    assert positron / xs.restricted_range(1.0, WATER, ECUT) == pytest.approx(1.013, abs=3e-3)


def test_a_grid_above_the_source_domain_is_refused() -> None:
    xs = AnalyticCrossSections(geometry_densities=((WATER, 1.0),))
    with pytest.raises(ValueError, match="energy domain"):
        build_positron_tables(xs, ecut=ECUT, e_max=40.0)
