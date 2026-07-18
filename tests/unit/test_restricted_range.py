"""Restricted-collision range and its inverse on the CrossSectionSource interface.

The exact-energy-loss stepping (DPM-style; Sempau et al. 2000) needs, per
substep, the mass path an electron can travel before its energy falls to the
cutoff — the restricted-collision range r(E) = int_ECUT^E dE'/S_col(E', ECUT) —
and the inverse map E(r) that yields the energy after a given mass path:
E_end = r^-1(r(E) - rho*s). These pins hold both to the defining differential
identity dr/dE = 1/S and to round-trip consistency, against the analytic water
source.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.data.analytic import AnalyticCrossSections

ECUT = 0.2
WATER = 0


@pytest.fixture(scope="module")
def xs() -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=((WATER, 1.0),))


def test_range_vanishes_at_the_cutoff(xs: AnalyticCrossSections) -> None:
    assert xs.restricted_range(ECUT, WATER, ECUT) == pytest.approx(0.0, abs=1e-12)


def test_range_is_monotone_in_energy(xs: AnalyticCrossSections) -> None:
    energies = np.geomspace(ECUT * 1.01, 20.0, 64)
    ranges = [xs.restricted_range(float(e), WATER, ECUT) for e in energies]
    assert np.all(np.diff(ranges) > 0.0)


@pytest.mark.parametrize("energy", [0.5, 1.0, 3.0, 6.0, 18.0])
def test_range_derivative_is_inverse_stopping_power(
    xs: AnalyticCrossSections, energy: float
) -> None:
    # dr/dE = 1/S_col,restricted(E): central difference at 0.1% spacing.
    h = 1.0e-3 * energy
    dr = xs.restricted_range(energy + h, WATER, ECUT) - xs.restricted_range(energy - h, WATER, ECUT)
    expected = 1.0 / xs.restricted_stopping_power(energy, WATER, ECUT)
    assert dr / (2.0 * h) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("energy", [0.5, 1.0, 6.0, 18.0])
def test_energy_after_zero_path_is_the_energy(xs: AnalyticCrossSections, energy: float) -> None:
    assert xs.energy_after_mass_path(energy, WATER, ECUT, 0.0) == pytest.approx(energy, rel=1e-9)


@pytest.mark.parametrize("energy", [0.5, 1.0, 6.0, 18.0])
def test_full_range_path_reaches_the_cutoff(xs: AnalyticCrossSections, energy: float) -> None:
    r = xs.restricted_range(energy, WATER, ECUT)
    assert xs.energy_after_mass_path(energy, WATER, ECUT, r) == pytest.approx(ECUT, rel=1e-6)


@pytest.mark.parametrize("energy", [1.0, 6.0, 18.0])
def test_partial_path_round_trip(xs: AnalyticCrossSections, energy: float) -> None:
    # r(E_end) == r(E) - mass_path, i.e. the inverse really inverts the range.
    mass_path = 0.4 * xs.restricted_range(energy, WATER, ECUT)
    e_end = xs.energy_after_mass_path(energy, WATER, ECUT, mass_path)
    assert ECUT < e_end < energy
    assert xs.restricted_range(e_end, WATER, ECUT) == pytest.approx(
        xs.restricted_range(energy, WATER, ECUT) - mass_path, rel=1e-6
    )


def test_overlong_path_clamps_to_the_cutoff(xs: AnalyticCrossSections) -> None:
    r = xs.restricted_range(6.0, WATER, ECUT)
    assert xs.energy_after_mass_path(6.0, WATER, ECUT, 10.0 * r) == ECUT
