"""Tungsten heavy alloy joins the material registry (Halcyon commissioning workstream).

Real MLC leaves are ~95 percent W heavy alloys with Ni/Cu binders; the pure-tungsten
entry models them as W at a caller-supplied density, which the ``TUNGSTEN`` docstring
records as a ~1-2 percent mu/rho approximation at MV energies. This entry retires that
approximation for the common W95/Ni3.5/Cu1.5 alloy at 18.0 g/cm^3 — the composition
published for Halcyon-class dual-layer MLC surrogates — as the pure data change the
registry anticipated.

Gated against NIST ESTAR like the ICRP media and tungsten: collision 3%, radiative 5%
(calibration regression pin), CSDA range 3%. The density-effect parameters additionally
get their own absolute gate, because unlike every other entry they are *fitted here*
rather than transcribed (see provenance), and the fit quality is a claim that needs its
own pin.

Provenance (verify before trusting further; queried 2026-08-11):

- Atomic weights W/Ni/Cu: IUPAC 2021 (Prohaska et al., doi:10.1515/pac-2019-0603).
- Composition W 0.95 / Ni 0.035 / Cu 0.015 by mass, density 18.0 g/cm^3: the heavy-alloy
  surrogate used in published approximate Halcyon MC work (Laakkonen et al., Phys. Med.
  Biol. 68, 2023, doi:10.1088/1361-6560/acbc61 lineage; nominal, machine-adjustable).
- Stopping powers, ranges, exact density-effect column, and I = 692.5 eV: NIST ESTAR
  (doi:10.18434/T4NC7P), *user-defined material* with exactly this composition and
  density. ESTAR computes I by the ICRU-37 Bragg additivity rule; the value is
  consistent with a hand evaluation from the elemental I values (~691 eV).
- Sternheimer parameters: cbar = 2 ln(I / hbar omega_p) + 1 analytically (the same
  relation reproduces the published SBS-1984 tungsten row's 5.4059 exactly); a, m, x0,
  x1 least-squares fitted to ESTAR's exact density-effect column over its full
  0.01-1000 MeV grid (max |delta error| 0.028); delta0 = 0.14 carried from the SBS-1984
  tungsten row (metallic conduction term; it only acts below x0 where delta is already
  near zero).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyRadMC.data import berger_seltzer
from pyRadMC.data.materials import (
    AVOGADRO,
    MATERIALS,
    TUNGSTEN_ALLOY,
    SternheimerParameters,
)

COLLISION_TOLERANCE = 0.03
RADIATIVE_TOLERANCE = 0.05
RANGE_TOLERANCE = 0.03
DELTA_TOLERANCE = 0.05  # absolute, on the fitted density-effect parameterization

ELECTRON_MASS_MEV = 0.510998950

# NIST ESTAR user-defined W95/Ni3.5/Cu1.5 at 18.0 g/cm^3: collision stopping power
# (MeV cm^2/g) at kinetic energy (MeV).
ESTAR_COLLISION = {
    0.2: 1.465,
    0.5: 1.102,
    1.0: 1.031,
    2.0: 1.051,
    6.0: 1.160,
    10.0: 1.217,
    20.0: 1.291,
}

# Same query: CSDA range (g/cm^2) at kinetic energy (MeV).
ESTAR_CSDA_RANGE = {0.2: 8.675e-2, 1.0: 7.589e-1, 6.0: 4.242, 20.0: 9.609}

# Same query: the exact density-effect parameter delta at kinetic energy (MeV).
ESTAR_DELTA = {0.2: 0.0342, 1.0: 0.3686, 2.0: 0.7174, 10.0: 2.097, 20.0: 2.942}

# <Z/A> from the composition and IUPAC 2021 weights (hand-derivable):
# 0.95*74/183.84 + 0.035*28/58.693 + 0.015*29/63.546.
Z_OVER_A = 0.405940


class TestRegistryEntry:
    def test_index_and_name(self) -> None:
        assert TUNGSTEN_ALLOY == 6
        assert MATERIALS[TUNGSTEN_ALLOY].name == "tungsten_alloy"

    def test_density_is_the_heavy_alloy_nominal(self) -> None:
        assert MATERIALS[TUNGSTEN_ALLOY].density == 18.0

    def test_composition_is_w95_ni35_cu15(self) -> None:
        assert MATERIALS[TUNGSTEN_ALLOY].composition == ((28, 0.035), (29, 0.015), (74, 0.95))

    def test_mean_excitation_is_the_estar_mixture_value(self) -> None:
        """I = 692.5 eV, ESTAR's ICRU-37 Bragg-additivity evaluation."""
        assert MATERIALS[TUNGSTEN_ALLOY].mean_excitation_mev == pytest.approx(692.5e-6)

    def test_electrons_per_gram_matches_composition(self) -> None:
        z_over_a = MATERIALS[TUNGSTEN_ALLOY].electrons_per_gram / AVOGADRO
        assert z_over_a == pytest.approx(Z_OVER_A, rel=1.0e-4)

    def test_sternheimer_cbar_is_the_plasma_energy_relation(self) -> None:
        """cbar = 2 ln(I / hbar omega_p) + 1 with hbar omega_p = 28.8159 sqrt(rho <Z/A>) eV.

        This is the exact Sternheimer relation, not a fit; the same evaluation
        reproduces the published SBS-1984 tungsten row (5.4059) from tungsten's own
        I and density.
        """
        material = MATERIALS[TUNGSTEN_ALLOY]
        plasma_ev = 28.8159 * math.sqrt(material.density * Z_OVER_A)
        cbar = 2.0 * math.log(material.mean_excitation_mev * 1.0e6 / plasma_ev) + 1.0
        assert material.sternheimer.cbar == pytest.approx(cbar, abs=1.0e-3)

    def test_sternheimer_row_is_pinned(self) -> None:
        """The fitted row (see module docstring); a change is a refit, not a tweak."""
        assert MATERIALS[TUNGSTEN_ALLOY].sternheimer == SternheimerParameters(
            a=0.1564, m=2.8413, x0=0.2323, x1=3.4878, cbar=5.3699, delta0=0.14
        )

    def test_density_effect_matches_estar_exact_column(self) -> None:
        """The fitted parameterization reproduces ESTAR's exact delta to 0.05."""
        for energy, expected in sorted(ESTAR_DELTA.items()):
            tau = energy / ELECTRON_MASS_MEV
            actual = berger_seltzer.density_effect(tau, MATERIALS[TUNGSTEN_ALLOY].sternheimer)
            assert actual == pytest.approx(expected, abs=DELTA_TOLERANCE), (
                f"delta({energy} MeV) = {actual:.4f}, ESTAR exact = {expected:.4f}"
            )


class TestElectronStoppingAgainstEstar:
    """The Berger-Seltzer machinery on the mixture, gated like the ICRP media."""

    def test_unrestricted_collision_stopping(self) -> None:
        for energy, expected in sorted(ESTAR_COLLISION.items()):
            actual = berger_seltzer.restricted_collision_stopping(
                energy, MATERIALS[TUNGSTEN_ALLOY], delta_cut=energy / 2.0
            )
            relative_error = abs(actual - expected) / expected
            assert relative_error < COLLISION_TOLERANCE, (
                f"S_col(tungsten_alloy, {energy} MeV) = {actual:.4f}, "
                f"ESTAR = {expected:.4f}, relative error {relative_error:.1%}"
            )

    def test_radiative_fit_pins_its_anchors(self) -> None:
        coeffs = berger_seltzer.radiative_fit_coefficients(
            MATERIALS[TUNGSTEN_ALLOY].radiative_anchors
        )
        for energy, expected in MATERIALS[TUNGSTEN_ALLOY].radiative_anchors:
            actual = berger_seltzer.radiative_stopping(energy, coeffs)
            assert actual == pytest.approx(expected, rel=RADIATIVE_TOLERANCE)

    def test_csda_range(self) -> None:
        log_energies, ranges = berger_seltzer.csda_range_table(MATERIALS[TUNGSTEN_ALLOY])
        for energy, expected in sorted(ESTAR_CSDA_RANGE.items()):
            actual = float(np.interp(math.log(energy), log_energies, ranges))
            relative_error = abs(actual - expected) / expected
            assert relative_error < RANGE_TOLERANCE, (
                f"R_CSDA(tungsten_alloy, {energy} MeV) = {actual:.4f}, "
                f"ESTAR = {expected:.4f}, relative error {relative_error:.1%}"
            )
