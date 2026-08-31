"""Tungsten joins the material registry (beam-limiting-device workstream, slice 0).

Jaws and MLC leaves are tungsten (or tungsten alloy); the collimation module
attenuates through ``mu_over_rho_total``, so the registry needs a tungsten row the
tabulated compiler can mix from EPDL/EEDL. Pure tungsten at a caller-supplied
density is the v1 model; the Ni/Cu atomic weights land alongside so a W-alloy
composition later is a pure data change.

Berger-Seltzer electron stopping is gated against ESTAR at the same tolerances as
the ICRP media (``test_icrp_media.py``). Z = 74 is far outside the low-Z envelope
the transport is tuned for — nothing *transports* electrons in tungsten in v1 —
but the compiled table carries the stopping columns, so they are pinned here. If a
gate misses, report the measured deviation for maintainer sign-off; do not loosen.

Provenance (verify before trusting further; transcribed 2026-07-15):

- Atomic weights W/Ni/Cu: IUPAC 2021 (Prohaska et al., doi:10.1515/pac-2019-0603).
- Density 19.30 g/cm^3, I = 727.0 eV, Sternheimer coefficients: PDG 2024
  atomic-properties page for tungsten (pdg.lbl.gov/2024/AtomicNuclearProperties,
  muE table header), reproducing Sternheimer, Berger & Seltzer (1984),
  doi:10.1016/0092-640X(84)90002-0.
- Stopping powers, ranges, Z/A = 0.402502: NIST ESTAR (doi:10.18434/T4NC7P),
  element 074.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc.data import berger_seltzer
from pyradmc.data.materials import (
    AVOGADRO,
    MATERIALS,
    STANDARD_ATOMIC_WEIGHT,
    TUNGSTEN,
    SternheimerParameters,
)

COLLISION_TOLERANCE = 0.03
RADIATIVE_TOLERANCE = 0.05
RANGE_TOLERANCE = 0.03

# NIST ESTAR tungsten, collision stopping power (MeV cm^2/g) at kinetic energy (MeV).
ESTAR_COLLISION = {
    0.2: 1.439,
    0.5: 1.085,
    1.0: 1.016,
    2.0: 1.037,
    6.0: 1.146,
    10.0: 1.203,
    20.0: 1.277,
}

# NIST ESTAR tungsten, CSDA range (g/cm^2) at kinetic energy (MeV).
ESTAR_CSDA_RANGE = {0.2: 8.835e-2, 1.0: 7.686e-1, 6.0: 4.265, 20.0: 9.594}

ESTAR_Z_OVER_A = 0.402502


class TestAtomicWeights:
    def test_tungsten_alloy_constituents_are_covered(self) -> None:
        """W plus the Ni/Cu of the common heavy alloys (IUPAC 2021)."""
        assert STANDARD_ATOMIC_WEIGHT[74] == 183.84
        assert STANDARD_ATOMIC_WEIGHT[28] == 58.693
        assert STANDARD_ATOMIC_WEIGHT[29] == 63.546


class TestRegistryEntry:
    def test_index_and_name(self) -> None:
        assert TUNGSTEN == 5
        assert MATERIALS[TUNGSTEN].name == "tungsten"

    def test_density_is_the_pdg_reference(self) -> None:
        assert MATERIALS[TUNGSTEN].density == 19.30

    def test_composition_is_pure_element(self) -> None:
        assert MATERIALS[TUNGSTEN].composition == ((74, 1.0),)

    def test_mean_excitation_is_icru37(self) -> None:
        """I(W) = 727.0 eV (SBS-1984, as in ESTAR)."""
        assert MATERIALS[TUNGSTEN].mean_excitation_mev == pytest.approx(727.0e-6)

    def test_sternheimer_is_sbs_1984(self) -> None:
        """The exact SBS-1984 tungsten row (PDG muE header transcription)."""
        assert MATERIALS[TUNGSTEN].sternheimer == SternheimerParameters(
            a=0.1551, m=2.8447, x0=0.2167, x1=3.4960, cbar=5.4059, delta0=0.14
        )

    def test_electrons_per_gram_matches_estar(self) -> None:
        z_over_a = MATERIALS[TUNGSTEN].electrons_per_gram / AVOGADRO
        assert z_over_a == pytest.approx(ESTAR_Z_OVER_A, rel=1.0e-4)


class TestElectronStoppingAgainstEstar:
    """The Berger-Seltzer machinery at Z = 74, gated like the ICRP media."""

    def test_unrestricted_collision_stopping(self) -> None:
        for energy, expected in sorted(ESTAR_COLLISION.items()):
            actual = berger_seltzer.restricted_collision_stopping(
                energy, MATERIALS[TUNGSTEN], delta_cut=energy / 2.0
            )
            relative_error = abs(actual - expected) / expected
            assert relative_error < COLLISION_TOLERANCE, (
                f"S_col(tungsten, {energy} MeV) = {actual:.4f}, "
                f"ESTAR = {expected:.4f}, relative error {relative_error:.1%}"
            )

    def test_radiative_fit_pins_its_anchors(self) -> None:
        coeffs = berger_seltzer.radiative_fit_coefficients(MATERIALS[TUNGSTEN].radiative_anchors)
        for energy, expected in MATERIALS[TUNGSTEN].radiative_anchors:
            actual = berger_seltzer.radiative_stopping(energy, coeffs)
            assert actual == pytest.approx(expected, rel=RADIATIVE_TOLERANCE)

    def test_csda_range(self) -> None:
        log_energies, ranges = berger_seltzer.csda_range_table(MATERIALS[TUNGSTEN])
        for energy, expected in sorted(ESTAR_CSDA_RANGE.items()):
            actual = float(np.interp(math.log(energy), log_energies, ranges))
            relative_error = abs(actual - expected) / expected
            assert relative_error < RANGE_TOLERANCE, (
                f"R_CSDA(tungsten, {energy} MeV) = {actual:.4f}, "
                f"ESTAR = {expected:.4f}, relative error {relative_error:.1%}"
            )
