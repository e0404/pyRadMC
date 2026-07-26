"""The ICRP reference media: registry data and electron-stopping validation.

The Berger-Seltzer machinery (``pyRadMC.data.berger_seltzer``) is then
gated against ESTAR per material at the same tolerances the water gate has always
used: collision 3%, radiative 5% (calibration regression pin), CSDA range 3%.

Provenance (verify before trusting further):

- Compositions, densities, I-values, Sternheimer coefficients: PDG 2024
  atomic-properties pages (pdg.lbl.gov/2024/AtomicNuclearProperties), which
  reproduce ICRP compositions and the Sternheimer, Berger & Seltzer (1984)
  density-effect table (doi:10.1016/0092-640X(84)90002-0).
- Stopping powers and ranges: NIST ESTAR (doi:10.18434/T4NC7P), materials 104 (air,
  dry, near sea level), 190 (lung, ICRP), 103 (adipose tissue, ICRP), 120 (cortical
  bone, ICRP).
"""

from __future__ import annotations

import pytest

from pyRadMC.data import berger_seltzer
from pyRadMC.data.materials import (
    ADIPOSE,
    AIR,
    AVOGADRO,
    CORTICAL_BONE,
    LUNG,
    MATERIALS,
    WATER,
)

COLLISION_TOLERANCE = 0.03
RADIATIVE_TOLERANCE = 0.05
RANGE_TOLERANCE = 0.03

# NIST ESTAR collision stopping power (MeV cm^2/g) at kinetic energy (MeV).
ESTAR_COLLISION: dict[int, dict[float, float]] = {
    AIR: {0.2: 2.469, 0.5: 1.802, 1.0: 1.661, 2.0: 1.684, 6.0: 1.870, 10.0: 1.979, 20.0: 2.134},
    LUNG: {0.2: 2.764, 0.5: 2.013, 1.0: 1.827, 2.0: 1.802, 6.0: 1.890, 10.0: 1.947, 20.0: 2.024},
    ADIPOSE: {
        0.2: 2.871,
        0.5: 2.081,
        1.0: 1.880,
        2.0: 1.850,
        6.0: 1.939,
        10.0: 1.997,
        20.0: 2.073,
    },
    CORTICAL_BONE: {
        0.2: 2.507,
        0.5: 1.825,
        1.0: 1.659,
        2.0: 1.643,
        6.0: 1.740,
        10.0: 1.799,
        20.0: 1.874,
    },
}

# NIST ESTAR CSDA range (g/cm^2) at kinetic energy (MeV).
ESTAR_CSDA_RANGE: dict[int, dict[float, float]] = {
    AIR: {0.2: 5.082e-2, 1.0: 4.912e-1, 6.0: 3.255, 20.0: 9.447},
    LUNG: {0.2: 4.534e-2, 1.0: 4.416e-1, 6.0: 3.088, 20.0: 9.425},
    ADIPOSE: {0.2: 4.359e-2, 1.0: 4.275e-1, 6.0: 3.017, 20.0: 9.305},
    CORTICAL_BONE: {0.2: 5.015e-2, 1.0: 4.857e-1, 6.0: 3.335, 20.0: 9.850},
}

# PDG <Z/A> (mol/g); the registry's electrons_per_gram divided by N_A must agree.
PDG_Z_OVER_A: dict[int, float] = {
    AIR: 0.49919,
    LUNG: 0.54965,
    ADIPOSE: 0.55947,
    CORTICAL_BONE: 0.52130,
}

MEDIA = sorted(PDG_Z_OVER_A)
_IDS = [MATERIALS[m].name if m < len(MATERIALS) else str(m) for m in MEDIA]


class TestRegistryEntries:
    def test_indices_and_names(self) -> None:
        """Indices are stable registry positions; water stays at zero."""
        assert (WATER, AIR, LUNG, ADIPOSE, CORTICAL_BONE) == (0, 1, 2, 3, 4)
        assert [MATERIALS[m].name for m in (AIR, LUNG, ADIPOSE, CORTICAL_BONE)] == [
            "air",
            "lung",
            "adipose",
            "cortical_bone",
        ]

    def test_densities_are_the_icrp_references(self) -> None:
        assert MATERIALS[AIR].density == 1.205e-3  # 20 C, 1 atm
        assert MATERIALS[LUNG].density == 1.050  # ICRP lung tissue (deflated)
        assert MATERIALS[ADIPOSE].density == 0.920
        assert MATERIALS[CORTICAL_BONE].density == 1.850

    def test_mean_excitation_energies(self) -> None:
        """I-values in eV: 85.7 / 75.3 / 63.2 / 106.4 (SBS-1984, as in ESTAR)."""
        assert MATERIALS[AIR].mean_excitation_mev == pytest.approx(85.7e-6)
        assert MATERIALS[LUNG].mean_excitation_mev == pytest.approx(75.3e-6)
        assert MATERIALS[ADIPOSE].mean_excitation_mev == pytest.approx(63.2e-6)
        assert MATERIALS[CORTICAL_BONE].mean_excitation_mev == pytest.approx(106.4e-6)

    @pytest.mark.parametrize("material", MEDIA, ids=_IDS)
    def test_electrons_per_gram_matches_pdg(self, material: int) -> None:
        """<Z/A> from our composition agrees with PDG's independent evaluation.

        PDG uses slightly older atomic weights (e.g. C 12.0107 vs 12.011), so the
        comparison is to 1e-4 relative, not exact. Exception, measured 2026-07-13:
        PDG's adipose page quotes <Z/A> = 0.55947, but its *own* mass-fraction table
        (identical to NIST's, which ESTAR computes from) yields 0.55846 — a 1.8e-3
        internal inconsistency on the PDG side. The composition is the ground truth
        here (it is what ESTAR's stopping powers, our validation reference, are
        built from), so adipose gets a tolerance that spans the discrepancy instead
        of a "corrected" transcription.
        """
        z_over_a = MATERIALS[material].electrons_per_gram / AVOGADRO
        tolerance = 2.0e-3 if material == ADIPOSE else 1.0e-4
        assert z_over_a == pytest.approx(PDG_Z_OVER_A[material], rel=tolerance)

    def test_bone_is_the_high_z_medium(self) -> None:
        """Cortical bone carries the P/Ca content that makes photoelectric matter."""
        composition = dict(MATERIALS[CORTICAL_BONE].composition)
        assert composition[20] == pytest.approx(0.209930)  # Ca
        assert composition[15] == pytest.approx(0.104970)  # P


class TestElectronStoppingAgainstEstar:
    """The material-parameterized Berger-Seltzer machinery, gated per medium."""

    @pytest.mark.parametrize("material", MEDIA, ids=_IDS)
    def test_unrestricted_collision_stopping(self, material: int) -> None:
        """Berger-Seltzer at delta_cut = T/2 reproduces the ESTAR collision column."""
        for energy, expected in sorted(ESTAR_COLLISION[material].items()):
            actual = berger_seltzer.restricted_collision_stopping(
                energy, MATERIALS[material], delta_cut=energy / 2.0
            )
            relative_error = abs(actual - expected) / expected
            assert relative_error < COLLISION_TOLERANCE, (
                f"S_col({MATERIALS[material].name}, {energy} MeV) = {actual:.4f}, "
                f"ESTAR = {expected:.4f}, relative error {relative_error:.1%}"
            )

    @pytest.mark.parametrize("material", MEDIA, ids=_IDS)
    def test_radiative_fit_pins_its_anchors(self, material: int) -> None:
        """Calibration regression pin: the fit passes through its own anchors."""
        coeffs = berger_seltzer.radiative_fit_coefficients(MATERIALS[material].radiative_anchors)
        for energy, expected in MATERIALS[material].radiative_anchors:
            actual = berger_seltzer.radiative_stopping(energy, coeffs)
            assert actual == pytest.approx(expected, rel=RADIATIVE_TOLERANCE)

    @pytest.mark.parametrize("material", MEDIA, ids=_IDS)
    def test_csda_range(self, material: int) -> None:
        """The range integral compounds both stopping channels; strongest single check."""
        import math

        import numpy as np

        log_energies, ranges = berger_seltzer.csda_range_table(MATERIALS[material])
        for energy, expected in sorted(ESTAR_CSDA_RANGE[material].items()):
            actual = float(np.interp(math.log(energy), log_energies, ranges))
            relative_error = abs(actual - expected) / expected
            assert relative_error < RANGE_TOLERANCE, (
                f"R_CSDA({MATERIALS[material].name}, {energy} MeV) = {actual:.4f}, "
                f"ESTAR = {expected:.4f}, relative error {relative_error:.1%}"
            )
