"""The material registry carries elemental composition and stopping-power data.

This file pins the model
and the water entry; the water numbers must equal the constants the analytic backend
has always used (AGENTS.md 2.2: the oracle does not move during a refactor).

Provenance of the pinned values:

- Atomic weights: IUPAC 2021 standard atomic weights (Prohaska et al.,
  doi:10.1515/pac-2019-0603).
- Water Sternheimer coefficients: Sternheimer, Berger & Seltzer, At. Data Nucl. Data
  Tables 30, 261 (1984), doi:10.1016/0092-640X(84)90002-0 (cross-checked against the
  PDG 2024 atomic-properties compilation, pdg.lbl.gov/2024/AtomicNuclearProperties).
- Water radiative anchors: NIST ESTAR liquid water, the transcription already pinned
  in the analytic backend.
"""

from __future__ import annotations

import pytest

from pyRadMC.data.materials import (
    MATERIALS,
    STANDARD_ATOMIC_WEIGHT,
    WATER,
    MaterialData,
    SternheimerParameters,
    electrons_per_gram_from_composition,
)


class TestAtomicWeights:
    """materials.py is the canonical home of the atomic-weight table."""

    def test_icru_media_elements_are_covered(self) -> None:
        """Every element of the ICRP reference media has a weight, trace metals included."""
        # H C N O Na Mg P S Cl Ar K Ca plus the Fe/Zn traces of the ICRP tissues.
        for z in (1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20, 26, 30):
            assert z in STANDARD_ATOMIC_WEIGHT

    def test_selected_iupac_2021_values(self) -> None:
        """Spot-pin IUPAC 2021 values, including the newly added trace metals."""
        assert STANDARD_ATOMIC_WEIGHT[1] == 1.008
        assert STANDARD_ATOMIC_WEIGHT[8] == 15.999
        assert STANDARD_ATOMIC_WEIGHT[26] == 55.845
        assert STANDARD_ATOMIC_WEIGHT[30] == 65.38

    def test_epdl_reexport_is_the_same_table(self) -> None:
        """The EPDL parser's table is this table, not a drifting copy."""
        from pyRadMC.data.tabulated import epdl

        assert epdl.STANDARD_ATOMIC_WEIGHT is STANDARD_ATOMIC_WEIGHT


class TestRegistryInvariants:
    """Structural invariants that must hold for every registry entry, present and future."""

    @pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m.name)
    def test_composition_is_normalized_mass_fractions(self, material: MaterialData) -> None:
        """Mass fractions are in (0, 1] and sum to one within transcription rounding."""
        assert material.composition, f"{material.name} has no composition"
        for z, fraction in material.composition:
            assert z in STANDARD_ATOMIC_WEIGHT, f"{material.name}: no weight for Z={z}"
            assert 0.0 < fraction <= 1.0
        total = sum(fraction for _, fraction in material.composition)
        # Published mass-fraction tables round to six decimals; do not renormalize
        # silently, but reject anything worse than transcription rounding.
        assert total == pytest.approx(1.0, abs=5.0e-5)

    @pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m.name)
    def test_stopping_data_is_physical(self, material: MaterialData) -> None:
        """I-value, density, and the radiative anchors are positive and ordered."""
        assert material.density > 0.0
        assert 0.0 < material.mean_excitation_mev < 1.0e-3  # eV-scale, stored in MeV
        anchor_energies = [e for e, _ in material.radiative_anchors]
        assert anchor_energies == sorted(anchor_energies)
        assert all(s > 0.0 for _, s in material.radiative_anchors)
        assert len(material.radiative_anchors) >= 3  # log-quadratic needs three points

    def test_names_are_unique(self) -> None:
        names = [m.name for m in MATERIALS]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m.name)
    def test_electrons_per_gram_consistent_with_composition(self, material: MaterialData) -> None:
        """The stored electron density matches N_A * sum(w_i Z_i / M_i).

        Water's stored value uses the molecular weight 18.01528 g/mol while the
        composition mixes IUPAC elemental weights (2*1.008 + 15.999 = 18.015), so
        exact equality is wrong to demand; 5e-5 relative covers that gap without
        letting a transcription error through.
        """
        derived = electrons_per_gram_from_composition(material.composition)
        assert derived == pytest.approx(material.electrons_per_gram, rel=5.0e-5)


class TestWaterEntry:
    """The water entry equals the constants the analytic backend has always used."""

    def test_water_is_index_zero(self) -> None:
        assert MATERIALS[WATER].name == "water"

    def test_composition_matches_h2o_formula(self) -> None:
        """Water's stored composition is exactly the H2O formula mass fractions."""
        from pyRadMC.data.tabulated.epdl import mass_fractions_from_formula

        expected = mass_fractions_from_formula({1: 2, 8: 1})
        stored = dict(MATERIALS[WATER].composition)
        assert stored.keys() == expected.keys()
        for z, fraction in expected.items():
            assert stored[z] == pytest.approx(fraction, rel=1.0e-12)

    def test_mean_excitation_is_icru37(self) -> None:
        """I(water) = 75 eV (ICRU Report 37) — not the ICRU-90 revision PDG adopted."""
        assert MATERIALS[WATER].mean_excitation_mev == 75.0e-6

    def test_sternheimer_is_sbs_1984(self) -> None:
        """The exact SBS-1984 liquid-water row, delta0 included."""
        assert MATERIALS[WATER].sternheimer == SternheimerParameters(
            a=0.09116, m=3.4773, x0=0.2400, x1=2.8004, cbar=3.5017, delta0=0.097
        )

    def test_radiative_anchors_are_the_estar_transcription(self) -> None:
        """The same ESTAR water anchors the analytic radiative fit passes through."""
        assert MATERIALS[WATER].radiative_anchors == (
            (1.0, 0.0128),
            (10.0, 0.1813),
            (20.0, 0.4008),
        )
