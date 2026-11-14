"""Pair-production screening data: Coulomb correction, mixture parameters, the DCS.

The sampler in :mod:`pyradmc.physics.pair` is tested against the differential
cross-section this module's subject exposes, so the DCS itself needs anchors that do
not come from the sampler:

- the Davies-Bethe-Maximon Coulomb correction against its defining infinite series;
- Tsai's radiation logarithms (the weights of the mixture rule) against the PDG
  radiation lengths, which are built from exactly those logarithms;
- the Butcher-Messel screening functions against the two analytic facts they were
  fitted to reproduce (the complete-screening limit and its 2/3 offset).
"""

from __future__ import annotations

import itertools
import math

import pytest
from scipy import integrate

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data.berger_seltzer import CLASSICAL_ELECTRON_RADIUS_CM
from pyradmc.data.interface import CrossSectionSource
from pyradmc.data.materials import (
    AVOGADRO,
    CORTICAL_BONE,
    MATERIALS,
    STANDARD_ATOMIC_WEIGHT,
    TUNGSTEN,
    WATER,
)
from pyradmc.data.pair import (
    FINE_STRUCTURE_CONSTANT,
    PAIR_COULOMB_ENERGY_MEV,
    PAIR_LOW_ENERGY_LIMIT_MEV,
    PairScreening,
    coulomb_correction,
    pair_screening_parameters,
    pair_sharing_dcs,
    radiation_logarithms,
    screening_phi1,
    screening_phi2,
)

# Radiation lengths in g/cm^2, PDG atomic and nuclear properties tables
# (pdg.lbl.gov/AtomicNuclearProperties; Tsai's formula with the same logarithms,
# Tsai (1974), doi:10.1103/RevModPhys.46.815). The water entry is the value the
# analytic backend already carries as ``_WATER_RADIATION_LENGTH_G_CM2``.
PDG_RADIATION_LENGTH: dict[str, float] = {"hydrogen": 63.04, "oxygen": 34.24, "tungsten": 6.76}
PDG_WATER_RADIATION_LENGTH = 36.08


def _inverse_radiation_length(z: int) -> float:
    """Tsai's 1/X0 for one element, in cm^2/g (Tsai (1974) eq. 3.66)."""
    l_rad, l_rad_prime = radiation_logarithms(z)
    return (
        4.0
        * FINE_STRUCTURE_CONSTANT
        * CLASSICAL_ELECTRON_RADIUS_CM**2
        * AVOGADRO
        / STANDARD_ATOMIC_WEIGHT[z]
        * (z * z * (l_rad - coulomb_correction(z)) + z * l_rad_prime)
    )


class TestCoulombCorrection:
    @pytest.mark.parametrize("z", [1, 8, 20, 29, 74, 82])
    def test_matches_the_defining_series(self, z: int) -> None:
        """f_c(Z) = a^2 sum_n 1/(n (n^2 + a^2)), a = alpha Z.

        The four-term polynomial is Davies, Bethe & Maximon's own approximation to
        this series, quoted as accurate to four figures up to a = 2/3 (Z = 92).
        """
        a = FINE_STRUCTURE_CONSTANT * z
        series = a * a * sum(1.0 / (n * (n * n + a * a)) for n in range(1, 200_000))
        # The truncated tail is below a^2 / (2 N^2) ~ 1e-11 a^2.
        assert coulomb_correction(z) == pytest.approx(series, rel=2e-4)

    def test_vanishes_like_z_squared(self) -> None:
        assert coulomb_correction(1) == pytest.approx(
            1.2020569 * FINE_STRUCTURE_CONSTANT**2, rel=1e-3
        )


class TestRadiationLogarithms:
    @pytest.mark.parametrize(("z", "name"), [(1, "hydrogen"), (8, "oxygen"), (74, "tungsten")])
    def test_reproduce_pdg_radiation_lengths(self, z: int, name: str) -> None:
        x0 = 1.0 / _inverse_radiation_length(z)
        assert x0 == pytest.approx(PDG_RADIATION_LENGTH[name], rel=3e-3)

    def test_reproduce_water_radiation_length(self) -> None:
        inverse = sum(w * _inverse_radiation_length(z) for z, w in MATERIALS[WATER].composition)
        assert 1.0 / inverse == pytest.approx(PDG_WATER_RADIATION_LENGTH, rel=3e-3)


class TestSharedWithBremsstrahlung:
    """Literal pins of the helpers the bremsstrahlung model shares.

    The fine-structure constant, ``coulomb_correction`` and Tsai's radiation
    logarithms are the same physical quantities for the screened Bethe-Heitler
    bremsstrahlung model (``pyradmc/data/bethe_heitler.py``) and for pair
    production, and must be the same *numbers*, or a medium would be screened
    differently for the photon that makes a pair and the lepton that radiates.
    Both modules now read one implementation, :mod:`pyradmc.data.screening`
    (``tests/unit/test_screening_shared.py``). The literals below were printed
    from the bremsstrahlung module while it still carried its own copy, to full
    double precision; the tolerance is libm rounding only, and they say which
    numbers the shared helper must return.
    """

    def test_fine_structure_constant(self) -> None:
        assert FINE_STRUCTURE_CONSTANT == 7.2973525693e-3

    @pytest.mark.parametrize(
        ("z", "expected"),
        [
            (1, 6.40083830230286e-05),
            (6, 0.0023006040397510067),
            (8, 0.004084720814309612),
            (20, 0.02514361676656564),
            (29, 0.05184058187270907),
            (74, 0.28174435078545906),
            (82, 0.3316239382143995),
        ],
    )
    def test_coulomb_correction(self, z: int, expected: float) -> None:
        assert coulomb_correction(z) == pytest.approx(expected, rel=1e-14, abs=0.0)

    @pytest.mark.parametrize(
        ("z", "expected"),
        [
            # Tsai's table B.2 for Z <= 4, where the Thomas-Fermi model fails.
            (1, (5.31, 6.144)),
            (2, (4.79, 5.621)),
            (3, (4.74, 5.805)),
            (4, (4.71, 5.924)),
            # Above: 20.863 / 4 - ln(Z) / 3 and 28.340 / 4 - 2 ln(Z) / 3.
            (5, (4.679270695855299, 6.0120413917106)),
            (6, (4.618496843590648, 5.890493687181297)),
            (8, (4.5226028194400545, 5.698705638880109)),
            (20, (4.217172575482003, 5.087845150964006)),
            (29, (4.093318056671175, 4.84013611334235)),
            (74, (3.7810616355986095, 4.21562327119722)),
            (82, (3.7468435842452488, 4.147187168490498)),
        ],
    )
    def test_radiation_logarithms(self, z: int, expected: tuple[float, float]) -> None:
        assert radiation_logarithms(z) == pytest.approx(expected, rel=1e-14, abs=0.0)

    def test_radiation_logarithms_are_tsai_to_the_fit_rounding(self) -> None:
        """The pinned form is Tsai's ln(184.15 Z^-1/3), ln(1194 Z^-2/3) to 5 figures.

        20.863 and 28.340 are his rounded ``4 ln 184.15`` and ``4 ln 1194``; the
        rounding shifts the logarithms by 6.4e-7 and 6.4e-5, independent of Z.
        """
        for z in (5, 8, 29, 74):
            l_rad, l_rad_prime = radiation_logarithms(z)
            assert l_rad == pytest.approx(math.log(184.15 * z ** (-1.0 / 3.0)), abs=1e-6)
            assert l_rad_prime == pytest.approx(math.log(1194.0 * z ** (-2.0 / 3.0)), abs=1e-4)


class TestScreeningFunctions:
    def test_complete_screening_limit(self) -> None:
        """Phi1(0) = 4 ln 183 and Phi1(0) - Phi2(0) = 2/3 (Bethe-Heitler, Thomas-Fermi)."""
        assert screening_phi1(0.0) == pytest.approx(4.0 * math.log(183.0), rel=2e-3)
        assert screening_phi1(0.0) - screening_phi2(0.0) == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_branches_join_at_delta_one(self) -> None:
        """The polynomial and logarithmic fits meet within the fit's own 1 percent."""
        below, above = 1.0, 1.0 + 1e-12
        assert screening_phi1(below) == pytest.approx(screening_phi1(above), rel=1e-2)
        assert screening_phi2(below) == pytest.approx(screening_phi2(above), rel=1e-2)
        assert screening_phi1(above) == screening_phi2(above)

    def test_monotone_decreasing_within_each_branch(self) -> None:
        for phi in (screening_phi1, screening_phi2):
            low = [phi(0.01 * i) for i in range(101)]
            high = [phi(1.0001 + 0.5 * i) for i in range(400)]
            assert all(a > b for a, b in itertools.pairwise(low))
            assert all(a > b for a, b in itertools.pairwise(high))


class TestMixtureParameters:
    def test_pure_element_reduces_to_its_own_z(self) -> None:
        params = pair_screening_parameters(((74, 1.0),))
        assert params.screening_constant == pytest.approx(136.0 * 74.0 ** (-1.0 / 3.0), rel=1e-12)
        assert params.coulomb_low == pytest.approx(8.0 / 3.0 * math.log(74.0), rel=1e-12)
        assert params.coulomb_high - params.coulomb_low == pytest.approx(
            8.0 * coulomb_correction(74), rel=1e-12
        )

    def test_water_sits_between_its_elements_and_is_oxygen_dominated(self) -> None:
        """Weights go as Z (Z + xi): oxygen carries ~95 percent of water's pair yield."""
        water = pair_screening_parameters(MATERIALS[WATER].composition)
        hydrogen = pair_screening_parameters(((1, 1.0),))
        oxygen = pair_screening_parameters(((8, 1.0),))
        assert oxygen.screening_constant < water.screening_constant < hydrogen.screening_constant
        assert hydrogen.coulomb_low < water.coulomb_low < oxygen.coulomb_low
        share = (water.coulomb_low - hydrogen.coulomb_low) / (
            oxygen.coulomb_low - hydrogen.coulomb_low
        )
        assert 0.92 < share < 0.97

    def test_independent_of_fraction_normalisation(self) -> None:
        composition = MATERIALS[WATER].composition
        scaled = tuple((z, 3.0 * w) for z, w in composition)
        assert pair_screening_parameters(scaled) == pytest.approx(
            pair_screening_parameters(composition), rel=1e-12
        )

    def test_rejects_empty_composition(self) -> None:
        with pytest.raises(ValueError, match="composition"):
            pair_screening_parameters(())

    def test_every_registry_material_has_parameters(self) -> None:
        for material in MATERIALS:
            params = pair_screening_parameters(material.composition)
            assert isinstance(params, PairScreening)
            assert params.screening_constant > 0.0
            assert params.coulomb_high >= params.coulomb_low >= 0.0


class TestSharingDcs:
    WATER_PARAMS = pair_screening_parameters(MATERIALS[WATER].composition)

    @pytest.mark.parametrize("energy", [2.0, 2.5, 6.0, 20.0, 49.9, 50.0, 100.0, 300.0])
    def test_symmetric_and_confined_to_the_kinematic_interval(self, energy: float) -> None:
        eps0 = ELECTRON_MASS_MEV / energy
        for i in range(1, 200):
            eps = i / 200.0
            value = pair_sharing_dcs(energy, eps, self.WATER_PARAMS)
            assert value >= 0.0
            assert value == pytest.approx(
                pair_sharing_dcs(energy, 1.0 - eps, self.WATER_PARAMS), rel=1e-9, abs=1e-12
            )
            if eps < eps0 or eps > 1.0 - eps0:
                assert value == 0.0
        assert pair_sharing_dcs(energy, 0.5, self.WATER_PARAMS) > 0.0

    def test_flat_below_the_low_energy_limit(self) -> None:
        """Below 2 MeV the model is the uniform split (stated approximation)."""
        energy = 0.75 * PAIR_LOW_ENERGY_LIMIT_MEV
        eps0 = ELECTRON_MASS_MEV / energy
        inside = [
            pair_sharing_dcs(energy, eps0 + f * (1 - 2 * eps0), self.WATER_PARAMS)
            for f in (0.01, 0.3, 0.5, 0.99)
        ]
        assert all(v == inside[0] > 0.0 for v in inside)
        assert pair_sharing_dcs(energy, 0.5 * eps0, self.WATER_PARAMS) == 0.0

    def test_no_pair_below_threshold(self) -> None:
        assert pair_sharing_dcs(1.0, 0.5, self.WATER_PARAMS) == 0.0

    def test_high_energy_shape_approaches_complete_screening(self) -> None:
        """At 300 MeV in tungsten the centre of the spectrum is nearly screened.

        Complete screening (delta -> 0) with the Coulomb correction gives the textbook
        ratio DCS(1/2) / DCS(eps) from Phi1(0), Phi2(0); the screened DCS must sit
        within a few percent of it away from the endpoints.
        """
        params = pair_screening_parameters(MATERIALS[TUNGSTEN].composition)
        f_half = params.coulomb_high / 2.0

        def complete(eps: float) -> float:
            return (eps**2 + (1 - eps) ** 2) * (screening_phi1(0.0) - f_half) + 2.0 / 3.0 * eps * (
                1 - eps
            ) * (screening_phi2(0.0) - f_half)

        for eps in (0.2, 0.35):
            screened = pair_sharing_dcs(300.0, eps, params) / pair_sharing_dcs(300.0, 0.5, params)
            assert screened == pytest.approx(complete(eps) / complete(0.5), rel=0.05)

    def test_coulomb_correction_switches_on_at_50_mev(self) -> None:
        """Only the high-energy branch carries f_c; in tungsten it lowers the DCS."""
        params = pair_screening_parameters(MATERIALS[TUNGSTEN].composition)
        below = pair_sharing_dcs(PAIR_COULOMB_ENERGY_MEV * (1 - 1e-9), 0.5, params)
        above = pair_sharing_dcs(PAIR_COULOMB_ENERGY_MEV, 0.5, params)
        expected_drop = 0.5 * (params.coulomb_high - params.coulomb_low) * (0.5 + 1.0 / 6.0)
        assert below - above == pytest.approx(expected_drop, rel=1e-6)

    @pytest.mark.parametrize(
        ("material", "drop"),
        [(WATER, 1.201e-3), (CORTICAL_BONE, 4.143e-3), (TUNGSTEN, 9.439e-2)],
    )
    def test_size_of_the_50_mev_step(self, material: int, drop: float) -> None:
        """Relative drop of the DCS at equal sharing across the switch, per medium.

        The number the module docstring quotes for the stated approximation: a
        per-mille effect in water and bone, nine percent in tungsten.
        """
        params = pair_screening_parameters(MATERIALS[material].composition)
        below = pair_sharing_dcs(PAIR_COULOMB_ENERGY_MEV * (1 - 1e-12), 0.5, params)
        above = pair_sharing_dcs(PAIR_COULOMB_ENERGY_MEV, 0.5, params)
        assert 1.0 - above / below == pytest.approx(drop, rel=1e-3)

    def test_integrable(self) -> None:
        total, _ = integrate.quad(
            lambda e: pair_sharing_dcs(20.0, e, self.WATER_PARAMS), 0.0, 1.0, limit=200
        )
        assert total > 0.0


class TestInterface:
    def test_source_answers_from_the_material_composition(self) -> None:
        from pyradmc.data.analytic import AnalyticCrossSections

        xs = AnalyticCrossSections()
        assert xs.pair_screening(WATER) == pair_screening_parameters(MATERIALS[WATER].composition)
        with pytest.raises(ValueError, match="material"):
            xs.pair_screening(xs.n_materials)

    def test_concrete_on_the_interface(self) -> None:
        """No implementation can omit it: the default answers for the whole registry."""
        assert "pair_screening" not in CrossSectionSource.__abstractmethods__
