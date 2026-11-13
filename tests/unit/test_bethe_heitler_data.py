"""Screened Bethe-Heitler bremsstrahlung: the data layer.

Pins, in order of how much each would hurt if wrong:

- the bookkeeping identity ``sigma(k > k_cut) <k> + S_rad(k < k_cut) == S_rad``,
  which is what lets the model replace the thin-target one without changing the
  radiative stopping power the engine is calibrated to;
- the constants of the screening functions, through two independent closed forms
  (Tsai's complete-screening spectrum and the radiation lengths tabulated by the
  Particle Data Group);
- the compound rule, against the exact element-by-element sum;
- the flattened tables, against the host queries they were built from.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from pyradmc import ECUT_MEV, ELECTRON_MASS_MEV, PCUT_MEV
from pyradmc.data import berger_seltzer
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.berger_seltzer import CLASSICAL_ELECTRON_RADIUS_CM
from pyradmc.data.bethe_heitler import (
    BetheHeitlerTables,
    born_radiative_stopping_power,
    build_bethe_heitler_tables,
    coulomb_correction,
    screening_parameters,
    shape_array,
    shape_integrals,
)
from pyradmc.data.interface import CrossSectionSource
from pyradmc.data.materials import (
    AIR,
    AVOGADRO,
    CORTICAL_BONE,
    MATERIALS,
    STANDARD_ATOMIC_WEIGHT,
    TUNGSTEN,
    WATER,
)
from pyradmc.data.tables import lookup_loglinear_2d
from pyradmc.physics.brems_bh import bethe_heitler_shape

ALPHA = 7.297_352_5693e-3  # CODATA 2018 fine-structure constant, transcribed independently
ALL_MATERIALS = tuple(range(len(MATERIALS)))
ENERGIES = (0.2, 1.0, 6.0, 20.0, 100.0, 300.0)


class _RegistrySource(CrossSectionSource):
    """Test instrument: every registry material, radiative stopping from its ESTAR fit.

    Only the radiative stopping power is real; the bremsstrahlung queries under test
    are concrete on the interface and need nothing else.
    """

    def radiative_stopping_power(self, energy: float, material: int) -> float:
        coefficients = berger_seltzer.radiative_fit_coefficients(
            MATERIALS[material].radiative_anchors
        )
        return berger_seltzer.radiative_stopping(energy, coefficients)

    def mu_over_rho(self, energy: float, material: int, process: int) -> float:
        raise NotImplementedError

    def mu_over_rho_total(self, energy: float, material: int) -> float:
        raise NotImplementedError

    def majorant(self, energy: float) -> float:
        raise NotImplementedError

    def restricted_stopping_power(self, energy: float, material: int, delta_cut: float) -> float:
        raise NotImplementedError

    def moller_cross_section(self, energy: float, material: int, delta_cut: float) -> float:
        raise NotImplementedError

    def csda_range(self, energy: float, material: int) -> float:
        raise NotImplementedError

    def scattering_power(self, energy: float, material: int) -> float:
        raise NotImplementedError


def _args(material: int) -> tuple[float, float, float, float, float]:
    s = screening_parameters(MATERIALS[material].composition)
    return (s.z_ratio, s.c_nuc, s.c_el, s.g_nuc, s.g_el)


def _clamp_points(energy: float, args: tuple[float, float, float, float, float]) -> list[float]:
    """Where the shape meets its clamp at zero on [PCUT, T], in ``t = ln(k / PCUT)``.

    Adaptive quadrature must be told about the kink: without it scipy returned an
    integral 7e-7 off for tungsten at 6 MeV with no warning (measured against a
    run with the break point supplied and against a panel-density doubling of the
    rule under test).
    """
    k = np.linspace(PCUT_MEV, energy, 4001)
    positive = np.array([bethe_heitler_shape(energy, float(x), *args) > 0.0 for x in k])
    points = []
    for i in np.flatnonzero(positive[1:] != positive[:-1]):
        a, b = float(k[i]), float(k[i + 1])
        for _ in range(60):
            mid = 0.5 * (a + b)
            if (bethe_heitler_shape(energy, mid, *args) > 0.0) == bool(positive[i]):
                a = mid
            else:
                b = mid
        points.append(math.log(0.5 * (a + b) / PCUT_MEV))
    return points


def _log_integrals(
    energy: float, args: tuple[float, float, float, float, float]
) -> tuple[float, float]:
    """Adaptive ``(int G dk, int G/k dk)`` over [PCUT, T], in ``t = ln(k / PCUT)``."""
    log_ratio = math.log(energy / PCUT_MEV)
    points = _clamp_points(energy, args) or None

    def shape(t: float) -> float:
        return bethe_heitler_shape(energy, PCUT_MEV * math.exp(t), *args)

    radiated, _ = integrate.quad(
        lambda t: shape(t) * PCUT_MEV * math.exp(t),
        0.0,
        log_ratio,
        points=points,
        epsabs=0.0,
        epsrel=1.0e-11,
        limit=400,
    )
    count, _ = integrate.quad(
        shape, 0.0, log_ratio, points=points, epsabs=0.0, epsrel=1.0e-11, limit=400
    )
    return radiated, count


class TestStoppingPowerIdentity:
    """sigma * <k> + restricted == total, on the interface."""

    @pytest.mark.parametrize("material", ALL_MATERIALS)
    @pytest.mark.parametrize("energy", ENERGIES)
    def test_discrete_plus_restricted_is_total(self, energy: float, material: int) -> None:
        """The mean photon energy is integrated independently, with scipy.

        ``<k>`` on [k_cut, T] is the ratio of the integrals of G and G/k; it comes
        from the scalar physics function through an adaptive quadrature that shares
        nothing with the fixed-panel rule behind the interface.
        """
        source = _RegistrySource()
        args = _args(material)
        total = source.radiative_stopping_power(energy, material)
        sigma = source.bremsstrahlung_cross_section(energy, material, PCUT_MEV)
        restricted = source.restricted_radiative_stopping_power(energy, material, PCUT_MEV)

        radiated, count = _log_integrals(energy, args)
        mean_k = radiated / count

        assert sigma > 0.0
        assert 0.0 < restricted < total
        assert sigma * mean_k + restricted == pytest.approx(total, rel=1.0e-9)

    @pytest.mark.parametrize("material", (WATER, TUNGSTEN))
    def test_no_discrete_channel_at_or_below_the_cut(self, material: int) -> None:
        """An electron at or below k_cut radiates nothing transportable."""
        source = _RegistrySource()
        for energy in (0.5 * PCUT_MEV, PCUT_MEV):
            assert source.bremsstrahlung_cross_section(energy, material, PCUT_MEV) == 0.0
            assert source.restricted_radiative_stopping_power(
                energy, material, PCUT_MEV
            ) == source.radiative_stopping_power(energy, material)

    def test_cross_section_falls_as_the_cut_rises(self) -> None:
        """Raising k_cut moves events into the restricted stopping power."""
        source = _RegistrySource()
        sigmas = [source.bremsstrahlung_cross_section(6.0, WATER, c) for c in (0.01, 0.05, 0.5)]
        restricted = [
            source.restricted_radiative_stopping_power(6.0, WATER, c) for c in (0.01, 0.05, 0.5)
        ]
        assert sigmas == sorted(sigmas, reverse=True)
        assert restricted == sorted(restricted)

    def test_analytic_source_answers_for_water(self) -> None:
        """The shipped analytic backend inherits the queries unchanged."""
        xs = AnalyticCrossSections()
        registry = _RegistrySource()
        for energy in (1.0, 6.0, 20.0):
            assert xs.bremsstrahlung_cross_section(energy, WATER, PCUT_MEV) == pytest.approx(
                registry.bremsstrahlung_cross_section(energy, WATER, PCUT_MEV), rel=1.0e-12
            )
        with pytest.raises(ValueError, match="material"):
            xs.bremsstrahlung_cross_section(6.0, TUNGSTEN, PCUT_MEV)
        with pytest.raises(ValueError, match="non-positive"):
            xs.bremsstrahlung_cross_section(6.0, WATER, 0.0)

    def test_tabulated_source_splits_its_own_radiative_table(self) -> None:
        """The tabulated backend normalizes to the radiative stopping it compiled.

        The split is concrete on the interface, so a tabulated source serves it
        from the ``radiative_stopping`` table it already carries: the two parts
        must add back to that table's interpolant, whatever its values.
        """
        from pyradmc.data.interface import PhotonProcess
        from pyradmc.data.tabulated.model import TabulatedData
        from pyradmc.data.tabulated.source import TabulatedCrossSections

        pe = np.geomspace(0.01, 25.0, 32)
        ee = np.geomspace(0.1, 25.0, 32)
        data = TabulatedData(
            photon_energies=pe,
            mu_over_rho={PhotonProcess.COMPTON: (0.1 / pe)[None, :]},
            electron_energies=ee,
            restricted_stopping=(2.0 / ee)[None, :],
            radiative_stopping=(0.01 * ee)[None, :],
            moller=(0.1 / ee)[None, :],
            csda_range=(0.5 * ee)[None, :],
            scattering_power=(1.0 / ee)[None, :],
            delta_cut=ECUT_MEV,
            materials=("water",),
            provenance="synthetic",
        )
        source = TabulatedCrossSections(data)
        args = _args(WATER)
        for energy in (0.5, 6.0, 20.0):
            radiated, count = _log_integrals(energy, args)
            sigma = source.bremsstrahlung_cross_section(energy, WATER, PCUT_MEV)
            restricted = source.restricted_radiative_stopping_power(energy, WATER, PCUT_MEV)
            assert sigma * radiated / count + restricted == pytest.approx(
                source.radiative_stopping_power(energy, WATER), rel=1.0e-9
            )


class TestShapeQuadrature:
    @pytest.mark.parametrize("material", (WATER, CORTICAL_BONE, AIR, TUNGSTEN))
    def test_array_shape_is_the_scalar_physics_function(self, material: int) -> None:
        """The vectorized shape used for quadrature cannot drift from the sampler's.

        The data layer integrates a NumPy transcription of the scalar function the
        sampler rejects against; the bookkeeping identity is only as good as their
        agreement, so it is pinned to rounding.
        """
        s = screening_parameters(MATERIALS[material].composition)
        for energy in ENERGIES:
            k = np.geomspace(1.0e-6 * energy, energy, 301)
            scalar = np.array([bethe_heitler_shape(energy, float(x), *_args(material)) for x in k])
            np.testing.assert_allclose(shape_array(energy, k, s), scalar, rtol=1e-13, atol=1e-15)

    @pytest.mark.parametrize("material", (WATER, TUNGSTEN))
    @pytest.mark.parametrize("energy", ENERGIES)
    def test_fixed_panel_integrals_match_adaptive_quadrature(
        self, energy: float, material: int
    ) -> None:
        s = screening_parameters(MATERIALS[material].composition)
        args = _args(material)
        below, above, count = shape_integrals(energy, PCUT_MEV, s)

        def shape(k: float) -> float:
            return bethe_heitler_shape(energy, k, *args)

        ref_below, _ = integrate.quad(shape, 0.0, PCUT_MEV, epsabs=0.0, epsrel=1e-12)
        ref_above, ref_count = _log_integrals(energy, args)
        assert below == pytest.approx(ref_below, rel=1e-8)
        assert above == pytest.approx(ref_above, rel=1e-9)
        assert count == pytest.approx(ref_count, rel=1e-9)


class TestScreeningConstants:
    """Independent closed forms for the constants of the screening functions."""

    def test_coulomb_correction_values(self) -> None:
        """Davies-Bethe-Maximon f(Z) against its defining series.

        ``f = a^2 sum_n 1 / (n (n^2 + a^2))`` with ``a = alpha Z``; the four-term
        polynomial in the implementation is its standard approximation, good to
        four digits up to uranium.
        """
        for z in (1, 8, 20, 74, 92):
            a = ALPHA * z
            series = a * a * sum(1.0 / (n * (n * n + a * a)) for n in range(1, 200_000))
            assert coulomb_correction(z) == pytest.approx(series, rel=2.0e-3, abs=1e-9)

    @pytest.mark.parametrize("z", (6, 8, 20, 74))
    def test_complete_screening_limit_is_tsai_closed_form(self, z: int) -> None:
        r"""At very high energy the shape is Tsai's complete-screening spectrum.

        .. math::

            k \frac{d\sigma}{dk} = 4 \alpha r_e^2 \left\{
                \left(\tfrac43 - \tfrac43 y + y^2\right)
                \left[Z^2 (L_{el} - f) + Z L_{inel}\right]
                + \tfrac19 (1 - y)(Z^2 + Z) \right\}

        with ``L_el = ln(184.15 Z^-1/3)`` and ``L_inel = ln(1194 Z^-2/3)``
        (Tsai 1974, doi:10.1103/RevModPhys.46.815; reproduced as eq. 34.29 of the
        Particle Data Group review). The shape function is this divided by
        ``(16/3) alpha r_e^2 Z^2``.
        """
        s = screening_parameters(((z, 1.0),))
        energy = 1.0e6
        total = energy + ELECTRON_MASS_MEV
        l_el = math.log(184.15 * z ** (-1.0 / 3.0))
        l_inel = math.log(1194.0 * z ** (-2.0 / 3.0))
        f = coulomb_correction(z)
        for y in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9):
            tsai = (4.0 / 3.0 - 4.0 / 3.0 * y + y * y) * (z * z * (l_el - f) + z * l_inel) + (
                1.0 - y
            ) * (z * z + z) / 9.0
            expected = 0.75 * tsai / (z * z)
            got = bethe_heitler_shape(
                energy, y * total, s.z_ratio, s.c_nuc, s.c_el, s.g_nuc, s.g_el
            )
            assert got == pytest.approx(expected, rel=5.0e-4)

    @pytest.mark.parametrize(
        ("material", "radiation_length"),
        [(WATER, 36.08), (AIR, 36.62), (TUNGSTEN, 6.76)],
    )
    def test_radiation_length_matches_particle_data_group(
        self, material: int, radiation_length: float
    ) -> None:
        r"""``1/X_0 = 4 \alpha r_e^2 N_A \sum_i (w_i/A_i)[Z_i^2 (L_{el}-f) + Z_i L_{inel}]``.

        The Particle Data Group's tabulated radiation lengths (g/cm^2; water
        36.08, dry air 36.62, tungsten 6.76) are computed with exactly this
        expression, with Tsai's tabulated ``L`` for hydrogen — so the test pins
        the absolute constants, the Coulomb correction and the low-Z table
        through numbers transcribed from a different source than the formula.
        """
        s = screening_parameters(MATERIALS[material].composition)
        bracket = 0.25 * 20.863 - s.c_nuc + s.z_ratio * (0.25 * 28.340 - s.c_el)
        inverse_x0 = (
            4.0 * ALPHA * CLASSICAL_ELECTRON_RADIUS_CM**2 * AVOGADRO * s.z2_per_gram * bracket
        )
        assert 1.0 / inverse_x0 == pytest.approx(radiation_length, rel=2.0e-3)

    def test_unscreened_limit_is_material_independent(self) -> None:
        """At large screening variable the nuclear charge drops out of the shape.

        ``phi_1/4 - ln(Z)/3 -> ln(2 E E'/(m k)) - 1/2`` for a bare nucleus: the
        geometric-mean compound rule is the one that preserves this, so the main
        bracket of two very different media must agree where screening is negligible
        (low energy, hard photon), up to the Coulomb correction that is removed here.
        """
        energy = 0.3
        k = 0.29
        values = []
        for z in (6, 74):
            s = screening_parameters(((z, 1.0),))
            c_nuc = s.c_nuc - coulomb_correction(z)
            values.append(bethe_heitler_shape(energy, k, 0.0, c_nuc, s.c_el, s.g_nuc, s.g_el))
        total = energy + ELECTRON_MASS_MEV
        y = k / total
        bare = math.log(2.0 * total * (total - k) / (ELECTRON_MASS_MEV * k)) - 0.5
        expected = (1.0 - y + 0.75 * y * y) * bare
        assert values[0] == pytest.approx(expected, rel=0.03)
        assert values[1] == pytest.approx(expected, rel=0.03)


def _elementwise_shape(
    energy: float, k: float, composition: tuple[tuple[int, float], ...]
) -> float:
    """Exact per-element sum of the shape, weighted by atom number times Z^2."""
    weights = [w * z * z / STANDARD_ATOMIC_WEIGHT[z] for z, w in composition]
    total = 0.0
    for (z, _), weight in zip(composition, weights, strict=True):
        s = screening_parameters(((z, 1.0),))
        total += weight * bethe_heitler_shape(
            energy, k, s.z_ratio, s.c_nuc, s.c_el, s.g_nuc, s.g_el
        )
    return total / sum(weights)


class TestCompoundRule:
    @pytest.mark.parametrize("z", (1, 8, 74))
    def test_single_element_is_invariant_under_splitting(self, z: int) -> None:
        single = screening_parameters(((z, 1.0),))
        split = screening_parameters(((z, 0.25), (z, 0.75)))
        assert tuple(split) == pytest.approx(tuple(single), rel=1.0e-12)

    @pytest.mark.parametrize("material", ALL_MATERIALS)
    def test_effective_parameters_track_the_element_sum(self, material: int) -> None:
        """The one-parameter-set compound shape stays within 0.3 percent of the exact sum.

        The kernel evaluates one set of screening parameters per material instead
        of looping over elements. Where screening is complete or absent the two are
        identical by construction; in between they differ because the screening
        functions are evaluated at a geometric-mean argument. Measured: the largest
        deviation over 0.2-300 MeV and 1e-4 < k/T < 0.98 is 0.17 percent (water,
        lung, adipose, cortical bone: the hydrogen-bearing media), of a shape whose
        absolute scale is renormalized to the stopping power anyway.
        """
        composition = MATERIALS[material].composition
        s = screening_parameters(composition)
        worst = 0.0
        for energy in ENERGIES:
            for k in np.geomspace(1.0e-4 * energy, 0.98 * energy, 60):
                exact = _elementwise_shape(energy, float(k), composition)
                got = bethe_heitler_shape(
                    energy, float(k), s.z_ratio, s.c_nuc, s.c_el, s.g_nuc, s.g_el
                )
                worst = max(worst, abs(got / exact - 1.0))
        assert worst < 3.0e-3


class TestBornNormalization:
    @pytest.mark.parametrize(
        ("material", "low", "high"),
        [
            (WATER, 0.97, 1.03),
            (AIR, 0.97, 1.03),
            (CORTICAL_BONE, 0.97, 1.03),
            (TUNGSTEN, 1.00, 1.07),
        ],
    )
    def test_born_stopping_power_is_close_to_estar(
        self, material: int, low: float, high: float
    ) -> None:
        """Un-normalized, the model reproduces ESTAR at 10 and 20 MeV.

        The transport never uses the absolute Born value: the interface rescales
        the shape to :meth:`radiative_stopping_power`. But a correct screened
        Bethe-Heitler cross-section has to land near the Seltzer-Berger totals at
        these energies, so the rescaling factor (EGS4's ``A'``) is an end-to-end
        check on the formula and its constants that shares nothing with them.
        Measured ESTAR/Born at (10, 20) MeV: water 0.988, 0.979; air 0.984, 0.995;
        cortical bone 0.995, 1.000; tungsten 1.060, 1.020. At 1 MeV the same ratio
        is 0.93-0.98 in the low-Z media and 1.45 in tungsten, which is the known
        failure of the Born approximation there and the reason the model is
        normalized rather than used absolutely.
        """
        s = screening_parameters(MATERIALS[material].composition)
        for energy, estar in MATERIALS[material].radiative_anchors[1:]:
            ratio = estar / born_radiative_stopping_power(energy, s)
            assert low < ratio < high, f"A' = {ratio:.4f} at {energy} MeV"


TABLE_E_MAX = 21.0


@pytest.fixture(scope="module")
def source() -> AnalyticCrossSections:
    return AnalyticCrossSections()


@pytest.fixture(scope="module")
def tables(source: AnalyticCrossSections) -> BetheHeitlerTables:
    return build_bethe_heitler_tables(
        source, ecut=ECUT_MEV, pcut=PCUT_MEV, e_max=TABLE_E_MAX, n_points=2048
    )


class TestFlattenedTables:
    def test_grid_is_the_electron_grid_of_the_cross_section_tables(
        self, source: AnalyticCrossSections, tables: BetheHeitlerTables
    ) -> None:
        """One lookup metadata pair serves both table sets in the kernel."""
        base = source.build_tables(ecut=ECUT_MEV, pcut=PCUT_MEV, e_max=TABLE_E_MAX)
        assert tables.n_points == base.n_points
        assert tables.electron_log_e_min == base.electron_log_e_min
        assert tables.electron_inv_dlog == base.electron_inv_dlog
        assert tables.cross_section.shape == (source.n_materials, tables.n_points)
        assert tables.stopping_radiative_restricted.shape == tables.cross_section.shape
        assert tables.screening.shape == (source.n_materials, 5)

    def test_lookup_parity_with_host_queries(
        self, source: AnalyticCrossSections, tables: BetheHeitlerTables
    ) -> None:
        """Interpolated values reproduce the exact host methods to 2e-4."""
        rng = np.random.default_rng(1234)
        for energy in np.exp(rng.uniform(math.log(ECUT_MEV), math.log(TABLE_E_MAX), 200)):
            e = float(energy)
            for values, exact in (
                (tables.cross_section, source.bremsstrahlung_cross_section(e, WATER, PCUT_MEV)),
                (
                    tables.stopping_radiative_restricted,
                    source.restricted_radiative_stopping_power(e, WATER, PCUT_MEV),
                ),
            ):
                got = lookup_loglinear_2d(
                    values,
                    WATER,
                    tables.electron_log_e_min,
                    tables.electron_inv_dlog,
                    tables.n_points,
                    e,
                )
                assert got == pytest.approx(exact, rel=2.0e-4)

    def test_screening_row_is_the_interface_answer(
        self, source: AnalyticCrossSections, tables: BetheHeitlerTables
    ) -> None:
        s = source.bremsstrahlung_screening(WATER)
        np.testing.assert_array_equal(
            tables.screening[WATER], [s.z_ratio, s.c_nuc, s.c_el, s.g_nuc, s.g_el]
        )
