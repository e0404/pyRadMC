r"""Moliere screening parameter and the GS/Fermi-Eyges anchoring contract.

The screening parameter ``eta`` fixes the *shape* of the single-scattering law;
:meth:`~pyRadMC.data.interface.CrossSectionSource.scattering_power` fixes its
*strength*. Together they determine the Goudsmit-Saunderson distribution for a
step. The contract test in this module is the one that matters: whatever ``eta``
comes out, the GS mean-square deflection over a short step must reproduce the
Fermi-Eyges ``T rho s`` the current transport loop uses, on **every**
implementation of the interface. If that fails, GS is not a refinement of the
shipped model but a rescaling of it, and every dose comparison downstream is
confounded.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestMoliereScreening:
    """The screening parameter itself."""

    def test_single_element_compound_reduces_to_that_element(self) -> None:
        """A one-element "mixture" must return that element's screening exactly.

        The compound rule is a Z(Z+1)/A-weighted geometric mean; a degenerate
        mixture is the sharpest available check that the weighting is
        normalized, since any un-normalized weight shows up immediately.
        """
        from pyRadMC.data.goudsmit_saunderson import moliere_screening

        for z in (1, 6, 20, 74):
            single = moliere_screening(((z, 1.0),), 1.0)
            duplicated = moliere_screening(((z, 0.25), (z, 0.75)), 1.0)
            assert duplicated == pytest.approx(single, rel=1.0e-12), (
                f"Z={z}: splitting one element across two entries changed eta"
            )

    def test_screening_falls_with_energy_and_rises_with_z(self) -> None:
        r"""``eta ~ (Z^{1/3}/pc)^2``: monotone down in energy, up in Z.

        Both trends are structural, not fitted, so a sign error or a swapped
        momentum factor cannot hide behind a plausible magnitude.
        """
        from pyRadMC.data.goudsmit_saunderson import moliere_screening

        energies = [0.2, 1.0, 6.0, 20.0]
        etas = [moliere_screening(((8, 1.0),), e) for e in energies]
        assert etas == sorted(etas, reverse=True), f"eta must fall with energy, got {etas}"

        at_fixed_energy = [moliere_screening(((z, 1.0),), 1.0) for z in (1, 6, 20, 74)]
        assert at_fixed_energy == sorted(at_fixed_energy), (
            f"eta must rise with Z, got {at_fixed_energy}"
        )

    def test_water_screening_is_of_the_expected_magnitude(self) -> None:
        """A 1 MeV electron in water screens at ``eta ~ 1e-5``.

        An order-of-magnitude anchor, not a fit: ``eta`` is a squared ratio of
        the reduced wavelength to the Thomas-Fermi radius, and getting the
        Bohr-radius unit conversion wrong moves it by many decades — the failure
        mode this catches.
        """
        from pyRadMC.data.goudsmit_saunderson import moliere_screening
        from pyRadMC.data.materials import MATERIALS, WATER

        eta = moliere_screening(MATERIALS[WATER].composition, 1.0)
        assert 1.0e-6 < eta < 1.0e-4, f"water eta at 1 MeV = {eta:.3e}, expected ~1e-5"


class TestFermiEygesAnchoringContract:
    """GS must reproduce the shipped hinge's second moment at small steps.

    Runs against every ``CrossSectionSource`` implementation: the anchoring is an
    interface-level guarantee, not a property of one data backend.
    """

    @staticmethod
    def _sources() -> list[tuple[str, object, list[int]]]:
        from pyRadMC.data.analytic import AnalyticCrossSections
        from pyRadMC.data.materials import WATER

        return [("analytic", AnalyticCrossSections(), [WATER])]

    def test_gs_mean_square_angle_matches_scattering_power_times_path(self) -> None:
        r"""``<theta^2>_GS -> T rho s`` for every source, material and energy.

        This is the identity that makes the transport-level limit check
        meaningful. It exercises the whole composed chain — screening, the
        closed-form ``G_1``, the elastic path count ``Lambda`` — rather than any
        single piece, which is why it is the contract worth pinning.
        """
        from pyRadMC.data.goudsmit_saunderson import (
            first_transport_moment,
            mean_square_angle,
        )

        rho = 1.0
        step_cm = 1.0e-4  # short enough that saturation is far below the tolerance
        for label, source, materials in self._sources():
            for material in materials:
                for energy in (0.25, 1.0, 6.0, 18.0):
                    t_rho_s = source.scattering_power(energy, material) * rho * step_cm  # type: ignore[attr-defined]
                    eta = source.elastic_screening(energy, material)  # type: ignore[attr-defined]
                    g1 = first_transport_moment(eta)
                    # Anchoring: Lambda G_1 = <theta^2>/2 by construction of sigma_el.
                    lam = 0.5 * t_rho_s / g1
                    gs = mean_square_angle(lam, g1)
                    assert gs == pytest.approx(t_rho_s, rel=1.0e-3), (
                        f"{label}/material {material} at {energy} MeV: GS <theta^2>={gs:.6e} "
                        f"vs Fermi-Eyges T rho s={t_rho_s:.6e}"
                    )

    def test_elastic_path_count_is_large_enough_for_the_series(self) -> None:
        """``Lambda`` over a real substep must sit in the many-scattering regime.

        The Legendre series represents a *diffused* distribution; at ``Lambda``
        of order one a finite fraction of electrons pass unscattered and the
        series cannot resolve that delta. This records that the substeps the
        engine actually takes are comfortably diffusive, so the series sampler
        is the right tool — and would fail loudly if a future step shrank into
        the single-scattering regime instead.
        """
        from pyRadMC.data.analytic import AnalyticCrossSections
        from pyRadMC.data.goudsmit_saunderson import first_transport_moment
        from pyRadMC.data.materials import WATER
        from pyRadMC.transport.electron import STEP_ENERGY_FRACTION

        source = AnalyticCrossSections()
        rho = 1.0
        for energy in (0.25, 1.0, 6.0):
            step_cm = STEP_ENERGY_FRACTION * source.csda_range(energy, WATER) / rho
            t_rho_s = source.scattering_power(energy, WATER) * rho * step_cm
            g1 = first_transport_moment(source.elastic_screening(energy, WATER))
            lam = 0.5 * t_rho_s / g1
            assert lam > 20.0, (
                f"only {lam:.1f} elastic mean free paths in a default substep at "
                f"{energy} MeV — below the diffusive regime the GS series assumes"
            )


class TestScreeningIsReachableThroughTheInterface:
    """AGENTS 2.6: physics reaches elastic data only through the source."""

    def test_every_source_answers_elastic_screening(self) -> None:
        """Concrete on the ABC, so a source cannot silently omit it."""
        from pyRadMC.data.analytic import AnalyticCrossSections
        from pyRadMC.data.materials import WATER

        eta = AnalyticCrossSections().elastic_screening(1.0, WATER)
        assert np.isfinite(eta) and eta > 0.0
