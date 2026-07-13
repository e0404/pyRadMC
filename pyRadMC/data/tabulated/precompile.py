"""Assemble compiled :class:`TabulatedData` from the EPICS source libraries.

The precompiler is the one place the provenance-specific parsers meet: it mixes EPDL
photon cross sections and EEDL elastic scattering into per-material tables on the
canonical geometric grids, fills the electron stopping quantities according to the
chosen strategy, and hands back a :class:`~pyRadMC.data.tabulated.model.TabulatedData`
the loader can wrap. It is compile-time only — no kernel ever sees it.

Two data sources are provenance-fixed regardless of strategy: **photons** come from
EPDL (validated sub-percent against NIST XCOM) and **elastic scattering power** from
EEDL (the transported large-angle moment; better than the analytic Highland form and
free of the delta-ray-restriction subtlety below). The strategy governs only the
electron **stopping**:

``berger-seltzer`` (default)
    Collision (restricted), radiative, Moller and CSDA range from the analytic ICRU-37
    backend evaluated on the grid — exact against ESTAR, and restriction handled
    cleanly by the Berger-Seltzer form. This is the shippable water configuration.

``eedl``
    Restricted collision (excitation MT=528 + ionization MT=534-572, integrating the
    differential electro-ionization spectra below ``delta_cut``) and radiative stopping
    from EEDL. A few percent from ESTAR (like the EEDL radiative evaluation), so it is
    the non-default provenance-consistent option, not the accuracy default. Moller and
    CSDA range stay analytic under both strategies.

Water-only for now: the analytic electron backend (Moller, CSDA, the berger-seltzer
stopping) is water-only, so multi-material compilation waits on the deferred materials
task. Photons, scattering and the EEDL electron stopping already mix arbitrary media.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from pyRadMC import ECUT_MEV, PCUT_MEV, RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.data.tables import TABLE_POINTS
from pyRadMC.data.tabulated import eedl, epdl
from pyRadMC.data.tabulated.model import TabulatedData

__all__ = ["ElectronStoppingStrategy", "compile_water"]

_WATER_FORMULA = {1: 2, 8: 1}  # H2O by atom count
_COHERENT_GRID_POINTS = 512  # momentum-transfer nodes for the coherent form-factor cumulative


class ElectronStoppingStrategy(str, Enum):
    """Which evaluation fills the electron stopping quantities; see the module docstring."""

    BERGER_SELTZER = "berger-seltzer"
    EEDL = "eedl"


def compile_water(
    epdl_text: str,
    eedl_text: str,
    *,
    strategy: ElectronStoppingStrategy = ElectronStoppingStrategy.BERGER_SELTZER,
    pcut: float = PCUT_MEV,
    ecut: float = ECUT_MEV,
    e_max: float = 30.0,
    n_points: int = TABLE_POINTS,
    delta_cut: float | None = None,
) -> TabulatedData:
    """Compile liquid-water :class:`TabulatedData` from EPDL and EEDL text.

    Photons come from ``epdl_text`` and elastic scattering from ``eedl_text``; the
    electron stopping quantities follow ``strategy``. Grids are geometric on
    ``[pcut/2, e_max]`` (photons) and ``[ecut/2, e_max]`` (electrons), matching
    :func:`pyRadMC.data.tables.build_cross_section_tables`. ``delta_cut`` defaults to
    ``ecut`` (the delta-ray production threshold the restricted quantities are built at).
    """
    if e_max <= max(pcut, ecut):
        raise ValueError(f"e_max={e_max} MeV does not cover the transport range")
    cut = ecut if delta_cut is None else delta_cut

    photon_energies = np.geomspace(0.5 * pcut, e_max, n_points)
    electron_energies = np.geomspace(0.5 * ecut, e_max, n_points)
    fractions = epdl.mass_fractions_from_formula(_WATER_FORMULA)

    # -- photons: EPDL, mixed by mass fraction --------------------------------
    channels = epdl.element_photon_channels(epdl_text, elements=set(_WATER_FORMULA))
    mixed = epdl.material_mu_over_rho(channels, fractions, photon_energies)
    mu_over_rho = {process: values[np.newaxis, :] for process, values in mixed.items()}

    # -- coherent form factor: EPDL MF=27, mixed by atom count ----------------
    x_max = RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV * e_max  # max momentum transfer, backscatter
    coherent_x = np.geomspace(x_max * 1.0e-7, x_max, _COHERENT_GRID_POINTS)
    form_factors = epdl.element_coherent_form_factor(epdl_text, elements=set(_WATER_FORMULA))
    form_factor_squared = epdl.material_form_factor_squared(
        form_factors, _WATER_FORMULA, coherent_x
    )
    coherent_cumulative = epdl.coherent_form_factor_cumulative(coherent_x, form_factor_squared)

    # -- elastic scattering power: EEDL, mixed by mass fraction ---------------
    scattering_elements = eedl.element_scattering_power(eedl_text, elements=set(_WATER_FORMULA))
    scattering = eedl.material_scattering_power(scattering_elements, fractions, electron_energies)

    # -- electron stopping ----------------------------------------------------
    # Moller (discrete channel) and CSDA range stay analytic under both strategies; the
    # strategy governs the continuous collision and radiative stopping.
    analytic = AnalyticCrossSections()
    moller = _evaluate(lambda e: analytic.moller_cross_section(e, WATER, cut), electron_energies)
    csda = _evaluate(lambda e: analytic.csda_range(e, WATER), electron_energies)
    if strategy is ElectronStoppingStrategy.EEDL:
        collision_elements = eedl.element_collision_stopping(
            eedl_text, cut, elements=set(_WATER_FORMULA)
        )
        restricted = eedl.material_collision_stopping(
            collision_elements, fractions, electron_energies
        )[np.newaxis, :]
        radiative_elements = eedl.element_radiative_stopping(
            eedl_text, elements=set(_WATER_FORMULA)
        )
        radiative = eedl.material_radiative_stopping(
            radiative_elements, fractions, electron_energies
        )[np.newaxis, :]
        stopping_provenance = "EEDL restricted collision + EEDL radiative"
    else:
        restricted = _evaluate(
            lambda e: analytic.restricted_stopping_power(e, WATER, cut), electron_energies
        )
        radiative = _evaluate(
            lambda e: analytic.radiative_stopping_power(e, WATER), electron_energies
        )
        stopping_provenance = "analytic ICRU-37 on grid"

    return TabulatedData(
        photon_energies=photon_energies,
        mu_over_rho=mu_over_rho,
        electron_energies=electron_energies,
        restricted_stopping=restricted,
        radiative_stopping=radiative,
        moller=moller,
        csda_range=csda,
        scattering_power=scattering[np.newaxis, :],
        delta_cut=cut,
        materials=("water",),
        provenance=(
            f"EPDL2023 photons + coherent form factors + EEDL2023 elastic scattering; "
            f"electron stopping: {strategy.value} ({stopping_provenance}); "
            f"Moller + CSDA analytic; delta_cut={cut} MeV"
        ),
        coherent_x=coherent_x,
        coherent_cumulative=coherent_cumulative[np.newaxis, :],
    )


def _evaluate(quantity: object, energies: np.ndarray) -> np.ndarray:
    """Tabulate a scalar water quantity over ``energies`` as a ``(1, n)`` material row."""
    row = np.array([quantity(float(e)) for e in energies])  # type: ignore[operator]
    return row[np.newaxis, :]
