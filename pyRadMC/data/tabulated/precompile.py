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
    Would take stopping from EEDL. It is **not implemented**: EEDL's average-energy-loss
    tables give *unrestricted* losses, but ``restricted_stopping`` must exclude delta
    rays above the cut, which needs the differential electro-ionization spectra
    (MF=26/MT=534), a separate slice. Raising here is deliberate — a silently
    unrestricted stopping power would bias the dose.

Water-only for now: the analytic electron backend is water-only (its collision and
range formulas do not carry elemental composition), so multi-material compilation waits
on that (deferred materials task). Photons and scattering already mix arbitrary media.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from pyRadMC import ECUT_MEV, PCUT_MEV
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.data.tables import TABLE_POINTS
from pyRadMC.data.tabulated import eedl, epdl
from pyRadMC.data.tabulated.model import TabulatedData

__all__ = ["ElectronStoppingStrategy", "compile_water"]

_WATER_FORMULA = {1: 2, 8: 1}  # H2O by atom count


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
    if strategy is ElectronStoppingStrategy.EEDL:
        raise NotImplementedError(
            "the 'eedl' stopping strategy needs restricted collision stopping from the "
            "EEDL differential electro-ionization spectra (MF=26/MT=534); not yet built. "
            "Use 'berger-seltzer'."
        )
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

    # -- elastic scattering power: EEDL, mixed by mass fraction ---------------
    scattering_elements = eedl.element_scattering_power(eedl_text, elements=set(_WATER_FORMULA))
    scattering = eedl.material_scattering_power(scattering_elements, fractions, electron_energies)

    # -- electron stopping: analytic ICRU-37 on the grid (berger-seltzer) -----
    analytic = AnalyticCrossSections()
    restricted = _evaluate(
        lambda e: analytic.restricted_stopping_power(e, WATER, cut), electron_energies
    )
    radiative = _evaluate(lambda e: analytic.radiative_stopping_power(e, WATER), electron_energies)
    moller = _evaluate(lambda e: analytic.moller_cross_section(e, WATER, cut), electron_energies)
    csda = _evaluate(lambda e: analytic.csda_range(e, WATER), electron_energies)

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
            f"EPDL2023 photons + EEDL2023 elastic scattering; electron stopping: "
            f"{strategy.value} (analytic ICRU-37 on grid); delta_cut={cut} MeV"
        ),
    )


def _evaluate(quantity: object, energies: np.ndarray) -> np.ndarray:
    """Tabulate a scalar water quantity over ``energies`` as a ``(1, n)`` material row."""
    row = np.array([quantity(float(e)) for e in energies])  # type: ignore[operator]
    return row[np.newaxis, :]
