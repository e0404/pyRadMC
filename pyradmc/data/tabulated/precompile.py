"""Assemble compiled :class:`TabulatedData` from the EPICS source libraries.

The precompiler is the one place the provenance-specific parsers meet: it mixes EPDL
photon cross sections and EEDL elastic scattering into per-material tables on the
canonical geometric grids, fills the electron stopping quantities according to the
chosen strategy, and hands back a :class:`~pyradmc.data.tabulated.model.TabulatedData`
the loader can wrap. It is compile-time only — no kernel ever sees it.

Materials come from the registry (:data:`pyradmc.data.materials.MATERIALS`): each
entry's elemental mass fractions are the mixing key for the per-element EPDL/EEDL
data, and its I-value, Sternheimer coefficients and ESTAR radiative anchors feed the
Berger-Seltzer electron machinery (:mod:`pyradmc.data.berger_seltzer`). Compiled rows
are a registry *prefix* so that material indices in the geometry index the table rows
unchanged; the loader enforces this.

Two data sources are provenance-fixed regardless of strategy: **photons** come from
EPDL (validated sub-percent against NIST XCOM), while the nuclear part of the
**elastic scattering power** comes from EEDL. The condensed atomic-electron part is
the Moliere-screened transport moment below ``delta_cut``; its above-cutoff moment
is excluded because those Moller deflections are transported explicitly. The
strategy governs only the electron **stopping**:

``berger-seltzer`` (default)
    Collision (restricted) from the Berger-Seltzer form with the material's ICRU-37
    I-value and SBS-1984 density effect; radiative from the material's ESTAR-anchor
    fit; both exact against ESTAR (gated sub-0.4 percent per medium in
    ``tests/unit/test_icrp_media.py``). This is the shippable configuration.

``eedl``
    Restricted collision (excitation MT=528 + ionization MT=534-572, integrating the
    differential electro-ionization spectra below ``delta_cut``) and radiative stopping
    from EEDL, mixed per element. A few percent from ESTAR (a genuine library
    difference), so it is the non-default provenance-consistent option, not the
    accuracy default.

The Moller cross-section and CSDA range stay Berger-Seltzer/analytic under both
strategies (the Moller channel is per-electron physics scaled by the material's
electron density; the range compounds the stopping the same way ESTAR does).
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from pyradmc import ECUT_MEV, PCUT_MEV, RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV
from pyradmc.data import berger_seltzer
from pyradmc.data.goudsmit_saunderson import subthreshold_moller_scattering_power
from pyradmc.data.materials import MATERIALS, STANDARD_ATOMIC_WEIGHT, MaterialData
from pyradmc.data.tables import TABLE_POINTS
from pyradmc.data.tabulated import eedl, epdl
from pyradmc.data.tabulated.model import TabulatedData

__all__ = ["ElectronStoppingStrategy", "compile_materials", "compile_water"]

_COHERENT_GRID_POINTS = 512  # momentum-transfer nodes for the coherent form-factor cumulative


class ElectronStoppingStrategy(str, Enum):
    """Which evaluation fills the electron stopping quantities; see the module docstring."""

    BERGER_SELTZER = "berger-seltzer"
    EEDL = "eedl"


def compile_materials(
    epdl_text: str,
    eedl_text: str,
    *,
    n_materials: int | None = None,
    strategy: ElectronStoppingStrategy = ElectronStoppingStrategy.BERGER_SELTZER,
    pcut: float = PCUT_MEV,
    ecut: float = ECUT_MEV,
    e_max: float = 30.0,
    n_points: int = TABLE_POINTS,
    delta_cut: float | None = None,
) -> TabulatedData:
    """Compile the first ``n_materials`` registry materials from EPDL and EEDL text.

    Photons come from ``epdl_text`` and elastic scattering from ``eedl_text``; the
    electron stopping quantities follow ``strategy``. Grids are geometric on
    ``[pcut/2, e_max]`` (photons) and ``[ecut/2, e_max]`` (electrons), matching
    :func:`pyradmc.data.tables.build_cross_section_tables`. ``delta_cut`` defaults to
    ``ecut`` (the delta-ray production threshold the restricted quantities are built
    at). ``n_materials`` defaults to the whole registry; the compiled rows are the
    registry prefix ``MATERIALS[:n_materials]``, aligned with geometry material
    indices.
    """
    if e_max <= max(pcut, ecut):
        raise ValueError(f"e_max={e_max} MeV does not cover the transport range")
    count = len(MATERIALS) if n_materials is None else n_materials
    if not 1 <= count <= len(MATERIALS):
        raise ValueError(f"n_materials={count} outside the registry (1..{len(MATERIALS)})")
    materials = MATERIALS[:count]
    cut = ecut if delta_cut is None else delta_cut

    photon_energies = np.geomspace(0.5 * pcut, e_max, n_points)
    electron_energies = np.geomspace(0.5 * ecut, e_max, n_points)
    elements = {z for material in materials for z, _ in material.composition}

    # -- photons: EPDL, mixed by mass fraction --------------------------------
    channels = epdl.element_photon_channels(epdl_text, elements=elements)
    mixed_rows = [
        epdl.material_mu_over_rho(channels, dict(m.composition), photon_energies) for m in materials
    ]
    processes = {process for row in mixed_rows for process in row}
    mu_over_rho = {
        process: np.stack([row.get(process, np.zeros_like(photon_energies)) for row in mixed_rows])
        for process in processes
    }

    # -- coherent form factor: EPDL MF=27, mixed by atoms per gram ------------
    # Only the F^2 *shape* matters (the magnitude is the MF=23 coherent channel), so
    # any per-material-consistent atom count works; mass fraction over atomic weight
    # is atoms per gram up to Avogadro.
    x_max = RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV * e_max  # max momentum transfer, backscatter
    coherent_x = np.geomspace(x_max * 1.0e-7, x_max, _COHERENT_GRID_POINTS)
    form_factors = epdl.element_coherent_form_factor(epdl_text, elements=elements)
    coherent_cumulative = np.stack(
        [
            epdl.coherent_form_factor_cumulative(
                coherent_x,
                epdl.material_form_factor_squared(form_factors, _atoms_per_gram(m), coherent_x),
            )
            for m in materials
        ]
    )

    # -- Class-II scattering: EEDL nuclear + subthreshold atomic electrons -----
    scattering_elements = eedl.element_scattering_power(eedl_text, elements=elements)
    nuclear_scattering = np.stack(
        [
            eedl.material_scattering_power(
                scattering_elements, dict(m.composition), electron_energies
            )
            for m in materials
        ]
    )
    soft_electron_scattering = np.stack(
        [
            _evaluate(
                lambda e, m=m: subthreshold_moller_scattering_power(m.composition, e, cut),
                electron_energies,
            )
            for m in materials
        ]
    )
    scattering = nuclear_scattering + soft_electron_scattering

    # -- electron stopping ----------------------------------------------------
    # Moller (discrete channel) and CSDA range stay Berger-Seltzer under both
    # strategies; the strategy governs the continuous collision and radiative stopping.
    moller = np.stack(
        [
            _evaluate(
                lambda e, m=m: berger_seltzer.restricted_moller_cross_section(e, m, cut),
                electron_energies,
            )
            for m in materials
        ]
    )
    csda = np.stack([_csda_on_grid(m, electron_energies) for m in materials])
    if strategy is ElectronStoppingStrategy.EEDL:
        collision_elements = eedl.element_collision_stopping(eedl_text, cut, elements=elements)
        restricted = np.stack(
            [
                eedl.material_collision_stopping(
                    collision_elements, dict(m.composition), electron_energies
                )
                for m in materials
            ]
        )
        radiative_elements = eedl.element_radiative_stopping(eedl_text, elements=elements)
        radiative = np.stack(
            [
                eedl.material_radiative_stopping(
                    radiative_elements, dict(m.composition), electron_energies
                )
                for m in materials
            ]
        )
        stopping_provenance = "EEDL restricted collision + EEDL radiative"
    else:
        restricted = np.stack(
            [
                _evaluate(
                    lambda e, m=m: berger_seltzer.restricted_collision_stopping(e, m, cut),
                    electron_energies,
                )
                for m in materials
            ]
        )
        radiative_rows = []
        for m in materials:
            coefficients = berger_seltzer.radiative_fit_coefficients(m.radiative_anchors)
            radiative_rows.append(
                _evaluate(
                    lambda e, c=coefficients: berger_seltzer.radiative_stopping(e, c),
                    electron_energies,
                )
            )
        radiative = np.stack(radiative_rows)
        stopping_provenance = "Berger-Seltzer ICRU-37 per material on grid"

    return TabulatedData(
        photon_energies=photon_energies,
        mu_over_rho=mu_over_rho,
        electron_energies=electron_energies,
        restricted_stopping=restricted,
        radiative_stopping=radiative,
        moller=moller,
        csda_range=csda,
        scattering_power=scattering,
        delta_cut=cut,
        materials=tuple(m.name for m in materials),
        provenance=(
            f"EPDL2023 photons + coherent form factors + EEDL2023 nuclear elastic "
            f"+ restricted Moliere electron scattering; "
            f"electron stopping: {strategy.value} ({stopping_provenance}); "
            f"Moller + CSDA Berger-Seltzer; delta_cut={cut} MeV; "
            f"materials: {', '.join(m.name for m in materials)}"
        ),
        coherent_x=coherent_x,
        coherent_cumulative=coherent_cumulative,
    )


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
    """Compile liquid-water-only :class:`TabulatedData`; see :func:`compile_materials`."""
    return compile_materials(
        epdl_text,
        eedl_text,
        n_materials=1,
        strategy=strategy,
        pcut=pcut,
        ecut=ecut,
        e_max=e_max,
        n_points=n_points,
        delta_cut=delta_cut,
    )


def _atoms_per_gram(material: MaterialData) -> dict[int, float]:
    """Relative atom counts ``w_i / M_i`` (atoms per gram up to Avogadro)."""
    return {z: fraction / STANDARD_ATOMIC_WEIGHT[z] for z, fraction in material.composition}


def _csda_on_grid(material: MaterialData, energies: np.ndarray) -> np.ndarray:
    """Interpolate the material's CSDA range table onto the compile grid."""
    log_energies, ranges = berger_seltzer.csda_range_table(material)
    return np.asarray(np.interp(np.log(energies), log_energies, ranges), dtype=np.float64)


def _evaluate(quantity: object, energies: np.ndarray) -> np.ndarray:
    """Tabulate a scalar quantity over ``energies`` as one material row."""
    return np.array([quantity(float(e)) for e in energies])  # type: ignore[operator]
