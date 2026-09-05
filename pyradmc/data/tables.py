"""Flattened cross-section tables and their kernel-side lookups.

``CrossSectionSource.build_tables`` flattens the host-side query API onto log-energy
grids so that kernels can interpolate instead of calling Python (AGENTS.md 2.6: this
module *is* the kernel-side face of the data interface — transport code never sees a
formula). Two grids are built, both with ``n_points`` nodes:

photon grid
    ``[PCUT/2, e_max]``: every transported photon energy, with margin below the cut.
electron grid
    ``[ECUT/2, e_max]``: every kinetic energy the electron loop queries, with margin.

Values are interpolated **linearly in the value on the log-energy grid**. At the
default 2048 points over these spans the local spacing is ~3e-3 in log energy and the
interpolation error is second order in it — a few 1e-5 relative for the smooth
channels, pinned in ``tests/unit/test_tables.py``. (This per-energy lookup is distinct
from the *spectrum-integration* requirement of AGENTS.md 2.7, which binds the
tabulated ``CrossSectionSource`` itself.)

The Woodcock majorant nodes carry a relative headroom above the float64 maximum so
that the kernel-side inequality ``rho * sum(channel lookups) <= majorant lookup``
survives float32 rounding everywhere. Enlarging a Woodcock majorant is *exact* — it
only adds delta scattering (Woodcock et al. (1965), ANL-7050) — whereas the reverse
error is a silent under-attenuation bias, so the headroom errs on the only safe side.

The lookup functions are single-source (AGENTS.md 2.5): pure scalar functions over
the :mod:`pyradmc.data.handles` aliases, running under NumPy on the host and compiled
to ``@wp.func`` by the Warp physics loader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pyradmc import RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV
from pyradmc.data.handles import Table1D, Table2D
from pyradmc.data.interface import PhotonProcess

if TYPE_CHECKING:
    from pyradmc.data.interface import CrossSectionSource

__all__ = [
    "MAJORANT_HEADROOM",
    "TABLE_POINTS",
    "CrossSectionTables",
    "build_cross_section_tables",
    "lookup_loglinear_1d",
    "lookup_loglinear_2d",
]

TABLE_POINTS: int = 2048
"""Default nodes per grid. Sized so interpolation error sits far below the 2e-4
parity budget of the table tests; memory is trivial at any plausible material count."""

COHERENT_POINTS: int = 512
"""Nodes on the coherent form-factor momentum-transfer grid; the cumulative is smooth
and the sampler inverts it, so far fewer nodes than the energy grids suffice."""

_RANGE_R_MIN_FRACTION: float = 1.0e-6
"""Lowest node of the shared log-range grid as a fraction of the largest
material's full range: below it the inverse lookup clamps to an energy within
interpolation error of the cutoff."""

_COHERENT_X_MIN_FRACTION: float = 1.0e-7
"""Lowest momentum-transfer node as a fraction of the maximum: small enough that the
neglected coherent mass below it (~F(0)^2 x_min^2 / 2) is negligible."""

MAJORANT_HEADROOM: float = 1.0 + 1.0e-6
"""Relative headroom on the majorant nodes: ~8 float32 ulps above the float64
maximum, so the kernel-side Woodcock inequality cannot be broken by rounding. Costs
one delta-scattering event per ~1e6 flights; exactness is unaffected (see module
docstring)."""


# -- kernel-side lookups (single-source; compiled to @wp.func for Warp) ----------


def lookup_loglinear_1d(
    values: Table1D, log_e_min: float, inv_dlog: float, n_points: int, energy: float
) -> float:
    """Interpolate one gridded quantity at ``energy``, linear in value over log E.

    Clamps flat outside the grid: linear extrapolation of a steep channel (the
    E^-3 photoelectric) would go negative below the grid, and no transported
    particle is ever outside it by construction — the clamp only guards float
    round-off at the edges.

    Parameters
    ----------
    values
        Grid node values, shape ``(n_points,)``.
    log_e_min, inv_dlog, n_points
        Grid metadata from :class:`CrossSectionTables`: log of the lowest node
        energy, inverse log-spacing, node count.
    energy
        Query energy in MeV; must be positive.
    """
    t = (math.log(energy) - log_e_min) * inv_dlog
    t = min(max(t, 0.0), float(n_points - 1))
    i = min(int(t), n_points - 2)
    w = t - float(i)
    return float(values[i]) * (1.0 - w) + float(values[i + 1]) * w


def lookup_loglinear_2d(
    values: Table2D,
    material: int,
    log_e_min: float,
    inv_dlog: float,
    n_points: int,
    energy: float,
) -> float:
    """Interpolate one per-material quantity; see :func:`lookup_loglinear_1d`.

    Parameters
    ----------
    values
        Grid node values, shape ``(n_materials, n_points)``.
    material
        Material row index.
    log_e_min, inv_dlog, n_points, energy
        As in :func:`lookup_loglinear_1d`.
    """
    t = (math.log(energy) - log_e_min) * inv_dlog
    t = min(max(t, 0.0), float(n_points - 1))
    i = min(int(t), n_points - 2)
    w = t - float(i)
    return float(values[material, i]) * (1.0 - w) + float(values[material, i + 1]) * w


# -- host-side container and builder ---------------------------------------------


@dataclass(frozen=True)
class CrossSectionTables:
    """Flattened interaction data on log-energy grids; frozen after construction.

    Arrays are float64 NumPy on the host; backends cast on upload (the Warp backend
    to float32). Units follow :mod:`pyradmc.data.interface`: mass coefficients in
    cm^2/g, stopping powers in MeV cm^2/g, ranges in g/cm^2, scattering power in
    rad^2 cm^2/g, the majorant macroscopic in 1/cm.

    Attributes
    ----------
    n_points
        Nodes per grid.
    ecut, pcut, e_max
        The cutoffs (MeV) the electron tables were built at and the upper grid
        edge; provenance so a mismatched engine configuration is detectable.
    photon_log_e_min, photon_inv_dlog
        Photon grid metadata for the lookup functions.
    electron_log_e_min, electron_inv_dlog
        Electron grid metadata.
    mu_compton, mu_photo, mu_pair, mu_rayleigh
        Per-channel mass attenuation coefficients, ``(n_materials, n_points)``.
        All four :class:`~pyradmc.data.interface.PhotonProcess` channels are always
        present — kernels are channel-complete (AGENTS.md 2.10); a disabled channel
        is an all-zero row, a *data* statement, not a flag.
    majorant
        Woodcock majorant with float32 headroom, ``(n_points,)``.
    stopping_restricted, stopping_radiative, moller, csda_range, scattering_power
        Electron-grid quantities, ``(n_materials, n_points)``; the restricted
        stopping power and Moller cross-section are evaluated at ``ecut``.
    log_eta
        ``ln`` of the Moliere elastic screening parameter
        (:meth:`~pyradmc.data.interface.CrossSectionSource.elastic_screening`)
        on the electron grid, ``(n_materials, n_points)`` — the kernel-side key
        of the Goudsmit-Saunderson deflection grid. Stored as the logarithm
        because that is the variable the GS grid is binned in (interpolation
        happens in the nearly-linear quantity, and the kernel skips a ``log``
        per hinge), and because the log-linear lookup then makes the row
        extremes an *exact* bound on every reachable key (see
        :func:`~pyradmc.data.goudsmit_saunderson.gs_window_from_tables`).
    restricted_range
        Restricted-collision range to ``ecut`` in g/cm^2 on the electron grid,
        ``(n_materials, n_points)`` — the exact-energy-loss substep's forward map
        (:meth:`~pyradmc.data.interface.CrossSectionSource.restricted_range`).
    energy_of_restricted_range
        The inverse map on a *shared* log-range grid, ``(n_materials, n_points)``:
        the energy whose restricted range equals the node, clamped into each
        material's own ``[ecut, e_max]``. ``range_log_r_min``/``range_inv_dlog``
        are its lookup metadata (same log-linear scheme as the energy grids).
    """

    n_points: int
    ecut: float
    pcut: float
    e_max: float
    photon_log_e_min: float
    photon_inv_dlog: float
    electron_log_e_min: float
    electron_inv_dlog: float
    mu_compton: npt.NDArray[np.float64]
    mu_photo: npt.NDArray[np.float64]
    mu_pair: npt.NDArray[np.float64]
    mu_rayleigh: npt.NDArray[np.float64]
    majorant: npt.NDArray[np.float64]
    stopping_restricted: npt.NDArray[np.float64]
    stopping_radiative: npt.NDArray[np.float64]
    moller: npt.NDArray[np.float64]
    csda_range: npt.NDArray[np.float64]
    scattering_power: npt.NDArray[np.float64]
    log_eta: npt.NDArray[np.float64]
    restricted_range: npt.NDArray[np.float64]
    energy_of_restricted_range: npt.NDArray[np.float64]
    range_log_r_min: float
    range_inv_dlog: float
    # Coherent form-factor cumulative A(x) for angular sampling: abscissae ``coherent_x``
    # (momentum transfer, 1/angstrom, ascending) and ``coherent_cumulative`` per material.
    # The default source yields the flat (Thomson) cumulative; the tabulated one its EPDL
    # form factor. Inverted by the coherent sampler with bisection, so no log-grid metadata.
    coherent_x: npt.NDArray[np.float64]
    coherent_cumulative: npt.NDArray[np.float64]
    n_coherent: int


def build_cross_section_tables(
    source: CrossSectionSource,
    ecut: float,
    pcut: float,
    e_max: float,
    n_points: int = TABLE_POINTS,
) -> CrossSectionTables:
    """Flatten a :class:`CrossSectionSource` onto the two lookup grids.

    Generic over sources: everything goes through the host query API, so the
    analytic and tabulated backends flatten identically.

    Parameters
    ----------
    source
        The host-side data source to flatten.
    ecut, pcut
        Electron and photon cutoffs in MeV (accuracy-defining, AGENTS.md 2.8);
        the electron tables are built *at* this ``ecut``.
    e_max
        Upper grid edge in MeV; must cover the highest primary energy. No
        transported particle can exceed it (secondaries only lose energy).
    n_points
        Nodes per grid.
    """
    if pcut <= 0.0 or ecut <= 0.0:
        raise ValueError(f"non-positive cutoff: pcut={pcut}, ecut={ecut}")
    if e_max <= max(pcut, ecut):
        raise ValueError(f"e_max={e_max} MeV does not cover the transport range")
    if n_points < 2:
        raise ValueError(f"need at least two grid nodes, got {n_points}")

    n_materials = source.n_materials
    photon_energies = np.geomspace(0.5 * pcut, e_max, n_points)
    electron_energies = np.geomspace(0.5 * ecut, e_max, n_points)

    channels = {
        process: np.empty((n_materials, n_points))
        for process in (
            PhotonProcess.COMPTON,
            PhotonProcess.PHOTOELECTRIC,
            PhotonProcess.PAIR,
            PhotonProcess.RAYLEIGH,
        )
    }
    for material in range(n_materials):
        for j, energy in enumerate(photon_energies):
            for process, table in channels.items():
                table[material, j] = source.mu_over_rho(float(energy), material, process)

    majorant = np.array([source.majorant(float(e)) for e in photon_energies]) * MAJORANT_HEADROOM

    electron_tables = {
        name: np.empty((n_materials, n_points))
        for name in (
            "stopping_restricted",
            "stopping_radiative",
            "moller",
            "csda_range",
            "scattering_power",
        )
    }
    log_eta = np.empty((n_materials, n_points))
    for material in range(n_materials):
        for j, energy in enumerate(electron_energies):
            e = float(energy)
            electron_tables["stopping_restricted"][material, j] = source.restricted_stopping_power(
                e, material, ecut
            )
            electron_tables["stopping_radiative"][material, j] = source.radiative_stopping_power(
                e, material
            )
            electron_tables["moller"][material, j] = source.moller_cross_section(e, material, ecut)
            electron_tables["csda_range"][material, j] = source.csda_range(e, material)
            electron_tables["scattering_power"][material, j] = source.scattering_power(
                e, material, ecut
            )
            log_eta[material, j] = math.log(source.elastic_screening(e, material))

    # Restricted-range forward map on the electron grid, plus its inverse on one
    # shared log-range grid (a common axis keeps the kernel lookup's scalar
    # metadata; per-material values clamp into their own [ecut, e_max], which the
    # interface's energy_after_mass_path already guarantees).
    restricted_range = np.empty((n_materials, n_points))
    for material in range(n_materials):
        for j, energy in enumerate(electron_energies):
            restricted_range[material, j] = source.restricted_range(float(energy), material, ecut)
    e_top = float(electron_energies[-1])
    r_top = max(float(restricted_range[m, -1]) for m in range(n_materials))
    range_nodes = np.geomspace(r_top * _RANGE_R_MIN_FRACTION, r_top, n_points)
    energy_of_restricted_range = np.empty((n_materials, n_points))
    for material in range(n_materials):
        r_top_mat = float(restricted_range[material, -1])
        for j, r in enumerate(range_nodes):
            energy_of_restricted_range[material, j] = source.energy_after_mass_path(
                e_top, material, ecut, r_top_mat - float(r)
            )

    def grid_metadata(energies: npt.NDArray[np.float64]) -> tuple[float, float]:
        log_min = float(np.log(energies[0]))
        log_max = float(np.log(energies[-1]))
        return log_min, (n_points - 1) / (log_max - log_min)

    photon_log_e_min, photon_inv_dlog = grid_metadata(photon_energies)
    electron_log_e_min, electron_inv_dlog = grid_metadata(electron_energies)
    range_log_r_min = float(np.log(range_nodes[0]))
    range_inv_dlog = (n_points - 1) / float(np.log(range_nodes[-1]) - np.log(range_nodes[0]))

    # Coherent form-factor cumulative: the momentum-transfer grid spans up to the maximum
    # transfer at e_max (backscatter), queried per material through the source so the flat
    # (Thomson) default and the tabulated form factor flatten the same way.
    x_max = RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV * e_max
    coherent_x = np.geomspace(x_max * _COHERENT_X_MIN_FRACTION, x_max, COHERENT_POINTS)
    coherent_cumulative = np.stack(
        [source.coherent_cumulative(coherent_x, material) for material in range(n_materials)]
    )

    return CrossSectionTables(
        n_points=n_points,
        ecut=ecut,
        pcut=pcut,
        e_max=e_max,
        photon_log_e_min=photon_log_e_min,
        photon_inv_dlog=photon_inv_dlog,
        electron_log_e_min=electron_log_e_min,
        electron_inv_dlog=electron_inv_dlog,
        mu_compton=channels[PhotonProcess.COMPTON],
        mu_photo=channels[PhotonProcess.PHOTOELECTRIC],
        mu_pair=channels[PhotonProcess.PAIR],
        mu_rayleigh=channels[PhotonProcess.RAYLEIGH],
        majorant=majorant,
        log_eta=log_eta,
        restricted_range=restricted_range,
        energy_of_restricted_range=energy_of_restricted_range,
        range_log_r_min=range_log_r_min,
        range_inv_dlog=range_inv_dlog,
        coherent_x=coherent_x,
        coherent_cumulative=coherent_cumulative,
        n_coherent=COHERENT_POINTS,
        **electron_tables,
    )
