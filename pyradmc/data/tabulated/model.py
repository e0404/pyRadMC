"""The canonical in-memory compiled cross-section tables.

One provenance-agnostic container the whole tabulated backend agrees on: every
source parser produces it, the precompiler serializes it, and the loader wraps it
behind :class:`~pyradmc.data.interface.CrossSectionSource`. Quantities and units
follow :mod:`pyradmc.data.interface` (mass coefficients in cm^2/g, mass stopping
powers in MeV cm^2/g); both energy grids are ascending, in MeV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TabulatedData"]


@dataclass(frozen=True)
class TabulatedData:
    """Material-resolved interaction data on two shared log-energy grids.

    Rows of every ``(n_materials, n_points)`` array are indexed by the material
    registry (:data:`pyradmc.data.materials.MATERIALS`); ``materials`` records the
    names in that order so a loaded table can be checked against the registry it
    will be used with. Electron quantities that depend on the delta-ray production
    threshold (``restricted_stopping``, ``moller``) are tabulated *at* ``delta_cut``;
    a source built from this data rejects a query at a different cut rather than
    silently returning the wrong restriction.
    """

    photon_energies: np.ndarray  # (n_photon,) MeV
    mu_over_rho: dict[int, np.ndarray]  # PhotonProcess -> (n_materials, n_photon), cm^2/g
    electron_energies: np.ndarray  # (n_electron,) MeV
    restricted_stopping: np.ndarray  # (n_materials, n_electron), MeV cm^2/g, at delta_cut
    radiative_stopping: np.ndarray  # (n_materials, n_electron), MeV cm^2/g
    moller: np.ndarray  # (n_materials, n_electron), cm^2/g, at delta_cut
    csda_range: np.ndarray  # (n_materials, n_electron), g/cm^2
    scattering_power: np.ndarray  # (n_materials, n_electron), rad^2 cm^2/g
    delta_cut: float  # MeV; the cut the restricted quantities were integrated at
    materials: tuple[str, ...]  # material names, aligned with the array rows
    provenance: str  # human-readable source citation, carried for auditability
    # Coherent (Rayleigh) angular sampling: the form-factor cumulative A(x) = int_0^x
    # F^2(x') x' dx' on a shared momentum-transfer grid ``coherent_x`` (inverse angstroms,
    # ascending). Optional: a table compiled without MF=27 form factors leaves both None,
    # and the source falls back to Thomson coherent scattering.
    coherent_x: np.ndarray | None = None  # (n_x,) 1/angstrom
    coherent_cumulative: np.ndarray | None = None  # (n_materials, n_x)
