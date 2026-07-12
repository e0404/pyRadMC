"""EPDL (Evaluated Photon Data Library) -> canonical photon cross sections.

Turns the photoatomic MF=23 cross sections of an EPDL ENDF-6 file into the mass
attenuation coefficients the tabulated backend consumes: per element, then mixed by
mass fraction into a material. This is the photon half of the tabulated compile; the
electron half (EEDL) and the full :class:`~pyRadMC.data.tabulated.model.TabulatedData`
assembly land in later slices, so this module deliberately stops at per-material
``{process -> mu/rho on a query grid}``.

EPDL facts this relies on (confirmed against ``EPDL2023.ALL``): sections are keyed by
``MAT = 100 * Z``; energies are in eV and cross sections in barns; the photon channels
we transport are the incoherent (Compton), total photoionization (photoelectric),
total pair, and coherent (Rayleigh) reactions. The reaction cross sections are smooth
and monotone over the transported MeV range (all atomic edges sit in the keV subshell
channels we do not read), so resampling by log-log interpolation between the dense
tabulated points is accurate to far better than the tabulation itself; the ENDF
interpolation law flagged per region is not separately honoured (a stated
approximation, valid only because no edge falls in ``[PCUT/2, e_max]`` for these
channels).

Units out: energies in MeV, mass attenuation coefficients ``mu/rho`` in cm^2/g.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from pyRadMC.data.interface import PhotonProcess
from pyRadMC.data.materials import AVOGADRO
from pyRadMC.data.tabulated.endf import read_tab1_by_mf

__all__ = [
    "MT_TO_PROCESS",
    "STANDARD_ATOMIC_WEIGHT",
    "element_photon_channels",
    "mass_fractions_from_formula",
    "material_mu_over_rho",
]

# EPDL MF=23 reaction (MT) -> transported photon channel.
#   504 incoherent scattering        -> Compton
#   522 total photoionization        -> photoelectric
#   516 total pair production         -> pair (nuclear + electron/triplet)
#   502 coherent scattering           -> Rayleigh
# Subshell photoionization (534-572) and the pair subchannels (515/517) are the
# totals' constituents and are intentionally not read.
MT_TO_PROCESS: dict[int, int] = {
    504: PhotonProcess.COMPTON,
    522: PhotonProcess.PHOTOELECTRIC,
    516: PhotonProcess.PAIR,
    502: PhotonProcess.RAYLEIGH,
}

# Standard (conventional) atomic weights, g/mol; IUPAC 2021 (Prohaska et al.,
# doi:10.1515/pac-2019-0603). Only the light elements of the ICRU-44 reference media
# are listed; the general element/composition home is MaterialData (deferred task b),
# which will supersede this table. Water needs H and O.
STANDARD_ATOMIC_WEIGHT: dict[int, float] = {
    1: 1.008,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    11: 22.98976928,
    12: 24.305,
    15: 30.973761998,
    16: 32.06,
    17: 35.45,
    18: 39.95,
    19: 39.0983,
    20: 40.078,
}


def element_photon_channels(
    text: str, elements: Iterable[int] | None = None
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Parse an EPDL file into per-element mass attenuation coefficients.

    Returns ``{Z: {process: (energy_MeV, mu_over_rho_cm2g)}}`` for the transported
    channels, with energies ascending. The barns-to-cm^2/g conversion is
    ``mu/rho = sigma[barn] * 1e-24 * N_A / M(Z)`` — the microscopic cross section
    times the number of atoms per gram.

    ``elements`` restricts extraction to those atomic numbers; a requested element
    absent from the file or from :data:`STANDARD_ATOMIC_WEIGHT` is an error (the caller
    asked for something that cannot be built). With ``elements=None`` every element in
    the file is returned, silently skipping any whose atomic weight is not tabulated —
    the full ``EPDL2023.ALL`` spans Z=1..100 and only the media constituents are
    weighted here.
    """
    want = None if elements is None else set(elements)
    sections = read_tab1_by_mf(text, 23)
    out: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for (mat, mt), (energy_ev, sigma_barns) in sections.items():
        process = MT_TO_PROCESS.get(mt)
        if process is None:
            continue
        z = mat // 100
        if want is not None and z not in want:
            continue
        molar_mass = STANDARD_ATOMIC_WEIGHT.get(z)
        if molar_mass is None:
            if want is None:
                continue  # unweighable and unrequested: skip it
            raise KeyError(f"no atomic weight for Z={z}; extend STANDARD_ATOMIC_WEIGHT")
        energy_mev = np.asarray(energy_ev, dtype=np.float64) * 1.0e-6
        mu_over_rho = np.asarray(sigma_barns, dtype=np.float64) * 1.0e-24 * AVOGADRO / molar_mass
        out.setdefault(z, {})[process] = (energy_mev, mu_over_rho)
    if want is not None and want - set(out):
        raise KeyError(f"elements {sorted(want - set(out))} not found in the EPDL text")
    return out


def mass_fractions_from_formula(formula: Mapping[int, float]) -> dict[int, float]:
    """Convert a stoichiometric formula ``{Z: atom_count}`` to mass fractions.

    ``w_i = n_i M_i / sum_j n_j M_j``, using :data:`STANDARD_ATOMIC_WEIGHT`; the
    returned fractions sum to one. (Water is ``{1: 2, 8: 1}``.)
    """
    masses = {z: count * STANDARD_ATOMIC_WEIGHT[z] for z, count in formula.items()}
    total = sum(masses.values())
    if total <= 0.0:
        raise ValueError(f"empty or non-positive formula {dict(formula)}")
    return {z: mass / total for z, mass in masses.items()}


def material_mu_over_rho(
    elements: Mapping[int, Mapping[int, tuple[np.ndarray, np.ndarray]]],
    mass_fractions: Mapping[int, float],
    grid_mev: np.ndarray,
) -> dict[int, np.ndarray]:
    """Mix per-element channels into a material on ``grid_mev`` by mass fraction.

    ``(mu/rho)_material(E) = sum_i w_i (mu/rho)_i(E)``, each element resampled onto
    ``grid_mev`` by log-log interpolation (:func:`_resample_loglog`). Returns
    ``{process: mu_over_rho}`` for every channel any constituent provides; an element
    missing a channel contributes zero to it.
    """
    grid = np.asarray(grid_mev, dtype=np.float64)
    processes = {p for z in mass_fractions for p in elements.get(z, {})}
    mixed: dict[int, np.ndarray] = {p: np.zeros_like(grid) for p in processes}
    for z, weight in mass_fractions.items():
        channels = elements.get(z, {})
        for process, (energy_mev, mu_over_rho) in channels.items():
            mixed[process] += weight * _resample_loglog(energy_mev, mu_over_rho, grid)
    return mixed


def _resample_loglog(
    energy_src: np.ndarray, mu_src: np.ndarray, energy_dst: np.ndarray
) -> np.ndarray:
    """Interpolate ``mu_src(energy_src)`` onto ``energy_dst``, linear in log-log.

    Only the strictly-positive tabulated points define the interpolant; query points
    below the lowest positive energy return zero (this is what makes a pair-production
    channel vanish below its threshold rather than extrapolate a spurious tail).
    Outside the positive support on the high side the value is held flat, which the
    transported range never reaches for these channels.
    """
    positive = mu_src > 0.0
    out = np.zeros_like(energy_dst, dtype=np.float64)
    if int(positive.sum()) < 2:
        return out
    log_e_src = np.log(energy_src[positive])
    log_mu_src = np.log(mu_src[positive])
    log_mu_dst = np.interp(np.log(energy_dst), log_e_src, log_mu_src)
    out = np.asarray(np.exp(log_mu_dst), dtype=np.float64)
    out[energy_dst < energy_src[positive][0]] = 0.0
    return out
