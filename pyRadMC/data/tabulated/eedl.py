"""EEDL (Evaluated Electron Data Library) -> radiative stopping power.

The electron counterpart of :mod:`pyRadMC.data.tabulated.epdl`. It builds the mass
radiative (bremsstrahlung) stopping power from two EEDL reactions: the bremsstrahlung
cross section ``sigma`` (MF=23/MT=527, barns) and the average energy the electron loses
per bremsstrahlung event ``<E_loss>`` (the LAW=8 energy-transfer subsection of
MF=26/MT=527, eV). Their product, per atom and per gram, is the radiative stopping
power::

    S_rad/rho (E) = (N_A / M(Z)) * sigma(E) * <E_loss>(E)   [MeV cm^2/g]

which is the definition ``S_rad = N * integral k (d(sigma)/dk) dk`` written with the
tabulated mean loss instead of the spectrum integral.

This is the ``eedl`` electron-stopping strategy's radiative source. The default
``berger-seltzer`` strategy takes radiative stopping from the analytic ESTAR fit
instead (exact against NIST ESTAR / ICRU-37, which EEDL's bremsstrahlung evaluation
departs from by a few percent in the MeV range); the choice is the precompiler's, and
recorded in the compiled provenance. Units out: energies in MeV, stopping power in
MeV cm^2/g.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from pyRadMC.data.materials import AVOGADRO
from pyRadMC.data.tabulated.endf import (
    read_mf26_angular_distributions,
    read_mf26_energy_transfer,
    read_tab1_by_mf,
)
from pyRadMC.data.tabulated.epdl import STANDARD_ATOMIC_WEIGHT

__all__ = [
    "element_radiative_stopping",
    "element_scattering_power",
    "material_radiative_stopping",
    "material_scattering_power",
]

_MT_BREMSSTRAHLUNG = 527
_MT_ELASTIC_LARGE_ANGLE = 525


def element_radiative_stopping(
    text: str, elements: Iterable[int] | None = None
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-element mass radiative stopping power from EEDL bremsstrahlung.

    Returns ``{Z: (energy_MeV, S_rad_MeV_cm2_g)}`` on the cross section's energy grid,
    with ``<E_loss>`` interpolated onto it (log-log). ``elements`` restricts extraction
    the way :func:`pyRadMC.data.tabulated.epdl.element_photon_channels` does: a
    requested element absent from the file or unweighted is an error; ``None`` returns
    every element with a tabulated atomic weight, skipping the rest.
    """
    want = None if elements is None else set(elements)
    cross_sections = read_tab1_by_mf(text, 23)
    energy_loss = read_mf26_energy_transfer(text, _MT_BREMSSTRAHLUNG)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for (mat, mt), (energy_ev, sigma_barns) in cross_sections.items():
        if mt != _MT_BREMSSTRAHLUNG or mat not in energy_loss:
            continue
        z = mat // 100
        if want is not None and z not in want:
            continue
        molar_mass = STANDARD_ATOMIC_WEIGHT.get(z)
        if molar_mass is None:
            if want is None:
                continue
            raise KeyError(f"no atomic weight for Z={z}; extend STANDARD_ATOMIC_WEIGHT")
        loss_energy_ev, loss_ev = energy_loss[mat]
        mean_loss_mev = _interp_loglog(energy_ev, loss_energy_ev, loss_ev) * 1.0e-6
        s_rad = sigma_barns * 1.0e-24 * AVOGADRO / molar_mass * mean_loss_mev
        out[z] = (np.asarray(energy_ev, dtype=np.float64) * 1.0e-6, s_rad)
    if want is not None and want - set(out):
        raise KeyError(f"elements {sorted(want - set(out))} not found in the EEDL text")
    return out


def material_radiative_stopping(
    elements: Mapping[int, tuple[np.ndarray, np.ndarray]],
    mass_fractions: Mapping[int, float],
    grid_mev: np.ndarray,
) -> np.ndarray:
    """Mix per-element radiative stopping into a material on ``grid_mev`` by mass.

    ``S_rad_material(E) = sum_i w_i S_rad_i(E)`` (Bragg additivity), each element
    resampled onto ``grid_mev`` by log-log interpolation.
    """
    grid = np.asarray(grid_mev, dtype=np.float64)
    total = np.zeros_like(grid)
    for z, weight in mass_fractions.items():
        energy_mev, s_rad = elements[z]
        total += weight * _interp_loglog(grid, energy_mev, s_rad)
    return total


def element_scattering_power(
    text: str, elements: Iterable[int] | None = None
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    r"""Per-element mass angular scattering power from EEDL elastic scattering.

    Returns ``{Z: (energy_MeV, T_rad2_cm2_g)}`` on the (dense) large-angle elastic cross
    section's energy grid. The scattering power is the mean-square-angle rate

    .. math:: T/\rho = (N_A / M)\, \sigma_{525}(E)\, \langle \theta^2 \rangle,
        \qquad \langle \theta^2 \rangle = 2\,\langle 1 - \mu \rangle,

    i.e. twice the first transport moment of the tabulated large-angle distribution
    (MF=26/MT=525) times the large-angle elastic cross section (MF=23/MT=525). The
    smoothly varying moment ``<1-mu>`` is evaluated at EEDL's sparse angular energies
    and interpolated (log-log) onto the dense cross section grid, so the cross section's
    energy structure is preserved through the sparse-shape gap.

    **Stated approximation:** the forward screened-Rutherford Coulomb tail beyond the
    last tabulated point (mu = 0.999999) is neglected. Being strongly forward peaked it
    is suppressed by the ``1 - mu`` weight; its share of the transport moment for water
    is under 1% below 1 MeV, ~5% at 10 MeV and ~11% at 20 MeV, so this underestimates
    the scattering power by at most that much at the top of the transported range.
    Restoring it (Seltzer screening, ICRU-35) is a refinement for penumbra accuracy.

    ``elements`` behaves as in :func:`element_radiative_stopping`.
    """
    want = None if elements is None else set(elements)
    cross_sections = read_tab1_by_mf(text, 23)
    distributions = read_mf26_angular_distributions(text, _MT_ELASTIC_LARGE_ANGLE)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for (mat, mt), (energy_ev, sigma_barns) in cross_sections.items():
        if mt != _MT_ELASTIC_LARGE_ANGLE or mat not in distributions:
            continue
        z = mat // 100
        if want is not None and z not in want:
            continue
        molar_mass = STANDARD_ATOMIC_WEIGHT.get(z)
        if molar_mass is None:
            if want is None:
                continue
            raise KeyError(f"no atomic weight for Z={z}; extend STANDARD_ATOMIC_WEIGHT")
        shape_energy_ev = np.array([e for e, _, _ in distributions[mat]])
        mean_1mu = np.array([_mean_one_minus_mu(mu, p) for _, mu, p in distributions[mat]])
        mean_on_grid = _interp_loglog(energy_ev, shape_energy_ev, mean_1mu)
        scattering = 2.0 * AVOGADRO / molar_mass * sigma_barns * 1.0e-24 * mean_on_grid
        out[z] = (np.asarray(energy_ev, dtype=np.float64) * 1.0e-6, scattering)
    if want is not None and want - set(out):
        raise KeyError(f"elements {sorted(want - set(out))} not found in the EEDL text")
    return out


def material_scattering_power(
    elements: Mapping[int, tuple[np.ndarray, np.ndarray]],
    mass_fractions: Mapping[int, float],
    grid_mev: np.ndarray,
) -> np.ndarray:
    """Mix per-element scattering power into a material on ``grid_mev`` by mass.

    ``T_material(E) = sum_i w_i T_i(E)`` (Bragg additivity), each element resampled
    onto ``grid_mev`` by log-log interpolation.
    """
    grid = np.asarray(grid_mev, dtype=np.float64)
    total = np.zeros_like(grid)
    for z, weight in mass_fractions.items():
        energy_mev, scattering = elements[z]
        total += weight * _interp_loglog(grid, energy_mev, scattering)
    return total


def _mean_one_minus_mu(mu: np.ndarray, probability: np.ndarray) -> float:
    """First transport moment ``<1 - mu>`` of a tabulated angular distribution."""
    weight = np.trapezoid(probability, mu)
    return float(np.trapezoid((1.0 - mu) * probability, mu) / weight)


def _interp_loglog(
    energy_dst: np.ndarray, energy_src: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Interpolate a strictly-positive quantity, linear in log-log; flat outside.

    Radiative stopping and the mean energy loss are positive across the tabulated
    range with no threshold, so (unlike the photon channels) no zero-support handling
    is needed — the log-log form is exact at the shared grid points a unit test pins.
    """
    log_values = np.interp(np.log(energy_dst), np.log(energy_src), np.log(values))
    return np.asarray(np.exp(log_values), dtype=np.float64)
