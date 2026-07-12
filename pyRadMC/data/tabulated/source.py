"""The tabulated cross-section source: a :class:`CrossSectionSource` over compiled data.

Wraps a provenance-agnostic :class:`~pyRadMC.data.tabulated.model.TabulatedData` and
answers the host query API by log-linear interpolation on its grids (the same basis
:mod:`pyRadMC.data.tables` uses, so the flattened runtime tables are consistent).
Both energy grids must be geometric (uniform in log energy); the precompiler emits
them that way.

The electron quantities that depend on the delta-ray cut (restricted stopping,
Moller) are tabulated at one ``delta_cut``: a query at a different cut raises rather
than returning a silently wrong restriction, so tables must be compiled at the run's
ECUT.
"""

from __future__ import annotations

import math

import numpy as np

from pyRadMC.data.interface import CrossSectionSource, PhotonProcess
from pyRadMC.data.materials import MATERIALS, WATER
from pyRadMC.data.tables import lookup_loglinear_2d
from pyRadMC.data.tabulated.model import TabulatedData

__all__ = ["TabulatedCrossSections"]

_PHOTON_PROCESSES = (
    PhotonProcess.COMPTON,
    PhotonProcess.PHOTOELECTRIC,
    PhotonProcess.PAIR,
    PhotonProcess.RAYLEIGH,
)

# Electron quantity name -> TabulatedData attribute, for integrate_product and queries.
_ELECTRON_TABLES = (
    "restricted_stopping",
    "radiative_stopping",
    "moller",
    "csda_range",
    "scattering_power",
)


def _grid_metadata(energies: np.ndarray) -> tuple[float, float, int]:
    """``(log_e_min, inv_dlog, n_points)`` for a geometric grid; validates uniformity."""
    n = int(energies.shape[0])
    if n < 2:
        raise ValueError("energy grid needs at least two nodes")
    log_e = np.log(energies)
    dlog = np.diff(log_e)
    if not np.allclose(dlog, dlog[0], rtol=1e-6):
        raise ValueError("energy grid is not geometric (uniform in log energy)")
    return float(log_e[0]), float((n - 1) / (log_e[-1] - log_e[0])), n


class TabulatedCrossSections(CrossSectionSource):
    """Interpolating source over compiled :class:`TabulatedData`.

    ``geometry_densities`` are the ``(material, maximum mass density)`` pairs present
    in the geometry, exactly as for the analytic source; the Woodcock majorant is
    taken over them.
    """

    def __init__(
        self,
        data: TabulatedData,
        geometry_densities: tuple[tuple[int, float], ...] = ((WATER, 1.0),),
    ) -> None:
        registry = tuple(m.name for m in MATERIALS)
        if data.materials != registry[: len(data.materials)]:
            raise ValueError(
                f"compiled materials {data.materials} do not match the registry {registry}"
            )
        for material, density in geometry_densities:
            if not 0 <= material < len(data.materials):
                raise ValueError(f"material index {material} not in the compiled table")
            if density <= 0.0:
                raise ValueError(f"non-positive density {density} for material {material}")

        self._data = data
        self._geometry_densities = tuple(geometry_densities)
        self._photon_energies = np.asarray(data.photon_energies, dtype=np.float64)
        self._electron_energies = np.asarray(data.electron_energies, dtype=np.float64)
        self._log_electron = np.log(self._electron_energies)
        self._p_log_min, self._p_inv_dlog, self._p_n = _grid_metadata(self._photon_energies)
        self._e_log_min, self._e_inv_dlog, self._e_n = _grid_metadata(self._electron_energies)
        # float64 views for the scalar lookups (which index float arrays).
        self._mu = {p: np.asarray(a, dtype=np.float64) for p, a in data.mu_over_rho.items()}
        self._electron = {
            name: np.asarray(getattr(data, name), dtype=np.float64) for name in _ELECTRON_TABLES
        }

    # -- photons ------------------------------------------------------------

    def mu_over_rho(self, energy: float, material: int, process: int) -> float:
        """Mass attenuation coefficient for one channel, in cm^2/g."""
        if energy <= 0.0:
            raise ValueError(f"non-positive photon energy {energy} MeV")
        table = self._mu.get(process)
        if table is None:
            return 0.0  # a channel absent from the compiled data is disabled
        return lookup_loglinear_2d(
            table, material, self._p_log_min, self._p_inv_dlog, self._p_n, energy
        )

    def mu_over_rho_total(self, energy: float, material: int) -> float:
        """Total mass attenuation coefficient: the sum over compiled channels."""
        return sum(self.mu_over_rho(energy, material, p) for p in self._mu)

    def majorant(self, energy: float) -> float:
        """Woodcock majorant over the declared geometry contents, in 1/cm."""
        return max(
            density * self.mu_over_rho_total(energy, material)
            for material, density in self._geometry_densities
        )

    # -- electrons ----------------------------------------------------------

    def restricted_stopping_power(self, energy: float, material: int, delta_cut: float) -> float:
        """Restricted collision stopping power at the compiled cut, in MeV cm^2/g."""
        self._check_delta_cut(delta_cut)
        return self._electron_lookup("restricted_stopping", energy, material)

    def radiative_stopping_power(self, energy: float, material: int) -> float:
        """Radiative (bremsstrahlung) mass stopping power, in MeV cm^2/g."""
        return self._electron_lookup("radiative_stopping", energy, material)

    def moller_cross_section(self, energy: float, material: int, delta_cut: float) -> float:
        """Moller cross section for delta rays above the compiled cut, in cm^2/g."""
        self._check_delta_cut(delta_cut)
        return self._electron_lookup("moller", energy, material)

    def csda_range(self, energy: float, material: int) -> float:
        """Continuous-slowing-down range, in g/cm^2."""
        return self._electron_lookup("csda_range", energy, material)

    def scattering_power(self, energy: float, material: int) -> float:
        """Multiple-scattering power, in rad^2 cm^2/g."""
        return self._electron_lookup("scattering_power", energy, material)

    # -- sub-grid product integration (AGENTS.md 2.7) -----------------------

    def integrate_product(
        self,
        quantity_a: str,
        quantity_b: str,
        material: int,
        e_lo: float,
        e_hi: float,
        *,
        n_sub: int = 64,
    ) -> float:
        r"""Integrate the product of two tabulated electron quantities over energy.

        Returns ``\int_{e_lo}^{e_hi} a(E) b(E) dE`` where ``a`` and ``b`` are the
        log-linear interpolants of the named quantities — capturing their intra-bin
        covariance, which multiplying separately averaged bin quantities discards
        (AGENTS.md 2.7). The interval is sampled on a fine geometric sub-grid and
        the product integrated by the trapezoidal rule; ``n_sub`` sets the resolution.
        """
        if e_hi <= e_lo:
            raise ValueError(f"empty interval [{e_lo}, {e_hi}]")
        energies = np.geomspace(e_lo, e_hi, n_sub + 1)
        a = self._interp_electron(quantity_a, material, energies)
        b = self._interp_electron(quantity_b, material, energies)
        return float(np.trapezoid(a * b, energies))

    # -- internals ----------------------------------------------------------

    def _check_delta_cut(self, delta_cut: float) -> None:
        if not math.isclose(delta_cut, self._data.delta_cut, rel_tol=1e-9, abs_tol=0.0):
            raise ValueError(
                f"tables were compiled at delta_cut={self._data.delta_cut} MeV but the "
                f"transport requested {delta_cut} MeV; recompile at the run's ECUT"
            )

    def _electron_lookup(self, name: str, energy: float, material: int) -> float:
        if energy <= 0.0:
            raise ValueError(f"non-positive electron energy {energy} MeV")
        return lookup_loglinear_2d(
            self._electron[name], material, self._e_log_min, self._e_inv_dlog, self._e_n, energy
        )

    def _interp_electron(self, name: str, material: int, energies: np.ndarray) -> np.ndarray:
        if name not in self._electron:
            raise ValueError(f"unknown electron quantity {name!r}")
        interp = np.interp(np.log(energies), self._log_electron, self._electron[name][material])
        return np.asarray(interp, dtype=np.float64)
