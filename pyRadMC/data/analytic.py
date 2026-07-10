"""Analytic photon cross-section parameterization for Phase 0.

One implementation of :class:`pyRadMC.data.interface.CrossSectionSource`. Its job is to
stand up the reference transport loop with smooth, closed-form cross-sections that are
correct to a few percent for water between 0.05 and 20 MeV. It is *not* final-accuracy
data; the tabulated backend (Phase 5) replaces it wherever accuracy matters.

Channels and their parameterizations:

Compton (incoherent)
    Klein-Nishina total cross-section per electron on free electrons at rest, times the
    material electron density. Klein & Nishina, Z. Phys. 52, 853 (1929),
    doi:10.1007/BF01366453. Electron binding is neglected; for water this overestimates
    the incoherent cross-section by a few percent at 50 keV and is negligible above
    ~200 keV.

Photoelectric
    A crude ``E^-3`` power law anchored at 50 keV, the photon transport cutoff. The
    approximate ``tau proportional to Z^4 / E^3`` scaling is standard; see Attix,
    *Introduction to Radiological Physics*, ch. 7 (1986), doi:10.1002/9783527617135.
    Above 1 MeV the photoelectric channel in water is below 1e-5 of the total, so the
    crudeness is irrelevant to the MeV transport this backend serves.

Pair production
    Near-threshold shape ``(1 - E_th/E)^3`` (Motz, Olsen & Koch, Rev. Mod. Phys. 41,
    581 (1969), doi:10.1103/RevModPhys.41.581) times a quadratic polynomial in
    ``ln(E/E_th)``, least-squares calibrated at construction so that the *total*
    attenuation coefficient reproduces the NIST/XCOM water table (Hubbell & Seltzer,
    NISTIR 5632 (1995), doi:10.18434/T4D01F) at the anchor energies. Nuclear and
    electron (triplet) pair production are not distinguished.

Rayleigh (coherent)
    Excluded, deliberately: **no Rayleigh by default** (AGENTS.md section 2.9). Because
    the pair channel is calibrated against totals that *include* coherent scattering,
    above 2 MeV the pair channel silently absorbs the coherent contribution. This keeps
    the total correct and misassigns less than 0.5 percent of the total to the wrong
    channel, which is acceptable for Phase 0 interaction branching.

All energies in MeV, mass attenuation coefficients in cm^2/g, macroscopic quantities in
1/cm (see :mod:`pyRadMC.data.interface`).
"""

from __future__ import annotations

import math

import numpy as np

from pyRadMC import ELECTRON_MASS_MEV
from pyRadMC.data.interface import CrossSectionSource, PhotonProcess
from pyRadMC.data.materials import MATERIALS, WATER

__all__ = ["AnalyticCrossSections"]

CLASSICAL_ELECTRON_RADIUS_CM: float = 2.817_940_3262e-13
"""Classical electron radius in cm (CODATA 2018)."""

PAIR_THRESHOLD_MEV: float = 2.0 * ELECTRON_MASS_MEV
"""Pair-production threshold, 2 m_e c^2. The (small) triplet threshold offset is ignored."""

_PHOTOELECTRIC_ANCHOR_ENERGY_MEV = 0.050
_PHOTOELECTRIC_ANCHOR_WATER = 0.030
"""tau/rho for water at 50 keV in cm^2/g, NIST XCOM, one significant figure.

One significant figure is all the E^-3 power law deserves. Verify against
https://physics.nist.gov/PhysRefData/Xcom/ before trusting it further than Phase 0 does.
"""

# Total mass attenuation coefficients for liquid water, coherent scattering included,
# cm^2/g. Hubbell & Seltzer, NISTIR 5632 (1995), doi:10.18434/T4D01F. Transcribed for
# use as pair-production calibration anchors; the unit test pins an independent
# transcription of the same table at 1, 2, 6, 10 and 15 MeV.
_WATER_TOTAL_MU_OVER_RHO_ANCHORS: tuple[tuple[float, float], ...] = (
    (2.0, 0.04942),
    (3.0, 0.03969),
    (4.0, 0.03403),
    (5.0, 0.03031),
    (6.0, 0.02770),
    (8.0, 0.02429),
    (10.0, 0.02219),
    (15.0, 0.01941),
    (20.0, 0.01813),
)


def _klein_nishina_total_cm2(energy_mev: float) -> float:
    """Total Klein-Nishina cross-section per free electron at rest, in cm^2.

    .. math::

        \\sigma_{KN}(\\alpha) = 2 \\pi r_e^2 \\left[
            \\frac{1+\\alpha}{\\alpha^2}\\left(
                \\frac{2(1+\\alpha)}{1+2\\alpha} - \\frac{\\ln(1+2\\alpha)}{\\alpha}
            \\right)
            + \\frac{\\ln(1+2\\alpha)}{2\\alpha}
            - \\frac{1+3\\alpha}{(1+2\\alpha)^2}
        \\right]

    with :math:`\\alpha = E / m_e c^2`. Klein & Nishina (1929),
    doi:10.1007/BF01366453; the closed form is standard, e.g. Attix (1986) eq. 7.15.
    """
    alpha = energy_mev / ELECTRON_MASS_MEV
    one_plus_2a = 1.0 + 2.0 * alpha
    log_term = math.log(one_plus_2a)
    bracket = (
        (1.0 + alpha) / alpha**2 * (2.0 * (1.0 + alpha) / one_plus_2a - log_term / alpha)
        + log_term / (2.0 * alpha)
        - (1.0 + 3.0 * alpha) / one_plus_2a**2
    )
    return 2.0 * math.pi * CLASSICAL_ELECTRON_RADIUS_CM**2 * bracket


class AnalyticCrossSections(CrossSectionSource):
    """Closed-form Phase 0 photon cross-sections; see the module docstring.

    Parameters
    ----------
    geometry_densities
        The ``(material, maximum mass density in g/cm^3)`` pairs present in the
        geometry. The Woodcock majorant is taken over exactly these; transporting
        through a geometry containing a denser voxel than declared here silently
        biases the transport (see :meth:`majorant`).
    """

    def __init__(
        self, geometry_densities: tuple[tuple[int, float], ...] = ((WATER, 1.0),)
    ) -> None:
        for material, density in geometry_densities:
            if not 0 <= material < len(MATERIALS):
                raise ValueError(f"unknown material index {material}")
            if density <= 0.0:
                raise ValueError(f"non-positive density {density} for material {material}")
        self._geometry_densities = geometry_densities
        self._pair_coeffs = self._fit_pair_coefficients()

    # -- photons ------------------------------------------------------------

    def mu_over_rho(self, energy: float, material: int, process: int) -> float:
        """Mass attenuation coefficient for one channel, in cm^2/g."""
        if energy <= 0.0:
            raise ValueError(f"non-positive photon energy {energy} MeV")
        if not 0 <= material < len(MATERIALS):
            raise ValueError(f"unknown material index {material}")

        if process == PhotonProcess.COMPTON:
            electrons = MATERIALS[material].electrons_per_gram
            return electrons * _klein_nishina_total_cm2(energy)
        if process == PhotonProcess.PHOTOELECTRIC:
            return self._photoelectric(energy)
        if process == PhotonProcess.PAIR:
            return self._pair(energy)
        if process == PhotonProcess.RAYLEIGH:
            # No Rayleigh by default; see the module docstring.
            return 0.0
        raise ValueError(f"unknown photon process {process}")

    def mu_over_rho_total(self, energy: float, material: int) -> float:
        """Total mass attenuation coefficient: the sum over enabled channels."""
        return (
            self.mu_over_rho(energy, material, PhotonProcess.COMPTON)
            + self.mu_over_rho(energy, material, PhotonProcess.PHOTOELECTRIC)
            + self.mu_over_rho(energy, material, PhotonProcess.PAIR)
        )

    def majorant(self, energy: float) -> float:
        """Woodcock majorant over the declared geometry contents, in 1/cm.

        The maximum of ``rho * mu/rho_total`` over the ``(material, max density)``
        pairs declared at construction. Woodcock et al. (1965), ANL-7050.
        """
        return max(
            density * self.mu_over_rho_total(energy, material)
            for material, density in self._geometry_densities
        )

    # -- electrons (Phase 2; electrons are not transported in Phase 0) -------

    def restricted_stopping_power(
        self, energy: float, material: int, delta_cut: float
    ) -> float:
        """Not implemented in Phase 0: electron energy is deposited locally (KERMA)."""
        raise NotImplementedError("Phase 2: electron transport is not part of Phase 0")

    def radiative_stopping_power(self, energy: float, material: int) -> float:
        """Not implemented in Phase 0: electron energy is deposited locally (KERMA)."""
        raise NotImplementedError("Phase 2: electron transport is not part of Phase 0")

    def csda_range(self, energy: float, material: int) -> float:
        """Not implemented in Phase 0: electron energy is deposited locally (KERMA)."""
        raise NotImplementedError("Phase 2: electron transport is not part of Phase 0")

    def scattering_power(self, energy: float, material: int) -> float:
        """Not implemented in Phase 0: electron energy is deposited locally (KERMA)."""
        raise NotImplementedError("Phase 2: electron transport is not part of Phase 0")

    # -- construction ---------------------------------------------------------

    def build_tables(self) -> object:
        """Not implemented in Phase 0: the reference backend uses the host API directly."""
        raise NotImplementedError("Phase 1: kernel table flattening is not part of Phase 0")

    # -- internals ------------------------------------------------------------

    def _photoelectric(self, energy: float) -> float:
        """Photoelectric tau/rho for water, E^-3 anchored at 50 keV. See module docstring."""
        return (
            _PHOTOELECTRIC_ANCHOR_WATER
            * (_PHOTOELECTRIC_ANCHOR_ENERGY_MEV / energy) ** 3
        )

    def _pair(self, energy: float) -> float:
        """Pair kappa/rho for water: threshold shape times a calibrated log-polynomial."""
        if energy <= PAIR_THRESHOLD_MEV:
            return 0.0
        shape = (1.0 - PAIR_THRESHOLD_MEV / energy) ** 3
        log_e = math.log(energy / PAIR_THRESHOLD_MEV)
        c0, c1, c2 = self._pair_coeffs
        # The fit is constrained by physical anchors, but extrapolation just above
        # threshold could dip negative; a negative cross-section is never acceptable.
        return max(0.0, shape * (c0 + c1 * log_e + c2 * log_e**2))

    def _fit_pair_coefficients(self) -> tuple[float, float, float]:
        """Least-squares calibrate the pair channel against the NIST water totals.

        Solves ``kappa_anchor(E) / shape(E) = c0 + c1 ln(E/E_th) + c2 ln^2(E/E_th)``
        where ``kappa_anchor = mu_total_NIST - mu_KN - tau``. See the module docstring
        for why this includes the (excluded) coherent channel above 2 MeV.
        """
        energies = np.array([e for e, _ in _WATER_TOTAL_MU_OVER_RHO_ANCHORS])
        totals = np.array([t for _, t in _WATER_TOTAL_MU_OVER_RHO_ANCHORS])
        electrons = MATERIALS[WATER].electrons_per_gram

        kn = np.array([electrons * _klein_nishina_total_cm2(float(e)) for e in energies])
        tau = np.array([self._photoelectric(float(e)) for e in energies])
        shape = (1.0 - PAIR_THRESHOLD_MEV / energies) ** 3
        target = (totals - kn - tau) / shape

        log_e = np.log(energies / PAIR_THRESHOLD_MEV)
        design = np.stack([np.ones_like(log_e), log_e, log_e**2], axis=1)
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
