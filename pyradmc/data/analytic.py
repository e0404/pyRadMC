"""Analytic closed-form photon cross-section parameterization.

One implementation of :class:`pyradmc.data.interface.CrossSectionSource`. Its job is to
stand up the reference transport loop with smooth, closed-form cross-sections that are
correct to a few percent for water between 0.05 and 20 MeV. It is *not* final-accuracy
data; the tabulated backend replaces it wherever accuracy matters.

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
    channel, which is acceptable for interaction branching.

All energies in MeV, mass attenuation coefficients in cm^2/g, macroscopic quantities in
1/cm (see :mod:`pyradmc.data.interface`).
"""

from __future__ import annotations

import math

import numpy as np

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data import berger_seltzer
from pyradmc.data.berger_seltzer import (
    CLASSICAL_ELECTRON_RADIUS_CM,
    moller_dcs_per_electron,
)
from pyradmc.data.goudsmit_saunderson import transport_moment_scattering_power
from pyradmc.data.interface import CrossSectionSource, PhotonProcess
from pyradmc.data.materials import MATERIALS, WATER

__all__ = ["AnalyticCrossSections", "moller_dcs_per_electron"]

PAIR_THRESHOLD_MEV: float = 2.0 * ELECTRON_MASS_MEV
"""Pair-production threshold, 2 m_e c^2. The (small) triplet threshold offset is ignored."""

_PHOTOELECTRIC_ANCHOR_ENERGY_MEV = 0.050
_PHOTOELECTRIC_ANCHOR_WATER = 0.030
"""tau/rho for water at 50 keV in cm^2/g, NIST XCOM, one significant figure.

One significant figure is all the E^-3 power law deserves. Verify against
https://physics.nist.gov/PhysRefData/Xcom/ before trusting it any further than this
parameterization already does.
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
    r"""Total Klein-Nishina cross-section per free electron at rest, in cm^2.

    .. math::

        \sigma_{KN}(\alpha) = 2 \pi r_e^2 \left[
            \frac{1+\alpha}{\alpha^2}\left(
                \frac{2(1+\alpha)}{1+2\alpha} - \frac{\ln(1+2\alpha)}{\alpha}
            \right)
            + \frac{\ln(1+2\alpha)}{2\alpha}
            - \frac{1+3\alpha}{(1+2\alpha)^2}
        \right]

    with :math:`\alpha = E / m_e c^2`. Klein & Nishina (1929),
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
    """Closed-form photon cross-sections; see the module docstring.

    Parameters
    ----------
    geometry_densities
        The ``(material, maximum mass density in g/cm^3)`` pairs present in the
        geometry. The Woodcock majorant is taken over exactly these; transporting
        through a geometry containing a denser voxel than declared here silently
        biases the transport (see :meth:`majorant`).
    """

    def __init__(self, geometry_densities: tuple[tuple[int, float], ...] = ((WATER, 1.0),)) -> None:
        for material, density in geometry_densities:
            if not 0 <= material < self.n_materials:
                raise ValueError(
                    f"material index {material} beyond the analytic source (water only)"
                )
            if density <= 0.0:
                raise ValueError(f"non-positive density {density} for material {material}")
        self._geometry_densities = geometry_densities
        self._pair_coeffs = self._fit_pair_coefficients()
        self._radiative_coeffs = berger_seltzer.radiative_fit_coefficients(
            MATERIALS[WATER].radiative_anchors
        )
        self._range_log_energies, self._range_values = berger_seltzer.csda_range_table(
            MATERIALS[WATER]
        )

    @property
    def n_materials(self) -> int:
        """Water only, permanently.

        The photoelectric anchor, pair calibration, I-value, density effect and
        radiative fit are all water-specific. The registry may grow past water;
        this source does not.
        """
        return 1

    @property
    def provenance(self) -> str:
        """Name the parameterization, so an archived dose is not mistaken for data.

        These are closed-form fits with stated few-percent accuracy in the soft
        spectrum, not measured cross-sections; a result built on them should say so
        in the same place a tabulated result cites its library.
        """
        return (
            "analytic closed-form water (Klein-Nishina without binding, power-law "
            "photoelectric, fitted pair, Berger-Seltzer ICRU-37 stopping)"
        )

    # -- photons ------------------------------------------------------------

    def mu_over_rho(self, energy: float, material: int, process: int) -> float:
        """Mass attenuation coefficient for one channel, in cm^2/g."""
        if energy <= 0.0:
            raise ValueError(f"non-positive photon energy {energy} MeV")
        if not 0 <= material < self.n_materials:
            raise ValueError(f"material index {material} beyond the analytic source (water only)")

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

    # -- electrons (water-only, like the photon channels) -----------

    def restricted_stopping_power(self, energy: float, material: int, delta_cut: float) -> float:
        """Restricted collision stopping power, Berger-Seltzer form, in MeV cm^2/g.

        Delegates to :func:`pyradmc.data.berger_seltzer.restricted_collision_stopping`
        (governing equation, citations and stated approximations there) with the water
        registry entry — whose I-value and Sternheimer coefficients are the constants
        this backend evaluated inline before the multi-material work moved them.
        """
        self._check_electron_args(energy, material)
        return berger_seltzer.restricted_collision_stopping(energy, MATERIALS[material], delta_cut)

    def radiative_stopping_power(self, energy: float, material: int) -> float:
        """Radiative stopping power, in MeV cm^2/g: a log-quadratic ESTAR fit.

        Like the pair channel, this is a calibration, not a theory: an exact
        log-quadratic (:func:`pyradmc.data.berger_seltzer.radiative_stopping`)
        through the material's transcribed NIST ESTAR anchors at 1, 10 and 20 MeV
        (Berger & Seltzer's data behind ESTAR; ICRU Report 37 (1984)).
        """
        self._check_electron_args(energy, material)
        return berger_seltzer.radiative_stopping(energy, self._radiative_coeffs)

    def moller_cross_section(self, energy: float, material: int, delta_cut: float) -> float:
        """Restricted Moller cross-section per unit mass, in cm^2/g.

        Delegates to :func:`pyradmc.data.berger_seltzer.restricted_moller_cross_section`
        (closed form and citation there). Zero at or below ``2 delta_cut``.
        """
        self._check_electron_args(energy, material)
        return berger_seltzer.restricted_moller_cross_section(
            energy, MATERIALS[material], delta_cut
        )

    def csda_range(self, energy: float, material: int) -> float:
        """CSDA range in g/cm^2: the range integral of the total stopping power.

        Precomputed at construction (:func:`pyradmc.data.berger_seltzer.csda_range_table`)
        and interpolated in log energy. Errs on the *under*-estimating side for range
        rejection (the table's grid starts above zero, truncating the sub-keV tail) by
        well under 1e-4 g/cm^2 — the safe side is documented in the interface as the
        *over*-estimate, so range rejection (when it arrives) must add its own safety
        margin anyway; the truncation here is orders of magnitude below any voxel
        dimension this engine will see.
        """
        self._check_electron_args(energy, material)
        log_e = math.log(energy)
        return float(np.interp(log_e, self._range_log_energies, self._range_values))

    def scattering_power(self, energy: float, material: int, delta_cut: float) -> float:
        r"""Mass angular scattering power, in rad^2 cm^2/g.

        The Class-II first transport moment of the Moliere-screened Rutherford law,
        ``T = 2 (N_A/A) [Z^2 sigma_tr + Z(sigma_tr - sigma_tr,M^hard)]``,
        from the material's composition
        (:func:`pyradmc.data.goudsmit_saunderson.transport_moment_scattering_power`).
        That is the strength Goudsmit-Saunderson theory pins for the shape the GS
        tables are built from; the Rossi-Greisen/Highland ``(14.1/pv)^2 / X_0`` core
        width used until 2026-09 lacked its energy-growing logarithm and
        under-scattered multi-MeV electrons. Atomic-electron scattering below
        ``delta_cut`` remains condensed; the hard Moller transport moment above
        the cut is removed because the transport loop applies it explicitly.
        """
        self._check_electron_args(energy, material)
        if delta_cut <= 0.0:
            raise ValueError(f"non-positive delta_cut {delta_cut} MeV")
        return transport_moment_scattering_power(MATERIALS[material].composition, energy, delta_cut)

    def _check_electron_args(self, energy: float, material: int) -> None:
        """Shared validation for the electron accessors."""
        if energy <= 0.0:
            raise ValueError(f"non-positive electron kinetic energy {energy} MeV")
        if not 0 <= material < self.n_materials:
            raise ValueError(f"material index {material} beyond the analytic source (water only)")

    # -- internals ------------------------------------------------------------

    def _photoelectric(self, energy: float) -> float:
        """Photoelectric tau/rho for water, E^-3 anchored at 50 keV. See module docstring."""
        return _PHOTOELECTRIC_ANCHOR_WATER * (_PHOTOELECTRIC_ANCHOR_ENERGY_MEV / energy) ** 3

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
