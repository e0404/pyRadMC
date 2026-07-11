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


_TWO_PI_RE2_MC2: float = 2.0 * math.pi * CLASSICAL_ELECTRON_RADIUS_CM**2 * ELECTRON_MASS_MEV
"""2 pi r_e^2 m_e c^2 in MeV cm^2: the Moller/Bethe prefactor per electron."""

_WATER_MEAN_EXCITATION_MEV: float = 75.0e-6
"""Mean excitation energy I of liquid water, 75 eV (ICRU Report 37, 1984)."""

_HIGHLAND_CONSTANT_MEV: float = 14.1
_WATER_RADIATION_LENGTH_G_CM2: float = 36.08

# Sternheimer density-effect parameters for liquid water: a, m, x0, x1, Cbar, delta0.
# Sternheimer, Berger & Seltzer, At. Data Nucl. Data Tables 30, 261 (1984),
# doi:10.1016/0092-640X(84)90002-0.
_STERNHEIMER_WATER: tuple[float, float, float, float, float, float] = (
    0.09116,
    3.4773,
    0.2400,
    2.8004,
    3.5017,
    0.097,
)

# Radiative stopping power anchors for water (MeV -> MeV cm^2/g), NIST ESTAR.
# The log-quadratic fit below passes through them exactly; the unit test pins them.
_RADIATIVE_ANCHORS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0128),
    (10.0, 0.1813),
    (20.0, 0.4008),
)


def _fit_radiative_coefficients() -> tuple[float, float, float]:
    """Exact log-quadratic through the three ESTAR radiative anchors."""
    log_e = np.log([e for e, _ in _RADIATIVE_ANCHORS])
    log_s = np.log([s for _, s in _RADIATIVE_ANCHORS])
    design = np.stack([np.ones_like(log_e), log_e, log_e**2], axis=1)
    c = np.linalg.solve(design, log_s)
    return float(c[0]), float(c[1]), float(c[2])


_RADIATIVE_FIT_COEFFS: tuple[float, float, float] = _fit_radiative_coefficients()


def _density_effect_water(tau: float) -> float:
    """Sternheimer density-effect correction delta for liquid water.

    Piecewise in x = log10(p / m_e c) = log10(sqrt(tau (tau + 2))): zero-ish below
    x0 (insulator form with the small delta0 conduction term), the standard
    a (x1 - x)^m interpolation between x0 and x1, and the asymptotic 4.6052 x - Cbar
    above. Sternheimer, Berger & Seltzer (1984), doi:10.1016/0092-640X(84)90002-0.
    """
    a, m, x0, x1, cbar, delta0 = _STERNHEIMER_WATER
    x = 0.5 * math.log10(tau * (tau + 2.0))
    if x >= x1:
        return 4.6052 * x - cbar
    if x >= x0:
        return 4.6052 * x - cbar + a * math.pow(x1 - x, m)
    return delta0 * math.pow(10.0, 2.0 * (x - x0))


def moller_dcs_per_electron(energy: float, energy_transfer: float) -> float:
    r"""Moller differential cross-section per electron, in cm^2/MeV.

    For primary kinetic energy T and delta kinetic energy W, with x = W/T:

    .. math::

        \frac{d\sigma}{dW} = \frac{2\pi r_e^2 mc^2}{\beta^2 T^2}
            \left[ \frac{1}{x^2} + \frac{1}{(1-x)^2}
                 + \left(\frac{\tau}{\tau+1}\right)^2
                 - \frac{2\tau+1}{(\tau+1)^2}\,\frac{1}{x(1-x)} \right],
        \qquad 0 < x \le \tfrac12.

    Moller (1932), doi:10.1002/andp.19324060506; e.g. Salvat et al., PENELOPE-2018,
    eq. (3.111). Zero outside the kinematic interval. Exposed at module level so
    tests can integrate it independently of the closed forms that use it.
    """
    if energy_transfer <= 0.0 or energy_transfer > energy / 2.0:
        return 0.0
    tau = energy / ELECTRON_MASS_MEV
    beta_sq = tau * (tau + 2.0) / (tau + 1.0) ** 2
    x = energy_transfer / energy
    f = (
        1.0 / x**2
        + 1.0 / (1.0 - x) ** 2
        + (tau / (tau + 1.0)) ** 2
        - (2.0 * tau + 1.0) / (tau + 1.0) ** 2 / (x * (1.0 - x))
    )
    return _TWO_PI_RE2_MC2 / (beta_sq * energy**2) * f


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

    def __init__(self, geometry_densities: tuple[tuple[int, float], ...] = ((WATER, 1.0),)) -> None:
        for material, density in geometry_densities:
            if not 0 <= material < len(MATERIALS):
                raise ValueError(f"unknown material index {material}")
            if density <= 0.0:
                raise ValueError(f"non-positive density {density} for material {material}")
        self._geometry_densities = geometry_densities
        self._pair_coeffs = self._fit_pair_coefficients()
        self._range_log_energies, self._range_values = self._build_range_table()

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

    # -- electrons (Phase 1; water-only, like the photon channels) -----------

    def restricted_stopping_power(self, energy: float, material: int, delta_cut: float) -> float:
        r"""Restricted collision stopping power, Berger-Seltzer form, in MeV cm^2/g.

        .. math::

            S(T, \Delta)/\rho = \frac{2\pi r_e^2 m c^2 N_e}{\beta^2}
                \left[ \ln\frac{\tau^2(\tau+2)}{2 (I/mc^2)^2}
                       + G^-(\tau, w) - \delta \right],
            \qquad w = \Delta/T,

        with the electron term (ICRU Report 37 (1984); Salvat et al., PENELOPE-2018,
        sec. 3.2.3, doi:10.1787/32da5043-en)

        .. math::

            G^-(\tau, w) = -1 - \beta^2 + \ln\bigl(4(1-w)w\bigr) + \frac{1}{1-w}
                + \frac{(\tau w)^2/2 + (2\tau+1)\ln(1-w)}{(\tau+1)^2}

        and the Sternheimer density effect :math:`\delta`. At ``delta_cut = T/2`` this
        is the unrestricted collision stopping power (the ESTAR collision column);
        that limit and the Moller consistency identity are both test-pinned.

        Shell corrections are neglected — percent-level below ~50 keV in water, far
        below ECUT's influence on dose at this engine's energies.
        """
        self._check_electron_args(energy, material)
        tau = energy / ELECTRON_MASS_MEV
        beta_sq = tau * (tau + 2.0) / (tau + 1.0) ** 2
        w = min(delta_cut / energy, 0.5)
        if w <= 0.0:
            raise ValueError(f"non-positive delta_cut {delta_cut} MeV")

        g_minus = (
            -1.0
            - beta_sq
            + math.log(4.0 * (1.0 - w) * w)
            + 1.0 / (1.0 - w)
            + ((tau * w) ** 2 / 2.0 + (2.0 * tau + 1.0) * math.log(1.0 - w)) / (tau + 1.0) ** 2
        )
        log_term = math.log(
            tau**2 * (tau + 2.0) / (2.0 * (_WATER_MEAN_EXCITATION_MEV / ELECTRON_MASS_MEV) ** 2)
        )
        electrons = MATERIALS[material].electrons_per_gram
        prefactor = _TWO_PI_RE2_MC2 * electrons / beta_sq
        return prefactor * (log_term + g_minus - _density_effect_water(tau))

    def radiative_stopping_power(self, energy: float, material: int) -> float:
        """Radiative stopping power, in MeV cm^2/g: a log-quadratic ESTAR fit.

        Like the pair channel, this is a calibration, not a theory: an exact
        log-quadratic through the transcribed NIST ESTAR water anchors at 1, 10 and
        20 MeV (Berger & Seltzer's data behind ESTAR; ICRU Report 37 (1984)). Below
        1 MeV it extrapolates smoothly; there the radiative share of the total
        stopping power is under one percent, so the extrapolation error is
        dosimetrically irrelevant.
        """
        self._check_electron_args(energy, material)
        log_e = math.log(energy)
        c0, c1, c2 = _RADIATIVE_FIT_COEFFS
        return math.exp(c0 + c1 * log_e + c2 * log_e**2)

    def moller_cross_section(self, energy: float, material: int, delta_cut: float) -> float:
        r"""Restricted Moller cross-section per unit mass, in cm^2/g.

        Closed-form integral of the Moller DCS (:func:`moller_dcs_per_electron`) from
        ``delta_cut`` to ``T/2``:

        .. math::

            \sigma(T, \Delta) = \frac{2\pi r_e^2 mc^2 N_e}{\beta^2 T}
                \left[ \left(\frac{\tau}{\tau+1}\right)^2\left(\tfrac12 - w\right)
                + \frac{1}{w} - \frac{1}{1-w}
                + \frac{2\tau+1}{(\tau+1)^2} \ln\frac{w}{1-w} \right]

        Moller (1932), doi:10.1002/andp.19324060506. Zero at or below ``2 delta_cut``.
        """
        self._check_electron_args(energy, material)
        if energy <= 2.0 * delta_cut:
            return 0.0
        tau = energy / ELECTRON_MASS_MEV
        beta_sq = tau * (tau + 2.0) / (tau + 1.0) ** 2
        w = delta_cut / energy

        bracket = (
            (tau / (tau + 1.0)) ** 2 * (0.5 - w)
            + 1.0 / w
            - 1.0 / (1.0 - w)
            + (2.0 * tau + 1.0) / (tau + 1.0) ** 2 * math.log(w / (1.0 - w))
        )
        electrons = MATERIALS[material].electrons_per_gram
        return _TWO_PI_RE2_MC2 * electrons / (beta_sq * energy) * bracket

    def csda_range(self, energy: float, material: int) -> float:
        """CSDA range in g/cm^2: the range integral of the total stopping power.

        Precomputed at construction by trapezoidal integration of ``1/S_total`` on a
        dense logarithmic grid from 1 keV, then interpolated in log energy. Errs on
        the *under*-estimating side for range rejection (the grid starts above zero,
        truncating the sub-keV tail) by well under 1e-4 g/cm^2 — the safe side is
        documented in the interface as the *over*-estimate, so range rejection (when
        it arrives) must add its own safety margin anyway; the truncation here is
        orders of magnitude below any voxel dimension this engine will see.
        """
        self._check_electron_args(energy, material)
        log_e = math.log(energy)
        return float(np.interp(log_e, self._range_log_energies, self._range_values))

    def scattering_power(self, energy: float, material: int) -> float:
        r"""Mass angular scattering power, in rad^2 cm^2/g.

        Rossi-Greisen form with the Highland constant:

        .. math::

            T/\rho = \left(\frac{14.1\,\mathrm{MeV}}{p v}\right)^2 \frac{1}{X_0},
            \qquad X_0(\text{water}) = 36.08\ \mathrm{g/cm^2}.

        Rossi & Greisen, Rev. Mod. Phys. 13, 240 (1941),
        doi:10.1103/RevModPhys.13.240; Highland, NIM 129, 497 (1975),
        doi:10.1016/0029-554X(75)90743-0. The step-length logarithmic correction of
        Highland's formula is neglected — a stated approximation, adequate for
        depth-dose observables; revisit before trusting penumbra shapes.
        """
        self._check_electron_args(energy, material)
        tau = energy / ELECTRON_MASS_MEV
        # p*v = p^2 c^2 / E_total, in MeV.
        pv = ELECTRON_MASS_MEV * tau * (tau + 2.0) / (tau + 1.0)
        return (_HIGHLAND_CONSTANT_MEV / pv) ** 2 / _WATER_RADIATION_LENGTH_G_CM2

    def _check_electron_args(self, energy: float, material: int) -> None:
        """Shared validation for the electron accessors."""
        if energy <= 0.0:
            raise ValueError(f"non-positive electron kinetic energy {energy} MeV")
        if not 0 <= material < len(MATERIALS):
            raise ValueError(f"unknown material index {material}")

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

    def _build_range_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Integrate 1/S_total on a dense log grid: the CSDA range lookup for water.

        Trapezoidal cumulative integration from 1 keV; the truncated sub-keV tail is
        below 1e-5 g/cm^2. 600 log-spaced points hold the interpolation error well
        under the 3 percent test tolerance.
        """
        energies = np.geomspace(1.0e-3, 30.0, 600)
        inverse_total = np.array(
            [
                1.0
                / (
                    self.restricted_stopping_power(float(e), WATER, delta_cut=float(e) / 2.0)
                    + self.radiative_stopping_power(float(e), WATER)
                )
                for e in energies
            ]
        )
        steps = np.diff(energies) * 0.5 * (inverse_total[1:] + inverse_total[:-1])
        ranges = np.concatenate(([0.0], np.cumsum(steps)))
        return np.log(energies), ranges
