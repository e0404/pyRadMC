r"""Positron interaction data: Bhabha scattering, collision stopping, annihilation.

The positron twins of :mod:`pyradmc.data.berger_seltzer`, parameterized by
:class:`~pyradmc.data.materials.MaterialData` in the same way, plus the two-photon
annihilation-in-flight cross-section. Host-side, compile-time functions: the
:class:`~pyradmc.data.interface.CrossSectionSource` accessors
(``positron_restricted_stopping_power``, ``bhabha_cross_section``,
``annihilation_cross_section``) delegate here, and no kernel ever calls them:
kernels see the flattened :class:`PositronTables` (:func:`build_positron_tables`).

The Class II scheme for positrons mirrors the electron one:

- collisions transferring less than the cut are the **restricted positron
  collision stopping power** (:func:`positron_restricted_collision_stopping`);
- collisions above it are discrete **Bhabha** events
  (:func:`restricted_bhabha_cross_section`, sampled by
  :mod:`pyradmc.physics.bhabha`);
- **annihilation in flight** is a second discrete channel
  (:func:`annihilation_cross_section_per_electron`, sampled by
  :mod:`pyradmc.physics.annihilation`).

Two differences from Moller that the transport must respect: a positron can hand
over its *whole* kinetic energy (the particles are distinguishable), so the Bhabha
channel opens at ``T > cut`` rather than ``T > 2 cut`` and the scattered positron
may come out below the cut; and the restriction parameter saturates at ``cut = T``
rather than ``T/2``.

Stated approximations
---------------------
- Target electrons are free and at rest; shell corrections are neglected, as for
  electrons (:func:`pyradmc.data.berger_seltzer.restricted_collision_stopping`).
- The mean excitation energy and the Sternheimer density effect are the
  electron's: both depend on the projectile only through its speed.
- **Radiative stopping of positrons is taken equal to the electron's.** The
  positron-to-electron ratio of the radiative stopping power is below one — far
  below at keV energies, approaching one in the MeV range and above for low Z
  (Kim et al., Phys. Rev. A 33, 3002 (1986), doi:10.1103/PhysRevA.33.3002;
  ICRU Report 37 (1984)) — so this overestimates positron bremsstrahlung
  where bremsstrahlung is a minor loss, and no positron radiative accessor is
  added to the interface.
- Annihilation: two-photon only, on free electrons at rest. One-photon
  annihilation (needs a bound electron; relative size of order
  :math:`\alpha^4 Z^5` per atom, negligible in tissue) and three-photon
  annihilation (about 1/372 of two-photon) are neglected. A positron that reaches
  the transport cutoff still annihilates *at rest* (existing approximation,
  unchanged).

All energies in MeV; mass stopping powers in MeV cm^2/g; cross-sections per
electron in cm^2 (differential: cm^2/MeV), per unit mass in cm^2/g.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data.berger_seltzer import (
    CLASSICAL_ELECTRON_RADIUS_CM,
    TWO_PI_RE2_MC2,
    density_effect,
)
from pyradmc.data.materials import MaterialData

if TYPE_CHECKING:
    from pyradmc.data.interface import CrossSectionSource

__all__ = [
    "PositronTables",
    "annihilation_cross_section_per_electron",
    "annihilation_dcs_per_electron",
    "bhabha_dcs_per_electron",
    "build_positron_tables",
    "positron_restricted_collision_stopping",
    "restricted_bhabha_cross_section",
]


def _bhabha_coefficients(tau: float) -> tuple[float, float, float, float, float]:
    """``(beta^2, B1, B2, B3, B4)`` of the Bhabha DCS; see :func:`bhabha_dcs_per_electron`."""
    gamma = tau + 1.0
    beta_sq = tau * (tau + 2.0) / (gamma * gamma)
    y = 1.0 / (gamma + 1.0)
    q = 1.0 - 2.0 * y
    return (
        beta_sq,
        2.0 - y * y,
        q * (3.0 + y * y),
        q * q + q * q * q,
        q * q * q,
    )


def bhabha_dcs_per_electron(energy: float, energy_transfer: float) -> float:
    r"""Bhabha differential cross-section per electron, in cm^2/MeV.

    For positron kinetic energy T and delta-ray kinetic energy W, with
    :math:`x = W/T`, :math:`\gamma = \tau + 1` and :math:`y = 1/(\gamma+1)`:

    .. math::

        \frac{d\sigma}{dW} = \frac{2\pi r_e^2 mc^2}{T^2}
            \left[ \frac{1}{\beta^2 x^2} - \frac{B_1}{x} + B_2 - B_3 x + B_4 x^2 \right],
        \qquad 0 < x \le 1,

    .. math::

        B_1 = 2 - y^2,\quad B_2 = (1-2y)(3+y^2),\quad
        B_3 = (1-2y)^2 + (1-2y)^3,\quad B_4 = (1-2y)^3 .

    Bhabha, Proc. R. Soc. A 154, 195 (1936), doi:10.1098/rspa.1936.0046; in this
    form ICRU Report 37 (1984) and the Geant4 Physics Reference Manual
    ("Moller and Bhabha scattering"). Algebraically identical to the
    :math:`b_i` form of Salvat et al., PENELOPE-2018, sec. 3.2
    (doi:10.1787/32da5043-en), which the unit test transcribes independently.
    Zero outside the kinematic interval. Material-free (per electron).
    """
    if energy_transfer <= 0.0 or energy_transfer > energy:
        return 0.0
    beta_sq, b1, b2, b3, b4 = _bhabha_coefficients(energy / ELECTRON_MASS_MEV)
    x = energy_transfer / energy
    bracket = 1.0 / (beta_sq * x * x) - b1 / x + b2 - b3 * x + b4 * x * x
    return TWO_PI_RE2_MC2 / (energy * energy) * bracket


def positron_restricted_collision_stopping(
    energy: float, material: MaterialData, delta_cut: float
) -> float:
    r"""Restricted positron collision stopping power, in MeV cm^2/g.

    Same structure as the electron's
    (:func:`pyradmc.data.berger_seltzer.restricted_collision_stopping`),

    .. math::

        S^+(T, \Delta)/\rho = \frac{2\pi r_e^2 m c^2 N_e}{\beta^2}
            \left[ \ln\frac{\tau^2(\tau+2)}{2 (I/mc^2)^2}
                   + G^+(\tau, d) - \delta \right],
        \qquad d = \min(\Delta, T)/mc^2,

    with the positron term (:math:`y = 1/(\gamma+1)`)

    .. math::

        G^+(\tau, d) = \ln\frac{4d}{\tau} - \frac{\beta^2}{\tau}\left[
            \tau + 2d - \frac{3 d^2 y}{2} - \left(d - \frac{d^3}{3}\right) y^2
            - \left(\frac{d^2}{2} - \frac{\tau d^3}{3} + \frac{d^4}{4}\right) y^3
        \right].

    Berger & Seltzer's restricted positron formula: ICRU Report 37 (1984);
    Rohrlich & Carlson, Phys. Rev. 93, 38 (1954), doi:10.1103/PhysRev.93.38; the
    same expression as Geant4's ``G4MollerBhabhaModel::ComputeDEDXPerVolume``. At
    :math:`\Delta \ge T` it reduces to the unrestricted

    .. math::

        F^+(\tau) = 2\ln 2 - \frac{\beta^2}{12}\left[23 + \frac{14}{\tau+2}
            + \frac{10}{(\tau+2)^2} + \frac{4}{(\tau+2)^3}\right],

    and for :math:`\Delta < T` it differs from that total by exactly the energy
    moment of the Bhabha DCS above :math:`\Delta`; both statements are test-pinned
    (``tests/unit/test_positron_data.py``).

    Shell corrections neglected; I and the density effect are the electron's (see
    the module docstring).
    """
    if energy <= 0.0:
        raise ValueError(f"non-positive positron kinetic energy {energy} MeV")
    if delta_cut <= 0.0:
        raise ValueError(f"non-positive delta_cut {delta_cut} MeV")
    tau = energy / ELECTRON_MASS_MEV
    gamma = tau + 1.0
    beta_sq = tau * (tau + 2.0) / (gamma * gamma)
    y = 1.0 / (gamma + 1.0)
    d = min(delta_cut, energy) / ELECTRON_MASS_MEV

    g_plus = math.log(4.0 * d / tau) - beta_sq / tau * (
        tau
        + 2.0 * d
        - 1.5 * d * d * y
        - (d - d**3 / 3.0) * y * y
        - (0.5 * d * d - tau * d**3 / 3.0 + 0.25 * d**4) * y**3
    )
    log_term = math.log(
        tau**2 * (tau + 2.0) / (2.0 * (material.mean_excitation_mev / ELECTRON_MASS_MEV) ** 2)
    )
    prefactor = TWO_PI_RE2_MC2 * material.electrons_per_gram / beta_sq
    return prefactor * (log_term + g_plus - density_effect(tau, material.sternheimer))


def restricted_bhabha_cross_section(
    energy: float, material: MaterialData, delta_cut: float
) -> float:
    r"""Restricted Bhabha cross-section per unit mass, in cm^2/g.

    Closed-form integral of the Bhabha DCS (:func:`bhabha_dcs_per_electron`) from
    ``delta_cut`` to ``T``, with :math:`w = \Delta/T`:

    .. math::

        \sigma(T, \Delta) = \frac{2\pi r_e^2 mc^2 N_e}{T}
            \left[ \frac{1}{\beta^2}\left(\frac1w - 1\right) + B_1 \ln w
                 + B_2 (1 - w) - \frac{B_3}{2}(1 - w^2) + \frac{B_4}{3}(1 - w^3) \right].

    Bhabha (1936), doi:10.1098/rspa.1936.0046. Zero at or below ``delta_cut`` —
    not ``2 delta_cut`` as for Moller: the positron may transfer all of T.
    """
    if energy <= 0.0:
        raise ValueError(f"non-positive positron kinetic energy {energy} MeV")
    if energy <= delta_cut:
        return 0.0
    beta_sq, b1, b2, b3, b4 = _bhabha_coefficients(energy / ELECTRON_MASS_MEV)
    w = delta_cut / energy
    bracket = (
        (1.0 / w - 1.0) / beta_sq
        + b1 * math.log(w)
        + b2 * (1.0 - w)
        - 0.5 * b3 * (1.0 - w * w)
        + b4 / 3.0 * (1.0 - w**3)
    )
    return TWO_PI_RE2_MC2 * material.electrons_per_gram / energy * bracket


def annihilation_cross_section_per_electron(energy: float) -> float:
    r"""Two-photon annihilation-in-flight cross-section per electron, in cm^2.

    For a positron of kinetic energy T (:math:`\gamma = 1 + T/mc^2`) on a free
    electron at rest:

    .. math::

        \sigma(\gamma) = \frac{\pi r_e^2}{\gamma + 1} \left[
            \frac{\gamma^2 + 4\gamma + 1}{\gamma^2 - 1}
            \ln\!\left(\gamma + \sqrt{\gamma^2 - 1}\right)
            - \frac{\gamma + 3}{\sqrt{\gamma^2 - 1}} \right].

    Dirac, Proc. Camb. Phil. Soc. 26, 361 (1930), doi:10.1017/S0305004100016091;
    Heitler, *The Quantum Theory of Radiation*, 3rd ed. (1954). Limits
    (both test-pinned): :math:`\pi r_e^2/\beta` as :math:`T \to 0` and
    :math:`\pi r_e^2 [\ln 2\gamma - 1]/\gamma` for :math:`\gamma \gg 1`.

    Exact at this order for any T > 0; evaluated with :math:`\gamma^2 - 1 =
    \tau(\tau + 2)` and ``log1p`` so that the keV range does not cancel. One- and
    three-photon annihilation neglected (module docstring).
    """
    if energy <= 0.0:
        raise ValueError(f"non-positive positron kinetic energy {energy} MeV")
    tau = energy / ELECTRON_MASS_MEV
    gamma = tau + 1.0
    p_sq = tau * (tau + 2.0)
    p = math.sqrt(p_sq)
    bracket = (gamma * gamma + 4.0 * gamma + 1.0) / p_sq * math.log1p(tau + p) - (gamma + 3.0) / p
    return math.pi * CLASSICAL_ELECTRON_RADIUS_CM**2 / (gamma + 1.0) * bracket


def annihilation_dcs_per_electron(energy: float, photon_energy: float) -> float:
    r"""Annihilation-in-flight DCS in one photon's energy, per electron, in cm^2/MeV.

    With :math:`\varepsilon = k/(T + 2mc^2)` the fraction of the available energy
    taken by the *labelled* photon:

    .. math::

        \frac{d\sigma}{d\varepsilon} = \frac{\pi r_e^2}{\gamma - 1}\,
            \frac{1}{\varepsilon}\left[ 1 + \frac{2\gamma}{(\gamma+1)^2}
            - \varepsilon - \frac{1}{(\gamma+1)^2\,\varepsilon} \right],
        \qquad
        \tfrac12\left(1 - \sqrt{\tfrac{\gamma-1}{\gamma+1}}\right) \le \varepsilon
        \le \tfrac12\left(1 + \sqrt{\tfrac{\gamma-1}{\gamma+1}}\right).

    This is the form :math:`S(\varepsilon)` of Heitler's result in which the
    physical two-photon spectrum is :math:`S(\varepsilon) + S(1-\varepsilon)`:
    integrating :math:`S` alone over the whole interval counts each annihilation
    once and gives :func:`annihilation_cross_section_per_electron` (test-pinned),
    and sampling one photon from :math:`S` and giving the other the remainder
    reproduces the symmetric pair. Heitler (1954); Nelson, Hirayama & Rogers,
    *The EGS4 Code System*, SLAC-265 (1985), ch. 2; Geant4 Physics Reference Manual,
    "Positron annihilation into two photons". Zero outside the interval.
    """
    if energy <= 0.0:
        raise ValueError(f"non-positive positron kinetic energy {energy} MeV")
    total = energy + 2.0 * ELECTRON_MASS_MEV
    eps = photon_energy / total
    half_width = 0.5 * math.sqrt(energy / total)
    if eps < 0.5 - half_width or eps > 0.5 + half_width:
        return 0.0
    gamma = 1.0 + energy / ELECTRON_MASS_MEV
    gp_sq = (gamma + 1.0) ** 2
    bracket = 1.0 + 2.0 * gamma / gp_sq - eps - 1.0 / (gp_sq * eps)
    per_fraction = (
        math.pi * CLASSICAL_ELECTRON_RADIUS_CM**2 * ELECTRON_MASS_MEV / energy * bracket / eps
    )
    return per_fraction / total


@dataclass(frozen=True)
class PositronTables:
    """Kernel-side data of the ``"bhabha-in-flight"`` positron model; frozen.

    Laid on the *same* electron log-energy grid as
    :class:`pyradmc.data.tables.CrossSectionTables` built with the same
    ``(ecut, e_max, n_points)``, so a kernel interpolates both with one metadata
    pair. Kept apart from that class because the model is optional: a run that
    does not select it neither builds nor uploads these.

    Attributes
    ----------
    n_points, ecut, e_max
        Grid size, the cutoff (MeV) the tables were built at, and the upper edge.
    electron_log_e_min, electron_inv_dlog
        Lookup metadata, identical to the cross-section tables' electron grid.
    bhabha
        Restricted Bhabha cross-section per unit mass at ``ecut``, in cm^2/g,
        ``(n_materials, n_points)``; zero at and below ``ecut``.
    annihilation
        Two-photon annihilation-in-flight cross-section per unit mass, in cm^2/g,
        ``(n_materials, n_points)``.
    restricted_range
        Positron restricted-collision range to ``ecut``, in g/cm^2, on the
        electron grid, ``(n_materials, n_points)``.
    energy_of_restricted_range
        Its inverse on a shared log-range grid of the positron's own,
        ``(n_materials, n_points)``, built exactly as the electron's
        (:func:`pyradmc.data.tables.build_cross_section_tables`);
        ``range_log_r_min``/``range_inv_dlog`` are its lookup metadata. The
        kernel takes a substep's loss as the difference of two lookups in this
        one table (AGENTS.md section 8).
    """

    n_points: int
    ecut: float
    e_max: float
    electron_log_e_min: float
    electron_inv_dlog: float
    bhabha: npt.NDArray[np.float64]
    annihilation: npt.NDArray[np.float64]
    restricted_range: npt.NDArray[np.float64]
    energy_of_restricted_range: npt.NDArray[np.float64]
    range_log_r_min: float
    range_inv_dlog: float


def build_positron_tables(
    source: CrossSectionSource,
    ecut: float,
    e_max: float,
    n_points: int | None = None,
) -> PositronTables:
    """Flatten the positron queries of a source onto the electron grid.

    Generic over sources, like
    :func:`pyradmc.data.tables.build_cross_section_tables`: everything goes
    through
    :meth:`~pyradmc.data.interface.CrossSectionSource.bhabha_cross_section`,
    :meth:`~pyradmc.data.interface.CrossSectionSource.annihilation_cross_section`,
    :meth:`~pyradmc.data.interface.CrossSectionSource.positron_restricted_range`
    and
    :meth:`~pyradmc.data.interface.CrossSectionSource.positron_energy_after_mass_path`.

    Parameters
    ----------
    source
        The host-side data source to flatten.
    ecut
        Electron cutoff in MeV (accuracy-defining, AGENTS.md 2.8): the Bhabha
        threshold and the foot of the range. The grid starts at ``ecut / 2``
        like the electron tables.
    e_max
        Upper grid edge in MeV; refused above the source's declared energy
        domain, like the cross-section tables.
    n_points
        Grid nodes; defaults to :data:`pyradmc.data.tables.TABLE_POINTS`.
    """
    from pyradmc.data.interface import require_energy_domain
    from pyradmc.data.tables import _RANGE_R_MIN_FRACTION, TABLE_POINTS

    require_energy_domain(source, e_max)
    if n_points is None:
        n_points = TABLE_POINTS
    if ecut <= 0.0:
        raise ValueError(f"non-positive cutoff: ecut={ecut}")
    if e_max <= ecut:
        raise ValueError(f"e_max={e_max} MeV does not cover the transport range")
    if n_points < 2:
        raise ValueError(f"need at least two grid nodes, got {n_points}")

    n_materials = source.n_materials
    # Must stay the electron grid of build_cross_section_tables (test-pinned).
    energies = np.geomspace(0.5 * ecut, e_max, n_points)
    bhabha = np.empty((n_materials, n_points))
    annihilation = np.empty((n_materials, n_points))
    restricted_range = np.empty((n_materials, n_points))
    for material in range(n_materials):
        for j, energy in enumerate(energies):
            e = float(energy)
            bhabha[material, j] = source.bhabha_cross_section(e, material, ecut)
            annihilation[material, j] = source.annihilation_cross_section(e, material)
            restricted_range[material, j] = source.positron_restricted_range(e, material, ecut)

    # The inverse on one shared log-range grid, as for the electron
    # (build_cross_section_tables): per-material values clamp into [ecut, e_max].
    e_top = float(energies[-1])
    r_top = max(float(restricted_range[m, -1]) for m in range(n_materials))
    range_nodes = np.geomspace(r_top * _RANGE_R_MIN_FRACTION, r_top, n_points)
    energy_of_restricted_range = np.empty((n_materials, n_points))
    for material in range(n_materials):
        r_top_mat = float(restricted_range[material, -1])
        for j, r in enumerate(range_nodes):
            energy_of_restricted_range[material, j] = source.positron_energy_after_mass_path(
                e_top, material, ecut, r_top_mat - float(r)
            )

    log_min = float(np.log(energies[0]))
    log_max = float(np.log(energies[-1]))
    return PositronTables(
        n_points=n_points,
        ecut=ecut,
        e_max=e_max,
        electron_log_e_min=log_min,
        electron_inv_dlog=(n_points - 1) / (log_max - log_min),
        bhabha=bhabha,
        annihilation=annihilation,
        restricted_range=restricted_range,
        energy_of_restricted_range=energy_of_restricted_range,
        range_log_r_min=float(np.log(range_nodes[0])),
        range_inv_dlog=(n_points - 1) / float(np.log(range_nodes[-1]) - np.log(range_nodes[0])),
    )
