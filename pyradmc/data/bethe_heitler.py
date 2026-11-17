r"""Screened Bethe-Heitler bremsstrahlung: screening data, normalization, tables.

The host-side data half of the screened Bethe-Heitler model whose sampling half is
:mod:`pyradmc.physics.brems_bh` (governing equation, references and the list of
stated approximations are there). Nothing here is called from a kernel; kernels see
the flattened :class:`BetheHeitlerTables`.

Normalization
-------------
The model supplies the *shape* of the photon spectrum, ``dsigma/dk`` proportional to
``G(k)/k``; its absolute scale is fixed by requiring the radiated energy to equal
the radiative stopping power the ``CrossSectionSource`` already answers with
(ESTAR/ICRU-37 totals, Seltzer & Berger's data),

.. math::

    N(T) \int_0^{T} G(k)\, dk = \frac{S_{rad}(T)}{\rho} ,

which is the role of the empirical factor ``A'(E)`` in EGS4 (Nelson, Hirayama &
Rogers, SLAC-265 (1985), sec. 2.7, doi:10.2172/1453993). With a photon threshold
``k_cut`` the radiative loss then splits exactly into

.. math::

    \frac{\sigma(T; k_{cut})}{\rho} = N \int_{k_{cut}}^{T} \frac{G(k)}{k}\, dk,
    \qquad
    \frac{S_{rad}^{<}(T; k_{cut})}{\rho} = N \int_0^{k_{cut}} G(k)\, dk,

a discrete cross-section for photons above the threshold and a restricted radiative
stopping power for the sub-threshold remainder, with the identity

.. math::

    \sigma \langle k \rangle + S_{rad}^{<} = S_{rad},
    \qquad \langle k \rangle = \frac{\int_{k_{cut}}^{T} G\, dk}
                                    {\int_{k_{cut}}^{T} G/k\, dk},

test-pinned (``tests/unit/test_bethe_heitler_data.py``). The un-normalized Born
value, :func:`born_radiative_stopping_power`, is kept as a diagnostic: its ratio to
ESTAR is within 3 percent of one at 10 and 20 MeV in water, air and cortical bone
and within 7 percent in tungsten, and within 2 percent of one from 30 MeV to 1 GeV
in every registry medium (measured values in the test).

Compounds
---------
See :func:`screening_parameters`.

All energies in MeV; cross-sections per unit mass in cm^2/g; mass stopping powers
in MeV cm^2/g.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data.berger_seltzer import CLASSICAL_ELECTRON_RADIUS_CM
from pyradmc.data.materials import AVOGADRO, STANDARD_ATOMIC_WEIGHT
from pyradmc.data.screening import (
    FINE_STRUCTURE_CONSTANT,
    coulomb_correction,
    radiation_logarithms,
)
from pyradmc.data.screening import TSAI_PHI1_COMPLETE as _PHI1_COMPLETE
from pyradmc.data.screening import TSAI_PSI1_COMPLETE as _PSI1_COMPLETE

if TYPE_CHECKING:
    from pyradmc.data.interface import CrossSectionSource

__all__ = [
    "FINE_STRUCTURE_CONSTANT",
    "BetheHeitlerTables",
    "BremsScreening",
    "born_radiative_stopping_power",
    "build_bethe_heitler_tables",
    "coulomb_correction",
    "screening_parameters",
    "shape_array",
    "shape_integrals",
]

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
_PANELS_PER_DECADE: int = 4
_PANEL_DECADES: int = 12


class BremsScreening(NamedTuple):
    """Per-material parameters of the screened Bethe-Heitler shape.

    The first five are the arguments of the functions in
    :mod:`pyradmc.physics.brems_bh`, in their order; all are dimensionless except
    the last.

    Attributes
    ----------
    z_ratio
        ``r = sum(n_i Z_i) / sum(n_i Z_i^2)``, the weight of the atomic-electron
        terms relative to the nuclear ones (``1/Z`` for an element).
    c_nuc
        Constant subtracted from ``phi_1 / 4``: the ``Z^2``-weighted mean of
        ``ln(Z)/3 + f(Z)``.
    c_el
        Constant subtracted from ``psi_1 / 4``: the ``Z``-weighted mean of
        ``2 ln(Z) / 3``.
    g_nuc
        Scale of the nuclear screening variable: the ``Z^2``-weighted geometric
        mean of ``Z^(-1/3)``.
    g_el
        Scale of the electron screening variable: the ``Z``-weighted geometric
        mean of ``Z^(-2/3)``.
    z2_per_gram
        ``sum(w_i Z_i^2 / A_i)`` in mol/g. Not needed by the normalized model;
        it restores the absolute Born cross-section
        (:func:`born_radiative_stopping_power`).
    """

    z_ratio: float
    c_nuc: float
    c_el: float
    g_nuc: float
    g_el: float
    z2_per_gram: float


def screening_parameters(composition: tuple[tuple[int, float], ...]) -> BremsScreening:
    r"""Collapse an elemental composition into one set of screening parameters.

    The exact cross-section of a compound is the atom-number-weighted sum of the
    elemental ones, each with its own screening variables. A kernel cannot afford a
    loop over up to thirteen elements per rejection round, so — as EGS4 does with
    its ``Z_G``, ``Z_A`` and ``Z_U`` averages (Nelson, Hirayama & Rogers, SLAC-265
    (1985), sec. 2.7, doi:10.2172/1453993) — the sum is replaced by the elemental
    form evaluated with averaged parameters. With atom fractions
    :math:`n_i \propto w_i / A_i`:

    .. math::

        r = \frac{\sum n_i Z_i}{\sum n_i Z_i^2}, \quad
        c_n = \left\langle \tfrac14 \phi_1(0) - (L_{el,i} - f_i)
              \right\rangle_{Z^2}, \quad
        c_e = \left\langle \tfrac14 \psi_1(0) - L_{inel,i} \right\rangle_{Z}, \quad
        \ln g_n = \left\langle -\tfrac13 \ln Z_i \right\rangle_{Z^2}, \quad
        \ln g_e = \left\langle -\tfrac23 \ln Z_i \right\rangle_{Z},

    where :math:`\langle\cdot\rangle_{Z^2}` and :math:`\langle\cdot\rangle_{Z}` are
    means weighted by :math:`n_i Z_i^2` and :math:`n_i Z_i`. The radiation logarithms
    are :math:`L_{el} = \ln(184.15 Z^{-1/3})` and
    :math:`L_{inel} = \ln(1194 Z^{-2/3})` for :math:`Z \ge 5`, so that there
    :math:`c_n = \langle \tfrac13 \ln Z + f \rangle` and
    :math:`c_e = \langle \tfrac23 \ln Z \rangle`.

    Properties of the rule (all test-pinned): it is exact for an element; it is
    exact for any compound in the complete-screening limit (the additive constants
    average linearly) and in the unscreened limit (the geometric mean of the
    screening scale is the one that makes the nuclear charge cancel); in between,
    the measured deviation from the element-by-element sum over 0.2-300 MeV is
    at most 0.17 percent of the shape in every registry material.

    Stated approximation for hydrogen to beryllium: the Thomas-Fermi model is not
    valid for :math:`Z < 5`, and Tsai tabulates :math:`L_{el}` and :math:`L_{inel}`
    from better atomic form factors instead. Those tabulated values are used for the
    complete-screening constants, while the *dependence* on the screening variable
    keeps the Thomas-Fermi functional form — i.e. the fitted functions are shifted
    by a constant. Of the registry elements only hydrogen is affected, and it
    carries about five percent of the bremsstrahlung of water.

    Parameters
    ----------
    composition
        ``((Z, mass_fraction), ...)`` as in
        :class:`~pyradmc.data.materials.MaterialData`; weights come from
        :data:`~pyradmc.data.materials.STANDARD_ATOMIC_WEIGHT`.
    """
    if not composition:
        raise ValueError("empty composition")
    sum_z = 0.0
    sum_z2 = 0.0
    sum_c_nuc = 0.0
    sum_c_el = 0.0
    sum_log_g_nuc = 0.0
    sum_log_g_el = 0.0
    for z, fraction in composition:
        if z < 1 or fraction < 0.0:
            raise ValueError(f"invalid composition entry (Z={z}, w={fraction})")
        atoms = fraction / STANDARD_ATOMIC_WEIGHT[z]
        log_z = math.log(z)
        l_el, l_inel = radiation_logarithms(z)
        weight_nuc = atoms * z * z
        weight_el = atoms * z
        sum_z += weight_el
        sum_z2 += weight_nuc
        sum_c_nuc += weight_nuc * (0.25 * _PHI1_COMPLETE - l_el + coulomb_correction(z))
        sum_c_el += weight_el * (0.25 * _PSI1_COMPLETE - l_inel)
        sum_log_g_nuc += weight_nuc * (-log_z / 3.0)
        sum_log_g_el += weight_el * (-2.0 * log_z / 3.0)
    if sum_z2 <= 0.0:
        raise ValueError("composition has no mass")
    return BremsScreening(
        z_ratio=sum_z / sum_z2,
        c_nuc=sum_c_nuc / sum_z2,
        c_el=sum_c_el / sum_z,
        g_nuc=math.exp(sum_log_g_nuc / sum_z2),
        g_el=math.exp(sum_log_g_el / sum_z),
        z2_per_gram=sum_z2,
    )


def shape_array(
    energy: float, photon_energies: npt.NDArray[np.float64], screening: BremsScreening
) -> npt.NDArray[np.float64]:
    """Vectorized :func:`pyradmc.physics.brems_bh.bethe_heitler_shape`, for quadrature.

    A NumPy transcription of the scalar physics function. The scalar one is what
    samples and what the kernels compile; this one exists only so the host can
    integrate it cheaply. Their agreement to rounding is test-pinned, because the
    stopping-power identity is only as good as it.
    """
    value = _unclamped_shape(energy, np.asarray(photon_energies, dtype=np.float64), screening)
    return np.asarray(np.maximum(value, 0.0), dtype=np.float64)


def _unclamped_shape(
    energy: float, k: npt.NDArray[np.float64], screening: BremsScreening
) -> npt.NDArray[np.float64]:
    """Evaluate the shape before its clamp at zero (negative where the clamp acts)."""
    total = energy + ELECTRON_MASS_MEV
    reduced = 100.0 * ELECTRON_MASS_MEV * k / (total * (total - k))
    gamma = reduced * screening.g_nuc
    epsilon = reduced * screening.g_el

    phi1 = (
        _PHI1_COMPLETE
        - 2.0 * np.log(1.0 + (0.55846 * gamma) ** 2)
        - 4.0 * (1.0 - 0.6 * np.exp(-0.9 * gamma) - 0.4 * np.exp(-1.5 * gamma))
    )
    phi_difference = (2.0 / 3.0) / (1.0 + 6.5 * gamma + 6.0 * gamma * gamma)
    psi1 = (
        _PSI1_COMPLETE
        - 2.0 * np.log(1.0 + (3.621 * epsilon) ** 2)
        - 4.0 * (1.0 - 0.7 * np.exp(-8.0 * epsilon) - 0.3 * np.exp(-29.2 * epsilon))
    )
    psi_difference = (2.0 / 3.0) / (1.0 + 40.0 * epsilon + 400.0 * epsilon * epsilon)

    main = 0.25 * phi1 - screening.c_nuc + screening.z_ratio * (0.25 * psi1 - screening.c_el)
    tip = 0.125 * (phi_difference + screening.z_ratio * psi_difference)
    y = k / total
    return np.asarray((1.0 - y + 0.75 * y * y) * main + (1.0 - y) * tip, dtype=np.float64)


def _quadrature(
    lo: float, hi: float, energy: float, screening: BremsScreening
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Gauss-Legendre nodes and weights on ``[lo, hi]`` within ``[0, T]``.

    Composite 16-point rule on panels whose edges are geometric in the photon
    energy (the screening variable spans many decades in ``k``) *and* geometric in
    the final total energy ``E - k`` (it varies on the scale of the electron mass
    at the tip, which a grid in ``k`` alone does not resolve at 300 MeV). Where the
    shape is clamped at zero (high Z, near the tip) the zero crossing is located by
    bisection and made a panel edge, so the integrand is analytic inside every
    panel and the rule converges to rounding; pinned against adaptive quadrature.
    """
    total = energy + ELECTRON_MASS_MEV
    steps = np.arange(_PANELS_PER_DECADE * _PANEL_DECADES + 1)
    k_edges = energy * 10.0 ** (-steps / _PANELS_PER_DECADE)
    n_tip = max(1, math.ceil(8.0 * math.log10(total / ELECTRON_MASS_MEV)))
    tip_edges = total - ELECTRON_MASS_MEV * (total / ELECTRON_MASS_MEV) ** (
        np.arange(n_tip + 1) / n_tip
    )
    edges = np.concatenate(([lo, hi], k_edges, tip_edges))
    edges = np.unique(edges[(edges >= lo) & (edges <= hi)])

    signs = _unclamped_shape(energy, edges, screening) > 0.0
    crossings = []
    for i in np.flatnonzero(signs[1:] != signs[:-1]):
        a, b = float(edges[i]), float(edges[i + 1])
        positive_at_a = bool(signs[i])
        for _ in range(60):
            mid = 0.5 * (a + b)
            if bool(_unclamped_shape(energy, np.array([mid]), screening)[0] > 0.0) == positive_at_a:
                a = mid
            else:
                b = mid
        crossings.append(0.5 * (a + b))
    if crossings:
        edges = np.unique(np.concatenate((edges, crossings)))

    half = 0.5 * np.diff(edges)
    mid_points = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid_points[:, None] + half[:, None] * _GL_NODES[None, :]).ravel()
    weights = (half[:, None] * _GL_WEIGHTS[None, :]).ravel()
    return nodes, weights


def shape_integrals(
    energy: float, k_cut: float, screening: BremsScreening
) -> tuple[float, float, float]:
    r"""Integrate the shape three ways, as the normalization needs.

    Returns ``(below, above, count)``:

    .. math::

        \text{below} = \int_0^{\min(k_{cut}, T)} G\, dk, \quad
        \text{above} = \int_{k_{cut}}^{T} G\, dk, \quad
        \text{count} = \int_{k_{cut}}^{T} \frac{G}{k}\, dk,

    in MeV, MeV and dimensionless; ``above`` and ``count`` are zero for
    ``T <= k_cut``. ``below + above`` is the full radiated-energy integral, so the
    split of the stopping power is exact by construction rather than to
    quadrature accuracy.

    Parameters
    ----------
    energy
        Electron kinetic energy T in MeV.
    k_cut
        Photon threshold in MeV.
    screening
        The material's :class:`BremsScreening`.
    """
    if energy <= 0.0:
        raise ValueError(f"non-positive electron kinetic energy {energy} MeV")
    if k_cut <= 0.0:
        raise ValueError(f"non-positive photon threshold {k_cut} MeV")
    nodes, weights = _quadrature(0.0, min(k_cut, energy), energy, screening)
    below = float(np.dot(weights, shape_array(energy, nodes, screening)))
    if energy <= k_cut:
        return (below, 0.0, 0.0)
    nodes, weights = _quadrature(k_cut, energy, energy, screening)
    shape = shape_array(energy, nodes, screening)
    above = float(np.dot(weights, shape))
    count = float(np.dot(weights, shape / nodes))
    return (below, above, count)


def born_radiative_stopping_power(energy: float, screening: BremsScreening) -> float:
    r"""Un-normalized radiative stopping power of the model itself, in MeV cm^2/g.

    .. math::

        \frac{S_{rad}^{Born}}{\rho} = \tfrac{16}{3} \alpha r_e^2 N_A
            \sum_i \frac{w_i Z_i^2}{A_i} \int_0^{T} G(k)\, dk .

    A diagnostic, **not** a transport quantity: the engine's radiative stopping
    power is the ``CrossSectionSource``'s, and the ratio of that to this value is
    the normalization the model applies (EGS4's ``A'``). Reported so the size of
    the rescaling — the model's absolute error — is visible rather than absorbed.
    """
    below, above, _ = shape_integrals(energy, energy, screening)
    prefactor = (
        16.0 / 3.0 * FINE_STRUCTURE_CONSTANT * CLASSICAL_ELECTRON_RADIUS_CM**2 * AVOGADRO
    ) * screening.z2_per_gram
    return prefactor * (below + above)


@dataclass(frozen=True)
class BetheHeitlerTables:
    """Kernel-side data of the screened Bethe-Heitler model; frozen after construction.

    Laid on the *same* electron log-energy grid as
    :class:`pyradmc.data.tables.CrossSectionTables` built with the same
    ``(ecut, e_max, n_points)``, so a kernel interpolates both with one metadata
    pair and :func:`pyradmc.data.tables.lookup_loglinear_2d`. Kept apart from that
    class because the model is optional: a run that does not select it should
    neither pay for these integrals nor upload them.

    Attributes
    ----------
    n_points, ecut, pcut, e_max
        Grid size and the cutoffs (MeV) the tables were built at; ``pcut`` is the
        photon threshold ``k_cut`` of both restricted quantities.
    electron_log_e_min, electron_inv_dlog
        Lookup metadata, identical to the cross-section tables' electron grid.
    cross_section
        Discrete bremsstrahlung cross-section for ``k > pcut`` per unit mass, in
        cm^2/g, ``(n_materials, n_points)``; zero at and below ``pcut``.
    stopping_radiative_restricted
        Radiative stopping power restricted to ``k < pcut``, in MeV cm^2/g,
        ``(n_materials, n_points)``.
    screening
        ``(n_materials, 5)``: the ``(z_ratio, c_nuc, c_el, g_nuc, g_el)`` arguments
        of the sampling functions, per material.
    """

    n_points: int
    ecut: float
    pcut: float
    e_max: float
    electron_log_e_min: float
    electron_inv_dlog: float
    cross_section: npt.NDArray[np.float64]
    stopping_radiative_restricted: npt.NDArray[np.float64]
    screening: npt.NDArray[np.float64]


def build_bethe_heitler_tables(
    source: CrossSectionSource,
    ecut: float,
    pcut: float,
    e_max: float,
    n_points: int | None = None,
) -> BetheHeitlerTables:
    """Flatten the bremsstrahlung queries of a source onto the electron grid.

    Generic over sources, like
    :func:`pyradmc.data.tables.build_cross_section_tables`: everything goes through
    :meth:`~pyradmc.data.interface.CrossSectionSource.bremsstrahlung_cross_section`,
    :meth:`~pyradmc.data.interface.CrossSectionSource.restricted_radiative_stopping_power`
    and :meth:`~pyradmc.data.interface.CrossSectionSource.bremsstrahlung_screening`.

    Parameters
    ----------
    source
        The host-side data source to flatten.
    ecut, pcut
        Electron and photon cutoffs in MeV (accuracy-defining, AGENTS.md 2.8). The
        grid starts at ``ecut / 2`` like the electron tables; ``pcut`` is the
        photon threshold of the discrete channel.
    e_max
        Upper grid edge in MeV. Refused above the source's declared energy
        domain (:attr:`~pyradmc.data.interface.CrossSectionSource.energy_max`),
        like the cross-section tables: the normalization is the source's
        radiative stopping power, which must not be extrapolated.
    n_points
        Grid nodes; defaults to :data:`pyradmc.data.tables.TABLE_POINTS`.
    """
    from pyradmc.data.interface import require_energy_domain
    from pyradmc.data.tables import TABLE_POINTS

    require_energy_domain(source, e_max)
    if n_points is None:
        n_points = TABLE_POINTS
    if pcut <= 0.0 or ecut <= 0.0:
        raise ValueError(f"non-positive cutoff: pcut={pcut}, ecut={ecut}")
    if e_max <= max(pcut, ecut):
        raise ValueError(f"e_max={e_max} MeV does not cover the transport range")
    if n_points < 2:
        raise ValueError(f"need at least two grid nodes, got {n_points}")

    n_materials = source.n_materials
    # Must stay the electron grid of build_cross_section_tables (test-pinned).
    energies = np.geomspace(0.5 * ecut, e_max, n_points)
    cross_section = np.empty((n_materials, n_points))
    restricted = np.empty((n_materials, n_points))
    screening = np.empty((n_materials, 5))
    for material in range(n_materials):
        screening[material] = source.bremsstrahlung_screening(material)[:5]
        for j, energy in enumerate(energies):
            e = float(energy)
            cross_section[material, j] = source.bremsstrahlung_cross_section(e, material, pcut)
            restricted[material, j] = source.restricted_radiative_stopping_power(e, material, pcut)

    log_min = float(np.log(energies[0]))
    log_max = float(np.log(energies[-1]))
    return BetheHeitlerTables(
        n_points=n_points,
        ecut=ecut,
        pcut=pcut,
        e_max=e_max,
        electron_log_e_min=log_min,
        electron_inv_dlog=(n_points - 1) / (log_max - log_min),
        cross_section=cross_section,
        stopping_radiative_restricted=restricted,
        screening=screening,
    )
