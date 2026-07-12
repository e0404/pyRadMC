"""EPDL -> canonical photon cross-section conversion (Phase 5, tabulated, slice B).

Exercises the deterministic conversion pipeline on synthetic ENDF-6 text (no
committed data, same authoring style as ``test_endf.py``): the MT->channel mapping,
the eV->MeV and barns->cm^2/g unit conversion, mass-fraction mixing of elements into
a material, and the log-log resample onto a query grid. The accuracy claim against
real EPDL data and NIST XCOM lives in the opt-in validation tier
(``tests/validation/test_epdl_water_nist.py``), which needs the ~90 MB library file.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.data.interface import PhotonProcess
from pyRadMC.data.materials import AVOGADRO
from pyRadMC.data.tabulated.epdl import (
    STANDARD_ATOMIC_WEIGHT,
    element_coherent_form_factor,
    element_photon_channels,
    mass_fractions_from_formula,
    material_form_factor_squared,
    material_mu_over_rho,
)

# -- synthetic ENDF authoring (mirrors tests/unit/test_endf.py) -----------------


def _line(fields: list[str], mat: int, mf: int, mt: int, num: int) -> str:
    """One 80-column ENDF record: six 11-char fields + MAT/MF/MT/line-number."""
    padded = (fields + [""] * 6)[:6]
    body = "".join(f"{f:<11}"[:11] for f in padded)
    return f"{body}{mat:>4}{mf:>2}{mt:>3}{num:>5}"


def _section(mat: int, mt: int, e_ev: list[float], barns: list[float]) -> list[str]:
    """A HEAD + TAB1 (single region) MF=23 section for one (element, channel)."""
    za = f"{mat // 100 * 1000:.1f}"  # ZA = 1000 * Z; MAT = 100 * Z in EPDL
    head = _line([za, "0.0", "0", "0", "0", "0"], mat, 23, mt, 1)
    ctrl = _line(["0.0", "0.0", "0", "0", "1", str(len(e_ev))], mat, 23, mt, 2)
    interp = _line([str(len(e_ev)), "5"], mat, 23, mt, 3)  # INT=5 (log-log)
    pairs: list[str] = []
    for x, y in zip(e_ev, barns, strict=True):
        pairs += [f"{x:.5E}", f"{y:.5E}"]  # 11-column ENDF fields
    # up to three (x, y) pairs per line
    data = [_line(pairs[i : i + 6], mat, 23, mt, 4 + i // 6) for i in range(0, len(pairs), 6)]
    send = _line(["0.0", "0.0", "0", "0", "0", "0"], mat, 23, 0, 99999)
    return [head, ctrl, interp, *data, send]


def _mf27_section(mat: int, x: list[float], form_factor: list[float]) -> list[str]:
    """A HEAD + TAB1 MF=27/MT=502 coherent form-factor section (x [1/A], F)."""
    za = f"{mat // 100 * 1000:.1f}"
    n = str(len(x))
    head = _line([za, "0.0", "0", "0", "0", "0"], mat, 27, 502, 1)
    ctrl = _line(["0.0", "1.0", "0", "0", "1", n], mat, 27, 502, 2)
    interp = _line([n, "2"], mat, 27, 502, 3)
    pairs: list[str] = []
    for xi, fi in zip(x, form_factor, strict=True):
        pairs += [f"{xi:.5E}", f"{fi:.5E}"]
    data = [_line(pairs[i : i + 6], mat, 27, 502, 4 + i // 6) for i in range(0, len(pairs), 6)]
    send = _line(["0.0", "0.0", "0", "0", "0", "0"], mat, 27, 0, 99999)
    return [head, ctrl, interp, *data, send]


def _barns_to_mu_over_rho(barns: float, z: int) -> float:
    return barns * 1.0e-24 * AVOGADRO / STANDARD_ATOMIC_WEIGHT[z]


# -- tests ----------------------------------------------------------------------


def test_element_channels_map_mt_and_convert_units() -> None:
    """MT numbers map to channels; energies go eV->MeV and barns->cm^2/g."""
    e_ev = [1.0e5, 1.0e6, 1.0e7]  # -> 0.1, 1.0, 10.0 MeV
    lines = (
        _section(100, 504, e_ev, [0.60, 0.30, 0.10])  # H incoherent (Compton)
        + _section(100, 522, e_ev, [1.0e-3, 1.0e-5, 1.0e-7])  # H photoionization
    )
    channels = element_photon_channels("\n".join(lines))

    assert set(channels) == {1}  # Z = MAT // 100
    assert set(channels[1]) == {PhotonProcess.COMPTON, PhotonProcess.PHOTOELECTRIC}

    e_mev, mu = channels[1][PhotonProcess.COMPTON]
    np.testing.assert_allclose(e_mev, [0.1, 1.0, 10.0], rtol=1e-9)
    np.testing.assert_allclose(
        mu, [_barns_to_mu_over_rho(b, 1) for b in (0.60, 0.30, 0.10)], rtol=1e-9
    )


def test_mass_fractions_of_water() -> None:
    """Water mass fractions come from the atomic-weight table and sum to one."""
    fractions = mass_fractions_from_formula({1: 2, 8: 1})
    assert fractions[1] == pytest.approx(0.1119, abs=5e-4)
    assert fractions[8] == pytest.approx(0.8881, abs=5e-4)
    assert sum(fractions.values()) == pytest.approx(1.0, rel=1e-12)


def test_material_mix_is_mass_weighted_sum_on_shared_grid() -> None:
    """On a grid equal to the native points, mixing is the exact mass-weighted sum."""
    e_ev = [1.0e6, 5.0e6, 1.0e7]  # 1, 5, 10 MeV
    grid = np.array([1.0, 5.0, 10.0])
    lines = (
        _section(100, 504, e_ev, [0.30, 0.12, 0.09])  # H Compton
        + _section(800, 504, e_ev, [2.40, 0.96, 0.72])  # O Compton
    )
    elements = element_photon_channels("\n".join(lines))
    fractions = mass_fractions_from_formula({1: 2, 8: 1})
    mixed = material_mu_over_rho(elements, fractions, grid)

    mu_h = np.array([_barns_to_mu_over_rho(b, 1) for b in (0.30, 0.12, 0.09)])
    mu_o = np.array([_barns_to_mu_over_rho(b, 8) for b in (2.40, 0.96, 0.72)])
    expected = fractions[1] * mu_h + fractions[8] * mu_o
    np.testing.assert_allclose(mixed[PhotonProcess.COMPTON], expected, rtol=1e-9)


def test_pair_channel_is_zero_below_threshold() -> None:
    """A channel that starts above a grid point contributes zero there (pair threshold)."""
    # Pair present only from 2 MeV up; the 1 MeV grid point is below its support.
    e_ev = [2.0e6, 5.0e6, 1.0e7]
    grid = np.array([1.0, 5.0, 10.0])
    lines = _section(100, 516, e_ev, [0.01, 0.05, 0.09])  # H pair
    elements = element_photon_channels("\n".join(lines))
    fractions = mass_fractions_from_formula({1: 2})  # single element for a clean check
    mixed = material_mu_over_rho(elements, fractions, grid)

    pair = mixed[PhotonProcess.PAIR]
    assert pair[0] == 0.0  # below the tabulated threshold
    assert pair[1] > 0.0 and pair[2] > 0.0


def test_coherent_form_factor_read_per_element() -> None:
    """MF=27/MT=502 form factors are keyed by Z with F(0)=Z; other MF ignored."""
    lines = _mf27_section(100, [0.0, 1.0, 2.0], [1.0, 0.5, 0.2]) + _mf27_section(
        800, [0.0, 1.0, 2.0], [8.0, 4.0, 1.0]
    )
    ff = element_coherent_form_factor("\n".join(lines))
    assert set(ff) == {1, 8}
    x_h, f_h = ff[1]
    np.testing.assert_allclose(x_h, [0.0, 1.0, 2.0], rtol=1e-5)
    np.testing.assert_allclose(f_h, [1.0, 0.5, 0.2], rtol=1e-5)  # F(0) = Z = 1


def test_material_form_factor_squared_is_atom_weighted_sum() -> None:
    """F^2_material(x) = sum_i n_i F_i^2(x), on a grid of the native positive abscissae."""
    lines = _mf27_section(100, [0.0, 1.0, 2.0], [1.0, 0.5, 0.2]) + _mf27_section(
        800, [0.0, 1.0, 2.0], [8.0, 4.0, 1.0]
    )
    elements = element_coherent_form_factor("\n".join(lines))
    grid = np.array([1.0, 2.0])  # positive native abscissae -> interpolation is identity
    f2 = material_form_factor_squared(elements, {1: 2, 8: 1}, grid)
    # 2 * F_H^2 + 1 * F_O^2 = 2*[0.25, 0.04] + [16, 1] = [16.5, 1.08]
    np.testing.assert_allclose(f2, [16.5, 1.08], rtol=1e-6)
