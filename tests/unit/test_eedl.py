"""EEDL -> radiative stopping power (Phase 5, tabulated, slice B, electrons).

Synthetic ENDF text (no committed data): a 23/527 bremsstrahlung cross section plus a
26/527 secondary-distribution section carrying a LAW=1 photon-spectrum subsection (to
be skipped) and a LAW=8 electron energy-transfer subsection (the average radiated
energy per event). Exercises the MF=26 walker and the ``sigma * <E_loss>`` radiative
stopping conversion and its mass-fraction mixing. The accuracy claim against real EEDL
and NIST ESTAR lives in the opt-in validation tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.data.materials import AVOGADRO
from pyRadMC.data.tabulated.eedl import (
    element_collision_stopping,
    element_radiative_stopping,
    element_scattering_power,
    material_radiative_stopping,
    material_scattering_power,
)
from pyRadMC.data.tabulated.endf import (
    read_mf26_angular_distributions,
    read_mf26_energy_transfer,
    read_mf26_spectra,
)
from pyRadMC.data.tabulated.epdl import STANDARD_ATOMIC_WEIGHT, mass_fractions_from_formula


def _line(fields: list[str], mat: int, mf: int, mt: int, num: int) -> str:
    padded = (fields + [""] * 6)[:6]
    body = "".join(f"{f:<11}"[:11] for f in padded)
    return f"{body}{mat:>4}{mf:>2}{mt:>3}{num:>5}"


def _xs_527(mat: int, e_ev: list[float], barns: list[float]) -> list[str]:
    """A HEAD + TAB1 MF=23/MT=527 bremsstrahlung cross section (barns)."""
    za = f"{mat // 100 * 1000:.1f}"
    n = str(len(e_ev))
    head = _line([za, "0.0", "0", "0", "0", "0"], mat, 23, 527, 1)
    ctrl = _line(["0.0", "0.0", "0", "0", "1", n], mat, 23, 527, 2)
    interp = _line([n, "2"], mat, 23, 527, 3)
    pairs: list[str] = []
    for x, y in zip(e_ev, barns, strict=True):
        pairs += [f"{x:.5E}", f"{y:.5E}"]
    data = [_line(pairs[i : i + 6], mat, 23, 527, 4 + i // 6) for i in range(0, len(pairs), 6)]
    send = _line(["0.0", "0.0", "0", "0", "0", "0"], mat, 23, 0, 99999)
    return [head, ctrl, interp, *data, send]


def _dist_527(mat: int, e_ev: list[float], et_ev: list[float]) -> list[str]:
    """MF=26/MT=527: a LAW=1 photon-spectrum subsection then a LAW=8 electron one."""
    za = f"{mat // 100 * 1000:.1f}"
    n = str(len(e_ev))
    ln = 1
    out: list[str] = []

    def add(fields: list[str]) -> None:
        nonlocal ln
        out.append(_line(fields, mat, 26, 527, ln))
        ln += 1

    add([za, "0.0", "0", "0", "2", "0"])  # HEAD: NK = 2 subsections
    # -- subsection 1: photon, LAW=1 (one incident energy, a 2-point spectrum) --
    add(["0.0", "0.0", "0", "1", "1", "2"])  # multiplicity TAB1 (LAW=1)
    add(["2", "2"])
    add(["10.0", "1.0", "1.0E+11", "1.0"])
    add(["0.0", "0.0", "0", "0", "1", "1"])  # LAW=1 TAB2: NR=1, NE=1
    add(["1", "2"])
    add(["0.0", "10.0", "0", "0", "4", "2"])  # list: NW=4, NEP=2 at E=10 eV
    add(["0.05", "1.0", "0.1", "0.5"])
    # -- subsection 2: electron, LAW=8 energy transfer (the target table) --
    add(["11.0", "0.0", "0", "8", "1", "2"])  # multiplicity TAB1 (LAW=8)
    add(["2", "2"])
    add(["10.0", "1.0", "1.0E+11", "1.0"])
    add(["0.0", "0.0", "0", "0", "1", n])  # LAW=8 TAB1: NR=1, NP=len
    add([n, "2"])
    pairs: list[str] = []
    for x, y in zip(e_ev, et_ev, strict=True):
        pairs += [f"{x:.5E}", f"{y:.5E}"]
    for i in range(0, len(pairs), 6):
        add(pairs[i : i + 6])
    out.append(_line(["0.0", "0.0", "0", "0", "0", "0"], mat, 26, 0, 99999))  # SEND
    return out


def _xs_525(mat: int, e_ev: list[float], barns: list[float]) -> list[str]:
    """A HEAD + TAB1 MF=23/MT=525 large-angle elastic cross section (barns)."""
    za = f"{mat // 100 * 1000:.1f}"
    n = str(len(e_ev))
    head = _line([za, "0.0", "0", "0", "0", "0"], mat, 23, 525, 1)
    ctrl = _line(["0.0", "0.0", "0", "0", "1", n], mat, 23, 525, 2)
    interp = _line([n, "2"], mat, 23, 525, 3)
    pairs: list[str] = []
    for x, y in zip(e_ev, barns, strict=True):
        pairs += [f"{x:.5E}", f"{y:.5E}"]
    data = [_line(pairs[i : i + 6], mat, 23, 525, 4 + i // 6) for i in range(0, len(pairs), 6)]
    send = _line(["0.0", "0.0", "0", "0", "0", "0"], mat, 23, 0, 99999)
    return [head, ctrl, interp, *data, send]


def _mf26_tab2(
    mat: int, mt: int, law: int, dists: list[tuple[float, list[float], list[float]]]
) -> list[str]:
    """A single MF=26 subsection with the given LAW and per-energy (x, y) tables."""
    za = f"{mat // 100 * 1000:.1f}"
    ne = str(len(dists))
    ln = 1
    out: list[str] = []

    def add(fields: list[str]) -> None:
        nonlocal ln
        out.append(_line(fields, mat, 26, mt, ln))
        ln += 1

    add([za, "0.0", "0", "0", "1", "0"])  # HEAD: NK = 1
    add(["11.0", "0.0", "0", str(law), "1", "2"])  # multiplicity TAB1
    add(["2", "2"])
    add(["10.0", "1.0", "1.0E+11", "1.0"])
    add(["0.0", "0.0", "0", "0", "1", ne])  # LAW TAB2: NR=1, NE
    add([ne, "2"])
    for energy, x, y in dists:
        n_pts = len(x)
        add(["0.0", f"{energy:.5E}", "0", "0", str(2 * n_pts), str(n_pts)])
        pairs: list[str] = []
        for xi, yi in zip(x, y, strict=True):
            pairs += [f"{xi:.6f}", f"{yi:.5E}"]
        for i in range(0, len(pairs), 6):
            add(pairs[i : i + 6])
    out.append(_line(["0.0", "0.0", "0", "0", "0", "0"], mat, 26, 0, 99999))  # SEND
    return out


def _ang_525(mat: int, dists: list[tuple[float, list[float], list[float]]]) -> list[str]:
    """MF=26/MT=525: a single LAW=2 subsection with per-energy angular tables P(mu)."""
    return _mf26_tab2(mat, 525, 2, dists)


def _xs(mat: int, mt: int, e_ev: list[float], barns: list[float]) -> list[str]:
    """A HEAD + TAB1 MF=23 cross section for one (element, reaction)."""
    za = f"{mat // 100 * 1000:.1f}"
    n = str(len(e_ev))
    head = _line([za, "0.0", "0", "0", "0", "0"], mat, 23, mt, 1)
    ctrl = _line(["0.0", "0.0", "0", "0", "1", n], mat, 23, mt, 2)
    interp = _line([n, "2"], mat, 23, mt, 3)
    pairs: list[str] = []
    for x, y in zip(e_ev, barns, strict=True):
        pairs += [f"{x:.5E}", f"{y:.5E}"]
    data = [_line(pairs[i : i + 6], mat, 23, mt, 4 + i // 6) for i in range(0, len(pairs), 6)]
    send = _line(["0.0", "0.0", "0", "0", "0", "0"], mat, 23, 0, 99999)
    return [head, ctrl, interp, *data, send]


def _law8(mat: int, mt: int, e_ev: list[float], et_ev: list[float]) -> list[str]:
    """MF=26/MT: a single LAW=8 (energy-transfer) subsection, (E, average loss)."""
    za = f"{mat // 100 * 1000:.1f}"
    n = str(len(e_ev))
    ln = 1
    out: list[str] = []

    def add(fields: list[str]) -> None:
        nonlocal ln
        out.append(_line(fields, mat, 26, mt, ln))
        ln += 1

    add([za, "0.0", "0", "0", "1", "0"])  # HEAD: NK = 1
    add(["11.0", "0.0", "0", "8", "1", "2"])  # multiplicity TAB1 (LAW=8)
    add(["2", "2"])
    add(["10.0", "1.0", "1.0E+11", "1.0"])
    add(["0.0", "0.0", "0", "0", "1", n])  # LAW=8 TAB1: NR=1, NP
    add([n, "2"])
    pairs: list[str] = []
    for x, y in zip(e_ev, et_ev, strict=True):
        pairs += [f"{x:.5E}", f"{y:.5E}"]
    for i in range(0, len(pairs), 6):
        add(pairs[i : i + 6])
    out.append(_line(["0.0", "0.0", "0", "0", "0", "0"], mat, 26, 0, 99999))  # SEND
    return out


def _s_rad(barns: float, z: int, et_ev: float) -> float:
    """sigma[barn] * (N_A/M) * <E_loss>, in MeV cm^2/g (the definition under test)."""
    return barns * 1.0e-24 * AVOGADRO / STANDARD_ATOMIC_WEIGHT[z] * (et_ev * 1.0e-6)


def _s_scat(barns: float, z: int, mean_1mu: float) -> float:
    """2 * (N_A/M) * sigma[barn] * <1-mu>, in rad^2 cm^2/g (the definition under test)."""
    return 2.0 * AVOGADRO / STANDARD_ATOMIC_WEIGHT[z] * barns * 1.0e-24 * mean_1mu


# Two angular distributions on mu in [-1, 1] whose trapezoidal first moment (what the
# code integrates) is exact on the two endpoints: uniform P=[1,1] -> <1-mu> = 2/2 = 1;
# forward P=[1,3] -> <1-mu> = trap[(2)(1),(0)(3)]/trap[1,3] = 2/4 = 0.5.
_UNIFORM = ([-1.0, 1.0], [1.0, 1.0])
_FORWARD = ([-1.0, 1.0], [1.0, 3.0])


def test_mf26_reader_skips_law1_and_returns_law8_transfer() -> None:
    """The walker steps over the LAW=1 spectrum and returns the LAW=8 (E, ET) table."""
    text = "\n".join(_dist_527(100, [1.0e5, 1.0e6, 1.0e7], [1.0e3, 1.0e4, 1.0e5]))
    tables = read_mf26_energy_transfer(text, 527)
    assert set(tables) == {100}
    energy, transfer = tables[100]
    np.testing.assert_allclose(energy, [1.0e5, 1.0e6, 1.0e7], rtol=1e-5)
    np.testing.assert_allclose(transfer, [1.0e3, 1.0e4, 1.0e5], rtol=1e-5)


def test_element_radiative_stopping_is_sigma_times_energy_loss() -> None:
    """Per element: S_rad(E) = sigma_527(E) * (N_A/M) * <E_loss>(E), energies in MeV."""
    e_ev = [1.0e5, 1.0e6, 1.0e7]
    barns = [0.02, 0.03, 0.04]
    et_ev = [1.0e3, 1.0e4, 1.0e5]
    text = "\n".join(_xs_527(100, e_ev, barns) + _dist_527(100, e_ev, et_ev))
    result = element_radiative_stopping(text, elements={1})

    energy_mev, s_rad = result[1]
    np.testing.assert_allclose(energy_mev, [0.1, 1.0, 10.0], rtol=1e-9)
    expected = [_s_rad(b, 1, t) for b, t in zip(barns, et_ev, strict=True)]
    np.testing.assert_allclose(s_rad, expected, rtol=1e-9)


def test_material_radiative_stopping_is_mass_weighted() -> None:
    """Water radiative stopping is the mass-fraction sum of its elements' on a grid."""
    e_ev = [1.0e6, 5.0e6, 1.0e7]
    grid = np.array([1.0, 5.0, 10.0])
    text = "\n".join(
        _xs_527(100, e_ev, [0.02, 0.03, 0.04])
        + _dist_527(100, e_ev, [1.0e4, 5.0e4, 1.0e5])
        + _xs_527(800, e_ev, [0.60, 0.90, 1.20])
        + _dist_527(800, e_ev, [2.0e4, 6.0e4, 1.1e5])
    )
    elements = element_radiative_stopping(text, elements={1, 8})
    fractions = mass_fractions_from_formula({1: 2, 8: 1})
    mixed = material_radiative_stopping(elements, fractions, grid)

    s_h = np.array(
        [_s_rad(b, 1, t) for b, t in zip([0.02, 0.03, 0.04], [1e4, 5e4, 1e5], strict=True)]
    )
    s_o = np.array(
        [_s_rad(b, 8, t) for b, t in zip([0.60, 0.90, 1.20], [2e4, 6e4, 1.1e5], strict=True)]
    )
    expected = fractions[1] * s_h + fractions[8] * s_o
    np.testing.assert_allclose(mixed, expected, rtol=1e-9)


def test_mf26_angular_reader_returns_per_energy_distributions() -> None:
    """The LAW=2 reader returns each incident energy's tabulated P(mu)."""
    text = "\n".join(_ang_525(100, [(1.0e6, *_UNIFORM), (1.0e7, *_FORWARD)]))
    dists = read_mf26_angular_distributions(text, 525)
    assert set(dists) == {100}
    energies = [e for e, _, _ in dists[100]]
    np.testing.assert_allclose(energies, [1.0e6, 1.0e7], rtol=1e-5)
    _, mu1, p1 = dists[100][1]
    np.testing.assert_allclose(mu1, [-1.0, 1.0], rtol=1e-5)
    np.testing.assert_allclose(p1, [1.0, 3.0], rtol=1e-5)


def test_element_scattering_power_is_twice_transport_moment() -> None:
    """Per element: T(E) = 2 (N_A/M) sigma_525(E) <1-mu>(E), <1-mu> from P(mu)."""
    e_ev = [1.0e6, 1.0e7]
    barns = [5.0, 1.0]
    text = "\n".join(
        _xs_525(100, e_ev, barns) + _ang_525(100, [(1.0e6, *_UNIFORM), (1.0e7, *_FORWARD)])
    )
    result = element_scattering_power(text, elements={1})

    energy_mev, scattering = result[1]
    np.testing.assert_allclose(energy_mev, [1.0, 10.0], rtol=1e-9)
    # <1-mu> = 1 (uniform) at 1 MeV, 2/3 (linear) at 10 MeV
    expected = [_s_scat(5.0, 1, 1.0), _s_scat(1.0, 1, 0.5)]
    np.testing.assert_allclose(scattering, expected, rtol=1e-6)


def test_material_scattering_power_is_mass_weighted() -> None:
    """Water scattering power is the mass-fraction sum of its elements' on a grid."""
    e_ev = [1.0e6, 1.0e7]
    grid = np.array([1.0, 10.0])
    text = "\n".join(
        _xs_525(100, e_ev, [5.0, 1.0])
        + _ang_525(100, [(1.0e6, *_UNIFORM), (1.0e7, *_FORWARD)])
        + _xs_525(800, e_ev, [40.0, 8.0])
        + _ang_525(800, [(1.0e6, *_UNIFORM), (1.0e7, *_FORWARD)])
    )
    elements = element_scattering_power(text, elements={1, 8})
    fractions = mass_fractions_from_formula({1: 2, 8: 1})
    mixed = material_scattering_power(elements, fractions, grid)

    t_h = np.array([_s_scat(5.0, 1, 1.0), _s_scat(1.0, 1, 0.5)])
    t_o = np.array([_s_scat(40.0, 8, 1.0), _s_scat(8.0, 8, 0.5)])
    expected = fractions[1] * t_h + fractions[8] * t_o
    np.testing.assert_allclose(mixed, expected, rtol=1e-6)


def test_mf26_spectra_reader_returns_law1_distributions() -> None:
    """The LAW=1 reader returns each incident energy's tabulated secondary spectrum."""
    text = "\n".join(_mf26_tab2(100, 534, 1, [(1.0e6, [0.0, 100.0], [0.01, 0.01])]))
    spectra = read_mf26_spectra(text, 534)
    assert set(spectra) == {100}
    energy, w, f = spectra[100][0]
    assert energy == pytest.approx(1.0e6, rel=1e-5)
    np.testing.assert_allclose(w, [0.0, 100.0], atol=1e-6)
    np.testing.assert_allclose(f, [0.01, 0.01], rtol=1e-5)


def test_element_collision_stopping_sums_excitation_and_restricted_ionization() -> None:
    """S_coll = (N_A/M)[sigma_528 <loss>_528 + sigma_534 integral_0^delta (B+W) f dW].

    Excitation loss 50 eV; ionization binding B=20 eV with a flat secondary spectrum
    W in [0, 100] eV (all below the 0.2 MeV cut), whose restricted mean loss is exactly
    B + W_max/2 = 70 eV (trapezoid of a linear integrand). Grid = the 528 energy nodes.
    """
    e_ev = [1.0e6, 1.0e7]
    spectrum = [(e, [0.0, 100.0], [0.01, 0.01]) for e in e_ev]  # f normalized over [0,100]
    text = "\n".join(
        _xs(100, 528, e_ev, [2.0, 2.0])  # excitation cross section
        + _law8(100, 528, e_ev, [50.0, 50.0])  # excitation average loss (eV)
        + _xs(100, 534, [20.0, *e_ev], [0.0, 3.0, 3.0])  # ionization: threshold B=20 eV
        + _mf26_tab2(100, 534, 1, spectrum)  # ionization secondary spectrum
    )
    result = element_collision_stopping(text, delta_cut=0.2, elements={1})

    energy_mev, s_coll = result[1]
    np.testing.assert_allclose(energy_mev, [1.0, 10.0], rtol=1e-9)
    prefactor = AVOGADRO / STANDARD_ATOMIC_WEIGHT[1] * 1.0e-24 * 1.0e-6
    expected = prefactor * (2.0 * 50.0 + 3.0 * 70.0)  # excitation + restricted ionization
    np.testing.assert_allclose(s_coll, [expected, expected], rtol=1e-6)
