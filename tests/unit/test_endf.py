"""ENDF-6 reader for EPICS data (Phase 5, tabulated, slice B).

Format-correct ENDF lines are generated in code (no committed data), matching the
real EPDL layout observed in ``EPDL2023.ALL``: HEAD then TAB1, ``MAT = 1000 * Z``,
eV/barns, and the exponent shorthand where ``E`` is dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.data.tabulated.endf import parse_endf_float, read_tab1_by_mf


class TestFloatParsing:
    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("1.234567+6", 1_234_567.0),
            (".999242000", 0.999242),
            ("9.56230E-8", 9.5623e-8),
            ("1.0000E+11", 1.0e11),
            ("-1.5-3", -1.5e-3),
            ("6.022+23", 6.022e23),
            ("0.0", 0.0),
            ("   2.0     ", 2.0),
            ("           ", 0.0),
        ],
    )
    def test_endf_number_forms(self, field: str, expected: float) -> None:
        assert parse_endf_float(field) == pytest.approx(expected, rel=1e-9, abs=0.0)


def _line(fields: list[str], mat: int, mf: int, mt: int, num: int) -> str:
    """One 80-column ENDF record: six 11-char fields + MAT/MF/MT/line-number."""
    padded = (fields + [""] * 6)[:6]  # always six fields, so columns 67+ align
    body = "".join(f"{f:<11}"[:11] for f in padded)
    return f"{body}{mat:>4}{mf:>2}{mt:>3}{num:>5}"


def _section(mat: int, mt: int, xs: list[float], ys: list[float]) -> list[str]:
    """A HEAD + TAB1 (single lin-lin region) MF=23 section, then a SEND boundary."""
    za = f"{mat // 100 * 1000:.1f}"  # ZA = 1000 * Z; MAT = 100 * Z here for the test
    head = _line([za, "0.999242", "0", "0", "0", "0"], mat, 23, mt, 1)
    ctrl = _line(["0.0", "0.0", "0", "0", "1", str(len(xs))], mat, 23, mt, 2)
    interp = _line([str(len(xs)), "2"], mat, 23, mt, 3)
    data_fields: list[str] = []
    for x, y in zip(xs, ys, strict=True):
        data_fields += [f"{x:.5E}", f"{y:.5E}"]
    data = _line(data_fields[:6], mat, 23, mt, 4)  # three (x, y) pairs on one line
    send = _line(["0.0", "0.0", "0", "0", "0", "0"], mat, 23, 0, 99999)
    return [head, ctrl, interp, data, send]


def test_reads_tab1_section() -> None:
    """A single MF=23/MT=504 section round-trips its (energy, cross-section) pairs."""
    xs = [1.0e3, 1.0e4, 1.0e5]  # eV
    ys = [0.10, 0.20, 0.30]  # barns
    text = "\n".join(_section(100, 504, xs, ys))
    tables = read_tab1_by_mf(text, 23)
    assert set(tables) == {(100, 504)}
    energy, sigma = tables[(100, 504)]
    np.testing.assert_allclose(energy, xs, rtol=1e-5)
    np.testing.assert_allclose(sigma, ys, rtol=1e-5)


def test_separates_materials_and_channels() -> None:
    """Multiple MAT/MT sections are keyed independently; other MFs are ignored."""
    lines = (
        _section(100, 504, [1.0e3, 1.0e4, 1.0e5], [0.1, 0.2, 0.3])  # H, incoherent
        + _section(100, 502, [1.0e3, 1.0e4, 1.0e5], [1.0, 0.5, 0.1])  # H, coherent
        + _section(800, 504, [1.0e3, 1.0e4, 1.0e5], [0.8, 1.6, 2.4])  # O, incoherent
    )
    tables = read_tab1_by_mf("\n".join(lines), 23)
    assert set(tables) == {(100, 504), (100, 502), (800, 504)}
    assert read_tab1_by_mf("\n".join(lines), 27) == {}  # no MF=27 present
    _, o_sigma = tables[(800, 504)]
    np.testing.assert_allclose(o_sigma, [0.8, 1.6, 2.4], rtol=1e-5)
