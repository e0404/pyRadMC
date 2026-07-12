"""A minimal ENDF-6 reader for the EPICS (EPDL/EEDL) interaction sublibraries.

Reads only what the tabulated backend needs: the TAB1 tables under a requested MF
(MF=23 photon cross sections, MF=27 form factors / scattering functions, and the
electron MF/MT the EEDL parser asks for). It is *not* a general ENDF-6 parser — it
assumes each ``(MAT, MF, MT)`` section is one HEAD record followed by one TAB1
record, which is how the EPICS photoatomic and electron files are laid out.

ENDF-6 line layout: six 11-column data fields, then MAT (columns 67-70), MF (71-72),
MT (73-75), and a line number (76-80). ``MAT = 1000 * Z``. Numbers use the exponent
shorthand where the ``E`` may be dropped (``"1.234567+6" == 1.234567e6``). Photon
energies are in eV and cross sections in barns; unit conversion is the caller's job.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["parse_endf_float", "read_tab1_by_mf"]

_FIELD = 11


def parse_endf_float(field: str) -> float:
    """Parse one ENDF number, including the ``1.234+6`` (no ``E``) exponent form."""
    s = field.strip()
    if not s:
        return 0.0
    if "e" in s or "E" in s:
        return float(s)
    for i in range(1, len(s)):  # an exponent sign that is not the leading sign
        if s[i] in "+-" and s[i - 1] not in "eE":
            return float(f"{s[:i]}e{s[i:]}")
    return float(s)


def _fields(line: str) -> list[str]:
    return [line[i * _FIELD : (i + 1) * _FIELD] for i in range(6)]


def _control(line: str) -> tuple[int, int, int]:
    """``(MAT, MF, MT)`` from a line; ``(-1, -1, -1)`` for a too-short/blank line."""
    if len(line) < 75:
        return -1, -1, -1

    def _int(s: str) -> int:
        s = s.strip()
        return int(s) if s else 0

    return _int(line[66:70]), _int(line[70:72]), _int(line[72:75])


def read_tab1_by_mf(
    text: str, mf_target: int
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Return ``{(MAT, MT): (x, y)}`` for every TAB1 section under ``mf_target``.

    ``x`` and ``y`` are the tabulated abscissae and ordinates (energy in eV and cross
    section in barns for MF=23), in file order.
    """
    groups: list[tuple[tuple[int, int, int], list[str]]] = []
    key: tuple[int, int, int] | None = None
    current: list[str] = []
    for line in text.splitlines():
        mat, mf, mt = _control(line)
        if mat <= 0 or mt == 0:  # junk, or a SEND/FEND/MEND boundary
            if current:
                groups.append((key, current))  # type: ignore[arg-type]
                current, key = [], None
            continue
        this = (mat, mf, mt)
        if this != key:
            if current:
                groups.append((key, current))  # type: ignore[arg-type]
            current, key = [line], this
        else:
            current.append(line)
    if current:
        groups.append((key, current))  # type: ignore[arg-type]

    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for (mat, mf, mt), lines in groups:
        if mf == mf_target:
            out[(mat, mt)] = _parse_tab1(lines)
    return out


def _parse_tab1(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Parse one HEAD + TAB1 section into ``(x, y)`` arrays."""
    control = _fields(lines[1])  # lines[0] is the HEAD (ZA, AWR, ...)
    n_regions = int(parse_endf_float(control[4]))
    n_points = int(parse_endf_float(control[5]))
    n_interp_lines = math.ceil(2 * n_regions / 6)
    data_lines = lines[2 + n_interp_lines :]

    numbers: list[float] = []
    for line in data_lines:
        numbers.extend(parse_endf_float(f) for f in _fields(line) if f.strip())
    xy = np.array(numbers[: 2 * n_points], dtype=np.float64)
    return xy[0::2], xy[1::2]
