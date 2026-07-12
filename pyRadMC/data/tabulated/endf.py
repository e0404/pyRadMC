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

__all__ = [
    "parse_endf_float",
    "read_mf26_angular_distributions",
    "read_mf26_energy_transfer",
    "read_tab1_by_mf",
]

# One incident energy's tabulated distribution: (energy, abscissae, ordinates).
AngularDistribution = tuple[float, np.ndarray, np.ndarray]

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


def _group_sections(text: str) -> list[tuple[tuple[int, int, int], list[str]]]:
    """Split an ENDF file into ``((MAT, MF, MT), lines)`` sections, in file order.

    A section is a maximal run of lines sharing one ``(MAT, MF, MT)``; blank lines and
    the ``MT == 0`` SEND/FEND/MEND boundaries close the current run.
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
    return groups


def read_tab1_by_mf(
    text: str, mf_target: int
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Return ``{(MAT, MT): (x, y)}`` for every TAB1 section under ``mf_target``.

    ``x`` and ``y`` are the tabulated abscissae and ordinates (energy in eV and cross
    section in barns for MF=23), in file order.
    """
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for (mat, mf, mt), lines in _group_sections(text):
        if mf == mf_target:
            out[(mat, mt)] = _parse_tab1(lines)
    return out


def read_mf26_energy_transfer(
    text: str, mt_target: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return ``{MAT: (energy, average_energy_transfer)}`` for MF=26/``mt_target``.

    EEDL stores each electro-atomic reaction's secondary distribution as an MF=26
    section of ``NK`` subsections (an outgoing photon and/or electron). The average
    energy the primary transfers — the quantity that turns a cross section into a
    stopping power — is the LAW=8 subsection's TAB1 (ENDF-6 File 26, LAW 8: "energy
    transfer"). This walks the subsections, skipping the LAW=1 continuum spectra, and
    returns that table. Both columns are in eV, as tabulated.
    """
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for (mat, mf, mt), lines in _group_sections(text):
        if mf == 26 and mt == mt_target:
            out[mat] = _parse_mf26_energy_transfer(lines)
    return out


def read_mf26_angular_distributions(
    text: str, mt_target: int
) -> dict[int, list[AngularDistribution]]:
    """Return ``{MAT: [(energy, mu, P), ...]}`` for MF=26/``mt_target``.

    The LAW=2 subsection of an elastic-scattering reaction tabulates, per incident
    energy, the normalized angular distribution ``P(mu)`` over ``mu`` in
    ``[-1, 0.999999]`` (the forward Coulomb tail above that is analytic and not stored).
    This walks the subsections, skipping any LAW=1/LAW=8 companions, and returns those
    per-energy tables in file order; energies are in eV.
    """
    out: dict[int, list[AngularDistribution]] = {}
    for (mat, mf, mt), lines in _group_sections(text):
        if mf == 26 and mt == mt_target:
            out[mat] = _parse_mf26_angular(lines)
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


def _int_fields(line: str) -> list[int]:
    """Return the six control fields as integers (ENDF control records are integers)."""
    return [int(parse_endf_float(f)) for f in _fields(line)]


def _skip_tab1(lines: list[str], idx: int) -> int:
    """Advance ``idx`` past the TAB1 record starting at ``lines[idx]``."""
    _, _, _, _, n_regions, n_points = _int_fields(lines[idx])
    return idx + 1 + math.ceil(2 * n_regions / 6) + math.ceil(2 * n_points / 6)


def _read_tab1_data(lines: list[str], idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the ``(x, y)`` table of the TAB1 record starting at ``lines[idx]``."""
    _, _, _, _, n_regions, n_points = _int_fields(lines[idx])
    start = idx + 1 + math.ceil(2 * n_regions / 6)
    numbers: list[float] = []
    for line in lines[start : start + math.ceil(2 * n_points / 6)]:
        numbers.extend(parse_endf_float(f) for f in _fields(line) if f.strip())
    xy = np.array(numbers[: 2 * n_points], dtype=np.float64)
    return xy[0::2], xy[1::2]


def _skip_tab2_lists(lines: list[str], idx: int) -> int:
    """Advance past a LAW=1/LAW=2 body: a TAB2 header then one list record per energy."""
    _, _, _, _, n_regions, n_energies = _int_fields(lines[idx])
    idx += 1 + math.ceil(2 * n_regions / 6)
    for _e in range(n_energies):
        n_words = _int_fields(lines[idx])[4]  # list record: N1 = NW
        idx += 1 + math.ceil(n_words / 6)
    return idx


def _read_tab2_lists(lines: list[str], idx: int) -> list[AngularDistribution]:
    """Read a LAW=2 body's per-energy ``(energy, abscissae, ordinates)`` list records."""
    _, _, _, _, n_regions, n_energies = _int_fields(lines[idx])
    idx += 1 + math.ceil(2 * n_regions / 6)
    out: list[AngularDistribution] = []
    for _e in range(n_energies):
        _, _, _, _, n_words, n_points = _int_fields(lines[idx])
        energy = parse_endf_float(_fields(lines[idx])[1])  # list header: C2 = incident E
        idx += 1
        numbers: list[float] = []
        for line in lines[idx : idx + math.ceil(n_words / 6)]:
            numbers.extend(parse_endf_float(f) for f in _fields(line) if f.strip())
        idx += math.ceil(n_words / 6)
        xy = np.array(numbers[: 2 * n_points], dtype=np.float64)
        out.append((energy, xy[0::2], xy[1::2]))
    return out


def _walk_mf26_subsections(lines: list[str], target_law: int) -> tuple[int, int]:
    """Return ``(body_idx, law)`` where the ``target_law`` subsection's body begins.

    Walks the ``NK`` subsections, skipping the LAW bodies that are not the target
    (LAW=1/2 TAB2-with-lists, LAW=8 TAB1). LAWs other than 1, 2, 8 are not expected in
    the EEDL reactions this reads and raise rather than being silently mis-walked.
    """
    n_subsections = _int_fields(lines[0])[4]  # HEAD: N1 = NK
    idx = 1
    for _ in range(n_subsections):
        law = _int_fields(lines[idx])[3]  # multiplicity TAB1: L2 = LAW
        idx = _skip_tab1(lines, idx)  # past the multiplicity table
        if law == target_law:
            return idx, law
        if law in (1, 2):
            idx = _skip_tab2_lists(lines, idx)
        elif law == 8:
            idx = _skip_tab1(lines, idx)
        else:
            raise ValueError(f"unsupported MF=26 LAW={law}")
    raise ValueError(f"no LAW={target_law} subsection found")


def _parse_mf26_energy_transfer(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the LAW=8 ``(E, energy_transfer)`` table of an MF=26 section."""
    idx, _ = _walk_mf26_subsections(lines, target_law=8)
    return _read_tab1_data(lines, idx)


def _parse_mf26_angular(lines: list[str]) -> list[AngularDistribution]:
    """Return the LAW=2 per-energy angular distributions of an MF=26 section."""
    idx, _ = _walk_mf26_subsections(lines, target_law=2)
    return _read_tab2_lists(lines, idx)
