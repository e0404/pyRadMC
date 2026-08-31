"""Synthetic IAEA phase-space fixtures, generated in code (never committed as binaries).

The reader (:mod:`pyradmc.geometry.phasespace`) is pinned independently in
``tests/unit/test_iaea_phsp.py`` against hand-authored bytes and a real reference
file, so writing fixtures to the same spec here to feed *source* tests is not
circular — these files exercise the source, not the reader.

Fixtures always store all six coordinates and carry no extra floats/ints; that is
the common case and keeps the helper small.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Rec", "write_phsp"]


@dataclass(frozen=True)
class Rec:
    """One record to write. ``typ`` is the IAEA code (1=photon, 2=e-, 3=e+, ...)."""

    typ: int
    E: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w_negative: bool = False
    weight: float = 1.0
    new_history: bool = True


def write_phsp(stem: Path, records: list[Rec], *, byte_order: int = 1234) -> Path:
    """Write ``{stem}.IAEAheader`` and ``{stem}.IAEAphsp``; return the phsp path.

    Encodes to the IAEA layout: the sign of the particle type carries ``w``'s sign,
    the sign of the energy carries ``new_history``, ``w`` itself is not stored.
    """
    prefix = "<" if byte_order == 1234 else ">"
    record_length = 1 + 4 + 4 * 6  # typ + E + x,y,z,u,v,weight
    payload = bytearray()
    for r in records:
        typ8 = -abs(r.typ) if r.w_negative else abs(r.typ)
        e_signed = -abs(r.E) if r.new_history else abs(r.E)
        payload += struct.pack(
            prefix + "bfffffff",
            typ8,
            e_signed,
            r.x,
            r.y,
            r.z,
            r.u,
            r.v,
            r.weight,
        )

    header = f"""$IAEA_INDEX:
1000

$FILE_TYPE:
0

$CHECKSUM:
{record_length * len(records)}

$RECORD_CONTENTS:
    1     // X is stored ?
    1     // Y is stored ?
    1     // Z is stored ?
    1     // U is stored ?
    1     // V is stored ?
    1     // W is stored ?
    1     // Weight is stored ?
    0     // Extra floats stored ?
    0     // Extra longs stored ?

$RECORD_CONSTANT:

$RECORD_LENGTH:
{record_length}

$BYTE_ORDER:
{byte_order}

$ORIG_HISTORIES:
{len(records)}

$PARTICLES:
{len(records)}
"""
    (stem.parent / f"{stem.name}.IAEAheader").write_text(header)
    phsp = stem.parent / f"{stem.name}.IAEAphsp"
    phsp.write_bytes(bytes(payload))
    return phsp
