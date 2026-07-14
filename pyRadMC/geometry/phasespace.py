"""IAEA phase-space file reader.

Phase 5, phase-space source, Slice 1. Reads the two-file IAEA phase-space format
(Capote et al., *Phase-space database for external beam radiotherapy*, IAEA
(NDS)-0484, Vienna, 2006): an ASCII ``.IAEAheader`` describing the record layout
and a binary ``.IAEAphsp`` of fixed-length records. This module is pure I/O — it
decodes records into :class:`PhspRecord`; the sampling of records into primaries
lives in the source layer (:mod:`pyRadMC.geometry.source`), never here.

Byte layout of one record, in read order:

* ``typ`` : ``int8``. ``abs(typ)`` is the IAEA particle code
  (:data:`IAEA_PHOTON` = 1, :data:`IAEA_ELECTRON` = 2, :data:`IAEA_POSITRON` = 3;
  4 = neutron, 5 = proton). The *sign* of ``typ`` carries the sign of the third
  direction cosine ``w`` — ``typ < 0`` means ``w < 0`` — because ``w`` is never
  written to disk.
* ``E`` : ``float32``. The *sign* of ``E`` flags the first particle of a new
  primary history (``E < 0`` -> :attr:`PhspRecord.new_history`); the stored value
  is ``abs(E)``.
* ``x, y, z, u, v, weight`` : ``float32`` each, but only those whose
  ``$RECORD_CONTENTS`` flag is 1. A flag of 0 means the value is constant across
  the file and read once from ``$RECORD_CONSTANT``. The seven flags are ordered
  ``x, y, z, u, v, w, weight``; ``w``'s flag must be 1 (``w`` is reconstructed,
  never a stored constant).
* ``w`` is reconstructed: ``w = sign_w * sqrt(1 - u**2 - v**2)``, with the
  degenerate ``u**2 + v**2 > 1`` case renormalizing ``u, v`` and setting ``w = 0``.
* ``extra_floats`` (``Nf`` x ``float32``) then ``extra_ints`` (``Ni`` x ``int32``).
"""

from __future__ import annotations

import mmap
import struct
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, NamedTuple

import numpy as np

from pyRadMC.geometry.source import Primary, Source
from pyRadMC.rng import RNGState, uniform

__all__ = [
    "IAEA_ELECTRON",
    "IAEA_PHOTON",
    "IAEA_POSITRON",
    "IAEAHeader",
    "IAEAPhaseSpace",
    "PhaseSpaceSource",
    "PhspRecord",
    "read_iaea_header",
]

IAEA_PHOTON = 1
IAEA_ELECTRON = 2
IAEA_POSITRON = 3

# IAEA particle code -> transport kind name. Only the particles a photon MC
# transports are supported; neutrons (4) and protons (5) are rejected on load
# rather than silently dropped, since dropping them would bias the fluence.
_IAEA_KIND = {IAEA_PHOTON: "photon", IAEA_ELECTRON: "electron", IAEA_POSITRON: "positron"}

_HEADER_EXT = ".IAEAheader"
_PHSP_EXT = ".IAEAphsp"

# The six per-particle coordinates carried by RECORD_CONTENTS in file order. ``w``
# sits between ``v`` and ``weight`` in the flag list but is never a stored float
# (its sign rides on the particle type), so it is excluded from the read/constant
# machinery and handled separately.
_COORDS = ("x", "y", "z", "u", "v", "weight")
_W_FLAG_INDEX = 5  # position of w within the seven RECORD_CONTENTS flags


class PhspRecord(NamedTuple):
    """One decoded phase-space particle.

    Energies and directions are physical: ``energy`` is always positive and
    ``(u, v, w)`` is a unit vector. ``particle_type`` is the IAEA integer code
    (mapping to transport particle kinds is the source layer's job).
    """

    particle_type: int
    energy: float
    x: float
    y: float
    z: float
    u: float
    v: float
    w: float
    weight: float
    new_history: bool
    extra_floats: tuple[float, ...]
    extra_ints: tuple[int, ...]


@dataclass(frozen=True)
class IAEAHeader:
    """Parsed ``.IAEAheader``: everything needed to decode the binary records."""

    byte_order: str  # struct prefix: "<" (little, $BYTE_ORDER 1234) or ">" (big, 4321)
    record_length: int  # bytes per record
    n_particles: int
    n_extra_floats: int
    n_extra_ints: int
    stored: tuple[bool, ...]  # per _COORDS: True if stored per-particle
    constants: dict[str, float]  # coordinate -> constant value, for non-stored coords


def _iaea_path(path: Path | str) -> tuple[Path, Path]:
    """Resolve a header/phsp/stem path to the ``(.IAEAheader, .IAEAphsp)`` pair."""
    p = Path(path)
    stem = p.with_suffix("") if p.suffix in (_HEADER_EXT, _PHSP_EXT) else p
    return stem.with_suffix(_HEADER_EXT), stem.with_suffix(_PHSP_EXT)


def _parse_sections(text: str) -> dict[str, list[str]]:
    """Split a header into ``$KEYWORD:`` sections, dropping ``//`` comments/blanks.

    Each section maps to its list of non-empty, comment-stripped value lines.
    """
    sections: dict[str, list[str]] = {}
    key: str | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("$") and stripped.endswith(":"):
            key = stripped[1:-1]
            sections[key] = []
            continue
        if key is None:
            continue
        value = stripped.split("//", 1)[0].strip()
        if value:
            sections[key].append(value)
    return sections


def read_iaea_header(path: Path | str) -> IAEAHeader:
    """Parse a ``.IAEAheader`` and validate it against its own record geometry."""
    header_path, _ = _iaea_path(path)
    sections = _parse_sections(header_path.read_text())

    contents = sections.get("RECORD_CONTENTS", [])
    # Seven coordinate flags (x,y,z,u,v,w,weight) then Nf, Ni.
    if len(contents) < 9:
        raise ValueError(
            f"RECORD_CONTENTS needs 7 flags plus Nf and Ni, got {len(contents)} entries"
        )
    flags = [int(tok) for tok in contents[:7]]
    n_extra_floats = int(contents[7])
    n_extra_ints = int(contents[8])
    if flags[_W_FLAG_INDEX] != 1:
        raise ValueError(
            "w must be stored (RECORD_CONTENTS flag 6 = 1); w is reconstructed, never a constant"
        )
    stored = tuple(bool(flags[i]) for i in (0, 1, 2, 3, 4, 6))  # x,y,z,u,v,weight

    const_values = [float(tok) for tok in sections.get("RECORD_CONSTANT", [])]
    n_constants_expected = sum(not s for s in stored)
    if len(const_values) != n_constants_expected:
        raise ValueError(
            f"RECORD_CONSTANT has {len(const_values)} values but "
            f"{n_constants_expected} coordinates are constant"
        )
    constants: dict[str, float] = {}
    it = iter(const_values)
    for coord, is_stored in zip(_COORDS, stored, strict=True):
        if not is_stored:
            constants[coord] = next(it)

    byte_order = _byte_order(sections)
    record_length = _record_length(sections, stored, n_extra_floats, n_extra_ints)
    n_particles = _n_particles(sections, header_path, record_length)

    return IAEAHeader(
        byte_order=byte_order,
        record_length=record_length,
        n_particles=n_particles,
        n_extra_floats=n_extra_floats,
        n_extra_ints=n_extra_ints,
        stored=stored,
        constants=constants,
    )


def _byte_order(sections: dict[str, list[str]]) -> str:
    tokens = sections.get("BYTE_ORDER", [])
    order = tokens[0] if tokens else "1234"
    if order == "1234":
        return "<"
    if order == "4321":
        return ">"
    raise ValueError(f"unknown BYTE_ORDER {order!r}; expected 1234 or 4321")


def _record_length(
    sections: dict[str, list[str]],
    stored: tuple[bool, ...],
    n_extra_floats: int,
    n_extra_ints: int,
) -> int:
    """Bytes per record implied by the flags, cross-checked against RECORD_LENGTH."""
    computed = 1 + 4 + 4 * sum(stored) + 4 * n_extra_floats + 4 * n_extra_ints
    declared_tokens = sections.get("RECORD_LENGTH", [])
    if declared_tokens:
        declared = int(declared_tokens[0])
        if declared != computed:
            raise ValueError(
                f"RECORD_LENGTH {declared} disagrees with the {computed} bytes "
                "implied by RECORD_CONTENTS"
            )
    return computed


def _n_particles(sections: dict[str, list[str]], header_path: Path, record_length: int) -> int:
    """Particle count: the on-disk record count is authoritative, PARTICLES advisory.

    A record that is not on disk cannot be read, so the file size wins; a PARTICLES
    header that disagrees (real files do — e.g. the Varian TrueBeam files are off by
    one) is a warning, not a fatal error. A file size that is not a whole number of
    records *is* fatal: that is genuine truncation/corruption.
    """
    _, phsp_path = _iaea_path(header_path)
    declared_tokens = sections.get("PARTICLES", [])
    declared = int(declared_tokens[0]) if declared_tokens else None

    if phsp_path.exists():
        size = phsp_path.stat().st_size
        if size % record_length != 0:
            raise ValueError(
                f"{phsp_path.name} size {size} is not a multiple of record length {record_length}"
            )
        on_disk = size // record_length
        if declared is not None and declared != on_disk:
            warnings.warn(
                f"PARTICLES header says {declared} but {phsp_path.name} holds {on_disk} "
                "records; using the on-disk count",
                stacklevel=2,
            )
        return on_disk
    if declared is None:
        raise ValueError("cannot determine particle count: no PARTICLES and no phsp file")
    return declared


def _compute_w(u: float, v: float, sign_w: float) -> tuple[float, float, float]:
    """Reconstruct ``w`` (and, if degenerate, renormalize ``u, v``)."""
    tmp = u * u + v * v
    if tmp <= 1.0:
        return u, v, sign_w * (1.0 - tmp) ** 0.5
    scale = tmp**0.5
    return u / scale, v / scale, 0.0


def _build_struct(header: IAEAHeader) -> struct.Struct:
    """Build the struct for one record's *stored* fields, in read order."""
    fmt = header.byte_order + "bf"  # int8 type, float32 energy
    fmt += "f" * sum(header.stored)  # stored coordinates
    fmt += "f" * header.n_extra_floats
    fmt += "i" * header.n_extra_ints
    return struct.Struct(fmt)


def _record_dtype(header: IAEAHeader) -> np.dtype:
    """NumPy structured dtype for bulk-decoding records straight from the mmap.

    Names the type byte, the energy float, and each *stored* coordinate at its byte
    offset, with ``itemsize`` the full record length so a strided view selects one
    record per stride. Constant coordinates and extra floats/ints are not named —
    they are filled from the header (constants) or ignored (extras) after gather.
    """
    order = header.byte_order
    names = ["typ", "e"]
    formats: list[object] = [np.int8, order + "f4"]
    offsets = [0, 1]
    off = 5  # after int8 type + float32 energy
    for coord, is_stored in zip(_COORDS, header.stored, strict=True):
        if is_stored:
            names.append(coord)
            formats.append(order + "f4")
            offsets.append(off)
            off += 4
    return np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": header.record_length}
    )


def _decode_record(raw: bytes, header: IAEAHeader, struct_: struct.Struct) -> PhspRecord:
    """Decode one raw record's bytes into a :class:`PhspRecord`."""
    fields = struct_.unpack(raw)
    typ = fields[0]
    e_signed = fields[1]
    new_history = e_signed < 0.0
    energy = abs(e_signed)

    # Fill the stored coordinates in order, taking constants for the rest.
    values: dict[str, float] = {}
    idx = 2
    for coord, is_stored in zip(_COORDS, header.stored, strict=True):
        if is_stored:
            values[coord] = fields[idx]
            idx += 1
        else:
            values[coord] = header.constants[coord]

    n_f = header.n_extra_floats
    extra_floats = tuple(fields[idx : idx + n_f])
    extra_ints = tuple(fields[idx + n_f : idx + n_f + header.n_extra_ints])

    sign_w = -1.0 if typ < 0 else 1.0
    u, v, w = _compute_w(values["u"], values["v"], sign_w)

    return PhspRecord(
        particle_type=abs(typ),
        energy=energy,
        x=values["x"],
        y=values["y"],
        z=values["z"],
        u=u,
        v=v,
        w=w,
        weight=values["weight"],
        new_history=new_history,
        extra_floats=extra_floats,
        extra_ints=extra_ints,
    )


class IAEAPhaseSpace:
    """Iterable, context-managed reader over the records of an IAEA phsp file.

    Iterating yields :class:`PhspRecord` in file order. ``len`` is the particle
    count from the header. The reader opens the binary lazily on ``__enter__`` and
    each ``__iter__`` rewinds to the first record, so a single instance can be
    iterated more than once within its ``with`` block.
    """

    def __init__(self, path: Path | str) -> None:
        self.header = read_iaea_header(path)
        _, self._phsp_path = _iaea_path(path)
        self._fh: BinaryIO | None = None
        self._struct = _build_struct(self.header)

    def __enter__(self) -> IAEAPhaseSpace:
        """Open the binary phsp file for iteration."""
        self._fh = self._phsp_path.open("rb")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the binary phsp file."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __len__(self) -> int:
        """Return the number of particle records in the file."""
        return self.header.n_particles

    def __iter__(self) -> Iterator[PhspRecord]:
        """Yield each :class:`PhspRecord` in file order (rewinds to the start)."""
        if self._fh is None:
            raise RuntimeError("iterate an IAEAPhaseSpace inside its `with` block")
        self._fh.seek(0)
        length = self.header.record_length
        for _ in range(self.header.n_particles):
            yield _decode_record(self._fh.read(length), self.header, self._struct)


class PhaseSpaceSource(Source):
    """A primary source that samples particles from an IAEA phase-space file.

    Unlike the analytic beam sources, a phase-space file is a recorded mix of
    photons, electrons and positrons at non-unit statistical weights; each emitted
    :class:`~pyRadMC.geometry.source.Primary` therefore carries its own ``kind`` and
    ``weight``. :meth:`emit` draws one record per history from the history's RNG
    stream (uniform random sampling with replacement), so the source stays a pure
    function of ``(seed, history)`` like the rest of the engine.

    **Streaming.** The file is memory-mapped, not read into RAM: only the header
    plus a one-pass scan of the per-record type byte are held eagerly, and each
    :meth:`emit` decodes a single record on demand. A multi-gigabyte, tens-of-
    millions-of-particle file (a real linac phase space) is therefore usable
    without loading it — resident memory tracks the pages actually touched.

    **Unsupported particles.** A photon MC transports only photons, electrons and
    positrons; a real file may carry the odd neutron/proton (IAEA codes 4/5). With
    ``skip_unsupported`` (the default) those records are excluded from the sampled
    population and their count is warned — one stray particle in tens of millions
    should not reject the file, and excluding a ~1e-8 fraction is negligible. With
    ``skip_unsupported=False`` any unsupported record makes construction raise.

    Because a phase-space particle can be any type at any position, this source
    breaks the unique beamlet ownership the Dij design relies on; it plugs into
    ``run`` on either backend, never ``run_dij``. Per-history :meth:`emit` serves
    the reference engine; :meth:`sample_batch` serves the warp engine, which
    transports a whole chunk at once (AGENTS.md 7.2).
    """

    def __init__(self, path: Path | str, *, skip_unsupported: bool = True) -> None:
        self.header = read_iaea_header(path)
        _, self._phsp_path = _iaea_path(path)
        self._struct = _build_struct(self.header)
        self._record_dtype = _record_dtype(self.header)
        self._reclen = self.header.record_length
        n = self.header.n_particles
        if n == 0:
            raise ValueError(f"phase-space file {path} contains no particles")

        self._file: BinaryIO | None = self._phsp_path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # One vectorized pass reading the per-record type byte (offset 0) and energy
        # float (offset 1) via a strided structured view, to (a) find records this MC
        # cannot transport and (b) get the maximum energy for sizing the transport
        # tables. The owned columns are copied out and the view dropped immediately:
        # a view left alive (e.g. held by a traceback if the code below raises) would
        # keep the map's buffer exported and block closing it in __del__.
        scan_dtype = np.dtype(
            {
                "names": ["typ", "e"],
                "formats": [np.int8, self.header.byte_order + "f4"],
                "offsets": [0, 1],
                "itemsize": self._reclen,
            }
        )
        view = np.frombuffer(self._mmap, dtype=scan_dtype, count=n)
        codes = np.abs(view["typ"].astype(np.int16))
        energies = np.abs(view["e"].astype(np.float64))
        del view
        self._max_energy = float(energies.max())
        supported = (codes == IAEA_PHOTON) | (codes == IAEA_ELECTRON) | (codes == IAEA_POSITRON)
        unsupported = np.flatnonzero(~supported)
        if unsupported.size:
            if not skip_unsupported:
                first = int(codes[unsupported[0]])
                raise ValueError(
                    f"unsupported particle type {first} in {path}; this photon MC "
                    "transports only photons, electrons and positrons "
                    "(pass skip_unsupported=True to drop them)"
                )
            warnings.warn(
                f"{unsupported.size} of {n} records in {self._phsp_path.name} are "
                "unsupported particle types (not photon/electron/positron); "
                "dropping them from the sampled population",
                stacklevel=2,
            )
        # Sorted, and tiny in the expected case; used to remap a compact sample
        # index over the supported subset back to the record's file position.
        self._unsupported: tuple[int, ...] = tuple(int(i) for i in unsupported)
        self._n_valid = n - len(self._unsupported)
        if self._n_valid == 0:
            raise ValueError(f"phase-space file {path} has no transportable particles")

    @property
    def max_energy(self) -> float:
        """Highest particle energy in the file, in MeV (for table sizing)."""
        return self._max_energy

    def _record_index(self, k: int) -> int:
        """Map the k-th *supported* record to its position in the full file.

        Walks the (sorted, usually empty) unsupported list, bumping the target
        index past each unsupported record at or below it.
        """
        actual = k
        for u in self._unsupported:
            if u <= actual:
                actual += 1
            else:
                break
        return actual

    def __len__(self) -> int:
        """Return the number of transportable particles available to sample."""
        return self._n_valid

    def _record_for_state(self, rng_state: RNGState) -> PhspRecord:
        """Sample one supported record from the file; consumes one uniform."""
        k = int(uniform(rng_state) * self._n_valid)
        if k >= self._n_valid:  # guard the uniform == 1.0 corner
            k = self._n_valid - 1
        idx = self._record_index(k)
        off = idx * self._reclen
        return _decode_record(self._mmap[off : off + self._reclen], self.header, self._struct)

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one primary, sampled uniformly from the file; consumes one uniform."""
        rec = self._record_for_state(rng_state)
        return Primary(
            energy=rec.energy,
            x=rec.x,
            y=rec.y,
            z=rec.z,
            ux=rec.u,
            uy=rec.v,
            uz=rec.w,
            kind=_IAEA_KIND[rec.particle_type],
            weight=rec.weight,
        )

    def _record_indices(self, k: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`_record_index`: map supported ranks to file positions."""
        idx = k.astype(np.int64, copy=True)
        for u in self._unsupported:  # sorted ascending, tiny in the expected case
            idx += idx >= u
        return idx

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, np.ndarray]:
        """Sample ``n`` primaries for histories ``[history_offset, history_offset + n)``.

        Returns the primaries as column arrays for bulk upload to a device backend,
        keyed ``particle_type`` (IAEA code 1/2/3), ``energy``, ``x/y/z``, ``ux/uy/uz``
        and ``weight``, geometry as float32 to match the device queue. Sampling and
        decoding are both vectorized: record indices come from a single ``PCG64(seed)``
        stream advanced to ``history_offset``, so history ``h`` always draws the
        ``h``-th value regardless of chunking (chunk-invariant), and the records are
        gathered from the mmap through :data:`_record_dtype` in one fancy-indexed read.

        This is a *different* stream from the reference :meth:`emit` (which spawns a
        per-history generator), so the two backends draw different records — both
        unbiased estimators of the same dose, compared statistically, never bit-wise.
        """
        bitgen = np.random.PCG64(seed)
        bitgen.advance(history_offset)
        u01 = np.random.Generator(bitgen).random(n)
        k = (u01 * self._n_valid).astype(np.int64)
        np.clip(k, 0, self._n_valid - 1, out=k)  # guard the u01 == 1 corner
        recs = np.frombuffer(self._mmap, dtype=self._record_dtype, count=self.header.n_particles)[
            self._record_indices(k)
        ]

        typ = recs["typ"].astype(np.int32)
        sign_w = np.where(typ < 0, np.float32(-1.0), np.float32(1.0))
        stored_names = recs.dtype.names

        def col(name: str) -> np.ndarray:
            if name in stored_names:
                return np.asarray(recs[name], dtype=np.float32)
            return np.full(n, self.header.constants[name], dtype=np.float32)

        u = col("u")
        v = col("v")
        tmp = u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2
        w = np.where(tmp <= 1.0, sign_w * np.sqrt(np.maximum(0.0, 1.0 - tmp)), 0.0).astype(
            np.float32
        )
        over = tmp > 1.0
        if over.any():  # degenerate direction: renormalize u, v (w stays 0)
            scale = np.sqrt(tmp[over]).astype(np.float32)
            u = u.copy()
            v = v.copy()
            u[over] /= scale
            v[over] /= scale

        return {
            "particle_type": np.abs(typ),
            "energy": np.abs(recs["e"].astype(np.float32)),
            "x": col("x"),
            "y": col("y"),
            "z": col("z"),
            "ux": u,
            "uy": v,
            "uz": w,
            "weight": col("weight"),
        }

    def close(self) -> None:
        """Release the memory map and file handle."""
        mm = getattr(self, "_mmap", None)
        if mm is not None:
            mm.close()
            self._mmap = None  # type: ignore[assignment]
        if getattr(self, "_file", None) is not None:
            assert self._file is not None
            self._file.close()
            self._file = None

    def __enter__(self) -> PhaseSpaceSource:
        """Return self; the map is already open from construction."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release the memory map and file handle."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup if the source was not closed explicitly."""
        self.close()
