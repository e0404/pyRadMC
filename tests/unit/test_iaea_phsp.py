"""IAEA phase-space reader: header parse + fixed-length binary record iteration.

Phase 5, phase-space source, Slice 1. Pure I/O — no physics. The byte layout is
the IAEA (NDS)-0484 format (Capote et al., Vienna 2006); this test pins it against
*hand-authored* record bytes with hand-computed expected fields, so the reader is
anchored to the external spec, not to a fixture writer that could share its bug.

Layout of one record, in read order (little-endian ``$BYTE_ORDER: 1234``):

* ``typ``  : int8. ``abs(typ)`` is the IAEA particle code (1=photon, 2=electron,
  3=positron). The *sign* of ``typ`` carries the sign of the third direction
  cosine ``w`` (``typ < 0`` -> ``w < 0``).
* ``E``    : float32. The *sign* of ``E`` flags the first particle of a new
  primary history (``E < 0`` -> ``new_history``); the stored energy is ``abs(E)``.
* ``x,y,z,u,v,weight`` : float32 each, but only those whose ``$RECORD_CONTENTS``
  flag is 1; a flag of 0 means the value is a header constant (``$RECORD_CONSTANT``).
* ``w`` is never stored: ``w = sign_w * sqrt(1 - u**2 - v**2)``.
* ``extra_floats`` (Nf x float32) then ``extra_ints`` (Ni x int32).
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from pyRadMC.geometry.phasespace import IAEAPhaseSpace, read_iaea_header

# IAEA particle codes.
PHOTON, ELECTRON, POSITRON = 1, 2, 3


def _header(
    *,
    stored: tuple[int, int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1, 1),
    constants: str = "",
    n_extra_floats: int = 0,
    n_extra_ints: int = 0,
    byte_order: int = 1234,
    record_length: int,
    particles: int,
) -> str:
    """A minimal but spec-valid ``.IAEAheader`` body.

    ``stored`` is the seven RECORD_CONTENTS flags (x,y,z,u,v,w,weight); ``//``
    comments are included deliberately so the parser is exercised against them.
    """
    flags = "\n".join(f"    {s}     // flag" for s in stored)
    return f"""$IAEA_INDEX:
1000

$FILE_TYPE:
0

$CHECKSUM:
{record_length * particles}

$RECORD_CONTENTS:
{flags}
    {n_extra_floats}     // Extra floats stored ?
    {n_extra_ints}     // Extra longs stored ?

$RECORD_CONSTANT:
{constants}
$RECORD_LENGTH:
{record_length}

$BYTE_ORDER:
{byte_order}

$ORIG_HISTORIES:
1

$PARTICLES:
{particles}
"""


def _write(tmp_path: Path, header: str, payload: bytes, stem: str = "case") -> Path:
    (tmp_path / f"{stem}.IAEAheader").write_text(header)
    (tmp_path / f"{stem}.IAEAphsp").write_bytes(payload)
    return tmp_path / f"{stem}.IAEAphsp"


class TestRecordDecoding:
    def test_single_photon_all_stored(self, tmp_path: Path) -> None:
        """A photon record with every coordinate stored + one extra int.

        Bytes and expected fields are authored by hand: a new-history photon at
        (3,4,5) cm, E=1 MeV, weight=2, direction (u,v)=(0.6,0.0) with w chosen
        negative via the particle-type sign, so ``w`` must reconstruct to exactly
        -0.8. One extra long = 13.
        """
        typ = -PHOTON  # negative -> w < 0
        e_signed = -1.0  # negative -> new history; |E| = 1 MeV
        payload = struct.pack(
            "<bfffffffi",
            typ,
            e_signed,
            3.0,  # x
            4.0,  # y
            5.0,  # z
            0.6,  # u
            0.0,  # v
            2.0,  # weight
            13,  # extra int
        )
        record_length = 1 + 4 + 4 * 6 + 4 * 1  # typ + E + 6 floats + 1 long = 33
        assert len(payload) == record_length
        header = _header(n_extra_ints=1, record_length=record_length, particles=1)
        phsp_path = _write(tmp_path, header, payload)

        with IAEAPhaseSpace(phsp_path) as phsp:
            assert len(phsp) == 1
            (rec,) = list(phsp)

        assert rec.particle_type == PHOTON
        assert rec.new_history is True
        assert rec.energy == pytest.approx(1.0)
        assert (rec.x, rec.y, rec.z) == pytest.approx((3.0, 4.0, 5.0))
        assert rec.u == pytest.approx(0.6)
        assert rec.v == pytest.approx(0.0)
        assert rec.w == pytest.approx(-0.8)  # sign from typ<0, |w|=sqrt(1-0.36)
        assert rec.weight == pytest.approx(2.0)
        assert rec.extra_ints == (13,)
        assert rec.extra_floats == ()
        # Direction is a unit vector.
        assert rec.u**2 + rec.v**2 + rec.w**2 == pytest.approx(1.0)

    def test_positive_type_gives_positive_w(self, tmp_path: Path) -> None:
        """Non-negative particle type -> w >= 0; non-negative energy -> continued history."""
        payload = struct.pack(
            "<bfffffff",
            ELECTRON,  # positive -> w >= 0, continued history since E > 0
            2.5,  # E
            0.0,
            0.0,
            0.0,
            0.0,  # u
            0.8,  # v -> w = +0.6
            1.0,  # weight
        )
        record_length = 1 + 4 + 4 * 6
        header = _header(record_length=record_length, particles=1)
        phsp_path = _write(tmp_path, header, payload)
        with IAEAPhaseSpace(phsp_path) as phsp:
            (rec,) = list(phsp)
        assert rec.particle_type == ELECTRON
        assert rec.new_history is False
        assert rec.energy == pytest.approx(2.5)
        assert rec.w == pytest.approx(0.6)


class TestConstants:
    def test_constant_coordinate_read_from_header(self, tmp_path: Path) -> None:
        """A z that is not stored per-particle is taken from RECORD_CONSTANT.

        With z's flag = 0, the record omits the z float (shorter record), and every
        particle's z is the header constant. Order of RECORD_CONSTANT follows the
        coordinate order x,y,z,u,v,weight (w never appears).
        """
        stored = (1, 1, 0, 1, 1, 1, 1)  # z not stored
        payload = struct.pack(
            "<bffffff",  # typ, E, x, y, u, v, weight (no z)
            PHOTON,
            6.0,  # E
            1.0,  # x
            2.0,  # y
            0.0,  # u
            0.0,  # v -> w = +1.0
            1.0,  # weight
        )
        record_length = 1 + 4 + 4 * 5  # z dropped
        header = _header(
            stored=stored,
            constants="-7.5     // z constant\n",
            record_length=record_length,
            particles=1,
        )
        phsp_path = _write(tmp_path, header, payload)
        with IAEAPhaseSpace(phsp_path) as phsp:
            (rec,) = list(phsp)
        assert rec.x == pytest.approx(1.0)
        assert rec.y == pytest.approx(2.0)
        assert rec.z == pytest.approx(-7.5)  # from the header constant
        assert rec.w == pytest.approx(1.0)


class TestHeaderAndIteration:
    def test_multiple_records_and_new_history_flags(self, tmp_path: Path) -> None:
        """Three particles: a fresh history, then two secondaries of it."""
        rec_len = 1 + 4 + 4 * 6

        def one(typ: int, e_signed: float) -> bytes:
            return struct.pack("<bfffffff", typ, e_signed, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        payload = one(PHOTON, -1.0) + one(ELECTRON, 0.3) + one(-POSITRON, 0.2)
        header = _header(record_length=rec_len, particles=3)
        phsp_path = _write(tmp_path, header, payload)
        with IAEAPhaseSpace(phsp_path) as phsp:
            assert len(phsp) == 3
            recs = list(phsp)
        assert [r.new_history for r in recs] == [True, False, False]
        assert [r.particle_type for r in recs] == [PHOTON, ELECTRON, POSITRON]

    def test_header_exposes_counts(self, tmp_path: Path) -> None:
        rec_len = 1 + 4 + 4 * 6
        header = _header(record_length=rec_len, particles=7)
        (tmp_path / "h.IAEAheader").write_text(header)
        hdr = read_iaea_header(tmp_path / "h.IAEAheader")
        assert hdr.n_particles == 7
        assert hdr.record_length == rec_len
        assert hdr.byte_order == "<"

    def test_record_length_mismatch_is_rejected(self, tmp_path: Path) -> None:
        """A RECORD_LENGTH inconsistent with the flags is a corrupt header, not silent."""
        header = _header(record_length=999, particles=1)  # flags imply 33, not 999
        (tmp_path / "bad.IAEAheader").write_text(header)
        with pytest.raises(ValueError, match="RECORD_LENGTH"):
            read_iaea_header(tmp_path / "bad.IAEAheader")

    def test_particles_off_by_one_warns_and_trusts_the_file(self, tmp_path: Path) -> None:
        """The on-disk record count wins; a stale PARTICLES header only warns.

        Real files disagree here (the Varian TrueBeam files declare one more particle
        than the phsp holds), so a mismatch must not be fatal: the size on disk is the
        count you can actually read.
        """
        rec_len = 1 + 4 + 4 * 6
        payload = struct.pack("<bfffffff", PHOTON, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        header = _header(record_length=rec_len, particles=2)  # header claims 2, file has 1
        phsp_path = _write(tmp_path, header, payload)
        with pytest.warns(UserWarning, match="PARTICLES header says 2"):
            hdr = read_iaea_header(phsp_path)
        assert hdr.n_particles == 1
        with (
            pytest.warns(UserWarning, match="using the on-disk count"),
            IAEAPhaseSpace(phsp_path) as phsp,
        ):
            assert len(list(phsp)) == 1

    def test_truncated_file_is_rejected(self, tmp_path: Path) -> None:
        """A file size that is not a whole number of records is corruption, not a warning."""
        rec_len = 1 + 4 + 4 * 6
        payload = struct.pack("<bfffffff", PHOTON, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)[:-3]
        header = _header(record_length=rec_len, particles=1)
        phsp_path = _write(tmp_path, header, payload)
        with pytest.raises(ValueError, match="not a multiple of record length"):
            read_iaea_header(phsp_path)

    def test_big_endian(self, tmp_path: Path) -> None:
        """A big-endian ($BYTE_ORDER: 4321) file decodes identically."""
        rec_len = 1 + 4 + 4 * 6
        payload = struct.pack(">bfffffff", PHOTON, 4.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0)
        header = _header(byte_order=4321, record_length=rec_len, particles=1)
        phsp_path = _write(tmp_path, header, payload)
        with IAEAPhaseSpace(phsp_path) as phsp:
            (rec,) = list(phsp)
        assert rec.energy == pytest.approx(4.0)
        assert (rec.x, rec.y, rec.z) == pytest.approx((1.0, 2.0, 3.0))


def test_degenerate_direction_is_renormalized(tmp_path: Path) -> None:
    """u**2 + v**2 > 1 (float noise) must not yield NaN: renormalize, w = 0."""
    rec_len = 1 + 4 + 4 * 6
    payload = struct.pack("<bfffffff", PHOTON, 1.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.0)
    header = _header(record_length=rec_len, particles=1)
    phsp_path = _write(tmp_path, header, payload)
    with IAEAPhaseSpace(phsp_path) as phsp:
        (rec,) = list(phsp)
    assert not math.isnan(rec.w)
    assert rec.w == pytest.approx(0.0)
    assert rec.u**2 + rec.v**2 + rec.w**2 == pytest.approx(1.0)
