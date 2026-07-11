"""BeamletGridSource: the stratified beamlet lattice (Phase 3).

The source is the semantic anchor of the Dij design: a beamlet is a rectangle of
the field, the lattice tiles the field exactly, and which beamlet a history feeds
is decided by the *caller* (deterministically, from the history index) — never
sampled. Positions within a beamlet consume exactly the two uniforms that
ParallelBeamSource consumes for the whole field, so a 1x1 lattice is bit-identical
to the open field. That equivalence is what lets the dij tier anchor Dij columns
to open-field dose runs.
"""

from __future__ import annotations

import math

import pytest

from pyRadMC.geometry.source import BeamletGridSource, ParallelBeamSource
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED

ENERGY = 6.0
FIELD = dict(energy=ENERGY, z=-1.0, x_range=(-4.0, 4.0), y_range=(-2.0, 6.0))


def _source(n_x: int = 4, n_y: int = 2) -> BeamletGridSource:
    return BeamletGridSource(n_x=n_x, n_y=n_y, **FIELD)


class TestLattice:
    def test_beamlet_count(self) -> None:
        assert _source(4, 2).n_beamlets == 8

    def test_bounds_tile_the_field_exactly(self) -> None:
        """Beamlet rectangles partition the field: no gaps, no overlap, exact edges."""
        src = _source(4, 2)
        x_edges = sorted({b[0] for b in map(src.beamlet_bounds, range(8))})
        y_edges = sorted({b[2] for b in map(src.beamlet_bounds, range(8))})
        assert x_edges == pytest.approx([-4.0, -2.0, 0.0, 2.0])
        assert y_edges == pytest.approx([-2.0, 2.0])
        for j in range(8):
            x_lo, x_hi, y_lo, y_hi = src.beamlet_bounds(j)
            assert x_hi - x_lo == pytest.approx(2.0)
            assert y_hi - y_lo == pytest.approx(4.0)
        # The union covers the field: upper edges of the last row/column are exact.
        assert src.beamlet_bounds(7)[1] == pytest.approx(4.0)
        assert src.beamlet_bounds(7)[3] == pytest.approx(6.0)

    def test_index_convention_is_x_major(self) -> None:
        """j = jx * n_y + jy: consecutive j sweeps y fastest. Convention is pinned."""
        src = _source(4, 2)
        assert src.beamlet_bounds(0)[:1] == (pytest.approx(-4.0),)
        assert src.beamlet_bounds(1)[2] == pytest.approx(2.0)  # j=1 is (jx=0, jy=1)
        assert src.beamlet_bounds(2)[0] == pytest.approx(-2.0)  # j=2 is (jx=1, jy=0)


class TestEmit:
    def test_position_lands_inside_the_beamlet(self) -> None:
        src = _source(4, 2)
        rng = HostRNG()
        for j in range(src.n_beamlets):
            x_lo, x_hi, y_lo, y_hi = src.beamlet_bounds(j)
            for h in range(50):
                p = src.emit(j, rng.init_state(SEED, h))
                assert x_lo <= p.x < x_hi
                assert y_lo <= p.y < y_hi
                assert p.z == FIELD["z"]
                assert (p.ux, p.uy, p.uz) == (0.0, 0.0, 1.0)
                assert p.energy == ENERGY

    def test_single_beamlet_is_bit_identical_to_the_open_field(self) -> None:
        """A 1x1 lattice consumes the same two uniforms as ParallelBeamSource.

        This is the anchor for the dij-tier test that a one-beamlet Dij run
        reproduces the open-field dose bit for bit on one target.
        """
        beamlet = _source(1, 1)
        open_field = ParallelBeamSource(**FIELD)
        rng = HostRNG()
        for h in range(100):
            a = beamlet.emit(0, rng.init_state(SEED, h))
            b = open_field.emit(rng.init_state(SEED, h))
            assert a == b

    def test_emit_rejects_out_of_range_beamlet(self) -> None:
        src = _source(2, 2)
        rng = HostRNG()
        with pytest.raises(IndexError):
            src.emit(4, rng.init_state(SEED, 0))
        with pytest.raises(IndexError):
            src.emit(-1, rng.init_state(SEED, 0))


class TestValidation:
    def test_rejects_non_positive_energy(self) -> None:
        with pytest.raises(ValueError):
            BeamletGridSource(
                energy=0.0, z=0.0, x_range=(0.0, 1.0), y_range=(0.0, 1.0), n_x=1, n_y=1
            )

    def test_rejects_empty_field(self) -> None:
        with pytest.raises(ValueError):
            BeamletGridSource(
                energy=1.0, z=0.0, x_range=(1.0, 1.0), y_range=(0.0, 1.0), n_x=1, n_y=1
            )

    def test_rejects_non_positive_lattice(self) -> None:
        for n_x, n_y in ((0, 1), (1, 0), (-2, 3)):
            with pytest.raises(ValueError):
                BeamletGridSource(
                    energy=1.0, z=0.0, x_range=(0.0, 1.0), y_range=(0.0, 1.0), n_x=n_x, n_y=n_y
                )


def test_bounds_are_exact_at_the_field_edges() -> None:
    """float arithmetic must not let the last beamlet fall short of the field edge."""
    src = BeamletGridSource(energy=1.0, z=0.0, x_range=(0.0, 0.7), y_range=(0.0, 0.3), n_x=7, n_y=3)
    assert src.beamlet_bounds(src.n_beamlets - 1)[1] == src.x_range[1]
    assert src.beamlet_bounds(src.n_beamlets - 1)[3] == src.y_range[1]
    widths = [src.beamlet_bounds(j)[1] - src.beamlet_bounds(j)[0] for j in range(src.n_beamlets)]
    assert all(math.isclose(w, 0.1, rel_tol=1e-12) for w in widths)
