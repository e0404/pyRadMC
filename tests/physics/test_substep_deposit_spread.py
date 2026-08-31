"""Sub-step deposit spreading: where a substep's continuous energy loss is filed.

The condensed-history loop charges each half-substep's collision loss as a *point*
deposit at that half's midpoint. That is invisible while dose is scored at the
transport voxel scale, and wrong below it: when substeps are boundary-limited the
half-step endpoints are pinned to voxel faces, so every deposit pair lands in the
same place and the sub-voxel dose profile becomes a tent — peaked at the voxel
centre, zero at the faces.

``deposit_resolution_cm`` splits each half-step into pieces no longer than the given
length and files one deposit per piece, which resolves the deposit to whatever scale
the scorer actually bins at. ``None`` (the default) is exactly one deposit at the
midpoint, i.e. the behaviour every existing result was produced with, and that
identity is pinned here rather than assumed.

These are unit tests on the deposit *positions*, not on dose: the mechanism is
cheap to check exactly, and the dosimetric consequence is measured in the
integration tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.rng.host import HostRNG
from pyRadMC.transport.electron import electron_steps, substep_deposit_count


class TestSubstepDepositCount:
    """The piece count is ceil(length / resolution), clamped."""

    def test_none_resolution_is_a_single_deposit(self) -> None:
        """The default must be exactly one piece — the midpoint deposit, unchanged."""
        assert substep_deposit_count(1.0, None) == 1
        assert substep_deposit_count(0.0, None) == 1

    def test_count_covers_the_length_at_the_requested_resolution(self) -> None:
        assert substep_deposit_count(0.10, 0.025) == 4
        assert substep_deposit_count(0.101, 0.025) == 5  # ceil, never floor
        assert substep_deposit_count(0.025, 0.025) == 1
        assert substep_deposit_count(0.0, 0.025) == 1  # degenerate step still deposits once

    def test_count_is_capped(self) -> None:
        """A tiny resolution against a long step must not explode the deposit count.

        The cap bounds the cost of a mis-specified resolution; past it the deposit
        is simply coarser than asked, which is a resolution shortfall and not a
        correctness failure.
        """
        n = substep_deposit_count(1.0e6, 1.0e-9)
        assert 1 < n <= 1024

    def test_resolution_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            substep_deposit_count(1.0, 0.0)
        with pytest.raises(ValueError, match="positive"):
            substep_deposit_count(1.0, -0.1)


def _collect(resolution, energy=6.0, seed=7):
    """Transport one 6 MeV electron; return (positions, amounts) of every deposit."""
    grid = VoxelGrid.uniform_water(shape=(40, 40, 40), spacing=(0.25, 0.25, 0.25))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    positions: list[tuple[float, float, float]] = []
    amounts: list[float] = []

    def deposit(x, y, z, amount, scored):
        positions.append((x, y, z))
        amounts.append(amount)

    escaped = electron_steps(
        False,
        energy,
        1.0,
        5.0,
        5.0,
        0.5,
        0.0,
        0.0,
        1.0,
        grid,
        xs,
        HostRNG().init_state(20260711, seed),
        deposit,
        lambda entry: None,
        pcut=0.050,
        ecut=0.200,
        deposit_resolution_cm=resolution,
    )
    return np.array(positions), np.array(amounts), escaped


class TestDepositPlacement:
    def test_default_is_unchanged_bit_for_bit(self) -> None:
        """``None`` must reproduce the midpoint deposits exactly, not merely closely.

        Every archived result was produced on this path; if it moves at all, it
        moves silently and every comparison against an older run becomes invalid.
        """
        a_pos, a_amt, a_esc = _collect(None)
        b_pos, b_amt, b_esc = _collect(None)
        np.testing.assert_array_equal(a_pos, b_pos)
        np.testing.assert_array_equal(a_amt, b_amt)
        assert a_esc == b_esc
        assert len(a_pos) > 10, "the test electron must actually deposit something"

    def test_spreading_increases_the_deposit_count(self) -> None:
        coarse_pos, _, _ = _collect(None)
        fine_pos, _, _ = _collect(0.01)
        assert len(fine_pos) > 3 * len(coarse_pos)

    def test_spreading_conserves_energy_exactly(self) -> None:
        """Splitting a deposit must move energy, never create or destroy it.

        Same history, same RNG stream, same trajectory — only the filing changes,
        so the books must agree to float64 accumulation.
        """
        _, coarse_amt, coarse_esc = _collect(None)
        _, fine_amt, fine_esc = _collect(0.01)
        assert coarse_amt.sum() + coarse_esc == pytest.approx(fine_amt.sum() + fine_esc, rel=1e-12)

    def test_pieces_lie_along_the_step_and_share_it_equally(self) -> None:
        """The first substep's deposits are evenly spaced midpoints of equal pieces.

        A straight-ahead electron starting on the axis: the leading deposits share
        one half-step, so their z positions must be equally spaced and their
        amounts equal. Uneven spacing would bias where energy lands within a step,
        which is the very thing this feature exists to fix.
        """
        pos, amt, _ = _collect(0.01)
        z = pos[:, 2]
        # Leading run of deposits with a common spacing: the first half-step.
        gaps = np.diff(z[:6])
        assert np.all(gaps > 0.0)
        np.testing.assert_allclose(gaps, gaps[0], rtol=1e-9)
        np.testing.assert_allclose(amt[:6], amt[0], rtol=1e-12)
        # First midpoint sits half a piece in, not at the step start.
        assert z[0] == pytest.approx(0.5 + 0.5 * gaps[0], rel=1e-9)
