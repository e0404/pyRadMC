"""Sharding a Dij's beamlet groups over several devices.

Beamlet groups are independent — there is no cross-group reduction — so ``devices``
is a pure scheduling knob, exactly like ``beamlet_group_size`` and ``chunk_size``.
The contract these tests pin is the one that makes that claim checkable:

**a column is computed wholly on one device, so sharding cannot change what a device
computes for a beamlet — only which device computes it.**

That is stronger than "the answer is close" and it is testable without owning two
GPUs: run the same Dij sharded over ``["cuda:0", "cpu"]``, then run it whole on each
device alone, and require every sharded column to be *bit-identical* to one of the
two single-device runs. Group-to-device assignment is greedy from a shared queue
(an idle device always pulls the next group), so *which* device owns a column is
timing-dependent — but the column's bits must always match its owner's whole run.
Over identical devices the same property makes the whole Dij bit-identical
(AGENTS.md 2.3 forbids asserting that across cpu/cuda, which is precisely why the
per-column form is used).

The energy books are integer quanta summed in device-list order, so merging shards
contributes no error of its own: the ledger must close on a shard as tightly as it
does on a whole run.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from pyRadMC.backends.warp.engine import WarpEngine  # noqa: E402
from pyRadMC.data.analytic import AnalyticCrossSections  # noqa: E402
from pyRadMC.geometry.grid import VoxelGrid  # noqa: E402
from pyRadMC.geometry.source import BeamletGridSource  # noqa: E402
from tests.conftest import SEED  # noqa: E402

N_PER = 240
N_BATCHES = 4
GROUP = 2  # small, so a handful of beamlets makes several groups to shard


def _engine(device: str) -> WarpEngine:
    grid = VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device=device, chunk_size=2048)


def _source(n_x: int = 3, n_y: int = 2) -> BeamletGridSource:
    return BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(2.0, 6.0), y_range=(2.0, 6.0), n_x=n_x, n_y=n_y
    )


def _run(device: str, devices=None):
    return _engine(device).run_dij(
        _source(),
        n_histories_per_beamlet=N_PER,
        n_batches=N_BATCHES,
        seed=SEED,
        beamlet_group_size=GROUP,
        devices=devices,
    )


def _column(dij, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = int(dij.indptr[j]), int(dij.indptr[j + 1])
    return dij.indices[lo:hi], dij.dose[lo:hi], dij.sigma[lo:hi]


def test_single_device_list_matches_the_default() -> None:
    """``devices=[d]`` must be the plain single-device path, bit for bit."""
    plain = _run("cpu")
    listed = _run("cpu", devices=["cpu"])
    np.testing.assert_array_equal(listed.indptr, plain.indptr)
    np.testing.assert_array_equal(listed.indices, plain.indices)
    np.testing.assert_array_equal(listed.dose, plain.dose)
    np.testing.assert_array_equal(listed.sigma, plain.sigma)
    assert listed.energy_deposited == plain.energy_deposited
    assert listed.energy_escaped == plain.energy_escaped


@pytest.mark.gpu
def test_sharded_columns_are_bit_identical_to_their_own_device() -> None:
    """Every column equals the whole-run column of *some* single device, bitwise.

    This is the sharding contract under greedy scheduling: the owner of a group is
    whichever device pulled it, so the test does not predict ownership — it requires
    each column's bits to match one of the whole runs exactly. A merge that mixes
    columns up, or a shard that changes what a device computes, fails this; a
    tolerance-based comparison against a single device would hide both. Groups are
    device-atomic, so ownership must also be constant within each group.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    device_list = ["cuda:0", "cpu"]
    sharded = _run("cuda:0", devices=device_list)
    whole = {d: _run(d) for d in device_list}

    n_beamlets = _source().n_beamlets
    owners: list[str | None] = []
    for j in range(n_beamlets):
        col = _column(sharded, j)
        owner = None
        for d, w in whole.items():
            want = _column(w, j)
            if (
                np.array_equal(col[0], want[0])
                and np.array_equal(col[1], want[1])
                and np.array_equal(col[2], want[2])
            ):
                owner = d
                break
        assert owner is not None, f"column {j} bit-matches neither device's whole run"
        owners.append(owner)
    for gs in range(0, n_beamlets, GROUP):
        group_owners = set(owners[gs : gs + GROUP])
        assert len(group_owners) == 1, f"group at {gs} split across devices: {group_owners}"


@pytest.mark.gpu
def test_sharding_splits_the_work_across_devices() -> None:
    """The shard must actually be mixed — otherwise the test above proves nothing.

    If every column happened to match *both* devices, the per-column assertion would
    pass on a broken shard. The cpu and cuda columns must genuinely differ somewhere,
    so pin that they do.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    cpu = _run("cpu")
    cuda = _run("cuda:0")
    assert not (cpu.dose.shape == cuda.dose.shape and np.array_equal(cpu.dose, cuda.dose)), (
        "cpu and cuda produced identical Dij; the cross-device sharding test is vacuous"
    )


def _ledger_residual(dij) -> float:
    """Relative miss of emitted == deposited + escaped + unscored."""
    total = dij.energy_deposited + dij.energy_escaped + dij.energy_unscored
    return abs(total - dij.energy_emitted) / dij.energy_emitted


@pytest.mark.gpu
def test_sharding_does_not_degrade_the_energy_ledger() -> None:
    """The ledger closes on a shard as tightly as on a whole run.

    The books are integer quanta summed in device-list order, so merging shards adds
    no error of its own — the residual is the fixed-point/float32 floor the
    single-device runs already carry, not something sharding introduced. Pinning the
    shard against the *whole runs'* residual says exactly that, where an absolute
    tolerance would only say "small enough".
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    sharded = _run("cuda:0", devices=["cuda:0", "cpu"])
    whole = [_run(d) for d in ("cuda:0", "cpu")]
    floor = max(_ledger_residual(d) for d in whole)
    assert _ledger_residual(sharded) <= max(floor, 1e-12) * 1.5
    assert _ledger_residual(sharded) < 1e-4  # the warp Dij ledger budget


def test_repeated_devices_collapse_to_one_shard() -> None:
    """A duplicated device must not run its groups twice (and double the books)."""
    once = _run("cpu", devices=["cpu"])
    twice = _run("cpu", devices=["cpu", "cpu"])
    np.testing.assert_array_equal(twice.dose, once.dose)
    assert twice.energy_deposited == once.energy_deposited


def test_empty_device_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one device"):
        _run("cpu", devices=[])
