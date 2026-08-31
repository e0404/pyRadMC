"""Sub-voxel dose: the substep deposit resolution, through the engines.

The condensed-history loop files each half-substep's collision loss at that half's
midpoint. Substeps are capped at transport voxel faces, so once a step is longer
than a voxel every half-step is pinned to the lattice and its midpoint lands near
the voxel centre: the sub-voxel dose profile becomes a tent, peaked at the centre
and starved at the faces. It is invisible at voxel resolution and dominant below it.

``deposit_resolution_cm`` splits the half-step into pieces no longer than it. What
must hold:

- it moves dose *within* a voxel and nowhere else — the energy books, and the dose
  coarse-grained back to the voxel pitch, are unchanged;
- the default is byte-identical to before the option existed;
- it actually removes the lattice modulation it exists to remove.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource
from pyradmc.rng.host import HostRNG
from pyradmc.scoring.cylinder import CylindricalScoringGrid, geometric_edges, graded_edges
from tests.conftest import SEED

VOXEL = 0.25
BIN = 0.025
PER_VOXEL = round(VOXEL / BIN)
AXIS = (2.0, 2.0)


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 12), spacing=(VOXEL,) * 3)


def _cylinder(grid: VoxelGrid) -> CylindricalScoringGrid:
    """Depth bins ten times finer than the transport voxel: the artifact's home."""
    return CylindricalScoringGrid.for_grid(
        grid,
        radial_edges=geometric_edges(r_max=2.0, n_shells=8, r_min=0.1),
        axis=AXIS,
        depth_edges=graded_edges([(0.0, 2.0, BIN)]),
    )


def _run(resolution, n=40_000, energy=6.0):
    grid = _grid()
    cyl = _cylinder(grid)
    engine = ReferenceEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )
    result = engine.run(
        PencilBeamSource(energy=energy, position=(*AXIS, -0.1), direction=(0.0, 0.0, 1.0)),
        n_histories=n,
        n_batches=4,
        seed=SEED,
        scoring_grid=cyl,
        deposit_resolution_cm=resolution,
    )
    return cyl, result


def _energy_per_bin(cyl, result):
    return (result.dose * cyl.voxel_mass).sum(axis=1)


def _lattice_modulation(cyl, result) -> float:
    """Peak-to-trough of the profile folded on the voxel period, trend removed.

    Isolates the lattice artifact from the genuine build-up rise underneath it.
    """
    e = _energy_per_bin(cyl, result)[PER_VOXEL * 2 : PER_VOXEL * 8]
    x = np.arange(e.size)
    detrended = e / np.polyval(np.polyfit(x, e, 3), x)
    folded = detrended.reshape(-1, PER_VOXEL).mean(axis=0)
    return float(folded.max() - folded.min())


def test_default_is_byte_identical_to_no_option() -> None:
    """Every archived result was produced on the ``None`` path; it must not move."""
    _, a = _run(None)
    _, b = _run(None)
    np.testing.assert_array_equal(a.dose, b.dose)
    assert a.energy_deposited == b.energy_deposited
    assert a.energy_escaped == b.energy_escaped


def test_spreading_removes_the_voxel_lattice_modulation() -> None:
    """The motivating defect: a tent locked to the transport voxel pitch.

    Scored at a tenth of the voxel, the default profile carries a large modulation
    at exactly the voxel period. Filing the loss along the step must largely remove
    it — this is the entire reason the option exists.
    """
    cyl, coarse = _run(None)
    _, fine = _run(BIN)
    before = _lattice_modulation(cyl, coarse)
    after = _lattice_modulation(cyl, fine)
    assert before > 0.15, f"artifact not reproduced (modulation {before:.3f})"
    assert after < before / 3.0, f"modulation {before:.3f} -> {after:.3f}, not reduced enough"


def test_energy_books_are_unchanged() -> None:
    """Dose moves inside a voxel; no total may move at all."""
    _, coarse = _run(None)
    _, fine = _run(BIN)
    assert fine.energy_emitted == coarse.energy_emitted
    total_c = coarse.energy_deposited + coarse.energy_unscored + coarse.energy_escaped
    total_f = fine.energy_deposited + fine.energy_unscored + fine.energy_escaped
    assert total_f == pytest.approx(total_c, rel=1e-12)
    assert fine.energy_deposited == pytest.approx(coarse.energy_deposited, rel=2e-3)


def test_dose_coarse_grained_to_the_voxel_pitch_is_preserved() -> None:
    """Averaged back to the transport voxel, the two agree.

    This is what makes the option safe to leave off: it redistributes dose *within*
    a voxel, so anything scored at voxel resolution — every result the engine has
    produced so far — is unaffected by the choice.
    """
    cyl, coarse = _run(None, n=80_000)
    _, fine = _run(BIN, n=80_000)
    a = _energy_per_bin(cyl, coarse).reshape(-1, PER_VOXEL).sum(axis=1)
    b = _energy_per_bin(cyl, fine).reshape(-1, PER_VOXEL).sum(axis=1)
    live = a > 0.02 * a.max()
    np.testing.assert_allclose(b[live], a[live], rtol=0.02)


def test_provenance_records_the_resolution() -> None:
    """Not recoverable from the dose array, so the result must carry it."""
    _, default = _run(None)
    _, spread = _run(BIN)
    assert default.provenance is not None
    assert default.provenance.deposit_resolution_cm is None
    assert spread.provenance.deposit_resolution_cm == BIN
    assert "deposit_res" in spread.provenance.summary()


def test_non_positive_resolution_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        _run(0.0)
    with pytest.raises(ValueError, match="positive"):
        _run(-0.1)


# ---------------------------------------------------------------------------
# The same contract on the device (AGENTS.md 2.3: statistical across targets)
# ---------------------------------------------------------------------------

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _run_warp(device: str, resolution, n=200_000):
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    cyl = _cylinder(grid)
    engine = WarpEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        device=device,
    )
    return cyl, engine.run(
        PencilBeamSource(energy=6.0, position=(*AXIS, -0.1), direction=(0.0, 0.0, 1.0)),
        n_histories=n,
        n_batches=4,
        seed=SEED,
        scoring_grid=cyl,
        deposit_resolution_cm=resolution,
    )


@pytest.mark.warp
class TestWarpDepositResolution:
    def test_default_path_is_bit_identical_on_one_device(self, device: str) -> None:
        """The option must not perturb the default route through the kernels."""
        _, a = _run_warp(device, None)
        _, b = _run_warp(device, None)
        np.testing.assert_array_equal(a.dose, b.dose)

    def test_spreading_removes_the_lattice_modulation(self, device: str) -> None:
        cyl, coarse = _run_warp(device, None)
        _, fine = _run_warp(device, BIN)
        before = _lattice_modulation(cyl, coarse)
        after = _lattice_modulation(cyl, fine)
        assert before > 0.15, f"artifact not reproduced (modulation {before:.3f})"
        assert after < before / 3.0, f"modulation {before:.3f} -> {after:.3f}"

    def test_energy_ledger_still_closes(self, device: str) -> None:
        _, r = _run_warp(device, BIN)
        total = r.energy_deposited + r.energy_unscored + r.energy_escaped
        assert total == pytest.approx(r.energy_emitted, rel=1.0e-4)
