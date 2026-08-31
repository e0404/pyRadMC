"""Mono-energetic pencil-beam kernels: cylindrical scoring through every engine.

A pencil-beam kernel is the dose an infinitely narrow mono-energetic beam deposits in
a homogeneous water medium, binned by depth and by radius about the beam axis. The
geometry that makes it is a *scoring* choice, not a physics one — transport still runs
on the rectilinear water phantom, Woodcock tracking and all — so the properties split
cleanly in two:

**Exact**, because the RNG streams never see the scoring geometry:

- the energy ledger ``emitted == deposited + unscored + escaped`` still closes;
- ``emitted`` and ``escaped`` are *bit-identical* to the same run scored on the
  transport grid. If they ever differ, the scoring geometry has leaked into
  transport, which is the one thing this design must not do.

**Statistical**, because they compare targets (AGENTS.md 2.3):

- the reference oracle against Warp cpu/cuda, via the chi-squared detection oracle.

Bit equality across targets is never asserted here, and the reference is never
adjusted to match a backend (AGENTS.md 2.2).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource
from pyradmc.rng.host import HostRNG
from pyradmc.scoring.cylinder import (
    CylindricalScoringGrid,
    geometric_edges,
    uniform_edges,
)
from pyradmc.scoring.grid import ScoringGrid
from tests.conftest import SEED, assert_chi2_consistent_batched

ENERGY = 6.0
"""Primary energy in MeV: inside the 1-20 MeV domain this engine is tuned for."""

ENERGY_BALANCE_RTOL = 1.0e-4  # warp: float32 transport + 1e-9 MeV scoring quanta


def _grid() -> VoxelGrid:
    """Water box wide enough that the scoring cylinder sits well inside it.

    The kernel is defined in a semi-infinite medium; the box stands in for that,
    and the cylinder only ever bins a region the box fully surrounds, so the outer
    shells still see full lateral scatter.
    """
    return VoxelGrid.uniform_water(shape=(24, 24, 20), spacing=(0.5, 0.5, 0.5))


def _axis() -> tuple[float, float]:
    return (6.0, 6.0)  # the lateral centre of the 12 x 12 cm box


def _source() -> PencilBeamSource:
    """Infinitely narrow mono-energetic beam on the axis, entering at z = 0."""
    x, y = _axis()
    return PencilBeamSource(energy=ENERGY, position=(x, y, -0.1), direction=(0.0, 0.0, 1.0))


def _cylinder(grid: VoxelGrid) -> CylindricalScoringGrid:
    return CylindricalScoringGrid.for_grid(
        grid,
        radial_edges=geometric_edges(r_max=5.0, n_shells=12, r_min=0.1),
        axis=_axis(),
        depth_edges=uniform_edges(10.0, 20),
    )


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _ref_engine(grid: VoxelGrid) -> ReferenceEngine:
    return ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())


# ---------------------------------------------------------------------------
# Exact properties on the reference oracle
# ---------------------------------------------------------------------------


class TestTransportIsInvariantToTheScoringGeometry:
    """Choosing cylindrical bins must not move a single history."""

    def test_emitted_and_escaped_are_bit_identical_to_grid_scoring(self) -> None:
        """The two runs differ only in where deposits are filed, never in physics.

        ``energy_emitted`` and ``energy_escaped`` are accumulated in the transport
        loop, upstream of the scorer, so on one target they must agree to the last
        bit — not merely to a tolerance.
        """
        grid = _grid()
        engine = _ref_engine(grid)
        common = {"n_histories": 400, "n_batches": 8, "seed": SEED}
        on_grid = engine.run(_source(), scoring_grid=None, **common)
        on_cylinder = engine.run(_source(), scoring_grid=_cylinder(grid), **common)

        assert on_cylinder.energy_emitted == on_grid.energy_emitted
        assert on_cylinder.energy_escaped == on_grid.energy_escaped

    def test_total_scored_energy_is_unchanged_where_the_regions_coincide(self) -> None:
        """Cylinder plus its unscored remainder equals the whole box's deposit.

        Both runs deposit the same energy inside the transport grid; the cylindrical
        run merely books the part outside its shells to the unscored bucket rather
        than to a voxel. Their sum is therefore the grid run's deposit, exactly.
        """
        grid = _grid()
        engine = _ref_engine(grid)
        common = {"n_histories": 400, "n_batches": 8, "seed": SEED}
        on_grid = engine.run(_source(), scoring_grid=None, **common)
        on_cylinder = engine.run(_source(), scoring_grid=_cylinder(grid), **common)

        inside_box = on_cylinder.energy_deposited + on_cylinder.energy_unscored
        assert inside_box == pytest.approx(on_grid.energy_deposited, rel=1e-12)


class TestEnergyLedger:
    def test_ledger_closes_on_the_reference(self) -> None:
        """``emitted == deposited + unscored + escaped``, float64 exact."""
        grid = _grid()
        result = _ref_engine(grid).run(
            _source(),
            n_histories=400,
            n_batches=8,
            seed=SEED,
            scoring_grid=_cylinder(grid),
        )
        total = result.energy_deposited + result.energy_unscored + result.energy_escaped
        assert total == pytest.approx(result.energy_emitted, rel=1e-12)

    def test_shell_dose_times_mass_reproduces_the_scored_energy(self) -> None:
        """The mass normalization is invertible: sum(dose * mass * n) == deposited.

        This is what pins the annulus mass map *through* the engine rather than only
        in the constructor's own arithmetic — a shell mass wrong by a factor would
        pass the unit test's formula and fail here.
        """
        grid = _grid()
        cyl = _cylinder(grid)
        n_histories = 400
        result = _ref_engine(grid).run(
            _source(),
            n_histories=n_histories,
            n_batches=8,
            seed=SEED,
            scoring_grid=cyl,
        )
        recovered = float((result.dose * cyl.voxel_mass).sum()) * n_histories
        assert recovered == pytest.approx(result.energy_deposited, rel=1e-10)


class TestKernelShape:
    """Coarse physical sanity, at the level a shape error would break."""

    def test_dose_falls_monotonically_with_radius_at_the_entrance(self) -> None:
        """A pencil beam's lateral profile is steeply peaked on the axis.

        Not a tolerance on the values — just the ordering, which a transposed
        index or a swapped mass map destroys immediately.
        """
        grid = _grid()
        cyl = _cylinder(grid)
        result = _ref_engine(grid).run(
            _source(), n_histories=2_000, n_batches=8, seed=SEED, scoring_grid=cyl
        )
        near_surface = result.dose[1]  # second depth bin: past the build-up edge
        assert near_surface[0] > near_surface[3] > near_surface[-1]

    def test_almost_all_deposited_energy_lands_inside_a_wide_cylinder(self) -> None:
        """A 5 cm radius captures the overwhelming majority of a 6 MeV pencil beam.

        Guards the containment convention: an inverted radial test would push most
        of the energy into the unscored bucket instead.
        """
        grid = _grid()
        result = _ref_engine(grid).run(
            _source(), n_histories=800, n_batches=8, seed=SEED, scoring_grid=_cylinder(grid)
        )
        inside_box = result.energy_deposited + result.energy_unscored
        assert result.energy_deposited / inside_box > 0.95


class TestElectronPrimaries:
    """``primary_kind='electron'`` — the existing range-validation instrument.

    Electron *beams* as a clinical modality stay out of scope (AGENTS.md 6); this
    exercises the same sanctioned instrument the engine already documents, now read
    out in the depth-and-radius binning where an electron range is legible.
    """

    def test_electron_pencil_beam_stops_within_its_csda_range(self) -> None:
        """A 6 MeV electron in water has a CSDA range of about 3.0 cm (ICRU 37).

        Beyond it only bremsstrahlung reaches, so the deposited energy past 4 cm
        must be a small fraction of the total — a bound loose enough to be about
        the range and not about the tolerance.
        """
        grid = _grid()
        cyl = _cylinder(grid)
        result = _ref_engine(grid).run(
            _source(),
            n_histories=400,
            n_batches=8,
            seed=SEED,
            scoring_grid=cyl,
            primary_kind="electron",
        )
        energy_per_depth = (result.dose * cyl.voxel_mass).sum(axis=1)
        beyond_range = energy_per_depth[cyl.depth_edges[:-1] >= 4.0].sum()
        assert beyond_range / energy_per_depth.sum() < 0.05

    def test_electron_ledger_closes(self) -> None:
        grid = _grid()
        result = _ref_engine(grid).run(
            _source(),
            n_histories=200,
            n_batches=4,
            seed=SEED,
            scoring_grid=_cylinder(grid),
            primary_kind="electron",
        )
        total = result.energy_deposited + result.energy_unscored + result.energy_escaped
        assert total == pytest.approx(result.energy_emitted, rel=1e-12)


# ---------------------------------------------------------------------------
# Cross-target: statistical equivalence only (AGENTS.md 2.3)
# ---------------------------------------------------------------------------

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


@pytest.fixture(scope="module")
def reference_kernel():
    """One reference run, shared by every device under test."""
    grid = _grid()
    cyl = _cylinder(grid)
    return cyl, _ref_engine(grid).run(
        _source(), n_histories=8_000, n_batches=8, seed=SEED, scoring_grid=cyl
    )


def _warp_engine(grid: VoxelGrid, device: str):
    from pyradmc.backends.warp.engine import WarpEngine

    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device)


@pytest.mark.warp
class TestWarpMatchesTheReferenceKernel:
    def test_kernel_is_statistically_consistent_with_the_oracle(
        self, device: str, reference_kernel
    ) -> None:
        """Chi-squared over the shells both estimates resolve (AGENTS.md 4)."""
        cyl, ref = reference_kernel
        grid = _grid()
        got = _warp_engine(grid, device).run(
            _source(), n_histories=8_000, n_batches=8, seed=SEED, scoring_grid=cyl
        )
        mask = (ref.dose > 0.01 * ref.dose.max()) & (ref.dose_sigma > 0.0)
        mask &= got.dose_sigma > 0.0
        assert np.count_nonzero(mask) > 40, "mask too small to detect anything"
        assert_chi2_consistent_batched(
            ref.dose, ref.dose_sigma, 8, got.dose, got.dose_sigma, 8, mask=mask
        )

    def test_ledger_closes_on_the_device(self, device: str) -> None:
        grid = _grid()
        result = _warp_engine(grid, device).run(
            _source(), n_histories=2_000, n_batches=8, seed=SEED, scoring_grid=_cylinder(grid)
        )
        total = result.energy_deposited + result.energy_unscored + result.energy_escaped
        assert total == pytest.approx(result.energy_emitted, rel=ENERGY_BALANCE_RTOL)

    def test_repeated_runs_are_bit_reproducible_on_one_device(self, device: str) -> None:
        """Within one target and seed, exact (AGENTS.md 2.3) — fixed-point scoring.

        The radial search is data-dependent branching, so this also pins that the
        shell lookup introduced no scheduling sensitivity.
        """
        grid = _grid()
        engine = _warp_engine(grid, device)
        common = {
            "n_histories": 1_000,
            "n_batches": 4,
            "seed": SEED,
            "scoring_grid": _cylinder(grid),
        }
        a = engine.run(_source(), **common)
        b = engine.run(_source(), **common)
        np.testing.assert_array_equal(a.dose, b.dose)
        np.testing.assert_array_equal(a.dose_sigma, b.dose_sigma)

    def test_rectilinear_scoring_is_untouched_by_the_cylindrical_path(self, device: str) -> None:
        """The default grid route must be bit-identical to before this feature.

        Cylindrical routing entered the kernels' single ``_deposit`` choke point;
        this pins that the rectilinear branch through it did not shift.
        """
        grid = _grid()
        engine = _warp_engine(grid, device)
        common = {"n_histories": 1_000, "n_batches": 4, "seed": SEED}
        default = engine.run(_source(), scoring_grid=None, **common)
        explicit = engine.run(_source(), scoring_grid=ScoringGrid.for_grid(grid), **common)
        np.testing.assert_array_equal(default.dose, explicit.dose)
        assert default.energy_deposited == explicit.energy_deposited
