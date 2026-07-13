"""Voxel grid indexing and source emission.

Indexing errors do not crash a Monte Carlo code; they shift dose by one voxel. The
edge-case tests here are exact, deliberately.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tests.conftest import SEED


class TestVoxelGrid:
    def test_uniform_water_construction(self) -> None:
        from pyRadMC.data.materials import WATER
        from pyRadMC.geometry.grid import VoxelGrid

        grid = VoxelGrid.uniform_water(shape=(4, 5, 6), spacing=(0.1, 0.2, 0.3))
        assert grid.density.shape == (4, 5, 6)
        assert grid.material.shape == (4, 5, 6)
        assert np.all(grid.density == 1.0)
        assert np.all(grid.material == WATER)
        assert grid.voxel_volume == pytest.approx(0.1 * 0.2 * 0.3)

    def test_voxel_index_maps_corners_and_boundaries(self) -> None:
        """Half-open voxels [lo, hi): a boundary position belongs to the upper voxel."""
        from pyRadMC.geometry.grid import VoxelGrid

        grid = VoxelGrid.uniform_water(
            shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0), origin=(-2.0, -2.0, -2.0)
        )
        assert grid.voxel_index(-2.0, -2.0, -2.0) == (0, 0, 0)
        assert grid.voxel_index(-1.999, -1.001, -0.5) == (0, 0, 1)
        # Internal boundary at x = -1.0 belongs to voxel 1.
        assert grid.voxel_index(-1.0, -2.0, -2.0) == (1, 0, 0)
        # Last interior position.
        assert grid.voxel_index(1.999, 1.999, 1.999) == (3, 3, 3)

    def test_axis_index_is_clamped_to_valid_range(self) -> None:
        """A ULP-edge position at the upper face maps to the last voxel, never n.

        On the GPU (float32) a position one rounding-ULP below the upper face passes
        the ``pos < hi`` containment check yet ``floor((pos - lo) / spacing)`` can
        round up to ``n``; unclamped that indexed out of bounds (an illegal memory
        access under Warp). The clamp keeps such a point in voxel ``n - 1`` and a
        below-origin straggler in voxel 0, without moving any interior point.
        """
        from pyRadMC.geometry.grid import point_axis_index

        # position == upper face -> floor gives n; must clamp to n - 1.
        assert point_axis_index(10.0, 0.0, 1.0, 10) == 9
        # just past the upper face (a float-error straggler) also clamps.
        assert point_axis_index(10.0001, 0.0, 1.0, 10) == 9
        # below the origin clamps to 0.
        assert point_axis_index(-0.001, 0.0, 1.0, 10) == 0
        # interior points are untouched (half-open convention preserved).
        assert point_axis_index(0.0, 0.0, 1.0, 10) == 0
        assert point_axis_index(3.7, 0.0, 1.0, 10) == 3
        assert point_axis_index(9.999, 0.0, 1.0, 10) == 9

    def test_contains_is_half_open_on_the_upper_faces(self) -> None:
        from pyRadMC.geometry.grid import VoxelGrid

        grid = VoxelGrid.uniform_water(shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))
        assert grid.contains(0.0, 0.0, 0.0)
        assert grid.contains(3.999, 3.999, 3.999)
        assert not grid.contains(4.0, 2.0, 2.0)  # upper face excluded
        assert not grid.contains(-1e-12, 2.0, 2.0)
        assert not grid.contains(2.0, 2.0, 4.0)

    def test_mismatched_arrays_are_rejected(self) -> None:
        from pyRadMC.data.materials import WATER
        from pyRadMC.geometry.grid import VoxelGrid

        with pytest.raises(ValueError, match="shape"):
            VoxelGrid(
                shape=(2, 2, 2),
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                density=np.ones((2, 2, 3)),
                material=np.full((2, 2, 2), WATER, dtype=np.int32),
            )

    def test_distance_to_entry(self) -> None:
        """Slab clipping: head-on entry, oblique entry, miss, pointing away, inside."""
        import math

        from pyRadMC.geometry.grid import VoxelGrid

        grid = VoxelGrid.uniform_water(shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))

        assert grid.distance_to_entry(2.0, 2.0, -3.0, 0.0, 0.0, 1.0) == pytest.approx(3.0)
        assert grid.distance_to_entry(2.0, 2.0, 2.0, 1.0, 0.0, 0.0) == 0.0  # inside
        # Oblique 45-degree approach in the x-z plane.
        s = 1.0 / np.sqrt(2.0)
        assert grid.distance_to_entry(-1.0, 2.0, -1.0, s, 0.0, s) == pytest.approx(np.sqrt(2.0))
        # Pointing away.
        assert grid.distance_to_entry(2.0, 2.0, -3.0, 0.0, 0.0, -1.0) == math.inf
        # Parallel ray that never crosses the x-slab.
        assert grid.distance_to_entry(5.0, 2.0, -3.0, 0.0, 0.0, 1.0) == math.inf
        # Would cross the z-slab but misses the box transversely.
        assert grid.distance_to_entry(10.0, 2.0, -3.0, 0.0, 0.0, 1.0) == math.inf

    def test_distance_to_voxel_boundary(self) -> None:
        """Distance to the next internal voxel face; the electron substep boundary cap.

        Half-open voxels are lower-inclusive, so a point exactly on a face moving into
        the upper voxel measures a full voxel ahead, while moving back along that axis
        measures zero (the degenerate case the transport loop nudges across).
        """
        import math

        from pyRadMC.geometry.grid import distance_to_voxel_boundary

        o = (0.0, 0.0, 0.0)
        sp = (1.0, 1.0, 1.0)
        n = (4, 4, 4)

        # Inside voxel (0,0,0): the +x face is 0.7 ahead, the -x face 0.3 behind.
        assert distance_to_voxel_boundary(
            0.3, 0.5, 0.5, 1.0, 0.0, 0.0, *o, *sp, *n
        ) == pytest.approx(0.7)
        assert distance_to_voxel_boundary(
            0.3, 0.5, 0.5, -1.0, 0.0, 0.0, *o, *sp, *n
        ) == pytest.approx(0.3)

        # Diagonal in x-z: z reaches its face (0.2 away) before x (0.7 away), so z limits.
        r = 1.0 / math.sqrt(2.0)
        assert distance_to_voxel_boundary(0.3, 0.5, 0.8, r, 0.0, r, *o, *sp, *n) == pytest.approx(
            0.2 / r
        )

        # On an internal face (x=1.0 -> voxel 1, lower-inclusive): +x sees a full voxel,
        # -x sees zero (the caller biases the step just across so this cannot recur).
        assert distance_to_voxel_boundary(
            1.0, 0.5, 0.5, 1.0, 0.0, 0.0, *o, *sp, *n
        ) == pytest.approx(1.0)
        assert distance_to_voxel_boundary(
            1.0, 0.5, 0.5, -1.0, 0.0, 0.0, *o, *sp, *n
        ) == pytest.approx(0.0)

    def test_density_extrema_per_material(self) -> None:
        """The majorant declaration must see the *maximum* density in the grid."""
        from pyRadMC.data.materials import WATER
        from pyRadMC.geometry.grid import VoxelGrid

        grid = VoxelGrid.uniform_water(shape=(3, 3, 3), spacing=(1.0, 1.0, 1.0))
        grid.density[1, 2, 0] = 1.19
        assert grid.max_density_by_material() == ((WATER, 1.19),)


class TestSources:
    def test_pencil_beam_is_deterministic(self) -> None:
        from pyRadMC.geometry.source import PencilBeamSource
        from pyRadMC.rng.host import HostRNG

        source = PencilBeamSource(energy=6.0, position=(1.0, 2.0, -0.5), direction=(0.0, 0.0, 1.0))
        state = HostRNG().init_state(SEED, 40)
        for _ in range(10):
            p = source.emit(state)
            assert p.energy == 6.0
            assert (p.x, p.y, p.z) == (1.0, 2.0, -0.5)
            assert (p.ux, p.uy, p.uz) == (0.0, 0.0, 1.0)

    def test_pencil_beam_direction_is_normalized(self) -> None:
        from pyRadMC.geometry.source import PencilBeamSource

        source = PencilBeamSource(energy=2.0, position=(0.0, 0.0, 0.0), direction=(3.0, 0.0, 4.0))
        assert (source.direction[0], source.direction[2]) == pytest.approx((0.6, 0.8))

    def test_parallel_beam_covers_field_uniformly(self) -> None:
        """Positions uniform over the rectangular field: chi-squared on both marginals."""
        from pyRadMC.geometry.source import ParallelBeamSource
        from pyRadMC.rng.host import HostRNG

        source = ParallelBeamSource(
            energy=2.0,
            z=-1.0,
            x_range=(-2.0, 2.0),
            y_range=(0.0, 1.0),
        )
        state = HostRNG().init_state(SEED, 41)
        n = 20_000
        emitted = [source.emit(state) for _ in range(n)]

        xs = np.array([p.x for p in emitted])
        ys = np.array([p.y for p in emitted])
        assert np.all((xs >= -2.0) & (xs < 2.0))
        assert np.all((ys >= 0.0) & (ys < 1.0))
        for values, lo, hi, label in [(xs, -2.0, 2.0, "x"), (ys, 0.0, 1.0, "y")]:
            observed, _ = np.histogram(values, bins=np.linspace(lo, hi, 17))
            chi2, p_value = stats.chisquare(observed)
            assert p_value > 0.01, f"{label} not uniform: chi2={chi2:.1f}, p={p_value:.2e}"

        assert all(p.energy == 2.0 for p in emitted[:100])
        assert all((p.ux, p.uy, p.uz) == (0.0, 0.0, 1.0) for p in emitted[:100])
        assert all(p.z == -1.0 for p in emitted[:100])
