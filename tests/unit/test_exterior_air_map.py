"""The exterior-air region and its safety map: classification and the conservative bound.

The whole approximation rests on one geometric promise: from any point of a voxel, a
ball of radius ``safety[voxel]`` holds nothing but exterior voxels and stays inside the
grid. It is pinned here three ways: against a brute-force box-to-box distance, against
random points and random ball samples, and on hand-computed values.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from pyradmc.data.materials import WATER
from pyradmc.geometry.exterior import (
    EXTERIOR_AIR_DENSITY_MAX,
    EXTERIOR_AUTO,
    EXTERIOR_MIN_SAFETY_CM,
    SAFETY_GUARD_CM,
    ExteriorAirMap,
    exterior_region,
    resolve_exterior_air,
    safety_distance,
)
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from tests.conftest import SEED

AIR = 1.205e-3


def _grid(density: np.ndarray, spacing=(0.5, 0.5, 0.5), origin=(0.0, 0.0, 0.0)) -> VoxelGrid:
    return VoxelGrid(
        shape=density.shape,
        spacing=spacing,
        origin=origin,
        density=density,
        material=np.full(density.shape, WATER, dtype=np.int32),
    )


def _body_with_cavities() -> np.ndarray:
    """Air, a water box, lung-density and air-density cavities sealed inside the box."""
    d = np.full((20, 20, 20), AIR)
    d[4:16, 4:16, 4:16] = 1.0
    d[6:9, 6:9, 6:9] = 0.26  # lung
    d[11:14, 11:14, 11:14] = AIR  # a sealed gas pocket
    return d


class TestRegion:
    def test_air_connected_to_the_boundary_is_exterior(self) -> None:
        region = exterior_region(_grid(_body_with_cavities()))
        assert region[0, 0, 0]
        assert region[2, 10, 10]

    def test_tissue_and_lung_are_not(self) -> None:
        region = exterior_region(_grid(_body_with_cavities()))
        assert not region[10, 10, 5]
        assert not region[7, 7, 7]

    def test_a_sealed_air_pocket_is_not(self) -> None:
        region = exterior_region(_grid(_body_with_cavities()))
        assert not region[12, 12, 12]

    def test_a_channel_to_the_boundary_makes_the_pocket_exterior(self) -> None:
        d = _body_with_cavities()
        d[12, 12, 14:] = AIR  # bore from the pocket out through the box's top
        assert exterior_region(_grid(d))[12, 12, 12]

    def test_diagonal_contact_does_not_connect(self) -> None:
        """Six-connectivity: a corner touch is not a path, the conservative reading."""
        d = np.ones((5, 5, 5))
        d[0, 0, 0] = AIR
        d[1, 1, 1] = AIR
        region = exterior_region(_grid(d))
        assert region[0, 0, 0]
        assert not region[1, 1, 1]

    def test_threshold_is_strict(self) -> None:
        d = np.full((3, 3, 3), EXTERIOR_AIR_DENSITY_MAX)
        assert not exterior_region(_grid(d)).any()
        d = np.full((3, 3, 3), np.nextafter(EXTERIOR_AIR_DENSITY_MAX, 0.0))
        assert exterior_region(_grid(d)).all()

    def test_threshold_is_below_every_tissue(self) -> None:
        """Lung at full inspiration sits near 0.1 g/cm^3; the bound keeps a decade clear."""
        assert EXTERIOR_AIR_DENSITY_MAX <= 0.01


def _brute_force_gap(region: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Exact smallest distance from each voxel's box to any non-region box or the outside."""
    out = np.zeros(region.shape)
    shape = region.shape
    blocked = np.argwhere(~region)
    sp = np.asarray(spacing)
    for idx in itertools.product(*(range(n) for n in shape)):
        if not region[idx]:
            continue
        p = np.asarray(idx)
        # Grid faces: the voxel's own distance to each of the six.
        best = min(min(p[a] * sp[a], (shape[a] - 1 - p[a]) * sp[a]) for a in range(3))
        if blocked.size:
            gaps = np.maximum(np.abs(blocked - p) - 1, 0) * sp
            best = min(best, float(np.sqrt((gaps * gaps).sum(axis=1)).min()))
        out[idx] = best
    return out


class TestSafety:
    @pytest.mark.parametrize("spacing", [(0.5, 0.5, 0.5), (0.3, 0.3, 0.25), (1.0, 0.2, 0.4)])
    def test_equals_the_exact_box_to_box_distance(self, spacing) -> None:
        rng = np.random.default_rng(SEED)
        region = rng.random((14, 12, 10)) > 0.04
        computed = safety_distance(region, spacing)
        np.testing.assert_allclose(computed, _brute_force_gap(region, spacing), atol=1e-12)

    def test_hand_values_in_an_open_box(self) -> None:
        """All air, 11 voxels of 1 cm a side: the centre voxel is 5 cm from each face."""
        s = safety_distance(np.ones((11, 11, 11), dtype=bool), (1.0, 1.0, 1.0))
        assert s[5, 5, 5] == pytest.approx(5.0)
        assert s[0, 5, 5] == 0.0
        assert s[1, 5, 5] == pytest.approx(1.0)

    def test_hand_values_next_to_a_blocked_voxel(self) -> None:
        region = np.ones((21, 21, 21), dtype=bool)
        region[10, 10, 10] = False
        s = safety_distance(region, (1.0, 1.0, 1.0))
        assert s[11, 10, 10] == 0.0  # touches the blocked box
        assert s[12, 10, 10] == pytest.approx(1.0)
        assert s[12, 12, 10] == pytest.approx(np.sqrt(2.0))

    def test_a_region_voxel_only(self) -> None:
        region = np.ones((9, 9, 9), dtype=bool)
        region[4, 4, 4] = False
        assert safety_distance(region, (1.0, 1.0, 1.0))[4, 4, 4] == 0.0


class TestBallContainment:
    @pytest.mark.parametrize("min_safety", [0.0, EXTERIOR_MIN_SAFETY_CM])
    def test_no_ball_ever_leaves_the_region(self, min_safety: float) -> None:
        """The promise itself: random points, random directions, radius the map's value."""
        rng = np.random.default_rng(SEED)
        d = _body_with_cavities()
        origin = (-3.0, 1.0, 2.0)
        spacing = (0.5, 0.4, 0.3)
        grid = _grid(d, spacing=spacing, origin=origin)
        air = ExteriorAirMap.from_grid(grid, min_safety_cm=min_safety)
        region = exterior_region(grid)
        candidates = np.argwhere(air.safety > 0.0)
        assert candidates.size > 0

        n = 200_000
        picks = candidates[rng.integers(0, len(candidates), n)]
        points = np.asarray(origin) + (picks + rng.random((n, 3))) * np.asarray(spacing)
        directions = rng.normal(size=(n, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = air.safety[picks[:, 0], picks[:, 1], picks[:, 2]]
        ends = points + directions * radii[:, None] * rng.random((n, 1)) ** 0.0

        cells = np.floor((ends - np.asarray(origin)) / np.asarray(spacing)).astype(int)
        inside = np.all((cells >= 0) & (cells < np.asarray(d.shape)), axis=1)
        assert inside.all(), "a ball left the grid"
        assert region[cells[:, 0], cells[:, 1], cells[:, 2]].all(), "a ball reached the body"


class TestMap:
    def test_below_the_minimum_is_zero_and_the_rest_is_guarded(self) -> None:
        grid = _grid(_body_with_cavities())
        air = ExteriorAirMap.from_grid(grid)
        raw = safety_distance(exterior_region(grid), grid.spacing)
        useful = raw >= EXTERIOR_MIN_SAFETY_CM
        assert useful.any()
        np.testing.assert_array_equal(air.safety[~useful], 0.0)
        np.testing.assert_allclose(air.safety[useful], raw[useful] - SAFETY_GUARD_CM)

    def test_a_body_voxel_and_its_neighbours_are_zero(self) -> None:
        air = ExteriorAirMap.from_grid(_grid(_body_with_cavities()))
        assert air.safety[10, 10, 10] == 0.0
        assert air.safety[3, 10, 10] == 0.0  # touches the box

    def test_is_read_only(self) -> None:
        air = ExteriorAirMap.from_grid(_grid(_body_with_cavities()))
        with pytest.raises(ValueError, match="read-only"):
            air.safety[0, 0, 0] = 1.0

    def test_matches_its_grid_only(self) -> None:
        grid = _grid(_body_with_cavities())
        air = ExteriorAirMap.from_grid(grid)
        assert air.matches(grid)
        assert air.matches(_grid(_body_with_cavities()))
        changed = _body_with_cavities()
        changed[0, 0, 0] = 0.002
        assert not air.matches(_grid(changed))
        assert not air.matches(_grid(_body_with_cavities(), spacing=(0.5, 0.5, 0.4)))
        assert not air.matches(_grid(_body_with_cavities(), origin=(1.0, 0.0, 0.0)))

    def test_from_region_is_data(self) -> None:
        """A caller's region goes through the same bound; a test instrument's way in."""
        grid = _grid(np.full((30, 30, 30), AIR))
        region = np.zeros(grid.shape, dtype=bool)
        region[5:25, 5:25, 5:25] = True
        air = ExteriorAirMap.from_region(grid, region)
        assert air.safety[15, 15, 15] > 0.0
        assert air.safety[2, 2, 2] == 0.0

    def test_from_region_refuses_a_wrong_shape(self) -> None:
        grid = _grid(np.full((6, 6, 6), AIR))
        with pytest.raises(ValueError, match="shape"):
            ExteriorAirMap.from_region(grid, np.ones((5, 6, 6), dtype=bool))

    def test_refuses_a_negative_minimum(self) -> None:
        with pytest.raises(ValueError, match="min_safety_cm"):
            ExteriorAirMap.from_grid(_grid(np.full((6, 6, 6), AIR)), min_safety_cm=-1.0)

    def test_provenance_names_threshold_minimum_and_digest(self) -> None:
        air = ExteriorAirMap.from_grid(_grid(_body_with_cavities()))
        text = air.provenance
        assert str(EXTERIOR_AIR_DENSITY_MAX) in text
        assert str(EXTERIOR_MIN_SAFETY_CM) in text
        assert air.digest[:12] in text

    def test_a_tampered_map_has_another_digest(self) -> None:
        air = ExteriorAirMap.from_grid(_grid(_body_with_cavities()))
        inflated = dataclasses.replace(air, safety=air.safety * 3.0)
        assert inflated.digest != air.digest


class TestResolve:
    def _grid(self) -> VoxelGrid:
        return _grid(_body_with_cavities())

    def test_none_or_no_field_is_inactive(self) -> None:
        field = MagneticField.uniform((1.5, 0.0, 0.0))
        grid = self._grid()
        assert resolve_exterior_air(None, grid, field) is None
        assert resolve_exterior_air(EXTERIOR_AUTO, grid, None) is None
        assert resolve_exterior_air(ExteriorAirMap.from_grid(grid), grid, None) is None

    def test_a_zero_field_is_inactive(self) -> None:
        zero = MagneticField.uniform((0.0, 0.0, 0.0))
        assert resolve_exterior_air(EXTERIOR_AUTO, self._grid(), zero) is None

    def test_a_nonuniform_map_is_inactive(self) -> None:
        grid = self._grid()
        values = np.zeros((*grid.shape, 3))
        values[..., 0] = 1.5
        field = MagneticField.on_grid(grid, values)
        assert resolve_exterior_air(EXTERIOR_AUTO, grid, field) is None

    def test_auto_derives_the_map_under_a_uniform_field(self) -> None:
        grid = self._grid()
        field = MagneticField.uniform((1.5, 0.0, 0.0))
        air = resolve_exterior_air(EXTERIOR_AUTO, grid, field)
        assert air is not None
        assert air.digest == ExteriorAirMap.from_grid(grid).digest

    def test_a_given_map_is_used_as_given(self) -> None:
        grid = self._grid()
        air = ExteriorAirMap.from_grid(grid, min_safety_cm=0.5)
        assert resolve_exterior_air(air, grid, MagneticField.uniform((1.5, 0.0, 0.0))) is air

    def test_a_map_with_no_safety_anywhere_is_inactive(self) -> None:
        grid = VoxelGrid.uniform_water(shape=(6, 6, 6), spacing=(0.5, 0.5, 0.5))
        field = MagneticField.uniform((1.5, 0.0, 0.0))
        assert resolve_exterior_air(EXTERIOR_AUTO, grid, field) is None
