"""A charged particle in a uniform field, driven through the reference step loop.

Analytic oracle: in a uniform static field a charged particle moves on a helix whose
projection perpendicular to B is a circle of radius ``r_g sin(alpha)`` (``alpha`` the
pitch angle), turning through ``s / r_g`` per unit path length, electrons one way and
positrons the other; the component of the motion along B is a constant of the motion
and the field does no work. A near-vacuum medium (the grid refuses true vacuum) makes
energy loss and scattering negligible over a few turns, so the deposit positions the
loop files trace the orbit and can be fitted against it. The sign pin is Bielajew
(1988): sighting along B, positrons turn counter-clockwise and electrons clockwise —
an electron moving along +x in B along +z curves toward +y.

The step cap is pinned too: in water at 2 MeV and 1.5 T the gyration cap, not the
energy-limited GS step, bounds the substep.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc import ECUT_MEV, PCUT_MEV
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.physics.lorentz import gyroradius_cm
from pyradmc.rng.host import HostRNG
from pyradmc.transport.electron import LORENTZ_STEP_ANGLE_MAX, electron_steps
from tests.conftest import SEED

B_TESLA = 1.5
ENERGY = 1.0
# g/cm^3 of water: loss and scattering negligible over a few turns. At 1e-4 the
# multiple scattering still moves the fitted radius by 1-2 %, so go three decades lower.
NEAR_VACUUM = 1.0e-6


def _grid(shape, spacing, density: float) -> VoxelGrid:
    return VoxelGrid(
        shape=shape,
        spacing=spacing,
        density=np.full(shape, density, dtype=np.float64),
        material=np.full(shape, WATER, dtype=np.int32),
    )


def _track(
    grid: VoxelGrid,
    field: MagneticField | None,
    *,
    is_positron: bool,
    energy: float,
    start: tuple[float, float, float],
    direction: tuple[float, float, float],
    history: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Deposit positions along one history, plus the exact energy ledger terms."""
    xs = AnalyticCrossSections()
    points: list[tuple[float, float, float]] = []
    deposited = 0.0
    spawned = 0.0

    def deposit(x: float, y: float, z: float, amount: float, scored: float) -> None:
        nonlocal deposited
        points.append((x, y, z))
        deposited += amount

    def spawn(entry) -> None:
        nonlocal spawned
        spawned += entry[1] * entry[2]

    escaped = electron_steps(
        is_positron,
        energy,
        1.0,
        *start,
        *direction,
        grid,
        xs,
        HostRNG().init_state(SEED, history),
        deposit,
        spawn,
        PCUT_MEV,
        ECUT_MEV,
        field=field,
    )
    return np.asarray(points), deposited + spawned, escaped


def _fit_circle(xy: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle: centre and radius."""
    x, y = xy[:, 0], xy[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(cx), float(cy), float(math.sqrt(c + cx * cx + cy * cy))


PITCH = math.radians(45.0)
HELIX_GRID = ((20, 20, 30), (0.1, 0.1, 0.1))  # 2 x 2 x 3 cm, fine voxels
START = (1.0, 1.0, 0.05)
DIRECTION = (math.cos(PITCH), 0.0, math.sin(PITCH))  # transverse part along +x


class TestHelix:
    field = MagneticField.uniform((0.0, 0.0, B_TESLA))

    def _orbit(self, is_positron: bool):
        grid = _grid(*HELIX_GRID, NEAR_VACUUM)
        points, booked, escaped = _track(
            grid,
            self.field,
            is_positron=is_positron,
            energy=ENERGY,
            start=START,
            direction=DIRECTION,
        )
        return points, booked, escaped

    def test_electron_circles_at_the_gyroradius(self) -> None:
        points, _, _ = self._orbit(is_positron=False)
        assert len(points) > 200, "expected several turns of deposits"
        cx, cy, radius = _fit_circle(points[:, :2])
        want = gyroradius_cm(ENERGY, B_TESLA) * math.sin(PITCH)
        # Deposits sit at chord midpoints, inside the arc by up to 1 - cos(delta_u/4).
        assert radius == pytest.approx(want, rel=0.015)
        assert np.std(np.hypot(points[:, 0] - cx, points[:, 1] - cy)) < 0.02 * want

    def test_electron_curves_toward_plus_y_in_plus_z_field(self) -> None:
        points, _, _ = self._orbit(is_positron=False)
        _, cy, radius = _fit_circle(points[:, :2])
        assert cy > START[1] + 0.5 * radius  # centre on the +y side: e- turns toward +y
        assert np.all(points[:, 1] > START[1] - 0.02)

    def test_positron_curves_the_other_way(self) -> None:
        points, _, _ = self._orbit(is_positron=True)
        _, cy, radius = _fit_circle(points[:, :2])
        assert radius == pytest.approx(gyroradius_cm(ENERGY, B_TESLA) * math.sin(PITCH), rel=0.015)
        assert cy < START[1] - 0.5 * radius
        assert np.all(points[:, 1] < START[1] + 0.02)

    def test_the_field_does_no_work(self) -> None:
        """Exact ledger: everything emitted is deposited, handed on or escaped."""
        for is_positron in (False, True):
            _, booked, escaped = self._orbit(is_positron)
            rest = 2.0 * 0.51099895069 if is_positron else 0.0
            assert booked + escaped == pytest.approx(ENERGY + rest, rel=1e-12)

    def test_motion_along_the_field_is_straight(self) -> None:
        grid = _grid(*HELIX_GRID, NEAR_VACUUM)
        points, _, _ = _track(
            grid,
            self.field,
            is_positron=False,
            energy=ENERGY,
            start=START,
            direction=(0.0, 0.0, 1.0),
        )
        lateral = np.hypot(points[:, 0] - START[0], points[:, 1] - START[1])
        # Only the (negligible) multiple scattering of the near-vacuum moves it.
        assert lateral.max() < 0.05 * gyroradius_cm(ENERGY, B_TESLA)
        assert points[-1, 2] > 2.5  # it left through the far face, not sideways


class TestFieldFreePaths:
    def test_zero_field_reproduces_no_field_exactly(self) -> None:
        grid = _grid((8, 8, 40), (0.5, 0.5, 0.1), 1.0)
        kwargs = dict(
            is_positron=False, energy=3.0, start=(2.0, 2.0, 0.05), direction=(0.0, 0.0, 1.0)
        )
        for history in range(5):
            a = _track(grid, None, history=history, **kwargs)
            b = _track(grid, MagneticField.uniform((0.0, 0.0, 0.0)), history=history, **kwargs)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1] == b[1] and a[2] == b[2]


class TestStepCap:
    """In water at 2 MeV and 1.5 T the gyration cap bounds the substep, not the GS step."""

    E0 = 2.0

    def _max_spacing(self, field: MagneticField | None) -> float:
        grid = _grid((10, 10, 10), (1.0, 1.0, 1.0), 1.0)  # coarse: faces rarely bind
        worst = 0.0
        for history in range(40):
            points, _, _ = _track(
                grid,
                field,
                is_positron=False,
                energy=self.E0,
                start=(5.0, 5.0, 0.5),
                direction=(0.0, 0.0, 1.0),
                history=history,
            )
            if len(points) > 1:
                worst = max(worst, float(np.linalg.norm(np.diff(points, axis=0), axis=1).max()))
        return worst

    def test_consecutive_deposits_are_bounded_by_the_gyration_cap(self) -> None:
        cap = LORENTZ_STEP_ANGLE_MAX * gyroradius_cm(self.E0, B_TESLA)
        with_field = self._max_spacing(MagneticField.uniform((B_TESLA, 0.0, 0.0)))
        without = self._max_spacing(None)
        assert with_field <= cap * 1.001
        assert without > cap, "the cap must actually be the binding limiter at this energy"
