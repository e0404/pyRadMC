"""The exact helix chord of a long exterior-air step, against the helix itself.

Analytic oracle: in a uniform field a charged particle at constant energy moves as

    p(s) = p0 + a (a.u) s + r [u_perp sin(s/r) + (a x u_perp) (1 - cos(s/r))].

Both the formula and the compiled form are held to it far beyond the 0.2 rad of the
curvature cap, up to tens of gyrations, because a long air step at 1.5 T is exactly
that. The loop test then drives the reference stepper through a near-vacuum exterior
with whole-radian steps and asks whether each substep start is on the orbit; the
midpoint chord of the small-step scheme is not (mutant test).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc import ECUT_MEV, PCUT_MEV
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.exterior import ExteriorAirMap
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.physics.lorentz import gyroradius_cm, helix_chord, rotate_about_axis
from pyradmc.rng.host import HostRNG
from pyradmc.transport import electron as electron_module
from pyradmc.transport.electron import electron_steps
from tests.conftest import SEED

ANGLES = [0.0, 1e-5, 1e-3, 0.0199, 0.0201, 0.2, 1.0, math.pi / 2, math.pi, 3.0 * math.pi, 40.0]


def _exact_chord(u, a, theta: float) -> np.ndarray:
    """Displacement per unit path of the true helix, for a right-handed turn ``theta``."""
    u, a = np.asarray(u, dtype=float), np.asarray(a, dtype=float)
    parallel = a * float(a @ u)
    perp = u - parallel
    if theta == 0.0:
        return u.copy()
    return parallel + (perp * math.sin(theta) + np.cross(a, perp) * (1.0 - math.cos(theta))) / theta


def _midpoint_call(u, a, theta: float) -> tuple[float, float, float]:
    """``helix_chord`` as the loop calls it: on the direction already turned by theta/2."""
    half = 0.5 * theta
    mid = rotate_about_axis(*u, *a, half)
    return helix_chord(*mid, *a, half)


@pytest.mark.parametrize("theta", [*ANGLES, *(-t for t in ANGLES if t > 0.0)])
@pytest.mark.parametrize(
    ("u", "a"),
    [
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.6, 0.0, 0.8), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        ((0.48, 0.6, 0.64), (0.8, 0.6, 0.0)),
    ],
    ids=["perpendicular", "pitched", "along-B", "skew"],
)
def test_chord_is_the_exact_helix_displacement(u, a, theta: float) -> None:
    np.testing.assert_allclose(_midpoint_call(u, a, theta), _exact_chord(u, a, theta), atol=1e-9)


@pytest.mark.parametrize("theta", ANGLES)
def test_chord_length_is_two_r_sin_half_theta(theta: float) -> None:
    """Perpendicular motion: |chord| per unit path is sin(theta/2) / (theta/2)."""
    c = np.asarray(_midpoint_call((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), theta))
    expected = 1.0 if theta == 0.0 else abs(math.sin(0.5 * theta) / (0.5 * theta))
    assert np.linalg.norm(c) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("theta", ANGLES)
def test_motion_along_the_field_is_preserved(theta: float) -> None:
    u, a = (0.48, 0.6, 0.64), (0.8, 0.6, 0.0)
    along = u[0] * a[0] + u[1] * a[1] + u[2] * a[2]
    c = _midpoint_call(u, a, theta)
    assert c[0] * a[0] + c[1] * a[1] + c[2] * a[2] == pytest.approx(along, abs=1e-12)


def test_displacement_never_exceeds_the_path() -> None:
    """What the safety bound leans on: chord length <= 1 per unit path, at any angle."""
    rng = np.random.default_rng(SEED)
    for _ in range(2000):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        c = _midpoint_call(tuple(u), tuple(a), float(rng.uniform(-60.0, 60.0)))
        assert np.linalg.norm(c) <= 1.0 + 1e-12


def test_midpoint_rule_on_a_shrinking_radius_is_small() -> None:
    """r_g falls as the particle slows; the constant-radius chord's error stays small.

    The worst a long step can see: the energy-fraction cap lets 20 % of the CSDA range
    go in one step, here 0.5 MeV at 1.5 T with a linear loss of 10 % over several turns.
    The turn is taken at the mid-energy radius as the loop does; the end point against a
    fine integration of the true spiral is off by well under the radius.
    """
    e0, de, s, b = 0.5, 0.05, 3.0, 1.5
    n = 400_000
    ds = s / n
    t = (np.arange(n) + 0.5) * ds
    radii = np.array([gyroradius_cm(e, b) for e in e0 - de * t / s])
    phase = np.cumsum(ds / radii) - 0.5 * ds / radii
    true_end = np.array([np.sum(np.cos(phase)) * ds, np.sum(np.sin(phase)) * ds, 0.0])
    r_mid = gyroradius_cm(e0 - 0.5 * de, b)
    theta = s / r_mid
    end = np.asarray(_midpoint_call((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), theta)) * s
    assert np.linalg.norm(end - true_end) < 0.05 * r_mid


# --- compiled form ----------------------------------------------------------------

wp_or_none = None
try:
    import warp as wp_or_none  # type: ignore[no-redef]
except ModuleNotFoundError as exc:
    if exc.name != "warp":
        raise


@pytest.mark.warp
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_compiled_chord_agrees_with_the_helix(device: str) -> None:
    """float32, fast_math: the same formula through a throwaway kernel, per device.

    A same-formula precision check, not a cross-target dose equality (AGENTS.md 2.3).
    The tolerance carries the float32 angle: at 40 rad one ulp of the angle is 4e-6.
    """
    if wp_or_none is None:
        pytest.skip("warp-lang optional dependency not installed")
    wp = wp_or_none
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp import kernels  # noqa: F401 - module options, physics
    from pyradmc.backends.warp.physics import warp_physics

    physics = warp_physics()
    compiled_chord = physics.helix_chord
    compiled_rotate = physics.rotate_about_axis

    @wp.kernel
    def probe(
        u: wp.array(dtype=wp.vec3),
        a: wp.array(dtype=wp.vec3),
        theta: wp.array(dtype=float),
        out: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        half = 0.5 * theta[i]
        mx, my, mz = compiled_rotate(u[i][0], u[i][1], u[i][2], a[i][0], a[i][1], a[i][2], half)
        cx, cy, cz = compiled_chord(mx, my, mz, a[i][0], a[i][1], a[i][2], half)
        out[i] = wp.vec3(cx, cy, cz)

    rng = np.random.default_rng(SEED)
    n = 4000
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    a = rng.normal(size=(n, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    theta = rng.uniform(-40.0, 40.0, n)
    theta[: n // 4] = 10.0 ** rng.uniform(-6.0, -1.0, n // 4)
    out = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(
        probe,
        dim=n,
        inputs=[
            wp.array(u.astype(np.float32), dtype=wp.vec3, device=device),
            wp.array(a.astype(np.float32), dtype=wp.vec3, device=device),
            wp.array(theta.astype(np.float32), dtype=float, device=device),
            out,
        ],
        device=device,
    )
    got = out.numpy()
    expected = np.array([_exact_chord(u[i], a[i], float(theta[i])) for i in range(n)])
    # An angle error dtheta moves a chord per unit path by at most dtheta / |theta|;
    # float32 and the approximate intrinsics are all below 1e-3 absolute here.
    np.testing.assert_allclose(got, expected, atol=1e-3)


# --- through the reference loop ---------------------------------------------------

B_TESLA = 1.5
ENERGY = 1.0
# Lower than test_lorentz_helix's 1e-6: these steps are tens of cm, and at 1e-6 the hinge
# angle of each (about 9 mrad) moves the gyro centre by a percent within a dozen steps.
NEAR_VACUUM = 1.0e-9
N_SUBSTEPS = 40  # at this density nothing ends a history: stop it


class _EnoughError(Exception):
    """Raised by the recording grid to end a history that would otherwise not end."""


def _vacuum_box() -> VoxelGrid:
    shape = (60, 60, 60)
    return VoxelGrid(
        shape=shape,
        spacing=(1.0, 1.0, 1.0),
        density=np.full(shape, NEAR_VACUUM, dtype=np.float64),
        material=np.full(shape, WATER, dtype=np.int32),
    )


class _RecordingGrid(VoxelGrid):
    """A grid that records where each substep starts: ``voxel_index`` is read there once."""

    def voxel_index(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        starts = self.__dict__.setdefault("starts", [])
        if len(starts) == N_SUBSTEPS:
            raise _EnoughError
        starts.append((x, y, z))
        return super().voxel_index(x, y, z)


def _orbit(direction: tuple[float, float, float], *, is_positron: bool) -> tuple[np.ndarray, list]:
    box = _vacuum_box()
    grid = _RecordingGrid(
        shape=box.shape, spacing=box.spacing, density=box.density, material=box.material
    )
    air = ExteriorAirMap.from_grid(grid)
    rng = HostRNG()
    state = rng.init_state(SEED, 0)
    chords: list[float] = []

    spy_target = electron_module.helix_chord

    def spy(*args: float) -> tuple[float, float, float]:
        chords.append(abs(args[6]))
        return spy_target(*args)

    electron_module.helix_chord = spy  # type: ignore[assignment]
    try:
        electron_steps(
            is_positron,
            ENERGY,
            1.0,
            30.0,
            30.0,
            30.0,
            *direction,
            grid,
            AnalyticCrossSections(),
            state,
            lambda *_: None,
            lambda _: None,
            PCUT_MEV,
            ECUT_MEV,
            field=MagneticField.uniform((0.0, 0.0, B_TESLA)),
            exterior_safety=air.safety,
        )
    except _EnoughError:
        pass  # a pitched launch leaves through a z face first, which is also fine
    finally:
        electron_module.helix_chord = spy_target  # type: ignore[assignment]
    return np.asarray(grid.__dict__["starts"]), chords


@pytest.mark.parametrize("is_positron", [False, True], ids=["electron", "positron"])
def test_long_steps_stay_on_the_gyration_circle(is_positron: bool) -> None:
    """Perpendicular launch: every substep start is on the circle, at whole radians a step."""
    points, half_angles = _orbit((1.0, 0.0, 0.0), is_positron=is_positron)
    assert len(points) > 30
    assert max(half_angles) > 2.0, "the steps were not long; nothing was tested"

    r = gyroradius_cm(ENERGY, B_TESLA)
    # An electron moving along +x in B along +z curves toward +y (test_lorentz_helix).
    side = -1.0 if is_positron else 1.0
    centre = np.array([30.0, 30.0 + side * r])
    radii = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1])
    np.testing.assert_allclose(radii, r, rtol=2e-3)
    np.testing.assert_allclose(points[:, 2], 30.0, atol=1e-6)


def test_pitch_is_preserved_over_long_steps() -> None:
    """A pitched launch keeps its axial speed: drift along B equals path times cos(alpha)."""
    c_alpha = 0.6
    points, _ = _orbit((0.8, 0.0, c_alpha), is_positron=False)
    r = gyroradius_cm(ENERGY, B_TESLA) * 0.8
    centre = np.array([30.0, 30.0 + r])
    radii = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1])
    assert len(points) > 5
    np.testing.assert_allclose(radii, r, rtol=2e-3)
    assert points[-1, 2] > points[0, 2]


def test_the_small_step_chord_fails_the_orbit_at_these_angles(monkeypatch) -> None:
    """Mutant: without the chord shortening a long step spirals off the circle."""

    def straight(ux, uy, uz, ax, ay, az, half):
        return ux, uy, uz

    box = _vacuum_box()
    grid = _RecordingGrid(
        shape=box.shape, spacing=box.spacing, density=box.density, material=box.material
    )
    air = ExteriorAirMap.from_grid(grid)
    state = HostRNG().init_state(SEED, 0)
    monkeypatch.setattr(electron_module, "helix_chord", straight)
    with pytest.raises(_EnoughError):
        electron_steps(
            False,
            ENERGY,
            1.0,
            30.0,
            30.0,
            30.0,
            1.0,
            0.0,
            0.0,
            grid,
            AnalyticCrossSections(),
            state,
            lambda *_: None,
            lambda _: None,
            PCUT_MEV,
            ECUT_MEV,
            field=MagneticField.uniform((0.0, 0.0, B_TESLA)),
            exterior_safety=air.safety,
        )
    r = gyroradius_cm(ENERGY, B_TESLA)
    points = np.asarray(grid.__dict__["starts"])
    radii = np.hypot(points[:, 0] - 30.0, points[:, 1] - (30.0 + r))
    assert np.abs(radii / r - 1.0).max() > 0.05
