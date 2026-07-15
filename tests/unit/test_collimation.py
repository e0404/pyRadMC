"""Beam-limiting-device geometry: frame, jaws, MLC, stack (BLD workstream, slices 1-3).

Pure ray/solid intersection math — no cross-sections, no RNG — so every expected
value here is hand geometry: slab crossings divided by direction cosines, circle
chords from the Pythagorean relation, projective scaling through the focal point.

Conventions under test (the module docstring is the normative statement):

- The local beam frame has its origin at the focal spot; ``w`` increases downstream,
  ``u`` is the leaf-travel direction, ``v`` the leaf-width direction.
- Edge and tip positions are physical coordinates at the device mid-plane.
- ``path_lengths`` uses the full-line convention (no ``t >= 0`` clamp): the caller
  guarantees the trajectory traversed the devices, which holds both for focal-spot
  sources (devices downstream of emission) and the planar exit source (devices
  upstream). ``from_origin=True`` (stack level) restricts to the forward half-line.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyRadMC.data.materials import TUNGSTEN
from pyRadMC.geometry.collimation import (
    MLC,
    BeamFrame,
    BeamLimitingStack,
    JawPair,
    project_between_planes,
)


def _rays(*rays: tuple[tuple[float, float, float], tuple[float, float, float]]):
    """Pack (origin, direction) pairs into the (n, 3) arrays path_lengths takes."""
    origins = np.array([r[0] for r in rays], dtype=np.float64)
    directions = np.array([r[1] for r in rays], dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origins, directions


class TestBeamFrame:
    def test_identity_frame_is_a_passthrough(self) -> None:
        frame = BeamFrame(origin=(0.0, 0.0, 0.0))
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 6.0]])
        directions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        p_local, d_local = frame.to_local(points, directions)
        np.testing.assert_array_equal(p_local, points)
        np.testing.assert_array_equal(d_local, directions)

    def test_round_trip_engine_to_local(self) -> None:
        """A rotated, translated frame maps points/directions consistently."""
        s = 1.0 / math.sqrt(2.0)
        frame = BeamFrame(
            origin=(10.0, -3.0, 5.0),
            u_axis=(s, s, 0.0),
            v_axis=(-s, s, 0.0),
            w_axis=(0.0, 0.0, 1.0),
        )
        rng = np.random.default_rng(20260715)
        points = rng.normal(size=(32, 3))
        directions = rng.normal(size=(32, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        p_local, d_local = frame.to_local(points, directions)
        # Reconstruct engine coordinates from the local ones by hand.
        axes = np.array([frame.u_axis, frame.v_axis, frame.w_axis])
        np.testing.assert_allclose(p_local @ axes + frame.origin, points, atol=1e-12)
        np.testing.assert_allclose(d_local @ axes, directions, atol=1e-12)
        # Directions are rotated only, never translated.
        np.testing.assert_allclose(np.linalg.norm(d_local, axis=1), np.ones(32), atol=1e-12)

    def test_axes_are_normalized(self) -> None:
        frame = BeamFrame(origin=(0.0, 0.0, 0.0), u_axis=(2.0, 0.0, 0.0))
        assert frame.u_axis == (1.0, 0.0, 0.0)

    def test_non_orthogonal_axes_raise(self) -> None:
        with pytest.raises(ValueError, match="orthonormal"):
            BeamFrame(origin=(0.0, 0.0, 0.0), u_axis=(1.0, 0.0, 0.0), v_axis=(1.0, 0.1, 0.0))

    def test_left_handed_axes_raise(self) -> None:
        """u x v must equal +w: a mirrored frame silently flips the leaf banks."""
        with pytest.raises(ValueError, match="right-handed"):
            BeamFrame(
                origin=(0.0, 0.0, 0.0),
                u_axis=(1.0, 0.0, 0.0),
                v_axis=(0.0, 1.0, 0.0),
                w_axis=(0.0, 0.0, -1.0),
            )


class TestProjection:
    def test_scalar_projection_scales_through_the_focal_point(self) -> None:
        # An isocenter-plane setting at SAD 100 projects to half at z = 50.
        assert project_between_planes(4.0, 100.0, 50.0) == pytest.approx(2.0)

    def test_array_projection(self) -> None:
        tips = np.array([-3.0, 0.0, 5.0])
        np.testing.assert_allclose(
            project_between_planes(tips, 100.0, 40.0), np.array([-1.2, 0.0, 2.0])
        )

    def test_zero_source_plane_raises(self) -> None:
        with pytest.raises(ValueError, match="plane"):
            project_between_planes(1.0, 0.0, 50.0)


class TestJawPair:
    """One jaw pair limiting the local u coordinate: slab z in [40, 47], edges +-2."""

    def jaws(self, **overrides: object) -> JawPair:
        settings: dict = {
            "axis": "u",
            "z_top": 40.0,
            "z_bottom": 47.0,
            "edge_neg": -2.0,
            "edge_pos": 2.0,
            "material": TUNGSTEN,
            "density": 18.0,
        }
        settings.update(overrides)
        return JawPair(**settings)

    def test_open_field_ray_sees_nothing(self) -> None:
        origins, directions = _rays(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        np.testing.assert_array_equal(
            self.jaws().path_lengths(origins, directions), np.array([0.0])
        )

    def test_axial_ray_under_a_block_sees_the_full_thickness(self) -> None:
        origins, directions = _rays(
            ((3.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # +u block
            ((-2.5, 1.0, 0.0), (0.0, 0.0, 1.0)),  # -u block
        )
        np.testing.assert_allclose(
            self.jaws().path_lengths(origins, directions), np.array([7.0, 7.0])
        )

    def test_oblique_ray_is_the_slant_thickness(self) -> None:
        """Fully inside a block laterally: (z_bottom - z_top) / cos(theta)."""
        theta = math.radians(30.0)
        direction = (math.sin(theta), 0.0, math.cos(theta))
        origins, directions = _rays(((10.0, 0.0, 0.0), direction))
        np.testing.assert_allclose(
            self.jaws().path_lengths(origins, directions),
            np.array([7.0 / math.cos(theta)]),
        )

    def test_edge_clipping_ray_gets_the_partial_length(self) -> None:
        """Ray from the origin along (1, 0, 7): u = z/7, crosses u = 6 at z = 42.

        With edge_pos = 6 the ray is inside the +u block only for z in [42, 47],
        a length of 5 / cos(theta) = 5 * sqrt(50) / 7.
        """
        origins, directions = _rays(((0.0, 0.0, 0.0), (1.0, 0.0, 7.0)))
        expected = 5.0 * math.sqrt(50.0) / 7.0
        np.testing.assert_allclose(
            self.jaws(edge_pos=6.0).path_lengths(origins, directions),
            np.array([expected]),
        )

    def test_ray_parallel_to_the_slab(self) -> None:
        """d_w = 0: empty outside the slab; infinite within a block (transmission 0)."""
        jaws = self.jaws()
        origins, directions = _rays(
            ((3.0, 0.0, 20.0), (0.0, 1.0, 0.0)),  # outside the slab in z
            ((3.0, 0.0, 43.0), (0.0, 1.0, 0.0)),  # inside slab, inside +u block
            ((0.0, 0.0, 43.0), (0.0, 1.0, 0.0)),  # inside slab, open field
        )
        lengths = jaws.path_lengths(origins, directions)
        assert lengths[0] == 0.0
        assert math.isinf(lengths[1])
        assert lengths[2] == 0.0

    def test_devices_behind_the_ray_origin_still_count(self) -> None:
        """Full-line convention: a planar source below the jaws sees them upstream."""
        origins, directions = _rays(((3.0, 0.0, 60.0), (0.0, 0.0, 1.0)))
        np.testing.assert_allclose(self.jaws().path_lengths(origins, directions), np.array([7.0]))

    def test_v_axis_pair_limits_the_other_coordinate(self) -> None:
        jaws = self.jaws(axis="v")
        origins, directions = _rays(
            ((3.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # u = 3 is open for a v-pair
            ((0.0, 3.0, 0.0), (0.0, 0.0, 1.0)),  # v = 3 is blocked
        )
        np.testing.assert_allclose(jaws.path_lengths(origins, directions), [0.0, 7.0])

    def test_validation_raises(self) -> None:
        with pytest.raises(ValueError, match="z_top"):
            self.jaws(z_top=-1.0)
        with pytest.raises(ValueError, match="z_bottom"):
            self.jaws(z_bottom=39.0)
        with pytest.raises(ValueError, match="edge"):
            self.jaws(edge_neg=3.0)  # crosses edge_pos = 2
        with pytest.raises(ValueError, match="axis"):
            self.jaws(axis="w")
        with pytest.raises(ValueError, match="density"):
            self.jaws(density=0.0)


class TestMLC:
    """Three leaf pairs, 2 cm wide, slab z in [48, 52] (H = 4), tip radius 10."""

    HEIGHT = 4.0
    RADIUS = 10.0

    def mlc(self, **overrides: object) -> MLC:
        settings: dict = {
            "z_top": 48.0,
            "z_bottom": 52.0,
            "leaf_edges_v": (-3.0, -1.0, 1.0, 3.0),
            "tips_neg": (-1.0, -1.0, -1.0),
            "tips_pos": (1.0, 1.0, 1.0),
            "tip_radius": self.RADIUS,
            "material": TUNGSTEN,
            "density": 18.0,
        }
        settings.update(overrides)
        return MLC(**settings)

    def axial(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        return _rays(((u, v, 0.0), (0.0, 0.0, 1.0)))

    def test_open_gap_between_tips_sees_nothing(self) -> None:
        np.testing.assert_array_equal(self.mlc().path_lengths(*self.axial(0.0, 0.0)), [0.0])

    def test_outside_the_bank_in_v_sees_nothing(self) -> None:
        """The outer leaf edges end the bank; the orthogonal jaws bound the field."""
        np.testing.assert_array_equal(self.mlc().path_lengths(*self.axial(5.0, 10.0)), [0.0])

    def test_axial_ray_deep_in_a_leaf_sees_the_slab_thickness(self) -> None:
        """Past the tip arc the leaf is slab-thick, on both sides of the disc center."""
        for u in (5.0, 15.0, -5.0, -15.0):
            np.testing.assert_allclose(self.mlc().path_lengths(*self.axial(u, 0.0)), [self.HEIGHT])

    def test_axial_ray_through_the_tip_arc_gets_the_exact_chord(self) -> None:
        """At lateral depth delta past the apex the chord is 2*sqrt(R^2 - (R-delta)^2).

        delta = 0.05 is inside the arc-only band (delta < R - sqrt(R^2 - (H/2)^2)
        = 0.2021 for R = 10, H = 4), so the chord is clipped by neither the slab
        nor the flat leaf body — the pure Pythagorean value.
        """
        delta = 0.05
        expected = 2.0 * math.sqrt(self.RADIUS**2 - (self.RADIUS - delta) ** 2)
        np.testing.assert_allclose(
            self.mlc().path_lengths(*self.axial(1.0 + delta, 0.0)), [expected]
        )
        np.testing.assert_allclose(
            self.mlc().path_lengths(*self.axial(-1.0 - delta, 0.0)), [expected]
        )

    def test_closed_pair_touches_at_a_single_tangent_point(self) -> None:
        """Opposing tips at the same u touch tangentially: zero chord exactly there,
        the arc chord a hair to either side."""
        mlc = self.mlc(tips_neg=(-1.0, 0.5, -1.0), tips_pos=(1.0, 0.5, 1.0))
        np.testing.assert_allclose(mlc.path_lengths(*self.axial(0.5, 0.0)), [0.0])
        delta = 0.01
        expected = 2.0 * math.sqrt(self.RADIUS**2 - (self.RADIUS - delta) ** 2)
        np.testing.assert_allclose(mlc.path_lengths(*self.axial(0.5 + delta, 0.0)), [expected])
        np.testing.assert_allclose(mlc.path_lengths(*self.axial(0.5 - delta, 0.0)), [expected])

    def test_tip_profile_is_monotone_from_zero_to_slab_thickness(self) -> None:
        """Thickness vs lateral depth into a leaf rises 0 -> H without overshoot."""
        u_values = 1.0 + np.linspace(-0.5, 1.5, 201)
        origins = np.column_stack([u_values, np.zeros_like(u_values), np.zeros_like(u_values)])
        directions = np.tile([0.0, 0.0, 1.0], (u_values.size, 1))
        lengths = self.mlc().path_lengths(origins, directions)
        assert np.all(np.diff(lengths) >= -1e-12)
        assert lengths[0] == 0.0
        assert lengths[-1] == pytest.approx(self.HEIGHT)
        assert np.all(lengths <= self.HEIGHT + 1e-12)

    def test_no_interleaf_gap_across_adjacent_leaves(self) -> None:
        """A ray crossing the v boundary inside the banks sees one solid slab."""
        theta = math.radians(25.0)
        direction = (0.0, math.sin(theta), math.cos(theta))
        # Starts in leaf 0's strip (v = -2 at z = 48 -> v = -2 + 4*tan(25) = -0.13
        # at z = 52): crosses into leaf 1's strip mid-slab, deep in +u material.
        origins, directions = _rays(((5.0, -2.0 + -48.0 * math.tan(theta), 0.0), direction))
        np.testing.assert_allclose(
            self.mlc().path_lengths(origins, directions),
            [self.HEIGHT / math.cos(theta)],
        )

    def test_validation_raises(self) -> None:
        with pytest.raises(ValueError, match="tip_radius"):
            self.mlc(tip_radius=1.5)  # < H/2 = 2
        with pytest.raises(ValueError, match="tips_neg"):
            self.mlc(tips_neg=(-1.0, -1.0))  # 2 tips for 3 pairs
        with pytest.raises(ValueError, match="leaf_edges_v"):
            self.mlc(leaf_edges_v=(-3.0, 1.0, -1.0, 3.0))  # not increasing
        with pytest.raises(ValueError, match="tip"):
            self.mlc(tips_neg=(-1.0, 2.0, -1.0))  # crosses tips_pos[1] = 1
        with pytest.raises(ValueError, match="z_bottom"):
            self.mlc(z_bottom=47.0)
        with pytest.raises(ValueError, match="density"):
            self.mlc(density=-1.0)


def _example_devices() -> tuple[JawPair, JawPair, MLC]:
    jaw_u = JawPair(
        axis="u",
        z_top=28.0,
        z_bottom=35.0,
        edge_neg=-2.0,
        edge_pos=2.0,
        material=TUNGSTEN,
        density=18.0,
    )
    jaw_v = JawPair(
        axis="v",
        z_top=36.0,
        z_bottom=43.0,
        edge_neg=-2.5,
        edge_pos=2.5,
        material=TUNGSTEN,
        density=17.5,
    )
    mlc = MLC(
        z_top=48.0,
        z_bottom=52.0,
        leaf_edges_v=(-3.0, -1.0, 1.0, 3.0),
        tips_neg=(-1.0, -1.0, -1.0),
        tips_pos=(1.0, 1.0, 1.0),
        tip_radius=10.0,
        material=TUNGSTEN,
        density=18.0,
    )
    return jaw_u, jaw_v, mlc


class TestBeamLimitingStack:
    def stack(self, frame: BeamFrame | None = None) -> BeamLimitingStack:
        return BeamLimitingStack(
            frame=frame if frame is not None else BeamFrame(origin=(0.0, 0.0, 0.0)),
            devices=_example_devices(),
        )

    def test_columns_are_the_per_device_answers(self) -> None:
        """Composition is concatenation: devices do not interact geometrically."""
        stack = self.stack()
        origins, directions = _rays(
            ((3.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # under the +u jaw, deep in the MLC leaf
            ((0.0, 0.0, 0.0), (0.05, 0.02, 1.0)),  # a diverging open-field ray
            ((0.0, -3.0, 0.0), (0.0, 0.0, 1.0)),  # under the -v jaw, outside the MLC bank
        )
        lengths = stack.path_lengths(origins, directions)
        assert lengths.shape == (3, 3)
        for column, device in enumerate(_example_devices()):
            np.testing.assert_allclose(lengths[:, column], device.path_lengths(origins, directions))

    def test_rotated_frame_is_equivalent(self) -> None:
        """The adapter contract: rotating frame and rays together changes nothing."""
        s = 1.0 / math.sqrt(2.0)
        rotated = BeamFrame(
            origin=(5.0, -2.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
            v_axis=(-s, 0.0, s),
            w_axis=(s, 0.0, s),
        )
        axes = np.array([rotated.u_axis, rotated.v_axis, rotated.w_axis])
        rng = np.random.default_rng(20260715)
        local_origins = rng.uniform(-4.0, 4.0, size=(64, 3))
        local_origins[:, 2] = 0.0
        local_directions = rng.normal(0.0, 0.05, size=(64, 3))
        local_directions[:, 2] = 1.0
        local_directions /= np.linalg.norm(local_directions, axis=1, keepdims=True)

        identity = self.stack()
        engine_origins = local_origins @ axes + rotated.origin
        engine_directions = local_directions @ axes
        np.testing.assert_allclose(
            self.stack(rotated).path_lengths(engine_origins, engine_directions),
            identity.path_lengths(local_origins, local_directions),
            atol=1e-9,
        )

    def test_from_origin_ignores_devices_behind_the_start(self) -> None:
        """The pre-solve's scattered photons start mid-stack and fly forward only."""
        stack = self.stack()
        origins, directions = _rays(
            ((3.0, 0.0, 60.0), (0.0, 0.0, 1.0)),  # below everything, heading away
            ((3.0, 0.0, 31.5), (0.0, 0.0, 1.0)),  # inside the +u jaw block, halfway
        )
        lengths = stack.path_lengths(origins, directions, from_origin=True)
        np.testing.assert_allclose(lengths[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(lengths[1], [3.5, 0.0, 4.0])
        # The full-line default still sees all of it.
        np.testing.assert_allclose(stack.path_lengths(origins, directions)[0], [7.0, 0.0, 4.0])

    def test_properties(self) -> None:
        stack = self.stack()
        assert stack.materials == (TUNGSTEN, TUNGSTEN, TUNGSTEN)
        assert stack.densities == (18.0, 17.5, 18.0)
        assert stack.exit_z == 52.0

    def test_overlapping_devices_raise(self) -> None:
        jaw_u, _jaw_v, mlc = _example_devices()
        overlapping = JawPair(
            axis="v",
            z_top=30.0,  # inside jaw_u's [28, 35]
            z_bottom=43.0,
            edge_neg=-2.5,
            edge_pos=2.5,
            material=TUNGSTEN,
            density=17.5,
        )
        with pytest.raises(ValueError, match="overlap"):
            BeamLimitingStack(
                frame=BeamFrame(origin=(0.0, 0.0, 0.0)), devices=(jaw_u, overlapping, mlc)
            )

    def test_unordered_devices_raise(self) -> None:
        jaw_u, _jaw_v, mlc = _example_devices()
        with pytest.raises(ValueError, match="ascending"):
            BeamLimitingStack(frame=BeamFrame(origin=(0.0, 0.0, 0.0)), devices=(mlc, jaw_u))

    def test_empty_stack_raises(self) -> None:
        with pytest.raises(ValueError, match="device"):
            BeamLimitingStack(frame=BeamFrame(origin=(0.0, 0.0, 0.0)), devices=())
