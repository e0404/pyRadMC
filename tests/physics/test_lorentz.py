"""Charged-particle motion in a static magnetic field: exact kinematic identities.

Everything here is algebra, not statistics: momentum, gyroradius and the Rodrigues
rotation have closed forms, so the tolerances are float precision. The one physics
convention worth pinning by hand is the handedness: an electron moving along +x in a
field along +z curves toward +y (F = q v x B with q = -e), a positron toward -y.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from pyradmc import ELECTRON_MASS_MEV, MEV_C_PER_TESLA_CM
from pyradmc.physics.direction import sample_isotropic_direction
from pyradmc.physics.lorentz import (
    gyroradius_cm,
    lorentz_rotation_angle,
    momentum_mev_c,
    rotate_about_axis,
)
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

EXACT = 1e-12


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class TestMomentum:
    def test_one_mev_hand_value(self) -> None:
        """p = sqrt(T (T + 2 m)) at T = 1 MeV is 1.4220 MeV/c (four figures)."""
        assert momentum_mev_c(1.0) == pytest.approx(1.4220, abs=5e-5)

    @pytest.mark.parametrize("e_kin", [0.05, 0.2, 1.0, 6.0, 20.0])
    def test_matches_relativistic_energy_momentum_relation(self, e_kin: float) -> None:
        """(pc)^2 = E_total^2 - m^2, computed independently of the implementation."""
        total = e_kin + ELECTRON_MASS_MEV
        want = math.sqrt(total * total - ELECTRON_MASS_MEV * ELECTRON_MASS_MEV)
        assert momentum_mev_c(e_kin) == pytest.approx(want, rel=EXACT)

    def test_limits(self) -> None:
        """Non-relativistic sqrt(2 m T) at low T; ultra-relativistic p -> T + m at high T."""
        classical = math.sqrt(2.0 * ELECTRON_MASS_MEV * 1e-6)
        assert momentum_mev_c(1e-6) == pytest.approx(classical, rel=1e-5)
        assert momentum_mev_c(1e4) == pytest.approx(1e4 + ELECTRON_MASS_MEV, rel=1e-8)


class TestGyroradius:
    @pytest.mark.parametrize(
        ("b_tesla", "anchor_cm", "half_unit_cm"),
        [
            (1.5, 0.316, 5e-4),  # MR-linac (Unity): 3.2 mm
            (0.35, 1.36, 5e-3),  # MR-linac (MRIdian): 14 mm
        ],
    )
    def test_one_mev_against_constants_and_external_anchor(
        self, b_tesla: float, anchor_cm: float, half_unit_cm: float
    ) -> None:
        """r_g = p / (2.99792458 B) with p from the kinematics, plus a 3-figure anchor.

        The expected value is recomputed here from the package constants (an
        implementation mistake in the formula cannot cancel); the literal anchors are
        the textbook numbers to three significant figures.
        """
        p = math.sqrt(1.0 * (1.0 + 2.0 * ELECTRON_MASS_MEV))
        want = p / (MEV_C_PER_TESLA_CM * b_tesla)
        got = gyroradius_cm(1.0, b_tesla)
        assert got == pytest.approx(want, rel=EXACT)
        assert got == pytest.approx(anchor_cm, abs=half_unit_cm)

    def test_constant_is_the_speed_of_light(self) -> None:
        """p[MeV/c] = c[1e8 m/s] * B[T] * r[cm] exactly: the constant is c / 1e8, no fit."""
        assert MEV_C_PER_TESLA_CM == 2.99792458

    def test_inverse_in_field_strength(self) -> None:
        """r_g B is independent of B for fixed energy."""
        strong = gyroradius_cm(3.0, 1.5) * 1.5
        weak = gyroradius_cm(3.0, 0.35) * 0.35
        assert strong == pytest.approx(weak, rel=EXACT)

    def test_grows_with_energy(self) -> None:
        energies = [0.2, 0.5, 1.0, 2.0, 6.0, 20.0]
        radii = [gyroradius_cm(e, 1.5) for e in energies]
        assert all(a < b for a, b in itertools.pairwise(radii))
        # ECUT-scale electrons at 1.5 T: ~1.1 mm; the plan's working numbers.
        assert radii[0] == pytest.approx(0.11, abs=0.01)


class TestRotateAboutAxis:
    def test_x_about_z_by_quarter_turn_is_y(self) -> None:
        """Right-handed convention: positive theta about +z takes +x toward +y."""
        vx, vy, vz = rotate_about_axis(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5 * math.pi)
        assert vx == pytest.approx(0.0, abs=EXACT)
        assert vy == pytest.approx(1.0, abs=EXACT)
        assert vz == pytest.approx(0.0, abs=EXACT)

    def test_zero_angle_returns_the_identical_floats(self) -> None:
        """theta = 0 must be a no-op bit for bit: B = 0 runs are pinned bit-identical."""
        u = (0.6, -0.48, 0.64)
        a = (0.267261241912424, 0.534522483824849, 0.801783725737273)
        assert rotate_about_axis(*u, *a, 0.0) == u

    def test_rotation_about_own_axis_is_identity(self) -> None:
        """u parallel to the axis: a x u = 0 and (a . u) a = u, for any angle."""
        a = (0.6, -0.48, 0.64)
        for theta in (0.1, 1.0, math.pi, -2.5):
            v = rotate_about_axis(*a, *a, theta)
            assert v == pytest.approx(a, abs=EXACT)
        anti = (-0.6, 0.48, -0.64)
        assert rotate_about_axis(*anti, *a, 1.3) == pytest.approx(anti, abs=EXACT)

    def test_random_rotations_preserve_norm_axial_component_and_angle(self) -> None:
        """A rotation about a is an isometry fixing a: |v| = |u|, a . v = a . u, and the
        transverse part turns by exactly theta.
        """
        state = HostRNG().init_state(SEED, 20)
        for _ in range(500):
            u = sample_isotropic_direction(state)
            a = sample_isotropic_direction(state)
            theta = (2.0 * state.random() - 1.0) * math.pi
            v = rotate_about_axis(*u, *a, theta)

            assert abs(_norm(v) - 1.0) < 1e-12
            assert abs(_dot(a, v) - _dot(a, u)) < 1e-12

            # Transverse parts: angle from u_perp to v_perp, signed about a.
            au = _dot(a, u)
            u_perp = np.array(u) - au * np.array(a)
            v_perp = np.array(v) - au * np.array(a)
            cos_t = float(np.dot(u_perp, v_perp)) / float(np.dot(u_perp, u_perp))
            sin_t = float(np.dot(np.cross(u_perp, v_perp), a)) / float(np.dot(u_perp, u_perp))
            assert math.atan2(sin_t, cos_t) == pytest.approx(theta, abs=1e-9)

    def test_composition_equals_rotation_by_the_sum(self) -> None:
        state = HostRNG().init_state(SEED, 21)
        for _ in range(200):
            u = sample_isotropic_direction(state)
            a = sample_isotropic_direction(state)
            t1 = (2.0 * state.random() - 1.0) * math.pi
            t2 = (2.0 * state.random() - 1.0) * math.pi
            two_step = rotate_about_axis(*rotate_about_axis(*u, *a, t1), *a, t2)
            one_step = rotate_about_axis(*u, *a, t1 + t2)
            assert two_step == pytest.approx(one_step, abs=1e-12)

    @pytest.mark.parametrize("n_chords", [4, 7, 64, 1000])
    def test_full_circle_of_chords_returns_to_start(self, n_chords: int) -> None:
        """N rotations by 2 pi / N compose to the identity: a closed gyration orbit."""
        u = (0.36, 0.48, 0.8)
        a = (0.0, 0.0, 1.0)
        v = u
        for _ in range(n_chords):
            v = rotate_about_axis(*v, *a, 2.0 * math.pi / n_chords)
        assert v == pytest.approx(u, abs=1e-12 * n_chords)

    def test_inverse_rotation_undoes(self) -> None:
        u = (0.36, 0.48, 0.8)
        a = (-0.48, 0.64, 0.6)
        v = rotate_about_axis(*rotate_about_axis(*u, *a, 0.7), *a, -0.7)
        assert v == pytest.approx(u, abs=EXACT)


class TestLorentzRotationAngle:
    def test_magnitude_is_arc_over_radius(self) -> None:
        assert abs(lorentz_rotation_angle(0.05, 0.25, False)) == pytest.approx(0.2, rel=EXACT)
        assert abs(lorentz_rotation_angle(0.05, 0.25, True)) == pytest.approx(0.2, rel=EXACT)

    def test_sign_electron_positive_positron_negative(self) -> None:
        """About +B_hat, the electron turns by +s/r_g and the positron by -s/r_g."""
        assert lorentz_rotation_angle(0.1, 1.0, False) == pytest.approx(+0.1, rel=EXACT)
        assert lorentz_rotation_angle(0.1, 1.0, True) == pytest.approx(-0.1, rel=EXACT)

    @pytest.mark.parametrize(("is_positron", "charge"), [(False, -1.0), (True, +1.0)])
    def test_curvature_follows_q_v_cross_b(self, is_positron: bool, charge: float) -> None:
        """End to end: an electron along +x in B along +z curves toward +y.

        The initial acceleration is along q (v x B); the first-order change of the
        direction over a short chord must point the same way. Evaluated for a generic
        (u, B_hat) pair as well, so the pin is the physics, not one axis-aligned case.
        """
        for u, b_hat in [
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((0.36, 0.48, 0.8), (-0.48, 0.64, 0.6)),
        ]:
            theta = lorentz_rotation_angle(1e-4, 1.0, is_positron)
            v = rotate_about_axis(*u, *b_hat, theta)
            dv = np.array(v) - np.array(u)
            force = charge * np.cross(u, b_hat)
            assert np.dot(dv, force) > 0.0
            # The axis-aligned case literally: +x, B +z -> electron +y, positron -y.
            if u == (1.0, 0.0, 0.0):
                assert math.copysign(1.0, v[1]) == -charge
