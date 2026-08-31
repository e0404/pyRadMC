"""Transmission-mask wrappers: user-supplied 0..1 fluence shaping on a plane.

Configuration 2 of the BLD workstream: a 2D transmission mask over the open field
(sequenced MLC shapes, aperture optimization) applied as a deterministic weight —
no cross-sections, zero extra uniforms, so correlated Dij sampling and the
vectorized batch routes survive exactly as for the attenuation wrappers.

Mask convention under test: ``mask[i, j]`` covers pixel ``i`` along ``u_axis`` and
``j`` along ``v_axis``; pixel centres tile the rectangle; lookups are bilinear
between pixel centres (clamped at the border band), and rays crossing the plane
outside the rectangle — or running parallel to it — carry weight zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.geometry.collimation import TransmissionMaskBeamletSource, TransmissionMaskSource
from pyradmc.geometry.source import ParallelBeamSource, PencilBeamSource, SpectralBeamletSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG, uniform
from tests.conftest import SEED

ENERGY = 6.0
CENTER = (8.0, 8.0, 0.0)
WIDTHS = (4.0, 4.0)  # 2x2 mask below => pixel centres at 7/9 on each axis


def _mask_source(inner: object, mask: np.ndarray) -> TransmissionMaskSource:
    return TransmissionMaskSource(
        inner,  # type: ignore[arg-type]
        mask=mask,
        plane_center=CENTER,
        width_u=WIDTHS[0],
        width_v=WIDTHS[1],
    )


def _pencil(x: float, y: float) -> PencilBeamSource:
    return PencilBeamSource(ENERGY, (x, y, -50.0), (0.0, 0.0, 1.0))


MASK_2X2 = np.array([[0.1, 0.4], [0.7, 1.0]])


class TestLookup:
    @pytest.mark.parametrize(
        ("x", "y", "expected"),
        [
            (7.0, 7.0, 0.1),  # pixel (0, 0)
            (7.0, 9.0, 0.4),  # pixel (0, 1): j runs along v = y
            (9.0, 7.0, 0.7),  # pixel (1, 0): i runs along u = x
            (9.0, 9.0, 1.0),
        ],
    )
    def test_pixel_centres_pick_up_exact_values(self, x: float, y: float, expected: float) -> None:
        source = _mask_source(_pencil(x, y), MASK_2X2)
        assert source.emit(HostRNG().init_state(SEED, 0)).weight == pytest.approx(expected)

    def test_bilinear_midpoint_is_the_mean_of_four(self) -> None:
        source = _mask_source(_pencil(8.0, 8.0), MASK_2X2)
        assert source.emit(HostRNG().init_state(SEED, 0)).weight == pytest.approx(0.55)

    def test_all_ones_mask_is_weight_inert(self) -> None:
        """A fully open mask changes nothing, bit for bit."""
        source = _mask_source(_pencil(7.3, 8.6), np.ones((5, 7)))
        assert source.emit(HostRNG().init_state(SEED, 0)).weight == 1.0

    def test_outside_the_rectangle_is_zero(self) -> None:
        source = _mask_source(_pencil(14.0, 8.0), MASK_2X2)  # u = +6 > width/2
        assert source.emit(HostRNG().init_state(SEED, 0)).weight == 0.0

    def test_ray_parallel_to_the_plane_is_zero(self) -> None:
        inner = PencilBeamSource(ENERGY, (7.0, 7.0, 0.0), (1.0, 0.0, 0.0))
        source = _mask_source(inner, MASK_2X2)
        assert source.emit(HostRNG().init_state(SEED, 0)).weight == 0.0

    def test_upstream_plane_still_counts(self) -> None:
        """Full-line convention, like the attenuation stack: emission below the
        mask plane is still shaped by it."""
        inner = PencilBeamSource(ENERGY, (9.0, 9.0, 30.0), (0.0, 0.0, 1.0))
        source = _mask_source(inner, MASK_2X2)
        assert source.emit(HostRNG().init_state(SEED, 0)).weight == pytest.approx(1.0)


class TestValidation:
    def test_mask_geometry_is_exposed_for_device_generators(self) -> None:
        inner = _pencil(8.0, 8.0)
        mask = MASK_2X2.copy()
        source = TransmissionMaskSource(
            inner,
            mask=mask,
            plane_center=(8.0, 9.0, 10.0),
            width_u=4.0,
            width_v=6.0,
            u_axis=(0.0, 1.0, 0.0),
            v_axis=(0.0, 0.0, 1.0),
        )
        assert source.inner is inner
        np.testing.assert_array_equal(source.mask, mask)
        assert source.plane_center == (8.0, 9.0, 10.0)
        assert source.width_u == 4.0
        assert source.width_v == 6.0
        assert source.u_axis == (0.0, 1.0, 0.0)
        assert source.v_axis == (0.0, 0.0, 1.0)

    def test_values_outside_unit_interval_raise(self) -> None:
        with pytest.raises(ValueError, match=r"0\.\.1"):
            _mask_source(_pencil(8.0, 8.0), np.array([[0.5, 1.5]]))

    def test_non_2d_mask_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            _mask_source(_pencil(8.0, 8.0), np.ones(4))

    def test_non_positive_width_raises(self) -> None:
        with pytest.raises(ValueError, match="width"):
            TransmissionMaskSource(
                _pencil(8.0, 8.0),
                mask=MASK_2X2,
                plane_center=CENTER,
                width_u=0.0,
                width_v=4.0,
            )


class TestRngAndRoutes:
    def test_emit_consumes_exactly_the_inner_draws(self) -> None:
        inner = ParallelBeamSource(ENERGY, -50.0, (6.0, 10.0), (6.0, 10.0))
        source = _mask_source(inner, MASK_2X2)
        rng = HostRNG()
        wrapped_state = rng.init_state(SEED, 5)
        source.emit(wrapped_state)
        inner_state = rng.init_state(SEED, 5)
        inner.emit(inner_state)
        assert uniform(wrapped_state) == uniform(inner_state)

    def test_sample_batch_matches_emit(self) -> None:
        inner = ParallelBeamSource(ENERGY, -50.0, (5.0, 11.0), (5.0, 11.0))
        source = _mask_source(inner, MASK_2X2)
        n = 64
        batch = source.sample_batch(SEED, 0, n)
        rng = HostRNG()
        emit_weights = [source.emit(rng.init_state(SEED, i)).weight for i in range(n)]
        np.testing.assert_allclose(batch["weight"], emit_weights, rtol=1e-5, atol=1e-7)


class TestBeamletTwin:
    def _source(self) -> TransmissionMaskBeamletSource:
        inner = SpectralBeamletSource(
            spectrum=Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0)),
            focal_point=(8.0, 8.0, -100.0),
            centers=((6.0, 8.0, 0.0), (10.0, 8.0, 0.0)),
            width_u=2.0,
            width_v=2.0,
        )
        # Bixel 0 (x in [5, 7]) fully open, bixel 1 (x in [9, 11]) at half
        # transmission. 8 pixels along u (centres -3.5..3.5 about the plane
        # centre) keep the 1.0 -> 0.5 interpolation band inside |u| < 0.5, so
        # both bixels' rays cross plateaus and pick up exact values.
        mask = np.array([[1.0]] * 4 + [[0.5]] * 4)
        return TransmissionMaskBeamletSource(
            inner,
            mask=mask,
            plane_center=(8.0, 8.0, 0.0),
            width_u=8.0,
            width_v=8.0,
        )

    def test_beamlet_count_passes_through(self) -> None:
        assert self._source().n_beamlets == 2

    def test_batch_weights_follow_the_mask(self) -> None:
        source = self._source()
        open_batch = source.sample_beamlet_batch(SEED, 0, 64, beamlet=0)
        half_batch = source.sample_beamlet_batch(SEED, 0, 64, beamlet=1)
        np.testing.assert_array_equal(open_batch["weight"], np.float32(1.0))
        np.testing.assert_array_equal(half_batch["weight"], np.float32(0.5))
