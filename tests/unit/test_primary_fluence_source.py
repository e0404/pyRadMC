"""The primary-fluence virtual source: Tacke et al. (2006) geometry, as a weight.

Photons are born on a plane rectangle upstream of the beam-limiting devices and
aimed from a 2D-Gaussian focal spot — the
:class:`~pyradmc.geometry.source.GaussianSpotBeamSource` geometry, unchanged —
and the measured radial primary fluence psi(r) enters as the per-history
statistical **weight**, evaluated at the ray's radius projected onto the plane
the table was measured at.

Two properties carry the whole design and are pinned here:

1. **The geometry is bit-identical to the Gaussian-spot source.** Same six
   uniforms in the same order, same energy, same start point, same direction —
   only ``weight`` differs. That is what keeps correlated Dij sampling,
   chunk-invariance and the vectorized batch route working unchanged, and it
   makes the fluence model auditable as "the spot source, reweighted".
2. **The weight is psi at the projected radius**, computed here from the raw
   geometry rather than from the source's own helper, so the test is not
   circular.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc.geometry.fluence import RadialFluence
from pyradmc.geometry.source import (
    GaussianSpotBeamletSource,
    GaussianSpotBeamSource,
    PrimaryFluenceBeamletSource,
    PrimaryFluenceBeamSource,
)
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG, uniform
from tests.conftest import SEED

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
CENTER = (8.0, 8.0, -30.0)  # emission plane 70 cm from the spot, upstream of the jaws
PLANE_DISTANCE = CENTER[2] - FOCAL[2]
WIDTHS = (6.0, 6.0)
SIGMA = 0.15

# Horn out to 3 cm at the reference distance, then a roll-off to zero at 4.2 cm.
# A table radius r sits at 0.7 r on the emission plane, so the zero edge falls at
# plane radius 2.94 cm inside a rectangle of half-width 3: one field samples the
# flat region, the horn, the roll-off and the zero beyond it.
FLUENCE = RadialFluence(
    radii=(0.0, 2.0, 3.0, 3.6, 4.2),
    values=(1.0, 1.04, 1.08, 0.40, 0.0),
    reference_distance=100.0,
)

_SPOT_KWARGS: dict[str, object] = dict(
    spectrum=SPECTRUM,
    focal_point=FOCAL,
    center=CENTER,
    width_u=WIDTHS[0],
    width_v=WIDTHS[1],
    sigma_u=SIGMA,
    sigma_v=SIGMA,
)


def _source(**overrides: object) -> PrimaryFluenceBeamSource:
    kwargs = dict(_SPOT_KWARGS, fluence=FLUENCE)
    kwargs.update(overrides)
    return PrimaryFluenceBeamSource(**kwargs)  # type: ignore[arg-type]


def _expected_weight(x: float, y: float, fluence: RadialFluence = FLUENCE) -> float:
    """psi at the plane radius projected to the table's reference distance.

    Independent of the source: the plane is z = CENTER[2] with the default axes,
    so the projection is the similar-triangles scaling of the off-axis distance.
    """
    r_plane = math.hypot(x - FOCAL[0], y - FOCAL[1])
    return fluence.at_radius(r_plane * fluence.reference_distance / PLANE_DISTANCE)


class TestContract:
    def test_max_energy_is_the_spectrum_top(self) -> None:
        assert _source().max_energy == SPECTRUM.max_energy

    def test_emit_consumes_exactly_six_uniforms(self) -> None:
        """The fluence is a weight, not a draw: the stream is the spot source's."""
        rng = HostRNG()
        state = rng.init_state(SEED, 21)
        _source().emit(state)
        after_emit = uniform(state)
        reference = rng.init_state(SEED, 21)
        for _ in range(6):
            uniform(reference)
        assert after_emit == uniform(reference)

    def test_fluence_is_exposed(self) -> None:
        assert _source().fluence is FLUENCE


class TestGeometryIsUnchanged:
    def test_emit_matches_the_gaussian_spot_source_except_for_weight(self) -> None:
        """Bit-identical energy, start point and direction on the same stream."""
        spot = GaussianSpotBeamSource(**_SPOT_KWARGS)  # type: ignore[arg-type]
        source = _source()
        rng = HostRNG()
        for i in range(300):
            a = source.emit(rng.init_state(SEED, i))
            b = spot.emit(rng.init_state(SEED, i))
            assert a[:8] == b[:8], f"history {i}: geometry diverged from the spot source"
            assert b.weight == 1.0

    def test_a_flat_covering_table_degenerates_to_the_spot_source(self) -> None:
        """psi == 1 over the whole rectangle reproduces the spot source exactly."""
        flat = RadialFluence(radii=(0.0, 50.0), values=(1.0, 1.0))
        spot = GaussianSpotBeamSource(**_SPOT_KWARGS)  # type: ignore[arg-type]
        source = _source(fluence=flat)
        rng = HostRNG()
        for i in range(200):
            assert source.emit(rng.init_state(SEED, i)) == spot.emit(rng.init_state(SEED, i))


class TestWeighting:
    def test_weight_is_psi_at_the_projected_radius(self) -> None:
        rng = HostRNG()
        source = _source()
        for i in range(500):
            p = source.emit(rng.init_state(SEED, i))
            assert p.weight == pytest.approx(_expected_weight(p.x, p.y), abs=1e-12)

    def test_corners_beyond_the_table_carry_zero_weight(self) -> None:
        """Outside the primary collimator's field the source must contribute nothing."""
        rng = HostRNG()
        source = _source()
        zeroed = [
            p
            for p in (source.emit(rng.init_state(SEED, i)) for i in range(2_000))
            if p.weight == 0.0
        ]
        assert zeroed, "the 6 x 6 rectangle should reach past the 4.2 cm table edge"
        for p in zeroed:
            projected = math.hypot(p.x - FOCAL[0], p.y - FOCAL[1]) * 100.0 / PLANE_DISTANCE
            assert projected >= FLUENCE.max_radius

    def test_weight_at_reports_the_same_number_as_emit(self) -> None:
        source = _source()
        for x, y in ((8.0, 8.0), (9.4, 8.0), (8.0, 10.0), (10.5, 10.5)):
            assert source.weight_at((x, y, CENTER[2])) == pytest.approx(_expected_weight(x, y))

    def test_on_axis_weight_is_the_axis_value(self) -> None:
        assert _source().weight_at((FOCAL[0], FOCAL[1], CENTER[2])) == pytest.approx(1.0)


class TestBatchRoute:
    def test_sample_batch_is_chunk_invariant(self) -> None:
        source = _source()
        whole = source.sample_batch(SEED, 0, 64)
        parts = [source.sample_batch(SEED, 0, 24), source.sample_batch(SEED, 24, 40)]
        for name in whole:
            np.testing.assert_array_equal(
                whole[name], np.concatenate([p[name] for p in parts]), err_msg=name
            )

    def test_batch_geometry_matches_the_spot_source(self) -> None:
        spot = GaussianSpotBeamSource(**_SPOT_KWARGS)  # type: ignore[arg-type]
        mine = _source().sample_batch(SEED, 0, 2_000)
        theirs = spot.sample_batch(SEED, 0, 2_000)
        for name in ("particle_type", "energy", "x", "y", "z", "ux", "uy", "uz"):
            np.testing.assert_array_equal(mine[name], theirs[name], err_msg=name)
        assert np.all(theirs["weight"] == 1.0)

    def test_batch_weights_are_psi_at_the_batch_positions(self) -> None:
        batch = _source().sample_batch(SEED, 0, 2_000)
        expected = [
            _expected_weight(float(x), float(y))
            for x, y in zip(batch["x"], batch["y"], strict=True)
        ]
        np.testing.assert_allclose(batch["weight"], expected, rtol=1e-5, atol=1e-7)

    def test_batch_reaches_the_zero_region(self) -> None:
        batch = _source().sample_batch(SEED, 0, 2_000)
        assert np.any(batch["weight"] == 0.0)
        assert np.any(batch["weight"] > 1.0)


class TestBeamletVariant:
    CENTERS = ((6.0, 8.0, -30.0), (10.0, 8.0, -30.0))

    def _source(self, **overrides: object) -> PrimaryFluenceBeamletSource:
        kwargs: dict[str, object] = dict(
            spectrum=SPECTRUM,
            fluence=FLUENCE,
            focal_point=FOCAL,
            centers=self.CENTERS,
            width_u=2.0,
            width_v=2.0,
            sigma_u=SIGMA,
            sigma_v=SIGMA,
        )
        kwargs.update(overrides)
        return PrimaryFluenceBeamletSource(**kwargs)  # type: ignore[arg-type]

    def test_beamlet_count(self) -> None:
        assert self._source().n_beamlets == 2

    def test_geometry_matches_the_gaussian_spot_beamlet_source(self) -> None:
        spot = GaussianSpotBeamletSource(
            spectrum=SPECTRUM,
            focal_point=FOCAL,
            centers=self.CENTERS,
            width_u=2.0,
            width_v=2.0,
            sigma_u=SIGMA,
            sigma_v=SIGMA,
        )
        source = self._source()
        rng = HostRNG()
        for beamlet in (0, 1):
            for i in range(100):
                a = source.emit(beamlet, rng.init_state(SEED, i))
                b = spot.emit(beamlet, rng.init_state(SEED, i))
                assert a[:8] == b[:8]

    def test_draws_are_beamlet_blind_but_weights_are_not(self) -> None:
        """Correlated replay: same energy and in-rectangle offset in every bixel;
        the weight differs because the bixels sit at different off-axis radii."""
        source = self._source()
        rng = HostRNG()
        a = source.emit(0, rng.init_state(SEED, 9))
        b = source.emit(1, rng.init_state(SEED, 9))
        assert a.energy == b.energy
        np.testing.assert_allclose(
            [a.x - self.CENTERS[0][0], a.y - self.CENTERS[0][1]],
            [b.x - self.CENTERS[1][0], b.y - self.CENTERS[1][1]],
            atol=1e-12,
        )
        assert a.weight == pytest.approx(_expected_weight(a.x, a.y))
        assert b.weight == pytest.approx(_expected_weight(b.x, b.y))
        assert a.weight != b.weight

    def test_beamlet_batch_weights_track_the_bixel_position(self) -> None:
        source = self._source()
        for beamlet, center in enumerate(self.CENTERS):
            batch = source.sample_beamlet_batch(SEED, 0, 256, beamlet=beamlet)
            expected = [
                _expected_weight(float(x), float(y))
                for x, y in zip(batch["x"], batch["y"], strict=True)
            ]
            np.testing.assert_allclose(batch["weight"], expected, rtol=1e-5, atol=1e-7)
            assert center is self.CENTERS[beamlet]

    def test_out_of_range_beamlet_raises(self) -> None:
        with pytest.raises(IndexError, match="beamlet 2"):
            self._source().emit(2, HostRNG().init_state(SEED, 0))


class TestPlaneValidation:
    def test_a_plane_through_the_focal_spot_raises(self) -> None:
        """Zero axial separation has no projection to the reference distance: the
        radius would blow up. The only genuinely ill-posed placement."""
        with pytest.raises(ValueError, match="downstream"):
            _source(center=(12.0, 8.0, FOCAL[2]))

    def test_a_beam_along_minus_z_is_accepted(self) -> None:
        """The central axis is derived from the aperture centre, not assumed to be
        +z, so a mirrored convention works and reads the same radii."""
        mirrored = _source(focal_point=(8.0, 8.0, 100.0), center=(8.0, 8.0, 30.0))
        assert mirrored.weight_at((8.0, 8.0, 30.0)) == pytest.approx(1.0)
        assert mirrored.weight_at((10.0, 8.0, 30.0)) == pytest.approx(_expected_weight(10.0, 8.0))

    def test_non_coplanar_beamlet_centres_raise(self) -> None:
        """One reference plane is what a measured primary fluence is defined on."""
        with pytest.raises(ValueError, match="same plane"):
            PrimaryFluenceBeamletSource(
                spectrum=SPECTRUM,
                fluence=FLUENCE,
                focal_point=FOCAL,
                centers=((6.0, 8.0, -30.0), (10.0, 8.0, -25.0)),
                width_u=2.0,
                width_v=2.0,
                sigma_u=SIGMA,
                sigma_v=SIGMA,
            )
