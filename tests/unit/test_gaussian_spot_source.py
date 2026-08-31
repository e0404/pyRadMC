"""The planar Gaussian-spot source: photons born on a plane, aimed from a spot.

The maintainer's simplified head-input source (BLD workstream): primaries start
**on** a rectangular region of a plane (e.g. just upstream of the limiting
devices) travelling as if they came in a straight line from a 2D-Gaussian focal
spot — the standard finite-source-size model, so downstream collimation acquires
a geometric penumbra with no extra machinery.

Contract under test: ``emit`` consumes **exactly six uniforms in fixed order**
(two spectrum, two in-rectangle offsets, two Box-Muller for the spot — drawn even
at zero sigma so the count never varies), which keeps the vectorized
``PCG64.advance(6 * offset)`` batch chunk-invariant and beamlet-blind, exactly
like the spectral fan it generalizes: at ``sigma = 0`` the emitted *lines* are the
fan's lines, only the start point moves from the focal spot to the plane.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.geometry.source import (
    GaussianSpotBeamletSource,
    GaussianSpotBeamSource,
    SpectralBeamSource,
)
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG, uniform
from tests.conftest import SEED

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
CENTER = (8.0, 8.0, -30.0)  # the emission plane, upstream of a hypothetical stack
WIDTHS = (6.0, 6.0)
SIGMA = 0.15


def _source(**overrides: object) -> GaussianSpotBeamSource:
    kwargs: dict[str, object] = dict(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        sigma_u=SIGMA,
        sigma_v=SIGMA,
        center=CENTER,
        width_u=WIDTHS[0],
        width_v=WIDTHS[1],
    )
    kwargs.update(overrides)
    return GaussianSpotBeamSource(**kwargs)  # type: ignore[arg-type]


class TestContract:
    def test_max_energy_is_the_spectrum_top(self) -> None:
        assert _source().max_energy == SPECTRUM.max_energy

    def test_emit_consumes_exactly_six_uniforms(self) -> None:
        """Fixed draw count, sigma zero or not — the vectorized-batch requirement."""
        rng = HostRNG()
        for sigma in (0.0, SIGMA):
            source = _source(sigma_u=sigma, sigma_v=sigma)
            state = rng.init_state(SEED, 21)
            source.emit(state)
            after_emit = uniform(state)
            reference = rng.init_state(SEED, 21)
            for _ in range(6):
                uniform(reference)
            assert after_emit == uniform(reference), f"sigma={sigma}"

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            _source(sigma_u=-0.1)

    def test_primaries_start_on_the_plane_rectangle(self) -> None:
        rng = HostRNG()
        source = _source()
        for i in range(200):
            p = source.emit(rng.init_state(SEED, i))
            assert p.z == CENTER[2]  # default axes: the plane is z = center_z
            assert abs(p.x - CENTER[0]) <= WIDTHS[0] / 2 + 1e-12
            assert abs(p.y - CENTER[1]) <= WIDTHS[1] / 2 + 1e-12
            assert p.ux**2 + p.uy**2 + p.uz**2 == pytest.approx(1.0, abs=1e-12)
            assert p.kind is None and p.weight == 1.0


class TestGeometry:
    def test_zero_sigma_reproduces_the_fan_lines(self) -> None:
        """At sigma = 0 the first four draws match the spectral fan's, so the same
        stream gives the same energy and the same line — started on the plane."""
        fan = SpectralBeamSource(
            spectrum=SPECTRUM,
            focal_point=FOCAL,
            center=CENTER,
            width_u=WIDTHS[0],
            width_v=WIDTHS[1],
        )
        source = _source(sigma_u=0.0, sigma_v=0.0)
        rng = HostRNG()
        for i in range(100):
            planar = source.emit(rng.init_state(SEED, i))
            through = fan.emit(rng.init_state(SEED, i))
            assert planar.energy == through.energy
            np.testing.assert_allclose(
                [planar.ux, planar.uy, planar.uz],
                [through.ux, through.uy, through.uz],
                atol=1e-12,
            )
            # The start point lies on the fan line from the focal spot.
            t = (planar.z - FOCAL[2]) / through.uz
            np.testing.assert_allclose(
                [FOCAL[0] + t * through.ux, FOCAL[1] + t * through.uy],
                [planar.x, planar.y],
                atol=1e-9,
            )

    def test_back_projected_spot_is_gaussian(self) -> None:
        """Extending each emitted line back to the focal plane recovers the spot.

        Fixed seed; the sample mean and standard deviation of the recovered spot
        offsets must sit near (0, sigma) — a distribution-shape regression pin,
        not a hypothesis test.
        """
        rng = HostRNG()
        source = _source()
        n = 4_000
        offsets_u = np.empty(n)
        offsets_v = np.empty(n)
        for i in range(n):
            p = source.emit(rng.init_state(SEED, i))
            t = (FOCAL[2] - p.z) / p.uz
            offsets_u[i] = p.x + t * p.ux - FOCAL[0]
            offsets_v[i] = p.y + t * p.uy - FOCAL[1]
        for offsets in (offsets_u, offsets_v):
            assert abs(offsets.mean()) < 4.0 * SIGMA / np.sqrt(n)
            assert offsets.std() == pytest.approx(SIGMA, rel=0.05)


class TestBatchRoute:
    def test_sample_batch_is_chunk_invariant(self) -> None:
        source = _source()
        whole = source.sample_batch(SEED, 0, 64)
        parts = [source.sample_batch(SEED, 0, 24), source.sample_batch(SEED, 24, 40)]
        for name in whole:
            np.testing.assert_array_equal(
                whole[name], np.concatenate([p[name] for p in parts]), err_msg=name
            )

    def test_batch_obeys_the_plane_contract(self) -> None:
        batch = _source().sample_batch(SEED, 0, 2_000)
        assert np.all(batch["z"] == np.float32(CENTER[2]))
        assert np.all(np.abs(batch["x"] - CENTER[0]) <= WIDTHS[0] / 2 + 1e-6)
        assert np.all(batch["weight"] == 1.0)
        assert np.all(batch["particle_type"] == 1)


class TestBeamletVariant:
    CENTERS = ((6.0, 8.0, -30.0), (10.0, 8.0, -30.0))

    def _source(self) -> GaussianSpotBeamletSource:
        return GaussianSpotBeamletSource(
            spectrum=SPECTRUM,
            focal_point=FOCAL,
            sigma_u=SIGMA,
            sigma_v=SIGMA,
            centers=self.CENTERS,
            width_u=2.0,
            width_v=2.0,
        )

    def test_beamlet_count(self) -> None:
        assert self._source().n_beamlets == 2

    def test_correlated_replay_across_beamlets(self) -> None:
        """Same stream => same energy and same in-rectangle offset in every bixel."""
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

    def test_beamlet_batch_is_beamlet_blind(self) -> None:
        source = self._source()
        a = source.sample_beamlet_batch(SEED, 0, 32, beamlet=0)
        b = source.sample_beamlet_batch(SEED, 0, 32, beamlet=1)
        np.testing.assert_array_equal(a["energy"], b["energy"])
        np.testing.assert_allclose(
            a["x"] - self.CENTERS[0][0], b["x"] - self.CENTERS[1][0], atol=1e-5
        )
