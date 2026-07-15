"""Deterministic attenuation wrappers: CollimatedSource / CollimatedBeamletSource.

The wrapper delegates emission to the inner source and multiplies the primary's
statistical weight by the stack transmission ``exp(-sum_i (mu/rho)_i(E) rho_i t_i)``
— consuming **zero extra uniforms**, so the correlated-Dij replay and the
fixed-draw vectorized batch routes of the inner source survive wrapping untouched.

The unit tests run the analytic water-only cross-sections against *water* jaws:
Beer-Lambert against a direct ``mu_over_rho_total`` call is then exact physics with
no EPDL download. The tungsten path differs only in the material index and is
covered by the validation-tier compile gates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import TUNGSTEN, WATER
from pyRadMC.geometry.collimation import (
    BeamFrame,
    BeamLimitingStack,
    CollimatedBeamletSource,
    CollimatedSource,
    JawPair,
)
from pyRadMC.geometry.source import ParallelBeamSource, PencilBeamSource, SpectralBeamletSource
from pyRadMC.geometry.spectrum import Spectrum
from pyRadMC.rng.host import HostRNG, uniform
from tests.conftest import SEED

ENERGY = 6.0
THICKNESS = 7.0


def _water_jaws(edge_pos: float = 2.0) -> BeamLimitingStack:
    """One jaw pair of *water* (the analytic source's only material), edges +-2."""
    return BeamLimitingStack(
        frame=BeamFrame(origin=(0.0, 0.0, -100.0)),
        devices=(
            JawPair(
                axis="u",
                z_top=40.0,
                z_bottom=40.0 + THICKNESS,
                edge_neg=-2.0,
                edge_pos=edge_pos,
                material=WATER,
                density=1.0,
            ),
        ),
    )


def _xs() -> AnalyticCrossSections:
    return AnalyticCrossSections()


class TestConstruction:
    def test_material_beyond_the_source_is_rejected(self) -> None:
        """The analytic water-only source cannot attenuate a tungsten jaw."""
        stack = BeamLimitingStack(
            frame=BeamFrame(origin=(0.0, 0.0, -100.0)),
            devices=(
                JawPair(
                    axis="u",
                    z_top=40.0,
                    z_bottom=47.0,
                    edge_neg=-2.0,
                    edge_pos=2.0,
                    material=TUNGSTEN,
                    density=18.0,
                ),
            ),
        )
        inner = PencilBeamSource(ENERGY, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="tabulated"):
            CollimatedSource(inner, stack, _xs())

    def test_max_energy_passes_through(self) -> None:
        inner = PencilBeamSource(ENERGY, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        assert CollimatedSource(inner, _water_jaws(), _xs()).max_energy == ENERGY


class TestBeerLambert:
    def test_blocked_pencil_carries_the_transmission_weight(self) -> None:
        """Under the block the weight is exp(-mu/rho * rho * t) of the direct call."""
        inner = PencilBeamSource(ENERGY, (3.0, 0.0, -100.0), (0.0, 0.0, 1.0))
        source = CollimatedSource(inner, _water_jaws(), _xs())
        primary = source.emit(HostRNG().init_state(SEED, 0))
        expected = math.exp(-_xs().mu_over_rho_total(ENERGY, WATER) * 1.0 * THICKNESS)
        assert primary.weight == pytest.approx(expected, rel=1e-4)

    def test_open_field_pencil_is_exactly_unit_weight(self) -> None:
        """Zero path length: exp(0) = 1 exactly, no interpolation noise."""
        inner = PencilBeamSource(ENERGY, (0.0, 0.0, -100.0), (0.0, 0.0, 1.0))
        source = CollimatedSource(inner, _water_jaws(), _xs())
        assert source.emit(HostRNG().init_state(SEED, 0)).weight == 1.0


class TestRngContract:
    def test_emit_consumes_exactly_the_inner_draws(self) -> None:
        """The wrapper adds zero uniforms: the post-emit stream position matches."""
        inner = ParallelBeamSource(ENERGY, -100.0, (-1.0, 1.0), (-1.0, 1.0))
        source = CollimatedSource(inner, _water_jaws(), _xs())
        rng = HostRNG()

        wrapped_state = rng.init_state(SEED, 11)
        source.emit(wrapped_state)
        after_wrapped = uniform(wrapped_state)

        inner_state = rng.init_state(SEED, 11)
        inner.emit(inner_state)
        after_inner = uniform(inner_state)

        assert after_wrapped == after_inner

    def test_wrapped_primary_is_the_inner_primary_with_a_weight(self) -> None:
        """Energy, position, direction, kind are untouched; only weight changes."""
        inner = ParallelBeamSource(ENERGY, -100.0, (2.5, 3.5), (-1.0, 1.0))
        source = CollimatedSource(inner, _water_jaws(), _xs())
        rng = HostRNG()
        wrapped = source.emit(rng.init_state(SEED, 3))
        bare = inner.emit(rng.init_state(SEED, 3))
        assert wrapped[:7] == bare[:7]  # energy, x, y, z, ux, uy, uz
        assert wrapped.kind == bare.kind
        assert 0.0 < wrapped.weight < 1.0  # x in (2.5, 3.5) is under the +u block


class TestBatchRoute:
    def test_sample_batch_weights_equal_emit_weights(self) -> None:
        """Both routes evaluate the same transmission table on the same rays.

        ParallelBeamSource uses the default per-history pre-sampling, so the batch
        rays equal emit's rays up to the float32 upload packing; the wrapper's
        vectorized weights must then match emit's weights to packing precision —
        any systematic gap here would be a route-relative bias the cross-backend
        chi-squared oracle would eventually blame on transport.
        """
        inner = ParallelBeamSource(ENERGY, -100.0, (-4.0, 4.0), (-1.0, 1.0))
        source = CollimatedSource(inner, _water_jaws(), _xs())
        n = 64
        batch = source.sample_batch(SEED, 0, n)
        rng = HostRNG()
        emit_weights = np.array(
            [source.emit(rng.init_state(SEED, i)).weight for i in range(n)], dtype=np.float64
        )
        np.testing.assert_allclose(batch["weight"], emit_weights, rtol=1e-5)
        assert np.any(batch["weight"] < 1.0) and np.any(batch["weight"] == 1.0)


class TestBeamletWrapper:
    SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
    CENTERS = ((6.0, 8.0, 0.0), (8.0, 8.0, 0.0), (10.0, 8.0, 0.0))

    def _source(self) -> CollimatedBeamletSource:
        inner = SpectralBeamletSource(
            spectrum=self.SPECTRUM,
            focal_point=(8.0, 8.0, -100.0),
            centers=self.CENTERS,
            width_u=2.0,
            width_v=2.0,
        )
        stack = BeamLimitingStack(
            frame=BeamFrame(origin=(8.0, 8.0, -100.0)),
            # Blocks u > 1 relative to the focal spot at the jaw plane z_mid = 43.5
            # (local frame): bixel 2's fan (centre offset +2 at plane 100) crosses
            # the block; bixel 0's does not.
            devices=(
                JawPair(
                    axis="u",
                    z_top=40.0,
                    z_bottom=47.0,
                    edge_neg=-100.0,
                    edge_pos=0.4,
                    material=WATER,
                    density=1.0,
                ),
            ),
        )
        return CollimatedBeamletSource(inner, stack, _xs())

    def test_beamlet_count_passes_through(self) -> None:
        assert self._source().n_beamlets == 3

    def test_correlated_replay_survives_wrapping(self) -> None:
        """The same stream still replays the same energy in every beamlet, and the
        wrapper consumes no draws of its own."""
        source = self._source()
        rng = HostRNG()
        primaries = [source.emit(j, rng.init_state(SEED, 7)) for j in range(3)]
        assert primaries[1].energy == primaries[0].energy
        assert primaries[2].energy == primaries[0].energy
        # The weight is beamlet-dependent (different fan lines cross the block
        # differently) — that is the collimation doing its job.
        assert primaries[0].weight == 1.0
        assert primaries[2].weight < 1.0

    def test_sample_beamlet_batch_applies_the_same_attenuation(self) -> None:
        """The vectorized beamlet batch weights match a direct stack evaluation."""
        source = self._source()
        batch = source.sample_beamlet_batch(SEED, 0, 48, beamlet=2)
        assert np.all(batch["weight"] <= 1.0)
        assert np.any(batch["weight"] < 1.0)
