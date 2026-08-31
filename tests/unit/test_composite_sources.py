"""Composite (mixture) sources — the virtual-source-model building block.

``CompositeSource`` / ``CompositeBeamletSource`` emit each history from one component,
chosen by weight. These pin the mixing statistics, the shared contract (max_energy, and
for the beamlet form the shared n_beamlets), the validation, and that the default
pre-sampling columns still reproduce emit (so the composites transport on a device
backend with no extra work). Cross-backend transport is exercised in tests/integration
and tests/dij.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.geometry.source import (
    BeamletSource,
    CompositeBeamletSource,
    CompositeSource,
    Primary,
    Source,
)
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED


class _PointSource(Source):
    """A component that tags itself by a fixed x, so the mixture is countable."""

    def __init__(self, x: float, energy: float = 6.0) -> None:
        self._x = x
        self._energy = energy

    @property
    def max_energy(self) -> float:
        return self._energy

    def emit(self, rng_state: object) -> Primary:
        return Primary(self._energy, self._x, 5.0, -1.0, 0.0, 0.0, 1.0)


class _PointBeamletSource(BeamletSource):
    """A beamlet component tagged by a per-beamlet x offset."""

    def __init__(self, x0: float, n: int = 3, energy: float = 6.0) -> None:
        self._x0 = x0
        self._n = n
        self._energy = energy

    @property
    def max_energy(self) -> float:
        return self._energy

    @property
    def n_beamlets(self) -> int:
        return self._n

    def emit(self, beamlet: int, rng_state: object) -> Primary:
        return Primary(self._energy, self._x0 + beamlet, 5.0, -1.0, 0.0, 0.0, 1.0)


class TestCompositeSource:
    def test_mixing_fraction_matches_weights(self) -> None:
        """Over many histories the share from each component tracks its weight."""
        source = CompositeSource([(_PointSource(2.0), 0.75), (_PointSource(8.0), 0.25)])
        batch = source.sample_batch(SEED, 0, 40_000)
        frac_a = float(np.mean(np.isclose(batch["x"], 2.0)))
        assert frac_a == pytest.approx(0.75, abs=0.01)  # ~sqrt(0.75*0.25/4e4) ~ 0.002

    def test_max_energy_is_the_component_maximum(self) -> None:
        source = CompositeSource([(_PointSource(0.0, 6.0), 1.0), (_PointSource(0.0, 18.0), 1.0)])
        assert source.max_energy == 18.0

    def test_default_sample_batch_matches_emit(self) -> None:
        """Selection + delegation is a pure function of the history stream."""
        source = CompositeSource([(_PointSource(2.0), 0.5), (_PointSource(8.0), 0.5)])
        rng = HostRNG()
        batch = source.sample_batch(SEED, 0, 100)
        for i in range(100):
            assert batch["x"][i] == pytest.approx(source.emit(rng.init_state(SEED, i)).x)

    def test_weights_need_not_be_normalized(self) -> None:
        source = CompositeSource([(_PointSource(2.0), 3.0), (_PointSource(8.0), 1.0)])
        frac_a = float(np.mean(np.isclose(source.sample_batch(SEED, 0, 40_000)["x"], 2.0)))
        assert frac_a == pytest.approx(0.75, abs=0.01)

    @pytest.mark.parametrize("bad", [[], [(_PointSource(0.0), -1.0)], [(_PointSource(0.0), 0.0)]])
    def test_invalid_components_are_rejected(self, bad: list) -> None:
        with pytest.raises(ValueError):
            CompositeSource(bad)


class TestCompositeBeamletSource:
    def test_shared_beamlet_count(self) -> None:
        source = CompositeBeamletSource(
            [(_PointBeamletSource(0.0, n=4), 0.5), (_PointBeamletSource(1.0, n=4), 0.5)]
        )
        assert source.n_beamlets == 4

    def test_mismatched_beamlet_counts_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_beamlets"):
            CompositeBeamletSource(
                [(_PointBeamletSource(0.0, n=4), 1.0), (_PointBeamletSource(0.0, n=3), 1.0)]
            )

    def test_mixture_is_per_beamlet(self) -> None:
        """For a fixed beamlet the component share tracks the weights."""
        source = CompositeBeamletSource(
            [(_PointBeamletSource(0.0), 0.8), (_PointBeamletSource(100.0), 0.2)]
        )
        batch = source.sample_beamlet_batch(SEED, 0, 40_000, beamlet=2)
        # component A emits x = 0 + 2 = 2; component B emits x = 100 + 2 = 102.
        frac_a = float(np.mean(np.isclose(batch["x"], 2.0)))
        assert frac_a == pytest.approx(0.8, abs=0.01)
        assert np.all(np.isclose(batch["x"], 2.0) | np.isclose(batch["x"], 102.0))
