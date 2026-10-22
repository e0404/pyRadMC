"""A subclass of a built-in beam keeps the host pre-sampling route on Warp.

The Warp engine generates :class:`PencilBeamSource` and :class:`ParallelBeamSource`
primaries in-kernel. A subclass may override ``emit``/``sample_batch`` (reweight,
reposition), and the built-in generator would silently bypass the override, so the
in-kernel route must be chosen by exact type, as it already is for
:class:`SpectralBeamSource` (``test_spectral_source_inkernel_warp.py``). Found when a
reweighting ``ParallelBeamSource`` subclass transported as the open field.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource, PencilBeamSource
from tests.conftest import SEED

pytestmark = pytest.mark.warp

pytest.importorskip("warp", reason="warp-lang optional dependency not installed")


class HalfWeightParallel(ParallelBeamSource):
    """Every primary at weight 0.5: the dose must halve if the override is honoured."""

    def sample_batch(self, seed: int, history_offset: int, n: int):
        columns = super().sample_batch(seed, history_offset, n)
        columns["weight"] = np.full(n, 0.5, dtype=np.float32)
        return columns


class HalfWeightPencil(PencilBeamSource):
    """The pencil-beam twin of :class:`HalfWeightParallel`."""

    def sample_batch(self, seed: int, history_offset: int, n: int):
        columns = super().sample_batch(seed, history_offset, n)
        columns["weight"] = np.full(n, 0.5, dtype=np.float32)
        return columns


@pytest.mark.parametrize(
    "source",
    [
        HalfWeightParallel(2.0, -1.0, (2.0, 6.0), (2.0, 6.0)),
        HalfWeightPencil(2.0, (4.0, 4.0, -1.0), (0.0, 0.0, 1.0)),
    ],
    ids=["parallel", "pencil"],
)
def test_beam_subclass_override_is_honoured(source) -> None:
    from pyradmc.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    engine = WarpEngine(grid=grid, cross_sections=AnalyticCrossSections(), device="cpu")
    result = engine.run(source, n_histories=2_000, n_batches=2, seed=SEED)
    # The emitted-energy book is exactly weight x energy x histories.
    assert result.energy_emitted == pytest.approx(0.5 * 2.0 * 2_000, rel=1e-6)


def test_beamlet_grid_subclass_override_is_honoured() -> None:
    """The Dij twin: a BeamletGridSource subclass must keep its own batch sampler."""
    from pyradmc.backends.warp.engine import WarpEngine
    from pyradmc.geometry.source import BeamletGridSource

    class HalfWeightGrid(BeamletGridSource):
        def sample_beamlet_batch(self, seed: int, history_offset: int, n: int, beamlet: int):
            columns = super().sample_beamlet_batch(seed, history_offset, n, beamlet)
            columns["weight"] = np.full(n, 0.5, dtype=np.float32)
            return columns

    source = HalfWeightGrid(2.0, -1.0, (2.0, 6.0), (2.0, 6.0), 2, 2)
    grid = VoxelGrid.uniform_water(shape=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    engine = WarpEngine(grid=grid, cross_sections=AnalyticCrossSections(), device="cpu")
    dij = engine.run_dij(source, n_histories_per_beamlet=500, n_batches=2, seed=SEED)
    assert dij.energy_emitted == pytest.approx(0.5 * 2.0 * 500 * 4, rel=1e-6)
