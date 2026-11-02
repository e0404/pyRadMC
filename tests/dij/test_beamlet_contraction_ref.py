"""Beamlet contraction on the reference engine: J^T a without assembling J.

An optimizer needs ``g_j = sum_v a_v D_vj`` (``a = dL/dD``), one number per
column, not the Dij itself. ``run_beamlet_contraction`` scores that sum per batch
directly. With the same seed and history mapping it transports exactly the
histories ``run_dij`` transports, so its mean must equal the contracted Dij columns
to float64 summation order (truncation off), its energy books must match, and its
per-batch sigma must be calibrated.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

KWARGS = dict(n_histories_per_beamlet=400, n_batches=4, transport_electrons=False)


def _engine() -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(8, 8, 6), spacing=(1.0, 1.0, 1.0))
    return ReferenceEngine(grid=grid, cross_sections=AnalyticCrossSections(), rng=HostRNG())


def _source() -> BeamletGridSource:
    return BeamletGridSource(2.0, -1.0, (2.0, 6.0), (2.0, 6.0), 2, 2)


def _weights(shape: tuple[int, int, int]) -> np.ndarray:
    """A signed, smooth voxel weight map, like dL/dD of a quadratic objective."""
    x, y, z = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return np.sin(0.7 * x) + 0.3 * y - 0.1 * z


@pytest.mark.parametrize("correlated", [True, False])
def test_contraction_equals_the_contracted_dij(correlated: bool) -> None:
    engine = _engine()
    a = _weights(engine.grid.shape)
    dij = engine.run_dij(_source(), seed=SEED, truncation=0.0, correlated=correlated, **KWARGS)
    result = engine.run_beamlet_contraction(
        _source(), a, seed=SEED, correlated=correlated, **KWARGS
    )
    expected = np.array([np.sum(a * dij.column_dense(j)) for j in range(dij.n_beamlets)])
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-15)
    assert result.energy_emitted == pytest.approx(dij.energy_emitted, rel=1e-12)
    assert result.energy_deposited == pytest.approx(dij.energy_deposited, rel=1e-12)
    assert result.energy_escaped == pytest.approx(dij.energy_escaped, rel=1e-12)
    assert result.n_histories_per_beamlet == KWARGS["n_histories_per_beamlet"]


def test_contraction_sigma_is_calibrated_across_seeds() -> None:
    """Independent seeds differ by about their quoted sigmas (pooled over beamlets)."""
    engine = _engine()
    a = _weights(engine.grid.shape)
    runs = [
        engine.run_beamlet_contraction(_source(), a, seed=SEED + k, correlated=False, **KWARGS)
        for k in range(6)
    ]
    values = np.array([r.values for r in runs])
    sigmas = np.array([r.sigma for r in runs])
    assert np.all(sigmas > 0.0)
    spread = values.std(axis=0, ddof=1)
    ratio = spread / sigmas.mean(axis=0)
    # 4 batches give a crude sigma; the pooled ratio must still be order one.
    assert 0.5 < float(np.median(ratio)) < 2.0


def test_contraction_refuses_a_weight_map_of_the_wrong_shape() -> None:
    engine = _engine()
    with pytest.raises(ValueError, match="shape"):
        engine.run_beamlet_contraction(_source(), np.zeros((2, 2, 2)), seed=SEED, **KWARGS)
