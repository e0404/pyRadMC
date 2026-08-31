"""Non-statistical engine invariants: energy bookkeeping and reproducibility.

These are exact properties. If one fails, it is a bug, never noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED


def _run(seed: int, n_histories: int = 400):
    from pyradmc.backends.ref.engine import ReferenceEngine

    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    # 6 MeV: pair production is a real channel, so annihilation photons are exercised.
    source = PencilBeamSource(energy=6.0, position=(8.0, 8.0, -1.0), direction=(0.0, 0.0, 1.0))
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    return engine.run(source, n_histories=n_histories, n_batches=4, seed=seed)


def test_energy_is_conserved_exactly() -> None:
    """emitted = deposited + escaped, to float accumulation precision.

    KERMA-mode photon transport creates no energy anywhere: every MeV emitted either
    ends in a voxel or leaves the grid. This is bookkeeping, not physics, so the
    tolerance is 1e-9 relative, not statistical.
    """
    result = _run(SEED)
    assert result.energy_emitted == pytest.approx(400 * 6.0)
    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=1e-9)
    assert result.energy_deposited > 0.0
    assert result.energy_escaped > 0.0


def test_same_seed_is_bit_identical_different_seed_is_not() -> None:
    """Within one target, one seed: bit-reproducible (AGENTS.md section 2.3).

    Bit-equality is asserted *within* the ref target only. Never write this test
    across targets.
    """
    a = _run(SEED, n_histories=200)
    b = _run(SEED, n_histories=200)
    assert np.array_equal(a.dose, b.dose)
    assert np.array_equal(a.dose_sigma, b.dose_sigma)
    assert a.energy_deposited == b.energy_deposited

    c = _run(SEED + 1, n_histories=200)
    assert not np.array_equal(a.dose, c.dose)


def test_history_count_must_divide_into_batches() -> None:
    """Unequal batches would weight batch means inconsistently; refuse them."""
    with pytest.raises(ValueError, match="divisible"):
        _run(SEED, n_histories=401)
