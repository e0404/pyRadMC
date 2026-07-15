"""A divergent polyenergetic SpectralBeamletSource drives the reference Dij (Phase 5).

Transport-level pins for the spectral source: the energy ledger closes exactly for a
polyenergetic beam (emitted energy is whatever the spectrum sampled, booked per
history), and the divergent fan geometry localizes each beamlet's dose under the
line from the focal spot through its own bixel.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import SpectralBeamletSource
from pyRadMC.geometry.spectrum import Spectrum
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -80.0)
CENTERS = ((4.0, 8.0, 0.0), (8.0, 8.0, 0.0), (12.0, 8.0, 0.0))


def _source() -> SpectralBeamletSource:
    return SpectralBeamletSource(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        centers=CENTERS,
        width_u=3.0,
        width_v=3.0,
    )


def _engine() -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    return ReferenceEngine(grid=grid, cross_sections=AnalyticCrossSections(), rng=HostRNG())


def test_spectral_dij_ledger_closes_exactly() -> None:
    """emitted == deposited + escaped to float precision, polyenergetic emission."""
    dij = _engine().run_dij(_source(), n_histories_per_beamlet=200, n_batches=2, seed=SEED)
    assert dij.n_beamlets == 3
    assert dij.energy_emitted == pytest.approx(dij.energy_deposited + dij.energy_escaped, rel=1e-12)
    # Polyenergetic: the total is bounded by the spectrum support, not equal to n*E.
    n = 3 * 200
    assert n * 0.5 < dij.energy_emitted < n * 6.0


def test_spectral_beamlet_columns_follow_their_fan_lines() -> None:
    """Each beamlet's dose x-centroid tracks its own focal-to-bixel line, in order."""
    dij = _engine().run_dij(
        _source(), n_histories_per_beamlet=600, n_batches=3, seed=SEED, transport_electrons=False
    )
    centroids = []
    for beamlet in range(3):
        by_x = dij.column_dense(beamlet).sum(axis=(1, 2))
        centroids.append(float(np.sum(np.arange(16) * by_x) / by_x.sum()))
    assert centroids[0] < centroids[1] < centroids[2]
    # The middle bixel is on the central axis: its centroid sits mid-grid.
    assert centroids[1] == pytest.approx(7.5, abs=1.0)
