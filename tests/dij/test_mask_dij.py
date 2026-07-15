"""A transmission-mask beamlet source scales its Dij columns by the mask value.

End-to-end pin for configuration 2 (aperture-optimization shapes): a bixel behind
a half-transmission mask region delivers half the dose of its mirror-image open
bixel. The device routes share the wrapper machinery the collimated Dij already
pins cross-backend (slice 5), so this runs on the reference engine only.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.collimation import TransmissionMaskBeamletSource
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import SpectralBeamletSource
from pyRadMC.geometry.spectrum import Spectrum
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED


def _source() -> TransmissionMaskBeamletSource:
    inner = SpectralBeamletSource(
        spectrum=Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0)),
        focal_point=(8.0, 8.0, -100.0),
        centers=((6.0, 8.0, 0.0), (10.0, 8.0, 0.0)),
        width_u=2.0,
        width_v=2.0,
    )
    # x < 8 fully open, x > 8 at half transmission; the plateau edges stay half
    # a pixel away from both bixels' rays (see the unit-tier mask tests).
    mask = np.array([[1.0]] * 4 + [[0.5]] * 4)
    return TransmissionMaskBeamletSource(
        inner, mask=mask, plane_center=(8.0, 8.0, 0.0), width_u=8.0, width_v=8.0
    )


def test_masked_column_is_half_the_open_mirror_column() -> None:
    """The mirror-symmetric bixel pair differs exactly by the mask factor.

    Correlated sampling replays the same energies and in-bixel offsets in both
    columns; the geometry is mirror-symmetric about x = 8, so after mirroring,
    the half-masked column is the open column scaled by exactly 0.5 up to the
    (deterministic, seed-fixed) transport noise of distinct spatial streams.
    The peak-ratio gate is statistical, centred on 0.5.
    """
    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    engine = ReferenceEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )
    dij = engine.run_dij(
        _source(),
        n_histories_per_beamlet=2_000,
        n_batches=4,
        seed=SEED,
        transport_electrons=False,
    )
    assert dij.energy_emitted == pytest.approx(dij.energy_deposited + dij.energy_escaped, rel=1e-9)
    open_peak = float(dij.column_dense(0).max())
    masked_peak = float(dij.column_dense(1).max())
    assert masked_peak / open_peak == pytest.approx(0.5, rel=0.10)
