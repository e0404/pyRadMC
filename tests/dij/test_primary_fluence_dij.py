"""A PrimaryFluenceBeamletSource assembles a Dij whose columns carry psi(r).

The beamlet variant's claim is that the measured primary fluence becomes a
*per-column* property: each bixel sits at its own off-axis radius, so its Dij
column is scaled by the fluence there, and a bixel outside the primary
collimator's field contributes an empty column. Both are pinned here, along with
the weighted energy ledger — the Dij books ``weight x energy`` emitted, and a
source whose weights are not 1 is exactly what would expose a book that ignores
them.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.fluence import RadialFluence
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import GaussianSpotBeamletSource, PrimaryFluenceBeamletSource
from pyRadMC.geometry.spectrum import Spectrum
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -80.0)
PLANE_Z = -8.0  # emission plane 72 cm from the spot; the reference distance is 80
REFERENCE_DISTANCE = 80.0
# Bixels on axis, 1.8 cm off axis, and 3.6 cm off axis on the plane. Projected to
# the 80 cm reference distance those are radii 0, 2.0 and 4.0 cm, so the table
# below gives them psi = 1.0, 0.5 and 0.0: one full column, one halved, one empty.
CENTERS = ((8.0, 8.0, PLANE_Z), (9.8, 8.0, PLANE_Z), (11.6, 8.0, PLANE_Z))
# The table is **plateaued** around each bixel's radius rather than sloping through
# it. A bixel is an area, so it samples a range of radii (+-0.31 cm here once
# projected), and on a sloping table its mean weight is not the weight at its
# centre — an on-axis bixel in particular can only ever average *below* psi(0),
# because the radius is one-sided there. Plateaus make each bixel's weight one
# exact number, which is what lets the ratios below be read as psi.
FLUENCE = RadialFluence(
    radii=(0.0, 1.0, 1.5, 2.5, 3.0, 3.5),
    values=(1.0, 1.0, 0.5, 0.5, 0.0, 0.0),
    reference_distance=REFERENCE_DISTANCE,
)
PSI = (1.0, 0.5, 0.0)


_GEOMETRY: dict[str, object] = dict(
    spectrum=SPECTRUM,
    focal_point=FOCAL,
    centers=CENTERS,
    width_u=0.4,  # narrow, so every history in a bixel sees nearly one psi
    width_v=0.4,
    sigma_u=0.05,
    sigma_v=0.05,
)


def _source() -> PrimaryFluenceBeamletSource:
    return PrimaryFluenceBeamletSource(fluence=FLUENCE, **_GEOMETRY)  # type: ignore[arg-type]


def _unweighted_source() -> GaussianSpotBeamletSource:
    """The exact geometric twin with unit weights — the control for the ratio test.

    Same six-uniform stream and bit-identical emission (pinned in
    ``tests/unit/test_primary_fluence_source.py``), so the two Dij runs transport
    *the same histories* and the per-column ratio is the fluence weight alone,
    with almost no residual statistical noise.
    """
    return GaussianSpotBeamletSource(**_GEOMETRY)  # type: ignore[arg-type]


def _engine() -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    return ReferenceEngine(grid=grid, cross_sections=AnalyticCrossSections(), rng=HostRNG())


def test_beamlet_weights_scale_the_columns() -> None:
    """Column totals track psi at each bixel, against an unweighted control.

    Ratios, not absolutes: the control shares the geometry and the seed, so the
    per-column ratio isolates the fluence weight from everything else.

    The tolerance is statistical rather than exact **on purpose**. The weight-1.0
    and weight-0.0 columns do come out exact (identical and empty respectively),
    but a weight of 0.5 changes how many boosts the soft-photon roulette can hand
    out before it hits ``PHOTON_ROULETTE_WEIGHT_CAP``, so those histories diverge
    after their first roulette game. Roulette is unbiased, so the ratio is right in
    expectation; it is not bit-exact, and tightening this to an equality would be
    pinning an accident.
    """
    kwargs = dict(n_histories_per_beamlet=400, n_batches=2, seed=SEED, transport_electrons=False)
    weighted = _engine().run_dij(_source(), **kwargs)
    control = _engine().run_dij(_unweighted_source(), **kwargs)

    for beamlet, psi in enumerate(PSI):
        reference = float(control.column_dense(beamlet).sum())
        assert reference > 0.0, f"control column {beamlet} is empty; the test cannot resolve psi"
        ratio = float(weighted.column_dense(beamlet).sum()) / reference
        assert ratio == pytest.approx(psi, abs=0.01), f"beamlet {beamlet}"


def test_a_bixel_outside_the_table_has_an_empty_column() -> None:
    """psi = 0 must produce no dose at all, not merely little — an exact zero."""
    dij = _engine().run_dij(
        _source(),
        n_histories_per_beamlet=300,
        n_batches=2,
        seed=SEED,
        transport_electrons=False,
    )
    assert dij.n_beamlets == 3
    assert np.all(dij.column_dense(2) == 0.0)
    assert dij.column_dense(0).sum() > 0.0


def test_weighted_dij_ledger_closes() -> None:
    """The emitted book must sum weight x energy; a weight-blind book would over-count."""
    dij = _engine().run_dij(_source(), n_histories_per_beamlet=200, n_batches=2, seed=SEED)
    assert dij.energy_emitted == pytest.approx(dij.energy_deposited + dij.energy_escaped, rel=1e-12)
    # Only bixels 0 and 1 emit, with weight 1.0 and 0.5 over 200 histories each, so
    # the book sits well below the 600-history unweighted bound.
    assert dij.energy_emitted < 600 * SPECTRUM.mean_energy
