"""The fan-beam edge estimator through transport, against the CRN oracle.

As for the parallel beam (``test_aperture_gradient_transport.py``): the
common-random-number central difference of the forward fan, one reweighted run,
is the oracle; the estimator must agree with it voxel by voxel (batched
chi-squared) for a rounded leaf tip with focused sides (smooth derivative) and a
focused jaw face (a pure jump on the edge line). The fan aperture is narrowed to
the edge region so the oracle's histories are not spent on rays the edge never
touches; the identity holds for any field covering it. KERMA mode keeps the tier
fast; the identity is transport-independent by construction.
"""

from __future__ import annotations

import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.aperture_gradient import (
    EdgeParameter,
    FanEdgeDerivativeSource,
    RayFiniteDifferenceSource,
)
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

XS = AnalyticCrossSections()
FOCAL = (5.0, 2.0, -100.0)  # engine frame; the phantom surface is the reference plane
N_BATCHES = 8


def _beam() -> SpectralBeamSource:
    """A 2 MeV +- 10 % fan through the 1 cm x 4 cm of surface around both edges."""
    return SpectralBeamSource(
        Spectrum(edges=[1.8, 2.2], weights=[1.0]),
        focal_point=FOCAL,
        center=(5.5, 2.0, 0.0),
        width_u=1.0,
        width_v=4.0,
    )


def _stack(focused_jaw: bool, tip: float, jaw_edge: float) -> BeamLimitingStack:
    """Jaw over a focused-sides MLC; strips map to the two 2 cm voxel rows."""
    return BeamLimitingStack(
        frame=BeamFrame(origin=FOCAL),
        devices=(
            JawPair(
                axis="u",
                z_top=30.0,
                z_bottom=37.8,
                edge_neg=-5.0,
                edge_pos=jaw_edge,
                material=WATER,
                density=5.0,
                focused=focused_jaw,
            ),
            MLC(
                z_top=50.0,
                z_bottom=56.0,
                leaf_edges_v=(-1.06, 0.0, 1.06),
                tips_neg=(-5.0, -5.0),
                tips_pos=(tip, 8.0),
                tip_radius=8.0,
                material=WATER,
                density=5.0,
                focused_sides=True,
            ),
        ),
    )


def _engine() -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(20, 2, 8), spacing=(0.5, 2.0, 1.0))
    return ReferenceEngine(grid=grid, cross_sections=XS, rng=HostRNG())


@pytest.mark.parametrize(
    ("parameter", "focused_jaw", "tip", "jaw_edge", "step"),
    [
        # Tip apex at 0.27 cm off axis at the MLC mid-plane (0.51 at the surface).
        (EdgeParameter(1, "pos", 0), False, 0.27, 5.0, 0.004),
        # Focused jaw edge 0.18 at its mid-plane (0.53 at the surface), MLC open.
        (EdgeParameter(0, "pos"), True, 8.0, 0.18, 0.004),
    ],
    ids=["rounded-tip", "focused-jaw"],
)
def test_fan_estimator_matches_the_finite_difference_dose(
    parameter: EdgeParameter, focused_jaw: bool, tip: float, jaw_edge: float, step: float
) -> None:
    engine = _engine()
    stack = _stack(focused_jaw, tip, jaw_edge)
    kwargs = dict(n_batches=N_BATCHES, transport_electrons=False)
    exact = engine.run(
        FanEdgeDerivativeSource(_beam(), stack, XS, parameter),
        n_histories=4_000,
        seed=SEED,
        **kwargs,
    )
    oracle = engine.run(
        RayFiniteDifferenceSource(_beam(), stack, XS, parameter, step=step),
        n_histories=16_000,
        seed=SEED + 1,
        **kwargs,
    )
    mask = exact.dose > 0.1 * exact.dose.max()
    mask &= (exact.dose_sigma > 0.0) & (oracle.dose_sigma > 0.0)
    assert mask.sum() >= 8
    assert_chi2_consistent_batched(
        exact.dose,
        exact.dose_sigma,
        N_BATCHES,
        oracle.dose,
        oracle.dose_sigma,
        N_BATCHES,
        mask=mask,
    )
