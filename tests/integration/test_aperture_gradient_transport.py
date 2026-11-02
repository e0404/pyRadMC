"""Aperture gradients through transport: the exact estimator against the CRN oracle.

The unit tier pins the source-side identity on a known response; here the response
is the engine's dose. The common-random-number central difference
(:class:`RayFiniteDifferenceSource`) is the oracle: unbiased for the finite
difference at its step, which sits inside one voxel column so its O(h^2) offset
from the derivative is far below the noise. The exact estimator
(:class:`EdgeDerivativeSource`) must agree with it voxel by voxel (batched
chi-squared, AGENTS.md 4) for both a rounded leaf tip (a smooth penumbra) and a
straight jaw edge (a pure boundary term). KERMA mode keeps the tier fast; the
identity is transport-independent by construction.

The beam is narrowed to the 1 cm around the edges. The identity holds for any
field covering the edge region, and the oracle spends histories only on rays
within +-h of the edge: over the unit tier's 9 cm field that is under 1 % of
them, too few interactions per voxel for an 8-batch sigma to be Gaussian.
"""

from __future__ import annotations

import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.geometry.aperture_gradient import (
    BixelChainRuleSource,
    EdgeDerivativeSource,
    EdgeParameter,
    RayFiniteDifferenceSource,
)
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.unit.test_aperture_gradient import ENERGY, JAW_POS, MLC_POS, XS, _stack

N_BATCHES = 8


def _beam() -> ParallelBeamSource:
    """Covers the pair-0 tip penumbra (5.25 to 5.83) and the jaw edge (5.6)."""
    return ParallelBeamSource(ENERGY, -1.0, (5.0, 6.0), (0.0, 4.0))


def _engine() -> ReferenceEngine:
    # 0.5 cm columns across the edges: the MLC tip (5.25) and the jaw edge (5.6)
    # each sit inside one column, as does each oracle's +-h window; the two
    # 2 cm rows are the two leaf strips.
    grid = VoxelGrid.uniform_water(shape=(20, 2, 8), spacing=(0.5, 2.0, 1.0))
    return ReferenceEngine(grid=grid, cross_sections=XS, rng=HostRNG())


@pytest.mark.parametrize(
    ("parameter", "step", "n_oracle"),
    [(MLC_POS, 0.02, 16_000), (JAW_POS, 0.04, 24_000)],
    ids=["rounded-tip", "straight-edge"],
)
def test_edge_estimator_matches_the_finite_difference_dose(
    parameter: EdgeParameter, step: float, n_oracle: int
) -> None:
    engine = _engine()
    kwargs = dict(n_batches=N_BATCHES, transport_electrons=False)
    exact = engine.run(
        EdgeDerivativeSource(_beam(), _stack(), XS, parameter),
        n_histories=4_000,
        seed=SEED,
        **kwargs,
    )
    oracle = engine.run(
        RayFiniteDifferenceSource(_beam(), _stack(), XS, parameter, step=step),
        n_histories=n_oracle,
        seed=SEED + 1,
        **kwargs,
    )
    mask = exact.dose > 0.1 * exact.dose.max()
    mask &= (exact.dose_sigma > 0.0) & (oracle.dose_sigma > 0.0)
    assert mask.sum() >= 12
    assert_chi2_consistent_batched(
        exact.dose,
        exact.dose_sigma,
        N_BATCHES,
        oracle.dose,
        oracle.dose_sigma,
        N_BATCHES,
        mask=mask,
    )


def test_oracle_rejects_the_bixel_chain_rule() -> None:
    """The oracle has power: DAO's bixel-spread gradient is distinguishable.

    The chain rule through a 5 mm bixel Dij carries the same derivative measure
    as the exact estimator but spreads it over the bixel [5.0, 5.5) instead of the
    tip's penumbra [5.25, 5.83]; the CRN oracle must tell the two apart, or a
    passing comparison above would say nothing.
    """
    engine = _engine()
    kwargs = dict(n_batches=N_BATCHES, transport_electrons=False)
    chain_rule = engine.run(
        BixelChainRuleSource(_beam(), _stack(), XS, MLC_POS, bixel_pitch=0.5),
        n_histories=4_000,
        seed=SEED,
        **kwargs,
    )
    oracle = engine.run(
        RayFiniteDifferenceSource(_beam(), _stack(), XS, MLC_POS, step=0.02),
        n_histories=16_000,
        seed=SEED + 1,
        **kwargs,
    )
    mask = chain_rule.dose > 0.1 * chain_rule.dose.max()
    mask &= (chain_rule.dose_sigma > 0.0) & (oracle.dose_sigma > 0.0)
    with pytest.raises(AssertionError):
        assert_chi2_consistent_batched(
            chain_rule.dose,
            chain_rule.dose_sigma,
            N_BATCHES,
            oracle.dose,
            oracle.dose_sigma,
            N_BATCHES,
            mask=mask,
        )
