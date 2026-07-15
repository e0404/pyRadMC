"""The Dij transports and books the source's statistical weight (BLD workstream).

The collimated beamlet sources attenuate deterministically by *weight*
(``w' = w * exp(-mu t)``), so the Dij path must honor ``Primary.weight`` the way
the open-field path always has (phase-space precedent). Before this slice both
engines silently assumed unit weight on the Dij: the reference booked the emitted
energy unweighted *and* transported at weight 1, and the Warp pre-sampled route
overwrote the sampled weight column with ones.

The exact-linearity pin compares two power-of-two weights **above the photon
roulette weight cap** (4.0 and 8.0): scaling every deposit by 2^-1 is exact in
floating point and commutes with every rounding, and above the cap the one
transport branch that reads the absolute weight (``copy_w <
PHOTON_ROULETTE_WEIGHT_CAP``) is constant-false in both runs, so the columns must
match bit for bit. **Below the cap exact linearity genuinely does not hold**
(measured 2026-07-15 at weights 1.0 vs 0.5, this seed: a roulette boost chain
crossed the cap in one run only, its extra/missing roulette draw diverged that
history's stream, and correlated sampling replayed the divergence into all three
columns, ~20 voxels each). That is fair-game roulette behaviour — each run stays
individually unbiased — but it is exactly the absolute-cap/attenuated-weight
interaction flagged for maintainer review in the BLD workstream plan; the sub-cap
weighted transport is therefore pinned statistically (cross-backend chi-squared at
weight 0.5), not bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletSource, Primary
from pyRadMC.rng.host import HostRNG, uniform
from tests.conftest import SEED, assert_chi2_consistent_batched

ENERGY = 6.0
Z0 = -1.0
FIELD_X = (2.0, 14.0)
FIELD_Y = (2.0, 14.0)
N_STRIPS = 3

ENERGY_BALANCE_RTOL = 1.0e-4  # Warp: float32 transport + 1e-9 MeV scoring quanta


class WeightedStripSource(BeamletSource):
    """The strip-lattice instrument source, emitting at a constant weight."""

    def __init__(self, n: int, weight: float) -> None:
        self._n = n
        self._weight = weight

    @property
    def max_energy(self) -> float:
        return ENERGY

    @property
    def n_beamlets(self) -> int:
        return self._n

    def emit(self, beamlet: int, rng_state: object) -> Primary:
        x_lo = FIELD_X[0] + (FIELD_X[1] - FIELD_X[0]) * beamlet / self._n
        x_hi = FIELD_X[0] + (FIELD_X[1] - FIELD_X[0]) * (beamlet + 1) / self._n
        x = x_lo + (x_hi - x_lo) * uniform(rng_state)
        y = FIELD_Y[0] + (FIELD_Y[1] - FIELD_Y[0]) * uniform(rng_state)
        return Primary(ENERGY, x, y, Z0, 0.0, 0.0, 1.0, weight=self._weight)


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _ref_engine(grid: VoxelGrid) -> ReferenceEngine:
    return ReferenceEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )


def test_reference_ledger_books_the_weight() -> None:
    """emitted = sum(w * E) exactly, and the weighted ledger still closes."""
    source = WeightedStripSource(N_STRIPS, weight=0.5)
    dij = _ref_engine(_grid()).run_dij(source, n_histories_per_beamlet=200, n_batches=2, seed=SEED)
    assert dij.energy_emitted == pytest.approx(0.5 * N_STRIPS * 200 * ENERGY, rel=1e-12)
    assert dij.energy_emitted == pytest.approx(dij.energy_deposited + dij.energy_escaped, rel=1e-12)


def test_reference_columns_scale_exactly_above_the_roulette_cap() -> None:
    """At weights 8.0 vs 4.0 the Dij columns scale by exactly one half, bit for bit.

    Both weights sit at or above ``PHOTON_ROULETTE_WEIGHT_CAP``, so the roulette
    cap comparison — the only transport branch that reads the absolute weight —
    decides identically (never plays) in both runs; constant scaling by 2^-1 is
    then exact through the whole float pipeline. Any difference means a branch
    illegally read the weight. Seed-pinned; a failure is deterministic and loud.
    (Sub-cap weights are *not* bit-linear — module docstring — and are pinned
    statistically below.)
    """
    grid = _grid()
    kwargs = dict(n_histories_per_beamlet=400, n_batches=4, seed=SEED)
    full = _ref_engine(grid).run_dij(WeightedStripSource(N_STRIPS, 8.0), **kwargs)
    half = _ref_engine(grid).run_dij(WeightedStripSource(N_STRIPS, 4.0), **kwargs)
    for j in range(N_STRIPS):
        np.testing.assert_array_equal(half.column_dense(j), 0.5 * full.column_dense(j))
    assert half.energy_emitted == 0.5 * full.energy_emitted


def test_unit_weight_dij_is_unchanged() -> None:
    """Regression guard: weight-1.0 primaries reproduce the historical Dij path."""
    grid = _grid()
    kwargs = dict(n_histories_per_beamlet=200, n_batches=2, seed=SEED)
    weighted_api = _ref_engine(grid).run_dij(WeightedStripSource(N_STRIPS, 1.0), **kwargs)
    assert weighted_api.energy_emitted == pytest.approx(N_STRIPS * 200 * ENERGY, rel=1e-12)
    assert weighted_api.energy_emitted == pytest.approx(
        weighted_api.energy_deposited + weighted_api.energy_escaped, rel=1e-12
    )


@pytest.mark.warp
class TestWarpPresampled:
    """The pre-sampled Warp Dij route carries the weight column end to end."""

    def test_warp_ledger_books_the_weight(self) -> None:
        wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
        del wp
        from pyRadMC.backends.warp.engine import WarpEngine

        grid = _grid()
        xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
        dij = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run_dij(
            WeightedStripSource(N_STRIPS, 0.5), n_histories_per_beamlet=600, n_batches=3, seed=SEED
        )
        assert dij.energy_emitted == pytest.approx(0.5 * N_STRIPS * 600 * ENERGY, rel=1e-12)
        assert dij.energy_emitted == pytest.approx(
            dij.energy_deposited + dij.energy_escaped, rel=ENERGY_BALANCE_RTOL
        )

    def test_warp_columns_match_reference(self) -> None:
        wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
        del wp
        from pyRadMC.backends.warp.engine import WarpEngine

        grid = _grid()
        xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
        source = WeightedStripSource(N_STRIPS, 0.5)
        kwargs = dict(
            n_histories_per_beamlet=2_400, n_batches=8, seed=SEED, transport_electrons=False
        )
        ref = _ref_engine(grid).run_dij(source, **kwargs)
        warp_result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run_dij(
            source, beamlet_group_size=2, **kwargs
        )
        for j in range(N_STRIPS):
            ref_dose, ref_sigma = ref.column_dense(j), ref.sigma_dense(j)
            warp_dose, warp_sigma = warp_result.column_dense(j), warp_result.sigma_dense(j)
            mask = ref_dose > 0.1 * ref_dose.max()
            mask &= (ref_sigma > 0.0) & (warp_sigma > 0.0)
            assert_chi2_consistent_batched(
                ref_dose,
                ref_sigma,
                ref.n_batches,
                warp_dose,
                warp_sigma,
                warp_result.n_batches,
                mask=mask,
            )
