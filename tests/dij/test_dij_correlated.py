"""Correlated sampling across beamlets (Phase 4): the stream-mapping pin.

Correlated sampling is a change of the *stream mapping* only (AGENTS.md 7.2):
with ``correlated=True``, the history at within-beamlet index ``rw`` of beamlet
``j`` draws the stream keyed on ``rw`` instead of the global index
``h = j * n_per + rw``, so corresponding histories of every beamlet replay the
same random sequence and only the entry position differs. The physics, the
scoring, and the batching are untouched.

The mapping pinned here on the reference engine is the specification every
backend follows, exactly like the Phase 3 ``h`` mapping fixed in
``ReferenceEngine.run_dij``.

The ``correlated`` flag is the Phase 4 experiment instrument (AGENTS.md 2.10);
the shipped default is decided from the noise/bias study at phase exit. A
correlated Dij's columns are *not* statistically independent: per-column sigmas
stay valid, but sigmas must never be combined across columns in quadrature.
The result carries ``correlated`` so downstream code can tell.
"""

from __future__ import annotations

import numpy as np

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletGridSource, ParallelBeamSource
from pyRadMC.rng.host import HostRNG
from pyRadMC.rng.interface import RNG, RNGState
from tests.conftest import SEED, assert_chi2_consistent_batched

FIELD_X = (2.0, 14.0)
FIELD_Y = (2.0, 14.0)
ENERGY = 6.0
Z0 = -1.0


class RecordingRNG(RNG):
    """HostRNG wrapper that records the ``history_index`` key of every stream.

    A test instrument: the stream mapping is engine behavior, and the only
    observable that pins it exactly is the sequence of keys the engine asks
    the RNG interface for.
    """

    def __init__(self) -> None:
        self._inner = HostRNG()
        self.keys: list[int] = []

    def init_state(self, seed: int, history_index: int) -> RNGState:
        self.keys.append(history_index)
        return self._inner.init_state(seed, history_index)

    def uniform(self, state: RNGState) -> float:
        return self._inner.uniform(state)


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(2.0, 2.0, 0.5))


def _engine(rng: RNG) -> ReferenceEngine:
    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return ReferenceEngine(grid=grid, cross_sections=xs, rng=rng)


def _lattice(n_x: int, n_y: int) -> BeamletGridSource:
    return BeamletGridSource(
        energy=ENERGY, z=Z0, x_range=FIELD_X, y_range=FIELD_Y, n_x=n_x, n_y=n_y
    )


class TestStreamMapping:
    """The (seed, key) a history draws is the whole definition of the mode."""

    def test_correlated_streams_key_on_the_within_beamlet_index(self) -> None:
        """Every beamlet replays the keys 0..n_per-1: the repetition pin.

        Loop order (batch-major, then beamlet, then within-batch index) is
        fixed by ``run_dij``; with n_per=4 in 2 batches over 2 beamlets the
        correlated keys repeat identically for each beamlet within a batch.
        """
        rng = RecordingRNG()
        dij = _engine(rng).run_dij(
            _lattice(2, 1),
            n_histories_per_beamlet=4,
            n_batches=2,
            seed=SEED,
            transport_electrons=False,
            correlated=True,
        )
        assert rng.keys == [0, 1, 0, 1, 2, 3, 2, 3]
        assert dij.correlated is True

    def test_default_run_dij_is_correlated(self) -> None:
        """Correlated sampling is the shipped configuration (Phase 4 exit).

        The noise/bias study (``examples/phase4_noise_bias_study.py``) settled
        the default in correlated's favour, so ``run_dij`` without the flag
        keys streams on the within-beamlet index. One configuration ships
        (AGENTS.md 2.10); ``correlated=False`` survives only as the test
        instrument exercised below.
        """
        rng = RecordingRNG()
        dij = _engine(rng).run_dij(
            _lattice(2, 1),
            n_histories_per_beamlet=4,
            n_batches=2,
            seed=SEED,
            transport_electrons=False,
        )
        assert rng.keys == [0, 1, 0, 1, 2, 3, 2, 3]
        assert dij.correlated is True

    def test_independent_instrument_keys_on_the_global_index(self) -> None:
        """``correlated=False`` is the test instrument: the Phase 3 h-mapping.

        Its one remaining job is to isolate column independence for the
        fluence-sum identity, whose quadrature sigma requires it
        (``tests/dij/test_dij_ref.py``).
        """
        rng = RecordingRNG()
        dij = _engine(rng).run_dij(
            _lattice(2, 1),
            n_histories_per_beamlet=4,
            n_batches=2,
            seed=SEED,
            transport_electrons=False,
            correlated=False,
        )
        assert rng.keys == [0, 1, 4, 5, 2, 3, 6, 7]
        assert dij.correlated is False


class TestSingleBeamletAnchor:
    """On a 1x1 lattice rw == h: correlated mode degenerates to the open field."""

    def test_correlated_1x1_column_equals_open_field_dose_bitwise(self) -> None:
        engine = _engine(HostRNG())
        dij = engine.run_dij(
            _lattice(1, 1),
            n_histories_per_beamlet=1_200,
            n_batches=6,
            seed=SEED,
            truncation=0.0,
            correlated=True,
        )
        run = engine.run(
            ParallelBeamSource(energy=ENERGY, z=Z0, x_range=FIELD_X, y_range=FIELD_Y),
            n_histories=1_200,
            n_batches=6,
            seed=SEED,
        )
        np.testing.assert_array_equal(dij.column_dense(0), run.dose)
        np.testing.assert_array_equal(dij.sigma_dense(0), run.dose_sigma)

    def test_beamlet_zero_is_bitwise_invariant_to_the_mode(self) -> None:
        """For beamlet 0, rw == h: the remap is the identity on its histories."""
        engine = _engine(HostRNG())
        kwargs = dict(
            source=_lattice(2, 1),
            n_histories_per_beamlet=400,
            n_batches=2,
            seed=SEED,
            transport_electrons=False,
            truncation=0.0,
        )
        ind = engine.run_dij(**kwargs)
        corr = engine.run_dij(correlated=True, **kwargs)
        np.testing.assert_array_equal(corr.column_dense(0), ind.column_dense(0))
        np.testing.assert_array_equal(corr.sigma_dense(0), ind.sigma_dense(0))


class TestCorrelatedColumnsStayUnbiased:
    """Sharing streams across columns must not move any single column.

    Each correlated column reuses the key set {0..n_per-1}, which is exactly a
    valid independent stream set for *one* beamlet — so per-column statistics
    are untouched by construction, and this pins it: the correlated column is
    chi-squared-consistent with an independently sampled estimate of the same
    column (different seed, so different histories; batched sigmas on both
    sides are individually valid — no cross-column combination anywhere).
    """

    def test_correlated_column_matches_independent_estimate(self) -> None:
        engine = _engine(HostRNG())
        kwargs = dict(
            source=_lattice(2, 1),
            n_histories_per_beamlet=2_000,
            n_batches=8,
            transport_electrons=False,
            truncation=0.0,
        )
        corr = engine.run_dij(seed=SEED, correlated=True, **kwargs)
        ind = engine.run_dij(seed=SEED + 1, **kwargs)
        # Column 1 is the interesting one: in correlated mode it replays
        # beamlet 0's sequences from a shifted entry position.
        dose_c, sigma_c = corr.column_dense(1), corr.sigma_dense(1)
        dose_i, sigma_i = ind.column_dense(1), ind.sigma_dense(1)
        # Beamlet 1's geometric footprint: x 8..14 cm -> ix 4..6, y 2..14 cm
        # -> iy 1..6 on the 2 cm voxels — deterministic, unlike a threshold
        # relative to a noisy per-voxel maximum.
        mask = np.zeros(dose_i.shape, dtype=bool)
        mask[4:7, 1:7, :] = True
        mask &= (sigma_c > 0.0) & (sigma_i > 0.0)
        assert np.count_nonzero(mask) > 100, "mask too small to detect anything"
        assert_chi2_consistent_batched(
            dose_c, sigma_c, corr.n_batches, dose_i, sigma_i, ind.n_batches, mask=mask
        )


class TestDifferenceVarianceCollapses:
    """The variance-reduction claim itself, measured.

    Two beamlets of a uniform lattice over uniform water are geometric
    translates; with shared streams their histories are translates too, so the
    *difference* between a column and its neighbour's column translated onto
    the same frame is nearly deterministic. Its variance over independent
    realizations (different seeds) must sit far below the same quantity under
    independent streams — that variance ratio is the correlated-sampling
    speedup for plan-difference-like quantities.

    Seeds are fixed: the assertion is deterministic, not a statistical flake
    risk. The lattice pitch (6 cm) is an exact multiple of the voxel pitch
    (2 cm), so the translation is a pure 3-voxel index shift.
    """

    N_SEEDS = 6
    SHIFT = 3  # beamlet pitch in x-voxels: 6 cm / 2 cm

    def _column_differences(self, correlated: bool) -> np.ndarray:
        engine = _engine(HostRNG())
        diffs = []
        for s in range(self.N_SEEDS):
            dij = engine.run_dij(
                _lattice(2, 1),
                n_histories_per_beamlet=300,
                n_batches=2,
                seed=SEED + s,
                transport_electrons=False,
                truncation=0.0,
                correlated=correlated,
            )
            a = dij.column_dense(0)
            b = dij.column_dense(1)
            b_on_a = np.zeros_like(b)
            b_on_a[: -self.SHIFT] = b[self.SHIFT :]  # translate b onto a's frame
            diffs.append(a - b_on_a)
        return np.stack(diffs)

    def test_translated_difference_variance_far_below_independent(self) -> None:
        d_ind = self._column_differences(correlated=False)
        d_corr = self._column_differences(correlated=True)
        var_ind = np.var(d_ind, axis=0, ddof=1)
        var_corr = np.var(d_corr, axis=0, ddof=1)
        # Beamlet 0's geometric footprint: x 2..8 cm -> ix 1..3, y 2..14 cm
        # -> iy 1..6 — inside the valid range of the 3-voxel shift by design.
        mask = np.zeros(var_ind.shape, dtype=bool)
        mask[1:4, 1:7, :] = True
        assert np.count_nonzero(mask) > 100, "mask too small to detect anything"
        ratio = var_corr[mask].sum() / var_ind[mask].sum()
        assert ratio < 0.25, f"difference-variance ratio {ratio:.3f}: correlation buys too little"
