"""Stochastic Dij truncation in the assembler: unbiased sparsification.

``truncation_mode="stochastic"`` replaces the hard cut of sub-threshold entries by
Russian roulette (Thieke et al., Med. Phys. 29 (2002) 676, doi:10.1118/1.1469633;
the per-entry rule is pinned in ``tests/physics/test_dij_roulette.py``). Here:

- the exact statistics of the column sums against the untruncated matrix (the
  oracle: mean zero, variance ``sum m (t - m)`` over the tail entries);
- the reproducibility contract of the draw — a pure function of
  ``(seed, global column, voxel)``, so it is invariant to how columns are grouped
  into ``add_block`` calls, independent between columns, and on a stream of its own;
- that the hard mode and ``truncation = 0`` are untouched.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from pyradmc import DIJ_TRUNCATION_RELATIVE
from pyradmc.physics.roulette import roulette_dij_entry
from pyradmc.rng.host import HostRNG, uniform_block
from pyradmc.scoring.dij import (
    DIJ_ROULETTE_SALT,
    DijAssembler,
    DijResult,
    _roulette_row,
    _roulette_uniforms,
)
from tests.conftest import SEED

N_COLS = 48
N_VOX = 3_000
SHAPE = (10, 15, 20)


def _block(seed: int = SEED, n_cols: int = N_COLS) -> tuple[np.ndarray, np.ndarray]:
    """A dense block with a populated sub-threshold tail in every column.

    Column maximum 1 at one voxel; 30 % empty voxels; the rest log-uniform over
    1e-7..1e-1 of the maximum, so about two thirds of the nonzero entries fall
    below the default 1e-3 threshold. Sigma is 20 % of the dose.
    """
    gen = np.random.default_rng(seed)
    dose = 10.0 ** gen.uniform(-7.0, -1.0, size=(n_cols, N_VOX))
    dose[gen.random((n_cols, N_VOX)) < 0.3] = 0.0
    dose[np.arange(n_cols), gen.integers(0, N_VOX, n_cols)] = 1.0
    return dose, 0.2 * dose


def _assemble(
    dose: np.ndarray,
    sigma: np.ndarray,
    truncation: float = DIJ_TRUNCATION_RELATIVE,
    mode: str = "stochastic",
    seed: int | None = SEED,
    splits: tuple[int, ...] = (),
) -> DijResult:
    kwargs: dict[str, object] = {"truncation_mode": mode}
    if seed is not None:
        kwargs["seed"] = seed
    asm = DijAssembler(
        grid_shape=SHAPE,
        n_beamlets=dose.shape[0],
        n_histories_per_beamlet=10,
        n_batches=2,
        truncation=truncation,
        **kwargs,  # type: ignore[arg-type]
    )
    edges = (0, *splits, dose.shape[0])
    for lo, hi in itertools.pairwise(edges):
        asm.add_block(lo, dose[lo:hi], sigma[lo:hi])
    return asm.finalize(energy_emitted=1.0, energy_deposited=0.0, energy_escaped=0.0)


def _assert_same(a: DijResult, b: DijResult) -> None:
    for name in ("indptr", "indices", "dose", "sigma"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype, name
        assert np.array_equal(x, y), name


class TestExactStatistics:
    """The stochastic matrix against the untruncated one, same dense block.

    Per column ``j`` the difference of column sums ``D_j`` has mean 0 and variance
    ``V_j = sum over tail entries of m (t - m)`` (the entries are independent
    Bernoulli draws scaled by ``t``). With ~1400 tail entries per column ``D_j / sqrt(V_j)``
    is normal to good approximation. Fixed seed: deterministic.

    False alarm: the chi-square over the 48 columns is two-sided at 1 %, the pooled
    z-score two-sided at 0.1 % (|z| < 3.29), the stored-variance z-score two-sided
    at 0.1 %; by the union bound at most 1.2 % for the class.
    """

    def _tail(self, dose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = DIJ_TRUNCATION_RELATIVE * dose.max(axis=1, keepdims=True)
        tail = (dose > 0.0) & (dose < t)
        return t, tail

    def test_column_sums_are_unbiased_with_the_bernoulli_variance(self) -> None:
        dose, sigma = _block()
        full = _assemble(dose, sigma, truncation=0.0)
        sto = _assemble(dose, sigma)
        t, tail = self._tail(dose)
        assert tail.sum(axis=1).min() > 1_000, "the tail must be populated"
        variance = np.where(tail, dose * (t - dose), 0.0).sum(axis=1)
        diff = np.array(
            [sto.column_dense(j).sum() - full.column_dense(j).sum() for j in range(N_COLS)]
        )
        z = diff / np.sqrt(variance)
        chi2 = float(np.sum(z**2))
        p_chi2 = float(stats.chi2.cdf(chi2, N_COLS))
        assert 0.005 < p_chi2 < 0.995, f"chi2={chi2:.1f} over {N_COLS} columns"
        z_pool = float(diff.sum() / math.sqrt(variance.sum()))
        assert abs(z_pool) < 3.29, f"pooled z = {z_pool:.2f}"

    def test_tail_survivors_are_stored_as_the_threshold(self) -> None:
        dose, sigma = _block()
        sto = _assemble(dose, sigma)
        t, tail = self._tail(dose)
        for j in range(N_COLS):
            col = sto.column_dense(j).ravel()
            kept_tail = tail[j] & (col > 0.0)
            assert np.all(col[kept_tail] == t[j, 0])
            above = dose[j] >= t[j, 0]
            assert np.array_equal(col[above], dose[j][above])
            assert np.array_equal(sto.sigma_dense(j).ravel()[above], sigma[j][above])

    def test_expected_stored_variance_is_the_total_variance(self) -> None:
        """sum of stored sigma^2 over the tail vs sum of s^2 + m (t - m)."""
        dose, sigma = _block()
        sto = _assemble(dose, sigma)
        t, tail = self._tail(dose)
        m, s, tt = dose[tail], sigma[tail], np.broadcast_to(t, dose.shape)[tail]
        var_kept = s**2 * tt / m + tt * (tt - m)
        p = m / tt
        target = float(np.sum(s**2 + m * (tt - m)))
        spread = math.sqrt(float(np.sum(var_kept**2 * p * (1.0 - p))))
        stored = 0.0
        for j in range(N_COLS):
            sig = sto.sigma_dense(j).ravel()
            stored += float(np.sum(sig[tail[j]] ** 2))
        assert abs(stored - target) < 3.29 * spread, (stored, target, spread)


class TestReproducibility:
    def test_same_seed_is_byte_identical(self) -> None:
        dose, sigma = _block()
        _assert_same(_assemble(dose, sigma), _assemble(dose, sigma))

    @pytest.mark.parametrize("splits", [(1,), (5, 17), tuple(range(1, N_COLS))])
    def test_grouping_into_blocks_is_inert(self, splits: tuple[int, ...]) -> None:
        """The draw is keyed by the global column index, not the block's row."""
        dose, sigma = _block()
        _assert_same(_assemble(dose, sigma, splits=splits), _assemble(dose, sigma))

    def test_a_different_seed_changes_the_kept_tail(self) -> None:
        dose, sigma = _block()
        a, b = _assemble(dose, sigma), _assemble(dose, sigma, seed=SEED + 1)
        assert not np.array_equal(a.indices, b.indices)

    def test_identical_columns_draw_independently(self) -> None:
        """Correlated-Dij columns can coincide; their roulette must not.

        Under independence two columns with the same ``n`` tail entries agree on
        an entry with probability ``p^2 + (1 - p)^2``; the agreement count is tested
        against that binomial at 4 sigma (false alarm 6e-5).
        """
        dose, sigma = _block(n_cols=1)
        dose2, sigma2 = np.repeat(dose, 2, axis=0), np.repeat(sigma, 2, axis=0)
        sto = _assemble(dose2, sigma2)
        t = DIJ_TRUNCATION_RELATIVE * dose[0].max()
        tail = (dose[0] > 0.0) & (dose[0] < t)
        k0 = sto.column_dense(0).ravel()[tail] > 0.0
        k1 = sto.column_dense(1).ravel()[tail] > 0.0
        p = dose[0][tail] / t
        q = p**2 + (1.0 - p) ** 2
        agree = int(np.count_nonzero(k0 == k1))
        assert abs(agree - q.sum()) < 4.0 * math.sqrt(float(np.sum(q * (1.0 - q))))

    def test_the_stream_is_not_a_transport_stream(self) -> None:
        """Salted sub-seed: column ``j`` never replays the transport stream ``(seed, j)``.

        Under correlated sampling the transport streams are ``(seed, rw)`` for the
        within-beamlet index ``rw``, so the columns' indices collide with history
        indices; the salt is what keeps the roulette off them.
        """
        assert DIJ_ROULETTE_SALT != 0
        for j in range(4):
            ours = _roulette_uniforms(SEED, j, 64)
            transport = uniform_block(HostRNG().init_state(SEED, j), 64)
            assert not np.array_equal(ours, transport)
            # The stream is a pure function of (seed, column, voxel): a shorter
            # draw is a prefix of a longer one.
            assert np.array_equal(_roulette_uniforms(SEED, j, 16), ours[:16])


class TestHardModeAndNoOp:
    def test_hard_is_the_default_and_is_recorded(self) -> None:
        dose, sigma = _block()
        asm = DijAssembler(
            grid_shape=SHAPE,
            n_beamlets=N_COLS,
            n_histories_per_beamlet=10,
            n_batches=2,
            truncation=DIJ_TRUNCATION_RELATIVE,
        )
        asm.add_block(0, dose, sigma)
        default = asm.finalize(energy_emitted=1.0, energy_deposited=0.0, energy_escaped=0.0)
        assert default.truncation_mode == "hard"
        _assert_same(default, _assemble(dose, sigma, mode="hard", seed=None))
        assert _assemble(dose, sigma).truncation_mode == "stochastic"

    def test_hard_mode_ignores_the_seed(self) -> None:
        dose, sigma = _block()
        _assert_same(
            _assemble(dose, sigma, mode="hard", seed=SEED),
            _assemble(dose, sigma, mode="hard", seed=None),
        )

    def test_zero_truncation_is_a_no_op_in_both_modes(self) -> None:
        dose, sigma = _block()
        hard = _assemble(dose, sigma, truncation=0.0, mode="hard", seed=None)
        sto = _assemble(dose, sigma, truncation=0.0)
        _assert_same(hard, sto)
        assert hard.indices.size == np.count_nonzero(dose)

    def test_stochastic_mode_needs_a_seed(self) -> None:
        dose, sigma = _block(n_cols=1)
        with pytest.raises(ValueError, match="seed"):
            _assemble(dose, sigma, seed=None)

    def test_unknown_mode_is_rejected(self) -> None:
        dose, sigma = _block(n_cols=1)
        with pytest.raises(ValueError, match="truncation_mode"):
            _assemble(dose, sigma, mode="soft")

    def test_negative_truncation_raises_in_stochastic_mode_too(self) -> None:
        dose, sigma = _block(n_cols=1)
        with pytest.raises(ValueError, match="negative truncation"):
            _assemble(dose, sigma, truncation=-1e-3)


class TestSingleSource:
    """The assembler's vectorized rule is the scalar physics rule, bit for bit."""

    def test_vectorized_rule_equals_the_scalar_rule(self) -> None:
        gen = np.random.default_rng(SEED)
        t = 1.0e-3
        m = np.concatenate(
            [
                [0.0, t, np.nextafter(t, 0.0), 2.0 * t, 1.0, 1e-12],
                10.0 ** gen.uniform(-7.0, 0.0, 5_000),
            ]
        )
        s = 0.3 * m * gen.random(m.size)
        u = np.concatenate([[0.0, 0.5, 0.999999, 0.1, 0.2, 0.0], gen.random(5_000)])
        value, sig = _roulette_row(m, s, t, u)
        for i in range(m.size):
            want = roulette_dij_entry(float(m[i]), float(s[i]), t, float(u[i]))
            assert (float(value[i]), float(sig[i])) == want, i

    def test_vectorized_rule_with_zero_threshold_keeps_every_nonzero(self) -> None:
        m = np.array([0.0, 1e-30, 1.0])
        value, sig = _roulette_row(m, 0.1 * m, 0.0, np.zeros(3))
        assert np.array_equal(value, m)
        assert np.array_equal(sig, 0.1 * m)
