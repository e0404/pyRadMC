"""Device stochastic truncation kernels on synthetic dose maps.

The count and compaction passes of ``truncation_mode="stochastic"``
(:mod:`pyradmc.backends.warp.sparsify`) apply the roulette of
``roulette_dij_entry`` with the uniform ``keyed_uniform(seed ^ DIJ_ROULETTE_SALT,
global column, voxel)``. Pinned here, within one device:

- the two passes agree (the counts size the compaction exactly, indices ascend);
- the output is independent of the chunk layout and of how columns are split into
  launches, given the global column offset;
- ``truncation = 0`` reproduces the hard kernels byte for byte;
- the decisions are the scalar rule applied to the keyed uniforms;
- the column sums are unbiased with the Bernoulli variance (the exact-statistics
  oracle of the host assembler, ``tests/unit/test_dij_stochastic_truncation.py``,
  on the device's own draws).

No assertion compares the device's stochastic matrix to the host assembler's: the
two draw independent uniforms (AGENTS.md 2.3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from pyradmc import DIJ_TRUNCATION_RELATIVE  # noqa: E402
from pyradmc.backends.warp import kernels, sparsify  # noqa: E402
from pyradmc.physics.roulette import roulette_dij_entry  # noqa: E402
from pyradmc.rng.warp_shim import keyed_uniform  # noqa: E402
from pyradmc.scoring.dij import DIJ_ROULETTE_SALT  # noqa: E402
from tests.conftest import SEED  # noqa: E402

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
ROULETTE_SEED = SEED ^ DIJ_ROULETTE_SALT


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _block(n_cols: int = 48, n_voxels: int = 3_000, seed: int = SEED):
    """Same shape as the host assembler's oracle block: a populated tail per column."""
    gen = np.random.default_rng(seed)
    dose = 10.0 ** gen.uniform(-7.0, -1.0, size=(n_cols, n_voxels))
    dose[gen.random((n_cols, n_voxels)) < 0.3] = 0.0
    dose[np.arange(n_cols), gen.integers(0, n_voxels, n_cols)] = 1.0
    return dose, 0.2 * dose


def _layout(n_voxels: int, n_chunks: int) -> tuple[int, int]:
    chunk = -(-n_voxels // n_chunks)
    return -(-n_voxels // chunk), chunk


def _sweep(
    device: str,
    mean: np.ndarray,
    sigma: np.ndarray,
    trunc: float,
    *,
    stochastic: bool = True,
    column_offset: int = 0,
    roulette_seed: int = ROULETTE_SEED,
    n_chunks: int | None = None,
):
    """Column max, count pass, host prefix sums, compaction: as the engine drives it."""
    group, n_voxels = mean.shape
    if n_chunks is None:
        n_chunks, chunk = kernels.truncation_chunk_layout(n_voxels, group)
    else:
        n_chunks, chunk = _layout(n_voxels, n_chunks)
    mean_dev = wp.array(mean.reshape(-1), dtype=wp.float64, device=device)
    sigma_dev = wp.array(sigma.reshape(-1), dtype=wp.float64, device=device)
    col_max = wp.zeros(group, dtype=wp.float64, device=device)
    wp.launch(
        kernels.column_max,
        dim=group * n_voxels,
        inputs=[mean_dev, n_voxels, col_max],
        device=device,
    )
    chunk_counts = wp.zeros(group * n_chunks, dtype=wp.int32, device=device)
    if stochastic:
        wp.launch(
            sparsify.count_kept_per_chunk_stochastic,
            dim=group * n_chunks,
            inputs=[
                mean_dev,
                sigma_dev,
                n_voxels,
                n_chunks,
                chunk,
                float(trunc),
                col_max,
                column_offset,
                roulette_seed,
                chunk_counts,
            ],
            device=device,
        )
    else:
        wp.launch(
            kernels.count_kept_per_chunk,
            dim=group * n_chunks,
            inputs=[mean_dev, n_voxels, n_chunks, chunk, float(trunc), col_max, chunk_counts],
            device=device,
        )
    per_chunk = chunk_counts.numpy().reshape(group, n_chunks)
    counts = per_chunk.sum(axis=1).astype(np.int32)
    column_base = np.zeros(group, dtype=np.int64)
    np.cumsum(counts[:-1], out=column_base[1:])
    within = np.zeros((group, n_chunks), dtype=np.int64)
    np.cumsum(per_chunk[:, :-1], axis=1, out=within[:, 1:])
    chunk_base = wp.array(
        (column_base[:, None] + within).reshape(-1).astype(np.int32), dtype=wp.int32, device=device
    )
    nnz = int(counts.sum())
    # Sentinels, so an entry the compaction never wrote cannot pass for a real one.
    out_i = wp.array(np.full(nnz, -1, dtype=np.int64), dtype=wp.int64, device=device)
    out_d = wp.array(np.full(nnz, -1.0), dtype=wp.float64, device=device)
    out_s = wp.array(np.full(nnz, -1.0), dtype=wp.float64, device=device)
    common = [mean_dev, sigma_dev, n_voxels, n_chunks, chunk, float(trunc), col_max]
    if stochastic:
        wp.launch(
            sparsify.compact_column_chunked_stochastic,
            dim=group * n_chunks,
            inputs=[*common, column_offset, roulette_seed, chunk_base, out_i, out_d, out_s],
            device=device,
        )
    else:
        wp.launch(
            kernels.compact_column_chunked,
            dim=group * n_chunks,
            inputs=[*common, chunk_base, out_i, out_d, out_s],
            device=device,
        )
    wp.synchronize_device(device)
    return counts, out_i.numpy().copy(), out_d.numpy().copy(), out_s.numpy().copy()


@wp.kernel
def _draw_keyed(seed: int, n_voxels: int, column_offset: int, out: wp.array(dtype=float)):
    tid = wp.tid()
    local = tid // n_voxels
    out[tid] = keyed_uniform(seed, column_offset + local, tid - local * n_voxels)


def _columns(counts, idx, dose, sigma, n_voxels: int):
    """Dense (value, sigma) maps from a compacted block."""
    group = counts.size
    value = np.zeros((group, n_voxels))
    sig = np.zeros((group, n_voxels))
    start = 0
    for j, n in enumerate(counts):
        sl = slice(start, start + int(n))
        value[j, idx[sl]] = dose[sl]
        sig[j, idx[sl]] = sigma[sl]
        start += int(n)
    return value, sig


def _assert_same(a, b) -> None:
    for x, y in zip(a, b, strict=True):
        assert x.dtype == y.dtype
        assert np.array_equal(x, y)


class TestPassesAndLayout:
    def test_count_pass_sizes_the_compaction_exactly(self, device: str) -> None:
        dose, sigma = _block()
        counts, idx, val, sig = _sweep(device, dose, sigma, DIJ_TRUNCATION_RELATIVE)
        assert np.all(idx >= 0), "the compaction left entries the count pass reserved"
        assert np.all(val > 0.0) and np.all(sig >= 0.0)
        start = 0
        for n in counts:
            assert np.all(np.diff(idx[start : start + int(n)]) > 0)
            start += int(n)

    @pytest.mark.parametrize("n_chunks", [1, 7, 64, 3_000])
    def test_chunk_layout_is_inert(self, device: str, n_chunks: int) -> None:
        dose, sigma = _block(n_cols=6)
        want = _sweep(device, dose, sigma, DIJ_TRUNCATION_RELATIVE)
        _assert_same(_sweep(device, dose, sigma, DIJ_TRUNCATION_RELATIVE, n_chunks=n_chunks), want)

    def test_splitting_columns_into_launches_is_inert(self, device: str) -> None:
        """Columns 2..5 launched alone at offset 2 equal columns 2..5 of the whole block."""
        dose, sigma = _block(n_cols=6)
        whole = _sweep(device, dose, sigma, DIJ_TRUNCATION_RELATIVE)
        head = _sweep(device, dose[:2], sigma[:2], DIJ_TRUNCATION_RELATIVE)
        tail = _sweep(device, dose[2:], sigma[2:], DIJ_TRUNCATION_RELATIVE, column_offset=2)
        joined = (
            np.concatenate([head[0], tail[0]]),
            *(np.concatenate([h, t]) for h, t in zip(head[1:], tail[1:], strict=True)),
        )
        _assert_same(joined, whole)

    def test_zero_truncation_is_the_hard_kernels_byte_for_byte(self, device: str) -> None:
        dose, sigma = _block(n_cols=6)
        _assert_same(
            _sweep(device, dose, sigma, 0.0), _sweep(device, dose, sigma, 0.0, stochastic=False)
        )

    def test_a_different_seed_changes_the_kept_tail(self, device: str) -> None:
        dose, sigma = _block(n_cols=6)
        a = _sweep(device, dose, sigma, DIJ_TRUNCATION_RELATIVE)
        b = _sweep(device, dose, sigma, DIJ_TRUNCATION_RELATIVE, roulette_seed=ROULETTE_SEED + 1)
        assert not np.array_equal(a[1], b[1])


class TestTheRule:
    def test_decisions_are_the_scalar_rule_on_the_keyed_uniforms(self, device: str) -> None:
        """Kept set and stored value equal the host rule fed the device's own uniforms.

        The comparison ``u < m / t`` is one IEEE float64 division on each side (the
        sparsify module compiles without fast-math), so the decisions must agree
        entry for entry; the stored sigma is the same float64 formula and is held
        to rounding. A same-formula check, not a comparison of two Monte Carlo
        results.
        """
        dose, sigma = _block(n_cols=6)
        offset = 11
        group, n_voxels = dose.shape
        counts, idx, val, sig = _sweep(
            device, dose, sigma, DIJ_TRUNCATION_RELATIVE, column_offset=offset
        )
        got_v, got_s = _columns(counts, idx, val, sig, n_voxels)
        u_dev = wp.zeros(group * n_voxels, dtype=float, device=device)
        wp.launch(
            _draw_keyed,
            dim=group * n_voxels,
            inputs=[ROULETTE_SEED, n_voxels, offset, u_dev],
            device=device,
        )
        u = u_dev.numpy().reshape(group, n_voxels).astype(np.float64)
        for j in range(group):
            t = DIJ_TRUNCATION_RELATIVE * float(dose[j].max())
            for v in range(n_voxels):
                want_v, want_s = roulette_dij_entry(
                    float(dose[j, v]), float(sigma[j, v]), t, float(u[j, v])
                )
                assert got_v[j, v] == want_v, (j, v)
                assert got_s[j, v] == pytest.approx(want_s, rel=1e-14, abs=0.0), (j, v)


class TestExactStatistics:
    """Column sums against ``truncation = 0`` on the same maps; Bernoulli variance.

    False alarm as for the host oracle: chi-square over 48 columns two-sided at
    1 %, pooled z two-sided at 0.1 %, stored-variance z two-sided at 0.1 %.
    """

    def test_column_sums_are_unbiased_with_the_bernoulli_variance(self, device: str) -> None:
        dose, sigma = _block()
        n_cols, n_voxels = dose.shape
        sto_v, _ = _columns(*_sweep(device, dose, sigma, DIJ_TRUNCATION_RELATIVE), n_voxels)
        full_v, _ = _columns(*_sweep(device, dose, sigma, 0.0), n_voxels)
        t = DIJ_TRUNCATION_RELATIVE * dose.max(axis=1, keepdims=True)
        tail = (dose > 0.0) & (dose < t)
        variance = np.where(tail, dose * (t - dose), 0.0).sum(axis=1)
        diff = sto_v.sum(axis=1) - full_v.sum(axis=1)
        z = diff / np.sqrt(variance)
        chi2 = float(np.sum(z**2))
        assert 0.005 < float(stats.chi2.cdf(chi2, n_cols)) < 0.995, chi2
        z_pool = float(diff.sum() / math.sqrt(variance.sum()))
        assert abs(z_pool) < 3.29, z_pool

    def test_expected_stored_variance_is_the_total_variance(self, device: str) -> None:
        dose, sigma = _block()
        _, n_voxels = dose.shape
        _, sto_s = _columns(*_sweep(device, dose, sigma, DIJ_TRUNCATION_RELATIVE), n_voxels)
        t = np.broadcast_to(DIJ_TRUNCATION_RELATIVE * dose.max(axis=1, keepdims=True), dose.shape)
        tail = (dose > 0.0) & (dose < t)
        m, s, tt = dose[tail], sigma[tail], t[tail]
        var_kept = s**2 * tt / m + tt * (tt - m)
        p = m / tt
        target = float(np.sum(s**2 + m * (tt - m)))
        spread = math.sqrt(float(np.sum(var_kept**2 * p * (1.0 - p))))
        stored = float(np.sum(sto_s[tail] ** 2))
        assert abs(stored - target) < 3.29 * spread, (stored, target, spread)
