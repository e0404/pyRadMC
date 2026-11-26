"""Contract tests for ``keyed_uniform``, the kernel-side keyed draw.

``keyed_uniform(seed, key, subkey)`` (:mod:`pyradmc.rng.warp_shim`) is one U[0, 1)
draw that is a pure function of its three integers, by two-level counter hashing.
The device's stochastic Dij truncation keys it on ``(salted seed, global column,
voxel)`` so that its count and compaction passes, and any split of the matrix,
make the same decision. Pinned within one device (AGENTS.md 2.3): the range,
uniformity, purity in the keys, decorrelation of neighbouring keys, and behaviour
at keys whose product with the voxel count would overflow int32.

Statistical checks have fixed keys, so they are deterministic on each device.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


@pytest.fixture(scope="module")
def draw_keyed():
    from pyradmc.rng.warp_shim import keyed_uniform

    @wp.kernel
    def draw(seed: int, n_voxels: int, column_offset: int, out: wp.array(dtype=float)):
        tid = wp.tid()
        local = tid // n_voxels
        out[tid] = keyed_uniform(seed, column_offset + local, tid - local * n_voxels)

    return draw


def keyed(draw, device: str, seed: int, n_cols: int, n_voxels: int, offset: int = 0):
    """``(n_cols, n_voxels)`` draws for columns ``offset ..`` and voxels ``0 ..``."""
    out = wp.zeros(n_cols * n_voxels, dtype=float, device=device)
    wp.launch(draw, dim=n_cols * n_voxels, inputs=[seed, n_voxels, offset, out], device=device)
    return out.numpy().reshape(n_cols, n_voxels).astype(np.float64)


def test_range_and_uniformity(draw_keyed, device: str) -> None:
    """Chi-square over 64 bins of 2e5 draws, p > 1e-3; values in [0, 1)."""
    u = keyed(draw_keyed, device, SEED, 50, 4_000).ravel()
    assert np.all(u >= 0.0) and np.all(u < 1.0)
    counts, _ = np.histogram(u, bins=64, range=(0.0, 1.0))
    assert stats.chisquare(counts).pvalue > 1e-3


def test_is_a_pure_function_of_its_keys(draw_keyed, device: str) -> None:
    """Columns 3.. drawn at offset 3 equal columns 3.. of a launch from 0; repeatable."""
    whole = keyed(draw_keyed, device, SEED, 8, 500)
    part = keyed(draw_keyed, device, SEED, 5, 500, offset=3)
    assert np.array_equal(part, whole[3:])
    assert np.array_equal(keyed(draw_keyed, device, SEED, 8, 500), whole)


def test_neighbouring_keys_are_uncorrelated(draw_keyed, device: str) -> None:
    """Adjacent voxels and adjacent columns: |r| < 4 / sqrt(n) (false alarm 6e-5 each)."""
    u = keyed(draw_keyed, device, SEED, 200, 1_000)
    r_voxel = np.corrcoef(u[:, :-1].ravel(), u[:, 1:].ravel())[0, 1]
    r_column = np.corrcoef(u[:-1].ravel(), u[1:].ravel())[0, 1]
    assert abs(r_voxel) < 4.0 / math.sqrt(u[:, 1:].size)
    assert abs(r_column) < 4.0 / math.sqrt(u[1:].size)


def test_seed_changes_the_draws(draw_keyed, device: str) -> None:
    a = keyed(draw_keyed, device, SEED, 4, 100)
    b = keyed(draw_keyed, device, SEED + 1, 4, 100)
    assert not np.array_equal(a, b)


def test_keys_past_the_int32_product_stay_uniform(draw_keyed, device: str) -> None:
    """Columns 3000.. of a 1e6-voxel grid: ``column * n_voxels`` would exceed 2^31.

    Two-level hashing never forms that product; the draws stay in range with the
    U[0, 1) mean (4 sigma).
    """
    u = keyed(draw_keyed, device, SEED, 2, 1_000_000, offset=3_000)
    assert np.all(u >= 0.0) and np.all(u < 1.0)
    assert abs(u.mean() - 0.5) < 4.0 * math.sqrt(1.0 / 12.0 / u.size)
