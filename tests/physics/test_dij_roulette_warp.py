"""The Dij-entry roulette compiled under Warp, in float64.

Same source, different compiler (see ``test_warp_physics.py``): the kernels apply
``roulette_dij_entry`` to the Dij's float64 dose maps, so the loader must instantiate
it in float64, not in the float32 every other physics function compiles to. Checked
against the analytic rule, not against host draws — the device's uniforms come from
``keyed_uniform`` (``pyradmc.rng.warp_shim``, contract-tested in
``tests/unit/test_warp_keyed_uniform.py``), a counter-based hash of
``(seed, column, voxel)`` that shares nothing with the host stream (AGENTS.md 2.3).

Statistical checks have fixed keys, so they are deterministic on each device.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

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
def kernels():
    from pyradmc.backends.warp.physics import warp_physics
    from pyradmc.rng.warp_shim import keyed_uniform

    rule = warp_physics().roulette_dij_entry

    @wp.kernel
    def apply_rule(
        m: wp.array(dtype=wp.float64),
        s: wp.array(dtype=wp.float64),
        t: wp.array(dtype=wp.float64),
        u: wp.array(dtype=wp.float64),
        out_v: wp.array(dtype=wp.float64),
        out_s: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        v, sg = rule(m[i], s[i], t[i], u[i])
        out_v[i] = v
        out_s[i] = sg

    @wp.kernel
    def draw_keyed(
        seed: int,
        n_voxels: int,
        column_offset: int,
        out: wp.array(dtype=float),
    ):
        tid = wp.tid()
        local = tid // n_voxels
        v = tid - local * n_voxels
        out[tid] = keyed_uniform(seed, column_offset + local, v)

    return apply_rule, draw_keyed


def _apply(kernels, device: str, m, s, t, u) -> tuple[np.ndarray, np.ndarray]:
    apply_rule, _ = kernels
    arrays = [wp.array(np.asarray(a, dtype=np.float64), device=device) for a in (m, s, t, u)]
    n = len(m)
    out_v = wp.zeros(n, dtype=wp.float64, device=device)
    out_s = wp.zeros(n, dtype=wp.float64, device=device)
    wp.launch(apply_rule, dim=n, inputs=[*arrays, out_v, out_s], device=device)
    return out_v.numpy().copy(), out_s.numpy().copy()


def _keyed(kernels, device: str, seed: int, n_cols: int, n_voxels: int, offset: int = 0):
    _, draw_keyed = kernels
    out = wp.zeros(n_cols * n_voxels, dtype=float, device=device)
    wp.launch(
        draw_keyed, dim=n_cols * n_voxels, inputs=[seed, n_voxels, offset, out], device=device
    )
    return out.numpy().reshape(n_cols, n_voxels).astype(np.float64)


class TestRuleInFloat64:
    def test_entries_keep_their_float64_value(self, kernels, device: str) -> None:
        """At or above threshold the entry comes back unchanged, to the last bit.

        A value that does not fit float32 (1 + 2^-40) would be rounded by a float32
        instantiation; float64 returns it as stored. Values are copies, not
        arithmetic, so this is the rule's contract, not a cross-target comparison.
        """
        m = np.array([1.0 + 2.0**-40, 2.0, 5.0, 0.0, 0.0])
        s = np.array([1.0e-3 + 2.0**-50, 0.1, 0.0, 0.0, 0.0])
        t = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
        u = np.array([0.5, 0.0, 0.9, 0.0, 0.5])
        value, sigma = _apply(kernels, device, m, s, t, u)
        assert value.tolist() == [m[0], 2.0, 5.0, 0.0, 0.0]
        assert sigma.tolist() == [s[0], 0.1, 0.0, 0.0, 0.0]

    def test_tail_entries_survive_as_the_threshold_or_drop(self, kernels, device: str) -> None:
        t0 = 1.0e-3 + 2.0**-45  # not representable in float32
        m = np.array([0.3e-3, 0.3e-3, 0.7e-3])
        s = np.array([0.05e-3, 0.05e-3, 0.0])
        t = np.full(3, t0)
        u = np.array([0.1, 0.5, 0.69])
        value, sigma = _apply(kernels, device, m, s, t, u)
        assert value[0] == t0 and value[2] == t0
        assert value[1] == 0.0 and sigma[1] == 0.0
        # Same formula in float64 on both sides: a precision check, not dose equality.
        want = [math.sqrt(s[i] ** 2 * t0 / m[i] + t0 * (t0 - m[i])) for i in (0, 2)]
        assert sigma[[0, 2]] == pytest.approx(want, rel=1e-14)


def test_keep_frequency_through_keyed_uniforms(kernels, device: str) -> None:
    """The rule fed device uniforms keeps a tail entry with probability m / t.

    40000 keys at p = 0.3, binomial bound 3.89 sigma (false alarm 1e-4).
    """
    n, p = 40_000, 0.3
    u = _keyed(kernels, device, SEED, 40, 1_000).ravel()
    value, _ = _apply(kernels, device, np.full(n, p), np.zeros(n), np.ones(n), u)
    kept = int(np.count_nonzero(value))
    assert abs(kept - n * p) < 3.89 * math.sqrt(n * p * (1.0 - p))
