r"""The Goudsmit-Saunderson grid sampler, compiled under Warp, on each device.

Same pattern as ``test_warp_physics.py``: the single-source sampler
(:func:`pyradmc.physics.gs.sample_gs_cos_theta_grid`) is compiled to ``@wp.func``
by the physics loader and driven from a probe kernel against the same oracles the
host tier uses — the exact first-moment anchor, the host-drawn distribution
(statistical, AGENTS.md 2.3), and the one-uniform stream-parity contract that
every paired GS-vs-Gaussian transport comparison leans on.

The grid uploaded here is a small real window built by ``build_gs_grid`` — the
same constructor, node parameters and fixed build seed as the reference
backend's lazy cache — cast to float32 exactly as the engine casts it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyradmc.data.goudsmit_saunderson import build_gs_grid  # noqa: E402

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

N = 30_000
LOG_ETA = -11.45  # off-node; eta ~ 1.06e-5, the low-MeV soft-tissue regime
THETA2_A = 0.02
THETA2_B = 0.05
THETA2_SUBFLOOR = 1.0e-13  # keyed at the window floor; deflection immeasurable


@pytest.fixture(scope="module")
def grid():
    """One small real window: two eta columns around LOG_ETA, Lambda ~ 10-190."""
    return build_gs_grid(LOG_ETA, LOG_ETA, 0.08, theta2_min=5.0e-3)


@pytest.fixture(scope="module", params=DEVICES)
def device_draws(request, grid):
    device = request.param
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    from pyradmc.backends.warp.physics import warp_physics
    from pyradmc.rng.warp_shim import WarpRNGState, init_slot, uniform

    p = warp_physics()
    gs_grid = p.sample_gs_cos_theta_grid
    hinge = p.sample_hinge_cos_theta

    n_eta, n_theta2, n_u = grid.values.shape
    fy_floor = float(math.log(5.0e-3) * grid.bins_per_log)

    @wp.kernel
    def sample_gs(
        seed: int,
        values: wp.array3d(dtype=float),
        ix0: int,
        iy0: int,
        n_eta_k: int,
        n_theta2_k: int,
        n_u_k: int,
        fx: float,
        fy_a: float,
        fy_b: float,
        slots: wp.array(dtype=wp.uint32),
        out: wp.array2d(dtype=float),
    ):
        tid = wp.tid()
        slots[tid] = init_slot(seed, tid)
        state = WarpRNGState()
        state.slots = slots
        state.idx = tid

        out[tid, 0] = gs_grid(
            values, ix0, iy0, n_eta_k, n_theta2_k, n_u_k, fx, fy_a, THETA2_A, state
        )
        out[tid, 1] = gs_grid(
            values, ix0, iy0, n_eta_k, n_theta2_k, n_u_k, fx, fy_b, THETA2_B, state
        )

        # Stream parity: from identical states, each model draws once; the next
        # uniform is then identical iff the consumption was.
        slots[tid] = init_slot(seed, tid)
        out[tid, 2] = hinge(THETA2_A, state)
        out[tid, 3] = uniform(state)
        slots[tid] = init_slot(seed, tid)
        out[tid, 4] = gs_grid(
            values, ix0, iy0, n_eta_k, n_theta2_k, n_u_k, fx, fy_a, THETA2_A, state
        )
        out[tid, 5] = uniform(state)

        # Sub-floor step, keyed at the window floor: forward to the last float.
        out[tid, 6] = gs_grid(
            values, ix0, iy0, n_eta_k, n_theta2_k, n_u_k, fx, fy_floor, THETA2_SUBFLOOR, state
        )

    values32 = wp.array(grid.values.astype(np.float32), dtype=float, device=device)
    slots = wp.zeros(N, dtype=wp.uint32, device=device)
    out = wp.zeros((N, 7), dtype=float, device=device)
    wp.launch(
        sample_gs,
        dim=N,
        inputs=[
            SEED,
            values32,
            grid.ix0,
            grid.iy0,
            n_eta,
            n_theta2,
            n_u,
            float(LOG_ETA * grid.bins_per_log),
            float(math.log(THETA2_A) * grid.bins_per_log),
            float(math.log(THETA2_B) * grid.bins_per_log),
            slots,
        ],
        outputs=[out],
        device=device,
    )
    wp.synchronize_device(device)
    return out.numpy()


class TestFirstMomentAnchor:
    """The anchoring survives the device: float32 cast, blend, and expm1 shim."""

    @pytest.mark.parametrize("column, theta2", [(0, THETA2_A), (1, THETA2_B)])
    def test_sampled_first_moment_matches_the_request(
        self, device_draws: np.ndarray, column: int, theta2: float
    ) -> None:
        one_minus_cos = 1.0 - device_draws[:, column].astype(np.float64)
        measured = float(np.mean(one_minus_cos))
        expected = float(-np.expm1(-0.5 * theta2))
        sem = float(np.std(one_minus_cos)) / np.sqrt(N)
        assert abs(measured - expected) < 4.0 * sem + 1.0e-3 * expected, (
            f"requested <1-cos>={expected:.4e}, device sampled {measured:.4e}"
        )

    def test_subfloor_step_is_forward(self, device_draws: np.ndarray) -> None:
        """The anchor ``1 - e^(-theta2/2)`` underflows float32's resolution at 1,
        so the deflection must come back exactly forward — never NaN, never
        backscatter (the max(-1, nan) landmine the zero-row guard exists for)."""
        assert np.all(device_draws[:, 6] == 1.0)


class TestDistribution:
    """Device draws against host draws from the same float64 grid (chi-squared)."""

    @pytest.mark.parametrize("column, theta2", [(0, THETA2_A), (1, THETA2_B)])
    def test_device_distribution_matches_host(
        self, grid, device_draws: np.ndarray, column: int, theta2: float
    ) -> None:
        from pyradmc.physics.gs import sample_gs_cos_theta_grid

        rng = np.random.default_rng(SEED + 1)
        fx = LOG_ETA * grid.bins_per_log
        fy = math.log(theta2) * grid.bins_per_log
        host = np.array(
            [
                sample_gs_cos_theta_grid(
                    grid.values,
                    grid.ix0,
                    grid.iy0,
                    grid.values.shape[0],
                    grid.values.shape[1],
                    grid.n_u,
                    fx,
                    fy,
                    theta2,
                    rng,
                )
                for _ in range(N)
            ]
        )
        device = device_draws[:, column].astype(np.float64)

        edges = np.quantile(1.0 - host, np.linspace(0.0, 1.0, 21))
        edges[0], edges[-1] = 0.0, 2.0
        obs, _ = np.histogram(1.0 - device, bins=edges)
        exp, _ = np.histogram(1.0 - host, bins=edges)
        keep = (obs + exp) > 20
        chi2 = float(np.sum((obs[keep] - exp[keep]) ** 2 / (obs[keep] + exp[keep])))
        p = float(stats.chi2.sf(chi2, int(keep.sum()) - 1))
        assert p > 0.001, f"device vs host GS draws: chi2={chi2:.1f} p={p:.2e}"


class TestStreamParity:
    """One uniform per hinge, both models, verified through the device RNG itself."""

    def test_next_uniform_is_identical_after_either_model(self, device_draws: np.ndarray) -> None:
        """From identical RNG states, the draw *after* the sampler is bit-equal
        between the Gaussian hinge and GS — the two consumed identically. This
        is the device-side form of the one-uniform contract; if it ever breaks,
        every paired GS-vs-Gaussian comparison on this backend measures stream
        divergence, not physics."""
        np.testing.assert_array_equal(device_draws[:, 3], device_draws[:, 5])
