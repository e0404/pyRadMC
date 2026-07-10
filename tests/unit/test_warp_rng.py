"""Contract tests for the Warp kernel-side RNG shim.

The same contract the host RNG satisfies (``tests/unit/test_interface_contracts.py``),
restated for kernel-side state: draws in the half-open interval [0, 1), streams that
are pure functions of ``(seed, history_index)`` independent of launch decomposition,
and — specific to the shim — bit-identity with warp's builtin stream, so that wrapping
the state in a struct can never silently change the random sequence.

Reproducibility assertions here are all *within* one device, per AGENTS.md 2.3.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]


def _skip_without_device(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")


@pytest.fixture(scope="module")
def kernels():
    """Compile the test kernels once for the module."""
    from pyRadMC.rng.warp_shim import WarpRNGState, init_slot, uniform

    @wp.kernel
    def draw_many(
        seed: int,
        n_draws: int,
        slots: wp.array(dtype=wp.uint32),
        out: wp.array2d(dtype=float),
    ):
        tid = wp.tid()
        slots[tid] = init_slot(seed, tid)
        state = WarpRNGState()
        state.slots = slots
        state.idx = tid
        for j in range(n_draws):
            out[tid, j] = uniform(state)

    @wp.kernel
    def draw_one_history(
        seed: int,
        history: int,
        slots: wp.array(dtype=wp.uint32),
        out: wp.array(dtype=float),
    ):
        slots[0] = init_slot(seed, history)
        state = WarpRNGState()
        state.slots = slots
        state.idx = 0
        for j in range(out.shape[0]):
            out[j] = uniform(state)

    @wp.kernel
    def draw_builtin(seed: int, history: int, out: wp.array(dtype=float)):
        s = wp.rand_init(seed, history)
        for j in range(out.shape[0]):
            out[j] = wp.randf(s)

    return draw_many, draw_one_history, draw_builtin


@pytest.mark.parametrize("device", DEVICES)
def test_uniform_in_half_open_interval(kernels, seed: int, device: str) -> None:
    """No draw may be negative, and none may round to exactly 1.0.

    An exact 1.0 gives an infinite Woodcock path and a division by zero in the Kahn
    loop; float32 rounding-to-one is the classic failure of 32-bit generators. 1e7
    draws through the actual shim the physics routines call.
    """
    _skip_without_device(device)
    draw_many, _, _ = kernels
    n_threads, n_draws = 10_000, 1_000
    slots = wp.zeros(n_threads, dtype=wp.uint32, device=device)
    out = wp.zeros((n_threads, n_draws), dtype=float, device=device)
    wp.launch(draw_many, dim=n_threads, inputs=[seed, n_draws, slots], outputs=[out], device=device)
    u = out.numpy()
    assert np.all(u >= 0.0)
    assert np.all(u < 1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_shim_stream_matches_builtin(kernels, seed: int, device: str) -> None:
    """The struct wrapper must not alter warp's builtin stream in any way.

    If the shim ever draws differently from bare ``wp.rand_init``/``wp.randf`` —
    an extra draw, a dropped draw, a reordering — every downstream statistical
    equivalence still passes while the documented seeding contract silently lies.
    Within one device this is exact bit equality, which AGENTS.md 2.3 requires.
    """
    _skip_without_device(device)
    _, draw_one_history, draw_builtin = kernels
    for history in (0, 1, 12345):
        slots = wp.zeros(1, dtype=wp.uint32, device=device)
        via_shim = wp.zeros(64, dtype=float, device=device)
        direct = wp.zeros(64, dtype=float, device=device)
        wp.launch(
            draw_one_history,
            dim=1,
            inputs=[seed, history, slots],
            outputs=[via_shim],
            device=device,
        )
        wp.launch(draw_builtin, dim=1, inputs=[seed, history], outputs=[direct], device=device)
        assert np.array_equal(via_shim.numpy(), direct.numpy())


@pytest.mark.parametrize("device", DEVICES)
def test_stream_independent_of_launch_partitioning(kernels, seed: int, device: str) -> None:
    """Counter-based seeding: the stream depends on (seed, history_index) only.

    History ``i`` drawn inside a wide launch must reproduce, bit for bit on the same
    device, history ``i`` drawn alone in a dim-1 launch. This is what makes a result
    independent of batch decomposition and thread scheduling.
    """
    _skip_without_device(device)
    draw_many, draw_one_history, _ = kernels
    n_threads, n_draws = 8, 16
    slots = wp.zeros(n_threads, dtype=wp.uint32, device=device)
    wide = wp.zeros((n_threads, n_draws), dtype=float, device=device)
    wp.launch(
        draw_many, dim=n_threads, inputs=[seed, n_draws, slots], outputs=[wide], device=device
    )
    wide_np = wide.numpy()

    for history in range(n_threads):
        slot = wp.zeros(1, dtype=wp.uint32, device=device)
        alone = wp.zeros(n_draws, dtype=float, device=device)
        wp.launch(
            draw_one_history, dim=1, inputs=[seed, history, slot], outputs=[alone], device=device
        )
        assert np.array_equal(alone.numpy(), wide_np[history])

    # Distinct histories must not share a stream.
    assert not np.array_equal(wide_np[0], wide_np[1])
