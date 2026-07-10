"""Kernel-side RNG shim for the Warp backend (cpu and cuda targets).

The stream is warp's builtin per-thread PRNG, exactly as documented in
:mod:`pyRadMC.rng.interface`: ``wp.rand_init(seed, history_index)`` seeds it
counter-based per history, ``wp.randf`` draws from [0, 1) with 24-bit resolution.
This module only packages that state so the single-source physics functions can
consume it unchanged.

Why the struct: Warp user functions (``@wp.func``) receive scalar arguments **by
value** — a bare ``wp.uint32`` state would not advance across the ``uniform(state)``
call boundary, and every draw would repeat (probed, not assumed; warp 1.15). Array
*element* writes do cross that boundary, so the state lives in a per-particle slot
of a ``uint32`` array and the struct carries the (array, index) pair. The cost is a
load/store to that slot per draw; that is the price of physics source that compiles
identically for ``ref``, ``cpu`` and ``cuda`` (AGENTS.md 2.5), and it touches one
cached word per thread.

The shim's stream is contract-tested to be bit-identical (within one device) to
bare ``wp.rand_init``/``wp.randf``; see ``tests/unit/test_warp_rng.py``.

This module is Warp kernel code: exempt from ``mypy --strict`` (AGENTS.md 5) and
importable only when the ``warp`` optional dependency is installed.
"""

from __future__ import annotations

import warp as wp

__all__ = ["WarpRNGState", "init_slot", "spawn_stream", "uniform"]


@wp.struct
class WarpRNGState:
    """Opaque per-particle RNG state: a slot index into a uint32 state array.

    Physics routines annotate their parameter ``RNGState`` and pass it to
    ``uniform`` unopened, exactly as with the host backend's ``numpy`` generator.
    """

    slots: wp.array(dtype=wp.uint32)
    idx: int


@wp.func
def init_slot(seed: int, history_index: int) -> wp.uint32:
    """Seed one history's stream: a pure function of ``(seed, history_index)``.

    Counter-based (AGENTS.md 2.3): independent of thread assignment, launch size,
    and batch decomposition — contract-tested.
    """
    return wp.rand_init(seed, history_index)


@wp.func
def uniform(state: WarpRNGState) -> float:
    """Draw from U[0, 1) — the only RNG call physics routines make.

    ``wp.randf`` maps 24 random bits onto [0, 1); it cannot produce 1.0, which the
    transport loop requires (see :class:`pyRadMC.rng.interface.RNG.uniform`).
    """
    s = state.slots[state.idx]
    r = wp.randf(s)
    state.slots[state.idx] = s
    return r


@wp.func
def spawn_stream(state: WarpRNGState) -> wp.uint32:
    """Derive a child stream for a spawned secondary from the parent's stream.

    Queued particles are transported concurrently, so a secondary cannot continue
    its parent's sequential stream the way the reference backend's stack does; it
    gets its own state, seeded from two parent draws. The child stream is thereby
    a deterministic function of ``(seed, history_index)`` and the parent's draw
    history, which preserves within-device run-to-run bit reproducibility no
    matter how kernels schedule the queue.
    """
    s = state.slots[state.idx]
    child = wp.rand_init(wp.randi(s), wp.randi(s))
    state.slots[state.idx] = s
    return child
