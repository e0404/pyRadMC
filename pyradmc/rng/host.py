"""Host-side RNG: one NumPy ``Generator`` per history, counter-based seeding.

The implementation behind the ``ref`` backend (and any host-side sampling test). The
per-history state is a ``numpy.random.Generator`` over PCG64, constructed from a
``SeedSequence`` keyed on ``(seed, history_index)``. That makes the stream a pure
function of those two integers — independent of creation order, batching, or thread
assignment — which is the reproducibility contract of AGENTS.md section 2.3.

``uniform`` draws ``Generator.random()``, a float64 in the half-open interval [0, 1):
53-bit mantissa construction cannot round to 1.0. Both properties are contract-tested
in ``tests/unit/test_interface_contracts.py``.
"""

from __future__ import annotations

import numpy as np

from pyradmc.rng.interface import RNG

__all__ = ["HostRNG", "uniform"]


def uniform(state: np.random.Generator) -> float:
    """Draw one float64 from U[0, 1) — the only RNG call physics routines make.

    Module-level rather than a method so that physics code reads ``uniform(state)``
    identically across targets; kernel targets bind their own ``uniform``
    (:mod:`pyradmc.rng.warp_shim`) at kernel compile time and never import this one.
    """
    return float(state.random())


class HostRNG(RNG):
    """Counter-based NumPy RNG; see the module docstring."""

    def init_state(self, seed: int, history_index: int) -> np.random.Generator:
        """Create the generator for one history as a pure function of the arguments.

        ``SeedSequence(seed, spawn_key=(history_index,))`` hashes both integers into
        the PCG64 state, so histories neither share nor overlap streams regardless of
        how they are partitioned into batches.
        """
        seed_seq = np.random.SeedSequence(entropy=seed, spawn_key=(history_index,))
        return np.random.Generator(np.random.PCG64(seed_seq))

    def uniform(self, state: np.random.Generator) -> float:
        """Draw from U[0, 1); delegates to the module-level :func:`uniform`."""
        return uniform(state)
