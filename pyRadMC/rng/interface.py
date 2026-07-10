"""Random number generation interface.

Physics routines never see which target they are running on. They call ``uniform(state)``
and nothing else; see AGENTS.md section 2.6.

This matters because the per-target RNG APIs are genuinely incompatible:

``ref``
    NumPy ``Generator``, one per history, seeded from a counter.
``warp`` (cpu and cuda)
    ``wp.rand_init`` / ``wp.randf``, per-thread state.
``numba`` (cuda)
    ``numba.cuda.random`` xoroshiro128p device state arrays.

Reproducibility contract, restated from AGENTS.md section 2.3:

- Within one target, one seed, one thread count: bit-reproducible.
- Across targets: **no reproducibility of any kind.** Not the stream, not the result.
  Do not write tests that assume otherwise.

The seeding scheme is counter-based per history rather than sequential, so that the
random stream a history sees does not depend on how histories are distributed over
threads. This is what makes within-target reproducibility independent of scheduling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["RNG"]


class RNG(ABC):
    """Per-history random number source.

    Implementations are thin. Anything richer than ``uniform`` that a physics routine
    needs (exponential, Gaussian, azimuthal angle) is *derived* in ``physics/`` from
    ``uniform``, so that all targets share one derivation and one set of tests.
    """

    @abstractmethod
    def init_state(self, seed: int, history_index: int) -> Any:
        """Create the state for one history.

        Counter-based: the stream is a pure function of ``(seed, history_index)`` and
        is independent of thread assignment or batch decomposition.
        """

    @abstractmethod
    def uniform(self, state: Any) -> float:
        """Draw from U[0, 1).

        The half-open interval matters. A returned exact 1.0 will produce a division by
        zero or an infinite path length in at least three places in the transport loop.
        Implementations must guarantee the exclusion and are contract-tested for it.
        """
