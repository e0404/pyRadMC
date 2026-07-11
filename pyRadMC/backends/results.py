"""Backend-agnostic transport results.

Every backend returns the same result type from ``run``, so tests and callers compare
backends without caring which engine produced what. Defined here, above the backend
subpackages, because a result is not backend code (AGENTS.md section 3: backends hold
launch and memory management only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TransportResult"]


@dataclass(frozen=True)
class TransportResult:
    """One engine run: batched dose estimate plus exact energy bookkeeping.

    ``energy_emitted == energy_deposited + energy_escaped`` holds to accumulation
    precision of the producing backend — float64 exact for ``ref``, float32
    transport arithmetic plus scoring quantization for ``warp`` — and is asserted
    in the integration tier at each backend's documented tolerance.

    ``energy_escaped`` is a *ledger*, not purely physical escape: since Phase 3 it
    also carries the net weight-energy Russian roulette removes from the transported
    population (kills positive, survivor boosts negative), which is exactly what
    keeps the identity above exact per run under variance reduction.
    """

    dose: np.ndarray
    """Per-voxel dose, MeV/g per emitted history."""
    dose_sigma: np.ndarray
    """Per-voxel 1-sigma standard error from batch statistics."""
    energy_emitted: float
    energy_deposited: float
    energy_escaped: float
    n_histories: int
    n_batches: int
