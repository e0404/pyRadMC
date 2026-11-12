"""Energy grids whose nodes do not depend on how far up they reach.

A geometric grid built as ``geomspace(lo, e_max, n)`` moves *every* node when
``e_max`` changes, so widening a table's energy range to serve a higher-energy
source would silently perturb the interpolation — and hence the results — of
every lower-energy run that never needed the extra range. The grids here are
instead defined by a *reference span* ``[lo, e_ref]`` with ``n_ref`` nodes, which
fixes the log spacing, and are extended above ``e_ref`` by appending nodes at that
same spacing. Every node at or below ``e_ref`` is then bit-identical to
``geomspace(lo, e_ref, n_ref)`` regardless of the requested top, and a wider grid
only ever *adds* nodes.

Pure host-side construction; no kernel sees this module.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

__all__ = ["extendable_log_grid"]


def extendable_log_grid(
    e_lo: float, e_ref: float, n_ref: int, e_top: float
) -> npt.NDArray[np.float64]:
    """Geometric grid over ``[e_lo, max(e_ref, e_top)]`` with append-only extension.

    Parameters
    ----------
    e_lo, e_ref, n_ref
        The reference span and its node count: the first ``n_ref`` nodes are exactly
        ``np.geomspace(e_lo, e_ref, n_ref)``, which also fixes the log spacing
        ``ln(e_ref / e_lo) / (n_ref - 1)``.
    e_top
        The energy the grid must reach. At or below ``e_ref`` the reference span is
        returned unchanged (a grid never shrinks below its reference, so nodes stay
        put); above it, nodes ``e_ref * exp(k * dlog)`` are appended up to and
        including the first one at or above ``e_top``.
    """
    if not 0.0 < e_lo < e_ref:
        raise ValueError(f"reference span must satisfy 0 < e_lo < e_ref, got {e_lo}, {e_ref}")
    if n_ref < 2:
        raise ValueError(f"reference span needs at least two nodes, got {n_ref}")
    base = np.geomspace(e_lo, e_ref, n_ref)
    if e_top <= e_ref:
        return base
    dlog = math.log(e_ref / e_lo) / (n_ref - 1)
    # Round up, and guard the float division against landing one node short.
    n_extra = math.ceil(math.log(e_top / e_ref) / dlog - 1.0e-9)
    if e_ref * math.exp(n_extra * dlog) < e_top:
        n_extra += 1
    extra = e_ref * np.exp(dlog * np.arange(1, n_extra + 1, dtype=np.float64))
    return np.concatenate((base, extra))
