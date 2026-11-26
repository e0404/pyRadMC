"""Stochastic Dij truncation on the device: the count and compaction passes.

The ``truncation_mode="stochastic"`` counterpart of the chunked hard-truncation
kernels in :mod:`pyradmc.backends.warp.kernels` (``count_kept_per_chunk``,
``compact_column_chunked``): the same tiling of each column into chunks, the same
per-column ``col_max`` and float64 threshold product ``t = truncation * col_max``,
the same exclusive-prefix write cursors — only the keep rule differs. Each entry goes
through :func:`pyradmc.physics.roulette.roulette_dij_entry`, the one definition the
host assembler applies too, compiled here in float64 (Russian roulette on the
sub-threshold tail; Thieke et al., Med. Phys. 29 (2002) 676, doi:10.1118/1.1469633).

The roulette's uniform for voxel ``v`` of column ``j`` is
``keyed_uniform(seed ^ DIJ_ROULETTE_SALT, j, v)``
(:func:`pyradmc.rng.warp_shim.keyed_uniform`), with ``j`` the *global* beamlet
index (the group's first column plus the local one). It is therefore a pure function
of ``(seed, j, v)``: the count pass and the compaction pass draw the same uniform for
the same entry, and nothing — chunk layout, beamlet group size, batch lanes, the
device a group was sharded to — can change a decision. It shares nothing with the
host assembler's stream (AGENTS.md 2.3): host and device apply the same rule to
independent draws, so their stochastic matrices agree statistically, never bitwise.

Approximation stated at the point of implementation: ``wp.randf`` has 24 bits, so
the keep probability ``m / t`` is effectively compared on a grid of 2^-24; the
stored value's expectation is ``m`` to within ``t * 2^-24`` (relative 6e-8 of the
threshold), against exactly ``m`` with the host's 53-bit draws.

Module options: no ``fast_math``, unlike the transport kernels. The two passes must
reach the same decision for every entry, or the compaction writes outside the
extents the count pass sized; under IEEE semantics ``u < m / t`` is evaluated the
same way in both kernels, whereas fast-math licenses each kernel's compilation to
rewrite the division (e.g. as a reciprocal multiply) independently.

This module is Warp kernel code: exempt from ``mypy --strict`` (AGENTS.md 5) and
importable only when the ``warp`` optional dependency is installed.
"""

from __future__ import annotations

import warp as wp

from pyradmc.backends.warp.physics import warp_physics
from pyradmc.rng.warp_shim import keyed_uniform

__all__ = ["compact_column_chunked_stochastic", "count_kept_per_chunk_stochastic"]

wp.set_module_options({"enable_backward": False})

roulette_dij_entry = warp_physics().roulette_dij_entry


@wp.func
def _stochastic_entry(
    mean: wp.float64,
    sigma: wp.float64,
    threshold: wp.float64,
    roulette_seed: int,
    column: int,
    voxel: int,
):
    """Return one entry's stored ``(value, sigma)``: the decision both passes make.

    The uniform is drawn only where the rule consults it (a tail entry), which is
    a cost saving, not a change: elsewhere the rule ignores ``u``.
    """
    u = wp.float64(0.0)
    if mean > wp.float64(0.0) and mean < threshold:
        u = wp.float64(keyed_uniform(roulette_seed, column, voxel))
    return roulette_dij_entry(mean, sigma, threshold, u)


@wp.kernel
def count_kept_per_chunk_stochastic(
    mean: wp.array(dtype=wp.float64),
    sigma: wp.array(dtype=wp.float64),
    n_voxels: int,
    n_chunks: int,
    chunk: int,
    truncation: wp.float64,
    col_max: wp.array(dtype=wp.float64),
    column_offset: int,
    roulette_seed: int,
    chunk_counts: wp.array(dtype=wp.int32),
):
    """Count the entries one chunk of one column stores under the roulette."""
    tid = wp.tid()
    local = tid // n_chunks
    start = (tid - local * n_chunks) * chunk
    stop = min(start + chunk, n_voxels)
    base = local * n_voxels
    thr = truncation * col_max[local]
    column = column_offset + local
    c = int(0)  # noqa: UP018, RUF046 - Warp needs a dynamic int var for the loop counter
    for v in range(start, stop):
        value, _s = _stochastic_entry(
            mean[base + v], sigma[base + v], thr, roulette_seed, column, v
        )
        if value > wp.float64(0.0):
            c += 1
    chunk_counts[tid] = c


@wp.kernel
def compact_column_chunked_stochastic(
    mean: wp.array(dtype=wp.float64),
    sigma: wp.array(dtype=wp.float64),
    n_voxels: int,
    n_chunks: int,
    chunk: int,
    truncation: wp.float64,
    col_max: wp.array(dtype=wp.float64),
    column_offset: int,
    roulette_seed: int,
    chunk_base: wp.array(dtype=wp.int32),
    out_indices: wp.array(dtype=wp.int64),
    out_dose: wp.array(dtype=wp.float64),
    out_sigma: wp.array(dtype=wp.float64),
):
    """Scatter one chunk's stored entries into the compact CSC arrays, ascending."""
    tid = wp.tid()
    local = tid // n_chunks
    start = (tid - local * n_chunks) * chunk
    stop = min(start + chunk, n_voxels)
    base = local * n_voxels
    thr = truncation * col_max[local]
    column = column_offset + local
    cursor = chunk_base[tid]
    for v in range(start, stop):
        value, stored = _stochastic_entry(
            mean[base + v], sigma[base + v], thr, roulette_seed, column, v
        )
        if value > wp.float64(0.0):
            out_indices[cursor] = wp.int64(v)
            out_dose[cursor] = value
            out_sigma[cursor] = stored
            cursor += 1
