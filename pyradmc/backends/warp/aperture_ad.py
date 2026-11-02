"""Warp-taped chord gradients: the geometry of the pathwise aperture-gradient estimator.

The one module that differentiates with Warp's tape. It compiles the **same**
scalar collimation twins the device pre-solve uses
(:func:`pyradmc.geometry.collimation.jaw_path_length` /
:func:`~pyradmc.geometry.collimation.mlc_path_length`, through
:func:`~pyradmc.backends.warp.physics.warp_physics`) into kernels whose edge
coordinate is a **per-ray** array with ``requires_grad``, so one backward pass of
a launch over ``n`` rays yields ``n`` independent derivatives
``d(chord_i)/d(edge)``. The geometry stage is where the tape fits: no dynamic
step loop, no queues, no reused buffers — unlike the transport kernels, which
keep ``enable_backward=False`` (see :mod:`pyradmc.backends.warp.kernels`).

What the tape returns is the *pathwise* derivative: the derivative of each ray's
chord at fixed ray. Where the chord jumps (a straight face hit end-on) that is
zero, and the boundary term is lost — see
:mod:`pyradmc.geometry.aperture_gradient`. Float32, like every Warp kernel here.

This module is Warp kernel plumbing: exempt from ``mypy --strict`` (AGENTS.md 5).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from pyradmc.backends.warp.physics import warp_physics
from pyradmc.geometry.collimation import MLC, JawPair

__all__ = ["edge_chord_gradients"]

# This module exists to be differentiated: keep the adjoint kernels.
wp.set_module_options({"enable_backward": True})

_p = warp_physics()
jaw_path_length = _p.jaw_path_length
mlc_path_length = _p.mlc_path_length


@wp.kernel
def _jaw_chords(
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    q_axis: int,
    z_top: float,
    z_bottom: float,
    edge_neg: wp.array(dtype=float),
    edge_pos: wp.array(dtype=float),
    chords: wp.array(dtype=float),
):
    i = wp.tid()
    p = origins[i]
    d = directions[i]
    p_q = wp.where(q_axis == 0, p[0], p[1])
    d_q = wp.where(q_axis == 0, d[0], d[1])
    chords[i] = jaw_path_length(
        p_q, d_q, p[2], d[2], z_top, z_bottom, edge_neg[i], edge_pos[i], 0, 0
    )


@wp.kernel
def _mlc_chords(
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    z_top: float,
    z_bottom: float,
    tip_radius: float,
    leaf_edges_v: wp.array(dtype=float),
    tips_neg: wp.array(dtype=float),
    tips_pos: wp.array(dtype=float),
    chords: wp.array(dtype=float),
):
    i = wp.tid()
    p = origins[i]
    d = directions[i]
    # One leaf pair, addressed at tips offset i: each ray reads its own copy of the
    # tips, so the backward pass separates the per-ray derivatives.
    chords[i] = mlc_path_length(
        p[0], p[1], p[2], d[0], d[1], d[2], z_top, z_bottom, tip_radius, 1, 0, i,
        leaf_edges_v, tips_neg, tips_pos, 0, 0,
    )  # fmt: skip


def edge_chord_gradients(
    block: JawPair | MLC,
    side: str,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Chord through ``block`` and its derivative with respect to its ``side`` edge.

    ``block`` is an isolated block (:func:`~pyradmc.geometry.aperture_gradient.isolated_block`:
    a jaw pair or a one-pair, parallel-sided MLC). ``origins``/``directions`` are
    ``(n, 3)`` **local beam-frame** rays. Returns ``(chords, d_chords)``, both
    ``(n,)`` float64 arrays converted from the float32 kernel results.
    """
    if side not in ("pos", "neg"):
        raise ValueError(f"side must be 'pos' or 'neg', got {side!r}")
    n = len(origins)
    # Chords use the full-line convention, so moving each origin along its ray is
    # exact. Re-anchoring at the block mid-plane (in float64, before the float32
    # cast) keeps the kernel's ray parameters O(slab), not O(source distance):
    # otherwise the interval ends cancel to ~1e-3 cm near tangency.
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    z_mid = 0.5 * (block.z_top + block.z_bottom)
    d_w = directions[:, 2]
    shift = np.where(d_w != 0.0, (z_mid - origins[:, 2]) / np.where(d_w != 0.0, d_w, 1.0), 0.0)
    origins = origins + shift[:, None] * directions
    p = wp.array(origins.astype(np.float32), dtype=wp.vec3, device=device)
    d = wp.array(np.asarray(directions, dtype=np.float32), dtype=wp.vec3, device=device)
    chords = wp.zeros(n, dtype=float, device=device, requires_grad=True)

    def per_ray(value: float, grad: bool) -> wp.array:
        return wp.array(
            np.full(n, value, dtype=np.float32), dtype=float, device=device, requires_grad=grad
        )

    pos = side == "pos"
    tape = wp.Tape()
    if isinstance(block, MLC):
        if block.n_pairs != 1 or block.focused_sides:
            raise ValueError("pass an isolated one-pair, parallel-sided MLC block")
        neg_edges = per_ray(block.tips_neg[0], not pos)
        pos_edges = per_ray(block.tips_pos[0], pos)
        leaf_edges = wp.array(
            np.asarray(block.leaf_edges_v, dtype=np.float32), dtype=float, device=device
        )
        with tape:
            wp.launch(
                _mlc_chords,
                dim=n,
                inputs=[
                    p,
                    d,
                    block.z_top,
                    block.z_bottom,
                    block.tip_radius,
                    leaf_edges,
                    neg_edges,
                    pos_edges,
                ],
                outputs=[chords],
                device=device,
            )
    else:
        if block.focused:
            raise ValueError("pass an unfocused jaw block")
        neg_edges = per_ray(block.edge_neg, not pos)
        pos_edges = per_ray(block.edge_pos, pos)
        q_axis = 0 if block.axis == "u" else 1
        with tape:
            wp.launch(
                _jaw_chords,
                dim=n,
                inputs=[p, d, q_axis, block.z_top, block.z_bottom, neg_edges, pos_edges],
                outputs=[chords],
                device=device,
            )
    tape.backward(grads={chords: wp.ones(n, dtype=float, device=device)})
    edge = pos_edges if pos else neg_edges
    return chords.numpy().astype(np.float64), edge.grad.numpy().astype(np.float64)
