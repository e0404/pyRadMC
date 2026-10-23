"""Demo: every control point of a 6 MV arc in one engine launch.

A VMAT arc delivers one fan from many gantry angles, each with its own MLC
aperture. :class:`~pyradmc.geometry.segments.SegmentedFanSource` given **one fan and
one beam frame per segment** makes each control point a beamlet of one source: a
single ``run_dij`` returns the dose of every control point, and a single ``run`` of
its :meth:`~pyradmc.geometry.segments.SegmentedFanSource.plan` view returns the
MU-weighted arc dose (scaled by the plan's ``total_weight``) — instead of one launch
per gantry angle. Physics: the head pre-solve's attenuation mode (narrow-beam MLC
transmission; no head scatter, no contaminant electrons, no air).

Setup: Varian 6 MV (Ali & Rogers 2012), SAD 100 cm, a full arc of 36 control points
(10 degrees apart, rotation about the y axis) around a 16 x 6.4 x 16 cm water box, a
10-pair focused-side MLC (5 mm leaves at the isocentre, 6 cm leaf height) of dense
water standing in for tungsten (no data dependency), and at every angle an aperture
conformed to the beam's-eye view of a 2 cm-radius cylindrical target placed 2 cm off
the isocentre, so the apertures slide across the field as the gantry turns.

The figure reads at a glance: the arc dose in the central axial slice (one plan
run), two single control points of the same arc (columns of one ``run_dij``), a
profile through the target of the one-launch arc against the MU-weighted sum of 36
separate per-angle runs, and the wall time of the three routes.
Run from the repository root::

    python examples/segmented_arc_demo.py [--histories N] [--angles A]

Renders ``segmented_arc_demo.png`` beside this script.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv
from pyradmc.rng.host import HostRNG

SAD = 100.0
SHAPE, SPACING = (64, 16, 64), (0.25, 0.4, 0.25)
ISO = tuple(0.5 * n * d for n, d in zip(SHAPE, SPACING, strict=True))  # box centre
FIELD_U, FIELD_V = 9.0, 5.0  # field at the isocentre: leaf travel x leaf width (cm)
N_PAIRS = 10
MLC_Z = (50.0, 56.0)
TO_MID = 0.5 * sum(MLC_Z) / SAD  # isocentre plane -> MLC mid-plane scaling
LEAF_EDGES = tuple(float(e) for e in np.linspace(-0.5, 0.5, N_PAIRS + 1) * FIELD_V * TO_MID)
TARGET_CENTRE = (ISO[0] + 2.0, ISO[2])  # (x, z), the cylinder runs along y
TARGET_RADIUS = 2.0
TARGET_LENGTH = 3.0  # along y, centred on the isocentre


def beam_axes(angle: float) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """``(u, v, w)`` at a gantry angle: ``w`` focal spot -> isocentre, ``v`` = +y."""
    a = math.radians(angle)
    return (math.cos(a), 0.0, -math.sin(a)), (0.0, 1.0, 0.0), (math.sin(a), 0.0, math.cos(a))


def focal(angle: float) -> tuple[float, float, float]:
    """Return the focal spot of the beam at a gantry angle."""
    w = beam_axes(angle)[2]
    return (ISO[0] - SAD * w[0], ISO[1] - SAD * w[1], ISO[2] - SAD * w[2])


def make_fan(spectrum, angle: float) -> SpectralBeamSource:
    """One control point's open fan: its aperture axes are its frame's u and v."""
    u, v, _ = beam_axes(angle)
    return SpectralBeamSource(spectrum, focal(angle), ISO, FIELD_U, FIELD_V, u_axis=u, v_axis=v)


def make_frame(angle: float) -> BeamFrame:
    """One control point's beam frame, origin on its focal spot."""
    u, v, w = beam_axes(angle)
    return BeamFrame(origin=focal(angle), u_axis=u, v_axis=v, w_axis=w)


def template() -> BeamLimitingStack:
    """Return the shared MLC structure (frame and tips are replaced per control point)."""
    return BeamLimitingStack(
        frame=make_frame(0.0),
        devices=(
            MLC(
                z_top=MLC_Z[0],
                z_bottom=MLC_Z[1],
                leaf_edges_v=LEAF_EDGES,
                tips_neg=(0.0,) * N_PAIRS,
                tips_pos=(0.0,) * N_PAIRS,
                tip_radius=8.0,
                material=WATER,
                density=19.3,
                focused_sides=True,
            ),
        ),
    )


def conformal_tips(angle: float) -> np.ndarray:
    """(pairs, 2) mid-plane tips covering the target's beam's-eye view at ``angle``."""
    u, _, w = beam_axes(angle)
    f = np.asarray(focal(angle))
    c = np.array([TARGET_CENTRE[0], ISO[1], TARGET_CENTRE[1]])
    depth = float((c - f) @ np.asarray(w))
    centre_u = float((c - f) @ np.asarray(u)) * SAD / depth  # at the isocentre plane
    half_u = TARGET_RADIUS * SAD / depth + 0.3
    rows = np.asarray(LEAF_EDGES) / TO_MID  # leaf boundaries at the isocentre
    tips = np.zeros((N_PAIRS, 2))
    for i in range(N_PAIRS):
        if max(abs(rows[i]), abs(rows[i + 1])) <= 0.5 * TARGET_LENGTH + 1e-9:
            tips[i] = (centre_u - half_u, centre_u + half_u)
    return np.clip(tips * TO_MID, -0.5 * FIELD_U * TO_MID, 0.5 * FIELD_U * TO_MID)


def make_engine(grid: VoxelGrid, xs):
    """Warp on CUDA if present, else Warp CPU, else the (slow) reference engine."""
    try:
        import warp as wp

        from pyradmc.backends.warp.engine import WarpEngine

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        return WarpEngine(grid=grid, cross_sections=xs, device=device), f"warp:{device}"
    except ImportError:
        from pyradmc.backends.ref.engine import ReferenceEngine

        return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()), "ref"


def main() -> None:
    """Build the arc, run it in one launch and per angle, render the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--histories", type=int, default=None, help="per control point (200k CUDA, 10k CPU)"
    )
    parser.add_argument("--angles", type=int, default=36)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()

    grid = VoxelGrid.uniform_water(shape=SHAPE, spacing=SPACING)
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    spectrum = ali_rogers_mv("varian-6mv")
    angles = np.arange(args.angles) * 360.0 / args.angles
    tips = np.stack([conformal_tips(a) for a in angles])
    mu = np.ones(args.angles)
    # One source for the whole arc: a fan and a frame per control point, one template.
    arc = SegmentedFanSource(
        [make_fan(spectrum, a) for a in angles],
        template(),
        xs,
        frames=[make_frame(a) for a in angles],
        tips_neg=tips[..., 0],
        tips_pos=tips[..., 1],
    )
    engine, backend = make_engine(grid, xs)
    n = args.histories or (200_000 if backend == "warp:cuda:0" else 10_000)
    if backend == "ref":
        n = min(n, 500)
        print("no Warp: the reference engine is slow, histories capped at 500")
    kw = dict(n_batches=args.batches, seed=args.seed)

    def timed(label: str, call):
        call()  # warm-up: kernel compilation and table caches
        t0 = time.perf_counter()
        out = call()
        wall = time.perf_counter() - t0
        print(f"{label}: {wall:.3f} s")
        return out, wall

    plan_source = arc.plan(mu)
    plan, t_plan = timed(
        f"one plan run, {args.angles} control points x {n:,}",
        lambda: engine.run(plan_source, n_histories=args.angles * n, **kw),
    )
    dij, t_dij = timed(
        f"one run_dij, {args.angles} columns x {n:,}",
        lambda: engine.run_dij(arc, n_histories_per_beamlet=n, truncation=0.0, **kw),
    )
    singles = [
        SegmentedFanSource(make_fan(spectrum, a), template(), xs, frames=[make_frame(a)],
                           tips_neg=tips[k, None, :, 0], tips_pos=tips[k, None, :, 1])
        for k, a in enumerate(angles)
    ]  # fmt: skip

    def per_angle():
        return [
            engine.run(s.plan([1.0]), n_histories=n, n_batches=args.batches, seed=args.seed + k)
            for k, s in enumerate(singles)
        ]

    runs, t_runs = timed(f"{args.angles} per-angle runs x {n:,}", per_angle)
    print(f"({backend}; best of one warm call each, not a benchmark)")

    # ------------------------------------------------------------------ figure
    iy = SHAPE[1] // 2
    x = (np.arange(SHAPE[0]) + 0.5) * SPACING[0] - ISO[0]
    z = (np.arange(SHAPE[2]) + 0.5) * SPACING[2] - ISO[2]
    extent = (x[0], x[-1], z[0], z[-1])
    arc_dose = plan_source.total_weight * plan.dose
    arc_sigma = plan_source.total_weight * plan.dose_sigma
    summed = sum(w * r.dose for w, r in zip(mu, runs, strict=True))
    summed_sigma = np.sqrt(sum((w * r.dose_sigma) ** 2 for w, r in zip(mu, runs, strict=True)))
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    target_x = TARGET_CENTRE[0] - ISO[0] + TARGET_RADIUS * np.cos(theta)
    target_z = TARGET_CENTRE[1] - ISO[2] + TARGET_RADIUS * np.sin(theta)

    fig, axes = plt.subplots(1, 5, figsize=(22.0, 4.8))
    ax = axes[0]
    ax.imshow(arc_dose[:, iy, :].T, origin="lower", extent=extent, cmap="magma")
    ax.plot(target_x, target_z, color="#7fd4ff", linewidth=1.0)
    ax.set_title(f"arc dose, {args.angles} control points\n(one run of the plan view)")
    for k, ax in zip((0, args.angles // 4), axes[1:3], strict=True):
        column = dij.column_dense(k).reshape(SHAPE)
        ax.imshow(column[:, iy, :].T, origin="lower", extent=extent, cmap="magma")
        ax.plot(target_x, target_z, color="#7fd4ff", linewidth=1.0)
        ax.set_title(f"control point at {angles[k]:.0f} deg\n(column {k} of one run_dij)")
    for ax in axes[:3]:
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("z (cm)")
    ax = axes[3]
    iz = round(TARGET_CENTRE[1] / SPACING[2] - 0.5)
    ax.errorbar(x, arc_dose[:, iy, iz], yerr=arc_sigma[:, iy, iz], color="#d8853b",
                label="one arc launch")  # fmt: skip
    ax.plot(x, summed[:, iy, iz], color="#2b5fa8", label=f"sum of {args.angles} runs")
    ax.fill_between(x, (summed - summed_sigma)[:, iy, iz], (summed + summed_sigma)[:, iy, iz],
                    color="#2b5fa8", alpha=0.25)  # fmt: skip
    ax.axvspan(*(TARGET_CENTRE[0] - ISO[0] + np.array([-1, 1]) * TARGET_RADIUS), color="0.9")
    ax.set_title("profile through the target (z = target centre)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("dose per control-point history (MeV/g)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend()
    ax = axes[4]
    labels = (f"{args.angles} per-angle\nruns", "one run_dij", "one plan run")
    walls = (t_runs, t_dij, t_plan)
    ax.bar(labels, walls, color=("#9aa5b1", "#2b5fa8", "#d8853b"))
    for i, wall in enumerate(walls):
        ax.text(i, wall, f"{wall * 1e3:.0f} ms", ha="center", va="bottom")
    ax.set_ylabel("wall time, warm (s)")
    ax.set_title(f"same histories, {backend}")

    fig.suptitle("A 6 MV arc: every control point in one launch")
    fig.tight_layout()
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=120)
    print(f"saved {stem.name}.png")


if __name__ == "__main__":
    main()
