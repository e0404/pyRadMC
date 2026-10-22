"""Demo: every segment of a 6 MV fan in one engine launch, dose per segment.

A step-and-shoot field is one photon fan delivered through several MLC settings.
:class:`~pyradmc.geometry.segments.SegmentedFanSource` makes each setting a
beamlet, so a single ``run_dij`` returns one dose column per segment — instead of
one head pre-solve plus one transport run per segment — and its
:meth:`~pyradmc.geometry.segments.SegmentedFanSource.plan` view returns the
MU-weighted plan dose from a single ``run`` (scaled by the plan's ``total_weight``,
the total MU). Physics: the head pre-solve's
attenuation mode (narrow-beam MLC transmission; no head scatter, no contaminant
electrons).

Setup: Varian 6 MV (Ali & Rogers 2012) from a point focal spot 100 cm above a
12 x 10 x 12 cm water phantom, a 20-pair focused-side MLC (5 mm leaves at the
surface, 6 cm leaf height, 8 cm tip radius), four segments: an open 8 x 8 cm
square, its left half, a C-shape with a closed middle, and a 2 x 2 cm off-axis
square.

The figure reads at a glance: the top row is each segment's beam's-eye-view
transmission (computed from the same source, at 2 MeV), the bottom row its dose
at 5 cm depth, and the last column the plan dose of one ``run`` against the
weighted sum of the four columns. The thin stripe at x = 0 across closed rows is
leakage where two closed rounded tips touch at their apices: geometry, not noise.

With the EPICS libraries available (cached or downloadable) the leaves are real
tungsten; ``--material water`` uses a dense-water stand-in with no data dependency.
Run from the repository root::

    python examples/segmented_fan_demo.py [--histories N] [--material M]

Renders ``segmented_fan_demo.png`` beside this script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyradmc.data.materials import TUNGSTEN, WATER
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv
from pyradmc.rng.host import HostRNG

SAD = 100.0
CENTRE = (6.0, 5.0)  # beam axis at the surface (engine x, y)
FIELD = 10.0  # fan field at the surface (cm)
N_PAIRS = 20
MLC_Z = (50.0, 56.0)
TO_MID = 0.5 * sum(MLC_Z) / SAD  # surface -> MLC mid-plane scaling
MU = np.array([1.0, 2.0, 1.0, 3.0])
NAMES = ("open 8 x 8", "left half", "C-shape", "2 x 2 off-axis")


def segment_tips() -> np.ndarray:
    """(4, pairs, 2) leaf tips at the MLC mid-plane, stated at the surface first."""
    rows = (np.arange(N_PAIRS) + 0.5) / N_PAIRS * FIELD - FIELD / 2  # leaf centres (surface)
    tips = np.zeros((4, N_PAIRS, 2))
    in_field = np.abs(rows) < 4.0
    tips[0, in_field] = (-4.0, 4.0)
    tips[1, in_field] = (-4.0, 0.0)
    middle = np.abs(rows) < 1.5
    tips[2, in_field] = (-4.0, 4.0)
    tips[2, middle] = (1.0, 4.0)  # the C: a closed bar left of x = 1 in the middle rows
    small = (rows > 0.5) & (rows < 2.5)
    tips[3, small] = (1.0, 3.0)
    return tips * TO_MID


def template(material: int, density: float) -> BeamLimitingStack:
    """Build the shared MLC structure (its tips are only segment 0's defaults)."""
    edges = tuple(float(e) for e in np.linspace(-0.5, 0.5, N_PAIRS + 1) * FIELD * TO_MID)
    tips = segment_tips()[0]
    return BeamLimitingStack(
        frame=BeamFrame(origin=(CENTRE[0], CENTRE[1], -SAD)),
        devices=(
            MLC(
                z_top=MLC_Z[0],
                z_bottom=MLC_Z[1],
                leaf_edges_v=edges,
                tips_neg=tuple(float(t) for t in tips[:, 0]),
                tips_pos=tuple(float(t) for t in tips[:, 1]),
                tip_radius=8.0,
                material=material,
                density=density,
                focused_sides=True,
            ),
        ),
    )


def make_cross_sections(material_choice: str, grid: VoxelGrid):
    """Compiled EPICS table for tungsten leaves, or analytic water (dense stand-in)."""
    if material_choice == "tungsten":
        from pyradmc.data.tabulated.build import download_library
        from pyradmc.data.tabulated.precompile import compile_materials
        from pyradmc.data.tabulated.source import TabulatedCrossSections

        print("compiling the tabulated table (EPICS cache/download) ...")
        epdl = download_library("epdl").read_text(encoding="latin-1")
        eedl = download_library("eedl").read_text(encoding="latin-1")
        data = compile_materials(epdl, eedl, e_max=7.0)
        xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
        return xs, TUNGSTEN, 19.3
    from pyradmc.data.analytic import AnalyticCrossSections

    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material()), WATER, 19.3


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


def bev_transmission(source: SegmentedFanSource, segment: int):
    """Transmission at 2 MeV of the focal-spot rays through a grid on the surface."""
    n = 120
    axis = np.linspace(-FIELD / 2, FIELD / 2, n)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    targets = np.stack([CENTRE[0] + gx.ravel(), CENTRE[1] + gy.ravel(), np.zeros(gx.size)], 1)
    focal = np.array(source.fan.focal_point)
    directions = targets - focal
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.tile(focal, (gx.size, 1))
    energies = np.full(gx.size, 2.0)
    return axis, source.transmission(segment, energies, origins, directions).reshape(n, n)


def main() -> None:
    """Build the segments, run one Dij and one plan run, render the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--histories", type=int, default=None, help="per segment (default 2M CUDA, 100k CPU)"
    )
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--material", choices=("tungsten", "water"), default="tungsten")
    args = parser.parse_args()

    grid = VoxelGrid.uniform_water(shape=(48, 40, 24), spacing=(0.25, 0.25, 0.5))
    xs, material, density = make_cross_sections(args.material, grid)
    fan = SpectralBeamSource(
        ali_rogers_mv("varian-6mv"),
        focal_point=(CENTRE[0], CENTRE[1], -SAD),
        center=(CENTRE[0], CENTRE[1], 0.0),
        width_u=FIELD,
        width_v=FIELD,
    )
    tips = segment_tips()
    # The cheap path: one template, per-segment tip arrays — no MLC object per segment.
    source = SegmentedFanSource(
        fan, template(material, density), xs, tips_neg=tips[..., 0], tips_pos=tips[..., 1]
    )
    engine, backend = make_engine(grid, xs)
    n = args.histories or (2_000_000 if backend == "warp:cuda:0" else 100_000)
    if backend == "ref":
        n = min(n, 4_000)
        print("no Warp: the reference engine is slow, histories capped at 4000")

    t0 = time.perf_counter()
    dij = engine.run_dij(
        source, n_histories_per_beamlet=n, n_batches=args.batches, seed=args.seed, truncation=0.0
    )
    print(f"one run_dij, {source.n_segments} segments x {n:,}: {time.perf_counter() - t0:.2f} s")
    t0 = time.perf_counter()
    plan_source = source.plan(MU)
    plan = engine.run(
        plan_source,
        n_histories=source.n_segments * n,
        n_batches=args.batches,
        seed=args.seed + 1,
    )
    print(f"one plan run, {source.n_segments * n:,} histories: {time.perf_counter() - t0:.2f} s")
    print(f"({backend}; first calls include kernel compilation; not a benchmark)")

    # ------------------------------------------------------------------ figure
    iz = int(5.0 / grid.spacing[2])
    x = (np.arange(grid.shape[0]) + 0.5) * grid.spacing[0] - CENTRE[0]
    y = (np.arange(grid.shape[1]) + 0.5) * grid.spacing[1] - CENTRE[1]
    extent = (x[0], x[-1], y[0], y[-1])
    columns = [dij.column_dense(s).reshape(grid.shape) for s in range(source.n_segments)]
    vmax = max(c[:, :, iz].max() for c in columns)

    fig, axes = plt.subplots(2, 5, figsize=(18.0, 7.2))
    for s, name in enumerate(NAMES):
        axis, bev = bev_transmission(source, s)
        ax = axes[0, s]
        ax.imshow(
            bev.T,
            origin="lower",
            extent=(axis[0], axis[-1], axis[0], axis[-1]),
            cmap="cividis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"segment {s}: {name}\nBEV transmission, 2 MeV")
        ax.set_xlabel("x at surface (cm)")
        ax.set_ylabel("y at surface (cm)")
        ax = axes[1, s]
        ax.imshow(
            columns[s][:, :, iz].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax
        )
        ax.set_title(f"dose at 5 cm, {MU[s]:g} MU")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")

    weighted = sum(w * c for w, c in zip(MU, columns, strict=True))
    sigma = np.sqrt(
        sum(
            (w * dij.sigma_dense(s).reshape(grid.shape)) ** 2
            for s, w in enumerate(MU)  # columns are correlated: a lower bound, see AGENTS 8
        )
    )
    ax = axes[0, 4]
    # run() returns sum_s (MU_s / W) column_s per history; W = total MU scales it back.
    plan_dose = plan_source.total_weight * plan.dose
    plan_sigma = plan_source.total_weight * plan.dose_sigma
    ax.imshow(plan_dose[:, :, iz].T, origin="lower", extent=extent, cmap="magma")
    ax.set_title("plan dose at 5 cm\n(one run of the plan view)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax = axes[1, 4]
    iy = int((CENTRE[1] + 1.75) / grid.spacing[1])  # a row through the C's closed bar
    ax.errorbar(
        x, plan_dose[:, iy, iz], yerr=plan_sigma[:, iy, iz], color="#d8853b", label="plan run"
    )
    ax.plot(x, weighted[:, iy, iz], color="#2b5fa8", label="sum MU x column")
    ax.fill_between(
        x,
        weighted[:, iy, iz] - sigma[:, iy, iz],
        weighted[:, iy, iz] + sigma[:, iy, iz],
        color="#2b5fa8",
        alpha=0.25,
    )
    ax.set_title(f"profile at y = {y[iy]:.2f} cm, 5 cm depth")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("dose per history (MeV/g)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend()

    fig.suptitle(f"Four segments of a 6 MV fan in one launch ({args.material} leaves, {backend})")
    fig.tight_layout()
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=130)
    print(f"saved {stem.name}.png")


if __name__ == "__main__":
    main()
