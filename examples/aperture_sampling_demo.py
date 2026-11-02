"""Demo: aperture importance sampling of a small 6 MV segment, against uniform sampling.

A segment of a :class:`~pyradmc.geometry.segments.SegmentedFanSource` samples its
fan's whole field uniformly by default, so a 2 x 2 cm opening in a 10 x 10 cm fan
transports 96 % of its histories through closed leaves, at a percent of
transmission each. ``aperture_sampling=delta`` draws a share ``1 - delta`` of the
histories inside the opening (widened by the rounded tips' partial-transmission
band) and multiplies the weight by the likelihood ratio, so the dose is unchanged in
expectation and its noise drops.

Setup: Varian 6 MV (Ali & Rogers 2012) from a point focal spot 100 cm above a
12 x 10 x 12 cm water phantom; focused tungsten jaws tracking the segment and a
20-pair focused-side MLC (5 mm leaves at the surface, 6 cm leaf height, 8 cm tip
radius) open on a 2 x 2 cm off-axis square, with the closed pairs meeting below it.

The figure: the beam's-eye-view transmission with the sampling region outlined, the
5 cm-depth dose of both runs at **equal histories**, and a profile with its 1 sigma
band, then the ratio of the two sigma maps. The numbers printed are the sigma and
wall-time ratios and the efficiency gain ``(sigma_u^2 t_u) / (sigma_a^2 t_a)``.

With the EPICS libraries available (cached or downloadable) the devices are real
tungsten; ``--material water`` uses a dense-water stand-in with no data dependency.
Run from the repository root::

    python examples/aperture_sampling_demo.py [--histories N] [--delta D] [--material M]

Renders ``aperture_sampling_demo.png`` beside this script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from pyradmc.data.materials import TUNGSTEN, WATER
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv
from pyradmc.rng.host import HostRNG

SAD = 100.0
CENTRE = (6.0, 5.0)  # beam axis at the surface (engine x, y)
FIELD = 10.0  # fan field at the surface (cm)
N_PAIRS = 20
JAW_V = (28.0, 31.9)  # the v (y) jaws, then the u (x) jaws, then the MLC (local w, cm)
JAW_U = (31.9, 35.8)
MLC_Z = (50.0, 56.0)
OPENING = (1.0, 3.0, 0.5, 2.5)  # u_lo, u_hi, v_lo, v_hi at the surface (cm)


def _to(z: tuple[float, float]) -> float:
    """Surface -> device mid-plane scaling (divergent, through the focal spot)."""
    return 0.5 * (z[0] + z[1]) / SAD


def make_cross_sections(material_choice: str, grid: VoxelGrid):
    """Compiled EPICS table for tungsten devices, or analytic water (dense stand-in)."""
    if material_choice == "tungsten":
        from pyradmc.data.tabulated.build import download_library
        from pyradmc.data.tabulated.precompile import compile_materials
        from pyradmc.data.tabulated.source import TabulatedCrossSections

        print("compiling the tabulated table (EPICS cache/download) ...")
        epdl = download_library("epdl").read_text(encoding="latin-1")
        eedl = download_library("eedl").read_text(encoding="latin-1")
        data = compile_materials(epdl, eedl, e_max=7.0)
        xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
        return xs, TUNGSTEN
    from pyradmc.data.analytic import AnalyticCrossSections

    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material()), WATER


def make_source(xs, material: int, delta: float | None) -> SegmentedFanSource:
    """One segment: the opening, jaws 5 mm outside it, closed pairs meeting below it."""
    u_lo, u_hi, v_lo, v_hi = OPENING
    rows = (np.arange(N_PAIRS) + 0.5) / N_PAIRS * FIELD - FIELD / 2  # leaf centres (surface)
    is_open = (rows > v_lo) & (rows < v_hi)
    tips_neg = np.full(N_PAIRS, 0.5 * (u_lo + u_hi))
    tips_pos = tips_neg.copy()
    tips_neg[is_open], tips_pos[is_open] = u_lo, u_hi
    k = _to(MLC_Z)
    edges = tuple(float(e) for e in np.linspace(-0.5, 0.5, N_PAIRS + 1) * FIELD * k)
    template = BeamLimitingStack(
        frame=BeamFrame(origin=(CENTRE[0], CENTRE[1], -SAD)),
        devices=(
            JawPair("v", *JAW_V, (v_lo - 0.5) * _to(JAW_V), (v_hi + 0.5) * _to(JAW_V),
                    material, 19.3, focused=True),
            JawPair("u", *JAW_U, (u_lo - 0.5) * _to(JAW_U), (u_hi + 0.5) * _to(JAW_U),
                    material, 19.3, focused=True),
            MLC(MLC_Z[0], MLC_Z[1], edges, tuple(tips_neg * k), tuple(tips_pos * k),
                tip_radius=8.0, material=material, density=19.3, focused_sides=True),
        ),
    )  # fmt: skip
    fan = SpectralBeamSource(
        ali_rogers_mv("varian-6mv"),
        focal_point=(CENTRE[0], CENTRE[1], -SAD),
        center=(CENTRE[0], CENTRE[1], 0.0),
        width_u=FIELD,
        width_v=FIELD,
    )
    return SegmentedFanSource.from_stacks(fan, [template], xs, aperture_sampling=delta)


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


def timed(engine, source, n: int, batches: int, seed: int, repeats: int):
    """Best-of-``repeats`` wall time of one Dij run (the result of the first)."""
    result, best = None, np.inf
    for r in range(repeats):
        t0 = time.perf_counter()
        dij = engine.run_dij(
            source, n_histories_per_beamlet=n, n_batches=batches, seed=seed + r, truncation=0.0
        )
        best = min(best, time.perf_counter() - t0)
        if result is None:
            result = dij
    return result, best


def main() -> None:
    """Run both samplers at equal histories and render the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--histories", type=int, default=None, help="default 2M CUDA, 200k CPU")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.1, help="defensive uniform share")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--material", choices=("tungsten", "water"), default="tungsten")
    args = parser.parse_args()

    grid = VoxelGrid.uniform_water(shape=(48, 40, 24), spacing=(0.25, 0.25, 0.5))
    xs, material = make_cross_sections(args.material, grid)
    uniform = make_source(xs, material, None)
    sampled = make_source(xs, material, args.delta)
    engine, backend = make_engine(grid, xs)
    n = args.histories or (2_000_000 if backend == "warp:cuda:0" else 200_000)
    repeats = 3
    if backend == "ref":
        n, repeats = min(n, 4_000), 1
        print("no Warp: the reference engine is slow, histories capped at 4000")
    engine.run_dij(uniform, n_histories_per_beamlet=1_000, n_batches=args.batches, seed=1)
    engine.run_dij(sampled, n_histories_per_beamlet=1_000, n_batches=args.batches, seed=1)

    dij_u, t_u = timed(engine, uniform, n, args.batches, args.seed, repeats)
    dij_a, t_a = timed(engine, sampled, n, args.batches, args.seed + 100, repeats)
    dose_u = dij_u.column_dense(0).reshape(grid.shape)
    dose_a = dij_a.column_dense(0).reshape(grid.shape)
    sig_u = dij_u.sigma_dense(0).reshape(grid.shape)
    sig_a = dij_a.sigma_dense(0).reshape(grid.shape)
    # Regions and the reference level from the *pooled* dose: either run's own maximum
    # would favour that run's upward fluctuations.
    pooled = 0.5 * (dose_u + dose_a)
    top = pooled.max()
    field = pooled > 0.5 * top
    level = float(pooled[field].mean())
    regions = (
        ("open field (> 50 %)", field),
        ("near field (2-20 %)", (pooled > 0.02 * top) & (pooled < 0.2 * top)),
        ("far, leakage only (< 2 %)", (pooled > 0.0) & (pooled < 0.02 * top)),
    )

    def rms(sigma: np.ndarray, region: np.ndarray) -> float:
        return float(np.sqrt(np.mean(sigma[region] ** 2)) / level)

    print(f"{backend}, {n:,} histories each, region |A|/|F| = {sampled.aperture_fraction[0]:.3f}")
    for name, region in regions:
        su, sa = rms(sig_u, region), rms(sig_a, region)
        gain = su**2 * t_u / (sa**2 * t_a)
        print(f"  {name}: sigma {100 * su:.3f} % -> {100 * sa:.3f} %; gain x{gain:.2f}")
    print(f"  wall time ratio (aperture / uniform): {t_a / t_u:.2f} (best of {repeats})")

    # ------------------------------------------------------------------ figure
    iz = int(5.0 / grid.spacing[2])
    x = (np.arange(grid.shape[0]) + 0.5) * grid.spacing[0] - CENTRE[0]
    y = (np.arange(grid.shape[1]) + 0.5) * grid.spacing[1] - CENTRE[1]
    extent = (x[0], x[-1], y[0], y[-1])
    vmax = pooled[:, :, iz].max()
    fig, axes = plt.subplots(1, 5, figsize=(21.0, 4.4))

    ax = axes[0]
    axis = np.linspace(-FIELD / 2, FIELD / 2, 160)
    gu, gv = np.meshgrid(axis, axis, indexing="ij")
    focal = np.array(uniform.fan.focal_point)
    targets = np.stack([CENTRE[0] + gu.ravel(), CENTRE[1] + gv.ravel(), np.zeros(gu.size)], 1)
    rays = targets - focal
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    bev = uniform.transmission(0, np.full(gu.size, 2.0), np.tile(focal, (gu.size, 1)), rays)
    ax.imshow(
        bev.reshape(gu.shape).T,
        origin="lower",
        extent=(-FIELD / 2, FIELD / 2, -FIELD / 2, FIELD / 2),
        cmap="cividis",
        norm=LogNorm(vmin=1e-3, vmax=1.0),
    )
    v_edges, u_lo, u_hi = sampled.aperture_region(0)
    for b in np.flatnonzero(u_hi > u_lo):  # the fan's aperture plane is the surface here
        ax.add_patch(
            Rectangle(
                (u_lo[b], v_edges[b]),
                u_hi[b] - u_lo[b],
                v_edges[b + 1] - v_edges[b],
                fill=False,
                edgecolor="#d8853b",
                linewidth=1.2,
            )
        )
    ax.set_title("BEV transmission at 2 MeV (log)\nsampling region outlined")
    ax.set_xlabel("x at surface (cm)")
    ax.set_ylabel("y at surface (cm)")

    for ax, dose, label in ((axes[1], dose_u, "uniform"), (axes[2], dose_a, "aperture-sampled")):
        ax.imshow(dose[:, :, iz].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"dose at 5 cm, {label}\n{n:,} histories")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")

    ax = axes[3]
    # Averaged over the rows inside the opening and 4-6 cm depth (0.25 mm voxels are
    # noisy one by one); the band is the voxel sigmas in quadrature, a lower bound.
    rows = (y > OPENING[2] + 0.25) & (y < OPENING[3] - 0.25)
    slab = slice(int(4.0 / grid.spacing[2]), int(6.0 / grid.spacing[2]))
    count = rows.sum() * (slab.stop - slab.start)
    for dose, sigma, label, colour in (
        (dose_u, sig_u, "uniform", "#2b5fa8"),
        (dose_a, sig_a, f"aperture, delta = {args.delta:g}", "#d8853b"),
    ):
        profile = dose[:, rows, slab].mean(axis=(1, 2)) / level
        band = np.sqrt((sigma[:, rows, slab] ** 2).sum(axis=(1, 2))) / count / level
        ax.plot(x, profile, color=colour, label=label, linewidth=1.2)
        ax.fill_between(x, profile - band, profile + band, color=colour, alpha=0.3)
    ax.set_title("profile through the opening, 4-6 cm deep\n(1 sigma bands)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("dose / open-field mean")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend()

    ax = axes[4]
    with np.errstate(divide="ignore", invalid="ignore"):
        # Below a few percent of the maximum a voxel holds a handful of deposits per
        # batch and its batch sigma is itself too noisy to compare voxel by voxel; the
        # printed region RMS covers it.
        dense = (pooled > 0.05 * top) & (sig_u > 0)
        ratio = np.where(dense, sig_a / np.where(sig_u > 0, sig_u, 1.0), np.nan)
    image = ax.imshow(
        ratio[:, :, iz].T, origin="lower", extent=extent, cmap="RdBu_r", vmin=0.0, vmax=2.0
    )
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title("sigma ratio at 5 cm, aperture / uniform\n(voxels above 5 % of max; < 1 better)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

    fig.suptitle(f"Aperture importance sampling of a 2 x 2 cm segment ({args.material}, {backend})")
    fig.tight_layout()
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=120)
    print(f"saved {stem.name}.png")


if __name__ == "__main__":
    main()
