"""Dose scoring decoupled from the transport grid + dose-to-water.

Transport runs on a fine 2 mm CT-like phantom (water body, cortical-bone slab,
low-density lung block); dose is *accumulated* on a separate 3 mm dose grid —
the pyRadPlan-style decoupling. Scoring directly on the coarse grid shrinks the
Dij and device buffers by ~(2/3)^3 ~ 0.30x and lowers per-voxel sigma, while the
transport physics keeps the full 2 mm heterogeneity; a non-integer 2->3 mm ratio
exercises the exact voxel-overlap mass rebin. The same run is repeated with
``scoring_mode="dose_to_water"``: with the same seed the transport replays
identically, so the D_w/D_m ratio per voxel is the deposit-weighted
stopping-power ratio — ~1.10-1.12 inside the bone slab, 1 in water and
(near) 1 in lung-density water — the textbook Siebers et al. (2000) picture.

A subregion dose grid (no CT-coverage requirement) is also run: deposits beyond
its far edge land in the *unscored* energy-ledger bucket, never in a clamped
edge voxel, and the printed three-bucket balance closes.

On first run this **downloads the ~120 MB EPDL/EEDL libraries** into
``~/.cache/pyRadMC/epics`` (cached thereafter). Needs the ``examples`` extra
(matplotlib). Defaults to Warp (GPU if present). Run from the repository root::

    python examples/dose_grid_demo.py [--histories N] [--backend B]

Renders ``dose_grid_demo.png`` beside this script. Wall-clock is
machine-dependent and not a benchmark.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyRadMC.data.materials import CORTICAL_BONE, LUNG, WATER
from pyRadMC.data.tabulated import build
from pyRadMC.data.tabulated.precompile import compile_materials
from pyRadMC.data.tabulated.source import TabulatedCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource
from pyRadMC.rng.host import HostRNG
from pyRadMC.scoring.grid import ScoringGrid

SPACING_CM = 0.2  # 2 mm transport (CT) grid
DOSE_SPACING_CM = 0.3  # 3 mm dose grid: non-integer ratio, the pyRadPlan case
SHAPE = (60, 60, 90)  # 12 x 12 x 18 cm
BONE_Z = (20, 30)  # slab across the field, 4-6 cm depth (transport-voxel indices)
LUNG_Z = (45, 75)  # lung block behind it, 9-15 cm


def phantom() -> VoxelGrid:
    """Water body with a cortical-bone slab and a lung block, on the 2 mm grid."""
    density = np.ones(SHAPE)
    material = np.full(SHAPE, WATER, dtype=np.int32)
    material[:, :, BONE_Z[0] : BONE_Z[1]] = CORTICAL_BONE
    density[:, :, BONE_Z[0] : BONE_Z[1]] = 1.92  # ICRP cortical bone
    material[:, :, LUNG_Z[0] : LUNG_Z[1]] = LUNG
    density[:, :, LUNG_Z[0] : LUNG_Z[1]] = 0.30
    return VoxelGrid(
        shape=SHAPE,
        spacing=(SPACING_CM,) * 3,
        density=density,
        material=material,
    )


def make_engine(backend: str, grid: VoxelGrid, xs: TabulatedCrossSections):
    """Return an engine and its name; 'auto' prefers Warp (CUDA, else CPU)."""
    if backend in ("auto", "warp"):
        try:
            import warp as wp

            from pyRadMC.backends.warp.engine import WarpEngine

            device = "cuda:0" if wp.is_cuda_available() else "cpu"
            return WarpEngine(grid=grid, cross_sections=xs, device=device), f"warp:{device}"
        except Exception:
            if backend == "warp":
                raise
    from pyRadMC.backends.ref.engine import ReferenceEngine

    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()), "ref"


def central_axis(dose: np.ndarray, sigma: np.ndarray, spacing: float):
    """Mean dose and combined sigma over the central 2x2 columns, with depths."""
    nx, ny, nz = dose.shape
    sl = (slice(nx // 2 - 1, nx // 2 + 1), slice(ny // 2 - 1, ny // 2 + 1))
    axis = dose[sl].mean(axis=(0, 1))
    axis_sigma = np.sqrt((sigma[sl] ** 2).sum(axis=(0, 1))) / 4.0
    depths = (np.arange(nz) + 0.5) * spacing
    return depths, axis, axis_sigma


def main() -> None:
    """Transport once per scoring configuration and render the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--histories", type=int, default=2_000_000)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--backend", choices=("auto", "warp", "ref"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None, help="EPICS library cache")
    args = parser.parse_args()

    grid = phantom()
    print("fetching EPDL/EEDL libraries (cached after first run) ...")
    epdl = build.download_library("epdl", args.cache_dir)
    eedl = build.download_library("eedl", args.cache_dir)
    print("compiling the tabulated material registry ...")
    data = compile_materials(
        epdl.read_text(encoding="latin-1"), eedl.read_text(encoding="latin-1"), e_max=8.0
    )
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    engine, backend_name = make_engine(args.backend, grid, xs)

    extent_cm = tuple(n * SPACING_CM for n in SHAPE)
    source = ParallelBeamSource(
        energy=6.0,
        z=-1.0,
        x_range=(0.2 * extent_cm[0], 0.8 * extent_cm[0]),
        y_range=(0.2 * extent_cm[1], 0.8 * extent_cm[1]),
    )

    # The 3 mm dose grid covering the phantom (36 mm-aligned; ceil to cover).
    coarse_shape = tuple(int(np.ceil(n * SPACING_CM / DOSE_SPACING_CM)) for n in SHAPE)
    coarse = ScoringGrid.rebin(
        grid, shape=coarse_shape, spacing=(DOSE_SPACING_CM,) * 3, origin=grid.origin
    )
    # A subregion dose grid: stops at 12 cm depth, inside a 18 cm phantom.
    sub = ScoringGrid.rebin(
        grid, shape=(40, 40, 40), spacing=(DOSE_SPACING_CM,) * 3, origin=(0.0, 0.0, 0.0)
    )

    def run(label: str, **kwargs):
        t0 = time.perf_counter()
        result = engine.run(
            source, n_histories=args.histories, n_batches=args.batches, seed=args.seed, **kwargs
        )
        print(f"  {label}: {time.perf_counter() - t0:.1f} s on {backend_name} (not a benchmark)")
        return result

    print(f"transporting {args.histories:,} histories per configuration ...")
    fine = run("fine 2 mm (transport grid)")
    coarse_dm = run("coarse 3 mm dose grid", scoring_grid=coarse)
    coarse_dw = run("coarse 3 mm, dose-to-water", scoring_grid=coarse, scoring_mode="dose_to_water")
    sub_dm = run("subregion 3 mm dose grid", scoring_grid=sub)

    n_fine = int(np.prod(SHAPE))
    n_coarse = int(np.prod(coarse_shape))
    print(f"scoring voxels: {n_fine:,} (2 mm) -> {n_coarse:,} (3 mm), {n_coarse / n_fine:.2f}x")
    balance = sub_dm.energy_deposited + sub_dm.energy_unscored + sub_dm.energy_escaped
    print(
        "subregion ledger (MeV): emitted "
        f"{sub_dm.energy_emitted:.6g} = deposited {sub_dm.energy_deposited:.6g} "
        f"+ unscored {sub_dm.energy_unscored:.6g} + escaped {sub_dm.energy_escaped:.6g} "
        f"(closes to {abs(balance / sub_dm.energy_emitted - 1.0):.2e} relative)"
    )

    # --- figure ---------------------------------------------------------------
    z_fine, d_fine, s_fine = central_axis(fine.dose, fine.dose_sigma, SPACING_CM)
    z_coarse, d_coarse, s_coarse = central_axis(
        coarse_dm.dose, coarse_dm.dose_sigma, DOSE_SPACING_CM
    )
    _, d_water, _ = central_axis(coarse_dw.dose, coarse_dw.dose_sigma, DOSE_SPACING_CM)

    fig, (ax_pdd, ax_ratio) = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    ax_pdd.errorbar(
        z_fine,
        d_fine,
        yerr=s_fine,
        fmt=".",
        ms=3,
        lw=0.8,
        alpha=0.7,
        label=f"2 mm transport grid ({n_fine:,} voxels)",
    )
    ax_pdd.errorbar(
        z_coarse,
        d_coarse,
        yerr=s_coarse,
        fmt="o",
        ms=4,
        lw=0.8,
        label=f"3 mm dose grid ({n_coarse:,} voxels, same transport)",
    )
    ax_pdd.set_ylabel("central-axis dose (MeV/g per history)")
    ax_pdd.set_title("Transport at 2 mm, dose scored at 3 mm: same physics, 0.30x memory")
    ax_pdd.legend()

    ratio = np.divide(d_water, d_coarse, out=np.ones_like(d_water), where=d_coarse > 0)
    ax_ratio.plot(z_coarse, ratio, "o-", ms=4, lw=1.0, color="tab:red")
    ax_ratio.axhline(1.0, color="grey", lw=0.8)
    ax_ratio.set_ylabel("$D_w / D_m$ (same seed)")
    ax_ratio.set_xlabel("depth z (cm)")
    ax_ratio.set_title("Dose-to-water vs dose-to-medium: the stopping-power ratio, per medium")
    for z_range, name, color in (
        (BONE_Z, "cortical bone", "tab:orange"),
        (LUNG_Z, "lung (0.3 g/cm$^3$)", "tab:blue"),
    ):
        for ax in (ax_pdd, ax_ratio):
            ax.axvspan(z_range[0] * SPACING_CM, z_range[1] * SPACING_CM, alpha=0.12, color=color)
        ax_ratio.text(
            (z_range[0] + z_range[1]) / 2 * SPACING_CM,
            ax_ratio.get_ylim()[1],
            name,
            ha="center",
            va="top",
            fontsize=8,
            color=color,
        )
    fig.tight_layout()

    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=160)
    print(f"saved {stem.name}.png")


if __name__ == "__main__":
    main()
