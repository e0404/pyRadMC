"""Multi-material transport through bone and lung heterogeneities.

Compiles the whole material registry (water, air, lung, adipose, cortical bone) from
the EPICS libraries via :func:`pyRadMC.data.tabulated.precompile.compile_materials`,
then transports a 6 MeV photon pencil beam through two phantoms: homogeneous water,
and water with a 2 cm cortical-bone slab (rho = 1.85) followed by an 8 cm inflated-lung
region (lung tissue at rho = 0.26 — the material carries the composition, the voxel
density the inflation). The overlaid central-axis depth-dose curves show the classic
heterogeneity signatures: extra attenuation through and beyond the bone, reduced
attenuation (a shallower dose slope) across the low-density lung, and the water curve
recovered downstream shifted by the radiological path difference.

The profiles run smoothly through the interfaces because each electron substep is
capped at the next voxel face (:func:`pyRadMC.geometry.grid.distance_to_voxel_boundary`),
so a step's density and material match the voxel it is actually in rather than plowing
one medium's stopping power across the boundary into the neighbour's mass. Any small
residual feature at the bone edges is second order (the multiple-scattering hinge can
still deflect the short post-hinge segment across the face) plus genuine interface
dosimetry, not the single-voxel spike the earlier start-voxel step produced.

On first run this **downloads the ~120 MB EPDL/EEDL libraries** into
``~/.cache/pyRadMC/epics`` (cached thereafter). Defaults to Warp (GPU if present),
else the reference engine. Run from the repository root::

    python examples/materials_demo.py [--histories N] [--backend B]

Renders ``materials_demo.png`` and ``materials_demo.csv`` beside this
script. Wall-clock is machine-dependent and not a benchmark.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyRadMC.data.materials import CORTICAL_BONE, LUNG, WATER
from pyRadMC.data.tabulated import build
from pyRadMC.data.tabulated.precompile import compile_materials
from pyRadMC.data.tabulated.source import TabulatedCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import PencilBeamSource
from pyRadMC.rng.host import HostRNG

ENERGY_MEV = 6.0
N_Z = 260
DZ_CM = 0.1
BONE_CM = (4.0, 6.0)  # cortical bone slab, rho = 1.85
LUNG_CM = (10.0, 18.0)  # inflated lung, rho = 0.26
LUNG_INFLATED_DENSITY = 0.26


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


def heterogeneous_grid() -> VoxelGrid:
    """Water phantom with a cortical-bone slab and an inflated-lung region."""
    shape = (20, 20, N_Z)
    density = np.ones(shape, dtype=np.float64)
    material = np.full(shape, WATER, dtype=np.int32)
    z_centers = (np.arange(N_Z) + 0.5) * DZ_CM
    bone = (z_centers >= BONE_CM[0]) & (z_centers < BONE_CM[1])
    lung = (z_centers >= LUNG_CM[0]) & (z_centers < LUNG_CM[1])
    material[:, :, bone] = CORTICAL_BONE
    density[:, :, bone] = 1.85
    material[:, :, lung] = LUNG
    density[:, :, lung] = LUNG_INFLATED_DENSITY
    return VoxelGrid(shape=shape, spacing=(2.0, 2.0, DZ_CM), density=density, material=material)


def run_pdd(backend: str, grid: VoxelGrid, xs: TabulatedCrossSections, args) -> np.ndarray:
    """Transport the pencil beam and return the laterally integrated depth dose."""
    engine, backend_name = make_engine(backend, grid, xs)
    source = PencilBeamSource(
        energy=ENERGY_MEV, position=(20.125, 20.125, -1.0), direction=(0.0, 0.0, 1.0)
    )
    t0 = time.perf_counter()
    result = engine.run(source, n_histories=args.histories, n_batches=args.batches, seed=args.seed)
    dt = time.perf_counter() - t0
    print(f"  {args.histories:,} histories in {dt:.1f} s ({backend_name}); not a benchmark")
    return result.dose.sum(axis=(0, 1))


def main() -> None:
    """Compile the registry, run both phantoms, render the comparison figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--histories", type=int, default=10_000_000)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--backend", choices=("auto", "warp", "ref"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None, help="EPICS library cache")
    args = parser.parse_args()

    print("fetching EPDL/EEDL libraries (cached after first run) ...")
    epdl = build.download_library("epdl", args.cache_dir)
    eedl = build.download_library("eedl", args.cache_dir)
    print("compiling the material registry ...")
    data = compile_materials(
        epdl.read_text(encoding="latin-1"), eedl.read_text(encoding="latin-1"), e_max=8.0
    )
    print(f"  {data.provenance}")

    water_grid = VoxelGrid.uniform_water(shape=(20, 20, N_Z), spacing=(2.0, 2.0, DZ_CM))
    hetero_grid = heterogeneous_grid()

    print("transporting: homogeneous water ...")
    xs_water = TabulatedCrossSections(data, geometry_densities=water_grid.max_density_by_material())
    pdd_water = run_pdd(args.backend, water_grid, xs_water, args)
    print("transporting: bone + inflated lung ...")
    xs_hetero = TabulatedCrossSections(
        data, geometry_densities=hetero_grid.max_density_by_material()
    )
    pdd_hetero = run_pdd(args.backend, hetero_grid, xs_hetero, args)

    depth = (np.arange(N_Z) + 0.5) * DZ_CM
    norm = pdd_water.max()

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axvspan(*BONE_CM, color="#c9c2b6", alpha=0.6, label="cortical bone (rho 1.85)")
    ax.axvspan(*LUNG_CM, color="#d8ecf5", alpha=0.6, label="lung (rho 0.26)")
    ax.plot(depth, 100.0 * pdd_water / norm, color="#3b7dd8", label="homogeneous water")
    ax.plot(depth, 100.0 * pdd_hetero / norm, color="#d8553b", label="bone + lung phantom")
    ax.set_title(f"Heterogeneity depth dose, {ENERGY_MEV:.0f} MeV photons")
    ax.set_xlabel("depth (cm)")
    ax.set_ylabel("dose (% of water maximum)")
    ax.set_xlim(0.0, N_Z * DZ_CM)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=160)
    print(f"saved {stem.name}.png")

    with Path(f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_cm", "water_pdd", "heterogeneous_pdd"])
        for z, w, h in zip(depth, pdd_water / norm, pdd_hetero / norm, strict=True):
            writer.writerow([f"{z:.3f}", f"{w:.6g}", f"{h:.6g}"])
    print(f"saved {stem.name}.csv")


if __name__ == "__main__":
    main()
