"""Phase 5 demo: the tabulated cross-section backend against EGSnrc.

Compiles liquid water from the EPICS libraries (EPDL photons with coherent form factors,
EEDL elastic scattering, ICRU-37 electron stopping) via
:func:`pyRadMC.data.tabulated.precompile.compile_water`, transports a 6 MeV monoenergetic
photon pencil beam through a water phantom, and overlays the central-axis depth dose on
the maintainer's EGSnrc full-physics benchmark
(``tests/validation/data/center_ray_dose_EGSNrc_intRadius6.0mm.txt``).

The tabulated backend reproduces this benchmark at the field-standard 5%/3mm gamma
criterion (``tests/validation/test_ranges_and_pdd.py``); the 2%/2mm difference is set by
the benchmark's geometry/metadata, not the cross-section data (see that test's docstring
and AGENTS.md 7.2). The pencil beam's laterally integrated (slice-sum) dose is the
central-axis dose of a broad beam under lateral scatter equilibrium, which is what the
benchmark scored.

On first run this **downloads the ~120 MB EPDL/EEDL libraries** into
``~/.cache/pyRadMC/epics`` (cached thereafter). Defaults to Warp (GPU if present), else
the reference engine. Run from the repository root::

    python examples/phase5_tabulated_demo.py [--histories N] [--backend B]

Renders ``phase5_tabulated_demo.png`` and ``phase5_tabulated_demo.csv`` beside this
script. Wall-clock is machine-dependent and not a benchmark.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyRadMC.data.tabulated import build
from pyRadMC.data.tabulated.precompile import compile_water
from pyRadMC.data.tabulated.source import TabulatedCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import PencilBeamSource
from pyRadMC.rng.host import HostRNG

BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "validation"
    / "data"
    / ("center_ray_dose_EGSNrc_intRadius6.0mm.txt")
)
BENCHMARK_COLUMN = 24  # 6.0 MeV column (after the depth column); see the gamma-gate test
ENERGY_MEV = 6.0
N_Z = 260  # 1 cm past COMPARE_MAX_CM so the exit face's backscatter deficit stays out of frame
DZ_CM = 0.1
COMPARE_MAX_CM = 25.0


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


def main() -> None:
    """Compile the tabulated backend, run the PDD, and render the figure + CSV."""
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
    print("compiling tabulated water ...")
    data = compile_water(
        epdl.read_text(encoding="latin-1"), eedl.read_text(encoding="latin-1"), e_max=8.0
    )
    print(f"  {data.provenance}")

    grid = VoxelGrid.uniform_water(shape=(20, 20, N_Z), spacing=(2.0, 2.0, DZ_CM))
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    engine, backend_name = make_engine(args.backend, grid, xs)
    source = PencilBeamSource(
        energy=ENERGY_MEV, position=(20.125, 20.125, -1.0), direction=(0.0, 0.0, 1.0)
    )

    print(f"transporting {args.histories:,} histories on {backend_name} ...")
    t0 = time.perf_counter()
    result = engine.run(source, n_histories=args.histories, n_batches=args.batches, seed=args.seed)
    dt = time.perf_counter() - t0

    depth = (np.arange(N_Z) + 0.5) * DZ_CM
    pdd = result.dose.sum(axis=(0, 1))  # pencil-kernel superposition -> broad-beam axis

    benchmark = np.loadtxt(BENCHMARK, skiprows=1)
    bench_depth = benchmark[:, 0]
    bench_dose = benchmark[:, BENCHMARK_COLUMN]
    mask = (bench_depth >= 0.3) & (bench_depth <= COMPARE_MAX_CM)
    ours_on_ref = np.interp(bench_depth[mask], depth, pdd)
    scale = float(np.sum(bench_dose[mask] * ours_on_ref) / np.sum(ours_on_ref**2))
    norm = bench_dose[mask].max()

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(depth, 100.0 * pdd * scale / norm, color="#3b7dd8", label="pyRadMC tabulated")
    ax.plot(
        bench_depth[mask],
        100.0 * bench_dose[mask] / norm,
        "o",
        ms=3,
        color="#d8853b",
        label="EGSnrc benchmark",
    )
    ax.set_title(f"Central-axis depth dose, {ENERGY_MEV:.0f} MeV photons in water")
    ax.set_xlabel("depth in water (cm)")
    ax.set_ylabel("dose (% of benchmark max)")
    ax.set_xlim(0, COMPARE_MAX_CM)
    ax.legend()
    fig.tight_layout()

    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=160)
    print(f"saved {stem.name}.png")

    with Path(f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_cm", "tabulated_pdd_scaled", "egsnrc"])
        for z, ours, ref in zip(
            bench_depth[mask], ours_on_ref * scale, bench_dose[mask], strict=True
        ):
            writer.writerow([f"{z:.3f}", f"{ours:.6g}", f"{ref:.6g}"])
    print(f"saved {stem.name}.csv")
    print(f"  {args.histories:,} histories in {dt:.1f} s ({backend_name}); not a benchmark")


if __name__ == "__main__":
    main()
