"""Transport an IAEA phase-space source into a water phantom.

Reads a real IAEA phase-space file (default: the Varian TrueBeam 6 MV file at
``D:/data/phsp``), places a water phantom downstream of the recorded particles,
runs the reference engine, and plots the central-axis depth-dose and a lateral
profile at the depth of maximum dose.

**This is a demonstration of the pipeline on real data, not a clinical field.**
Two honest caveats:

- *Uncollimated.* The Varian file is a **patient-independent** phase space stored
  *above the movable jaws* — pyradmc models no jaws/MLC or treatment head, so the
  simulated field is the full, unshaped divergent cone. A clinical field would need
  the collimation/forward-transport model (a later addition). The central-axis PDD
  is still physically meaningful because on-axis fluence and water attenuation
  dominate it; the field *shape* and penumbra are not.
- *Air gap is vacuum.* The region between the phase-space plane and the phantom is
  transported as vacuum, not air: particles fly straight in with no interaction.
  For photons over a short gap this is a <1% effect; for the ~1% contamination
  electrons it slightly overstates their (surface/buildup) contribution.

By default this runs on the Warp backend (GPU if a CUDA device is present, else
Warp-CPU), falling back to the reference engine if Warp is not installed. Run from
the repository root::

    python examples/phasespace_demo.py [PHSP] [--histories N] [--backend B]

Renders ``phasespace_demo.png`` and ``phasespace_demo.csv`` beside
this script. Wall-clock is machine-dependent and not a benchmark.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.phasespace import PhaseSpaceSource

DEFAULT_PHSP = Path("D:/data/phsp/Varian_TrueBeam6MV_01.IAEAheader")

# Phantom placement. The Varian records sit at z up to ~27.4 cm (origin at the
# target, +z toward the phantom); a water box beginning below that catches the
# beam, with the few-cm gap transported as vacuum. Lateral extent is wide enough
# for on-axis scatter equilibrium at 6 MV.
FRONT_Z_CM = 30.0
DEPTH_CM = 30.0
HALF_WIDTH_CM = 15.0
SPACING = (0.3, 0.3, 0.3)
# The uncollimated divergent beam spreads fluence over a wide area, so the
# on-axis column is starved of deposits; average the PDD over a central column a
# few cm wide (the field is broad) to get a usable curve at demo statistics.
ON_AXIS_ROI_CM = 3.0


def build_grid() -> VoxelGrid:
    """Build a water phantom downstream of the phase-space plane, on the axis."""
    nx = round(2 * HALF_WIDTH_CM / SPACING[0])
    ny = round(2 * HALF_WIDTH_CM / SPACING[1])
    nz = round(DEPTH_CM / SPACING[2])
    origin = (-HALF_WIDTH_CM, -HALF_WIDTH_CM, FRONT_Z_CM)
    return VoxelGrid.uniform_water(shape=(nx, ny, nz), spacing=SPACING, origin=origin)


def central_axis_pdd(dose: np.ndarray, grid: VoxelGrid) -> tuple[np.ndarray, np.ndarray]:
    """Depth (cm from phantom front) and dose averaged over a small on-axis column."""
    half_i = max(0, round(0.5 * ON_AXIS_ROI_CM / grid.spacing[0]))
    half_j = max(0, round(0.5 * ON_AXIS_ROI_CM / grid.spacing[1]))
    ci = grid.shape[0] // 2
    cj = grid.shape[1] // 2
    roi = dose[ci - half_i : ci + half_i + 1, cj - half_j : cj + half_j + 1, :]
    pdd = roi.mean(axis=(0, 1))
    depth = (np.arange(grid.shape[2]) + 0.5) * grid.spacing[2]
    return depth, pdd


def lateral_profile(dose: np.ndarray, grid: VoxelGrid, iz: int) -> tuple[np.ndarray, np.ndarray]:
    """Off-axis dose vs x at a given depth slice (y through the axis)."""
    cj = grid.shape[1] // 2
    x = grid.origin[0] + (np.arange(grid.shape[0]) + 0.5) * grid.spacing[0]
    return x, dose[:, cj, iz]


def make_engine(backend: str, grid: VoxelGrid, xs: AnalyticCrossSections):
    """Build the transport engine; 'auto' prefers Warp GPU, then Warp CPU, then ref."""
    if backend in ("auto", "warp"):
        try:
            import warp as wp

            from pyradmc.backends.warp.engine import WarpEngine

            device = "cuda:0" if wp.is_cuda_available() else "cpu"
            return WarpEngine(grid=grid, cross_sections=xs, device=device), f"warp:{device}"
        except ImportError:
            if backend == "warp":
                raise
    from pyradmc.backends.ref.engine import ReferenceEngine
    from pyradmc.rng.host import HostRNG

    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()), "ref"


def main() -> None:
    """Run the demo and render the figure + CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "phsp", nargs="?", type=Path, default=DEFAULT_PHSP, help="IAEA .IAEAheader path"
    )
    parser.add_argument("--histories", type=int, default=10_000_000)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--backend", choices=("auto", "warp", "ref"), default="auto")
    args = parser.parse_args()

    if not args.phsp.exists():
        sys.exit(
            f"phase-space header not found: {args.phsp}\n"
            "Pass the path to an IAEA .IAEAheader file as the first argument."
        )

    grid = build_grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine, backend_name = make_engine(args.backend, grid, xs)

    print(f"loading {args.phsp.name} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        source = PhaseSpaceSource(args.phsp)
    print(f"  {len(source):,} transportable particles; phantom {grid.shape} @ {grid.spacing} cm")

    print(f"transporting {args.histories:,} histories on {backend_name} ...")
    t0 = time.perf_counter()
    result = engine.run(source, n_histories=args.histories, n_batches=args.batches, seed=args.seed)
    source.close()
    dt = time.perf_counter() - t0

    dose = result.dose
    depth, pdd = central_axis_pdd(dose, grid)
    iz_max = int(np.argmax(pdd))
    dmax = pdd[iz_max]
    pdd_pct = 100.0 * pdd / dmax if dmax > 0 else pdd
    x, lat = lateral_profile(dose, grid, iz_max)
    lat_pct = 100.0 * lat / lat.max() if lat.max() > 0 else lat

    fig, (ax_pdd, ax_lat) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax_pdd.plot(depth, pdd_pct, color="#3b7dd8")
    ax_pdd.axvline(depth[iz_max], color="#888", lw=0.8, ls="--")
    ax_pdd.set_title("Central-axis depth dose (uncollimated)")
    ax_pdd.set_xlabel("depth in water (cm)")
    ax_pdd.set_ylabel("dose (% of dmax)")
    ax_pdd.set_ylim(0, 105)

    ax_lat.plot(x, lat_pct, color="#d8853b")
    ax_lat.set_title(f"Lateral profile at dmax ({depth[iz_max]:.1f} cm)")
    ax_lat.set_xlabel("off-axis x (cm)")
    ax_lat.set_ylabel("dose (% of profile max)")
    ax_lat.set_ylim(0, 105)
    fig.tight_layout()

    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=160)
    print(f"saved {stem.name}.png")

    with Path(f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_cm", "pdd_percent"])
        writer.writerows(zip(depth, pdd_pct, strict=True))
    print(f"saved {stem.name}.csv")

    print(
        f"  dmax at {depth[iz_max]:.1f} cm; {args.histories:,} histories "
        f"in {dt:.1f} s ({backend_name})"
    )
    print("  uncollimated field, air gap = vacuum; wall-clock is not a benchmark")


if __name__ == "__main__":
    main()
