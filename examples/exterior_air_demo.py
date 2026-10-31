"""Long exact-helix steps through exterior air: same dose, measured wall time.

Renders one figure (``exterior_air_demo.png``, saved beside this script) for a 6 MV beam
through an elliptical tissue cylinder with two lungs in an air margin at 1.5 T, run twice on
the same grid: the current stepping, and with an :class:`~pyradmc.geometry.exterior.ExteriorAirMap`
handed to the engine.

- **Left**: dose along the beam through the isocentre, both paths, including the shell next
  to the air, where electrons have to curve back to deliver their dose.
- **Right**: the ratio of the two over the same axis. Both paths use the same seeds, so most
  of each history is shared and the ratio is paired: the bars are the unpaired seed spread and
  overstate its uncertainty. The independent-seed comparison is in the validation tier. The
  wall time per pass of each path is in the title.

The exterior path is the default under a uniform field (and warns once per engine, which this
script shows); ``exterior_air=None`` is how the current stepping is run here. A named
approximation (AGENTS.md section 8). Run it from the repository root (requires the
``examples`` extra, i.e. matplotlib, and the cached EPICS libraries)::

    python examples/exterior_air_demo.py

Runtime is under a minute on a CUDA device; pass ``--histories`` to trade statistics for
time on the CPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from pyradmc import AIR, LUNG, WATER, MagneticField
from pyradmc.backends.warp.engine import WarpEngine
from pyradmc.data.tabulated import build
from pyradmc.data.tabulated.precompile import compile_materials
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry.exterior import ExteriorAirMap
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv

SEED = 20260711
SEEDS = 6
B_TESLA = 1.5
SPACING = 0.4
MARGIN_CM = 3.0
AXES_CM = (15.0, 10.0)
LENGTH_CM = 16.0

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
CURRENT, EXTERIOR = "#52514e", "#2a78d6"


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def phantom() -> VoxelGrid:
    """Elliptical tissue cylinder along z with two lungs, air around it in x and y."""
    a, b = AXES_CM
    nx = round(2.0 * (a + MARGIN_CM) / SPACING)
    ny = round(2.0 * (b + MARGIN_CM) / SPACING)
    nz = round(LENGTH_CM / SPACING)
    x = (np.arange(nx) + 0.5) * SPACING - 0.5 * nx * SPACING
    y = (np.arange(ny) + 0.5) * SPACING - 0.5 * ny * SPACING
    xg, yg = np.meshgrid(x, y, indexing="ij")
    density = np.full((nx, ny), 1.205e-3)
    material = np.full((nx, ny), AIR, dtype=np.int32)
    body = (xg / a) ** 2 + (yg / b) ** 2 <= 1.0
    density[body] = 1.0
    material[body] = WATER
    for side in (-1.0, 1.0):
        lung = ((xg - side * 7.0) / 4.5) ** 2 + ((yg + 1.0) / 6.0) ** 2 <= 1.0
        density[lung] = 0.26
        material[lung] = LUNG
    shape = (nx, ny, nz)
    return VoxelGrid(
        shape=shape,
        spacing=(SPACING, SPACING, SPACING),
        density=np.broadcast_to(density[:, :, None], shape).copy(),
        material=np.broadcast_to(material[:, :, None], shape).copy(),
    )


def main() -> None:
    """Run both paths on one grid and save the comparison figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=1_500_000, help="per seed and path")
    parser.add_argument("--device", default="cuda:0", help="Warp device")
    args = parser.parse_args()
    t0 = time.perf_counter()

    grid = phantom()
    epdl, eedl = build.library_path("epdl", None), build.library_path("eedl", None)
    data = compile_materials(
        epdl.read_text(encoding="latin-1"), eedl.read_text(encoding="latin-1"), e_max=8.0
    )
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    c = tuple(n * s / 2.0 for n, s in zip(grid.shape, grid.spacing, strict=True))
    beam = SpectralBeamSource(
        spectrum=ali_rogers_mv("varian-6mv"),
        focal_point=(c[0], c[1] - 100.0, c[2]),
        center=c,
        width_u=10.0,
        width_v=10.0,
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
    )
    field = MagneticField.uniform((0.0, 0.0, B_TESLA))

    profiles = {}
    walls = {}
    for name, air in (("current", None), ("exterior", ExteriorAirMap.from_grid(grid))):
        engine = WarpEngine(
            grid=grid, cross_sections=xs, device=args.device, magnetic_field=field, exterior_air=air
        )
        engine.run(beam, n_histories=20_000, n_batches=1, seed=SEED)
        runs = []
        best = float("inf")
        for k in range(SEEDS):
            start = time.perf_counter()
            result = engine.run(beam, n_histories=args.histories, n_batches=1, seed=SEED + k)
            best = min(best, time.perf_counter() - start)
            # Central 4 x 4 cm of the field along the beam: enough voxels for a 2 % curve.
            ix, iz = grid.shape[0] // 2, grid.shape[2] // 2
            d = np.asarray(result.dose)[ix - 5 : ix + 5, :, iz - 5 : iz + 5]
            runs.append(d.mean(axis=(0, 2)))
        profiles[name] = np.stack(runs)
        walls[name] = best
        engine.release_device_memory()

    y = (np.arange(grid.shape[1]) + 0.5) * SPACING - c[1]
    inside = np.abs(y) <= AXES_CM[1]
    mean_c, mean_e = profiles["current"].mean(axis=0), profiles["exterior"].mean(axis=0)
    sem_c = profiles["current"].std(axis=0, ddof=1) / np.sqrt(SEEDS)
    sem_e = profiles["exterior"].std(axis=0, ddof=1) / np.sqrt(SEEDS)
    norm = mean_c[inside].max()
    ratio = mean_e[inside] / mean_c[inside]
    sigma = ratio * np.hypot(sem_e[inside] / mean_e[inside], sem_c[inside] / mean_c[inside])

    fig, (ax_dose, ax_ratio) = plt.subplots(1, 2, figsize=(11.0, 4.2), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    _style(ax_dose)
    _style(ax_ratio)
    ax_dose.plot(y[inside], mean_c[inside] / norm, color=CURRENT, lw=2.4, label="current stepping")
    ax_dose.plot(
        y[inside], mean_e[inside] / norm, color=EXTERIOR, lw=1.2, label="exterior-air path"
    )
    ax_dose.set_xlabel("position along the beam (cm)", color=INK)
    ax_dose.set_ylabel("dose (relative)", color=INK)
    ax_dose.set_title(f"6 MV, {B_TESLA:g} T along the body axis", color=INK, fontsize=10)
    ax_dose.legend(frameon=False, labelcolor=INK, fontsize=9)
    ax_ratio.axhline(1.0, color=INK_SECONDARY, lw=0.8)
    ax_ratio.errorbar(y[inside], ratio, yerr=sigma, color=EXTERIOR, lw=1.0, fmt="-")
    ax_ratio.set_xlabel("position along the beam (cm)", color=INK)
    ax_ratio.set_ylabel("exterior-air path / current stepping", color=INK)
    ax_ratio.set_ylim(0.9, 1.1)
    ax_ratio.set_title(
        f"wall per pass: current {walls['current']:.2f} s, exterior {walls['exterior']:.2f} s",
        color=INK,
        fontsize=10,
    )
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"figure written to {out} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
