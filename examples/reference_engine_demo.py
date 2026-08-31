"""The reference photon engine on a water phantom.

Runs two small problems on the pure-NumPy reference backend and renders one 2x2 figure
(``reference_engine_demo.png``, saved beside this script) — a depth-dose curve and a
central-plane dose map for each:

- **Broad 2 MeV parallel beam** (top row): the depth dose, with its 2-sigma band, pulls
  away from the primary-only exponential ``exp(-mu z)`` with depth — scatter buildup,
  the physics this engine exists to get right. The slice shows the whole irradiated cube
  attenuating with depth.
- **2 MeV pencil beam** (bottom row): on the narrow central axis, scattered photons
  mostly *leave* the axis instead of accumulating, so the central-axis depth dose hugs
  the same exponential the broad beam departs from. The slice, on a log scale over
  three decades, shows the primary column and the faint scatter halo around it.

There is no electron buildup at the surface in either case: electrons are not
transported in this example (KERMA approximation).

Run it from the repository root (requires the ``examples`` extra, i.e. matplotlib)::

    python examples/reference_engine_demo.py

Runtime is a few seconds. The printed energy balance is exact bookkeeping, not an
estimate; if emitted != deposited + escaped, the engine is broken.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure

from pyradmc.backends.ref.engine import ReferenceEngine, TransportResult
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource, PencilBeamSource
from pyradmc.rng.host import HostRNG

SEED = 20260711
ENERGY_MEV = 2.0

# Reference dataviz palette (validated): series blue, neutral ink, blue sequential ramp.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
SERIES_BLUE = "#2a78d6"
BLUE_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


def run_broad_beam() -> tuple[TransportResult, VoxelGrid, float]:
    """Broad parallel beam onto a 16 cm water cube; returns result, grid, and mu."""
    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = ParallelBeamSource(energy=ENERGY_MEV, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    result = engine.run(source, n_histories=80_000, n_batches=10, seed=SEED)
    mu = 1.0 * xs.mu_over_rho_total(ENERGY_MEV, WATER)
    return result, grid, mu


def run_pencil_beam() -> tuple[TransportResult, VoxelGrid]:
    """Pencil beam down the axis of a finer water cube."""
    grid = VoxelGrid.uniform_water(shape=(32, 32, 32), spacing=(0.5, 0.5, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    # Aim through a voxel *center* (voxels are half-open): a beam on the 8.0 boundary
    # would fill the [8.0, 8.5) column and make the scatter halo look skewed.
    source = PencilBeamSource(
        energy=ENERGY_MEV, position=(8.25, 8.25, -1.0), direction=(0.0, 0.0, 1.0)
    )
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    result = engine.run(source, n_histories=150_000, n_batches=10, seed=SEED)
    return result, grid


def _style(ax: Axes) -> None:
    """Recessive panel styling shared by all four axes."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)


def _plot_depth_dose(
    ax: Axes,
    depth: np.ndarray,
    dose: np.ndarray,
    sigma: np.ndarray,
    mu: float,
    title: str,
) -> None:
    """Relative depth dose with a 2-sigma band against the primary-only exponential."""
    scale = 1.0 / dose.max()
    primary = dose[0] * np.exp(-mu * (depth - depth[0]))

    ax.fill_between(
        depth,
        (dose - 2 * sigma) * scale,
        (dose + 2 * sigma) * scale,
        color=SERIES_BLUE,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        depth,
        dose * scale,
        color=SERIES_BLUE,
        lw=2,
        label="Monte Carlo (±2σ)",  # noqa: RUF001 — deliberate Greek sigma in the plot label
    )
    ax.plot(
        depth,
        primary * scale,
        color=INK_SECONDARY,
        lw=1.6,
        ls="--",
        label=r"primary only: $e^{-\mu z}$",
    )
    ax.set_xlabel("depth z (cm)", color=INK)
    ax.set_ylabel("dose (relative)", color=INK)
    ax.set_title(title, color=INK, fontsize=10)
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower left")
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def _plot_slice(
    fig: Figure,
    ax: Axes,
    plane: np.ndarray,
    extent_cm: float,
    norm: Normalize,
    title: str,
    colorbar_label: str,
) -> None:
    """Central-plane dose map (rows = depth, columns = x) with a labeled colorbar."""
    cmap = LinearSegmentedColormap.from_list("pyradmc_blue", [SURFACE, *BLUE_RAMP])
    cmap.set_bad(SURFACE)
    image = ax.imshow(
        np.ma.masked_less_equal(plane, 0.0),
        cmap=cmap,
        norm=norm,
        extent=(0.0, extent_cm, extent_cm, 0.0),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlabel("x (cm)", color=INK)
    ax.set_ylabel("depth z (cm)", color=INK)
    ax.set_title(title, color=INK, fontsize=10)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.92)
    colorbar.set_label(colorbar_label, color=INK, fontsize=9)
    colorbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def main() -> None:
    """Run both problems, print the energy books, and render the 2x2 figure."""
    t0 = time.perf_counter()
    broad, broad_grid, mu = run_broad_beam()
    pencil, pencil_grid = run_pencil_beam()
    runtime = time.perf_counter() - t0

    for name, r in [("broad beam", broad), ("pencil beam", pencil)]:
        balance = r.energy_deposited + r.energy_escaped
        print(
            f"{name}: {r.n_histories} histories | emitted {r.energy_emitted:.1f} MeV = "
            f"deposited {r.energy_deposited:.1f} + escaped {r.energy_escaped:.1f} "
            f"(balance error {abs(balance - r.energy_emitted) / r.energy_emitted:.1e})"
        )
    print(f"total transport time: {runtime:.1f} s")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), facecolor=SURFACE, constrained_layout=True)
    fig.suptitle(
        "pyradmc — reference photon engine, 2 MeV in water (KERMA approximation)",
        color=INK,
        fontsize=12,
    )
    for ax in axes.flat:
        _style(ax)

    # --- top row: broad beam ------------------------------------------------------
    depth_broad = (np.arange(broad_grid.shape[2]) + 0.5) * broad_grid.spacing[2]
    dose_broad = broad.dose.mean(axis=(0, 1))
    # Quadrature slice sigma; an under-estimate (in-slice correlations), band only.
    sigma_broad = np.sqrt((broad.dose_sigma**2).sum(axis=(0, 1))) / (16 * 16)
    _plot_depth_dose(
        axes[0, 0],
        depth_broad,
        dose_broad,
        sigma_broad,
        mu,
        "Broad beam — depth dose departs from the exponential",
    )
    gap_mid = float((dose_broad[12] / dose_broad.max() + np.exp(-mu * (depth_broad[12] - 0.5))) / 2)
    axes[0, 0].annotate(
        "scatter buildup",
        xy=(12.5, gap_mid),
        xytext=(9.6, 0.42),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "lw": 0.8},
    )

    # Average over y: a single 1 cm plane at this history count is ~25 percent noise
    # per voxel, which would bury the ~45 percent attenuation gradient.
    plane_broad = broad.dose.mean(axis=1).T
    plane_broad = plane_broad / plane_broad.max()
    _plot_slice(
        fig,
        axes[0, 1],
        plane_broad,
        extent_cm=16.0,
        norm=Normalize(vmin=0.0, vmax=1.0),
        title="Broad beam — dose map (averaged over y)",
        colorbar_label="dose relative to peak (linear)",
    )

    # --- bottom row: pencil beam --------------------------------------------------
    ix = int(8.25 / pencil_grid.spacing[0])
    iy = int(8.25 / pencil_grid.spacing[1])
    depth_pencil = (np.arange(pencil_grid.shape[2]) + 0.5) * pencil_grid.spacing[2]
    dose_axis = pencil.dose[ix, iy, :]
    sigma_axis = pencil.dose_sigma[ix, iy, :]
    _plot_depth_dose(
        axes[1, 0],
        depth_pencil,
        dose_axis,
        sigma_axis,
        mu,
        "Pencil beam — central axis stays near the exponential",
    )
    axes[1, 0].annotate(
        "scatter leaves the narrow axis",
        xy=(10.0, float(dose_axis[20] / dose_axis.max())),
        xytext=(6.4, 0.35),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "lw": 0.8},
    )

    plane_pencil = pencil.dose[:, iy, :].T
    plane_pencil = plane_pencil / plane_pencil.max()
    _plot_slice(
        fig,
        axes[1, 1],
        plane_pencil,
        extent_cm=16.0,
        norm=LogNorm(vmin=1e-3, vmax=1.0),
        title="Pencil beam — central plane: primary column + scatter halo",
        colorbar_label="dose relative to peak (log scale)",
    )

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
