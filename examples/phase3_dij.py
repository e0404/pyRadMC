"""Phase 3 example: the beamlet-resolved dose influence matrix (Dij).

Renders one figure (``phase3_dij.png``, saved beside this script):

- **Left**: the dose column of one central beamlet, side view through its axis on a
  log scale. It reads like a pencil beam: a sharp core under the 2 x 2 cm beamlet
  (edges marked), buildup over the secondary-electron range, and a scatter halo
  that fades into white where the default Dij truncation (1e-3 of the column
  maximum, AGENTS.md 2.8) cuts the tail.
- **Middle**: the fluence-sum identity. The open field is recombined from the Dij
  columns at uniform weights and overlaid on a *directly simulated* open-field run
  of independent histories; the lateral profiles at 5 cm depth agree within the
  error band. Columns partition the field's dose — nothing is lost or double
  counted by beamlet tagging.
- **Right**: the payoff. A wedge plan — beamlet weights ramping across the field —
  is a matrix product with the same Dij, no re-simulation. The profile follows the
  weights (steps shown), which is exactly the loop a treatment-plan optimizer runs
  thousands of times.

Run it from the repository root (requires the ``examples`` and ``warp`` extras)::

    python examples/phase3_dij.py

Runtime is well under a minute on a laptop; it uses CUDA when available and falls
back to Warp-CPU with fewer histories.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from pyRadMC.backends.warp.engine import WarpEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletGridSource, ParallelBeamSource

SEED = 20260711

# Reference dataviz palette (validated): categorical slots 1-3, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
SERIES = ["#2a78d6", "#1baf7a", "#eda100"]  # blue, aqua, yellow

SHAPE = (64, 64, 64)
SPACING = (0.4, 0.4, 0.4)  # 25.6 cm water cube
FIELD = (2.8, 22.8)  # 20 x 20 cm field, 2 cm margin to the phantom edge
N_X, N_Y = 10, 10  # 2 x 2 cm beamlets
PROFILE_DEPTH_CM = 5.0
ENERGY = 6.0


def _style(ax: Axes) -> None:
    """Recessive panel styling shared by all axes."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def run_all() -> dict:
    """One Dij run and one direct open-field run on the best available device."""
    grid = VoxelGrid.uniform_water(shape=SHAPE, spacing=SPACING)
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    lattice = BeamletGridSource(
        energy=ENERGY, z=-1.0, x_range=FIELD, y_range=FIELD, n_x=N_X, n_y=N_Y
    )
    open_field = ParallelBeamSource(energy=ENERGY, z=-1.0, x_range=FIELD, y_range=FIELD)

    cuda = wp.is_cuda_available()
    device = "cuda:0" if cuda else "cpu"
    n_per = 10_000_000 if cuda else 5_000
    engine = WarpEngine(grid=grid, cross_sections=xs, device=device, chunk_size=1_048_576)
    engine.run_dij(lattice, n_histories_per_beamlet=500, n_batches=1, seed=SEED)  # warm

    t0 = time.perf_counter()
    dij = engine.run_dij(lattice, n_histories_per_beamlet=n_per, n_batches=8, seed=SEED)
    dij_seconds = time.perf_counter() - t0

    # Independent histories for the middle panel's honest comparison.
    direct = engine.run(open_field, n_histories=dij.n_beamlets * n_per, n_batches=8, seed=SEED + 1)

    return {
        "grid": grid,
        "lattice": lattice,
        "dij": dij,
        "direct": direct,
        "device": device,
        "dij_seconds": dij_seconds,
    }


def column_panel(ax: Axes, data: dict) -> None:
    """Side view of one central beamlet's truncated dose column, log scale."""
    lattice: BeamletGridSource = data["lattice"]
    grid: VoxelGrid = data["grid"]
    j = (N_X // 2) * N_Y + (N_Y // 2)  # central beamlet
    x_lo, x_hi, y_lo, y_hi = lattice.beamlet_bounds(j)
    column = data["dij"].column_dense(j)
    iy = int((0.5 * (y_lo + y_hi) - grid.origin[1]) / grid.spacing[1])
    slab = column[:, iy, :]  # x-z plane through the beamlet axis
    peak = slab.max()
    masked = np.ma.masked_less_equal(slab, 0.0)
    extent = (
        grid.origin[2],
        grid.origin[2] + SHAPE[2] * grid.spacing[2],
        grid.origin[0],
        grid.origin[0] + SHAPE[0] * grid.spacing[0],
    )
    image = ax.imshow(
        masked,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
        norm=LogNorm(vmin=peak * 1.0e-3, vmax=peak),
    )
    image.cmap.set_bad(SURFACE)
    for edge in (x_lo, x_hi):
        ax.axhline(edge, color=SERIES[1], lw=0.9, ls="--")
    ax.grid(False)
    ax.set_xlabel("depth (cm)", fontsize=9, color=INK)
    ax.set_ylabel("x (cm)", fontsize=9, color=INK)
    ax.set_title("one Dij column (log dose)\nwhite = truncated tail", fontsize=10, color=INK)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=7, colors=INK_SECONDARY
    )


def _profile(dose: np.ndarray, sigma: np.ndarray, grid: VoxelGrid, iy: int, iz: int):
    x = grid.origin[0] + (np.arange(SHAPE[0]) + 0.5) * grid.spacing[0]
    return x, dose[:, iy, iz], sigma[:, iy, iz]


def fluence_sum_panel(ax: Axes, data: dict) -> None:
    """Uniform-weight recombination vs an independent open-field simulation."""
    grid: VoxelGrid = data["grid"]
    dij = data["dij"]
    direct = data["direct"]
    iy = SHAPE[1] // 2
    iz = int(PROFILE_DEPTH_CM / grid.spacing[2])

    recombined = dij.dose_for_weights(np.full(dij.n_beamlets, 1.0 / dij.n_beamlets))
    x, d_dij, _ = _profile(recombined, np.zeros_like(recombined), grid, iy, iz)
    _, d_ref, s_ref = _profile(direct.dose, direct.dose_sigma, grid, iy, iz)
    peak = d_ref.max()

    ax.fill_between(
        x,
        (d_ref - 2.0 * s_ref) / peak * 100.0,
        (d_ref + 2.0 * s_ref) / peak * 100.0,
        color=SERIES[0],
        alpha=0.25,
        lw=0,
        label="open field, simulated (±2 sigma)",
    )
    ax.plot(x, d_dij / peak * 100.0, color=SERIES[2], lw=1.4, label="Dij columns, unit weights")
    ax.set_xlabel("x (cm)", fontsize=9, color=INK)
    ax.set_ylabel("dose (% of open-field max)", fontsize=9, color=INK)
    ax.set_title(
        f"fluence sum at {PROFILE_DEPTH_CM:.0f} cm depth\ncolumns partition the field",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def wedge_panel(ax: Axes, data: dict) -> None:
    """Recombine a wedge plan as a matrix product: weights ramp across x, dose follows."""
    grid: VoxelGrid = data["grid"]
    dij = data["dij"]
    iy = SHAPE[1] // 2
    iz = int(PROFILE_DEPTH_CM / grid.spacing[2])

    ramp = np.linspace(0.2, 1.0, N_X)
    weights = np.repeat(ramp, N_Y)  # x-major beamlet indexing: j = jx * n_y + jy
    wedge = dij.dose_for_weights(weights / weights.sum())
    x, d_wedge, _ = _profile(wedge, np.zeros_like(wedge), grid, iy, iz)
    peak = d_wedge.max()

    lattice: BeamletGridSource = data["lattice"]
    edges = [lattice.beamlet_bounds(jx * N_Y)[0] for jx in range(N_X)]
    edges.append(lattice.beamlet_bounds((N_X - 1) * N_Y)[1])
    ax.stairs(
        ramp / ramp.max() * 100.0,
        edges,
        color=INK_SECONDARY,
        lw=1.0,
        ls="--",
        label="beamlet weights (scaled)",
    )
    ax.plot(x, d_wedge / peak * 100.0, color=SERIES[1], lw=1.4, label="recombined dose")
    ax.set_xlabel("x (cm)", fontsize=9, color=INK)
    ax.set_ylabel("dose (% of wedge max)", fontsize=9, color=INK)
    ax.set_title(
        "a wedge plan without re-simulation\ndose = Dij @ weights",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")


def main() -> None:
    """Run everything and save the figure beside this script."""
    data = run_all()
    dij = data["dij"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURFACE)
    for ax in axes:
        _style(ax)
    column_panel(axes[0], data)
    fluence_sum_panel(axes[1], data)
    wedge_panel(axes[2], data)
    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved {out}")

    n_hist = dij.n_beamlets * dij.n_histories_per_beamlet
    fill = dij.indices.size / (dij.n_voxels * dij.n_beamlets)
    balance = (dij.energy_deposited + dij.energy_escaped) / dij.energy_emitted
    print(f"  device            : {data['device']}")
    print(f"  beamlets          : {dij.n_beamlets} ({N_X}x{N_Y}, 2x2 cm)")
    print(f"  histories         : {n_hist:,} ({dij.n_histories_per_beamlet:,}/beamlet)")
    print(
        f"  Dij wall clock    : {data['dij_seconds']:.2f} s ({n_hist / data['dij_seconds']:,.0f}/s)"
    )
    print(f"  sparsity          : {dij.indices.size:,} entries ({fill:.1%} fill)")
    print(f"  energy balance    : deposited+escaped = {balance:.9f} x emitted")


if __name__ == "__main__":
    main()
