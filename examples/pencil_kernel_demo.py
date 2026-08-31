"""Mono-energetic pencil-beam kernels in water, scored in cylindrical shells.

An infinitely narrow mono-energetic beam entering a semi-infinite water medium, with
dose binned by depth and by radius about the beam axis — the geometry a pencil-beam
kernel is defined on (Mackie et al., Med. Phys. 12(2):188-196, 1985,
doi:10.1118/1.595774; Ahnesjo, Med. Phys. 16(4):577-592, 1989, doi:10.1118/1.596360).

Renders one figure (``pencil_kernel_demo.png``, saved beside this script):

- **Left**: the photon kernel as a depth-radius map, log-scaled. Dose spans several
  decades between the axis and the scatter tail; the forward-leaning shape is the
  secondary electrons carrying energy downstream.
- **Centre**: radial profiles at three depths. Equal-ratio shells resolve the
  near-axis core, where essentially all of the structure lives, at a per-shell sigma
  that a Cartesian grid would need orders more histories to reach.
- **Right**: photon vs electron primaries at the same energy, as energy per unit
  depth. The electron kernel stops at its CSDA range (marked); the photon kernel
  builds up and then attenuates. ``primary_kind="electron"`` is the engine's existing
  range-validation instrument — electron *beams* as a clinical modality are out of
  scope (AGENTS.md 6).

The phantom is a water **box** that circumscribes the scored cylinder, not a cylinder
with vacuum outside: transport stays on the rectilinear grid, and the box keeps the
outermost shell under full lateral scatter conditions. The cylinder is a scoring
geometry only — transport never sees it.

Run it from the repository root (requires the ``examples`` extra, i.e. matplotlib)::

    python examples/pencil_kernel_demo.py

Runtime is well under a minute on the Warp CPU backend; it falls back to the
reference engine (slower) if ``warp`` is not installed.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource
from pyradmc.scoring.cylinder import (
    CylindricalScoringGrid,
    geometric_edges,
    graded_edges,
)

SEED = 20260711
ENERGY = 6.0  # MeV
N_HISTORIES = 40_000
N_BATCHES = 8

# CSDA range of a 6 MeV electron in water, ICRU 37 / ESTAR: 3.052 g/cm^2.
CSDA_RANGE_CM = 3.05

# Reference dataviz palette (validated): categorical slots 1-3, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
SERIES = ["#2a78d6", "#1baf7a", "#eda100"]  # blue, aqua, yellow


def _style(ax: Axes) -> None:
    """Recessive panel styling shared by every axis."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def _phantom() -> tuple[VoxelGrid, tuple[float, float]]:
    """Water box 16 x 16 x 10 cm, and the lateral centre the beam enters on."""
    grid = VoxelGrid.uniform_water(shape=(64, 64, 50), spacing=(0.25, 0.25, 0.2))
    return grid, (8.0, 8.0)


def _engine(grid: VoxelGrid):
    """Build the Warp CPU backend if available, else the reference oracle."""
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    try:
        from pyradmc.backends.warp.engine import WarpEngine
    except ImportError:
        from pyradmc.backends.ref.engine import ReferenceEngine
        from pyradmc.rng.host import HostRNG

        return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    return WarpEngine(grid=grid, cross_sections=xs, device="cpu")


def _kernel(primary_kind: str) -> tuple[CylindricalScoringGrid, np.ndarray, np.ndarray]:
    """Run one pencil-beam kernel; returns the binning, dose and its sigma."""
    grid, axis = _phantom()
    cylinder = CylindricalScoringGrid.for_grid(
        grid,
        # Equal-ratio shells from 0.5 mm out to 6 cm, plus the central disc: constant
        # *relative* radial resolution, which is the natural binning for a quantity
        # falling roughly as a power law.
        radial_edges=geometric_edges(r_max=6.0, n_shells=28, r_min=0.05),
        axis=axis,
        # Graded depth: fine through the build-up, coarse where the curve is flat.
        depth_edges=graded_edges([(0.0, 2.0, 0.05), (2.0, 10.0, 0.25)]),
    )
    source = PencilBeamSource(
        energy=ENERGY, position=(axis[0], axis[1], -0.1), direction=(0.0, 0.0, 1.0)
    )
    result = _engine(grid).run(
        source,
        n_histories=N_HISTORIES,
        n_batches=N_BATCHES,
        seed=SEED,
        scoring_grid=cylinder,
        primary_kind=primary_kind,
    )
    return cylinder, result.dose, result.dose_sigma


def map_panel(ax: Axes, cyl: CylindricalScoringGrid, dose: np.ndarray) -> None:
    """Draw the photon kernel as a depth-radius map, log-scaled over its range."""
    floor = dose[dose > 0.0].max() * 1e-5
    mesh = ax.pcolormesh(
        cyl.radial_edges,
        cyl.depth_edges,
        np.maximum(dose, floor),
        norm=LogNorm(vmin=floor, vmax=dose.max()),
        cmap="magma",
        shading="flat",
    )
    bar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label("dose (MeV/g per history)", color=INK, fontsize=8)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=7)
    # Log radius: the shells are geometric, so this makes them equal-width columns.
    # The central disc spans r = 0 to r_min, which has no place on a log axis; it is
    # drawn running off the left spine rather than given a decade of its own, which a
    # symlog linear region would do and would badly overstate its extent.
    ax.set_xscale("log")
    ax.set_xlim(0.6 * cyl.radial_edges[1], cyl.radial_edges[-1])
    ax.invert_yaxis()
    ax.set_xlabel("radius r (cm)", color=INK)
    ax.set_ylabel("depth z (cm)", color=INK)
    ax.set_title(f"{ENERGY:.0f} MeV photon pencil kernel", color=INK, fontsize=10)


def radial_panel(
    ax: Axes, cyl: CylindricalScoringGrid, dose: np.ndarray, sigma: np.ndarray
) -> None:
    """Radial profiles at three depths, with the batch-estimated 1-sigma band."""
    radius = cyl.radial_centers
    for color, depth_cm in zip(SERIES, (0.5, 2.0, 5.0), strict=True):
        iz = int(np.argmin(np.abs(cyl.depth_centers - depth_cm)))
        profile = dose[iz]
        band = sigma[iz]
        ax.plot(radius, profile, color=color, lw=1.4, label=f"z = {cyl.depth_centers[iz]:.1f} cm")
        ax.fill_between(
            radius, profile - band, profile + band, color=color, alpha=0.25, linewidth=0
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("radius r (cm)", color=INK)
    ax.set_ylabel("dose (MeV/g per history)", color=INK)
    ax.set_title("Radial profiles (band: 1 sigma)", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)


def depth_panel(ax: Axes, kernels: dict[str, tuple[CylindricalScoringGrid, np.ndarray]]) -> None:
    """Energy per unit depth: the photon build-up against the electron range."""
    for color, (kind, (cyl, dose)) in zip(SERIES[: len(kernels)], kernels.items(), strict=True):
        # Integrate the kernel over the shells: sum(dose * mass) is the energy in
        # that depth bin, per history. Dividing by the bin thickness makes the two
        # curves comparable independently of the binning.
        per_depth = (dose * cyl.voxel_mass).sum(axis=1) / cyl.depth_thickness
        ax.plot(cyl.depth_centers, per_depth, color=color, lw=1.6, label=f"{kind} primary")
    ax.axvline(CSDA_RANGE_CM, color=INK_SECONDARY, lw=1.0, ls=":")
    ax.annotate(
        f"electron CSDA range\n{CSDA_RANGE_CM} cm",
        xy=(CSDA_RANGE_CM, 0.55),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=8,
    )
    ax.set_xlabel("depth z (cm)", color=INK)
    ax.set_ylabel("energy per cm depth (MeV/cm per history)", color=INK)
    ax.set_title(f"{ENERGY:.0f} MeV photon vs electron primaries", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    ax.set_xlim(0.0, 10.0)


def main() -> None:
    """Run both kernels and save the figure."""
    t0 = time.perf_counter()
    photon_cyl, photon_dose, photon_sigma = _kernel("photon")
    electron_cyl, electron_dose, _ = _kernel("electron")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), facecolor=SURFACE, constrained_layout=True)
    fig.suptitle(
        "pyradmc — mono-energetic pencil-beam kernels in water (cylindrical scoring)",
        color=INK,
        fontsize=12,
    )
    for ax in axes:
        _style(ax)
    map_panel(axes[0], photon_cyl, photon_dose)
    radial_panel(axes[1], photon_cyl, photon_dose, photon_sigma)
    depth_panel(
        axes[2],
        {"photon": (photon_cyl, photon_dose), "electron": (electron_cyl, electron_dose)},
    )

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"figure written to {out} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
