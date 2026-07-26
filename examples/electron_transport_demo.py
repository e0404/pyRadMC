"""Condensed-history electron transport on the reference backend.

Renders one figure (``electron_transport_demo.png``, saved beside this script):

- **Left**: depth dose of a broad monoenergetic 6 MeV photon beam with electrons
  transported against the same beam in the KERMA approximation.
  The difference *is* the phase: the buildup region — surface dose of a few percent
  climbing over the secondary-electron range — only exists when electrons carry their
  energy downstream before depositing it.
- **Right**: broad monoenergetic electron beams at 2, 5 and 10 MeV. Textbook shapes:
  sub-surface peak, steep distal falloff with R50 at ~0.8 of the CSDA range (marked),
  and only the bremsstrahlung tail beyond the practical range.

Run it from the repository root (requires the ``examples`` extra, i.e. matplotlib)::

    python examples/electron_transport_demo.py

Runtime is tens of seconds (the reference backend is deliberately unoptimized).
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource
from pyRadMC.rng.host import HostRNG

SEED = 20260711

# Reference dataviz palette (validated): categorical slots 1-3, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
SERIES = ["#2a78d6", "#1baf7a", "#eda100"]  # blue, aqua, yellow


def _style(ax: Axes) -> None:
    """Recessive panel styling shared by both axes."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def photon_panel(ax: Axes) -> None:
    """6 MeV photon PDD: electron transport vs the KERMA approximation."""
    grid = VoxelGrid.uniform_water(shape=(8, 8, 40), spacing=(2.0, 2.0, 0.2))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = ParallelBeamSource(energy=6.0, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    depth = (np.arange(40) + 0.5) * 0.2

    # KERMA deposits arrive in few large lumps, so that curve needs (cheap) extra
    # histories to look as smooth as the transport one.
    for label, transport_electrons, n_histories, color, style in [
        ("electrons transported", True, 30_000, SERIES[0], "-"),
        ("KERMA approximation", False, 100_000, INK_SECONDARY, "--"),
    ]:
        result = engine.run(
            source,
            n_histories=n_histories,
            n_batches=10,
            seed=SEED,
            transport_electrons=transport_electrons,
        )
        dose_z = result.dose.mean(axis=(0, 1))
        ax.plot(depth, dose_z / dose_z.max(), color=color, ls=style, lw=2, label=label)

    ax.annotate(
        "electron buildup",
        xy=(0.9, 0.55),
        xytext=(2.6, 0.32),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "lw": 0.8},
    )
    ax.set_xlabel("depth z (cm)", color=INK)
    ax.set_ylabel("dose (relative)", color=INK)
    ax.set_title("6 MeV photons: what electron transport adds", color=INK, fontsize=10)
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, 1.1)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")


def electron_panel(ax: Axes) -> None:
    """Broad electron beams at 2, 5, 10 MeV with CSDA-range markers."""
    xs_probe = AnalyticCrossSections()
    for energy, color in zip([2.0, 5.0, 10.0], SERIES, strict=True):
        r_csda = xs_probe.csda_range(energy, WATER)
        depth_cm = 1.4 * r_csda
        n_z = 36
        grid = VoxelGrid.uniform_water(shape=(8, 8, n_z), spacing=(2.0, 2.0, depth_cm / n_z))
        xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
        source = ParallelBeamSource(energy=energy, z=-0.5, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
        engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
        result = engine.run(
            source, n_histories=2_500, n_batches=10, seed=SEED, primary_kind="electron"
        )
        dose_z = result.dose.mean(axis=(0, 1))
        depth = (np.arange(n_z) + 0.5) * depth_cm / n_z
        ax.plot(depth, dose_z / dose_z.max(), color=color, lw=2)
        ax.axvline(r_csda, color=color, lw=1.0, ls=":", alpha=0.7)
        # Direct label just left of each curve's CSDA-range marker, clear of the peaks.
        ax.annotate(
            f"{energy:.0f} MeV",
            xy=(r_csda - 0.05, 0.72),
            color=INK,
            fontsize=9,
            ha="right",
        )

    ax.set_xlabel("depth z (cm)", color=INK)
    ax.set_ylabel("dose (relative)", color=INK)
    ax.set_title("Electron beams: R50 tracks the CSDA range (dotted)", color=INK, fontsize=10)
    ax.set_xlim(0.0, 7.0)
    ax.set_ylim(0.0, 1.12)


def main() -> None:
    """Run both panels and save the figure."""
    t0 = time.perf_counter()
    fig, (ax_photon, ax_electron) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), facecolor=SURFACE, constrained_layout=True
    )
    fig.suptitle(
        "pyRadMC — Class II condensed-history electron transport in water",
        color=INK,
        fontsize=12,
    )
    _style(ax_photon)
    _style(ax_electron)
    photon_panel(ax_photon)
    electron_panel(ax_electron)

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"figure written to {out} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
