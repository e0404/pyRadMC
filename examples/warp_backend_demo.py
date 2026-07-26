"""The Warp backend — CPU and CUDA from one physics source.

Renders one figure (``warp_backend_demo.png``, saved beside this script):

- **Left**: depth dose of a broad monoenergetic 6 MeV photon beam with full coupled
  photon-electron transport, computed three times — by the reference backend (the
  oracle, few histories, visible error band) and by the Warp backend on every
  available device (many histories, thin error band). The physics reads at a glance:
  identical buildup over the secondary-electron range, identical falloff, and the
  Warp curves sit inside the reference's uncertainty. Statistical equivalence, never
  bit equality (AGENTS.md 2.3).
- **Right**: throughput of the same workload per backend and device, histories per
  second on a log scale. The 1e6 histories/s design target on a GPU is
  marked.

Run it from the repository root (requires the ``examples`` and ``warp`` extras)::

    python examples/warp_backend_demo.py

Runtime is under a minute; most of it is the (deliberately unoptimized) reference
run and one-time kernel compilation.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.axes import Axes

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.backends.warp.engine import WarpEngine
from pyRadMC.data.analytic import AnalyticCrossSections
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

N_Z = 512
SPACING_Z = 0.1


def _style(ax: Axes) -> None:
    """Recessive panel styling shared by both axes."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def _problem() -> tuple[VoxelGrid, AnalyticCrossSections, ParallelBeamSource]:
    grid = VoxelGrid.uniform_water(shape=(128, 128, N_Z), spacing=(0.2, 0.2, SPACING_Z))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = ParallelBeamSource(energy=6.0, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
    return grid, xs, source


def _pdd(result) -> tuple[np.ndarray, np.ndarray]:
    """Central-axis-free depth dose: mean over the full slab cross-section."""
    dose = result.dose.mean(axis=(0, 1))
    sigma = np.sqrt((result.dose_sigma**2).sum(axis=(0, 1))) / (
        result.dose.shape[0] * result.dose.shape[1]
    )
    return dose, sigma


def run_all() -> tuple[dict, dict]:
    """One reference run plus one Warp run per available device, timed."""
    grid, xs, source = _problem()
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    throughput: dict[str, float] = {}

    ref_engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    n_ref = 4_000
    t0 = time.perf_counter()
    ref = ref_engine.run(source, n_histories=n_ref, n_batches=8, seed=SEED)
    throughput["ref (python)"] = n_ref / (time.perf_counter() - t0)
    curves["ref (python)"] = _pdd(ref)

    devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
    for device in devices:
        engine = WarpEngine(grid=grid, cross_sections=xs, device=device)
        engine.run(source, n_histories=1_000, n_batches=1, seed=SEED)  # compile + warm
        n = 40_000 if device == "cpu" else 1_000_000
        t0 = time.perf_counter()
        result = engine.run(source, n_histories=n, n_batches=10, seed=SEED)
        throughput[f"warp {device}"] = n / (time.perf_counter() - t0)
        curves[f"warp {device}"] = _pdd(result)

    return curves, throughput


def pdd_panel(ax: Axes, curves: dict) -> None:
    """Depth-dose overlay: all backends inside each other's error bands."""
    depth = (np.arange(N_Z) + 0.5) * SPACING_Z
    peak = max(dose.max() for dose, _ in curves.values())
    for color, (label, (dose, sigma)) in zip(SERIES, curves.items(), strict=False):
        ax.fill_between(
            depth,
            (dose - sigma) / peak * 100.0,
            (dose + sigma) / peak * 100.0,
            color=color,
            alpha=0.25,
            lw=0,
        )
        ax.plot(depth, dose / peak * 100.0, color=color, lw=1.4, label=label)
    ax.set_xlabel("depth in water (cm)", fontsize=9, color=INK)
    ax.set_ylabel("dose (% of maximum)", fontsize=9, color=INK)
    ax.set_title(
        "6 MeV broad beam: one physics source, three compilations",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def throughput_panel(ax: Axes, throughput: dict) -> None:
    """Histories/s per backend on a log scale, with the exit criterion marked."""
    labels = list(throughput)
    values = [throughput[k] for k in labels]
    bars = ax.barh(labels, values, color=SERIES[: len(labels)], height=0.55)
    ax.set_xscale("log")
    ax.axvline(1.0e6, color=INK_SECONDARY, lw=1.0, ls="--")
    ax.text(
        1.0e6 * 0.82,
        0.97,
        "1e6 histories/s target (GPU)",
        fontsize=7,
        color=INK_SECONDARY,
        rotation=90,
        va="top",
        ha="right",
        transform=ax.get_xaxis_transform(),
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value * 1.15,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:,.0f}/s",
            fontsize=8,
            color=INK,
            va="center",
        )
    ax.set_xlim(right=max(values) * 30.0)
    ax.set_xlabel("histories per second (same workload)", fontsize=9, color=INK)
    ax.set_title("throughput", fontsize=10, color=INK)


def main() -> None:
    """Run everything and save the figure beside this script."""
    curves, throughput = run_all()
    fig, (ax_pdd, ax_perf) = plt.subplots(1, 2, figsize=(11.0, 4.2), facecolor=SURFACE)
    _style(ax_pdd)
    _style(ax_perf)
    pdd_panel(ax_pdd, curves)
    throughput_panel(ax_perf, throughput)
    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved {out}")
    for label, value in throughput.items():
        print(f"  {label:>14}: {value:>12,.0f} histories/s")


if __name__ == "__main__":
    main()
