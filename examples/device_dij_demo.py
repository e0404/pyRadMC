"""A Dij that never leaves the GPU: ``WarpEngine.run_dij_device``.

An optimizer that recomputes the Dij every iteration and keeps its dose matrix on
the device does not want the engine to copy every column to the host and assemble a
SciPy matrix there, only to upload it again. ``run_dij_device`` runs the same
computation as ``run_dij`` and hands the CSC arrays over as Warp arrays on the
computing device; ``to_host()`` reproduces ``run_dij`` bit for bit.

Renders one figure (``device_dij_demo.png``, saved beside this script):

- **Left**: a wedge plan ``D w`` at 5 cm depth, computed twice — on the device, by a
  ten-line Warp kernel straight from the device CSC arrays, and on the host, as
  ``run_dij(...).dose_csc() @ w``. The curves overlay; the largest difference is
  printed (the device sum uses float64 atomics, so it agrees to rounding, not bit
  for bit — the *matrices* are bit-identical, which the test suite pins).
- **Right**: wall time per call of the two routes to a plan dose, best of several
  interleaved repeats: ``run_dij`` + ``dose_csc()`` + SciPy product on the host,
  against ``run_dij_device`` + the device product. Transport is identical, so the
  gap is the readback and host assembly the device route skips. Timings on a
  shared or power-throttled laptop GPU are indicative only.

Run it from the repository root (requires the ``examples`` and ``warp`` extras)::

    python examples/device_dij_demo.py

Runtime is well under a minute on a laptop; it uses CUDA when available and falls
back to Warp-CPU (where "device" memory is host memory, so the saving shrinks).
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.axes import Axes

from pyradmc.backends.warp.engine import WarpEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource

SEED = 20261102

# Reference dataviz palette (validated; as in dij_demo.py): blue, aqua, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
SERIES = ["#2a78d6", "#1baf7a"]

SHAPE = (64, 64, 64)
SPACING = (0.4, 0.4, 0.4)  # 25.6 cm water cube, 262k voxels
FIELD = (2.8, 22.8)  # 20 x 20 cm field
N_X, N_Y = 10, 10  # 100 beamlets of 2 x 2 cm
ENERGY = 6.0
PROFILE_DEPTH_CM = 5.0
REPEATS = 5


@wp.kernel
def csc_matvec(
    indptr: wp.array(dtype=wp.int64),
    indices: wp.array(dtype=wp.int64),
    values: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
):
    """``out += D w`` for a CSC ``D``: one thread per column, atomic row updates."""
    j = wp.tid()
    w = weights[j]
    for k in range(int(indptr[j]), int(indptr[j + 1])):
        wp.atomic_add(out, int(indices[k]), w * values[k])


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def _wedge(n_x: int, n_y: int) -> np.ndarray:
    """Weights ramping 0.2 -> 1.0 across x (beamlet j = ix * n_y + iy)."""
    ramp = np.linspace(0.2, 1.0, n_x)
    return np.repeat(ramp, n_y).astype(np.float64)


def run_all() -> dict:
    """Time both routes to a wedge plan dose, interleaved, and keep the last plans."""
    grid = VoxelGrid.uniform_water(shape=SHAPE, spacing=SPACING)
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    lattice = BeamletGridSource(
        energy=ENERGY, z=-1.0, x_range=FIELD, y_range=FIELD, n_x=N_X, n_y=N_Y
    )
    cuda = wp.is_cuda_available()
    device = "cuda:0" if cuda else "cpu"
    n_per = 40_000 if cuda else 400
    kwargs = dict(n_histories_per_beamlet=n_per, n_batches=4, seed=SEED, beamlet_group_size=25)
    engine = WarpEngine(grid=grid, cross_sections=xs, device=device)
    weights = _wedge(N_X, N_Y)
    w_dev = wp.array(weights, dtype=wp.float64, device=device)

    def host_route() -> np.ndarray:
        dij = engine.run_dij(lattice, **kwargs)
        return dij.dose_csc() @ weights

    def device_route() -> wp.array:
        dij = engine.run_dij_device(lattice, **kwargs)
        out = wp.zeros(dij.n_voxels, dtype=wp.float64, device=device)
        wp.launch(
            csc_matvec,
            dim=dij.n_beamlets,
            inputs=[dij.indptr, dij.indices, dij.dose, w_dev, out],
            device=device,
        )
        wp.synchronize_device(device)
        return out

    host_route(), device_route()  # warm: kernel modules, resident tables
    host_times, device_times = [], []
    for _ in range(REPEATS):  # interleaved, so drift hits both routes alike
        t0 = time.perf_counter()
        plan_host = host_route()
        host_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        plan_dev = device_route()
        device_times.append(time.perf_counter() - t0)
    nnz = engine.run_dij_device(lattice, **kwargs).nnz
    return {
        "grid": grid,
        "device": device,
        "plan_host": plan_host.reshape(SHAPE),
        "plan_device": plan_dev.numpy().reshape(SHAPE),
        "host_s": min(host_times),
        "device_s": min(device_times),
        "nnz": nnz,
    }


def profile_panel(ax: Axes, data: dict) -> None:
    """Lateral profile of the plan dose from both routes, overlaid."""
    grid: VoxelGrid = data["grid"]
    iy = SHAPE[1] // 2
    iz = int(PROFILE_DEPTH_CM / SPACING[2])
    x = grid.origin[0] + (np.arange(SHAPE[0]) + 0.5) * SPACING[0]
    host = data["plan_host"][:, iy, iz]
    dev = data["plan_device"][:, iy, iz]
    ax.plot(x, host, color=SERIES[0], lw=2.0, label="host: run_dij + SciPy")
    ax.plot(
        x,
        dev,
        ls="none",
        marker="o",
        ms=4.5,
        mfc=SERIES[1],
        mec=SURFACE,
        mew=1.0,
        label="device: run_dij_device + Warp kernel",
    )
    rel = np.abs(data["plan_device"] - data["plan_host"]).max() / data["plan_host"].max()
    ax.set_xlabel("x (cm)", fontsize=9, color=INK)
    ax.set_ylabel("plan dose (MeV/g per history)", fontsize=9, color=INK)
    ax.set_title(
        f"wedge plan D w at {PROFILE_DEPTH_CM:g} cm depth\nmax |device - host| / max = {rel:.1e}",
        fontsize=10,
        color=INK,
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)


def timing_panel(ax: Axes, data: dict) -> None:
    """Wall time per call of the two routes; the gap is the host work skipped."""
    labels = ["run_dij\n+ dose_csc() @ w", "run_dij_device\n+ device D w"]
    values = [1e3 * data["host_s"], 1e3 * data["device_s"]]
    bars = ax.barh(labels, values, color=SERIES[0], height=0.5)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.0f} ms",
            va="center",
            fontsize=8,
            color=INK,
        )
    saved = values[0] - values[1]
    ax.set_xlim(0, max(values) * 1.3)
    ax.set_xlabel("wall time per call (ms, best of interleaved repeats)", fontsize=9, color=INK)
    ax.set_title(
        f"host readback + assembly avoided: {saved:.0f} ms\n"
        f"({data['nnz']:,} entries, {N_X * N_Y} columns, {data['device']})",
        fontsize=10,
        color=INK,
    )
    ax.invert_yaxis()
    _style(ax)


def main() -> None:
    """Run, plot, save the figure beside this script."""
    t0 = time.perf_counter()
    data = run_all()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    profile_panel(left, data)
    timing_panel(right, data)
    fig.tight_layout()
    out = Path(__file__).with_name("device_dij_demo.png")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(
        f"{data['device']}: host route {1e3 * data['host_s']:.0f} ms, "
        f"device route {1e3 * data['device_s']:.0f} ms per call; "
        f"total {time.perf_counter() - t0:.1f} s; wrote {out}"
    )


if __name__ == "__main__":
    main()
