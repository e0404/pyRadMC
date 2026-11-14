"""What the bremsstrahlung model changes in a dose calculation — and what it does not.

Two bremsstrahlung models are registered (``PhysicsOptions(bremsstrahlung=...)``):

- ``"thin-target"``: a ``1/k`` photon spectrum, photons emitted along the electron.
  What ``"auto"`` selects up to ``HIGH_ENERGY_PHYSICS_THRESHOLD_MEV``.
- ``"bethe-heitler"``: the screened Bethe-Heitler spectrum, photons emitted at an
  angle of order ``1/gamma``. What ``"auto"`` selects above the threshold.

Both are normalized to the same radiative stopping power, so they radiate the same
energy and differ in how it is shared among photons: the screened spectrum emits more
photons, softer ones. This script measures what that does to depth dose, in a water
column wide enough to collect everything laterally (a pencil beam scored over the whole
cross-section is the central-axis depth dose of a broad beam):

- **6 MeV electrons**, the bottom of the clinical electron range: radiation is a few
  percent of the energy loss, and the effect is correspondingly small.
- **20 MeV electrons**, the top of the clinical range and of the analytic data's
  energy domain: the falloff and the bremsstrahlung tail beyond the practical range.
- **18 MeV photons**: the bremsstrahlung of the secondary electrons.

All three are below the threshold, so ``"auto"`` runs ``thin-target`` for each of them
and the ``bethe-heitler`` curves are an explicit choice. The comparison above the
threshold needs cross-section data that reaches there — the analytic source used here
ends at 20 MeV and the engines refuse a source beyond it — and is measured, on a
tabulated compile, in ``tests/validation/test_bethe_heitler_high_energy.py``.

Renders ``bremsstrahlung_model_demo.png`` beside this script and prints the measured
numbers. Top row: depth dose under both models. Bottom row: their difference in percent
of the maximum dose, with its one-sigma band.

What it shows (GPU run, eight replicates of 1e6 electrons or 8e6 photons per model):
softer photons mean fewer electrons lose a large part of their energy in one event, so
the electron falloff sharpens and moves shallower — R50 by 0.10 +- 0.01 mm at 6 MeV and
1.03 +- 0.01 mm at 20 MeV, up to 0.8 and 2.7 percent of the maximum dose in the
falloff — while the bremsstrahlung tail at 20 MeV rises by 6 percent of itself.
For 18 MeV photons the depth dose changes by at most 0.61 +- 0.06 percent of the maximum.

Every uncertainty is the spread of independent replicate runs, so derived quantities
(R50, the tail dose) carry one too; the two models use disjoint seeds.

Run it from the repository root (requires the ``examples`` and ``warp`` extras)::

    python examples/bremsstrahlung_model_demo.py

Runs on the GPU when one is available, else on the CPU with fewer histories; well under
a minute either way once the kernels are compiled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.axes import Axes

from pyradmc import HIGH_ENERGY_PHYSICS_THRESHOLD_MEV, PhysicsOptions
from pyradmc.backends.warp.engine import WarpEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource

SEED = 20260711
REPLICATES = 8
MODELS = ("thin-target", "bethe-heitler")
COLUMN_WIDTH_CM = 60.0

# Reference dataviz palette (validated): two categorical slots, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
COLORS = {"thin-target": "#2a78d6", "bethe-heitler": "#eda100"}


@dataclass(frozen=True)
class Case:
    """One beam: what is transported, how deep, and how many histories per replicate."""

    title: str
    energy_mev: float
    primary_kind: str
    depth_cm: float
    n_bins: int
    histories_gpu: int
    histories_cpu: int
    log_scale: bool


CASES = (
    Case("6 MeV electrons", 6.0, "electron", 4.8, 80, 1_000_000, 60_000, True),
    Case("20 MeV electrons", 20.0, "electron", 16.0, 80, 1_000_000, 30_000, True),
    Case("18 MeV photons", 18.0, "photon", 30.0, 60, 8_000_000, 250_000, False),
)


def depth_doses(case: Case, device: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Depth dose per replicate and model, in MeV/g per primary.

    Returns the bin-centre depths and, per model, an array ``(REPLICATES, n_bins)``.
    """
    spacing = case.depth_cm / case.n_bins
    grid = VoxelGrid.uniform_water(
        shape=(1, 1, case.n_bins), spacing=(COLUMN_WIDTH_CM, COLUMN_WIDTH_CM, spacing)
    )
    engine = WarpEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        device=device,
    )
    centre = 0.5 * COLUMN_WIDTH_CM
    beam = PencilBeamSource(
        energy=case.energy_mev, position=(centre, centre, -0.5), direction=(0.0, 0.0, 1.0)
    )
    n = case.histories_gpu if device.startswith("cuda") else case.histories_cpu
    doses: dict[str, np.ndarray] = {}
    for m, model in enumerate(MODELS):
        runs = []
        for r in range(REPLICATES):
            result = engine.run(
                beam,
                n_histories=n,
                n_batches=4,
                seed=SEED + 1000 * m + r,  # disjoint seeds: the two models are independent
                primary_kind=case.primary_kind,
                physics=PhysicsOptions(bremsstrahlung=model),
            )
            runs.append(result.dose[0, 0])
        doses[model] = np.array(runs)
    depth = (np.arange(case.n_bins) + 0.5) * spacing
    return depth, doses


def mean_and_error(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replicate mean and its standard error, along the replicate axis."""
    return samples.mean(axis=0), samples.std(axis=0, ddof=1) / np.sqrt(samples.shape[0])


def r50(depth: np.ndarray, dose: np.ndarray) -> float:
    """Depth of 50 percent of the maximum on the distal side, by linear interpolation."""
    peak = int(np.argmax(dose))
    half = 0.5 * dose[peak]
    below = peak + int(np.argmax(dose[peak:] < half))
    d0, d1 = dose[below - 1], dose[below]
    return float(depth[below - 1] + (d0 - half) / (d0 - d1) * (depth[below] - depth[below - 1]))


def report(case: Case, depth: np.ndarray, doses: dict[str, np.ndarray]) -> None:
    """Print the measured effect of the model choice for one beam."""
    thin, thin_err = mean_and_error(doses["thin-target"])
    screened, screened_err = mean_and_error(doses["bethe-heitler"])
    norm = thin.max()
    diff = 100.0 * (screened - thin) / norm
    diff_err = 100.0 * np.hypot(screened_err, thin_err) / norm
    worst = int(np.argmax(np.abs(diff) / diff_err))
    largest = int(np.argmax(np.abs(diff)))
    print(f"\n{case.title}")
    print(
        f"  largest difference: {diff[largest]:+.2f} +- {diff_err[largest]:.2f} % of maximum "
        f"at {depth[largest]:.1f} cm; most significant bin {diff[worst] / diff_err[worst]:+.1f} "
        f"sigma at {depth[worst]:.1f} cm"
    )
    chi2 = float(np.sum((diff / diff_err) ** 2))
    print(f"  all bins: chi2 = {chi2:.0f} on {depth.size} bins")

    def integral(samples: np.ndarray) -> np.ndarray:
        return samples.sum(axis=1)

    total = {model: mean_and_error(integral(doses[model])[:, None]) for model in MODELS}
    ratio = total["bethe-heitler"][0][0] / total["thin-target"][0][0]
    ratio_err = ratio * np.hypot(
        total["bethe-heitler"][1][0] / total["bethe-heitler"][0][0],
        total["thin-target"][1][0] / total["thin-target"][0][0],
    )
    print(f"  energy absorbed in the column, ratio: {ratio:.4f} +- {ratio_err:.4f}")

    if case.primary_kind != "electron":
        return
    ranges = {}
    for model in MODELS:
        per_replicate = np.array([r50(depth, d) for d in doses[model]])
        ranges[model] = mean_and_error(per_replicate[:, None])
    shift = ranges["bethe-heitler"][0][0] - ranges["thin-target"][0][0]
    shift_err = float(np.hypot(ranges["bethe-heitler"][1][0], ranges["thin-target"][1][0]))
    print(
        f"  R50: {ranges['thin-target'][0][0]:.3f} cm thin-target, "
        f"{ranges['bethe-heitler'][0][0]:.3f} cm bethe-heitler; "
        f"shift {10.0 * shift:+.2f} +- {10.0 * shift_err:.2f} mm"
    )
    # The bremsstrahlung tail: everything beyond 1.3 R50, past the practical range.
    tail = depth > 1.3 * ranges["thin-target"][0][0]
    tails = {model: mean_and_error(doses[model][:, tail].sum(axis=1)[:, None]) for model in MODELS}
    tail_ratio = tails["bethe-heitler"][0][0] / tails["thin-target"][0][0]
    tail_ratio_err = tail_ratio * np.hypot(
        tails["bethe-heitler"][1][0] / tails["bethe-heitler"][0][0],
        tails["thin-target"][1][0] / tails["thin-target"][0][0],
    )
    tail_level = 100.0 * thin[tail].mean() / norm
    print(
        f"  bremsstrahlung tail beyond {1.3 * ranges['thin-target'][0][0]:.1f} cm "
        f"(mean level {tail_level:.2f} % of maximum): "
        f"bethe-heitler / thin-target = {tail_ratio:.3f} +- {tail_ratio_err:.3f}"
    )


def _style(ax: Axes) -> None:
    """Recessive panel styling shared by all axes."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def plot(case: Case, depth: np.ndarray, doses: dict[str, np.ndarray], top: Axes, low: Axes) -> None:
    """Depth dose under both models (top) and their difference with its band (bottom)."""
    thin, thin_err = mean_and_error(doses["thin-target"])
    screened, screened_err = mean_and_error(doses["bethe-heitler"])
    norm = thin.max()
    for model, mean, err, style in (
        ("thin-target", thin, thin_err, "-"),
        ("bethe-heitler", screened, screened_err, "--"),
    ):
        top.fill_between(
            depth, (mean - err) / norm, (mean + err) / norm, color=COLORS[model], alpha=0.25, lw=0
        )
        top.plot(depth, mean / norm, color=COLORS[model], ls=style, lw=1.8, label=model)
    if case.log_scale:
        top.set_yscale("log")
        top.set_ylim(2.0e-4, 1.5)
    else:
        top.set_ylim(0.0, 1.1)
    above = case.energy_mev > HIGH_ENERGY_PHYSICS_THRESHOLD_MEV
    selected = "bethe-heitler" if above else "thin-target"
    top.set_title(f'{case.title}  ("auto": {selected})', color=INK, fontsize=10)
    top.set_ylabel("dose / maximum", color=INK)
    top.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper right")

    diff = 100.0 * (screened - thin) / norm
    diff_err = 100.0 * np.hypot(screened_err, thin_err) / norm
    low.axhline(0.0, color=INK_SECONDARY, lw=0.8)
    low.fill_between(depth, diff - diff_err, diff + diff_err, color=INK_SECONDARY, alpha=0.25, lw=0)
    low.plot(depth, diff, color=INK, lw=1.2)
    low.set_xlabel("depth (cm)", color=INK)
    low.set_ylabel(
        "bethe-heitler minus thin-target\n(% of maximum, band: 1 sigma)", color=INK, fontsize=8
    )
    for ax in (top, low):
        ax.set_xlim(0.0, case.depth_cm)


def main() -> None:
    """Run the three beams under both models, print the numbers, save the figure."""
    t0 = time.perf_counter()
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(f"device: {device}; {REPLICATES} replicates per model")

    fig, axes = plt.subplots(
        2,
        len(CASES),
        figsize=(13.0, 6.4),
        facecolor=SURFACE,
        constrained_layout=True,
        sharex="col",
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    fig.suptitle(
        "pyradmc — bremsstrahlung model: same radiated energy, different photons",
        color=INK,
        fontsize=12,
    )
    for column, case in enumerate(CASES):
        depth, doses = depth_doses(case, device)
        report(case, depth, doses)
        _style(axes[0, column])
        _style(axes[1, column])
        plot(case, depth, doses, axes[0, column], axes[1, column])

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"\nfigure written to {out} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
