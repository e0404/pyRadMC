"""What the pair and positron models change in a photon dose calculation.

Pair production and the positron each have two registered models
(``PhysicsOptions(pair=..., positron=...)``):

- ``pair="uniform-split"``: the pair's kinetic energy is shared uniformly and both
  leptons leave along the photon. ``pair="bethe-heitler"``: screened Bethe-Heitler
  sharing, each lepton at its own angle of order ``1/gamma``.
- ``positron="moller-at-rest"``: a positron is transported as an electron and
  annihilates where it stops. ``positron="bhabha-in-flight"``: its own collision
  stopping power, Bhabha delta rays, and two-photon annihilation in flight.

``"auto"`` selects the first of each up to ``HIGH_ENERGY_PHYSICS_THRESHOLD_MEV`` and
the second above it, so every beam here runs the first two by default and the others
are an explicit choice. This script measures what each choice does to depth dose in a
column wide enough to collect everything laterally (a pencil beam scored over the
whole cross-section is the central-axis depth dose of a broad beam):

- **6 MeV photons** in water: pair production is a tenth of the interactions.
- **18 MeV photons** in water: pair production is 41 percent of the interactions and
  the build-up region is made by its leptons.
- **18 MeV photons through lung**: 5 cm of water, 8 cm at 0.26 g/cm^3, water again;
  the positron-sensitive case, with a re-build-up behind the lung.

Renders ``pair_positron_model_demo.png`` beside this script and prints the measured
numbers. Top row: depth dose under the clinical models. Bottom row: the change each
alternative makes, in percent of the maximum dose, with its one-sigma band.

What it shows (GPU run, eight replicates of 3 to 4 million photons per model set; the
numbers are printed). The Bethe-Heitler pair model raises the build-up, by up to
0.8 percent of the maximum dose at 6 MeV and 1.4 percent at 18 MeV (0.83 +- 0.06
percent averaged over the build-up region), and leaves the dose beyond the maximum
unchanged within 0.1 percent: leptons that leave at an angle and share the energy more
evenly stop shallower. The positron model lowers the dose at every depth, by 0.16
percent of the maximum at 6 MeV and 0.8 to 1.1 percent at 18 MeV: a positron that
annihilates in flight hands the kinetic energy it had left to photons, most of which
leave the column. Together the two partly cancel in the build-up and the positron
effect remains behind it; the lung slab changes neither picture by more than a few
tenths of a percent.

Every uncertainty is the spread of independent replicate runs; the model sets use
disjoint seeds. The comparison above the threshold needs cross-section data that
reaches there and is measured on a tabulated compile in
``tests/validation/test_pair_positron_high_energy.py``.

Run it from the repository root (requires the ``examples`` and ``warp`` extras)::

    python examples/pair_positron_model_demo.py

Runs on the GPU when one is available, else on the CPU with fewer histories; under a
minute either way once the kernels are compiled.
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
COLUMN_WIDTH_CM = 60.0
CLINICAL = "clinical models"
MODEL_SETS = {
    CLINICAL: PhysicsOptions(pair="uniform-split", positron="moller-at-rest"),
    "pair: bethe-heitler": PhysicsOptions(pair="bethe-heitler"),
    "positron: bhabha-in-flight": PhysicsOptions(positron="bhabha-in-flight"),
    "both": PhysicsOptions(pair="bethe-heitler", positron="bhabha-in-flight"),
}

# Reference dataviz palette: categorical slots 1 to 3 in their fixed order, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
COLORS = {
    "pair: bethe-heitler": "#2a78d6",
    "positron: bhabha-in-flight": "#eb6834",
    "both": "#1baf7a",
}
LINESTYLES = {"pair: bethe-heitler": "-", "positron: bhabha-in-flight": "--", "both": "-."}


@dataclass(frozen=True)
class Case:
    """One beam and phantom: energy, depth binning, an optional lung slab, histories."""

    title: str
    energy_mev: float
    depth_cm: float
    n_bins: int
    lung_cm: tuple[float, float] | None
    histories_gpu: int
    histories_cpu: int


CASES = (
    Case("6 MeV photons, water", 6.0, 20.0, 40, None, 4_000_000, 60_000),
    Case("18 MeV photons, water", 18.0, 20.0, 40, None, 3_000_000, 40_000),
    Case("18 MeV photons, water / lung / water", 18.0, 20.0, 40, (5.0, 13.0), 3_000_000, 40_000),
)


def depth_doses(case: Case, device: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Depth dose per replicate and model set, in MeV/g per primary.

    Returns the bin-centre depths and, per model set, an array ``(REPLICATES, n_bins)``.
    """
    spacing = case.depth_cm / case.n_bins
    shape = (1, 1, case.n_bins)
    depth = (np.arange(case.n_bins) + 0.5) * spacing
    density = np.ones(shape)
    if case.lung_cm is not None:
        density[0, 0, (depth > case.lung_cm[0]) & (depth < case.lung_cm[1])] = 0.26
    grid = VoxelGrid(
        shape=shape,
        spacing=(COLUMN_WIDTH_CM, COLUMN_WIDTH_CM, spacing),
        density=density,
        material=np.zeros(shape, dtype=np.int32),
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
    for m, (name, physics) in enumerate(MODEL_SETS.items()):
        doses[name] = np.array(
            [
                engine.run(
                    beam, n_histories=n, n_batches=4, seed=SEED + 1_000 * m + r, physics=physics
                ).dose[0, 0]
                for r in range(REPLICATES)
            ]
        )
    return depth, doses


def mean_and_error(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replicate mean and its standard error, per depth bin."""
    return samples.mean(axis=0), samples.std(axis=0, ddof=1) / np.sqrt(samples.shape[0])


def changes(doses: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each alternative minus the clinical set, in percent of the clinical maximum."""
    base, base_err = mean_and_error(doses[CLINICAL])
    scale = 100.0 / base.max()
    out = {}
    for name in COLORS:
        mean, err = mean_and_error(doses[name])
        out[name] = (scale * (mean - base), scale * np.hypot(err, base_err))
    return out


def report(case: Case, depth: np.ndarray, doses: dict[str, np.ndarray]) -> None:
    """Print the build-up and beyond-maximum changes with their replicate errors."""
    base, _ = mean_and_error(doses[CLINICAL])
    peak = int(np.argmax(base))
    print(f"{case.title}: maximum at {depth[peak]:.2f} cm")
    for name in COLORS:
        delta = 100.0 * (doses[name] - doses[CLINICAL].mean(axis=0)) / base.max()
        parts = []
        for label, region in (("build-up", slice(0, peak)), ("beyond", slice(peak, None))):
            if region == slice(0, 0):
                continue
            per_replicate = delta[:, region].mean(axis=1)
            base_scatter = (
                100.0 * (doses[CLINICAL] - doses[CLINICAL].mean(axis=0))[:, region] / base.max()
            ).mean(axis=1)
            error = np.hypot(per_replicate.std(ddof=1), base_scatter.std(ddof=1)) / np.sqrt(
                REPLICATES
            )
            parts.append(f"{label} {per_replicate.mean():+.2f} +- {error:.2f}")
        change, change_error = changes(doses)[name]
        worst = int(np.argmax(np.abs(change)))
        parts.append(
            f"largest {change[worst]:+.2f} +- {change_error[worst]:.2f} at {depth[worst]:.2f} cm"
        )
        print(f"  {name:28s} mean change, % of maximum: " + ", ".join(parts))


def style(ax: Axes) -> None:
    """Recessive grid and axes on the chart surface."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)


def main() -> None:
    """Run every case under every model set, print the numbers, save the figure."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    start = time.perf_counter()
    fig, axes = plt.subplots(
        2, len(CASES), figsize=(15.0, 7.2), sharex="col", gridspec_kw={"height_ratios": [1.0, 1.0]}
    )
    fig.patch.set_facecolor(SURFACE)
    for column, case in enumerate(CASES):
        depth, doses = depth_doses(case, device)
        report(case, depth, doses)
        top, bottom = axes[0, column], axes[1, column]
        base, _ = mean_and_error(doses[CLINICAL])
        top.plot(depth, base / base.max(), color=INK, linewidth=2.0, label=CLINICAL)
        if case.lung_cm is not None:
            for ax in (top, bottom):
                ax.axvspan(*case.lung_cm, color=GRID_LINE, alpha=0.6, linewidth=0.0)
            top.text(
                0.5 * sum(case.lung_cm), 0.08, "lung", color=INK_SECONDARY, ha="center", fontsize=9
            )
        top.set_title(case.title, color=INK, fontsize=11, loc="left")
        top.set_ylim(0.0, 1.08)
        for name, (delta, error) in changes(doses).items():
            bottom.fill_between(
                depth, delta - error, delta + error, color=COLORS[name], alpha=0.2, linewidth=0.0
            )
            bottom.plot(
                depth,
                delta,
                color=COLORS[name],
                linewidth=2.0,
                linestyle=LINESTYLES[name],
                label=name,
            )
        bottom.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
        bottom.set_xlabel("depth (cm)", color=INK_SECONDARY)
        for ax in (top, bottom):
            style(ax)
    axes[0, 0].set_ylabel("dose / maximum, clinical models", color=INK_SECONDARY)
    axes[1, 0].set_ylabel("change (% of maximum dose)", color=INK_SECONDARY)
    axes[0, 0].legend(frameon=False, labelcolor=INK, loc="lower right")
    axes[1, 0].legend(frameon=False, labelcolor=INK, loc="best", fontsize=9)
    fig.suptitle(
        "Pair and positron models: what an explicit choice changes below the "
        f"{HIGH_ENERGY_PHYSICS_THRESHOLD_MEV:g} MeV threshold of 'auto'",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out = Path(__file__).with_name("pair_positron_model_demo.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out} on {device} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
