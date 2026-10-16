"""Plan-dose uncertainty from the Dij: what the column sigmas can and cannot give you.

A plan is ``Dij @ weights``. The obvious uncertainty for it is the quadrature sum of the
column variances, ``sqrt(variance @ weights**2)`` — and whether that is *right* depends
on how the Dij was sampled. This script measures it, three ways, at three beamlet
pitches (2, 1 and 0.5 cm), for an open field at uniform weights:

1. a Dij with **independent** columns (``correlated=False``): plan dose and quadrature
   sigma as a matrix-vector product;
2. the same Dij with **correlated** columns (``correlated=True``, the shipped default,
   which shares random streams across beamlets to make plan *differences* nearly
   noise-free): again plan dose and quadrature sigma — note the result caveat
   ``variance_csc`` raises;
3. a **direct forward simulation** of the open field with the same number of
   independent histories: dose and batch sigma from the transport itself.

Against those, the honest noise of each Dij plan is measured empirically: the same
Dij is recomputed under several seeds, and the seed-to-seed spread of the plan dose
is the variance the quadrature ought to reproduce.

Renders one figure (``dij_plan_sigma_demo.png``, saved beside this script):

- **Left**: the plan doses agree. Lateral profiles at 5 cm depth from both Dijs sit
  inside the direct run's error band — the fluence-sum identity holds in either
  sampling mode, so correlated sampling is not a bias. The correlated profile looks
  *smoother*, not noisier: every beamlet replays the same histories, so its noise is
  spatially coherent rather than smaller — that coherence is the covariance the
  middle panel measures.
- **Middle**: the variance ratios versus beamlet pitch. *Empirical / quadrature* is 1
  for independent columns at every pitch: the column sigmas of an independent Dij
  combine in quadrature into an exact plan sigma. For correlated columns it rises above
  1 as the beamlets get finer: shared streams make corresponding histories of
  neighbouring beamlets translates of each other, and their deposits in one voxel are
  *positively* correlated, which inflates the variance of a **sum** exactly as it
  deflates the variance of a difference. The quadrature cannot see that covariance —
  and numerically it is the *same* estimate in both modes, because the marginal column
  variances do not change — so it under-reports the noise of a correlated plan by the
  full covariance term. *Quadrature / direct* is 1 in both modes: the quadrature equals
  the variance an independent-history run has, which is what a correlated plan does
  not have.
- **Right**: the pull ``(plan - direct) / sqrt(sigma_quad**2 + sigma_direct**2)`` at
  0.5 cm pitch, pooled over the field core and the seeds. Independent columns give a
  unit normal; correlated columns give a visibly wider one — the same finding, seen as
  an over-dispersed test statistic.

Measured on the shipped physics (6 MeV, water, 0.4 cm voxels; CUDA, 1.6e7 histories
per Dij, 6 seeds): correlated empirical/quadrature = 1.02 at 2 cm pitch, 1.07 at 1 cm,
1.4 at 0.5 cm (1.46 in an 8-seed rerun), against 1.00 for independent columns at every
pitch; quadrature/direct is 1.00 in both modes. The take-away for a
planning adapter: ``variance_csc`` from an *independent* Dij is a valid plan-variance
source; from a *correlated* Dij it is a lower bound whose shortfall grows with beamlet
resolution, and the caveat it warns with is not decorative.

Run it from the repository root (requires the ``examples`` and ``warp`` extras)::

    python examples/dij_plan_sigma_demo.py

Runtime is under a minute on a laptop GPU; without CUDA it falls back to Warp-CPU with
far fewer histories, and the ratios become correspondingly noisy.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.axes import Axes

from pyradmc.backends.warp.engine import WarpEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, ParallelBeamSource
from pyradmc.scoring.dij import DijResult

SEED = 20260711

# Reference dataviz palette (validated): categorical slots 1-3, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
SERIES = ["#2a78d6", "#1baf7a", "#eda100"]  # blue, aqua, yellow
MODE_COLOR = {False: SERIES[0], True: SERIES[2]}
MODE_LABEL = {False: "independent columns", True: "correlated columns"}

SHAPE = (64, 64, 64)
SPACING = (0.4, 0.4, 0.4)  # 25.6 cm water cube
FIELD = (2.8, 22.8)  # 20 x 20 cm field, 2 cm margin to the phantom edge
PITCHES_CM = (2.0, 1.0, 0.5)  # 10x10, 20x20 and 40x40 beamlets over the field
ENERGY = 6.0
N_BATCHES = 8
N_SEEDS = 6
PROFILE_DEPTH_CM = 5.0


def _style(ax: Axes) -> None:
    """Recessive panel styling shared by all axes."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def core_mask(grid: VoxelGrid) -> np.ndarray:
    """Voxels 2 cm inside the field edges, between 1.5 and 20 cm depth."""
    x = grid.origin[0] + (np.arange(SHAPE[0]) + 0.5) * grid.spacing[0]
    y = grid.origin[1] + (np.arange(SHAPE[1]) + 0.5) * grid.spacing[1]
    z = grid.origin[2] + (np.arange(SHAPE[2]) + 0.5) * grid.spacing[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    inside = (xx > FIELD[0] + 2.0) & (xx < FIELD[1] - 2.0)
    inside &= (yy > FIELD[0] + 2.0) & (yy < FIELD[1] - 2.0)
    return inside & (zz > 1.5) & (zz < 20.0)


def quadrature_variance(dij: DijResult, weights: np.ndarray) -> np.ndarray:
    """``variance @ weights**2`` as a dense grid — the obvious plan variance.

    ``variance_csc`` warns when the Dij is correlated; the caveat is the point of
    this script, so it is surfaced once on the console rather than silenced.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        variance = dij.variance_csc()
    for w in caught:
        if not getattr(quadrature_variance, "_warned", False):
            print(f"  [{w.category.__name__} from variance_csc] {w.message}")
            quadrature_variance._warned = True  # type: ignore[attr-defined]
    return np.asarray(variance @ (weights**2)).reshape(dij.grid_shape)


def run_pitch(engine: WarpEngine, pitch_cm: float, n_total: int) -> dict:
    """Both Dij modes over ``N_SEEDS`` seeds plus one direct run, at one beamlet pitch."""
    n_side = round((FIELD[1] - FIELD[0]) / pitch_cm)
    lattice = BeamletGridSource(
        energy=ENERGY, z=-1.0, x_range=FIELD, y_range=FIELD, n_x=n_side, n_y=n_side
    )
    n_beamlets = lattice.n_beamlets
    n_per = max(N_BATCHES, (n_total // n_beamlets) // N_BATCHES * N_BATCHES)
    weights = np.full(n_beamlets, 1.0 / n_beamlets)

    out: dict = {"pitch": pitch_cm, "n_beamlets": n_beamlets, "n_per": n_per}
    for correlated in (False, True):
        plans, quads = [], []
        for s in range(N_SEEDS):
            dij = engine.run_dij(
                lattice,
                n_histories_per_beamlet=n_per,
                n_batches=N_BATCHES,
                seed=SEED + s,
                correlated=correlated,
            )
            plans.append(dij.dose_for_weights(weights))
            quads.append(quadrature_variance(dij, weights))
        out[correlated] = {"plan": np.stack(plans), "quad": np.stack(quads)}

    open_field = ParallelBeamSource(energy=ENERGY, z=-1.0, x_range=FIELD, y_range=FIELD)
    out["direct"] = engine.run(
        open_field, n_histories=n_beamlets * n_per, n_batches=N_BATCHES, seed=SEED + 999
    )
    return out


def empirical_over_quadrature(
    plan: np.ndarray, quad: np.ndarray, mask: np.ndarray, keep: np.ndarray
) -> float:
    """Seed-to-seed plan variance over the mean quadrature variance, summed over ``mask``."""
    v_emp = np.var(plan[keep], axis=0, ddof=1)
    return float(v_emp[mask].sum() / quad[keep].mean(axis=0)[mask].sum())


def summarize(result: dict, mask: np.ndarray) -> dict:
    """Variance ratios over the core, with leave-one-seed-out error bars, and pull rms."""
    direct = result["direct"]
    var_direct = direct.dose_sigma**2
    summary = {}
    for correlated in (False, True):
        plan, quad = result[correlated]["plan"], result[correlated]["quad"]
        m = mask & (quad.mean(axis=0) > 0.0) & (var_direct > 0.0)
        every = np.ones(N_SEEDS, dtype=bool)
        jack = []
        for s in range(N_SEEDS):
            keep = every.copy()
            keep[s] = False
            jack.append(empirical_over_quadrature(plan, quad, m, keep))
        jack_arr = np.asarray(jack)
        err = float(np.sqrt((N_SEEDS - 1) / N_SEEDS * np.sum((jack_arr - jack_arr.mean()) ** 2)))
        pull = (plan - direct.dose)[:, m] / np.sqrt(quad + var_direct)[:, m]
        summary[correlated] = {
            "emp_over_quad": empirical_over_quadrature(plan, quad, m, every),
            "emp_over_quad_err": err,
            "quad_over_direct": float(quad.mean(axis=0)[m].sum() / var_direct[m].sum()),
            "pull": pull.ravel(),
            "pull_rms": float(np.sqrt(np.mean(pull**2))),
        }
    return summary


def dose_panel(ax: Axes, grid: VoxelGrid, result: dict) -> None:
    """Lateral plan-dose profiles from both Dijs against the direct run's band."""
    direct = result["direct"]
    iy = SHAPE[1] // 2
    iz = int(PROFILE_DEPTH_CM / grid.spacing[2])
    x = grid.origin[0] + (np.arange(SHAPE[0]) + 0.5) * grid.spacing[0]
    d_ref = direct.dose[:, iy, iz]
    s_ref = direct.dose_sigma[:, iy, iz]
    peak = d_ref.max()
    ax.fill_between(
        x,
        (d_ref - 2.0 * s_ref) / peak * 100.0,
        (d_ref + 2.0 * s_ref) / peak * 100.0,
        color=INK_SECONDARY,
        alpha=0.25,
        lw=0,
        label="direct simulation (±2 sigma)",
    )
    for correlated, ls in ((False, "-"), (True, "--")):
        plan = result[correlated]["plan"][0][:, iy, iz]
        ax.plot(
            x,
            plan / peak * 100.0,
            color=MODE_COLOR[correlated],
            lw=1.3,
            ls=ls,
            label=f"Dij @ weights, {MODE_LABEL[correlated]}",
        )
    ax.set_xlabel("x (cm)", fontsize=9, color=INK)
    ax.set_ylabel("plan dose (% of max)", fontsize=9, color=INK)
    ax.set_title(
        f"plan dose at {PROFILE_DEPTH_CM:.0f} cm depth, {result['pitch']:g} cm beamlets"
        "\nboth samplings are unbiased",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower center")


def ratio_panel(ax: Axes, summaries: list[tuple[float, dict]]) -> None:
    """Variance ratios versus beamlet pitch for both sampling modes."""
    pitches = [p for p, _ in summaries]
    for correlated in (False, True):
        emp = [s[correlated]["emp_over_quad"] for _, s in summaries]
        err = [s[correlated]["emp_over_quad_err"] for _, s in summaries]
        ax.errorbar(
            pitches,
            emp,
            yerr=err,
            color=MODE_COLOR[correlated],
            marker="o",
            ms=5,
            lw=1.4,
            capsize=3,
            label=f"empirical / quadrature, {MODE_LABEL[correlated]}",
        )
        qd = [s[correlated]["quad_over_direct"] for _, s in summaries]
        ax.plot(
            pitches,
            qd,
            color=MODE_COLOR[correlated],
            marker="s",
            ms=4,
            lw=0.9,
            ls=":",
            alpha=0.8,
            label=f"quadrature / direct, {MODE_LABEL[correlated]}",
        )
    ax.axhline(1.0, color=INK_SECONDARY, lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(pitches)
    ax.set_xticklabels([f"{p:g}" for p in pitches])
    ax.invert_xaxis()
    ax.set_xlabel("beamlet pitch (cm)", fontsize=9, color=INK)
    ax.set_ylabel("plan-variance ratio (field core)", fontsize=9, color=INK)
    ax.set_title(
        "quadrature is exact for independent columns\nand a lower bound for correlated ones",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper left")


def pull_panel(ax: Axes, pitch: float, summary: dict) -> None:
    """Pull histograms of Dij plan vs direct run for both modes, with the unit normal."""
    bins = np.linspace(-5.0, 5.0, 61)
    for correlated in (False, True):
        rms = summary[correlated]["pull_rms"]
        ax.hist(
            summary[correlated]["pull"],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.4,
            color=MODE_COLOR[correlated],
            label=f"{MODE_LABEL[correlated]} (rms {rms:.2f})",
        )
    z = np.linspace(-5.0, 5.0, 301)
    ax.plot(z, np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi), color=INK_SECONDARY, lw=0.9, ls="--")
    ax.set_xlabel("(plan - direct) / sqrt(sigma_quad² + sigma_direct²)", fontsize=9, color=INK)
    ax.set_ylabel("density", fontsize=9, color=INK)
    ax.set_title(
        f"pull at {pitch:g} cm beamlets\nunit normal dashed",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)


def main() -> None:
    """Run every pitch on the best available device, save the figure, print the table."""
    grid = VoxelGrid.uniform_water(shape=SHAPE, spacing=SPACING)
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    cuda = wp.is_cuda_available()
    device = "cuda:0" if cuda else "cpu"
    n_total = 16_000_000 if cuda else 64_000
    engine = WarpEngine(grid=grid, cross_sections=xs, device=device)
    warm = BeamletGridSource(energy=ENERGY, z=-1.0, x_range=FIELD, y_range=FIELD, n_x=2, n_y=2)
    engine.run_dij(warm, n_histories_per_beamlet=N_BATCHES, n_batches=N_BATCHES, seed=SEED)

    mask = core_mask(grid)
    t0 = time.perf_counter()
    results = [run_pitch(engine, pitch, n_total) for pitch in PITCHES_CM]
    seconds = time.perf_counter() - t0
    summaries = [(r["pitch"], summarize(r, mask)) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), facecolor=SURFACE)
    for ax in axes:
        _style(ax)
    dose_panel(axes[0], grid, results[-1])
    ratio_panel(axes[1], summaries)
    pull_panel(axes[2], results[-1]["pitch"], summaries[-1][1])
    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved {out}")

    print(f"  device            : {device}")
    print(f"  histories per Dij : {n_total:,} nominal, {N_SEEDS} seeds, {N_BATCHES} batches")
    print(f"  wall clock        : {seconds:.1f} s for everything")
    print("  pitch  beamlets  columns      empirical/quadrature  quadrature/direct  pull rms")
    for r, (_, s) in zip(results, summaries, strict=True):
        for correlated in (False, True):
            row = s[correlated]
            mode = "correlated" if correlated else "independent"
            print(
                f"  {r['pitch']:>4g}   {r['n_beamlets']:>6d}   {mode:<11}"
                f"  {row['emp_over_quad']:.3f} +- {row['emp_over_quad_err']:.3f}"
                f"           {row['quad_over_direct']:.3f}          {row['pull_rms']:.3f}"
            )


if __name__ == "__main__":
    main()
