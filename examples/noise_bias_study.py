"""Per-beamlet Dij noise vs. the bias of the optimized plan.

The question this answers (docs/decisions.md): when a plan is optimized on
a *noisy* Dij, how wrong is the plan — not the matrix — and how does that
shrink as per-beamlet statistics grow? The optimizer exploits noise: it sees
favourable fluctuations as real and shapes weights around them. Measured here
against high-statistics ground truth for both stream mappings (independent and
correlated), with the workflow of clinical practice in mind: the
final plan is *recalculated independently and renormalized*, so what matters
is the true quality of the delivered plan, not what the optimizer believed.

Design choices that keep the numbers honest:

- **Two independent ground truths.** The baseline plan is optimized on truth A;
  every endpoint — baseline and noisy plans alike — is evaluated on truth B
  (the "independent recalculation"). Truth A's own noise exploitation therefore
  cancels out of the bias definition instead of contaminating it.
- **The decision metric is the RMS true-endpoint error**, which combines the
  noise-exploitation bias with the across-seed spread of the plan. Correlated
  sampling trades less bias for more spread (its noise is coherent across
  columns), so neither moment alone can decide the default.
- **Renormalized endpoints.** Each plan is scaled so its recalculated PTV D98
  matches the baseline's — the clinical renormalization step. Dose is linear in
  the weights, so this is one factor per plan. If correlated-mode spread is
  mostly common-mode scale, renormalization removes it and the residual (OAR
  D2) decides; the spread of the factor itself measures how much of the error
  was pure scale.
- **The plan-difference experiment** (second figure, ``*_delta.png``): every
  realization is optimized twice — default and strict OAR penalty — and the
  predicted and delivered endpoint *deltas* between the two plans are compared
  to the ground-truth delta. Re-optimizing after a constraint change and
  reading off the trade is the planner's iteration loop; noise shared between
  the two evaluations cancels in the difference.

Setup: a 16 cm water cube, an 8 x 8 lattice of 1 x 1 cm 6 MeV beamlets, a
central 4 cm PTV box with an OAR box directly downstream — the simplest
geometry with a real coverage-vs-sparing trade-off for the toy objective of
:mod:`pyRadMC.study`.

Renders ``noise_bias_study.png`` and the raw per-realization table
``noise_bias_study.csv`` beside this script; ``--full`` switches to the
decision-grade configuration (32 seeds, a sixth statistics level at the 2-3
percent target sigma, deeper truths) and ``--phantom halfslab`` adds a lateral
lung-density slab upstream of the PTV — uniform water is correlated sampling's
best case, so the default decision needs the heterogeneous curves too. Output
files gain matching ``_halfslab`` / ``_full`` suffixes. Run from the
repository root (``examples`` and ``warp`` extras)::

    python examples/noise_bias_study.py [--full] [--phantom halfslab]

Quick mode stays around a minute on a laptop GPU (Warp-CPU fallback reduces
statistics); ``--full`` takes several minutes. Wall-clock numbers printed at
the end are subject to the laptop power-state caveat and are not benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.axes import Axes
from matplotlib.ticker import NullFormatter, StrMethodFormatter

from pyRadMC.backends.warp.engine import WarpEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletGridSource
from pyRadMC.scoring.dij import DijResult
from pyRadMC.study import ToyPlanProblem, dose_at_volume, optimize_weights

SEED = 20260711
SEED_TRUTH_B = SEED + 999_999  # the independent recalculation's stream space

# Reference dataviz palette (validated): categorical slots 1-2, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
SERIES = {"independent": "#2a78d6", "correlated": "#1baf7a"}  # fixed assignment

# The plan-difference experiment: every realization is optimized twice, at the
# default and at a stricter OAR penalty, and the *delta* between the two plans
# is compared to the ground-truth delta. Re-optimizing after a constraint
# change and reading off the trade is the planner's iteration loop. Note both
# plans are evaluated on the *same* matrix, so the realization's noise is
# shared between them in either stream mapping; what this experiment isolates
# is how the two re-optimizations respond to noise. Measured result: the
# mappings are at parity here — the correlated-sampling win lives in the
# renormalized single-plan quality, not in within-matrix deltas.
OAR_PENALTY_STRICT = 8.0

SHAPE = (32, 32, 32)
SPACING = (0.5, 0.5, 0.5)  # 16 cm water cube
FIELD = (4.0, 12.0)  # 8 x 8 cm field
N_X, N_Y = 8, 8  # 1 x 1 cm beamlets
ENERGY = 6.0
Z0 = -1.0
N_BATCHES = 8

# PTV: central 4 cm box, 4-8 cm deep. OAR: same cross-section, directly
# downstream at 9-12 cm — beamlets crossing the PTV also feed it, so coverage
# and sparing genuinely compete.
PTV_SLICES = (slice(12, 20), slice(12, 20), slice(8, 16))
OAR_SLICES = (slice(12, 20), slice(12, 20), slice(18, 24))

# The heterogeneous variant: a lung-density (0.3 g/cm^3) half-slab covering
# x < 8 cm at 2-4 cm depth, strictly upstream of the PTV. Uniform water is
# correlated sampling's best case (beamlets are exact translates); this breaks
# that symmetry — half the beamlets traverse lung, half don't, and shared-stream
# histories decorrelate at the interface — while the optimizer must modulate
# across it. Variable-density water is the project's heterogeneity idiom until
# Real materials are available via the tabulated backend.
SLAB_SLICES = (slice(0, 16), slice(0, 32), slice(4, 8))
SLAB_DENSITY = 0.3


def _masks() -> tuple[np.ndarray, np.ndarray]:
    ptv = np.zeros(SHAPE, dtype=bool)
    ptv[PTV_SLICES] = True
    oar = np.zeros(SHAPE, dtype=bool)
    oar[OAR_SLICES] = True
    return ptv.ravel(), oar.ravel()


def per_beamlet_sigma(dij: DijResult) -> float:
    """Mean over columns of the median relative sigma in each column's high-dose region.

    The high-dose region (>= 50 percent of the column maximum) is where the
    project's 2-3 percent target is defined (AGENTS.md 1).
    """
    rels = []
    for j in range(dij.n_beamlets):
        lo, hi = int(dij.indptr[j]), int(dij.indptr[j + 1])
        dose, sigma = dij.dose[lo:hi], dij.sigma[lo:hi]
        if dose.size == 0:
            continue
        high = dose >= 0.5 * dose.max()
        rels.append(float(np.median(sigma[high] / dose[high])))
    return float(np.mean(rels))


def _endpoints(matrix, w: np.ndarray, problem: ToyPlanProblem) -> dict:
    """PTV/OAR endpoints of one plan on one matrix, as fractions of Rx."""
    d = (matrix @ w).ravel()
    rx = problem.prescription
    return {
        "ptv_d98": dose_at_volume(d[problem.ptv], 98.0) / rx,
        "ptv_d2": dose_at_volume(d[problem.ptv], 2.0) / rx,
        "oar_d2": dose_at_volume(d[problem.oar], 2.0) / rx,
    }


def _optimize_pair(matrix, problem: ToyPlanProblem) -> tuple[np.ndarray, np.ndarray]:
    """Run the two optimizations of the plan-difference experiment on one matrix."""
    kwargs = dict(
        ptv=problem.ptv,
        oar=problem.oar,
        prescription=problem.prescription,
        oar_max=problem.oar_max,
    )
    w1 = optimize_weights(ToyPlanProblem(dij=matrix, oar_penalty=problem.oar_penalty, **kwargs))
    w2 = optimize_weights(ToyPlanProblem(dij=matrix, oar_penalty=OAR_PENALTY_STRICT, **kwargs))
    return w1, w2


def evaluate_plan(record: dict, dij: DijResult, truth_csc, problem: ToyPlanProblem) -> dict:
    """Optimize twice on ``dij``, evaluate both plans on noisy and recalc truth, record."""
    noisy_csc = dij.dose_csc()
    w1, w2 = _optimize_pair(noisy_csc, problem)
    true_1 = _endpoints(truth_csc, w1, problem)
    true_2 = _endpoints(truth_csc, w2, problem)
    app_1 = _endpoints(noisy_csc, w1, problem)
    app_2 = _endpoints(noisy_csc, w2, problem)
    return record | {
        "sigma_beamlet": per_beamlet_sigma(dij),
        "ptv_d98_true": true_1["ptv_d98"],
        "ptv_d2_true": true_1["ptv_d2"],
        "oar_d2_true": true_1["oar_d2"],
        "ptv_d98_apparent": app_1["ptv_d98"],
        "oar_d2_apparent": app_1["oar_d2"],
        "ptv_d98_true_p2": true_2["ptv_d98"],
        "oar_d2_true_p2": true_2["oar_d2"],
        "ptv_d98_apparent_p2": app_2["ptv_d98"],
        "oar_d2_apparent_p2": app_2["oar_d2"],
    }


def run_study(full: bool, phantom: str) -> tuple[list[dict], dict]:
    """All transport and optimization; returns per-realization records and the baseline."""
    grid = VoxelGrid.uniform_water(shape=SHAPE, spacing=SPACING)
    if phantom == "halfslab":
        grid.density[SLAB_SLICES] = SLAB_DENSITY
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    lattice = BeamletGridSource(energy=ENERGY, z=Z0, x_range=FIELD, y_range=FIELD, n_x=N_X, n_y=N_Y)
    ptv, oar = _masks()

    cuda = wp.is_cuda_available()
    device = "cuda:0" if cuda else "cpu"
    if cuda:
        # Levels bracket the 2-3 percent target sigma (AGENTS.md 1) from above;
        # the decisive 262k level (~2 percent) only in --full. All divisible by
        # the batch count.
        levels = [256, 1_024, 4_096, 16_384, 65_536] + ([262_144] if full else [])
        n_seeds = 32 if full else 8
        n_truth = 2_097_152 if full else 524_288
    else:
        levels = [128, 512, 2_048]
        n_seeds = 8 if full else 4
        n_truth = 20_000

    engine = WarpEngine(grid=grid, cross_sections=xs, device=device)
    engine.run_dij(lattice, n_histories_per_beamlet=500, n_batches=1, seed=SEED)  # warm

    t0 = time.perf_counter()
    truth_a = engine.run_dij(
        lattice, n_histories_per_beamlet=n_truth, n_batches=N_BATCHES, seed=SEED
    )
    truth_b = engine.run_dij(
        lattice, n_histories_per_beamlet=n_truth, n_batches=N_BATCHES, seed=SEED_TRUTH_B
    )
    truth_seconds = time.perf_counter() - t0
    truth_b_csc = truth_b.dose_csc()

    # The prescription is a fixed physical number taken from truth A: the mean
    # PTV dose at unit weights. Every problem below shares it.
    rx = float(np.mean((truth_a.dose_csc() @ np.ones(truth_a.n_beamlets)).ravel()[ptv]))
    problem = ToyPlanProblem(dij=truth_a.dose_csc(), ptv=ptv, oar=oar, prescription=rx)

    # Baseline: optimized on truth A, scored on truth B — same protocol as the
    # noisy realizations, so truth A's own noise exploitation cancels out of
    # every bias below. The strict-penalty twin defines the ground-truth plan
    # delta for the plan-difference experiment.
    w_base, w_base_strict = _optimize_pair(truth_a.dose_csc(), problem)
    base_1 = _endpoints(truth_b_csc, w_base, problem)
    base_2 = _endpoints(truth_b_csc, w_base_strict, problem)
    baseline = {
        "device": device,
        "full": full,
        "phantom": phantom,
        "levels": levels,
        "n_seeds": n_seeds,
        "n_truth": n_truth,
        "truth_seconds": truth_seconds,
        "sigma_beamlet": per_beamlet_sigma(truth_a),
        "ptv_d98_true": base_1["ptv_d98"],
        "ptv_d2_true": base_1["ptv_d2"],
        "oar_d2_true": base_1["oar_d2"],
        "delta_ptv_d98": base_2["ptv_d98"] - base_1["ptv_d98"],
        "delta_oar_d2": base_2["oar_d2"] - base_1["oar_d2"],
    }

    records: list[dict] = []
    for mode_name, correlated in (("independent", False), ("correlated", True)):
        for level_idx, n_per in enumerate(levels):
            t0 = time.perf_counter()
            for seed_idx in range(n_seeds):
                seed = SEED + 1 + level_idx * 100 + seed_idx  # disjoint from both truths
                dij = engine.run_dij(
                    lattice,
                    n_histories_per_beamlet=n_per,
                    n_batches=N_BATCHES,
                    seed=seed,
                    correlated=correlated,
                )
                records.append(
                    evaluate_plan(
                        {
                            "mode": mode_name,
                            "n_per": n_per,
                            "seed": seed,
                            "baseline_ptv_d98_true": baseline["ptv_d98_true"],
                            "baseline_oar_d2_true": baseline["oar_d2_true"],
                        },
                        dij,
                        truth_b_csc,
                        problem,
                    )
                )
            elapsed = time.perf_counter() - t0
            print(
                f"  {mode_name:>11s}  n_per={n_per:>7,}  {n_seeds} realizations  {elapsed:6.1f} s"
            )
    return records, baseline


def _aggregate(records: list[dict], baseline: dict) -> dict:
    """Per (mode, level): sigma, D98 error moments, renormalized OAR error, scale spread."""
    base_d98 = baseline["ptv_d98_true"]
    base_oar = baseline["oar_d2_true"]
    out: dict = {}
    for mode in SERIES:
        rows = [r for r in records if r["mode"] == mode]
        levels = sorted({r["n_per"] for r in rows})
        agg: dict = {
            k: []
            for k in (
                "sigma",
                "bias",
                "bias_se",
                "spread",
                "rms",
                "oar_renorm_rms",
                "oar_renorm_bias",
                "scale_spread",
                "delta_d98_pred_rms",
                "delta_d98_deliv_rms",
                "delta_oar_pred_rms",
                "delta_oar_deliv_rms",
            )
        }
        for n_per in levels:
            sel = [r for r in rows if r["n_per"] == n_per]
            k = len(sel)
            d98 = np.array([r["ptv_d98_true"] for r in sel])
            oar = np.array([r["oar_d2_true"] for r in sel])
            err = d98 - base_d98
            # Clinical renormalization: scale each plan so its recalculated
            # D98 matches the baseline's. Dose is linear in the weights, so
            # every endpoint scales by the same factor.
            scale = base_d98 / d98
            oar_renorm_err = scale * oar - base_oar
            agg["sigma"].append(np.mean([r["sigma_beamlet"] for r in sel]) * 100.0)
            agg["bias"].append(err.mean() * 100.0)
            agg["bias_se"].append(err.std(ddof=1) / np.sqrt(k) * 100.0)
            agg["spread"].append(err.std(ddof=1) * 100.0)
            agg["rms"].append(np.sqrt(np.mean(err**2)) * 100.0)
            agg["oar_renorm_rms"].append(np.sqrt(np.mean(oar_renorm_err**2)) * 100.0)
            agg["oar_renorm_bias"].append(oar_renorm_err.mean() * 100.0)
            agg["scale_spread"].append(scale.std(ddof=1) * 100.0)
            # Plan-difference experiment: predicted deltas are what the planner
            # reads off the noisy matrix when tightening the OAR penalty;
            # delivered deltas are what the true dose actually changes by.
            for endpoint, base_key in (("ptv_d98", "delta_ptv_d98"), ("oar_d2", "delta_oar_d2")):
                pred = np.array(
                    [r[f"{endpoint}_apparent_p2"] - r[f"{endpoint}_apparent"] for r in sel]
                )
                deliv = np.array([r[f"{endpoint}_true_p2"] - r[f"{endpoint}_true"] for r in sel])
                short = "d98" if endpoint == "ptv_d98" else "oar"
                base_delta = baseline[base_key]
                agg[f"delta_{short}_pred_rms"].append(
                    np.sqrt(np.mean((pred - base_delta) ** 2)) * 100.0
                )
                agg[f"delta_{short}_deliv_rms"].append(
                    np.sqrt(np.mean((deliv - base_delta) ** 2)) * 100.0
                )
        out[mode] = {key: np.array(val) for key, val in agg.items()}
    return out


def _style(ax: Axes) -> None:
    """Recessive panel styling shared by all axes."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    # Explicit octave ticks: the default log minor labels collide at this span.
    ax.set_xticks([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("per-beamlet sigma in the high-dose region (%)", fontsize=9, color=INK)


def _rms_panel(ax: Axes, agg: dict) -> None:
    for mode, color in SERIES.items():
        a = agg[mode]
        ax.plot(a["sigma"], a["rms"], color=color, lw=1.4, marker="o", ms=4, label=mode)
    ax.set_yscale("log")
    ax.set_ylabel("RMS true PTV D98 error (% of Rx)", fontsize=9, color=INK)
    ax.set_title(
        "total plan error, before renormalization\n(bias and spread combined)",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def _decomposition_panel(ax: Axes, agg: dict) -> None:
    ax.axhline(0.0, color=INK_SECONDARY, lw=0.8, ls=":")
    for mode, color in SERIES.items():
        a = agg[mode]
        ax.fill_between(
            a["sigma"],
            a["bias"] - a["bias_se"],
            a["bias"] + a["bias_se"],
            color=color,
            alpha=0.2,
            lw=0,
        )
        ax.plot(a["sigma"], a["bias"], color=color, lw=1.4, marker="o", ms=4, label=f"{mode}: bias")
        ax.plot(a["sigma"], a["spread"], color=color, lw=1.2, ls="--", marker="o", ms=3)
    ax.set_ylabel("true PTV D98 error components (% of Rx)", fontsize=9, color=INK)
    ax.set_title(
        "error decomposition\n(bias solid, across-seed spread dashed)", fontsize=10, color=INK
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def _renorm_panel(ax: Axes, agg: dict) -> None:
    for mode, color in SERIES.items():
        a = agg[mode]
        ax.plot(a["sigma"], a["oar_renorm_rms"], color=color, lw=1.4, marker="o", ms=4, label=mode)
    ax.set_yscale("log")
    ax.set_ylabel("RMS OAR D2 error after renorm (% of Rx)", fontsize=9, color=INK)
    ax.set_title(
        "after recalc + renormalization to D98\nresidual OAR error is what remains",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def _scale_panel(ax: Axes, agg: dict) -> None:
    for mode, color in SERIES.items():
        a = agg[mode]
        ax.plot(a["sigma"], a["scale_spread"], color=color, lw=1.4, marker="o", ms=4, label=mode)
    ax.set_yscale("log")
    ax.set_ylabel("across-seed std of renorm factor (%)", fontsize=9, color=INK)
    ax.set_title(
        "how much error is pure scale\n(renormalization absorbs exactly this)",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def _delta_panel(ax: Axes, agg: dict, key: str, ylabel: str, title: str) -> None:
    for mode, color in SERIES.items():
        a = agg[mode]
        ax.plot(a["sigma"], a[key], color=color, lw=1.4, marker="o", ms=4, label=mode)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def _delta_figure(agg: dict, baseline: dict) -> plt.Figure:
    """Build the plan-difference figure: error of plan deltas under re-optimization."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), facecolor=SURFACE)
    for ax in axes.ravel():
        _style(ax)
    truth_d98 = baseline["delta_ptv_d98"] * 100.0
    truth_oar = baseline["delta_oar_d2"] * 100.0
    _delta_panel(
        axes[0, 0],
        agg,
        "delta_d98_pred_rms",
        "RMS error of predicted ΔD98 (% of Rx)",
        f"predicted coverage cost of OAR tightening\n(truth Δ = {truth_d98:+.1f} % Rx)",
    )
    _delta_panel(
        axes[0, 1],
        agg,
        "delta_d98_deliv_rms",
        "RMS error of delivered ΔD98 (% of Rx)",
        "delivered coverage cost\n(same plans, true dose)",
    )
    _delta_panel(
        axes[1, 0],
        agg,
        "delta_oar_pred_rms",
        "RMS error of predicted ΔOAR D2 (% of Rx)",
        f"predicted OAR gain of tightening\n(truth Δ = {truth_oar:+.1f} % Rx)",
    )
    _delta_panel(
        axes[1, 1],
        agg,
        "delta_oar_deliv_rms",
        "RMS error of delivered ΔOAR D2 (% of Rx)",
        "delivered OAR gain\n(same plans, true dose)",
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the study, save figure and raw table beside this script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--full",
        action="store_true",
        help="decision-grade statistics: 32 seeds, sixth level at ~2 percent sigma, deeper truths",
    )
    parser.add_argument(
        "--phantom",
        choices=("water", "halfslab"),
        default="water",
        help="halfslab adds a lateral lung-density slab upstream of the PTV",
    )
    args = parser.parse_args()

    print(f"noise/bias study ({'full' if args.full else 'quick'} mode, {args.phantom})")
    records, baseline = run_study(args.full, args.phantom)
    agg = _aggregate(records, baseline)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), facecolor=SURFACE)
    for ax in axes.ravel():
        _style(ax)
    _rms_panel(axes[0, 0], agg)
    _decomposition_panel(axes[0, 1], agg)
    _renorm_panel(axes[1, 0], agg)
    _scale_panel(axes[1, 1], agg)
    fig.tight_layout()

    stem = Path(__file__).with_suffix("")
    suffix = ("_halfslab" if args.phantom == "halfslab" else "") + ("_full" if args.full else "")
    out = Path(f"{stem}{suffix}.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved {out}")

    delta_out = Path(f"{stem}{suffix}_delta.png")
    _delta_figure(agg, baseline).savefig(delta_out, dpi=160, facecolor=SURFACE)
    print(f"saved {delta_out}")

    csv_path = Path(f"{stem}{suffix}.csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"saved {csv_path}")

    print(f"  device                : {baseline['device']}")
    print(
        f"  truths A/B            : {baseline['n_truth']:,}/beamlet each, "
        f"sigma {baseline['sigma_beamlet'] * 100:.2f} %, {baseline['truth_seconds']:.1f} s"
    )
    print(
        f"  baseline plan (A on B): PTV D98 {baseline['ptv_d98_true'] * 100:.1f} % Rx, "
        f"OAR D2 {baseline['oar_d2_true'] * 100:.1f} % Rx"
    )
    print(
        f"  truth plan delta      : D98 {baseline['delta_ptv_d98'] * 100:+.2f} % Rx, "
        f"OAR D2 {baseline['delta_oar_d2'] * 100:+.2f} % Rx under strict OAR penalty"
    )
    for label, key in (
        ("D98 RMS", "rms"),
        ("OAR renorm RMS", "oar_renorm_rms"),
        ("pred dOAR RMS", "delta_oar_pred_rms"),
    ):
        for mode in SERIES:
            a = agg[mode]
            pairs = ", ".join(
                f"{s:.1f}%->{v:.2f}%" for s, v in zip(a["sigma"], a[key], strict=True)
            )
            print(f"  {label} ({mode:>11s}) : {pairs}")
    print("  wall-clock figures are laptop power-state dependent; not benchmarks")


if __name__ == "__main__":
    main()
