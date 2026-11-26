"""Dij truncation, hard and stochastic: what each costs a plan.

The Dij drops entries below ``truncation * column_max`` (``truncation`` = 1e-3 by
default, AGENTS.md 2.8). How the sub-threshold tail is treated is
``truncation_mode``:

- ``"hard"`` (default) drops it. At low statistics a lone hit in a tail voxel
  clears the threshold and survives; at high statistics the tail converges below
  it and is dropped whole. The dose removed therefore *grows with the histories per
  beamlet*.
- ``"stochastic"`` (opt-in) plays Russian roulette on it: an entry of mean ``m < t``
  is kept with probability ``m / t`` and stored as ``t`` (Thieke et al., Med. Phys.
  29 (2002) 676, doi:10.1118/1.1469633). Every plan dose of non-negative scores
  is unbiased at every history count; the price is a per-voxel noise
  ``sum_j m (t - m)`` that does not fall with the histories.

Renders one figure (``dij_truncation_demo.png``, saved beside this script):

- **Left**: the plan dose kept, truncated over untruncated (same seed, so the Monte
  Carlo noise cancels and only the truncation shows), against histories per beamlet,
  summed over the whole phantom (solid) and over the voxels above 50 % of the
  maximum (dashed). The hard rule falls as the statistics improve; the stochastic
  mode stays at 1 within its roulette sigma (error bars, 2 sigma).
- **Right**: at the largest history count, the median per-voxel standard deviation
  relative to the local dose, against the local plan dose in units of its maximum:
  the Monte Carlo sigma of the untruncated plan, and the sigma the roulette adds.
  The roulette's is the larger in the low-dose region and the smaller in the field.

Setup: a 10 x 10 cm field of 6 MeV photons as 20 x 20 beamlets of 0.5 cm on a 30 cm
water cube in 1 cm voxels, full transport, unit weights, independent columns
(``correlated=False``, so the quadrature of the column sigmas is the plan's Monte
Carlo sigma), 8 batches, seed 20260711, 5000 to 80000 histories per beamlet.
Measured on CUDA (laptop RTX 4070) at 5k / 10k / 20k / 40k / 80k histories per
beamlet: the hard rule keeps 0.981 / 0.966 / 0.953 / 0.942 / 0.937 of the
whole-phantom plan dose and 0.995 / 0.992 / 0.988 / 0.985 / 0.983 above 50 % of the
maximum; the stochastic mode keeps 1.0000 at every count (roulette sigma 2e-5 to
5e-5 of the sum, so its error bars are hidden by the markers) with 1.5 to 3 times
as many stored entries (0.94M to 1.58M against 0.64M to 0.52M). At 80k the roulette
sigma exceeds the Monte Carlo sigma below about 3 % of the maximum (6 % against
4 % at 1-3 %, 42 % against 11 % at 0.03-0.1 %) and is a sixth of it at the top
(0.13 % against 0.81 %).

Run it from the repository root (requires the ``examples`` and ``warp`` extras)::

    python examples/dij_truncation_demo.py

Runtime is well under a minute on a laptop GPU; without CUDA it falls back to
Warp-CPU with an 8 x 8 lattice and a tenth of the histories, and the curves
become correspondingly noisier.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.axes import Axes

from pyradmc import DIJ_TRUNCATION_RELATIVE
from pyradmc.backends.warp.engine import WarpEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource
from pyradmc.scoring.dij import DijResult

SEED = 20260711

# Reference dataviz palette (validated): categorical slots 1-2, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
HARD, STOCHASTIC = "#2a78d6", "#eda100"  # blue, yellow
MC, ROULETTE = "#2a78d6", "#eda100"

SHAPE = (30, 30, 30)
SPACING = (1.0, 1.0, 1.0)
FIELD = (10.0, 20.0)  # 10 x 10 cm, centred
ENERGY = 6.0
HIGH_DOSE = 0.5  # fraction of the plan maximum for the dashed curves
DOSE_BINS = np.array([3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0001])


def _style(ax: Axes) -> None:
    """Recessive panel styling shared by all axes."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def roulette_variance(full: DijResult, weights: np.ndarray) -> np.ndarray:
    """Per voxel, ``sum_j w_j^2 m_ij (t_j - m_ij)`` over the tail entries of ``full``."""
    out = np.zeros(full.n_voxels)
    for j in range(full.n_beamlets):
        lo, hi = int(full.indptr[j]), int(full.indptr[j + 1])
        idx, m = full.indices[lo:hi], full.dose[lo:hi]
        if m.size == 0:
            continue
        t = DIJ_TRUNCATION_RELATIVE * m.max()
        tail = m < t
        out[idx[tail]] += weights[j] ** 2 * m[tail] * (t - m[tail])
    return out


def run_all() -> dict:
    """Untruncated, hard and stochastic Dij at each history count, one device."""
    cuda = wp.is_cuda_available()
    device = "cuda:0" if cuda else "cpu"
    n_side = 20 if cuda else 8
    counts = [5_000, 10_000, 20_000, 40_000, 80_000]
    if not cuda:
        counts = [n // 10 for n in counts]
    grid = VoxelGrid.uniform_water(shape=SHAPE, spacing=SPACING)
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine = WarpEngine(grid=grid, cross_sections=xs, device=device)
    source = BeamletGridSource(
        energy=ENERGY, z=-1.0, x_range=FIELD, y_range=FIELD, n_x=n_side, n_y=n_side
    )
    weights = np.ones(source.n_beamlets)
    engine.run_dij(source, n_histories_per_beamlet=80, n_batches=8, seed=SEED)  # warm

    rows = []
    t0 = time.perf_counter()
    for n in counts:
        kw = dict(n_histories_per_beamlet=n, n_batches=8, seed=SEED, correlated=False)
        full = engine.run_dij(source, truncation=0.0, **kw)
        hard = engine.run_dij(source, **kw)
        sto = engine.run_dij(source, truncation_mode="stochastic", **kw)
        d_full = full.dose_for_weights(weights).ravel()
        high = d_full >= HIGH_DOSE * d_full.max()
        var_rr = roulette_variance(full, weights)
        row = {"n": n, "full": full, "d_full": d_full, "var_rr": var_rr}
        for name, dij in (("hard", hard), ("stochastic", sto)):
            d = dij.dose_for_weights(weights).ravel()
            row[name] = (d.sum() / d_full.sum(), d[high].sum() / d_full[high].sum())
            row[f"{name}_nnz"] = dij.indices.size
        row["sigma"] = (
            np.sqrt(var_rr.sum()) / d_full.sum(),
            np.sqrt(var_rr[high].sum()) / d_full[high].sum(),
        )
        rows.append(row)
    return {"rows": rows, "device": device, "seconds": time.perf_counter() - t0, "n_side": n_side}


def kept_panel(ax: Axes, data: dict) -> None:
    """Kept plan-dose fraction against histories per beamlet, both modes."""
    rows = data["rows"]
    n = np.array([r["n"] for r in rows])
    for k, (style, region) in enumerate((("-", "whole phantom"), ("--", f">{HIGH_DOSE:.0%} max"))):
        hard = np.array([r["hard"][k] for r in rows])
        sto = np.array([r["stochastic"][k] for r in rows])
        two_sigma = 2.0 * np.array([r["sigma"][k] for r in rows])
        ax.plot(n, hard, style, color=HARD, marker="o", ms=4, lw=1.4, label=f"hard, {region}")
        ax.errorbar(
            n,
            sto,
            yerr=two_sigma,
            fmt=style,
            color=STOCHASTIC,
            marker="s",
            ms=4,
            lw=1.4,
            capsize=2,
            label=f"stochastic, {region}",
        )
    ax.axhline(1.0, color=INK_SECONDARY, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(n, [f"{v // 1000}k" if v >= 1000 else str(v) for v in n])
    ax.minorticks_off()
    ax.set_xlabel("histories per beamlet", fontsize=9, color=INK)
    ax.set_ylabel("plan dose kept (truncated / untruncated)", fontsize=9, color=INK)
    ax.set_title("the hard rule's bias grows with the statistics", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def noise_panel(ax: Axes, data: dict) -> None:
    """Median relative per-voxel sigma, Monte Carlo against roulette, by dose level."""
    row = data["rows"][-1]
    full: DijResult = row["full"]
    d_full = row["d_full"]
    weights = np.ones(full.n_beamlets)
    var_mc = np.asarray(full.variance_csc() @ weights**2).ravel()
    rel = d_full / d_full.max()
    centres, mc, rr = [], [], []
    for lo, hi in itertools.pairwise(DOSE_BINS):
        sel = (rel >= lo) & (rel < hi) & (d_full > 0.0)
        if np.count_nonzero(sel) < 10:
            continue
        centres.append(np.sqrt(lo * min(hi, 1.0)))
        mc.append(np.median(np.sqrt(var_mc[sel]) / d_full[sel]))
        rr.append(np.median(np.sqrt(row["var_rr"][sel]) / d_full[sel]))
    ax.plot(centres, np.array(mc) * 100.0, "-o", color=MC, ms=4, lw=1.4, label="Monte Carlo")
    ax.plot(
        centres, np.array(rr) * 100.0, "-s", color=ROULETTE, ms=4, lw=1.4, label="roulette (added)"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("local plan dose / plan maximum", fontsize=9, color=INK)
    ax.set_ylabel("median sigma / local dose (%)", fontsize=9, color=INK)
    ax.set_title(f"per-voxel noise at {row['n']:,} histories per beamlet", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def main() -> None:
    """Run the sweep, draw both panels, save the figure and print the numbers."""
    data = run_all()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), facecolor=SURFACE)
    for ax in axes:
        _style(ax)
    kept_panel(axes[0], data)
    noise_panel(axes[1], data)
    fig.suptitle(
        f"Dij truncation at {DIJ_TRUNCATION_RELATIVE:g} of the column maximum: "
        f"{data['n_side']}x{data['n_side']} beamlets, 6 MeV, water, unit weights "
        f"({data['device']}, seed {SEED})",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved {out} ({data['seconds']:.1f} s of Dij runs on {data['device']})")
    for r in data["rows"]:
        print(
            f"N={r['n']:>6}: hard kept {r['hard'][0]:.4f} / {r['hard'][1]:.4f}, "
            f"stochastic {r['stochastic'][0]:.4f} / {r['stochastic'][1]:.4f} "
            f"(roulette sigma {r['sigma'][0]:.1e} / {r['sigma'][1]:.1e}), "
            f"entries hard {r['hard_nnz']:,}, stochastic {r['stochastic_nnz']:,}"
        )


if __name__ == "__main__":
    main()
