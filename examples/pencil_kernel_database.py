"""Simulate a mono-energetic photon pencil-beam kernel database in water.

Runs one pencil-beam kernel per energy on a graded depth-by-radial-shell binning and
writes them to a single compressed ``.npz``. This is the lookup table an analytical
pencil-beam dose engine is built from.

Protocol
--------
Energies
    22 points, 0.2 to 15 MeV, graded so the spacing follows how fast the kernel
    changes with energy: 0.1 MeV steps to 0.6, then 0.25, 0.5, 1, 2 and 3 MeV.
Medium
    Water at 1.0 g/cm^3, homogeneous.
Radial
    97 shells, 0 to 30 cm, first shell the disc ``[0, 0.010)`` cm and the rest
    equal-ratio (about 8.7 percent per shell) out to 30 cm.
Depth
    152 bins, 0 to 32 cm, graded 0.010 / 0.025 / 0.25 / 1.00 cm so the build-up
    region is resolved and the flat tail is not over-binned.

Output
------
``dose`` is a **density** — MeV/g per emitted history, per bin — not a ring-integrated
energy. The scorer divides each bin's energy by that bin's own annulus mass
``rho * pi * (r_out^2 - r_in^2) * dz``, so the radial shape is the physical dose
profile and does not carry the ``2*pi*r*dr`` growth of the shell volume. Multiply by
``pyradmc.GY_PER_MEV_PER_G`` for Gy per history.

``r_bounds_cm`` and ``z_bounds_cm`` are bin **edges** (lengths ``nR+1`` and ``nZ+1``),
not centres.

The run sets ``deposit_resolution_cm`` because it bins depth well below the transport
voxel. A condensed-history substep is capped at the voxel face and its collision loss
is filed at the half-step midpoint, so without it the sub-voxel build-up profile
carries a large modulation locked to the voxel pitch. It costs roughly twice the
runtime here and moves no energy — only where inside a voxel it is filed.

Run from the repository root::

    python examples/pencil_kernel_database.py --out photon_kernels.npz

``--pilot`` runs a single energy at low statistics and reports the achieved
uncertainty and the projected cost of the full sweep, which is the sane thing to do
before committing a long run.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from pyradmc import __version__
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource
from pyradmc.scoring.cylinder import CylindricalScoringGrid, geometric_edges, graded_edges

logger = logging.getLogger("pencil_kernel_database")

SEED = 20260711

ENERGIES_MEV: tuple[float, ...] = (
    0.2, 0.3, 0.4, 0.5, 0.6,          # 0.1 MeV steps
    0.75, 1.0, 1.25, 1.5, 1.75,       # 0.25
    2.0, 2.5, 3.0,                    # 0.5
    4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,  # 1
    12.0,                             # 2
    15.0,                             # 3
)  # fmt: skip

RADIAL_R_MAX_CM = 30.0
RADIAL_N_SHELLS = 97
RADIAL_R_MIN_CM = 0.010

DEPTH_SEGMENTS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.5, 0.010),
    (0.5, 2.0, 0.025),
    (2.0, 6.0, 0.25),
    (6.0, 32.0, 1.00),
)

WATER_DENSITY = 1.0

# --- phantom ---------------------------------------------------------------
# A water box circumscribing the 30 cm scoring radius and the 32 cm scoring depth,
# with margin, so the outermost shells still sit under full lateral scatter and the
# deepest bin is not starved of backscatter. The kernel is defined in a semi-infinite
# medium; the box stands in for it, and its corners hold *more* material than an
# inscribed cylinder would.
PHANTOM_MARGIN_CM = 2.0
PHANTOM_SPACING_CM = 0.25

BATCHES = 10
"""Statistical batches. Ten gives a well-conditioned batch-variance estimate and
keeps the Welch-Satterthwaite inflation of the sigma estimate small."""

DEFAULT_HISTORIES = 4_000_000

DIVERGENCE_SIGMA_RAD = 0.0
"""Angular divergence of the primary, in radians.

Exactly zero: this is a *pencil* beam — an infinitely narrow, perfectly parallel
source (``PencilBeamSource`` with a fixed direction), which is what makes the result
a kernel rather than a beam model. The field is carried per energy because a
consumer that convolves these kernels with a diverging source needs somewhere to put
that source's divergence; it is not a property of the kernels simulated here.
"""


def build_binning() -> tuple[np.ndarray, np.ndarray]:
    """Radial and depth bin edges, in cm."""
    r_bounds = geometric_edges(
        r_max=RADIAL_R_MAX_CM, n_shells=RADIAL_N_SHELLS, r_min=RADIAL_R_MIN_CM
    )
    z_bounds = graded_edges(list(DEPTH_SEGMENTS))
    return r_bounds, z_bounds


def build_phantom(z_bounds: np.ndarray) -> tuple[VoxelGrid, tuple[float, float]]:
    """Water box surrounding the scored cylinder; returns it and the beam axis."""
    half_width = RADIAL_R_MAX_CM + PHANTOM_MARGIN_CM
    depth = float(z_bounds[-1]) + PHANTOM_MARGIN_CM
    n_lateral = round(2.0 * half_width / PHANTOM_SPACING_CM)
    n_depth = round(depth / PHANTOM_SPACING_CM)
    grid = VoxelGrid.uniform_water(
        shape=(n_lateral, n_lateral, n_depth),
        spacing=(PHANTOM_SPACING_CM,) * 3,
    )
    return grid, (half_width, half_width)


def simulate(
    energy_mev: float,
    grid: VoxelGrid,
    axis: tuple[float, float],
    cylinder: CylindricalScoringGrid,
    engine: object,
    n_histories: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One kernel: returns ``(dose, sigma, ledger_residual)`` on the (nZ, nR) binning.

    ``ledger_residual`` is the relative closure of
    ``emitted == deposited + unscored + escaped``; it is checked rather than assumed,
    because a kernel that silently lost energy would still look plausible.
    """
    source = PencilBeamSource(
        energy=energy_mev, position=(axis[0], axis[1], -0.1), direction=(0.0, 0.0, 1.0)
    )
    result = engine.run(  # type: ignore[attr-defined]
        source,
        n_histories=n_histories,
        n_batches=BATCHES,
        seed=SEED,
        scoring_grid=cylinder,
        # The depth bins are ten times finer than the transport voxel. Without this
        # the half-substep midpoint deposit prints the voxel lattice onto the
        # build-up region as a tent peaked at voxel centres; see the module docstring.
        deposit_resolution_cm=cylinder.deposit_resolution_cm,
    )
    booked = result.energy_deposited + result.energy_unscored + result.energy_escaped
    residual = abs(booked - result.energy_emitted) / result.energy_emitted
    return result.dose, result.dose_sigma, residual


def uncertainty_report(dose: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    """Relative 1-sigma statistics, banded by how much dose a bin actually carries.

    A single per-bin target over *every* bin is unreachable by construction: the
    outermost shells at 30 cm and the deepest at 32 cm sit many orders below the
    peak, and pushing those to half a percent is a variance-reduction problem, not a
    history-count one. So the achieved uncertainty is reported per dose band, and
    the statistics target is read against the band that carries the dose.

    Bands are fractions of the kernel maximum. ``axis`` is the innermost shell,
    which is the hardest bin in the table — it holds the highest dose but the
    smallest mass, so few histories reach it.
    """
    out: dict[str, float] = {}
    for name, lo in (("hi", 0.5), ("mid", 0.1), ("lo", 0.01)):
        mask = (dose > lo * dose.max()) & (sigma > 0.0)
        rel = sigma[mask] / dose[mask]
        out[f"{name}_n"] = float(mask.sum())
        out[f"{name}_median"] = float(np.median(rel)) if rel.size else float("nan")
        out[f"{name}_p95"] = float(np.percentile(rel, 95)) if rel.size else float("nan")
    axis_dose, axis_sigma = dose[:, 0], sigma[:, 0]
    live = (axis_dose > 0.01 * axis_dose.max()) & (axis_sigma > 0.0)
    rel_axis = axis_sigma[live] / axis_dose[live]
    out["axis_median"] = float(np.median(rel_axis)) if rel_axis.size else float("nan")
    return out


def main() -> None:
    """Run the sweep and write the database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("photon_kernels.npz"))
    parser.add_argument("--histories", type=int, default=DEFAULT_HISTORIES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="one energy at the given statistics; report uncertainty and projected cost",
    )
    parser.add_argument(
        "--pilot-energy",
        type=float,
        default=max(ENERGIES_MEV),
        help="energy to pilot; defaults to the highest, which is the costliest",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    r_bounds, z_bounds = build_binning()
    grid, axis = build_phantom(z_bounds)
    cylinder = CylindricalScoringGrid.for_grid(
        grid, radial_edges=r_bounds, axis=axis, depth_edges=z_bounds
    )
    logger.info(
        "phantom %s voxels at %.2f cm; cylinder %d depth x %d shells; %d histories/energy",
        "x".join(str(n) for n in grid.shape),
        PHANTOM_SPACING_CM,
        cylinder.n_depth,
        cylinder.n_shells,
        args.histories,
    )

    from pyradmc.backends.warp.engine import WarpEngine

    engine = WarpEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        device=args.device,
    )

    energies = (args.pilot_energy,) if args.pilot else ENERGIES_MEV
    tables: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    t_start = time.perf_counter()

    for i, energy in enumerate(energies):
        t0 = time.perf_counter()
        dose, sigma, residual = simulate(energy, grid, axis, cylinder, engine, args.histories)
        elapsed = time.perf_counter() - t0
        stats = uncertainty_report(dose, sigma)
        logger.info(
            "%2d/%2d %6.2f MeV %6.1f s ledger %.1e | sigma/dose median(p95): "
            ">50%%max %.2f%%(%.2f%%) n=%d | >10%% %.2f%%(%.2f%%) n=%d | "
            ">1%% %.2f%%(%.2f%%) n=%d | axis %.2f%%",
            i + 1,
            len(energies),
            energy,
            elapsed,
            residual,
            100 * stats["hi_median"],
            100 * stats["hi_p95"],
            int(stats["hi_n"]),
            100 * stats["mid_median"],
            100 * stats["mid_p95"],
            int(stats["mid_n"]),
            100 * stats["lo_median"],
            100 * stats["lo_p95"],
            int(stats["lo_n"]),
            100 * stats["axis_median"],
        )
        # dose comes back (nZ, nR); the database stores (nE, nR, nZ).
        tables.append(dose.T)
        sigmas.append(sigma.T)

    total = time.perf_counter() - t_start
    if args.pilot:
        logger.info(
            "pilot: %.1f s for 1 energy -> projected %.1f min for all %d energies",
            total,
            total * len(ENERGIES_MEV) / 60.0,
            len(ENERGIES_MEV),
        )
        return

    np.savez_compressed(
        args.out,
        energies_mev=np.asarray(energies, float),
        r_bounds_cm=r_bounds,
        z_bounds_cm=z_bounds,
        dose=np.asarray(tables, float),
        divergence_sigma_rad=np.full(len(energies), DIVERGENCE_SIGMA_RAD, dtype=float),
        # Provenance and units, so the table is interpretable without this script.
        dose_sigma=np.asarray(sigmas, float),
        dose_units=np.asarray("MeV/g per emitted history"),
        medium=np.asarray("water"),
        density_g_cm3=np.asarray(WATER_DENSITY),
        n_histories_per_energy=np.asarray(args.histories),
        provenance=np.asarray(
            f"pyradmc {__version__} warp/{args.device} seed={SEED} "
            f"batches={BATCHES} analytic cross-sections "
            f"deposit_resolution_cm={cylinder.deposit_resolution_cm:.4g}"
        ),
    )
    logger.info("wrote %s in %.1f min", args.out, total / 60.0)


if __name__ == "__main__":
    main()
