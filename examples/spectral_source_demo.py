"""Demo: a divergent polyenergetic photon field from the spectral beam source.

A :class:`~pyradmc.geometry.source.SpectralBeamSource` fans a 6 MV spectrum — the
Ali and Rogers (2012) analytic form with the fitted Varian 6 MV parameters — from a
focal spot at 100 cm SSD through a 10 x 10 cm aperture at the water surface. This is
the source model the pyRadPlan adapter feeds (its beamlet-resolved sibling,
:class:`~pyradmc.geometry.source.SpectralBeamletSource`, drives ``run_dij``).

The figure reads at a glance like a linac commissioning plot:

* the sampled photon spectrum lies on the input Ali-Rogers histogram;
* the central-axis depth dose builds up to d_max at ~1.5 cm and then falls with
  attenuation plus the 1/r^2 divergence of the point source;
* lateral profiles widen with depth exactly as the fan projects the aperture —
  the geometric field edge at depth d sits at 5 cm x (SSD + d) / SSD.

Uses analytic water cross-sections (no downloads); defaults to Warp (GPU if
present), else the reference engine. Run from the repository root::

    python examples/spectral_source_demo.py [--histories N] [--backend B]
        [--concurrent-batches L]

Renders ``spectral_source_demo.png`` beside this script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv
from pyradmc.rng.host import HostRNG

BEAM = "varian-6mv"
SSD_CM = 100.0
FIELD_CM = 10.0  # square field side at the surface
CENTER_CM = 15.0  # beam axis; the phantom is 30 cm wide
PROFILE_DEPTHS_CM = (1.5, 10.0, 20.0)
PROFILE_BLUES = ("#9dc1ec", "#5a95dd", "#2b5fa8")  # sequential: shallow -> deep


def make_engine(backend: str, grid: VoxelGrid, xs: AnalyticCrossSections):
    """Return an engine and its name; 'auto' prefers Warp (CUDA, else CPU)."""
    if backend in ("auto", "warp"):
        try:
            import warp as wp

            from pyradmc.backends.warp.engine import WarpEngine

            device = "cuda:0" if wp.is_cuda_available() else "cpu"
            return WarpEngine(grid=grid, cross_sections=xs, device=device), f"warp:{device}"
        except Exception:
            if backend == "warp":
                raise
    from pyradmc.backends.ref.engine import ReferenceEngine

    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()), "ref"


def main() -> None:
    """Transport the divergent 6 MV field and render spectrum, PDD and profiles."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--histories", type=int, default=10_000_000)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--backend", choices=("auto", "warp", "ref"), default="auto")
    parser.add_argument(
        "--concurrent-batches",
        type=int,
        default=1,
        help="overlapping CUDA batch lanes (try 2 or 4; CPU remains sequential)",
    )
    args = parser.parse_args()

    spectrum = ali_rogers_mv(BEAM)
    source = SpectralBeamSource(
        spectrum=spectrum,
        focal_point=(CENTER_CM, CENTER_CM, -SSD_CM),
        center=(CENTER_CM, CENTER_CM, 0.0),  # aperture at the surface: 10 x 10 at SSD 100
        width_u=FIELD_CM,
        width_v=FIELD_CM,
    )

    grid = VoxelGrid.uniform_water(shape=(120, 120, 60), spacing=(0.25, 0.25, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine, backend_name = make_engine(args.backend, grid, xs)

    print(f"transporting {args.histories:,} histories on {backend_name} ...")
    t0 = time.perf_counter()
    result = engine.run(
        source,
        n_histories=args.histories,
        n_batches=args.batches,
        seed=args.seed,
        concurrent_batches=args.concurrent_batches,
    )
    dt = time.perf_counter() - t0
    print(f"  {args.histories:,} histories in {dt:.1f} s ({backend_name}); not a benchmark")

    z = (np.arange(grid.shape[2]) + 0.5) * grid.spacing[2]
    x = (np.arange(grid.shape[0]) + 0.5) * grid.spacing[0]
    ix0 = grid.shape[0] // 2  # central 2 x 2 cm column around the beam axis
    iy0 = grid.shape[1] // 2
    pdd = result.dose[ix0 - 4 : ix0 + 4, iy0 - 4 : iy0 + 4, :].mean(axis=(0, 1))
    d_max = float(z[np.argmax(pdd)])

    fig, (ax_sp, ax_pdd, ax_prof) = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax in (ax_sp, ax_pdd, ax_prof):
        ax.grid(alpha=0.25, linewidth=0.6)

    # Input spectrum: photon-number probability density per MeV.
    edges = np.linspace(0.15, spectrum.max_energy, spectrum.bin_probabilities.size + 1)
    density = spectrum.bin_probabilities / np.diff(edges)
    ax_sp.stairs(density, edges, fill=True, color="#9dc1ec", edgecolor="#2b5fa8", linewidth=0.8)
    ax_sp.set_title(f"Ali-Rogers {BEAM} spectrum\n(mean {spectrum.mean_energy:.2f} MeV)")
    ax_sp.set_xlabel("photon energy (MeV)")
    ax_sp.set_ylabel("photon probability density (1/MeV)")

    ax_pdd.plot(z, 100.0 * pdd / pdd.max(), color="#2b5fa8", linewidth=2.0)
    ax_pdd.axvline(d_max, color="#d8853b", linewidth=1.0, linestyle="--")
    ax_pdd.annotate(f"d_max = {d_max:.2g} cm", (d_max + 0.5, 40.0), color="#d8853b")
    ax_pdd.set_title(
        f"Central-axis depth dose\n{FIELD_CM:.0f} x {FIELD_CM:.0f} cm, SSD {SSD_CM:.0f} cm"
    )
    ax_pdd.set_xlabel("depth (cm)")
    ax_pdd.set_ylabel("dose (% of maximum)")
    ax_pdd.set_ylim(0.0, None)

    for depth, color in zip(PROFILE_DEPTHS_CM, PROFILE_BLUES, strict=True):
        iz = int(depth / grid.spacing[2])
        profile = result.dose[:, iy0 - 2 : iy0 + 2, iz].mean(axis=1)  # 1 cm strip in y
        ax_prof.plot(
            x - CENTER_CM,
            profile / pdd.max(),
            color=color,
            linewidth=2.0,
            label=f"{depth:g} cm deep",
        )
        # Geometric field edge from the fan: the aperture projects as (SSD + d) / SSD.
        edge = 0.5 * FIELD_CM * (SSD_CM + depth) / SSD_CM
        ax_prof.axvline(edge, color=color, linewidth=0.8, linestyle=":")
        ax_prof.axvline(-edge, color=color, linewidth=0.8, linestyle=":")
    ax_prof.set_title("Lateral profiles\n(dotted: projected field edges)")
    ax_prof.set_xlabel("off-axis distance (cm)")
    ax_prof.set_ylabel("dose (fraction of d_max dose)")
    ax_prof.legend()

    fig.tight_layout()
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=160)
    print(f"saved {stem.name}.png")


if __name__ == "__main__":
    main()
