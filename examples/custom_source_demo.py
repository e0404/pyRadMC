"""Demo: defining a custom primary source through the public Source interface.

The engine's built-in sources are a pencil, a broad parallel field, and a beamlet
lattice. This shows a source they do *not* provide — a **Gaussian pencil beam** (a
photon beam whose lateral fluence is a 2-D Gaussian) — written by subclassing
:class:`pyradmc.geometry.source.Source`. Two things are all that a custom source needs:

* ``emit`` + ``max_energy`` — the host contract. The reference backend transports it
  directly, and (the *simple* GPU route) the default ``sample_batch`` host-samples it
  for a device backend with no extra work.
* an optional ``warp_sampler`` ``@wp.func`` — the *advanced* GPU route. If present, the
  Warp engine wraps it into a generator kernel and produces the primaries on-device with
  no host round trip. Here ``emit`` and ``warp_sampler`` are the same Box-Muller draw, so
  the two routes are interchangeable; the source below carries both.

The figure overlays the simulated central-axis-plane lateral dose profile on the input
Gaussian — the beam reproduces the profile it was given, broadened a little by scatter.
Uses analytic water cross-sections (no downloads); defaults to Warp (GPU if present),
else the reference engine. Run from the repository root::

    python examples/custom_source_demo.py [--histories N] [--backend B]

Renders ``custom_source_demo.png`` beside this script.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import Primary, Source
from pyradmc.rng.host import HostRNG, uniform

ENERGY = 6.0
SIGMA_CM = 2.0  # lateral Gaussian sigma
CENTER = 10.0  # beam axis (x = y = CENTER), phantom is 20 cm wide
Z_ENTRY = -1.0

try:  # the warp_sampler is only defined when warp is available; emit works regardless.
    import warp as wp

    from pyradmc.rng.warp_shim import WarpRNGState
    from pyradmc.rng.warp_shim import uniform as wp_uniform
    from pyradmc.transport.particles import PHOTON

    @wp.func
    def _gaussian_sampler(history_index: int, state: WarpRNGState):
        """Draw a Gaussian-profile photon along +z; Box-Muller from two uniforms."""
        radius = SIGMA_CM * wp.sqrt(-2.0 * wp.log(wp_uniform(state)))
        angle = 2.0 * 3.14159265358979 * wp_uniform(state)
        x = CENTER + radius * wp.cos(angle)
        y = CENTER + radius * wp.sin(angle)
        return int(PHOTON), ENERGY, x, y, Z_ENTRY, 0.0, 0.0, 1.0, 1.0

    _WARP_SAMPLER = _gaussian_sampler
except Exception:  # warp not installed: the source still runs on the reference backend
    _WARP_SAMPLER = None


class GaussianPencilSource(Source):
    """A photon beam with a 2-D Gaussian lateral fluence profile — not a built-in.

    ``emit`` (host) and ``warp_sampler`` (device) are the same Box-Muller draw, so the
    beam is identical whichever GPU route the engine takes.
    """

    warp_sampler = _WARP_SAMPLER

    @property
    def max_energy(self) -> float:
        """The single beam energy, for cross-section table sizing."""
        return ENERGY

    def emit(self, rng_state: object) -> Primary:
        """Emit one Gaussian-profile photon; Box-Muller from two uniforms."""
        radius = SIGMA_CM * math.sqrt(-2.0 * math.log(uniform(rng_state)))
        angle = 2.0 * math.pi * uniform(rng_state)
        x = CENTER + radius * math.cos(angle)
        y = CENTER + radius * math.sin(angle)
        return Primary(ENERGY, x, y, Z_ENTRY, 0.0, 0.0, 1.0)


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
    """Transport the custom Gaussian source and overlay the lateral dose on its profile."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--histories", type=int, default=5_000_000)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--backend", choices=("auto", "warp", "ref"), default="auto")
    args = parser.parse_args()

    grid = VoxelGrid.uniform_water(shape=(80, 80, 40), spacing=(0.25, 0.25, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine, backend_name = make_engine(args.backend, grid, xs)

    print(f"transporting {args.histories:,} histories on {backend_name} ...")
    t0 = time.perf_counter()
    result = engine.run(
        GaussianPencilSource(), n_histories=args.histories, n_batches=args.batches, seed=args.seed
    )
    dt = time.perf_counter() - t0
    print(f"  {args.histories:,} histories in {dt:.1f} s ({backend_name}); not a benchmark")

    # Lateral dose profile along x at the depth of maximum dose, through the beam axis.
    depth = int(np.argmax(result.dose.sum(axis=(0, 1))))
    iy = grid.shape[1] // 2
    profile = result.dose[:, iy, depth]
    x = (np.arange(grid.shape[0]) + 0.5) * grid.spacing[0]
    gaussian = np.exp(-0.5 * ((x - CENTER) / SIGMA_CM) ** 2)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(x, profile / profile.max(), color="#3b7dd8", label="simulated lateral dose")
    ax.plot(x, gaussian, "--", color="#d8853b", label=f"input Gaussian (sigma {SIGMA_CM:g} cm)")
    ax.set_title(f"Custom Gaussian pencil source, {ENERGY:.0f} MeV photons in water")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("dose (normalized)")
    ax.legend()
    fig.tight_layout()

    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=160)
    print(f"saved {stem.name}.png")


if __name__ == "__main__":
    main()
