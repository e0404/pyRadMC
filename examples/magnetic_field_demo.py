"""Photon dose in a transverse magnetic field: the electron return effect.

Renders one figure (``magnetic_field_demo.png``, saved beside this script) for a
6 MV beam into a water / air / water slab phantom at 0, 0.2, 0.7, 1.5 and 3 T, the
field transverse to the beam as on an MR-linac (the sweep of Raaijmakers et al. 2008;
the MRIdian runs at 0.35 T, the Aurora-RT at 0.5 T, the Unity at 1.5 T):

- **Left**: central-axis depth dose. Secondary electrons that leave the upstream
  water into the air gyrate back (gyroradius 3 mm at 1 MeV and 1.5 T) and redeposit
  on the water side — the *electron return effect* (Raaijmakers, Raaymakers &
  Lagendijk 2005, PMB 50, 1363, doi:10.1088/0031-9155/50/7/002) — so the last
  millimetres before the gap gain dose while the downstream water, no longer fed by
  electrons crossing the gap, rebuilds from its own surface. The peak is about as
  high at 0.2 T as at 1.5 T and highest in between: an electron leaving along the
  normal reaches only one gyroradius into the gap, so in a 3 cm gap everything below
  about 1.4 MeV turns round even at 0.2 T (2.7 MeV at 0.35 T), which covers most of
  the energy fluence leaving a 6 MV beam, and at the strongest fields the compressed
  in-water kernel feeds the exit face from a smaller volume. That is what Raaijmakers
  et al. (2008, PMB 53, 909, doi:10.1088/0031-9155/53/4/006) report for broad fields:
  an exit-side dose increase of 40 / 52 / 42 / 25 % at 0.2 / 0.75 / 1.5 / 3 T for a
  10 x 10 cm2 field.
- **Right**: lateral profile across the beam in the last 4 mm of water before the
  gap. The returning electrons land one gyration *diameter* to the side (toward -y
  for a +z beam in a +x field), so the profile skews, and the skew is *wider* at the
  weaker field: a 1 MeV electron comes back 4.7 cm to the left at 0.2 T, 6 mm at
  1.5 T and 3 mm at 3 T. The right edge loses the electrons that would have returned
  from outside the field, the left edge gains them; at 0.2 T the dose rises gradually
  from the right edge to the left and a long tail reaches centimetres beyond the
  field, as Raaijmakers et al. (2008) describe. The in-water penumbra shift that
  grows *with* the field is under a millimetre here and invisible on 5 mm voxels.

Air is a real registry material here (EPICS-tabulated cross-sections, downloaded and
cached on first use), and the field is :class:`pyradmc.MagneticField` handed to the
engine as world data. Run it from the repository root (requires the ``examples``
extra, i.e. matplotlib)::

    python examples/magnetic_field_demo.py

Runtime is under a minute on a CUDA device; pass ``--histories`` to trade statistics
for time on the CPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from pyradmc import AIR, WATER, MagneticField
from pyradmc.data.tabulated import build
from pyradmc.data.tabulated.precompile import compile_materials
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv

SEED = 20260711

# Reference dataviz palette (validated): categorical slots 1-3, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e5e4e0"
SERIES = ["#52514e", "#2a78d6", "#ffa500", "#d64545", "#ff69b4"]
# 0 T grey, 0.2 T blue, 0.7 T orange, 1.5 T red, 3 T pink — zipped with FIELDS_TESLA.
AIR_BAND = "#d8ecf5"

FIELDS_TESLA = (0.0, 0.2, 0.7, 1.5, 3)

# Phantom: 20 x 20 cm lateral, 12 cm deep; water 0-4 cm, air 4-7 cm, water 7-12 cm.
NXY, DXY = 40, 0.5
NZ, DZ = 120, 0.1
AIR_CM = (4.0, 7.0)
AIR_DENSITY = 1.205e-3
FIELD_CM = 10.0  # square field side at the phantom surface
SSD_CM = 100.0


def _style(ax: Axes) -> None:
    """Recessive panel styling shared by both axes."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, lw=0.6)
    ax.set_axisbelow(True)


def phantom() -> VoxelGrid:
    """Water slab with an air gap; the engine frame has z along the beam."""
    shape = (NXY, NXY, NZ)
    z_centers = (np.arange(NZ) + 0.5) * DZ
    air = (z_centers >= AIR_CM[0]) & (z_centers < AIR_CM[1])
    density = np.ones(shape)
    material = np.full(shape, WATER, dtype=np.int32)
    density[:, :, air] = AIR_DENSITY
    material[:, :, air] = AIR
    return VoxelGrid(shape=shape, spacing=(DXY, DXY, DZ), density=density, material=material)


def make_engine(backend: str, grid: VoxelGrid, xs: TabulatedCrossSections, field):
    """Return an engine and its name; 'auto' prefers Warp (CUDA, else CPU)."""
    if backend in ("auto", "warp"):
        try:
            import warp as wp

            from pyradmc.backends.warp.engine import WarpEngine

            device = "cuda:0" if wp.is_cuda_available() else "cpu"
            engine = WarpEngine(grid=grid, cross_sections=xs, device=device, magnetic_field=field)
            return engine, f"warp:{device}"
        except Exception:
            if backend == "warp":
                raise
    from pyradmc.backends.ref.engine import ReferenceEngine
    from pyradmc.rng.host import HostRNG

    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG(), magnetic_field=field), "ref"


def main() -> None:
    """Transport the field sweep and render the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--histories", type=int, default=50_000_000)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--backend", choices=("auto", "warp", "ref"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None, help="EPICS library cache")
    args = parser.parse_args()
    t0 = time.perf_counter()

    print("fetching EPDL/EEDL libraries (cached after first run) ...")
    epdl = build.download_library("epdl", args.cache_dir)
    eedl = build.download_library("eedl", args.cache_dir)
    print("compiling the material registry ...")
    data = compile_materials(
        epdl.read_text(encoding="latin-1"), eedl.read_text(encoding="latin-1"), e_max=8.0
    )
    grid = phantom()
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    half = 0.5 * NXY * DXY
    source = SpectralBeamSource(
        ali_rogers_mv("varian-6mv"),
        focal_point=(half, half, -SSD_CM),
        center=(half, half, 0.0),
        width_u=FIELD_CM,
        width_v=FIELD_CM,
    )

    depth = (np.arange(NZ) + 0.5) * DZ
    y = (np.arange(NXY) + 0.5) * DXY - half
    core = slice(NXY // 2 - 4, NXY // 2 + 4)  # central 4 x 4 cm for the depth dose
    ere_band = slice(int((AIR_CM[0] - 0.4) / DZ), int(AIR_CM[0] / DZ))  # last 4 mm of water
    curves = {}
    for b in FIELDS_TESLA:
        # Transverse field along +x: the beam travels along +z, electrons are pushed
        # along -y. On an axial CT this is the superior-inferior bore axis.
        field = None if b == 0.0 else MagneticField.uniform((b, 0.0, 0.0))
        engine, name = make_engine(args.backend, grid, xs, field)
        t1 = time.perf_counter()
        result = engine.run(source, n_histories=args.histories, n_batches=args.batches, seed=SEED)
        print(
            f"  B = {b:4.2f} T on {name}: {time.perf_counter() - t1:5.1f} s, "
            f"{result.provenance.summary() if result.provenance else ''}"
        )
        pdd = result.dose[core, core, :].mean(axis=(0, 1))
        pdd_sigma = np.sqrt((result.dose_sigma[core, core, :] ** 2).sum(axis=(0, 1))) / 64
        profile = result.dose[core, :, ere_band].mean(axis=(0, 2))
        curves[b] = (pdd, pdd_sigma, profile)

    norm = curves[0.0][0][: int(AIR_CM[0] / DZ)].max()  # dmax of the field-free upstream slab
    fig, (ax_pdd, ax_prof) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), facecolor=SURFACE, constrained_layout=True
    )
    fig.suptitle(
        "pyradmc — 6 MV photons in a transverse magnetic field: the electron return effect",
        color=INK,
        fontsize=12,
    )
    for ax in (ax_pdd, ax_prof):
        _style(ax)

    water_mask = (depth < AIR_CM[0]) | (depth >= AIR_CM[1])
    ax_pdd.axvspan(*AIR_CM, color=AIR_BAND, alpha=0.7, label="air gap", zorder=0)
    for b, color in zip(FIELDS_TESLA, SERIES, strict=True):
        pdd, sigma, _ = curves[b]
        d = np.where(water_mask, pdd / norm, np.nan)  # dose to air is not the point here
        s = np.where(water_mask, sigma / norm, np.nan)
        ax_pdd.plot(depth, d, color=color, lw=2, label=f"B = {b:g} T")
        ax_pdd.fill_between(depth, d - s, d + s, color=color, alpha=0.2, lw=0)
    ax_pdd.annotate(
        "electrons return\nfrom the air",
        xy=(AIR_CM[0] - 0.15, curves[1.5][0][int(AIR_CM[0] / DZ) - 2] / norm),
        xytext=(1.0, 1.35),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "lw": 0.8},
    )
    ax_pdd.annotate(
        "build-up restarts",
        xy=(AIR_CM[1] + 0.3, curves[1.5][0][int(AIR_CM[1] / DZ) + 3] / norm),
        xytext=(8.0, 0.35),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "lw": 0.8},
    )
    ax_pdd.set_xlabel("depth z (cm)", color=INK)
    ax_pdd.set_ylabel("dose (relative to field-free dmax)", color=INK)
    ax_pdd.set_title("Central-axis depth dose, 10 x 10 cm field", color=INK, fontsize=10)
    ax_pdd.set_xlim(0.0, NZ * DZ)
    ax_pdd.set_ylim(0.0, 1.7)
    ax_pdd.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")

    for b, color in zip(FIELDS_TESLA, SERIES, strict=True):
        profile = curves[b][2]
        ax_prof.plot(y, profile / norm, color=color, lw=2, label=f"B = {b:g} T")
    ax_prof.annotate(
        "Lorentz force on electrons: -y",
        xy=(-6.0, 0.25),
        xytext=(-9.5, 1.05),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "lw": 0.8},
    )
    ax_prof.set_xlabel("lateral position y (cm)", color=INK)
    ax_prof.set_ylabel("dose (relative to field-free dmax)", color=INK)
    ax_prof.set_title(
        "Lateral profile in the last 4 mm before the gap (B along x)", color=INK, fontsize=10
    )
    ax_prof.set_xlim(-half, half)
    ax_prof.set_ylim(0.0, 1.7)
    ax_prof.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"figure written to {out} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
