"""Phase 5 demo: Monte Carlo dose on a CT-derived geometry.

Builds a synthetic patient-like CT phantom (a water body with a cortical-bone slab and
an off-axis lung region), writes it to a NRRD file, and reads it back through the CT
adapter (:func:`pyRadMC.adapters.ct.read_ct`) — the same path a real
DICOM/NIfTI CT takes: Hounsfield units become per-voxel density (a Schneider-like ramp)
and registry material (HU bins into the ICRP media). It then compiles the tabulated
multi-material backend from the EPICS libraries, transports a 6 MeV photon field, and
renders the classic picture: the material map and the Monte Carlo dose on the same
sagittal slice, so the dose visibly tracks the CT heterogeneities (deeper penetration
through the low-density lung, perturbation around the bone).

On first run this **downloads the ~120 MB EPDL/EEDL libraries** into
``~/.cache/pyRadMC/epics`` (cached thereafter). Needs the ``ct`` extra (SimpleITK) and,
for the figure, ``examples`` (matplotlib). Defaults to Warp (GPU if present). Run from
the repository root::

    python examples/phase5_ct_demo.py [--histories N] [--backend B]

Renders ``phase5_ct_demo.png`` beside this script. Wall-clock is machine-dependent and
not a benchmark.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813 - the library's own conventional alias

from pyRadMC.adapters.ct import read_ct
from pyRadMC.data.tabulated import build
from pyRadMC.data.tabulated.precompile import compile_materials
from pyRadMC.data.tabulated.source import TabulatedCrossSections
from pyRadMC.geometry.source import ParallelBeamSource
from pyRadMC.rng.host import HostRNG

SPACING_MM = (3.0, 3.0, 3.0)
SHAPE = (64, 40, 90)  # (x, y, z): ~19 x 12 x 27 cm


def synthetic_ct() -> sitk.Image:
    """Build a water body with a cortical-bone slab and an off-axis lung block (in HU).

    Deliberately no near-vacuum air pocket *inside* the body: dose-to-medium in a
    ~air-density voxel is dominated by 1/density variance (a rare energetic deposit into
    ~800x smaller mass is a huge dose-per-gram), which speckles the image at finite
    statistics without illustrating anything about the adapter. The air margins outside
    the body are fine — nothing deposits there.
    """
    hu = np.full(SHAPE, 0.0)  # water body
    hu[:5, :, :] = -1000.0  # air outside the body in x (skin-line-ish)
    hu[-5:, :, :] = -1000.0
    # Cortical-bone slab across the field at shallow depth.
    hu[5:-5, :, 24:32] = 1000.0
    # Low-density lung block, off-axis in x, at mid depth.
    hu[8:30, :, 40:72] = -700.0

    image = sitk.GetImageFromArray(np.transpose(hu, (2, 1, 0)))  # engine (x,y,z) -> ITK (z,y,x)
    image.SetSpacing(SPACING_MM)
    image.SetOrigin((0.0, 0.0, 0.0))
    return image


def make_engine(backend: str, grid, xs: TabulatedCrossSections):
    """Return an engine and its name; 'auto' prefers Warp (CUDA, else CPU)."""
    if backend in ("auto", "warp"):
        try:
            import warp as wp

            from pyRadMC.backends.warp.engine import WarpEngine

            device = "cuda:0" if wp.is_cuda_available() else "cpu"
            return WarpEngine(grid=grid, cross_sections=xs, device=device), f"warp:{device}"
        except Exception:
            if backend == "warp":
                raise
    from pyRadMC.backends.ref.engine import ReferenceEngine

    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()), "ref"


def main() -> None:
    """Build the CT, read it through the adapter, transport, and render dose on the slice."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--histories", type=int, default=20_000_000)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--backend", choices=("auto", "warp", "ref"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None, help="EPICS library cache")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        ct_path = Path(tmp) / "phantom.nrrd"
        sitk.WriteImage(synthetic_ct(), str(ct_path))
        print(f"reading CT through the adapter ({ct_path.name}) ...")
        grid = read_ct(str(ct_path))
    print(
        f"  grid {grid.shape}, spacing {grid.spacing} cm, materials present: "
        f"{sorted(int(m) for m in np.unique(grid.material))}"
    )

    print("fetching EPDL/EEDL libraries (cached after first run) ...")
    epdl = build.download_library("epdl", args.cache_dir)
    eedl = build.download_library("eedl", args.cache_dir)
    print("compiling the tabulated material registry ...")
    data = compile_materials(
        epdl.read_text(encoding="latin-1"), eedl.read_text(encoding="latin-1"), e_max=8.0
    )

    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    engine, backend_name = make_engine(args.backend, grid, xs)
    # A 6 MeV field entering along +z, covering the central body in x and all of y.
    nx, ny, nz = SHAPE
    x_cm = nx * SPACING_MM[0] / 10.0
    y_cm = ny * SPACING_MM[1] / 10.0
    source = ParallelBeamSource(
        energy=6.0, z=-1.0, x_range=(0.15 * x_cm, 0.85 * x_cm), y_range=(0.0, y_cm)
    )

    print(f"transporting {args.histories:,} histories on {backend_name} ...")
    t0 = time.perf_counter()
    result = engine.run(source, n_histories=args.histories, n_batches=args.batches, seed=args.seed)
    dt = time.perf_counter() - t0
    print(f"  {args.histories:,} histories in {dt:.1f} s ({backend_name}); not a benchmark")

    # Sagittal slice through the phantom centre (fix y).
    y0 = ny // 2
    density = grid.density[:, y0, :]
    dose = result.dose[:, y0, :]
    extent = (0.0, nz * SPACING_MM[2] / 10.0, nx * SPACING_MM[0] / 10.0, 0.0)  # depth x, x y

    fig, (ax_ct, ax_dose) = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    ax_ct.imshow(density, cmap="bone", extent=extent, aspect="auto", vmin=0.0, vmax=1.9)
    ax_ct.set_title("CT-derived density (sagittal slice)")
    ax_ct.set_ylabel("x (cm)")

    im = ax_dose.imshow(dose, cmap="turbo", extent=extent, aspect="auto")
    # Overlay density contours so the anatomy is visible on the dose panel too.
    ax_dose.contour(
        np.linspace(extent[0], extent[1], density.shape[1]),
        np.linspace(extent[3], extent[2], density.shape[0]),
        density,
        levels=[0.5, 1.4],
        colors="white",
        linewidths=0.6,
        alpha=0.7,
    )
    ax_dose.set_title("Monte Carlo dose (6 MeV, tabulated multi-material)")
    ax_dose.set_xlabel("depth z (cm)")
    ax_dose.set_ylabel("x (cm)")
    fig.colorbar(im, ax=ax_dose, label="dose (a.u.)", fraction=0.046, pad=0.02)
    fig.tight_layout()

    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=160)
    print(f"saved {stem.name}.png")


if __name__ == "__main__":
    main()
