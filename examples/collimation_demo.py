"""Demo: beam-limiting devices — jaws and a rounded-tip MLC shaping a 6 MV field.

The BLD workstream end to end: a Varian-type 6 MV spectrum (Ali & Rogers 2012)
from a 1 mm Gaussian focal spot is emitted on a plane above the head
(:class:`~pyRadMC.geometry.source.GaussianSpotBeamSource`), collimated by two jaw
pairs and a 10-pair rounded-tip MLC set to a staircase field with one fully
closed pair, and transported into a water phantom two ways:

* the **deterministic wrapper** (:class:`~pyRadMC.geometry.collimation.
  CollimatedSource`) — Beer-Lambert weights, the Dij-baseline configuration;
* the **head pre-solve** (``first_compton``) — an exit-plane phase space
  carrying collimator scatter and air-generated contaminant electrons. On a Warp
  engine it runs on the device and is transported with no copy back
  (:func:`~pyRadMC.backends.warp.presolve.presolve_head_device`); on the reference
  engine it runs on the host (:func:`~pyRadMC.geometry.head.presolve_head`).

The figure reads at a glance: the beam's-eye-view transmission map shows the
staircase aperture, the rounded tips and the closed-pair stripe; the inline
profile shows the closed pair's cold stripe with the under-leaf transmission
floor (and the pre-solve sitting slightly above the wrapper there — collimator
scatter); the depth dose is a normal 6 MV curve.

With the EPICS libraries available (cached or downloadable), the devices are real
tungsten from the compiled EPDL table; otherwise pass ``--material water`` for a
dependency-free water-equivalent stand-in (thicker, softer shielding — the
geometry chain is identical). Run from the repository root::

    python examples/collimation_demo.py [--histories N] [--backend B] [--material M]

Renders ``collimation_demo.png`` beside this script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyRadMC.data.materials import AIR, TUNGSTEN, WATER
from pyRadMC.geometry.collimation import (
    MLC,
    BeamFrame,
    BeamLimitingStack,
    CollimatedSource,
    JawPair,
    project_between_planes,
)
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.head import AirColumn, presolve_head
from pyRadMC.geometry.source import GaussianSpotBeamSource
from pyRadMC.geometry.spectrum import ali_rogers_mv
from pyRadMC.rng.host import HostRNG

BEAM = "varian-6mv"
SAD_CM = 100.0
CENTER_CM = 15.0  # beam axis; the phantom is 30 cm wide
SPOT_SIGMA_CM = 0.1  # 1 mm focal spot
FIELD_HALF_ISO = 5.0  # jaw opening at the isocenter plane
N_PAIRS = 10
CLOSED_PAIR = 6

JAW_U = (28.0, 35.0)  # local w extents of the devices (cm from the focal spot)
JAW_V = (36.0, 43.0)
MLC_Z = (48.0, 55.0)
TIP_RADIUS = 8.0


def make_stack(material: int, density: float) -> BeamLimitingStack:
    """Jaws plus a staircase MLC, all settings stated at the isocenter plane."""
    frame = BeamFrame(origin=(CENTER_CM, CENTER_CM, -SAD_CM))

    def at(setting: float, z_range: tuple[float, float]) -> float:
        return float(project_between_planes(setting, SAD_CM, sum(z_range) / 2.0))

    # The staircase: tips walk from 1 to 4 cm (at iso) across the pairs; one pair
    # is fully closed and shows up as a cold stripe in the field.
    tips_pos_iso = 1.0 + 3.0 * np.arange(N_PAIRS) / (N_PAIRS - 1)
    tips_neg_iso = -tips_pos_iso[::-1]
    tips_pos_iso[CLOSED_PAIR] = 0.4
    tips_neg_iso[CLOSED_PAIR] = 0.4
    edges_iso = np.linspace(-FIELD_HALF_ISO, FIELD_HALF_ISO, N_PAIRS + 1)

    mlc_mid = sum(MLC_Z) / 2.0
    return BeamLimitingStack(
        frame=frame,
        devices=(
            JawPair(
                axis="u",
                z_top=JAW_U[0],
                z_bottom=JAW_U[1],
                edge_neg=at(-FIELD_HALF_ISO, JAW_U),
                edge_pos=at(FIELD_HALF_ISO, JAW_U),
                material=material,
                density=density,
            ),
            JawPair(
                axis="v",
                z_top=JAW_V[0],
                z_bottom=JAW_V[1],
                edge_neg=at(-FIELD_HALF_ISO, JAW_V),
                edge_pos=at(FIELD_HALF_ISO, JAW_V),
                material=material,
                density=density,
            ),
            MLC(
                z_top=MLC_Z[0],
                z_bottom=MLC_Z[1],
                leaf_edges_v=tuple(
                    float(project_between_planes(e, SAD_CM, mlc_mid)) for e in edges_iso
                ),
                tips_neg=tuple(
                    float(project_between_planes(t, SAD_CM, mlc_mid)) for t in tips_neg_iso
                ),
                tips_pos=tuple(
                    float(project_between_planes(t, SAD_CM, mlc_mid)) for t in tips_pos_iso
                ),
                tip_radius=TIP_RADIUS,
                material=material,
                density=density,
            ),
        ),
    )


def make_cross_sections(material_choice: str, grid: VoxelGrid):
    """Compiled EPICS table for tungsten devices, or the analytic water source."""
    if material_choice == "tungsten":
        from pyRadMC.data.tabulated.build import download_library
        from pyRadMC.data.tabulated.precompile import compile_materials
        from pyRadMC.data.tabulated.source import TabulatedCrossSections

        print("compiling the tabulated table (EPICS cache/download) ...")
        epdl = download_library("epdl").read_text(encoding="latin-1")
        eedl = download_library("eedl").read_text(encoding="latin-1")
        data = compile_materials(epdl, eedl, e_max=7.0)
        xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
        return xs, TUNGSTEN, 19.30, AIR
    from pyRadMC.data.analytic import AnalyticCrossSections

    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return xs, WATER, 1.0, WATER


def make_engine(backend: str, grid: VoxelGrid, xs):
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


def transmission_map(stack: BeamLimitingStack, xs, material: int, density: float):
    """Beam's-eye-view transmission at 2 MeV on a ray grid through the iso plane."""
    half = 6.5
    n = 260
    axis = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    targets = np.stack([CENTER_CM + gx.ravel(), CENTER_CM + gy.ravel(), np.zeros(gx.size)], axis=1)
    focal = np.array([CENTER_CM, CENTER_CM, -SAD_CM])
    directions = targets - focal
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.tile(focal, (gx.size, 1))
    thickness = stack.path_lengths(origins, directions)
    mu = np.array(
        [density * xs.mu_over_rho_total(2.0, material) for _ in range(len(stack.devices))]
    )
    tau = (thickness * mu).sum(axis=1)
    return axis, np.exp(-tau).reshape(n, n)


def main() -> None:
    """Collimate, pre-solve, transport, and render the three-panel figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--histories", type=int, default=10_000_000)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--backend", choices=("auto", "warp", "ref"), default="auto")
    parser.add_argument("--material", choices=("tungsten", "water"), default="tungsten")
    args = parser.parse_args()

    grid = VoxelGrid.uniform_water(shape=(120, 120, 60), spacing=(0.25, 0.25, 0.5))
    xs, device_material, device_density, air_material = make_cross_sections(args.material, grid)
    stack = make_stack(device_material, device_density)

    spectrum = ali_rogers_mv(BEAM)
    source = GaussianSpotBeamSource(
        spectrum=spectrum,
        focal_point=(CENTER_CM, CENTER_CM, -SAD_CM),
        center=(CENTER_CM, CENTER_CM, -SAD_CM + 20.0),  # emission plane above the jaws
        width_u=3.0,
        width_v=3.0,
        sigma_u=SPOT_SIGMA_CM,
        sigma_v=SPOT_SIGMA_CM,
    )

    engine, backend_name = make_engine(args.backend, grid, xs)
    device = getattr(engine, "device", None)  # a Warp device string, or None for ref

    presolve_kwargs = dict(
        stack=stack,
        cross_sections=xs,
        n_histories=args.histories,
        seed=args.seed,
        exit_z=SAD_CM - 5.0,
        air=AirColumn(
            z_start=MLC_Z[1] + 0.5, z_end=SAD_CM - 5.0, material=air_material, density=1.205e-3
        ),
        mode="first_compton",
    )
    where = f"on device ({device})" if device is not None else "on the host"
    print(f"pre-solving the head ({args.histories:,} rays) {where} ...")
    t0 = time.perf_counter()
    if device is not None:
        # The Warp device pre-solve, transported with no copy back (DevicePhaseSpace).
        from pyRadMC.backends.warp.presolve import presolve_head_device

        phsp = presolve_head_device(
            source, device=device, return_device_source=True, **presolve_kwargs
        )
    else:
        phsp = presolve_head(source, **presolve_kwargs)
    print(f"  {len(phsp):,} exit-plane particles in {time.perf_counter() - t0:.1f} s")

    wrapped = CollimatedSource(source, stack, xs)
    runs = {}
    for label, src in (("wrapper", wrapped), ("pre-solve", phsp)):
        t0 = time.perf_counter()
        runs[label] = engine.run(
            src, n_histories=args.histories, n_batches=args.batches, seed=args.seed + 1
        )
        print(
            f"  {label}: {args.histories:,} histories in {time.perf_counter() - t0:.1f} s "
            f"({backend_name}); not a benchmark"
        )

    # ------------------------------------------------------------------ figure
    y = (np.arange(grid.shape[1]) + 0.5) * grid.spacing[1] - CENTER_CM
    z = (np.arange(grid.shape[2]) + 0.5) * grid.spacing[2]
    ix0 = grid.shape[0] // 2
    iy0 = grid.shape[1] // 2

    fig, (ax_bev, ax_prof, ax_pdd) = plt.subplots(1, 3, figsize=(14.0, 4.4))

    axis, bev = transmission_map(stack, xs, device_material, device_density)
    image = ax_bev.imshow(
        bev.T,
        origin="lower",
        extent=(axis[0], axis[-1], axis[0], axis[-1]),
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(image, ax=ax_bev, label="transmission at 2 MeV")
    ax_bev.set_title("Beam's-eye view of the stack\n(staircase MLC, one closed pair)")
    ax_bev.set_xlabel("x at isocenter (cm)")
    ax_bev.set_ylabel("y at isocenter (cm)")

    iz = int(5.0 / grid.spacing[2])  # inline profile at 5 cm depth
    reference = runs["wrapper"].dose[ix0 - 2 : ix0 + 2, :, iz].mean(axis=0)
    for label, color in (("wrapper", "#2b5fa8"), ("pre-solve", "#d8853b")):
        profile = runs[label].dose[ix0 - 2 : ix0 + 2, :, iz].mean(axis=0)
        ax_prof.plot(y, profile / reference.max(), color=color, linewidth=1.8, label=label)
    ax_prof.set_title("Inline profile at 5 cm depth\n(the closed pair's cold stripe)")
    ax_prof.set_xlabel("y off-axis (cm)")
    ax_prof.set_ylabel("dose (fraction of open max)")
    ax_prof.grid(alpha=0.25, linewidth=0.6)
    ax_prof.legend()

    for label, color in (("wrapper", "#2b5fa8"), ("pre-solve", "#d8853b")):
        pdd = runs[label].dose[ix0 - 3 : ix0 + 3, iy0 - 3 : iy0 + 3, :].mean(axis=(0, 1))
        ax_pdd.plot(z, 100.0 * pdd / pdd.max(), color=color, linewidth=1.8, label=label)
    ax_pdd.set_title(f"Central-axis depth dose\n({BEAM}, {args.material} devices)")
    ax_pdd.set_xlabel("depth (cm)")
    ax_pdd.set_ylabel("dose (% of maximum)")
    ax_pdd.grid(alpha=0.25, linewidth=0.6)
    ax_pdd.legend()

    fig.tight_layout()
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=160)
    print(f"saved {stem.name}.png")


if __name__ == "__main__":
    main()
