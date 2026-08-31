"""Demo: a measured-primary-fluence virtual source model (Tacke et al. 2006).

The virtual source model of Tacke, Szymanowski, Oelfke et al., Med. Phys. 33
(2006) 1125-1132 (doi:10.1118/1.2181298): photons are sampled on a plane in the
treatment head according to the machine's **measured radial primary fluence**,
then given a direction from a finite Gaussian focal spot and an energy from a
photon spectrum. In pyradmc that is
:class:`~pyradmc.geometry.source.PrimaryFluenceBeamSource` fed a
:class:`~pyradmc.geometry.fluence.RadialFluence` table.

Three things the figure shows:

* the **input fluence** psi(r) — the flattening filter's horn and the primary
  collimator's circular field edge, quoted at isocentre as a commissioning curve
  is (left);
* the **horn transfers to the dose**: a 30 x 30 cm profile at 10 cm depth against
  the same field run with a flat fluence, both normalized on the central axis.
  Nothing but the per-history weight differs between the two (middle);
* the **penumbra checks the focal-spot width**: the source sigma is *derived*
  from an assumed 5 mm 80-20 penumbra at isocentre by the usual quadrature
  back-calculation, and a point-spot control run isolates the transport blur, so
  the pair is a closed-loop test of that derivation (right).

That last panel is worth reading before trusting the derived spot size. The
assumed transport blur holds up (a point spot measures ~3.6 mm here), but the
measured total penumbra lands well above what ``P_total^2 = P_geom^2 +
P_transport^2`` predicts: 80-20 widths of a Gaussian spot and of the electron
transport kernel do **not** add in quadrature, because the transport kernel is
heavy-tailed rather than Gaussian. Quadrature therefore over-estimates the spot
size a given penumbra implies. Set ``sigma_u``/``sigma_v`` from a measured
profile fit where the number matters.

The fluence table here is **synthetic** — an analytic stand-in with the shape of
a Siemens Artiste commissioning curve (flat-to-horned core, 50% at a 24 cm
radius). Point ``--fluence-file`` at a real two-column ``radius_mm fluence``
table (the PPBKC ``primflu.dat`` layout) to run the model on measured data.

The emission plane sits at the field-defining collimator, so its rectangle is a
perfectly absorbing aperture and the geometric penumbra comes out of the source
geometry alone — no material data needed. For a real tungsten jaws-and-MLC head,
move the plane upstream of the devices and wrap in
:class:`~pyradmc.geometry.collimation.CollimatedSource`; see
``examples/collimation_demo.py``.

Uses analytic water cross-sections (no downloads); defaults to Warp (GPU if
present), else the reference engine. Run from the repository root::

    python examples/virtual_source_demo.py [--histories N] [--backend B]
        [--fluence-file PATH]

Renders ``virtual_source_demo.png`` beside this script.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.fluence import RadialFluence
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PrimaryFluenceBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv
from pyradmc.rng.host import HostRNG

BEAM = "siemens-6mv"  # the vendor of the machine the source paper studied
SAD_CM = 100.0
SSD_CM = 90.0  # phantom surface; isocentre therefore sits at 10 cm depth
COLLIMATOR_CM = 34.5  # field-defining device, from the machine's parameter file
AXIS_X_CM, AXIS_Y_CM = 18.0, 15.0  # beam axis in the engine frame

PENUMBRA_ISO_MM = 5.0  # the assumed 80-20 penumbra this model is tuned to
TRANSPORT_PENUMBRA_MM = 3.5  # typical 6 MV lateral electron transport + phantom scatter

WIDE_FIELD_CM = 30.0  # shows the horn
NARROW_FIELD_CM = 10.0  # shows the penumbra
PROFILE_DEPTH_CM = 10.0  # = the isocentre plane, by the SSD choice above

HORNED_BLUE = "#2b5fa8"
FLAT_GREY = "#8a8a8a"
ACCENT = "#d8853b"


def spot_sigma_cm() -> float:
    """Back out the focal-spot sigma from the assumed penumbra at isocentre.

    The 80-20 width of a Gaussian is ``2 * 0.84162 * sigma = 0.71481 * FWHM``.
    Removing the transport contribution in quadrature leaves the geometric
    penumbra, which is the source FWHM magnified by ``(SAD - d) / d`` for a
    collimator at distance ``d``::

        P_geom = sqrt(P_total^2 - P_transport^2)
        FWHM_source = P_geom / 0.71481 * d / (SAD - d)

    Stated assumption: ``TRANSPORT_PENUMBRA_MM``. It is an estimate, not a
    measurement — the right-hand panel is what checks the result.
    """
    p_geom_mm = math.sqrt(PENUMBRA_ISO_MM**2 - TRANSPORT_PENUMBRA_MM**2)
    fwhm_mm = p_geom_mm / 0.71481 * COLLIMATOR_CM / (SAD_CM - COLLIMATOR_CM)
    return fwhm_mm / 2.35482 / 10.0  # FWHM mm -> sigma cm


def synthetic_fluence() -> RadialFluence:
    """Build an analytic stand-in shaped like a flattened 6 MV commissioning curve.

    ``psi(r) = (1 + 0.20 (r/20)^2) * erfc((r - 23.7) / (sqrt(2) * 4.59)) / 2``,
    radius in cm at isocentre: a quadratic horn peaking near 1.08 at a 14 cm
    radius, rolled off by the primary collimator's circular field (50% at 23.7 cm,
    essentially closed by 33 cm).
    """
    r = np.linspace(0.0, 40.0, 401)
    horn = 1.0 + 0.20 * (r / 20.0) ** 2
    edge = np.array([0.5 * math.erfc((value - 23.7) / (math.sqrt(2.0) * 4.59)) for value in r])
    return RadialFluence(radii=r, values=horn * edge, reference_distance=SAD_CM)


def flat_fluence(table: RadialFluence) -> RadialFluence:
    """Build a uniform table over the same support — the control for the horn panel."""
    return RadialFluence(
        radii=(0.0, table.max_radius), values=(1.0, 1.0), reference_distance=SAD_CM
    )


def make_source(
    fluence: RadialFluence, field_iso_cm: float, sigma: float | None = None
) -> PrimaryFluenceBeamSource:
    """Build the virtual source for one square field, stated at the isocentre plane."""
    half = 0.5 * field_iso_cm * COLLIMATOR_CM / SAD_CM  # project the opening to the plane
    sigma = spot_sigma_cm() if sigma is None else sigma
    return PrimaryFluenceBeamSource(
        spectrum=ali_rogers_mv(BEAM),
        fluence=fluence,
        focal_point=(AXIS_X_CM, AXIS_Y_CM, -SSD_CM),
        center=(AXIS_X_CM, AXIS_Y_CM, -SSD_CM + COLLIMATOR_CM),
        width_u=2.0 * half,
        width_v=2.0 * half,
        sigma_u=sigma,
        sigma_v=sigma,
    )


def make_engine(backend: str, grid: VoxelGrid, xs: AnalyticCrossSections):  # type: ignore[no-untyped-def]
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


def penumbra_80_20_mm(x_cm: np.ndarray, normalized: np.ndarray, field_half_cm: float) -> float:
    """80-20 width on the +x side, scanning outward from the beam axis.

    Each level is taken at its *first* outward crossing and the search stops a
    few cm past the geometric edge, so a noisy voxel in the low-dose tail cannot
    drag the 20% point outward — which sorting the profile by dose would let it
    do.
    """
    window = (x_cm > 0.0) & (x_cm < field_half_cm + 3.0)
    x_w, y_w = x_cm[window], normalized[window]

    def crossing(level: float) -> float:
        below = np.nonzero(y_w < level)[0]
        if below.size == 0 or below[0] == 0:
            return float("nan")
        i = below[0]
        return float(np.interp(level, (y_w[i], y_w[i - 1]), (x_w[i], x_w[i - 1])))

    return 10.0 * (crossing(0.20) - crossing(0.80))


def rebin(values: np.ndarray, factor: int) -> np.ndarray:
    """Average adjacent samples in groups of ``factor``, dropping any remainder."""
    usable = values.size // factor * factor
    return values[:usable].reshape(-1, factor).mean(axis=1)


def main() -> None:
    """Transport the horned, flat and narrow fields and render the three panels."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--histories", type=int, default=8_000_000, help="per field")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--backend", choices=("auto", "warp", "ref"), default="auto")
    parser.add_argument(
        "--fluence-file",
        type=Path,
        default=None,
        help="two-column 'radius_mm fluence' table (PPBKC primflu.dat); default synthetic",
    )
    args = parser.parse_args()

    if args.fluence_file is None:
        fluence, source_label = synthetic_fluence(), "synthetic (Artiste-like)"
    else:
        fluence = RadialFluence.from_file(args.fluence_file, reference_distance=SAD_CM)
        source_label = args.fluence_file.name

    sigma = spot_sigma_cm()
    print(f"fluence: {source_label}, {fluence.radii.size} points to {fluence.max_radius:.1f} cm")
    print(f"focal spot: sigma = {10 * sigma:.2f} mm (FWHM {10 * sigma * 2.35482:.2f} mm)")

    grid = VoxelGrid.uniform_water(shape=(180, 60, 40), spacing=(0.2, 0.5, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine, backend_name = make_engine(args.backend, grid, xs)

    runs = {
        "horned": make_source(fluence, WIDE_FIELD_CM),
        "flat": make_source(flat_fluence(fluence), WIDE_FIELD_CM),
        "narrow": make_source(fluence, NARROW_FIELD_CM),
        # Point-spot control: same field, zero focal spot, so its penumbra is the
        # transport blur alone and the finite-spot run's geometric term follows.
        "point": make_source(fluence, NARROW_FIELD_CM, sigma=0.0),
    }
    doses = {}
    t0 = time.perf_counter()
    for name, source in runs.items():
        print(f"transporting {args.histories:,} histories ({name}) on {backend_name} ...")
        doses[name] = engine.run(
            source, n_histories=args.histories, n_batches=args.batches, seed=args.seed
        ).dose
    total = len(runs) * args.histories
    print(f"  {total:,} histories in {time.perf_counter() - t0:.1f} s; not a benchmark")

    x = (np.arange(grid.shape[0]) + 0.5) * grid.spacing[0] - AXIS_X_CM
    iz = int(PROFILE_DEPTH_CM / grid.spacing[2])
    iy0 = grid.shape[1] // 2
    strip = slice(iy0 - 6, iy0 + 6)  # 6 cm in y: the profile is flat there, so this is
    on_axis = np.abs(x) < 0.5  # free statistics, not smoothing along the gradient

    def profile(dose: np.ndarray) -> np.ndarray:
        """Lateral profile at the reference depth, normalized to the central axis."""
        lateral = dose[:, strip, iz].mean(axis=1)
        return lateral / lateral[on_axis].mean()

    fig, (ax_flu, ax_horn, ax_pen) = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax in (ax_flu, ax_horn, ax_pen):
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.title.set_fontsize(10)

    # --- 1. the input commissioning curve ---------------------------------------
    ax_flu.plot(fluence.radii, fluence.values, color=HORNED_BLUE, linewidth=2.0)
    ax_flu.axhline(1.0, color=FLAT_GREY, linewidth=0.8, linestyle="--")
    peak = int(np.argmax(fluence.values))  # the horn: interpolate on the falling side only
    r50 = float(
        np.interp(
            0.5 * fluence.values[peak], fluence.values[peak:][::-1], fluence.radii[peak:][::-1]
        )
    )
    ax_flu.axvline(r50, color=ACCENT, linewidth=1.0, linestyle=":")
    ax_flu.annotate(f"50% at r = {r50:.1f} cm", (r50 + 0.7, 0.55), color=ACCENT, fontsize=9)
    ax_flu.set_title(f"Input primary fluence psi(r)\n{source_label}, at isocentre")
    ax_flu.set_xlabel("off-axis radius at isocentre (cm)")
    ax_flu.set_ylabel("relative primary fluence")
    ax_flu.set_ylim(0.0, None)

    # --- 2. the horn transfers to the dose --------------------------------------
    # Rebinned to 1 cm laterally: the horn varies over ~10 cm, so this costs no
    # structure and buys back the sqrt(5) in per-voxel noise the fine penumbra
    # grid (panel 3) forces on the profile.
    bin_x, horned, flat = (
        rebin(v, 5) for v in (x, profile(doses["horned"]), profile(doses["flat"]))
    )
    predicted = flat * fluence.at_radii(np.abs(bin_x)) / fluence.at_radius(0.0)
    ax_horn.plot(bin_x, 100.0 * flat, color=FLAT_GREY, linewidth=2.0, label="flat psi(r) = 1")
    ax_horn.plot(bin_x, 100.0 * horned, color=HORNED_BLUE, linewidth=2.0, label="measured psi(r)")
    ax_horn.plot(
        bin_x,
        100.0 * predicted,
        color=ACCENT,
        linewidth=1.4,
        linestyle="--",
        label="flat run x psi(r)/psi(0)",
    )
    ax_horn.axhline(100.0, color="black", linewidth=0.6, alpha=0.4)
    ax_horn.set_xlim(-0.6 * WIDE_FIELD_CM, 0.6 * WIDE_FIELD_CM)
    ax_horn.set_ylim(0.0, None)
    ax_horn.set_title(
        f"{WIDE_FIELD_CM:.0f} x {WIDE_FIELD_CM:.0f} cm profile, {PROFILE_DEPTH_CM:.0f} cm deep\n"
        "same geometry and seed; only the weight differs"
    )
    ax_horn.set_xlabel("off-axis distance at isocentre (cm)")
    ax_horn.set_ylabel("dose (% of central axis)")
    ax_horn.legend(loc="lower center", fontsize=8)

    # --- 3. the penumbra checks the derived spot width --------------------------
    # The point-spot run carries the transport blur alone, so removing it in
    # quadrature isolates the geometric penumbra the source sigma was derived from.
    edges = {}
    for name, color, label in (
        ("point", FLAT_GREY, "point spot"),
        ("narrow", HORNED_BLUE, f"sigma {10 * sigma:.2f} mm"),
    ):
        p = profile(doses[name])
        edges[name] = penumbra_80_20_mm(x, p, 0.5 * NARROW_FIELD_CM)
        ax_pen.plot(
            x, 100.0 * p, color=color, linewidth=2.0, label=f"{label}: {edges[name]:.1f} mm"
        )
    design_geometric_mm = math.sqrt(PENUMBRA_ISO_MM**2 - TRANSPORT_PENUMBRA_MM**2)
    quadrature_mm = math.hypot(edges["point"], design_geometric_mm)

    for level in (80.0, 20.0):
        ax_pen.axhline(level, color=ACCENT, linewidth=0.8, linestyle=":")
    ax_pen.axvline(0.5 * NARROW_FIELD_CM, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax_pen.set_xlim(0.5 * NARROW_FIELD_CM - 2.5, 0.5 * NARROW_FIELD_CM + 2.5)
    ax_pen.set_ylim(0.0, 110.0)
    ax_pen.set_title(
        f"{NARROW_FIELD_CM:.0f} x {NARROW_FIELD_CM:.0f} cm edge: 80-20 penumbra\n"
        f"quadrature {quadrature_mm:.1f} mm vs {edges['narrow']:.1f} mm measured"
    )
    ax_pen.set_xlabel("off-axis distance at isocentre (cm)")
    ax_pen.set_ylabel("dose (% of central axis)")
    ax_pen.legend(loc="lower left", fontsize=8)

    print(f"80-20 penumbra: {edges['narrow']:.2f} mm total, {edges['point']:.2f} mm transport-only")
    print(
        f"  transport blur: {edges['point']:.2f} mm measured against the "
        f"{TRANSPORT_PENUMBRA_MM:.1f} mm assumed in the sigma derivation"
    )
    print(
        f"  quadrature predicted {quadrature_mm:.2f} mm total against {edges['narrow']:.2f} mm "
        f"measured (target {PENUMBRA_ISO_MM:.0f} mm); 80-20 widths do not add in quadrature"
    )

    fig.tight_layout()
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.png", dpi=160)
    print(f"saved {stem.name}.png")


if __name__ == "__main__":
    main()
