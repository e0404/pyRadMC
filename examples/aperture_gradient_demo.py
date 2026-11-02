"""Leaf-tip and jaw-edge dose gradients by Monte Carlo: four estimators, one oracle.

Renders one figure (``aperture_gradient_demo.png``, saved beside this script) for a
2 MeV parallel photon beam into water through a tungsten jaw pair over a rounded-tip
MLC, differentiating the dose with respect to two edge coordinates — the +x tip of
leaf pair 0 (a smooth, rounded penumbra) and the +x jaw edge (a straight face, a
pure jump in transmission). The estimators (:mod:`pyradmc.geometry.aperture_gradient`):

- **A, oracle** — the common-random-number central difference, one reweighted run;
  unbiased for the finite difference at its step (0.1 mm). Only rays within the
  edge's penumbra +- h carry a nonzero weight, so it runs on that slice of the
  field and is rescaled to the full field's fluence — exact, and still far less
  efficient than C. At h = 0.5 mm its O(h^2) bias was measured at chi2/ndf ~ 12
  against C on these 1 mm voxels (falling to 1.0 at 0.1 mm): the response varies
  within a voxel column, and a +-h window averages over it.
- **B, pathwise (Warp tape)** — each ray's ``dT/dx`` from Warp's autodiff of the
  collimation geometry: what a fully taped simulation would return. Correct at the
  rounded tip, **identically zero at the straight edge** (the boundary term is lost).
- **C, edge-sampled** — rays drawn from ``|dT/dx|`` itself: penumbra rays for the
  tip, the edge line for the jaw. Unbiased in both cases.
- **D, bixel chain rule** — what direct aperture optimization computes from a bixel
  Dij: the same derivative measure spread uniformly over the bixel holding the edge,
  as for a sharp leaf end.
- **E, profile-aware chain rule** — the same bixel Dij, but each bixel's ``dw/dx`` is
  the drop of the edge's transmission profile across it: the derivative mass goes to
  the bixels the penumbra actually crosses. Converges to C as the pitch shrinks.

Panels: (a) the forward dose across both edges; (b, c) the gradient profiles, depth
averaged over 1-4 cm; (d) the chain rules' error where an optimizer feels it: the
leaf gradient of the dose summed over a structure, as a function of how far the
structure's edge sits from the tip. The console reports each estimator's reduced
chi-squared against the oracle, its efficiency (1 / (relative variance x seconds))
over the gradient region, and, as the tip moves through one 5 mm bixel, each 5 mm
chain rule's worst structure-gradient error.

The parallel-beam estimators' restrictions (refused by the sources where they bite):
parallel beam, monoenergetic, unfocused parameter devices; the divergent-fan
estimator is demonstrated by ``aperture_gradient_fan_dao.py``. With the EPICS libraries available
(cached or downloadable) the devices are real tungsten; ``--devices surrogate``
uses water at 16.6 g/cm3, close to tungsten's attenuation at 2 MeV. Run from the
repository root (requires the ``examples`` extra, i.e. matplotlib)::

    python examples/aperture_gradient_demo.py

Runtime is about a minute on a CUDA device; ``--scale`` trades statistics for time.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pyradmc.data.materials import WATER
from pyradmc.geometry.aperture_gradient import (
    BixelChainRuleSource,
    EdgeDerivativeSource,
    EdgeParameter,
    PathwiseDerivativeSource,
    ProfileChainRuleSource,
    RayFiniteDifferenceSource,
    edge_value,
    with_edge,
)
from pyradmc.geometry.collimation import (
    MLC,
    BeamFrame,
    BeamLimitingStack,
    CollimatedSource,
    JawPair,
)
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource

ENERGY = 2.0  # MeV
FIELD_X = (0.5, 9.5)
FIELD_Y = (0.0, 4.0)
TIP = 5.15  # leaf pair 0, +x tip (cm): mid-voxel, off every 2.5 mm bixel boundary
JAW_EDGE = 8.35  # +x jaw edge, visible in strip 1 where pair 1 is open to the field edge
DEPTH_CM = (1.0, 4.0)  # gradient profiles are averaged over this depth band
FD_STEP = 0.01  # cm: small against the 1 mm voxels (0.05 measured biased, see docstring)
ORACLE_SLICE = (-0.3, 0.9)  # cm about the edge: covers the tip penumbra (0.6 cm) +- h
TIP_SWEEP = (5.05, 5.15, 5.25, 5.35, 5.45)  # tip positions through one 5 mm bixel
OAR_OFFSETS = np.linspace(-1.0, 2.0, 31)  # structure boundary minus tip position (cm)

MLC_TIP = EdgeParameter(device=1, side="pos", pair=0)
JAW_POS = EdgeParameter(device=0, side="pos")


class VectorizedParallelBeam(ParallelBeamSource):
    """The parallel beam with a vectorized batch route (two uniforms per history).

    Pre-sampling through ``emit`` costs ~10 us per history in Python; the demo needs
    tens of millions. Same chunk-invariant ``PCG64(seed)``-advanced stream as the
    library's vectorized sources; ``emit`` is unchanged.
    """

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, Any]:
        """Draw the field positions for histories ``[offset, offset + n)`` at once."""
        bitgen = np.random.PCG64(seed)
        bitgen.advance(2 * history_offset)
        u = np.random.Generator(bitgen).random((n, 2))
        x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * u[:, 0]
        y = self.y_range[0] + (self.y_range[1] - self.y_range[0]) * u[:, 1]
        f32 = np.float32
        return {
            "particle_type": np.ones(n, dtype=np.int32),
            "energy": np.full(n, self.energy, dtype=f32),
            "x": x.astype(f32),
            "y": y.astype(f32),
            "z": np.full(n, self.z, dtype=f32),
            "ux": np.zeros(n, dtype=f32),
            "uy": np.zeros(n, dtype=f32),
            "uz": np.ones(n, dtype=f32),
            "weight": np.ones(n, dtype=f32),
        }


def make_grid() -> VoxelGrid:
    """10 x 4 x 10 cm of water: 1 mm across the edges, one 2 cm row per leaf strip."""
    return VoxelGrid.uniform_water(shape=(100, 2, 20), spacing=(0.1, 2.0, 0.5))


def make_stack(material: int, density: float, tip: float = TIP) -> BeamLimitingStack:
    """Build a 7.8 cm jaw pair over a 6 cm MLC with 8 cm tip radius (parallel beam)."""
    return BeamLimitingStack(
        frame=BeamFrame(origin=(0.0, 0.0, -100.0)),
        devices=(
            JawPair(
                axis="u",
                z_top=28.0,
                z_bottom=35.8,
                edge_neg=0.0,
                edge_pos=JAW_EDGE,
                material=material,
                density=density,
            ),
            MLC(
                z_top=50.0,
                z_bottom=56.0,
                leaf_edges_v=(0.0, 2.0, 4.0),
                tips_neg=(1.05, 1.05),
                tips_pos=(tip, 9.5),
                tip_radius=8.0,
                material=material,
                density=density,
            ),
        ),
    )


def make_cross_sections(choice: str, grid: VoxelGrid):
    """Tabulated tungsten if the EPICS libraries are available, else the surrogate."""
    if choice in ("auto", "tungsten"):
        try:
            from pyradmc.data.materials import TUNGSTEN
            from pyradmc.data.tabulated.build import download_library
            from pyradmc.data.tabulated.precompile import compile_materials
            from pyradmc.data.tabulated.source import TabulatedCrossSections

            print("compiling the tabulated table (EPICS cache/download) ...")
            epdl = download_library("epdl").read_text(encoding="latin-1")
            eedl = download_library("eedl").read_text(encoding="latin-1")
            data = compile_materials(epdl, eedl, e_max=ENERGY * 1.5)
            xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
            return xs, TUNGSTEN, 19.30, "tungsten"
        except Exception:
            if choice == "tungsten":
                raise
    from pyradmc.data.analytic import AnalyticCrossSections

    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return xs, WATER, 16.6, "water at 16.6 g/cm3 (tungsten-like mu at 2 MeV)"


def make_engine(backend: str, grid: VoxelGrid, xs):
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
    from pyradmc.rng.host import HostRNG

    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()), "ref"


class Runner:
    """Runs sources at a fixed batch count and records wall time per run."""

    def __init__(self, engine, n_batches: int, seed: int) -> None:
        self.engine = engine
        self.n_batches = n_batches
        self.seed = seed

    def __call__(self, source, n_histories: int) -> tuple[Any, float]:
        """Transport ``source`` (rounded to whole batches); return result and seconds."""
        n = max(self.n_batches, n_histories // self.n_batches * self.n_batches)
        start = time.perf_counter()
        result = self.engine.run(source, n_histories=n, n_batches=self.n_batches, seed=self.seed)
        self.seed += 1  # every run independent of every other
        return result, time.perf_counter() - start


def depth_profile(result, row: int, grid: VoxelGrid) -> tuple[np.ndarray, np.ndarray]:
    """Lateral profile in one strip row, averaged over the depth band (mean, sigma)."""
    z = (np.arange(grid.shape[2]) + 0.5) * grid.spacing[2]
    band = (z >= DEPTH_CM[0]) & (z < DEPTH_CM[1])
    dose = result.dose[:, row, band]
    sigma = result.dose_sigma[:, row, band]
    return dose.mean(axis=1), np.sqrt((sigma**2).sum(axis=1)) / band.sum()


def compare(result, reference, roi: np.ndarray) -> tuple[float, int]:
    """Reduced chi-squared of two independent runs over the ROI voxels."""
    variance = result.dose_sigma**2 + reference.dose_sigma**2
    use = roi & (variance > 0.0)
    z2 = (result.dose - reference.dose) ** 2 / np.where(use, variance, 1.0)
    return float(z2[use].sum() / use.sum()), int(use.sum())


def efficiency(result, seconds: float, roi: np.ndarray) -> float:
    """Figure of merit 1 / (mean relative variance x seconds) over the ROI."""
    use = roi & (result.dose != 0.0)
    if not use.any():
        return float("nan")
    relative = (result.dose_sigma[use] / np.abs(result.dose[use])) ** 2
    return float(1.0 / (relative.mean() * seconds))


def structure_gradient(result, grid: VoxelGrid, boundary: float) -> tuple[float, float]:
    """Gradient summed over a structure: strip 0, the depth band, ``x >= boundary``.

    This is the number an optimizer actually consumes: a leaf gradient enters
    ``dL/dx`` only contracted with ``dL/dD``, and a structure whose edge sits a
    distance from the tip weighs the gradient there and beyond.
    """
    x = (np.arange(grid.shape[0]) + 0.5) * grid.spacing[0]
    z = (np.arange(grid.shape[2]) + 0.5) * grid.spacing[2]
    band = (z >= DEPTH_CM[0]) & (z < DEPTH_CM[1])
    region = np.zeros(grid.shape, dtype=bool)
    region[np.ix_(x >= boundary, [0], band)] = True
    total = float(result.dose[region].sum())
    sigma = float(np.sqrt((result.dose_sigma[region] ** 2).sum()))
    return total, sigma


def structure_ratios(exact, chain, grid: VoxelGrid, tip: float) -> np.ndarray:
    """Rows ``(offset, chain / exact, sigma)`` of the structure gradient per offset."""
    rows = []
    for offset in OAR_OFFSETS:
        c, c_sigma = structure_gradient(exact, grid, tip + offset)
        d, d_sigma = structure_gradient(chain, grid, tip + offset)
        ratio = d / c
        rows.append((offset, ratio, abs(ratio) * np.hypot(c_sigma / c, d_sigma / d)))
    return np.array(rows)


def run_oracle(run, beam, stack, xs, parameter, n_histories: int) -> tuple[Any, float]:
    """Run the finite-difference oracle on the slice of field its weights live on.

    A ray whose transmission is the same at ``x - h`` and ``x + h`` carries weight
    zero, so restricting the uniform beam to the slice holding the edge penumbra
    and scaling the per-history result by the slice's share of the field area is
    an exact reweighting, not an approximation.
    """
    x = edge_value(stack, parameter)
    lo = max(beam.x_range[0], x + ORACLE_SLICE[0])
    hi = min(beam.x_range[1], x + ORACLE_SLICE[1])
    narrow = VectorizedParallelBeam(beam.energy, beam.z, (lo, hi), beam.y_range)
    source = RayFiniteDifferenceSource(narrow, stack, xs, parameter, step=FD_STEP)
    result, seconds = run(source, n_histories)
    share = (hi - lo) / (beam.x_range[1] - beam.x_range[0])
    scaled = replace(result, dose=result.dose * share, dose_sigma=result.dose_sigma * share)
    return scaled, seconds


def main() -> None:
    """Run every estimator for both edges, report agreement and render the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=float, default=1.0, help="multiply all history counts")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--backend", choices=("auto", "warp", "ref"), default="auto")
    parser.add_argument("--devices", choices=("auto", "tungsten", "surrogate"), default="auto")
    args = parser.parse_args()

    grid = make_grid()
    xs, material, density, device_label = make_cross_sections(args.devices, grid)
    engine, engine_name = make_engine(args.backend, grid, xs)
    run = Runner(engine, args.batches, args.seed)
    beam = VectorizedParallelBeam(ENERGY, -1.0, FIELD_X, FIELD_Y)
    stack = make_stack(material, density)
    print(f"engine {engine_name}; devices: {device_label}")

    def n(base: float) -> int:
        return int(base * args.scale)

    forward, _ = run(CollimatedSource(beam, stack, xs), n(4e6))
    results: dict[str, dict[str, tuple[Any, float]]] = {}
    for name, parameter in (("tip", MLC_TIP), ("jaw", JAW_POS)):
        print(f"-- d dose / d {name} --")
        runs = {
            "A oracle": run_oracle(run, beam, stack, xs, parameter, n(8e6)),
            "B pathwise": run(PathwiseDerivativeSource(beam, stack, xs, parameter), n(4e6)),
            "C edge-sampled": run(EdgeDerivativeSource(beam, stack, xs, parameter), n(1e6)),
            "D chain rule 5 mm": run(
                BixelChainRuleSource(beam, stack, xs, parameter, bixel_pitch=0.5), n(1e6)
            ),
            "D chain rule 2.5 mm": run(
                BixelChainRuleSource(beam, stack, xs, parameter, bixel_pitch=0.25), n(1e6)
            ),
        }
        runs["D chain rule 1 mm"] = run(
            BixelChainRuleSource(beam, stack, xs, parameter, bixel_pitch=0.1), n(1e6)
        )
        for pitch in (0.5, 0.25, 0.1):
            label = f"E profile rule {10 * pitch:g} mm"
            source = ProfileChainRuleSource(beam, stack, xs, parameter, bixel_pitch=pitch)
            runs[label] = run(source, n(1e6))
        results[name] = runs
        exact = runs["C edge-sampled"][0]
        roi = np.abs(exact.dose) > 0.1 * np.abs(exact.dose).max()
        oracle = runs["A oracle"][0]
        for label, (result, seconds) in runs.items():
            chi2, ndf = compare(result, oracle, roi) if label != "A oracle" else (1.0, 0)
            fom = efficiency(result, seconds, roi) if label[0] in "ABC" else float("nan")
            print(
                f"  {label:20s} {seconds:6.1f} s   chi2/ndf vs oracle {chi2:8.2f} (ndf {ndf})"
                f"   efficiency {fom:10.3g}"
            )

    print("-- structure-integrated gradient, chain rule / exact (structure edge - tip) --")
    exact = results["tip"]["C edge-sampled"][0]
    ratios = {
        label: structure_ratios(exact, results["tip"][label][0], grid, TIP)
        for label in results["tip"]
        if label[0] in "DE"
    }
    for label, rows in ratios.items():
        for offset, ratio, sigma in rows[::5]:
            print(f"  {label:20s} offset {offset:+.1f} cm: {ratio:6.3f} +- {sigma:.3f}")

    print("-- worst structure-gradient error, 5 mm chain rules, tip moving through [5.0, 5.5) --")
    for tip in TIP_SWEEP:
        moved = with_edge(stack, MLC_TIP, tip)
        exact, _ = run(EdgeDerivativeSource(beam, moved, xs, MLC_TIP), n(1e6))
        line = f"  tip {tip:.2f} cm:"
        for label, cls in (("D", BixelChainRuleSource), ("E", ProfileChainRuleSource)):
            chain, _ = run(cls(beam, moved, xs, MLC_TIP, bixel_pitch=0.5), n(1e6))
            rows = structure_ratios(exact, chain, grid, tip)
            worst = int(np.argmax(np.abs(rows[:, 1] - 1.0)))
            offset, ratio, sigma = rows[worst]
            line += f"   {label}: {ratio:5.3f} +- {sigma:.3f} at {offset:+.1f} cm"
        print(line)

    render(grid, forward, results, ratios)


def render(grid: VoxelGrid, forward, results, ratios) -> None:
    """Draw the four panels and save the figure beside this script."""
    import matplotlib.pyplot as plt

    x = (np.arange(grid.shape[0]) + 0.5) * grid.spacing[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    ax = axes[0, 0]
    for row, label in (
        (0, f"strip 0 (leaf tip at {TIP} cm)"),
        (1, f"strip 1 (jaw edge at {JAW_EDGE} cm)"),
    ):
        mean, _ = depth_profile(forward, row, grid)
        ax.plot(x, mean / mean.max(), label=label)
    ax.set(
        title="(a) forward dose, 1-4 cm deep",
        xlabel="x (cm)",
        ylabel="relative dose",
        xlim=(3.5, 9.8),
    )
    ax.legend(fontsize=8)

    styles = {
        "A oracle": dict(color="0.4", marker="o", ms=3, ls="none"),
        "B pathwise": dict(color="tab:orange", lw=1.2),
        "C edge-sampled": dict(color="tab:blue", lw=1.8),
        "D chain rule 5 mm": dict(color="tab:red", lw=1.2, ls="--"),
        "D chain rule 2.5 mm": dict(color="tab:purple", lw=1.2, ls=":"),
        "D chain rule 1 mm": dict(color="tab:pink", lw=1.2, ls="-."),
        "E profile rule 5 mm": dict(color="tab:green", lw=1.2, ls="--"),
        "E profile rule 2.5 mm": dict(color="tab:olive", lw=1.2, ls=":"),
        "E profile rule 1 mm": dict(color="tab:cyan", lw=1.2, ls="-."),
    }
    profiles = (
        "A oracle",
        "B pathwise",
        "C edge-sampled",
        "D chain rule 5 mm",
        "E profile rule 5 mm",
    )
    for ax, name, row, window, title in (
        (axes[0, 1], "tip", 0, (4.0, 7.5), "(b) d dose / d tip  (rounded leaf end)"),
        (axes[1, 0], "jaw", 1, (7.0, 9.6), "(c) d dose / d jaw edge  (straight face)"),
    ):
        scale = np.abs(depth_profile(results[name]["C edge-sampled"][0], row, grid)[0]).max()
        for label in profiles:
            result = results[name][label][0]
            mean, sigma = depth_profile(result, row, grid)
            if label == "A oracle":
                ax.errorbar(x, mean / scale, yerr=sigma / scale, label=label, **styles[label])
            else:
                ax.plot(x, mean / scale, label=label, **styles[label])
        ax.axhline(0.0, color="0.8", lw=0.8)
        ax.set(title=title, xlabel="x (cm)", ylabel="gradient (relative)", xlim=window)
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    for label, rows in ratios.items():
        style = styles[label]
        ax.plot(rows[:, 0], rows[:, 1], color=style["color"], ls=style["ls"], label=label)
        ax.fill_between(
            rows[:, 0],
            rows[:, 1] - rows[:, 2],
            rows[:, 1] + rows[:, 2],
            color=style["color"],
            alpha=0.2,
        )
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set(
        title="(d) structure dose gradient, chain rules / exact",
        xlabel="structure edge minus leaf tip (cm)",
        ylabel="ratio",
        ylim=(0.0, None),
    )
    ax.legend(fontsize=8)

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
