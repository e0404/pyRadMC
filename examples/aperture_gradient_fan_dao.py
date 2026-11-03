"""Monte Carlo direct aperture optimization on a divergent 6 MV fan, fully on the GPU.

Each iteration computes a Monte Carlo forward dose for every segment and the
objective's gradient with respect to every leaf tip, both on the device, then takes
one projected Adam step on all tips and the segment MU weights:

- **forward** — all segments in one
  :meth:`~pyradmc.backends.warp.engine.WarpEngine.run_dij` over a
  :class:`~pyradmc.geometry.segments.SegmentedFanSource`: each primary carries its
  segment's narrow-beam MLC transmission (the quantity the gradient differentiates),
  evaluated in-kernel, and the correlated columns share one ray sample; the MU
  weights' gradient is the adjoint's dot product with those columns;
- **gradient** — every leaf edge of every segment is a beamlet of one
  :meth:`~pyradmc.backends.warp.engine.WarpEngine.run_beamlet_contraction` over a
  :class:`~pyradmc.geometry.segment_edges.SegmentedEdgeSource` (the segments' tips
  as arrays, no Python object per edge): rays generated in-kernel from device-built
  proposal tables, each weighted by its exact ``dT/dx``, and contracted with
  ``dL/dD`` on the device. One number per edge, no Dij.

Setup: water 12 x 10 x 12 cm on 2.5 x 2.5 x 5 mm voxels; Varian 6 MV Ali-Rogers
spectrum from a point focal spot 100 cm above the surface; a tungsten MLC (6 cm
leaves, 8 cm tip radius, focused sides, 5 mm leaves at the surface) with 10
step-and-shoot segments, i.e. 400 leaf edges; a C-shaped PTV around a cylindrical
OAR, 3-9 cm deep; electrons transported. The quadratic objective pulls the PTV to the
prescription and penalizes OAR and normal-tissue dose above 40 and 60 percent of it.

Renders ``aperture_gradient_fan_dao.png`` beside this script:

- row 1: the objective per iteration (the noisy in-loop loss, and the initial and final
  plans re-scored at high statistics), wall time per iteration, and the DVHs;
- row 2: the beam's-eye view at three iterations: the MU-weighted MLC transmission of
  all segments summed (the fluence) with the PTV (red) and OAR (white) outlines and the
  leaves of one segment (grey), all projected to the phantom surface;
- row 3: the initial and final plan dose on a slice across the beam at 6 cm depth, and
  the final dose along the beam through the OAR.

and ``aperture_gradient_fan_dao.gif``: the fluence and that iteration's (noisy)
dose at 6 cm, per iteration.
Needs a CUDA device and the EPICS libraries (cached or downloadable)::

    python examples/aperture_gradient_fan_dao.py

About 60 ms per iteration on an RTX 4070 laptop GPU (forward ~40 ms, gradient
~20 ms); at the default 500 iterations, with table compilation, the two re-scored
plan doses and rendering, about a minute and a half.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pyradmc.data.materials import TUNGSTEN
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segment_edges import SegmentedEdgeSource
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv

SAD = 100.0  # focal spot to phantom surface (cm)
CENTRE = (6.0, 5.0)  # beam axis and OAR axis at the surface (engine x, y)
FIELD = 10.0  # cm square at the surface
N_PAIRS = 20
N_SEGMENTS = 10
MLC_Z = (50.0, 56.0)
TO_MID = 0.5 * sum(MLC_Z) / SAD  # surface -> MLC mid-plane scaling
SEGMENT_SHOWN = 3


def make_grid() -> VoxelGrid:
    """Water, 12 x 10 x 12 cm, 2.5 x 2.5 x 5 mm voxels; surface at z = 0."""
    return VoxelGrid.uniform_water(shape=(48, 40, 24), spacing=(0.25, 0.25, 0.5))


def make_structures(grid: VoxelGrid) -> dict[str, np.ndarray]:
    """C-shaped PTV around a cylindrical OAR, 3-9 cm deep (coordinates at the voxel)."""
    axes = [(np.arange(n) + 0.5) * d for n, d in zip(grid.shape, grid.spacing, strict=True)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    # Structures are defined in the beam's-eye view, projected along the fan.
    magnification = SAD / (SAD + zz)
    u = (xx - CENTRE[0]) * magnification
    v = (yy - CENTRE[1]) * magnification
    r = np.hypot(u, v)
    theta = np.arctan2(v, u)
    depth = (zz >= 3.0) & (zz < 9.0)
    oar = (r < 1.5) & depth
    ptv = (r >= 2.0) & (r < 3.5) & (np.abs(theta) > np.pi / 4) & depth
    return {"ptv": ptv, "oar": oar, "normal": ~(ptv | oar)}


def make_beam(spectrum) -> SpectralBeamSource:
    """Build the 6 MV fan through the 10 x 10 cm field at the surface.

    The exact type, so the engine generates it in-kernel inside the segmented source.
    """
    return SpectralBeamSource(
        spectrum,
        focal_point=(CENTRE[0], CENTRE[1], -SAD),
        center=(CENTRE[0], CENTRE[1], 0.0),
        width_u=FIELD,
        width_v=FIELD,
    )


LEAF_EDGES = tuple(float(e) for e in (np.linspace(-0.5, 0.5, N_PAIRS + 1) * FIELD * TO_MID))


def initial_tips(rng: np.random.Generator) -> np.ndarray:
    """Random sub-openings of each row's PTV chord at the mid-plane; others closed."""
    tips = np.empty((N_SEGMENTS, N_PAIRS, 2))
    for s in range(N_SEGMENTS):
        for i in range(N_PAIRS):
            y_surface = 0.5 * (LEAF_EDGES[i] + LEAF_EDGES[i + 1]) / TO_MID
            half = np.sqrt(max(3.5**2 - y_surface**2, 0.0))
            if half > 0.3:
                a, b = np.sort(rng.uniform(-half, half, 2))
                b = max(b, a + 0.3)
            else:
                a = b = rng.uniform(-0.5, 0.5)
            tips[s, i] = (a * TO_MID, b * TO_MID)
    return tips


def segment_stack(tips: np.ndarray) -> BeamLimitingStack:
    """One segment's tungsten MLC in the fan's frame (local = surface-centred)."""
    return BeamLimitingStack(
        frame=BeamFrame(origin=(CENTRE[0], CENTRE[1], -SAD)),
        devices=(
            MLC(
                z_top=MLC_Z[0],
                z_bottom=MLC_Z[1],
                leaf_edges_v=LEAF_EDGES,
                tips_neg=tuple(float(t) for t in tips[:, 0]),
                tips_pos=tuple(float(t) for t in tips[:, 1]),
                tip_radius=8.0,
                material=TUNGSTEN,
                density=19.3,
                focused_sides=True,
            ),
        ),
    )


def project(tips: np.ndarray) -> np.ndarray:
    """Keep tips inside the field and each pair uncrossed (mid-plane coordinates)."""
    limit = 0.5 * FIELD * TO_MID
    tips = np.clip(tips, -limit, limit)
    crossed = tips[..., 0] > tips[..., 1]
    middle = 0.5 * (tips[..., 0] + tips[..., 1])
    tips[..., 0] = np.where(crossed, middle, tips[..., 0])
    tips[..., 1] = np.where(crossed, middle, tips[..., 1])
    return tips


class FanPlan:
    """Forward doses, objective and edge gradients for one fan, all on the device."""

    def __init__(self, engine, xs, beam, structures) -> None:
        self.engine = engine
        self.xs = xs
        self.beam = beam
        self.structures = structures
        self.prescription = 1.0
        self.seed = 1
        self.segments: SegmentedFanSource | None = None

    def _segments(self, tips: np.ndarray) -> SegmentedFanSource:
        """Return the segmented fan at these tips: one template, per-segment tip arrays.

        Built once; every later call re-targets it through ``with_settings``, so
        only the tips are new (no MLC object per segment).
        """
        if self.segments is None:
            self.segments = SegmentedFanSource(
                self.beam, segment_stack(tips[0]), self.xs,
                tips_neg=tips[..., 0], tips_pos=tips[..., 1],
            )  # fmt: skip
        self.segments = self.segments.with_settings(tips_neg=tips[..., 0], tips_pos=tips[..., 1])
        return self.segments

    def _seed(self) -> int:
        self.seed += 1
        return self.seed

    def segment_doses(self, tips: np.ndarray, n_per_segment: int) -> np.ndarray:
        """Per-segment dose per fan history (MeV/g): one segmented Dij, (S, *grid).

        Correlated sampling (the default) replays one ray sample through every
        segment, so the columns' differences are low-noise; no truncation, since
        every column enters the plan dose.
        """
        dij = self.engine.run_dij(
            self._segments(tips),
            n_histories_per_beamlet=n_per_segment,
            n_batches=2,
            seed=self._seed(),
            truncation=0.0,
        )
        return np.stack([dij.column_dense(s) for s in range(N_SEGMENTS)])

    def objective(self, dose: np.ndarray) -> tuple[float, np.ndarray]:
        """Loss and dL/dD: PTV to prescription, one-sided OAR and normal-tissue limits."""
        ptv, oar, normal = (self.structures[k] for k in ("ptv", "oar", "normal"))
        rx = self.prescription
        grad = np.zeros_like(dose)
        r_ptv = dose[ptv] - rx
        r_oar = np.maximum(0.0, dose[oar] - 0.4 * rx)
        r_nt = np.maximum(0.0, dose[normal] - 0.6 * rx)
        grad[ptv] = 2.0 * r_ptv / ptv.sum()
        grad[oar] = 2.0 * 2.0 * r_oar / oar.sum()
        grad[normal] = 2.0 * 0.5 * r_nt / normal.sum()
        loss = (r_ptv**2).mean() + 2.0 * (r_oar**2).mean() + 0.5 * (r_nt**2).mean()
        return float(loss) / rx**2, grad

    def tip_gradient(
        self, tips: np.ndarray, weights: np.ndarray, grad_dose: np.ndarray, n_per_edge: int
    ) -> np.ndarray:
        """dL/d(tip) for all 400 edges from one in-kernel contraction, (S, pairs, 2).

        The edges are segment-major, then leaf pair, then side (neg, pos): the
        ``(S, pairs, 2)`` layout of ``tips``, built from the same segmented source as
        the forward dose.
        """
        result = self.engine.run_beamlet_contraction(
            SegmentedEdgeSource(self._segments(tips)),
            grad_dose,
            n_histories_per_beamlet=n_per_edge,
            n_batches=2,
            seed=self._seed(),
        )
        g = result.values * np.repeat(weights, 2 * N_PAIRS)
        return g.reshape(N_SEGMENTS, N_PAIRS, 2)


def optimize(plan: FanPlan, tips, weights, iterations: int, n_fwd: int, n_edge: int, lr: float):
    """Projected Adam on tips (mid-plane cm) and MU weights, step decaying as 1/sqrt(k)."""
    m_t, v_t = np.zeros_like(tips), np.zeros_like(tips)
    m_w, v_w = np.zeros_like(weights), np.zeros_like(weights)
    beta1, beta2, eps = 0.9, 0.999, 1e-12
    history, timing = [], []
    for k in range(1, iterations + 1):
        t0 = time.perf_counter()
        doses = plan.segment_doses(tips, n_fwd)
        t1 = time.perf_counter()
        dose = np.tensordot(weights, doses, axes=1)
        loss, grad_dose = plan.objective(dose)
        history.append((tips.copy(), weights.copy(), loss, dose))
        g_tips = plan.tip_gradient(tips, weights, grad_dose, n_edge)
        g_w = doses.reshape(N_SEGMENTS, -1) @ grad_dose.ravel()
        t2 = time.perf_counter()
        for m, v, g in ((m_t, v_t, g_tips), (m_w, v_w, g_w)):
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * g * g
        decay = 1.0 / np.sqrt(k)
        step_t = lr * decay * (m_t / (1 - beta1**k)) / (np.sqrt(v_t / (1 - beta2**k)) + eps)
        step_w = 0.05 * decay * (m_w / (1 - beta1**k)) / (np.sqrt(v_w / (1 - beta2**k)) + eps)
        tips = project(tips - step_t)
        weights = np.maximum(weights * (1.0 - step_w), 0.0)
        timing.append((t1 - t0, t2 - t1))
        print(
            f"  iteration {k:3d}: loss {loss:.4e}   forward {t1 - t0:5.2f} s"
            f"   gradient {t2 - t1:5.2f} s",
            flush=True,
        )
    doses = plan.segment_doses(tips, n_fwd)
    dose = np.tensordot(weights, doses, axes=1)
    history.append((tips.copy(), weights.copy(), plan.objective(dose)[0], dose))
    return history, np.array(timing)


def bev_fluence(tips: np.ndarray, weights: np.ndarray, xs, us: np.ndarray, vs: np.ndarray):
    """MU-weighted narrow-beam transmission at the surface, summed over segments."""
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    directions = np.stack([uu.ravel(), vv.ravel(), np.full(uu.size, SAD)], axis=1)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.tile([CENTRE[0], CENTRE[1], -SAD], (uu.size, 1))
    mu = xs.mu_over_rho_total(2.0, TUNGSTEN) * 19.3  # display only: mid-spectrum
    total = np.zeros(uu.size)
    for s in range(N_SEGMENTS):
        lengths = segment_stack(tips[s]).path_lengths(origins, directions)[:, 0]
        total += weights[s] * np.exp(-mu * lengths)
    return total.reshape(uu.shape)


def draw_bev(ax, structures, tips, weights, xs, title: str, pixels=(200, 160)) -> None:
    """Fluence, structure outlines and one segment's leaves at the surface.

    ``pixels`` sets the ray grid of the fluence (host ray tracing, the slow part of
    rendering): the summary figure uses the default, the GIF a coarser grid.
    """
    us = np.linspace(-0.5 * FIELD, 0.5 * FIELD, pixels[0])
    vs = np.linspace(-0.5 * FIELD + 0.01, 0.5 * FIELD - 0.01, pixels[1])
    extent = (-0.5 * FIELD, 0.5 * FIELD, -0.5 * FIELD, 0.5 * FIELD)
    ax.imshow(
        bev_fluence(tips, weights, xs, us, vs).T, origin="lower", extent=extent, cmap="viridis"
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(1.5 * np.cos(theta), 1.5 * np.sin(theta), color="white", lw=1.2)
    for radius in (2.0, 3.5):
        arc = theta[np.abs(np.arctan2(np.sin(theta), np.cos(theta))) > np.pi / 4]
        ax.plot(radius * np.cos(arc), radius * np.sin(arc), color="tab:red", lw=1.2)
    for i in range(N_PAIRS):
        y0, y1 = LEAF_EDGES[i] / TO_MID, LEAF_EDGES[i + 1] / TO_MID
        a, b = tips[SEGMENT_SHOWN, i] / TO_MID
        ax.fill_between([-0.5 * FIELD, a], y0, y1, color="0.75", alpha=0.55, lw=0)
        ax.fill_between([b, 0.5 * FIELD], y0, y1, color="0.75", alpha=0.55, lw=0)
    ax.set(title=title, xlabel="u at the surface (cm)", ylabel="v at the surface (cm)")


def dvh(dose: np.ndarray, mask: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Cumulative dose-volume histogram of ``mask`` at ``levels``."""
    values = dose[mask]
    return np.array([(values >= level).mean() for level in levels])


DOSE_DEPTH = 6.0  # cm: the transverse dose slice, mid-target
DOSE_MAX = 1.3  # colour scale top, in units of the prescription
GIF_FRAMES = 120  # at most this many frames: long runs are subsampled


def draw_dose(ax, grid: VoxelGrid, dose: np.ndarray, structures, plane: str, title: str):
    """One dose slice (in units of the prescription) with PTV and OAR contours.

    ``plane="transverse"`` cuts perpendicular to the beam at ``DOSE_DEPTH`` (the plane
    the beam's-eye view looks at, magnified by the fan by about 6 % at that depth);
    ``plane="depth"`` cuts along the beam axis through the OAR (u horizontal, depth
    down). Coordinates are relative to the beam axis.
    """
    (nx, ny, nz), (dx, dy, dz) = grid.shape, grid.spacing
    x = (np.arange(nx) + 0.5) * dx - CENTRE[0]
    y = (np.arange(ny) + 0.5) * dy - CENTRE[1]
    z = (np.arange(nz) + 0.5) * dz
    if plane == "transverse":
        k = int(DOSE_DEPTH / dz)
        image = dose[:, :, k].T
        masks = {name: structures[name][:, :, k].T for name in ("ptv", "oar")}
        extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)
        coords, origin = (x, y), "lower"
        labels = ("x from the beam axis (cm)", "y from the beam axis (cm)")
    else:
        j = int(CENTRE[1] / dy)
        image = dose[:, j, :].T
        masks = {name: structures[name][:, j, :].T for name in ("ptv", "oar")}
        extent = (x[0] - dx / 2, x[-1] + dx / 2, z[-1] + dz / 2, z[0] - dz / 2)
        coords, origin = (x, z), "upper"
        labels = ("x from the beam axis (cm)", "depth (cm)")
    handle = ax.imshow(
        image, origin=origin, extent=extent, cmap="magma", vmin=0.0, vmax=DOSE_MAX,
        aspect="equal", interpolation="nearest",
    )  # fmt: skip
    for name, color in (("ptv", "tab:red"), ("oar", "white")):
        if masks[name].any():
            ax.contour(
                *coords, masks[name].astype(float), levels=[0.5], colors=color, linewidths=1.2
            )
    ax.set(title=title, xlabel=labels[0], ylabel=labels[1])
    return handle


def render(plan, grid, structures, history, timing, xs, clean, out: Path) -> None:
    """Summary figure and per-iteration GIF.

    ``clean`` holds the initial and final plan doses re-scored at high statistics
    (``(initial, final)``, absolute); the dose panels, the DVHs and the true-loss
    markers use them, while the loss curve and the GIF show what the optimizer saw
    (each iteration's own noisy forward dose).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    rx = plan.prescription
    initial, final = (d / rx for d in clean)
    last = len(history) - 1
    fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(range(len(history)), [h[2] for h in history], "-", label="in-loop (noisy forward)")
    ax.plot(
        [0, last], [plan.objective(clean[0])[0], plan.objective(clean[1])[0]], "o",
        color="tab:red", ms=8, label="re-scored at high statistics",
    )  # fmt: skip
    ax.set(title="(a) objective", xlabel="iteration", ylabel="loss / Rx^2", yscale="log")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    it = np.arange(1, len(timing) + 1)
    ax.bar(it, timing[:, 0], width=1.0, label="forward (10 segments)")
    ax.bar(it, timing[:, 1], width=1.0, bottom=timing[:, 0], label="gradient (400 edges)")
    ax.set(title="(b) wall time per iteration", xlabel="iteration", ylabel="seconds")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    levels = np.linspace(0.0, 1.4, 141)
    for dose, style, when in ((initial, "--", "initial"), (final, "-", "final")):
        for name, color in (("ptv", "tab:red"), ("oar", "tab:blue")):
            ax.plot(
                levels, dvh(dose, structures[name], levels), style, color=color,
                label=f"{name.upper()} {when}",
            )  # fmt: skip
    ax.set(
        title="(c) DVH, initial (dashed) and final", xlabel="dose / Rx", ylabel="volume fraction"
    )
    ax.legend(fontsize=8)

    picks = (0, len(history) // 2, last)
    for ax, k, label in zip(axes[1], picks, "def", strict=True):
        draw_bev(
            ax, structures, history[k][0], history[k][1], xs,
            f"({label}) fluence, iteration {k} (leaves: segment {SEGMENT_SHOWN})",
        )  # fmt: skip

    draw_dose(axes[2, 0], grid, initial, structures, "transverse",
              f"(g) initial dose at {DOSE_DEPTH:g} cm depth")  # fmt: skip
    draw_dose(axes[2, 1], grid, final, structures, "transverse",
              f"(h) final dose at {DOSE_DEPTH:g} cm depth")  # fmt: skip
    handle = draw_dose(axes[2, 2], grid, final, structures, "depth",
                       "(i) final dose along the beam, through the OAR")  # fmt: skip
    fig.colorbar(handle, ax=axes[2, :], label="dose / Rx", shrink=0.8)
    fig.savefig(out.with_suffix(".png"), dpi=110)
    print(f"saved {out.with_suffix('.png')}")

    anim_fig, (bev_ax, dose_ax) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    frames = sorted(set(range(0, len(history), max(1, len(history) // GIF_FRAMES))) | {last})
    colorbar = []

    def frame(k: int) -> None:
        bev_ax.clear()
        dose_ax.clear()
        tips, weights, loss, dose = history[k]
        draw_bev(bev_ax, structures, tips, weights, xs, f"fluence, iteration {k}", (100, 80))
        handle = draw_dose(dose_ax, grid, dose / rx, structures, "transverse",
                           f"dose at {DOSE_DEPTH:g} cm, in-loop loss {loss:.3f}")  # fmt: skip
        if not colorbar:
            colorbar.append(anim_fig.colorbar(handle, ax=dose_ax, label="dose / Rx"))

    FuncAnimation(anim_fig, frame, frames=frames).save(
        out.with_suffix(".gif"), writer=PillowWriter(fps=6)
    )
    print(f"saved {out.with_suffix('.gif')} ({len(frames)} frames)")


def main() -> None:
    """Check the forward normalization, optimize, and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--forward", type=int, default=400_000, help="histories per segment")
    parser.add_argument("--per-edge", type=int, default=2_000, help="gradient histories per edge")
    parser.add_argument("--lr", type=float, default=0.03, help="Adam step on tips (mid-plane cm)")
    parser.add_argument(
        "--evaluate", type=int, default=10_000_000,
        help="histories per segment for the initial and final plan doses shown",
    )  # fmt: skip
    args = parser.parse_args()

    from pyradmc.backends.warp.engine import WarpEngine
    from pyradmc.data.tabulated.build import download_library
    from pyradmc.data.tabulated.precompile import compile_materials
    from pyradmc.data.tabulated.source import TabulatedCrossSections

    grid = make_grid()
    structures = make_structures(grid)
    print("compiling the tabulated table (EPICS cache/download) ...")
    data = compile_materials(
        download_library("epdl").read_text(encoding="latin-1"),
        download_library("eedl").read_text(encoding="latin-1"),
        e_max=7.0,
    )
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    engine = WarpEngine(grid=grid, cross_sections=xs, device="cuda:0")
    beam = make_beam(ali_rogers_mv("varian-6mv"))
    plan = FanPlan(engine, xs, beam, structures)
    rng = np.random.default_rng(7)
    tips = project(initial_tips(rng))
    weights = rng.uniform(0.5, 1.5, N_SEGMENTS)

    initial = np.tensordot(weights, plan.segment_doses(tips, args.evaluate), axes=1)
    plan.prescription = 1.1 * float(initial[structures["ptv"]].mean())
    print(f"-- Monte Carlo direct aperture optimization: {2 * N_PAIRS * N_SEGMENTS} leaf edges --")
    history, timing = optimize(
        plan, tips, weights, args.iterations, args.forward, args.per_edge, args.lr
    )
    print(
        f"median per iteration: forward {np.median(timing[:, 0]):.2f} s, "
        f"gradient {np.median(timing[:, 1]):.2f} s"
    )
    # The in-loop loss carries each iteration's forward noise (its quadratic terms add
    # the voxel variance); re-score the initial and final plans at high statistics.
    final_tips, final_weights = history[-1][0], history[-1][1]
    final = np.tensordot(final_weights, plan.segment_doses(final_tips, args.evaluate), axes=1)
    print(
        f"loss re-scored at {args.evaluate:,} histories per segment: "
        f"{plan.objective(initial)[0]:.4f} -> {plan.objective(final)[0]:.4f} "
        f"(in-loop final {history[-1][2]:.4f})"
    )
    render(
        plan, grid, structures, history, timing, xs, (initial, final),
        Path(__file__).with_suffix(""),
    )  # fmt: skip


if __name__ == "__main__":
    main()
