"""Monte Carlo leaf gradients inside an optimizer loop: precision, cost, and a short DAO.

A **study**, not a quick demo (about four minutes on a CUDA laptop GPU): it measures
whether Monte Carlo gradients of a plan objective with respect to every leaf tip are
cheap and precise enough to drive a direct aperture optimization, and then runs one.

The optimizer needs ``g_e = sum_v dL/dD_v dD_v/dx_e``, one number per leaf edge, not the
Jacobian. Every edge of every segment becomes a beamlet of one ``run_dij``
(:class:`~pyradmc.geometry.aperture_gradient.EdgeDerivativeSource` each; the sign of a
closing edge is carried outside, since Dij truncation is relative to a positive column
maximum), and the columns are contracted with ``dL/dD`` afterwards.

Setup (the parallel-beam estimator's limits: parallel beam, 2 MeV, unfocused devices):
water 12 x 10 x 12 cm on 2.5 x 2.5 x 5 mm voxels; a 20-pair tungsten MLC with 5 mm
leaves and 8 cm tip radius; 10 step-and-shoot segments with MU weights, i.e. 400 leaf edges; a
C-shaped PTV around a cylindrical OAR, 3-9 cm deep; a quadratic objective with a
one-sided OAR and normal-tissue penalty.

1. **Precision** — the gradient at the initial iterate from N histories per edge,
   against a high-statistics reference: relative error of the gradient vector and its
   cosine to the reference.
2. **Optimization** — projected Adam on all 400 tip positions and the 10 MU weights,
   each iteration with a fresh forward dose and a fresh gradient run.

Renders ``aperture_gradient_feasibility.png`` (precision, loss, the beam's-eye-view
fluence and one segment's leaves at three iterations, DVHs) and
``aperture_gradient_feasibility.gif`` (the leaf setup per iteration) beside this
script. Requires the ``examples`` extra and EPICS data (cached or downloadable)::

    python examples/aperture_gradient_feasibility.py

The forward dose uses :class:`StripCollimatedBeam`, exact under a parallel beam with
parallel-sided leaves: a vertical ray in strip i crosses only pair i.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from aperture_gradient_demo import (
    VectorizedParallelBeam,
    make_cross_sections,
    make_engine,
)

from pyradmc.geometry.aperture_gradient import EdgeDerivativeSource, EdgeParameter
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletSource, Primary

ENERGY = 2.0  # MeV
FIELD_X = (1.0, 11.0)
FIELD_Y = (0.0, 10.0)
N_PAIRS = 20
N_SEGMENTS = 10
LEAF_EDGES = tuple(float(v) for v in np.linspace(0.0, 10.0, N_PAIRS + 1))
CENTRE = (6.0, 5.0)  # OAR axis in the beam's-eye view (cm)
SEGMENT_SHOWN = 3


# --- geometry -----------------------------------------------------------------------


def make_grid() -> VoxelGrid:
    """Water, 12 x 10 x 12 cm, 2.5 x 2.5 x 5 mm voxels."""
    return VoxelGrid.uniform_water(shape=(48, 40, 24), spacing=(0.25, 0.25, 0.5))


def make_structures(grid: VoxelGrid) -> dict[str, np.ndarray]:
    """C-shaped PTV around a cylindrical OAR, 3-9 cm deep; the rest is normal tissue."""
    x = (np.arange(grid.shape[0]) + 0.5) * grid.spacing[0]
    y = (np.arange(grid.shape[1]) + 0.5) * grid.spacing[1]
    z = (np.arange(grid.shape[2]) + 0.5) * grid.spacing[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r = np.hypot(xx - CENTRE[0], yy - CENTRE[1])
    theta = np.arctan2(yy - CENTRE[1], xx - CENTRE[0])
    depth = (zz >= 3.0) & (zz < 9.0)
    oar = (r < 1.5) & depth
    ptv = (r >= 2.0) & (r < 3.5) & (np.abs(theta) > np.pi / 4) & depth
    return {"ptv": ptv, "oar": oar, "normal": ~(ptv | oar)}


def initial_tips(rng: np.random.Generator) -> np.ndarray:
    """Random sub-openings of each row's PTV-disc chord; rows outside it closed.

    Shape ``(segments, pairs, 2)``: ``[..., 0]`` the -x tip, ``[..., 1]`` the +x tip.
    """
    tips = np.empty((N_SEGMENTS, N_PAIRS, 2))
    for s in range(N_SEGMENTS):
        for i in range(N_PAIRS):
            yc = 0.5 * (LEAF_EDGES[i] + LEAF_EDGES[i + 1])
            dy = abs(yc - CENTRE[1])
            if dy < 3.5:
                half = np.sqrt(3.5**2 - dy**2)
                a, b = np.sort(rng.uniform(CENTRE[0] - half, CENTRE[0] + half, 2))
                b = max(b, a + 0.3)
            else:
                a = b = CENTRE[0] + rng.uniform(-0.5, 0.5)
            tips[s, i] = (a, b)
    return tips


def segment_stack(tips: np.ndarray, material: int, density: float) -> BeamLimitingStack:
    """One segment's MLC (6 cm tungsten leaves, 8 cm tip radius) in a parallel frame."""
    return BeamLimitingStack(
        frame=BeamFrame(origin=(0.0, 0.0, -100.0)),
        devices=(
            MLC(
                z_top=50.0,
                z_bottom=56.0,
                leaf_edges_v=LEAF_EDGES,
                tips_neg=tuple(float(t) for t in tips[:, 0]),
                tips_pos=tuple(float(t) for t in tips[:, 1]),
                tip_radius=8.0,
                material=material,
                density=density,
            ),
        ),
    )


def strip_transmission(
    stack: BeamLimitingStack, mu: float, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Narrow-beam transmission of vertical rays at local (x, y), pair by pair.

    A vertical ray in strip i crosses only pair i of a parallel-sided MLC, so each
    ray is traced through its own one-pair MLC: exact, and 20x cheaper than tracing
    every ray through every pair.
    """
    mlc = stack.devices[0]
    assert isinstance(mlc, MLC)
    out = np.ones_like(x)
    strip = np.clip(np.searchsorted(mlc.leaf_edges_v, y, side="right") - 1, 0, mlc.n_pairs - 1)
    for i in np.unique(strip):
        sel = strip == i
        pair = MLC(
            z_top=mlc.z_top,
            z_bottom=mlc.z_bottom,
            leaf_edges_v=(mlc.leaf_edges_v[i], mlc.leaf_edges_v[i + 1]),
            tips_neg=(mlc.tips_neg[i],),
            tips_pos=(mlc.tips_pos[i],),
            tip_radius=mlc.tip_radius,
            material=mlc.material,
            density=mlc.density,
        )
        n = int(sel.sum())
        points = np.column_stack([x[sel], y[sel], np.zeros(n)])
        lengths = pair.path_lengths(points, np.tile([0.0, 0.0, 1.0], (n, 1)))
        out[sel] = np.exp(-mu * mlc.density * lengths)
    return out


class StripCollimatedBeam(VectorizedParallelBeam):
    """The parallel beam seen through one segment's MLC, weighted strip by strip."""

    def __init__(self, stack: BeamLimitingStack, mu_over_rho: float) -> None:
        super().__init__(ENERGY, -1.0, FIELD_X, FIELD_Y)
        object.__setattr__(self, "_stack", stack)
        object.__setattr__(self, "_mu", mu_over_rho)

    def emit(self, rng_state: Any) -> Primary:
        """Emit one ray of the open beam with its transmission weight."""
        p = super().emit(rng_state)
        local = np.array([p.x]) - self._stack.frame.origin[0], np.array([p.y])
        return p._replace(weight=float(strip_transmission(self._stack, self._mu, *local)[0]))

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, Any]:
        """Sample the vectorized open beam and reweight it pair by pair."""
        columns = super().sample_batch(seed, history_offset, n)
        x = columns["x"].astype(np.float64) - self._stack.frame.origin[0]
        y = columns["y"].astype(np.float64) - self._stack.frame.origin[1]
        columns["weight"] = strip_transmission(self._stack, self._mu, x, y).astype(np.float32)
        return columns


class EdgeBeamlets(BeamletSource):
    """Every leaf edge as one beamlet: column e is |dD/dx_e| (signs kept in ``signs``)."""

    def __init__(self, sources: list[EdgeDerivativeSource]) -> None:
        self._sources = sources
        self.signs = np.array([np.sign(s.weight_scale) for s in sources])

    @property
    def max_energy(self) -> float:
        """The beam energy."""
        return ENERGY

    @property
    def n_beamlets(self) -> int:
        """One beamlet per leaf edge."""
        return len(self._sources)

    def emit(self, beamlet: int, rng_state: Any) -> Primary:
        """Emit one edge ray with the magnitude of its derivative weight."""
        p = self._sources[beamlet].emit(rng_state)
        return p._replace(weight=abs(p.weight))

    def sample_beamlet_batch(
        self, seed: int, history_offset: int, n: int, beamlet: int
    ) -> dict[str, Any]:
        """Return the edge source's vectorized batch with positive weights."""
        columns = self._sources[beamlet].sample_batch(seed, history_offset, n)
        columns["weight"] = np.abs(columns["weight"])
        return columns


# --- the plan problem --------------------------------------------------------------


class Plan:
    """Forward dose, objective and Monte Carlo gradients for leaf tips and MU weights."""

    def __init__(self, engine, xs, material: int, density: float, structures) -> None:
        self.engine = engine
        self.xs = xs
        self.material = material
        self.density = density
        self.structures = structures
        self.mu = xs.mu_over_rho_total(ENERGY, material)
        self.beam = VectorizedParallelBeam(ENERGY, -1.0, FIELD_X, FIELD_Y)
        self.seed = 1
        self.prescription = 1.0

    def _seed(self) -> int:
        self.seed += 1
        return self.seed

    def segment_doses(self, tips: np.ndarray, n_per_segment: int) -> np.ndarray:
        """Per-segment dose (MeV/g per history of the open field), shape (S, *grid)."""
        doses = []
        for s in range(N_SEGMENTS):
            beam = StripCollimatedBeam(segment_stack(tips[s], self.material, self.density), self.mu)
            result = self.engine.run(
                beam, n_histories=n_per_segment, n_batches=2, seed=self._seed()
            )
            doses.append(result.dose)
        return np.array(doses)

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

    def edge_sources(self, tips: np.ndarray) -> list[EdgeDerivativeSource]:
        """One exact edge-gradient source per leaf tip, segment-major, (-x, +x) per pair."""
        sources = []
        for s in range(N_SEGMENTS):
            stack = segment_stack(tips[s], self.material, self.density)
            for i in range(N_PAIRS):
                for side in ("neg", "pos"):
                    parameter = EdgeParameter(device=0, side=side, pair=i)
                    sources.append(
                        EdgeDerivativeSource(self.beam, stack, self.xs, parameter, n_table=1025)
                    )
        return sources

    def tip_gradient(
        self, tips: np.ndarray, weights: np.ndarray, grad_dose: np.ndarray, n_per_edge: int
    ) -> np.ndarray:
        """dL/d(tip) for every edge from one Dij run, shape (S, pairs, 2)."""
        beamlets = EdgeBeamlets(self.edge_sources(tips))
        dij = self.engine.run_dij(
            beamlets,
            n_histories_per_beamlet=n_per_edge,
            n_batches=1,
            seed=self._seed(),
            truncation=0.0,
        )
        a = grad_dose.reshape(-1)
        g = np.array(
            [
                a[dij.indices[dij.indptr[e] : dij.indptr[e + 1]]]
                @ dij.dose[dij.indptr[e] : dij.indptr[e + 1]]
                for e in range(dij.n_beamlets)
            ]
        )
        g = g * beamlets.signs * np.repeat(weights, 2 * N_PAIRS)
        return g.reshape(N_SEGMENTS, N_PAIRS, 2)


# --- part 1: precision --------------------------------------------------------------


def precision_study(plan: Plan, tips, weights, grad_dose, sweep, reps: int, n_ref: int):
    """Gradient-vector error and cosine against a reference, per histories per edge."""
    t = time.perf_counter()
    g1 = plan.tip_gradient(tips, weights, grad_dose, n_ref).ravel()
    t_ref = time.perf_counter() - t
    g2 = plan.tip_gradient(tips, weights, grad_dose, n_ref).ravel()
    reference = 0.5 * (g1 + g2)
    ref_noise = np.linalg.norm(g1 - g2) / 2.0 / np.linalg.norm(reference)
    print(f"reference: {n_ref:.0e} histories/edge, {t_ref:.1f} s per run, noise {ref_noise:.4f}")
    print("  N/edge   wall[s]   |g - g_ref| / |g_ref|   cos(g, g_ref)")
    rows = []
    for n in sweep:
        errors, cosines, walls = [], [], []
        for _ in range(reps):
            t = time.perf_counter()
            g = plan.tip_gradient(tips, weights, grad_dose, n).ravel()
            walls.append(time.perf_counter() - t)
            errors.append(np.linalg.norm(g - reference) / np.linalg.norm(reference))
            cosines.append(g @ reference / np.linalg.norm(g) / np.linalg.norm(reference))
        rows.append((n, np.mean(errors), np.std(errors), np.mean(cosines)))
        print(
            f"  {n:6d}   {np.median(walls):7.2f}   {np.mean(errors):.3f} +- {np.std(errors):.3f}"
            f"          {np.mean(cosines):.4f}"
        )
    return np.array(rows)


# --- part 2: optimization ----------------------------------------------------------


def project(tips: np.ndarray) -> np.ndarray:
    """Keep tips inside the field and the -x tip at or below the +x tip."""
    tips = np.clip(tips, FIELD_X[0], FIELD_X[1])
    crossed = tips[..., 0] > tips[..., 1]
    middle = 0.5 * (tips[..., 0] + tips[..., 1])
    tips[..., 0] = np.where(crossed, middle, tips[..., 0])
    tips[..., 1] = np.where(crossed, middle, tips[..., 1])
    return tips


def optimize(plan: Plan, tips, weights, iterations: int, n_fwd: int, n_edge: int, lr: float):
    """Projected Adam on tips (cm) and MU weights (relative steps).

    The step decays as ``1/sqrt(k)``: with stochastic gradients a constant step
    leaves the iterate jittering at a noise floor set by the step (measured: the
    loss stalls and drifts back up after ~10 iterations at a constant 0.5 mm).
    """
    m_t, v_t = np.zeros_like(tips), np.zeros_like(tips)
    m_w, v_w = np.zeros_like(weights), np.zeros_like(weights)
    beta1, beta2, eps = 0.9, 0.999, 1e-12
    history = []
    for k in range(1, iterations + 1):
        t0 = time.perf_counter()
        doses = plan.segment_doses(tips, n_fwd)
        dose = np.tensordot(weights, doses, axes=1)
        loss, grad_dose = plan.objective(dose)
        history.append((tips.copy(), weights.copy(), loss, dose))
        g_tips = plan.tip_gradient(tips, weights, grad_dose, n_edge)
        g_w = np.array([np.sum(grad_dose * d) for d in doses])
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
        print(
            f"  iteration {k:3d}: loss {loss:.4e}   {time.perf_counter() - t0:5.2f} s", flush=True
        )
    doses = plan.segment_doses(tips, n_fwd)
    dose = np.tensordot(weights, doses, axes=1)
    history.append((tips.copy(), weights.copy(), plan.objective(dose)[0], dose))
    return history


# --- figure -------------------------------------------------------------------------


def bev_fluence(plan: Plan, tips: np.ndarray, weights: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """MU-weighted transmission summed over segments on a beam's-eye-view grid."""
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    total = np.zeros(xx.size)
    for s in range(N_SEGMENTS):
        stack = segment_stack(tips[s], plan.material, plan.density)
        total += weights[s] * strip_transmission(stack, plan.mu, xx.ravel(), yy.ravel())
    return total.reshape(xx.shape)


def draw_bev(ax, plan, structures, tips, weights, title: str) -> None:
    """Fluence, structure outlines and one segment's leaves in the beam's-eye view."""
    xs = np.linspace(FIELD_X[0], FIELD_X[1], 200)
    ys = np.linspace(FIELD_Y[0] + 0.01, FIELD_Y[1] - 0.01, 160)
    fluence = bev_fluence(plan, tips, weights, xs, ys)
    ax.imshow(
        fluence.T,
        origin="lower",
        extent=(FIELD_X[0], FIELD_X[1], FIELD_Y[0], FIELD_Y[1]),
        cmap="viridis",
        aspect="equal",
    )
    mid = structures["ptv"].shape[2] // 2
    grid_x = (np.arange(structures["ptv"].shape[0]) + 0.5) * 0.25
    grid_y = (np.arange(structures["ptv"].shape[1]) + 0.5) * 0.25
    for name, color in (("ptv", "tab:red"), ("oar", "white")):
        ax.contour(
            grid_x, grid_y, structures[name][:, :, mid].T, [0.5], colors=color, linewidths=1.2
        )
    for i in range(N_PAIRS):
        y0, y1 = LEAF_EDGES[i], LEAF_EDGES[i + 1]
        a, b = tips[SEGMENT_SHOWN, i]
        ax.fill_between([FIELD_X[0], a], y0, y1, color="0.75", alpha=0.55, lw=0)
        ax.fill_between([b, FIELD_X[1]], y0, y1, color="0.75", alpha=0.55, lw=0)
    ax.set(title=title, xlabel="x (cm)", ylabel="y (cm)")


def dvh(dose: np.ndarray, mask: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Cumulative dose-volume histogram of ``mask`` at ``levels``."""
    values = dose[mask]
    return np.array([(values >= level).mean() for level in levels])


def render(plan, structures, precision, history, out: Path) -> None:
    """Draw the summary figure and the per-iteration GIF."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.errorbar(
        precision[:, 0], precision[:, 1], yerr=precision[:, 2], marker="o", label="relative error"
    )
    ax.plot(precision[:, 0], 1.0 - precision[:, 3], "s--", label="1 - cosine")
    n = precision[:, 0]
    ax.plot(n, precision[0, 1] * np.sqrt(n[0] / n), color="0.6", lw=0.8, label="1/sqrt(N)")
    ax.set(
        xscale="log",
        yscale="log",
        title="(a) gradient precision, 400 edges",
        xlabel="histories per leaf edge",
        ylabel="error of the gradient vector",
    )
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    losses = [h[2] for h in history]
    ax.plot(range(len(losses)), losses, "o-")
    ax.set(title="(b) objective", xlabel="iteration", ylabel="loss / Rx^2", yscale="log")

    picks = (0, len(history) // 2, len(history) - 1)
    for ax, k in zip((axes[0, 2], axes[1, 0], axes[1, 1]), picks, strict=True):
        tips, weights, _loss, _ = history[k]
        label = "(c)" if ax is axes[0, 2] else "(d)" if ax is axes[1, 0] else "(e)"
        draw_bev(
            ax,
            plan,
            structures,
            tips,
            weights,
            f"{label} iteration {k}: fluence, segment {SEGMENT_SHOWN} leaves",
        )

    ax = axes[1, 2]
    levels = np.linspace(0.0, 1.4, 141)
    for k, style in ((0, "--"), (len(history) - 1, "-")):
        dose = history[k][3] / plan.prescription
        for name, color in (("ptv", "tab:red"), ("oar", "tab:blue")):
            ax.plot(
                levels,
                dvh(dose, structures[name], levels),
                style,
                color=color,
                label=f"{name.upper()} it {k}",
            )
    ax.set(
        title="(f) DVH, initial (dashed) and final", xlabel="dose / Rx", ylabel="volume fraction"
    )
    ax.legend(fontsize=8)

    fig.savefig(out.with_suffix(".png"), dpi=110)
    print(f"saved {out.with_suffix('.png')}")

    anim_fig, anim_ax = plt.subplots(figsize=(6, 5))

    def frame(k: int) -> None:
        anim_ax.clear()
        tips, weights, loss, _ = history[k]
        draw_bev(anim_ax, plan, structures, tips, weights, f"iteration {k}, loss {loss:.3e}")

    animation = FuncAnimation(anim_fig, frame, frames=len(history))
    animation.save(out.with_suffix(".gif"), writer=PillowWriter(fps=3))
    print(f"saved {out.with_suffix('.gif')}")


def main() -> None:
    """Measure the gradient precision, run the optimization, render the figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--forward", type=int, default=300_000, help="histories per segment")
    parser.add_argument("--per-edge", type=int, default=1_000, help="gradient histories per edge")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam step on the tips (cm)")
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--reference", type=int, default=100_000, help="reference histories/edge")
    args = parser.parse_args()

    grid = make_grid()
    structures = make_structures(grid)
    xs, material, density, label = make_cross_sections("tungsten", grid)
    engine, engine_name = make_engine("warp", grid, xs)
    plan = Plan(engine, xs, material, density, structures)
    rng = np.random.default_rng(7)
    tips = project(initial_tips(rng))
    weights = rng.uniform(0.5, 1.5, N_SEGMENTS)
    print(f"engine {engine_name}; devices {label}; {2 * N_PAIRS * N_SEGMENTS} leaf edges")

    doses = plan.segment_doses(tips, 2_000_000)
    dose = np.tensordot(weights, doses, axes=1)
    plan.prescription = 1.1 * float(dose[structures["ptv"]].mean())
    _, grad_dose = plan.objective(dose)

    print("-- part 1: gradient precision at the initial iterate --")
    sweep = (300, 1_000, 3_000, 10_000)
    precision = precision_study(plan, tips, weights, grad_dose, sweep, args.reps, args.reference)

    print("-- part 2: Monte Carlo direct aperture optimization --")
    history = optimize(plan, tips, weights, args.iterations, args.forward, args.per_edge, args.lr)
    render(plan, structures, precision, history, Path(__file__).with_suffix(""))


if __name__ == "__main__":
    main()
