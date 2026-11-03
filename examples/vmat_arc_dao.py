"""Monte Carlo direct aperture optimization of a coplanar arc, one aperture per angle.

A demonstrator of the aperture-gradient machinery at arc scale, not a planning system
(the planning loop belongs in pyRadPlan; nothing here changes the library). Each
gantry angle is one control point: a 6 MV fan through a tungsten MLC with a single
aperture and an MU weight. Every iteration computes, entirely by Monte Carlo on the
GPU,

- **forward** — each angle's dose per fan history as the columns of one
  :meth:`~pyradmc.backends.warp.engine.WarpEngine.run_dij` over a single
  :class:`~pyradmc.geometry.segments.SegmentedFanSource` whose segments are the
  angles, each with its own fan and beam frame (the MLC transmission evaluated
  in-kernel), summed with the MU weights;
- **gradient** — the objective's derivative with respect to every leaf tip of the
  arc, one :meth:`~pyradmc.backends.warp.engine.WarpEngine.run_beamlet_contraction`
  over a :class:`~pyradmc.geometry.segment_edges.SegmentedEdgeSource` of that source;
  the MU weights' gradient is the adjoint's dot product with each angle's dose;

then one Adam step on all tips and weights, the tips projected onto the machine's
limits: inside the field, each pair uncrossed, and a leaf-speed limit, each tip moving
at most ``--leaf-speed`` cm (at the isocentre) per degree of gantry rotation between
neighbouring control points (default 0.2 cm/deg, 2 cm per 10 deg; a real MLC at
2.5 cm/s under a 6 deg/s gantry allows ~0.4). The projection is a small QP per leaf
pair (``LeafProjection``), solved in Adam's own metric.

**Stochastic schedule.** The Adam step stays constant and the forward histories grow
instead, by ``--growth`` per iteration from ``--forward`` to ``--forward-max`` per
angle (edge histories forward / 200, at least ``--per-edge``): the noise of a fixed
small sample would hold a constant step in a noise ball, and a decaying step stalls
before the noise matters. At equal wall time this beat a 1/sqrt(k) decay at 100k
histories by 47 % in true loss, and an interior-point solver (ipax) on a fixed sample
by 20-23 % at every leaf-speed limit (docs/decisions.md, "Stochastic optimization of
the arc").

**Initialization.** The apertures start from the beam's-eye-view projection of the
target with the OAR's projection cut out (``--init avoid``), opening the same side of
the OAR from every angle where it splits a leaf row (the opposed beams cover the other
side), then projected onto the limits. Opening alternate sides at neighbouring angles
(``--init alternate``) is as good without a speed limit, but every limit forbids it:
its nearest feasible plan straddles the OAR from every angle. The plain conformal
projection (``--init conformal``) wraps the C-shaped target around the OAR and is a
local minimum for the tips, so the optimizer can only scale the MU down. Direct
aperture optimization is non-convex; the initialization matters. MU-rate and gantry-
speed constraints are not modelled.

Geometry: a water box 16 x 6.4 x 16 cm on 2.5 x 4 x 2.5 mm voxels; the gantry rotates
about the y axis through the isocentre at the box centre (SAD 100 cm); a C-shaped PTV
around a cylindrical OAR in the transverse (x, z) plane, 4 cm long in y (after the
TG-119 C-shape); electrons transported. Leaves travel in the rotation plane, 5 mm wide
at the isocentre along the gantry axis.

The whole arc costs two engine launches per iteration, whatever the number of angles:
at 36 angles 0.06 s plus 0.38 s per million forward histories per angle on an RTX 4070
laptop GPU, so the growing sample takes an iteration from ~0.1 s to ~0.8 s; the
example reports the times.
Renders ``vmat_arc_dao.png`` and ``vmat_arc_dao.gif`` beside this script.
Needs a CUDA device and the EPICS libraries (cached or downloadable)::

    python examples/vmat_arc_dao.py --spacing 10
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

SAD = 100.0  # focal spot to isocentre (cm)
SHAPE, SPACING = (64, 16, 64), (0.25, 0.4, 0.25)
ISO = tuple(0.5 * n * d for n, d in zip(SHAPE, SPACING, strict=True))  # box centre
FIELD_U, FIELD_V = 9.0, 5.0  # field at the isocentre: leaf travel x leaf width (cm)
N_PAIRS = 10  # 5 mm leaves at the isocentre
MLC_Z = (50.0, 56.0)  # tungsten slab, distance from the focal spot (cm)
TO_MID = 0.5 * sum(MLC_Z) / SAD  # isocentre plane -> MLC mid-plane scaling
LEAF_EDGES = tuple(float(e) for e in np.linspace(-0.5, 0.5, N_PAIRS + 1) * FIELD_V * TO_MID)
TIP_LIMIT = 0.5 * FIELD_U * TO_MID  # tips stay inside the field (mid-plane cm)
MARGIN = 0.3  # cm at the isocentre: the initial apertures' margin around the PTV
OAR_MARGIN = 0.2  # cm at the isocentre: the initial apertures' clearance of the OAR
DOSE_MAX = 1.3  # colour scale top, in units of the prescription
GIF_FRAMES = 100


def make_grid() -> VoxelGrid:
    """Water box, 16 x 6.4 x 16 cm, centred on the isocentre."""
    return VoxelGrid.uniform_water(shape=SHAPE, spacing=SPACING)


def make_structures(grid: VoxelGrid) -> dict[str, np.ndarray]:
    """C-shaped PTV (open towards +x) around a cylindrical OAR, 4 cm long in y."""
    axes = [(np.arange(n) + 0.5) * d for n, d in zip(grid.shape, grid.spacing, strict=True)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    r = np.hypot(xx - ISO[0], zz - ISO[2])
    phi = np.arctan2(zz - ISO[2], xx - ISO[0])
    length = np.abs(yy - ISO[1]) < 2.0
    oar = (r < 1.2) & length
    ptv = (r >= 1.7) & (r < 3.7) & (np.abs(phi) > np.pi / 4) & length
    return {"ptv": ptv, "oar": oar, "normal": ~(ptv | oar)}


def beam_axes(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(u, v, w)`` of the beam at a gantry angle (rotation about +y).

    ``w`` points from the focal spot through the isocentre (angle 0: along +z), ``u``
    is the leaf travel in the rotation plane, ``v`` the gantry axis; ``u x v = w``.
    """
    a = np.radians(angle_deg)
    w = np.array([np.sin(a), 0.0, np.cos(a)])
    u = np.array([np.cos(a), 0.0, -np.sin(a)])
    return u, np.array([0.0, 1.0, 0.0]), w


def focal_point(angle_deg: float) -> np.ndarray:
    """Return the focal spot of the beam at a gantry angle."""
    return np.asarray(ISO) - SAD * beam_axes(angle_deg)[2]


def make_fan(spectrum, angle_deg: float) -> SpectralBeamSource:
    """Build the exact-type fan of one gantry angle (generated in-kernel)."""
    u, v, _ = beam_axes(angle_deg)
    return SpectralBeamSource(
        spectrum,
        focal_point=tuple(focal_point(angle_deg)),
        center=ISO,
        width_u=FIELD_U,
        width_v=FIELD_V,
        u_axis=tuple(u),
        v_axis=tuple(v),
    )


def make_stack(angle_deg: float, tips: np.ndarray) -> BeamLimitingStack:
    """One angle's tungsten MLC in its beam frame; ``tips`` is ``(N_PAIRS, 2)`` mid-plane."""
    u, v, w = beam_axes(angle_deg)
    return BeamLimitingStack(
        frame=BeamFrame(
            origin=tuple(focal_point(angle_deg)), u_axis=tuple(u), v_axis=tuple(v), w_axis=tuple(w)
        ),
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


def bev_coordinates(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Project engine-frame points through the focal spot onto the isocentre plane."""
    u, v, w = beam_axes(angle_deg)
    d = points - focal_point(angle_deg)
    scale = SAD / (d @ w)
    return np.stack([(d @ u) * scale, (d @ v) * scale], axis=1)


def voxel_centres(grid: VoxelGrid, mask: np.ndarray) -> np.ndarray:
    """Engine-frame centres of the voxels in ``mask``."""
    index = np.argwhere(mask)
    return (index + 0.5) * np.asarray(grid.spacing)


def initial_tips(
    grid: VoxelGrid, structures, angles: np.ndarray, mode: str = "avoid"
) -> np.ndarray:
    """Return the initial apertures from the beam's-eye-view projection of the PTV.

    ``mode="conformal"``: each leaf row opens over the projected PTV plus ``MARGIN``.
    Around a C-shaped target this aperture covers the OAR from every angle, and it is
    a local minimum for the tips: closing a leaf towards the OAR first underdoses the
    PTV, so gradient descent only scales the MU down (measured; see the docstring).

    ``mode="avoid"`` (default): the projected PTV minus the projected OAR (plus
    ``OAR_MARGIN``). Where the OAR splits a row's PTV extent in two, one aperture per
    row can open only one side; every angle opens the same side, and the opposed
    beams half an arc later cover the other. ``mode="alternate"`` switches sides
    between neighbouring angles instead: as good without a leaf-speed limit, but no
    limit allows a leaf to cross the OAR at every control point.

    Returns ``(n_angles, N_PAIRS, 2)`` tips at the MLC mid-plane; a row the PTV does
    not reach (or with nothing left) is closed at the beam axis.
    """
    if mode not in ("conformal", "avoid", "alternate"):
        raise ValueError(f"unknown initialization {mode!r}")
    ptv = voxel_centres(grid, structures["ptv"])
    oar = voxel_centres(grid, structures["oar"])
    half_voxel = 0.5 * max(grid.spacing)
    rows = np.asarray(LEAF_EDGES) / TO_MID  # leaf boundaries at the isocentre
    tips = np.zeros((len(angles), N_PAIRS, 2))
    for a, angle in enumerate(angles):
        bev_ptv, bev_oar = bev_coordinates(ptv, angle), bev_coordinates(oar, angle)
        for i in range(N_PAIRS):
            row = (rows[i] - half_voxel, rows[i + 1] + half_voxel)
            inside = (bev_ptv[:, 1] > row[0]) & (bev_ptv[:, 1] < row[1])
            if not inside.any():
                continue
            lo = bev_ptv[inside, 0].min() - MARGIN - half_voxel
            hi = bev_ptv[inside, 0].max() + MARGIN + half_voxel
            blocked = (bev_oar[:, 1] > row[0]) & (bev_oar[:, 1] < row[1])
            if mode != "conformal" and blocked.any():
                o_lo = bev_oar[blocked, 0].min() - OAR_MARGIN - half_voxel
                o_hi = bev_oar[blocked, 0].max() + OAR_MARGIN + half_voxel
                sides = [(s_lo, s_hi) for s_lo, s_hi in ((lo, min(hi, o_lo)), (max(lo, o_hi), hi))
                         if s_hi - s_lo > 2.0 * half_voxel]  # fmt: skip
                if not sides:
                    continue
                lo, hi = sides[a % len(sides) if mode == "alternate" else 0]
            tips[a, i] = (lo * TO_MID, hi * TO_MID)
    return project(tips)


def project(tips: np.ndarray) -> np.ndarray:
    """Keep tips inside the field and each pair uncrossed (mid-plane coordinates)."""
    tips = np.clip(tips, -TIP_LIMIT, TIP_LIMIT)
    crossed = tips[..., 0] > tips[..., 1]
    middle = 0.5 * (tips[..., 0] + tips[..., 1])
    tips[..., 0] = np.where(crossed, middle, tips[..., 0])
    tips[..., 1] = np.where(crossed, middle, tips[..., 1])
    return tips


class LeafProjection:
    """Nearest feasible tips: in the field, uncrossed, leaf travel limited along the arc.

    Solves, for every leaf pair at once,

        min sum D (x - y)^2   s.t.  |x| <= TIP_LIMIT,  neg <= pos,
                                    |x(a+1) - x(a)| <= delta  (each tip, neighbouring angles)

    with ``D`` a positive metric (Adam's ``sqrt(v_hat)``, or 1). The constraints couple
    only the two tips of one pair along the arc, so each pair is a small QP (``2 x
    n_angles`` variables); all pairs are solved together by ADMM (OSQP's splitting,
    variables scaled by ``sqrt(D)``, residual-balanced penalty, warm-started between
    calls): 15-130 ms at 36 angles, feasible to ~1e-7 cm, then clipped and uncrossed
    exactly. ``delta=None`` drops the travel limit.
    """

    def __init__(self, n_angles: int, delta: float | None, tol: float = 1e-7) -> None:
        self.n_angles, self.n = n_angles, 2 * n_angles
        rows = [np.eye(self.n)]
        lo, hi = [np.full(self.n, -TIP_LIMIT)], [np.full(self.n, TIP_LIMIT)]
        cross = np.zeros((n_angles, self.n))
        cross[np.arange(n_angles), np.arange(n_angles)] = 1.0
        cross[np.arange(n_angles), n_angles + np.arange(n_angles)] = -1.0
        rows.append(cross), lo.append(np.full(n_angles, -np.inf)), hi.append(np.zeros(n_angles))
        if delta is not None:
            step = np.diff(np.eye(n_angles), axis=0)  # x(a+1) - x(a)
            for travel in (np.hstack([step, 0 * step]), np.hstack([0 * step, step])):
                rows.append(travel)
                lo.append(np.full(n_angles - 1, -delta))
                hi.append(np.full(n_angles - 1, delta))
        self.C, self.lo, self.hi = np.vstack(rows), np.concatenate(lo), np.concatenate(hi)
        self.delta, self.tol = delta, tol
        self.state = None

    def __call__(self, tips: np.ndarray, metric: np.ndarray | None = None) -> np.ndarray:
        """Project ``(n_angles, N_PAIRS, 2)`` tips; ``metric`` has their shape."""
        a = self.n_angles
        y = np.concatenate([tips[:, :, 0].T, tips[:, :, 1].T], axis=1)  # (pairs, 2A)
        d = np.ones_like(y) if metric is None else np.concatenate(
            [metric[:, :, 0].T, metric[:, :, 1].T], axis=1)  # fmt: skip
        d = np.clip(d / d.mean(axis=1, keepdims=True), 1e-3, 1e3)  # scale-free; conditioned
        s = np.sqrt(d)  # u = s x: objective |u - s y|^2 / 2, constraints C diag(1/s) u
        cs = self.C[None] / s[:, None, :]
        ctc = np.einsum("pmi,pmj->pij", cs, cs)
        eye, sigma, alpha = np.eye(self.n)[None], 1e-6, 1.6
        if self.state is None or self.state[0].shape[0] != y.shape[0]:
            z = np.clip(y @ self.C.T, self.lo, self.hi)
            lam, rho = np.zeros_like(z), np.ones(len(y))
        else:
            z, lam, rho = self.state
        k_inv = np.linalg.inv((1.0 + sigma) * eye + rho[:, None, None] * ctc)
        target = s * y
        u = target.copy()
        for it in range(1, 3001):
            rhs = target + sigma * u + np.einsum("pm,pmn->pn", rho[:, None] * z - lam, cs)
            u = np.einsum("pij,pj->pi", k_inv, rhs)
            cu = np.einsum("pmn,pn->pm", cs, u)
            relaxed = alpha * cu + (1 - alpha) * z
            z_new = np.clip(relaxed + lam / rho[:, None], self.lo, self.hi)
            lam = lam + rho[:, None] * (relaxed - z_new)
            dual = rho * np.abs(np.einsum("pm,pmn->pn", z_new - z, cs)).max(axis=1)
            z = z_new
            if it % 10:
                continue
            primal = np.abs(cu - z).max(axis=1)
            if max(primal.max(), dual.max()) < self.tol:
                break
            if it % 50 == 0:  # balance the residuals, then refactor
                rho = np.clip(rho * np.clip(np.sqrt((primal + 1e-12) / (dual + 1e-12)), 0.2, 5.0),
                              1e-4, 1e4)  # fmt: skip
                k_inv = np.linalg.inv((1.0 + sigma) * eye + rho[:, None, None] * ctc)
        self.state = (z, lam, rho)
        x = u / s
        return project(np.stack([x[:, :a].T, x[:, a:].T], axis=-1))

    def max_travel(self, tips: np.ndarray) -> float:
        """Largest tip travel between neighbouring angles (mid-plane cm)."""
        return float(np.abs(np.diff(tips, axis=0)).max()) if len(tips) > 1 else 0.0


class ArcPlan:
    """Per-angle forward doses, the objective and the tip gradients of one arc."""

    def __init__(self, engine, xs, spectrum, angles, structures, tips) -> None:
        self.engine = engine
        self.angles = angles
        self.structures = structures
        self.prescription = 1.0
        self.seed = 1
        # One source for the whole arc: segment a is angle a, with its own fan and frame.
        self.source = SegmentedFanSource(
            [make_fan(spectrum, angle) for angle in angles],
            make_stack(angles[0], tips[0]),
            xs,
            frames=[make_stack(angle, tips[a]).frame for a, angle in enumerate(angles)],
            tips_neg=tips[..., 0],
            tips_pos=tips[..., 1],
        )

    def _seed(self) -> int:
        self.seed += 1
        return self.seed

    def set_tips(self, tips: np.ndarray) -> None:
        """Re-target the arc to new tips (arrays only, no MLC objects)."""
        self.source = self.source.with_settings(tips_neg=tips[..., 0], tips_pos=tips[..., 1])

    def angle_doses(self, n_per_angle: int) -> np.ndarray:
        """Dose per fan history of every angle, ``(n_angles, *grid)``: one segmented Dij.

        Correlated sampling (the default) drives every angle's fan with the same
        uniforms; no truncation, since every column enters the plan dose.
        """
        dij = self.engine.run_dij(
            self.source,
            n_histories_per_beamlet=n_per_angle,
            n_batches=2,
            seed=self._seed(),
            truncation=0.0,
        )
        return np.stack([dij.column_dense(a) for a in range(len(self.angles))])

    def objective(self, dose: np.ndarray) -> tuple[float, np.ndarray]:
        """Loss and dL/dD: PTV to prescription, one-sided OAR and normal-tissue limits."""
        ptv, oar, normal = (self.structures[k] for k in ("ptv", "oar", "normal"))
        rx = self.prescription
        grad = np.zeros_like(dose)
        r_ptv = dose[ptv] - rx
        r_oar = np.maximum(0.0, dose[oar] - 0.5 * rx)
        r_nt = np.maximum(0.0, dose[normal] - 0.6 * rx)
        grad[ptv] = 2.0 * r_ptv / ptv.sum()
        grad[oar] = 2.0 * 2.0 * r_oar / oar.sum()
        grad[normal] = 2.0 * 0.5 * r_nt / normal.sum()
        loss = (r_ptv**2).mean() + 2.0 * (r_oar**2).mean() + 0.5 * (r_nt**2).mean()
        return float(loss) / rx**2, grad

    def tip_gradient(self, weights, grad_dose: np.ndarray, n_per_edge: int) -> np.ndarray:
        """dL/d(tip), ``(n_angles, N_PAIRS, 2)``: one edge contraction for the whole arc.

        The edges are angle-major, then leaf pair, then side (neg, pos): the layout of
        ``tips``.
        """
        result = self.engine.run_beamlet_contraction(
            SegmentedEdgeSource(self.source),
            grad_dose,
            n_histories_per_beamlet=n_per_edge,
            n_batches=2,
            seed=self._seed(),
        )
        return weights[:, None, None] * result.values.reshape(len(self.angles), N_PAIRS, 2)


def histories(k: int, start: int, top: int, growth: float, per_edge: int) -> tuple[int, int]:
    """Forward histories per angle and per edge at iteration ``k`` (even: two batches)."""
    n = min(top, int(start * growth ** (k - 1)))
    return 2 * (n // 2), 2 * (max(per_edge, n // 200) // 2)


def optimize(plan: ArcPlan, tips, weights, iterations: int, schedule, lr: float,
             projection: LeafProjection):  # fmt: skip
    """Projected Adam on tips (mid-plane cm) and MU weights; constant step, growing sample.

    ``schedule(k)`` gives the forward and per-edge histories of iteration ``k``. The tips
    are projected after every step in Adam's own metric ``sqrt(v_hat)``.
    """
    m_t, v_t = np.zeros_like(tips), np.zeros_like(tips)
    m_w, v_w = np.zeros_like(weights), np.zeros_like(weights)
    beta1, beta2, eps = 0.9, 0.999, 1e-12
    history, timing = [], []
    for k in range(1, iterations + 1):
        n_fwd, n_edge = schedule(k)
        t0 = time.perf_counter()
        plan.set_tips(tips)
        doses = plan.angle_doses(n_fwd)
        t1 = time.perf_counter()
        dose = np.tensordot(weights, doses, axes=1)
        loss, grad_dose = plan.objective(dose)
        history.append((tips.copy(), weights.copy(), loss, dose))
        g_tips = plan.tip_gradient(weights, grad_dose, n_edge)
        g_w = doses.reshape(len(weights), -1) @ grad_dose.ravel()
        t2 = time.perf_counter()
        for m, v, g in ((m_t, v_t, g_tips), (m_w, v_w, g_w)):
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * g * g
        metric = np.sqrt(v_t / (1 - beta2**k)) + eps
        tips = projection(tips - lr * (m_t / (1 - beta1**k)) / metric, metric)
        step_w = 0.05 * (m_w / (1 - beta1**k)) / (np.sqrt(v_w / (1 - beta2**k)) + eps)
        weights = np.maximum(weights * (1.0 - step_w), 0.0)
        t3 = time.perf_counter()
        timing.append((t1 - t0, t2 - t1, t3 - t2))
        if k == 1 or k % 10 == 0 or k == iterations:
            print(
                f"  iteration {k:4d}: loss {loss:.4e}   {n_fwd:>9,} histories/angle   forward "
                f"{t1 - t0:5.2f} s   gradient {t2 - t1:5.2f} s   projection {t3 - t2:5.3f} s",
                flush=True,
            )
    plan.set_tips(tips)
    dose = np.tensordot(weights, plan.angle_doses(schedule(iterations)[0]), axes=1)
    history.append((tips.copy(), weights.copy(), plan.objective(dose)[0], dose))
    return history, np.array(timing)


def dvh(dose: np.ndarray, mask: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Cumulative dose-volume histogram of ``mask`` at ``levels``."""
    values = dose[mask]
    return np.array([(values >= level).mean() for level in levels])


def draw_axial(ax, grid: VoxelGrid, dose: np.ndarray, structures, title: str):
    """Transverse (x, z) dose slice through the isocentre, with PTV and OAR contours."""
    (nx, _, nz), (dx, dy, dz) = grid.shape, grid.spacing
    x = (np.arange(nx) + 0.5) * dx - ISO[0]
    z = (np.arange(nz) + 0.5) * dz - ISO[2]
    j = int(ISO[1] / dy)
    handle = ax.imshow(
        dose[:, j, :].T, origin="upper", cmap="magma", vmin=0.0, vmax=DOSE_MAX,
        extent=(x[0] - dx / 2, x[-1] + dx / 2, z[-1] + dz / 2, z[0] - dz / 2),
        interpolation="nearest",
    )  # fmt: skip
    for name, color in (("ptv", "tab:red"), ("oar", "white")):
        ax.contour(x, z, structures[name][:, j, :].T.astype(float), levels=[0.5], colors=color,
                   linewidths=1.2)  # fmt: skip
    ax.set(title=title, xlabel="x from the isocentre (cm)", ylabel="z from the isocentre (cm)")
    return handle


def draw_aperture(ax, grid, structures, angle, tips_initial, tips_final, title: str) -> None:
    """Beam's-eye view of one angle: final leaves, initial tips, projected PTV/OAR."""
    rows = np.asarray(LEAF_EDGES) / TO_MID
    for i in range(N_PAIRS):
        y0, y1 = rows[i], rows[i + 1]
        a, b = tips_final[i] / TO_MID
        ax.fill_between([-0.5 * FIELD_U, a], y0, y1, color="0.55", alpha=0.6, lw=0)
        ax.fill_between([b, 0.5 * FIELD_U], y0, y1, color="0.55", alpha=0.6, lw=0)
        a0, b0 = tips_initial[i] / TO_MID
        ax.plot([a0, a0], [y0, y1], color="k", lw=1.0, ls="--")
        ax.plot([b0, b0], [y0, y1], color="k", lw=1.0, ls="--")
    # Structure outlines on top of the leaves. Several points per voxel, spread over
    # its volume: voxel centres alone project onto a lattice coarser than the bins.
    bins_u = np.linspace(-0.5 * FIELD_U, 0.5 * FIELD_U, 91)
    bins_v = np.linspace(-0.5 * FIELD_V, 0.5 * FIELD_V, 51)
    centres_u, centres_v = 0.5 * (bins_u[1:] + bins_u[:-1]), 0.5 * (bins_v[1:] + bins_v[:-1])
    jitter = np.random.default_rng(0)
    for name, color in (("ptv", "tab:red"), ("oar", "tab:blue")):
        centres = voxel_centres(grid, structures[name])
        spread = (jitter.random((8, *centres.shape)) - 0.5) * np.asarray(grid.spacing)
        bev = bev_coordinates((centres[None] + spread).reshape(-1, 3), angle)
        counts, _, _ = np.histogram2d(bev[:, 0], bev[:, 1], bins=(bins_u, bins_v))
        ax.contour(centres_u, centres_v, (counts > 0).T.astype(float), levels=[0.5],
                   colors=color, linewidths=1.8)  # fmt: skip
    ax.set(xlim=(-0.5 * FIELD_U, 0.5 * FIELD_U), ylim=(-0.5 * FIELD_V, 0.5 * FIELD_V),
           aspect="equal", title=title, xlabel="u at the isocentre (cm)",
           ylabel="v = gantry axis (cm)")  # fmt: skip


def render(plan, grid, structures, history, timing, clean, limit_iso, out: Path) -> None:
    """Summary figure and per-iteration GIF (see the module docstring)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    rx = plan.prescription
    angles = plan.angles
    initial, final = (d / rx for d in clean)
    last = len(history) - 1
    fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(range(len(history)), [h[2] for h in history], "-", label="in-loop (noisy forward)")
    ax.plot([0, last], [plan.objective(clean[0])[0], plan.objective(clean[1])[0]], "o",
            color="tab:red", ms=8, label="re-scored at high statistics")  # fmt: skip
    ax.set(title="(a) objective", xlabel="iteration", ylabel="loss / Rx^2", yscale="log")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    it = np.arange(1, len(timing) + 1)
    ax.bar(it, timing[:, 0], width=1.0, label=f"forward ({len(angles)} angles)")
    ax.bar(it, timing[:, 1], width=1.0, bottom=timing[:, 0],
           label=f"gradient ({2 * N_PAIRS * len(angles)} edges)")  # fmt: skip
    ax.bar(it, timing[:, 2], width=1.0, bottom=timing[:, 0] + timing[:, 1],
           label="leaf projection")  # fmt: skip
    ax.set(title="(b) wall time per iteration (growing sample)", xlabel="iteration",
           ylabel="seconds")  # fmt: skip
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    levels = np.linspace(0.0, 1.4, 141)
    for dose, style, when in ((initial, "--", "initial"), (final, "-", "final")):
        for name, color in (("ptv", "tab:red"), ("oar", "tab:blue")):
            ax.plot(levels, dvh(dose, structures[name], levels), style, color=color,
                    label=f"{name.upper()} {when}")  # fmt: skip
    ax.set(title="(c) DVH, initial (dashed) and final", xlabel="dose / Rx",
           ylabel="volume fraction")  # fmt: skip
    ax.legend(fontsize=8)

    draw_axial(axes[1, 0], grid, initial, structures, "(d) initial dose")
    handle = draw_axial(axes[1, 1], grid, final, structures, "(e) final dose")
    fig.colorbar(handle, ax=axes[1, :2], label="dose / Rx", shrink=0.8)

    axes[1, 2].remove()
    polar = fig.add_subplot(3, 3, 6, projection="polar")
    theta = np.radians(angles)
    width = np.radians(np.diff(angles).min() if len(angles) > 1 else 10.0) * 0.8
    polar.bar(theta, history[0][1], width=width, alpha=0.35, color="0.4", label="initial")
    polar.bar(theta, history[-1][1], width=width * 0.5, color="tab:orange", label="final")
    polar.set_theta_zero_location("N")
    polar.set_theta_direction(-1)
    polar.set_title("(f) MU weight per gantry angle")
    polar.legend(fontsize=8, loc="lower left")

    picks = sorted({0, len(angles) // 4})
    tips0, tips1 = history[0][0], history[-1][0]
    for ax, a, label in zip(axes[2, :2], picks, "gh", strict=False):
        title = f"({label}) gantry {angles[a]:g} deg: final leaves, initial dashed"
        draw_aperture(ax, grid, structures, angles[a], tips0[a], tips1[a], title)

    ax = axes[2, 2]
    for i, color in zip((N_PAIRS // 2 - 2, N_PAIRS // 2), ("tab:green", "tab:purple"),
                        strict=True):  # fmt: skip
        for side in (0, 1):
            ax.plot(angles, tips0[:, i, side] / TO_MID, "--", color=color, lw=1.0)
            ax.plot(angles, tips1[:, i, side] / TO_MID, "-", color=color, lw=1.5,
                    label=f"pair {i}" if side == 0 else None)  # fmt: skip
    travel = np.abs(np.diff(tips1, axis=0)).max() / TO_MID if len(angles) > 1 else 0.0
    limit = "none" if limit_iso is None else f"{limit_iso:.2f} cm"
    ax.set(title=f"(i) leaf tips along the arc, initial dashed\nmax travel per control point "
                 f"{travel:.2f} cm (limit {limit})", xlabel="gantry angle (deg)",
           ylabel="tip at the isocentre (cm)", ylim=(-0.5 * FIELD_U, 0.5 * FIELD_U))  # fmt: skip
    ax.legend(fontsize=8)
    fig.savefig(out.with_suffix(".png"), dpi=110)
    print(f"saved {out.with_suffix('.png')}")

    anim_fig, (dose_ax, leaf_ax) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    frames = sorted(set(range(0, len(history), max(1, len(history) // GIF_FRAMES))) | {last})
    centre_row = N_PAIRS // 2
    colorbar = []

    def frame(k: int) -> None:
        dose_ax.clear()
        leaf_ax.clear()
        tips, weights, loss, dose = history[k]
        handle = draw_axial(dose_ax, grid, dose / rx, structures,
                            f"iteration {k}: dose (noisy), loss {loss:.3f}")  # fmt: skip
        if not colorbar:
            colorbar.append(anim_fig.colorbar(handle, ax=dose_ax, label="dose / Rx"))
        lo, hi = tips[:, centre_row, 0] / TO_MID, tips[:, centre_row, 1] / TO_MID
        leaf_ax.fill_between(angles, lo, hi, color="tab:orange", alpha=0.5, step="mid")
        leaf_ax.plot(angles, weights / weights.max() * 0.45 * FIELD_U, "k.", ms=4,
                     label="MU (scaled)")  # fmt: skip
        leaf_ax.set(xlim=(angles[0], angles[-1]), ylim=(-0.5 * FIELD_U, 0.5 * FIELD_U),
                    title=f"centre leaf pair opening vs gantry angle (row {centre_row})",
                    xlabel="gantry angle (deg)", ylabel="u at the isocentre (cm)")  # fmt: skip
        leaf_ax.legend(fontsize=8, loc="lower right")

    FuncAnimation(anim_fig, frame, frames=frames).save(
        out.with_suffix(".gif"), writer=PillowWriter(fps=6)
    )
    print(f"saved {out.with_suffix('.gif')} ({len(frames)} frames)")


def main() -> None:
    """Build the conformal arc, optimize it, re-score and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spacing", type=float, default=10.0, help="gantry spacing (deg)")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--forward", type=int, default=100_000,
                        help="forward histories per angle at the first iteration")  # fmt: skip
    parser.add_argument("--forward-max", type=int, default=2_000_000,
                        help="cap of the growing forward histories per angle")  # fmt: skip
    parser.add_argument("--growth", type=float, default=1.03,
                        help="forward histories grow by this factor per iteration")  # fmt: skip
    parser.add_argument("--per-edge", type=int, default=500,
                        help="minimum histories per edge (else forward / 200)")  # fmt: skip
    parser.add_argument("--lr", type=float, default=0.03,
                        help="constant Adam step on tips (mid-plane cm)")  # fmt: skip
    parser.add_argument("--leaf-speed", type=float, default=0.2,
                        help="leaf travel limit, cm at the isocentre per degree of gantry "
                             "rotation (0: none)")  # fmt: skip
    parser.add_argument(
        "--init", choices=("avoid", "alternate", "conformal"), default="avoid",
        help="initial apertures: PTV minus OAR projection (one side, or alternating sides "
             "between angles), or the plain PTV projection",
    )  # fmt: skip
    parser.add_argument(
        "--evaluate", type=int, default=2_000_000,
        help="histories per angle for the initial and final doses shown",
    )  # fmt: skip
    args = parser.parse_args()

    from pyradmc.backends.warp.engine import WarpEngine
    from pyradmc.data.tabulated.build import download_library
    from pyradmc.data.tabulated.precompile import compile_materials
    from pyradmc.data.tabulated.source import TabulatedCrossSections

    grid = make_grid()
    structures = make_structures(grid)
    angles = np.arange(0.0, 360.0, args.spacing)
    print("compiling the tabulated table (EPICS cache/download) ...")
    data = compile_materials(
        download_library("epdl").read_text(encoding="latin-1"),
        download_library("eedl").read_text(encoding="latin-1"),
        e_max=7.0,
    )
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    engine = WarpEngine(grid=grid, cross_sections=xs, device="cuda:0")
    limit_iso = args.leaf_speed * args.spacing if args.leaf_speed > 0.0 else None
    if limit_iso is not None and args.init == "alternate":
        parser.error("--init alternate crosses the OAR at every control point; no leaf-speed "
                     "limit allows it (use --leaf-speed 0)")  # fmt: skip
    projection = LeafProjection(len(angles), None if limit_iso is None else limit_iso * TO_MID)
    tips = projection(initial_tips(grid, structures, angles, args.init))  # feasible start
    projection.state = None
    weights = np.ones(len(angles))
    plan = ArcPlan(engine, xs, ali_rogers_mv("varian-6mv"), angles, structures, tips)

    initial = np.tensordot(weights, plan.angle_doses(args.evaluate), axes=1)
    plan.prescription = float(initial[structures["ptv"]].mean())
    n_edges = 2 * N_PAIRS * len(angles)
    limit = "none" if limit_iso is None else f"{limit_iso:.2f} cm per control point"
    print(f"-- Monte Carlo VMAT arc: {len(angles)} angles x 1 aperture, {n_edges} leaf edges, "
          f"leaf travel limit {limit} --")  # fmt: skip
    history, timing = optimize(
        plan, tips, weights, args.iterations,
        lambda k: histories(k, args.forward, args.forward_max, args.growth, args.per_edge),
        args.lr, projection,
    )  # fmt: skip
    print(
        f"wall time {timing.sum():.1f} s; median per iteration: forward "
        f"{np.median(timing[:, 0]):.2f} s, gradient {np.median(timing[:, 1]):.2f} s, projection "
        f"{np.median(timing[:, 2]):.3f} s (two engine launches)"
    )
    final_tips, final_weights = history[-1][0], history[-1][1]
    travel = projection.max_travel(final_tips) / TO_MID
    print(f"max leaf travel between neighbouring angles: {travel:.3f} cm at the isocentre "
          f"(limit {limit})")  # fmt: skip
    plan.set_tips(final_tips)
    final = np.tensordot(final_weights, plan.angle_doses(args.evaluate), axes=1)
    print(
        f"loss re-scored at {args.evaluate:,} histories per angle: "
        f"{plan.objective(initial)[0]:.4f} -> {plan.objective(final)[0]:.4f} "
        f"(in-loop final {history[-1][2]:.4f})"
    )
    render(plan, grid, structures, history, timing, (initial, final), limit_iso,
           Path(__file__).with_suffix(""))  # fmt: skip


if __name__ == "__main__":
    main()
