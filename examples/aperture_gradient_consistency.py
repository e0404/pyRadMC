"""Measure the Monte Carlo DAO leaf gradient against its forward dose (6 MV fan).

The direct aperture optimization of ``aperture_gradient_fan_dao.py`` computes its
forward dose and its leaf gradient through two code paths that share no code for the
transmission derivative:

- forward: :class:`~pyradmc.geometry.segments.SegmentedFanSource` columns from
  ``run_dij`` (each primary weighted in-kernel by its segment's stack transmission);
- gradient: :class:`~pyradmc.geometry.aperture_gradient.EdgeGradientBeamletSource` of
  :class:`~pyradmc.geometry.aperture_gradient.FanEdgeDerivativeSource` edges,
  contracted with the adjoint ``a = dL/dD`` by ``run_beamlet_contraction``.

This script measures, per leaf edge, the edge contraction ``g_edge`` against the
common-random-number finite difference of the forward model in the contracted
quantity, ``g_FD = a . (D(x + h) - D(x - h)) / 2h`` (second-order one-sided at a
closed pair, whose tips may not cross), with sigmas from batches (edge) and from
independent replicate seeds (finite difference), at three steps ``h``. The displaced
tips are columns of one correlated ``run_dij``, so rays and transport draws coincide
and only the transmission weights differ. Each edge's finite difference runs on the
fan narrowed to the edge's leaf strip and blocked side and is scaled by the area
ratio: every other ray gives bit-identical columns under common random numbers, so
this is exact in expectation and only saves histories. Two adjoints: the demo objective's
``dL/dD`` at its initial plan (``real``) and ``a = 1`` (integral dose: opening a leaf
raises it, so the sign is obvious).

Geometry: the demo's — water 12 x 10 x 12 cm on 2.5 x 2.5 x 5 mm voxels, Varian 6 MV
Ali-Rogers spectrum from a point focal spot at 100 cm, tungsten MLC with focused sides
and 8 cm rounded tips, 10 segments x 20 pairs; electrons transported. Thirteen edges:
both sides of an open pair, of a nearly closed pair (0.16 cm mid-plane gap), closed
pairs (one in a PTV row, one the outermost pair), a tip at the field limit, one whose
penumbra the field edge cuts, and edges of five segments.

Findings recorded in docs/decisions.md ("DAO gradient consistency"). Both sides of
every pair agree since the transport fix of 2026-10-23; before it, every ``neg``-side
gradient came out at 0.2-0.3 of the finite difference, because negative-weight
histories lost their electrons' continuous-loss deposits
(``tests/integration/test_signed_weights.py``).

Needs a CUDA device and the EPICS libraries (cached or downloadable)::

    python examples/aperture_gradient_consistency.py [--study]

About a minute on an RTX 4070 laptop GPU at the defaults (12 replicate seeds x 1e6
histories per finite-difference column, 1e6 per edge); ``--study`` reproduces the table
of the decision record (40 x 4e6 and 4e6 per edge, about 7 minutes). Renders
``aperture_gradient_consistency.png`` beside this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

from pyradmc.geometry.aperture_gradient import (
    EdgeGradientBeamletSource,
    EdgeParameter,
    FanEdgeDerivativeSource,
)
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv

_spec = importlib.util.spec_from_file_location(
    "aperture_gradient_fan_dao", Path(__file__).with_name("aperture_gradient_fan_dao.py")
)
assert _spec is not None and _spec.loader is not None
dao = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dao)

LIMIT = 0.5 * dao.FIELD * dao.TO_MID  # the projection's tip limit at the mid-plane (cm)
HS = (0.002, 0.005, 0.02)  # finite-difference steps at the mid-plane (cm)
PRIMARY_H = 1  # index of the step the table reports
SEED_EDGE = 424242
SEED_FD = 100_000


def fan(spectrum, u=(-5.0, 5.0), v=(-5.0, 5.0)) -> SpectralBeamSource:
    """Return the demo fan (exact type: in-kernel routes) over a surface rectangle (cm)."""
    return SpectralBeamSource(
        spectrum,
        focal_point=(dao.CENTRE[0], dao.CENTRE[1], -dao.SAD),
        center=(dao.CENTRE[0] + 0.5 * sum(u), dao.CENTRE[1] + 0.5 * sum(v), 0.0),
        width_u=u[1] - u[0],
        width_v=v[1] - v[0],
    )


def edges(tips: np.ndarray):
    """(label, stencil, pair, side, base tips) of the 13 edges; base is one segment."""

    def base(s, mods=()):
        t = tips[s].copy()
        for pair, col, value in mods:
            t[pair, col] = value
        return t

    middle = 0.5 * (tips[3, 6, 0] + tips[3, 6, 1])
    return [
        ("open-neg", "central", 9, "neg", base(0)),
        ("open-pos", "central", 9, "pos", base(0)),
        ("near-closed-neg", "central", 16, "neg", base(0)),
        ("near-closed-pos", "central", 16, "pos", base(0)),
        ("closed-pos", "one-sided", 17, "pos", base(0)),
        ("closed-outer-neg", "one-sided", 0, "neg", base(1)),
        ("closed-ptv-row-pos", "one-sided", 6, "pos", base(3, [(6, 0, middle), (6, 1, middle)])),
        ("limit-pos", "central", 11, "pos", base(4, [(11, 1, LIMIT)])),
        ("near-limit-neg", "central", 8, "neg", base(4, [(8, 0, -2.55)])),
        ("seg5-pos", "central", 6, "pos", base(5)),
        ("seg7-neg", "central", 13, "neg", base(7)),
        ("seg9-pos", "central", 12, "pos", base(9)),
        ("seg5-near-closed-pos", "central", 3, "pos", base(5)),
    ]


def narrowed(spectrum, pair: int, side: str, x: float) -> tuple[SpectralBeamSource, float]:
    """Return the fan over the edge's strip (+0.5 mm) and blocked side, and the area ratio."""
    leaf = np.asarray(dao.LEAF_EDGES) / dao.TO_MID
    v = (max(leaf[pair] - 0.05, -5.0), min(leaf[pair + 1] + 0.05, 5.0))
    reach = (x + (1 if side == "pos" else -1) * -2 * max(HS)) / dao.TO_MID
    u = (max(reach - 0.3, -5.0), 5.0) if side == "pos" else (-5.0, min(reach + 0.3, 5.0))
    area = (u[1] - u[0]) * (v[1] - v[0]) / dao.FIELD**2
    return fan(spectrum, u, v), area


def stencils(kind: str, side: str, x: float):
    """Tip positions of the columns and, per h, (column indices, weights)."""
    positions, rows = [], []
    if kind == "central":
        for h in HS:
            rows.append(([len(positions), len(positions) + 1], np.array([-1.0, 1.0]) / (2 * h)))
            positions += [x - h, x + h]
    else:
        opening = 1.0 if side == "pos" else -1.0
        positions.append(x)
        for h in HS:
            rows.append(
                (
                    [0, len(positions), len(positions) + 1],
                    opening * np.array([-3.0, 4.0, -1.0]) / (2 * h),
                )
            )
            positions += [x + opening * h, x + 2 * opening * h]
    return positions, rows


def adjoints(engine, xs, spectrum, grid, tips, weights) -> dict[str, np.ndarray]:
    """Return the demo objective's dL/dD at the initial plan, and a = 1."""
    plan = SegmentedFanSource(
        fan(spectrum),
        dao.segment_stack(tips[0]),
        xs,
        tips_neg=tips[:, :, 0],
        tips_pos=tips[:, :, 1],
    )
    dij = engine.run_dij(
        plan, n_histories_per_beamlet=2_000_000, n_batches=10, seed=11, truncation=0.0
    )
    columns = np.array([dij.column_dense(s).reshape(grid.shape) for s in range(len(tips))])
    dose = np.tensordot(weights, columns, axes=1)
    fp = dao.FanPlan.__new__(dao.FanPlan)
    fp.structures = dao.make_structures(grid)
    fp.prescription = 1.1 * float(dose[fp.structures["ptv"]].mean())
    _, grad = fp.objective(dose)
    return {"real": grad, "simple": np.ones(grid.shape)}


def main() -> None:
    """Measure, print the table and the h study, render the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=12, help="FD replicate seeds")
    parser.add_argument("--fd-histories", type=int, default=1_000_000, help="per column")
    parser.add_argument("--edge-histories", type=int, default=1_000_000, help="per edge")
    parser.add_argument("--edge-batches", type=int, default=40)
    parser.add_argument(
        "--study", action="store_true", help="the decision record's 40 x 4e6 FD, 4e6 per edge"
    )
    args = parser.parse_args()
    if args.study:
        args.replicates, args.fd_histories, args.edge_histories = 40, 4_000_000, 4_000_000

    from pyradmc.backends.warp.engine import WarpEngine
    from pyradmc.data.tabulated.build import download_library
    from pyradmc.data.tabulated.precompile import compile_materials
    from pyradmc.data.tabulated.source import TabulatedCrossSections

    grid = dao.make_grid()
    print("compiling the tabulated table (EPICS cache/download) ...")
    data = compile_materials(
        download_library("epdl").read_text(encoding="latin-1"),
        download_library("eedl").read_text(encoding="latin-1"),
        e_max=7.0,
    )
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    engine = WarpEngine(grid=grid, cross_sections=xs, device="cuda:0")
    spectrum = ali_rogers_mv("varian-6mv")
    rng = np.random.default_rng(7)
    tips = dao.project(dao.initial_tips(rng))
    weights = rng.uniform(0.5, 1.5, dao.N_SEGMENTS)
    adj = adjoints(engine, xs, spectrum, grid, tips, weights)
    spec = edges(tips)

    # --- gradient: every edge a beamlet of one in-kernel contraction per adjoint ----
    full = fan(spectrum)
    sources = [
        FanEdgeDerivativeSource(full, dao.segment_stack(base), xs, EdgeParameter(0, side, pair))
        for _, _, pair, side, base in spec
    ]
    edge, t_edge = {}, 0.0
    for name, a in adj.items():
        t0 = time.perf_counter()
        edge[name] = engine.run_beamlet_contraction(
            EdgeGradientBeamletSource(sources),
            a,
            n_histories_per_beamlet=args.edge_histories,
            n_batches=args.edge_batches,
            seed=SEED_EDGE,
            correlated=False,
        )
        t_edge += time.perf_counter() - t0

    # --- forward: CRN finite differences on the narrowed fan, replicate seeds ---------
    fd: dict[str, dict[str, np.ndarray]] = {}
    t_fd = 0.0
    for e, (label, kind, pair, side, base) in enumerate(spec):
        col = 0 if side == "neg" else 1
        x = float(base[pair, col])
        beam, area = narrowed(spectrum, pair, side, x)
        positions, rows = stencils(kind, side, x)
        variants = np.repeat(base[None], len(positions), axis=0)
        variants[:, pair, col] = positions
        source = SegmentedFanSource(
            beam,
            dao.segment_stack(base),
            xs,
            tips_neg=variants[:, :, 0],
            tips_pos=variants[:, :, 1],
        )
        values = {name: np.empty((args.replicates, len(HS))) for name in adj}
        t0 = time.perf_counter()
        for r in range(args.replicates):
            dij = engine.run_dij(
                source,
                n_histories_per_beamlet=args.fd_histories,
                n_batches=1,
                seed=SEED_FD * (e + 1) + r,
                truncation=0.0,
                correlated=True,
            )
            cols = np.array([dij.column_dense(c).reshape(-1) for c in range(len(positions))])
            for name, a in adj.items():
                contracted = cols @ a.reshape(-1) * area
                values[name][r] = [w @ contracted[idx] for idx, w in rows]
        t_fd += time.perf_counter() - t0
        fd[label] = values
        print(f"  finite differences: {label} ({time.perf_counter() - t0:.1f} s)", flush=True)

    # --- table ------------------------------------------------------------------------
    labels = [s[0] for s in spec]
    for name in adj:
        print(f"\nadjoint '{name}', g_FD at h = {HS[PRIMARY_H]} cm (mid-plane)")
        print(f"{'edge':22s} {'g_FD':>21s} {'g_edge':>21s} {'z':>7s} {'ratio':>6s}")
        for j, label in enumerate(labels):
            g = fd[label][name][:, PRIMARY_H]
            m, s = g.mean(), g.std(ddof=1) / np.sqrt(g.size)
            ge, se = edge[name].values[j], edge[name].sigma[j]
            print(
                f"{label:22s} {m:+.4e} +- {s:.1e} {ge:+.4e} +- {se:.1e} "
                f"{(ge - m) / np.hypot(s, se):+7.2f} {ge / m:6.3f}"
            )
        print("h study (paired): (g_FD(h) - g_FD(0.005)) / g_FD(0.005)")
        for label in labels:
            g = fd[label][name]
            parts = []
            for k, h in enumerate(HS):
                if k == PRIMARY_H:
                    continue
                d = (g[:, k] - g[:, PRIMARY_H]) / abs(g[:, PRIMARY_H].mean())
                parts.append(f"h={h}: {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(d.size):.4f}")
            print(f"  {label:22s} " + "   ".join(parts))
    print(
        f"\ncost: edge contractions {t_edge:.1f} s, finite differences {t_fd:.1f} s "
        "(GPU possibly shared; rough)"
    )
    render(labels, edge, fd, Path(__file__).with_suffix(".png"))


def render(labels, edge, fd, out: Path) -> None:
    """Ratio g_edge / g_FD per edge (both adjoints) and the h study."""
    import matplotlib.pyplot as plt

    colors = {"pos": "#2a78d6", "neg": "#eb6834"}
    markers = {"pos": "o", "neg": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True, sharey=False)
    y = np.arange(len(labels))
    for ax, name in zip(axes[:2], ("real", "simple"), strict=True):
        for j, label in enumerate(labels):
            side = "neg" if label.endswith("neg") else "pos"
            g = fd[label][name][:, PRIMARY_H]
            m, s = g.mean(), g.std(ddof=1) / np.sqrt(g.size)
            ge, se = edge[name].values[j], edge[name].sigma[j]
            ratio = ge / m
            err = abs(ratio) * np.hypot(s / m, se / ge)
            ax.errorbar(ratio, j, xerr=err, fmt=markers[side], color=colors[side], ms=6, lw=1.5)
        ax.axvline(1.0, color="0.5", lw=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        title = "demo objective dL/dD" if name == "real" else "a = 1 (integral dose)"
        ax.set(title=f"g_edge / g_FD, {title}", xlabel="ratio (1 = consistent)")
        ax.grid(axis="x", color="0.9")
    axes[0].plot([], [], "o", color=colors["pos"], label="pos-side edge")
    axes[0].plot([], [], "s", color=colors["neg"], label="neg-side edge")
    axes[0].legend(loc="lower left", fontsize=8)
    ax = axes[2]
    for k, (h, mk) in enumerate(zip(HS, ("^", "o", "v"), strict=True)):
        if k == PRIMARY_H:
            continue
        for j, label in enumerate(labels):
            g = fd[label]["simple"]
            d = (g[:, k] - g[:, PRIMARY_H]) / abs(g[:, PRIMARY_H].mean())
            ax.errorbar(
                100 * d.mean(),
                j,
                xerr=100 * d.std(ddof=1) / np.sqrt(d.size),
                fmt=mk,
                color="#1baf7a" if k == 0 else "#6250d6",
                ms=6,
                lw=1.5,
                label=f"h = {h} cm" if j == 0 else None,
            )
    ax.axvline(0.0, color="0.5", lw=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(title="finite-difference step, a = 1", xlabel=f"g_FD(h) / g_FD({HS[PRIMARY_H]}) - 1 (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", color="0.9")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
