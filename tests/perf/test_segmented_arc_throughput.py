"""A VMAT arc's control points in one launch versus one source per gantry angle.

The forward workload of the VMAT arc demonstrator: a 6 MV Ali-Rogers fan, 9 x 5 cm
at the isocentre, through a 10-pair focused-side MLC at each of ``n`` gantry angles
(rotation about +y), 100k histories per control point, electrons transported, on a
16 x 6.4 x 16 cm water box at 2.5 x 4 x 2.5 mm. Per iteration the per-angle route
re-targets each angle's own :class:`SegmentedFanSource` (``with_settings``) and runs
its plan view, one launch per angle; the arc routes re-target one source with a fan
and a frame per control point and make **one** launch: ``run_dij`` (the per-angle
doses, as the per-angle route returns them) or ``run`` of the plan view (the
MU-weighted arc dose).

The MLC is dense *water* (analytic cross-sections): the transport cost does not
depend on the device material, and the tier stays EPDL-free. Timings are
interleaved, best of ``ROUNDS``, because the development GPU is shared and drifts in
power state (AGENTS.md section 8: check ``nvidia-smi`` clocks before believing an
absolute). Measured at adoption: docs/decisions.md, "Arcs in one launch". Alerts,
not pass/fail — with one structural floor: one launch for the arc must not be slower
than one launch per angle.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.perf, pytest.mark.warp, pytest.mark.gpu]

SAD = 100.0
SHAPE, SPACING = (64, 16, 64), (0.25, 0.4, 0.25)
ISO = tuple(0.5 * n * d for n, d in zip(SHAPE, SPACING, strict=True))
FIELD_U, FIELD_V = 9.0, 5.0
N_PAIRS = 10
MLC_Z = (50.0, 56.0)
TO_MID = 0.5 * sum(MLC_Z) / SAD
LEAF_EDGES = tuple(float(e) for e in np.linspace(-0.5, 0.5, N_PAIRS + 1) * FIELD_V * TO_MID)
N_PER_ANGLE = 100_000
N_BATCHES = 2
ROUNDS = {36: 5, 180: 3}


def _engine():
    from pyradmc.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=SHAPE, spacing=SPACING)
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device="cuda:0")


def _axes(angle: float):
    a = math.radians(angle)
    return (math.cos(a), 0.0, -math.sin(a)), (0.0, 1.0, 0.0), (math.sin(a), 0.0, math.cos(a))


def _focal(angle: float) -> tuple[float, float, float]:
    w = _axes(angle)[2]
    return (ISO[0] - SAD * w[0], ISO[1] - SAD * w[1], ISO[2] - SAD * w[2])


def _fan(spectrum, angle: float) -> SpectralBeamSource:
    u, v, _ = _axes(angle)
    return SpectralBeamSource(spectrum, _focal(angle), ISO, FIELD_U, FIELD_V, u_axis=u, v_axis=v)


def _frame(angle: float) -> BeamFrame:
    u, v, w = _axes(angle)
    return BeamFrame(origin=_focal(angle), u_axis=u, v_axis=v, w_axis=w)


def _template(angle: float) -> BeamLimitingStack:
    half = 0.5 * FIELD_U * TO_MID
    return BeamLimitingStack(
        frame=_frame(angle),
        devices=(
            MLC(
                z_top=MLC_Z[0],
                z_bottom=MLC_Z[1],
                leaf_edges_v=LEAF_EDGES,
                tips_neg=(-half,) * N_PAIRS,
                tips_pos=(half,) * N_PAIRS,
                tip_radius=8.0,
                material=WATER,
                density=19.3,
                focused_sides=True,
            ),
        ),
    )


def _tips(rng: np.random.Generator, n_angles: int) -> np.ndarray:
    """(angles, pairs, 2) random openings at the mid-plane, as an optimizer iterate."""
    half = 0.45 * FIELD_U * TO_MID
    a = rng.uniform(-half, half, (n_angles, N_PAIRS, 2))
    a.sort(axis=-1)
    return a


def _per_angle_runs(engine, sources, tips, seed: int) -> list:
    """The route being replaced: one plan run per angle, each re-targeted first."""
    doses = []
    for a, source in enumerate(sources):
        moved = source.with_settings(tips_neg=tips[a, None, :, 0], tips_pos=tips[a, None, :, 1])
        doses.append(
            engine.run(
                moved.plan([1.0]), n_histories=N_PER_ANGLE, n_batches=N_BATCHES, seed=seed + a
            ).dose
        )
    return doses


def _one_dij(engine, arc: SegmentedFanSource, tips, seed: int):
    source = arc.with_settings(tips_neg=tips[..., 0], tips_pos=tips[..., 1])
    return engine.run_dij(
        source, n_histories_per_beamlet=N_PER_ANGLE, n_batches=N_BATCHES, seed=seed
    )


def _one_plan(engine, arc: SegmentedFanSource, tips, weights, seed: int):
    plan = arc.with_settings(tips_neg=tips[..., 0], tips_pos=tips[..., 1]).plan(weights)
    n = N_PER_ANGLE * arc.n_segments
    return engine.run(plan, n_histories=n, n_batches=N_BATCHES, seed=seed)


@pytest.mark.parametrize("n_angles", [36, 180])
def test_one_arc_launch_versus_one_launch_per_angle(benchmark, n_angles: int) -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    engine = _engine()
    spectrum = ali_rogers_mv("varian-6mv")
    xs = AnalyticCrossSections()
    angles = np.arange(n_angles) * 360.0 / n_angles
    rng = np.random.default_rng(SEED)
    tips = _tips(rng, n_angles)
    weights = rng.uniform(0.5, 2.0, n_angles)
    template = _template(0.0)
    sources = [
        SegmentedFanSource(
            _fan(spectrum, a), _template(a), xs, tips_neg=tips[k, None, :, 0],
            tips_pos=tips[k, None, :, 1],
        )
        for k, a in enumerate(angles)
    ]  # fmt: skip
    arc = SegmentedFanSource(
        [_fan(spectrum, a) for a in angles],
        template,
        xs,
        frames=[_frame(a) for a in angles],
        tips_neg=tips[..., 0],
        tips_pos=tips[..., 1],
    )
    # Warm every kernel and table cache on all three routes.
    _per_angle_runs(engine, sources, tips, SEED)
    _one_dij(engine, arc, tips, SEED)
    _one_plan(engine, arc, tips, weights, SEED)

    timings: dict[str, list[float]] = {"per_angle_runs": [], "one_dij": [], "one_plan": []}
    for r in range(ROUNDS[n_angles]):
        tips = _tips(rng, n_angles)  # a new iterate every round, as in an optimizer
        seed = SEED + 1000 * r
        for name in timings:
            wp.synchronize_device("cuda:0")
            start = time.perf_counter()
            if name == "per_angle_runs":
                _per_angle_runs(engine, sources, tips, seed)
            elif name == "one_dij":
                _one_dij(engine, arc, tips, seed)
            else:
                _one_plan(engine, arc, tips, weights, seed)
            wp.synchronize_device("cuda:0")
            timings[name].append(time.perf_counter() - start)

    best = {name: min(values) for name, values in timings.items()}
    benchmark.extra_info.update({f"best_{k}_s": v for k, v in best.items()})
    for route in ("one_dij", "one_plan"):
        benchmark.extra_info[f"speedup_{route}"] = best["per_angle_runs"] / best[route]
    benchmark.pedantic(lambda: _one_plan(engine, arc, tips, weights, SEED), rounds=1, iterations=1)
    assert best["one_plan"] < best["per_angle_runs"], (
        f"one arc launch ({best['one_plan']:.3f} s) is slower than one launch per angle "
        f"({best['per_angle_runs']:.3f} s): the arc route is structurally broken"
    )
