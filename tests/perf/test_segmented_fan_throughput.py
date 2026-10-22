"""Segmented fan: all segments in one launch versus one pre-solve + run per segment.

The optimization-loop workload of the aperture-gradient DAO example: a 6 MV
Ali-Rogers fan, 10 x 10 cm at 100 cm, through a 20-pair focused-side MLC in 10
step-and-shoot segments, 400k histories per segment, electrons transported, on a
12 x 10 x 12 cm water phantom at 2.5 x 2.5 x 5 mm. Per iteration the old route
builds each segment's stack, pre-solves it on the device in attenuation mode and
transports it (20 launches with their own buffers, drains and syncs); the new
routes are one ``run_dij`` over a :class:`SegmentedFanSource` (per-segment doses)
and one ``run`` over its plan view (the MU-weighted sum). Both new routes start
from ``with_settings`` on new tips, as an optimizer would.

The MLC is dense *water* (analytic cross-sections): the transport cost does not
depend on the device material, and the tier stays EPDL-free.

Two baselines: the plain ten runs, and the DAO example's faster variant that shares
one host fan sample across the segments. Measured at adoption (2026-10-22, laptop
RTX 4070 at 2535 MHz, no throttle reason, GPU shared with other jobs; best of 7
interleaved): ten runs 1085 ms, shared ten runs 469 ms, one Dij 78 ms (6.0x the
shared baseline), one plan run 67 ms (7.0x).

Timings are interleaved, best of ``ROUNDS``, because the development GPU is
shared and drifts in power state (AGENTS.md section 8: only interleaved ratios are
trustworthy there; check ``nvidia-smi`` clocks before believing an absolute).
Alerts, not pass/fail — with one structural floor: one launch for all segments
must not be slower than ten.
"""

from __future__ import annotations

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
CENTRE = (6.0, 5.0)
FIELD = 10.0
N_PAIRS = 20
N_SEGMENTS = 10
N_PER_SEGMENT = 400_000
N_BATCHES = 2
MLC_Z = (50.0, 56.0)
TO_MID = 0.5 * sum(MLC_Z) / SAD
LEAF_EDGES = tuple(float(e) for e in np.linspace(-0.5, 0.5, N_PAIRS + 1) * FIELD * TO_MID)
ROUNDS = 5


def _engine():
    from pyradmc.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(48, 40, 24), spacing=(0.25, 0.25, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device="cuda:0")


def _fan() -> SpectralBeamSource:
    return SpectralBeamSource(
        ali_rogers_mv("varian-6mv"),
        focal_point=(CENTRE[0], CENTRE[1], -SAD),
        center=(CENTRE[0], CENTRE[1], 0.0),
        width_u=FIELD,
        width_v=FIELD,
    )


class _SharedPrimariesFan(SpectralBeamSource):
    """The DAO example's baseline trick: one host fan sample shared by every segment.

    The host sampling of the spectral fan was two thirds of each segment's
    pre-solve cost; memoizing it makes the ten-run baseline as fast as that route
    gets without changing what it computes (it only correlates the segments).
    """

    def sample_batch(self, seed: int, history_offset: int, n: int):
        """Return the cached batch for these arguments, sampling it once."""
        key = (seed, history_offset, n)
        cache = self.__dict__.setdefault("_batch_cache", {})
        if key not in cache:
            cache.clear()
            cache[key] = super().sample_batch(seed, history_offset, n)
        return cache[key]


def _shared_fan() -> _SharedPrimariesFan:
    fan = _fan()
    return _SharedPrimariesFan(fan.spectrum, fan.focal_point, fan.center, FIELD, FIELD)


def _tips(rng: np.random.Generator) -> np.ndarray:
    """(segments, pairs, 2) random openings at the mid-plane, like the DAO example."""
    half = 0.4 * FIELD * TO_MID
    a = rng.uniform(-half, half, (N_SEGMENTS, N_PAIRS, 2))
    a.sort(axis=-1)
    return a


def _stack(tips: np.ndarray) -> BeamLimitingStack:
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
                material=WATER,
                density=19.3,
                focused_sides=True,
            ),
        ),
    )


def _ten_runs(engine, fan, xs, tips, seed: int) -> list:
    """The route being replaced: per segment, build the stack, pre-solve, run."""
    from pyradmc.backends.warp.presolve import presolve_head_device

    doses = []
    for s in range(N_SEGMENTS):
        phase_space = presolve_head_device(
            fan,
            cross_sections=xs,
            n_histories=N_PER_SEGMENT,
            seed=seed,
            exit_z=MLC_Z[1] + 1.0,
            stack=_stack(tips[s]),
            mode="attenuation",
            device=engine.device,
            return_device_source=True,
        )
        doses.append(
            engine.run(
                phase_space, n_histories=N_PER_SEGMENT, n_batches=N_BATCHES, seed=seed + 1 + s
            ).dose
        )
    return doses


def _one_dij(engine, base: SegmentedFanSource, tips, seed: int):
    source = base.with_settings(tips_neg=tips[..., 0], tips_pos=tips[..., 1])
    return engine.run_dij(
        source, n_histories_per_beamlet=N_PER_SEGMENT, n_batches=N_BATCHES, seed=seed
    )


def _one_plan(engine, base: SegmentedFanSource, tips, weights, seed: int):
    plan = base.with_settings(tips_neg=tips[..., 0], tips_pos=tips[..., 1]).plan(weights)
    return engine.run(plan, n_histories=N_SEGMENTS * N_PER_SEGMENT, n_batches=N_BATCHES, seed=seed)


def test_one_launch_versus_ten_runs(benchmark) -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    engine = _engine()
    fan = _fan()
    shared = _shared_fan()
    xs = AnalyticCrossSections()
    rng = np.random.default_rng(SEED)
    tips = _tips(rng)
    weights = rng.uniform(0.5, 2.0, N_SEGMENTS)
    base = SegmentedFanSource(
        fan, _stack(tips[0]), xs, tips_neg=tips[..., 0], tips_pos=tips[..., 1]
    )
    # Warm every kernel and table cache on all three routes.
    _ten_runs(engine, fan, xs, tips, SEED)
    _ten_runs(engine, shared, xs, tips, SEED)
    _one_dij(engine, base, tips, SEED)
    _one_plan(engine, base, tips, weights, SEED)

    timings: dict[str, list[float]] = {
        "ten_runs": [],
        "ten_runs_shared": [],
        "one_dij": [],
        "one_plan": [],
    }
    for r in range(ROUNDS):
        tips = _tips(rng)  # a new iterate every round, as in an optimizer
        seed = SEED + r
        for name in timings:
            wp.synchronize_device("cuda:0")
            start = time.perf_counter()
            if name == "ten_runs":
                _ten_runs(engine, fan, xs, tips, seed)
            elif name == "ten_runs_shared":
                _ten_runs(engine, shared, xs, tips, seed)
            elif name == "one_dij":
                _one_dij(engine, base, tips, seed)
            else:
                _one_plan(engine, base, tips, weights, seed)
            wp.synchronize_device("cuda:0")
            timings[name].append(time.perf_counter() - start)

    best = {name: min(values) for name, values in timings.items()}
    benchmark.extra_info.update({f"best_{k}_s": v for k, v in best.items()})
    for route in ("one_dij", "one_plan"):
        benchmark.extra_info[f"speedup_{route}"] = best["ten_runs"] / best[route]
        benchmark.extra_info[f"speedup_{route}_vs_shared"] = best["ten_runs_shared"] / best[route]
    benchmark.pedantic(lambda: _one_dij(engine, base, tips, SEED), rounds=1, iterations=1)
    assert best["one_dij"] < best["ten_runs_shared"], (
        f"one launch ({best['one_dij']:.3f} s) is slower than ten runs "
        f"({best['ten_runs_shared']:.3f} s): the segmented route is structurally broken"
    )
