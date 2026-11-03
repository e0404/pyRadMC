"""Edge sets for an aperture gradient: vectorized build + upload, and the contraction.

The DAO example's gradient lists every leaf edge of every segment (10 segments x 20
pairs = 400 edges; an arc ~15 000). Built as one ``FanEdgeDerivativeSource`` per edge
that cost 0.13 ms per edge in Python plus a ~12 ms upload; ``SegmentedEdgeSource``
builds the same device columns with array arithmetic from a ``SegmentedFanSource``
(``with_settings`` on new tips, as an optimizer does). Recorded here: build + upload of
400 and of 15 000 edges (time and device bytes, alerting on non-linear scaling), and
one 400-edge contraction (2 000 histories per edge, 2 batches, electrons on) through
both routes, interleaved.

Measured at adoption (2026-10-22, laptop RTX 4070, clocks 2.5 GHz under load, no
throttle reason, GPU shared with another agent's jobs; best of 9 interleaved, EPICS
tungsten in the scratch probe, dense water here): build + upload 400 edges 1.4 ms host
(+0.7 ms of proposal kernels on the device) against 81 ms + 12 ms; 15 000 edges
~12 ms; contraction of 400 edges 34.5 -> 23.8 ms, build + contraction 118 -> 25 ms.

Alerts, not pass/fail (AGENTS.md 4), with one structural floor: the vectorized
build + upload must beat the per-edge build alone.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.aperture_gradient import (
    EdgeGradientBeamletSource,
    EdgeParameter,
    FanEdgeDerivativeSource,
)
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segment_edges import SegmentedEdgeSource
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.perf, pytest.mark.warp, pytest.mark.gpu]

SAD = 100.0
CENTRE = (6.0, 5.0)
FIELD = 10.0
MLC_Z = (50.0, 56.0)
TO_MID = 0.5 * sum(MLC_Z) / SAD
ROUNDS = 9
DEVICE = "cuda:0"


def _fan() -> SpectralBeamSource:
    return SpectralBeamSource(
        ali_rogers_mv("varian-6mv"),
        focal_point=(CENTRE[0], CENTRE[1], -SAD),
        center=(CENTRE[0], CENTRE[1], 0.0),
        width_u=FIELD,
        width_v=FIELD,
    )


def _tips(rng: np.random.Generator, n_segments: int, n_pairs: int) -> np.ndarray:
    half = 0.4 * FIELD * TO_MID
    tips = rng.uniform(-half, half, (n_segments, n_pairs, 2))
    tips.sort(axis=-1)
    return tips


def _stack(tips: np.ndarray) -> BeamLimitingStack:
    n_pairs = tips.shape[0]
    edges = np.linspace(-0.5, 0.5, n_pairs + 1) * FIELD * TO_MID
    return BeamLimitingStack(
        frame=BeamFrame(origin=(CENTRE[0], CENTRE[1], -SAD)),
        devices=(
            MLC(
                z_top=MLC_Z[0], z_bottom=MLC_Z[1], leaf_edges_v=tuple(edges),
                tips_neg=tuple(tips[:, 0]), tips_pos=tuple(tips[:, 1]), tip_radius=8.0,
                material=WATER, density=19.3, focused_sides=True,
            ),
        ),
    )  # fmt: skip


def _per_edge(fan, xs, tips: np.ndarray) -> EdgeGradientBeamletSource:
    edges = []
    for s in range(tips.shape[0]):
        stack = _stack(tips[s])
        for i in range(tips.shape[1]):
            for side in ("neg", "pos"):
                edges.append(FanEdgeDerivativeSource(fan, stack, xs, EdgeParameter(0, side, i)))
    return EdgeGradientBeamletSource(edges)


def _vectorized(base: SegmentedFanSource, tips: np.ndarray) -> SegmentedEdgeSource:
    return SegmentedEdgeSource(base.with_settings(tips_neg=tips[..., 0], tips_pos=tips[..., 1]))


def _best(fn, rounds: int = ROUNDS) -> float:
    out = []
    for _ in range(rounds):
        wp.synchronize_device(DEVICE)
        start = time.perf_counter()
        fn()
        wp.synchronize_device(DEVICE)
        out.append(time.perf_counter() - start)
    return min(out)


def test_edge_set_build_and_upload(benchmark) -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp import aperture

    fan = _fan()
    xs = AnalyticCrossSections()
    rng = np.random.default_rng(SEED)
    info: dict[str, float] = {}
    for n_segments, n_pairs in ((10, 20), (125, 60)):
        tips = _tips(rng, n_segments, n_pairs)
        base = SegmentedFanSource(
            fan, _stack(tips[0]), xs, tips_neg=tips[..., 0], tips_pos=tips[..., 1]
        )
        aperture.upload_edge_set(_vectorized(base, tips), DEVICE)  # warm caches and kernels
        n = 2 * n_segments * n_pairs
        info[f"build_upload_{n}_s"] = _best(
            lambda base=base, tips=tips: aperture.upload_edge_set(_vectorized(base, tips), DEVICE)
        )
        free = wp.get_device(DEVICE).free_memory
        held = aperture.upload_edge_set(_vectorized(base, tips), DEVICE)
        wp.synchronize_device(DEVICE)
        info[f"device_bytes_{n}"] = float(free - wp.get_device(DEVICE).free_memory)
        del held
    tips = _tips(rng, 10, 20)
    info["per_edge_build_400_s"] = _best(lambda: _per_edge(fan, xs, tips), rounds=3)
    per_edge = _per_edge(fan, xs, tips)
    info["per_edge_upload_400_s"] = _best(lambda: aperture.upload_edge_set(per_edge, DEVICE), 3)
    # Linear scaling: 37.5x the edges should cost well under ~40x the time.
    info["scaling_15000_over_400"] = info["build_upload_15000_s"] / info["build_upload_400_s"]
    benchmark.extra_info.update(info)
    benchmark.pedantic(
        lambda: aperture.upload_edge_set(_vectorized(base, _tips(rng, 125, 60)), DEVICE),
        rounds=1,
        iterations=1,
    )
    assert info["build_upload_400_s"] < info["per_edge_build_400_s"], info


def test_contraction_of_400_edges(benchmark) -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(48, 40, 24), spacing=(0.25, 0.25, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine = WarpEngine(grid=grid, cross_sections=xs, device=DEVICE)
    weights = np.random.default_rng(SEED).normal(size=grid.shape)
    fan = _fan()
    rng = np.random.default_rng(SEED)
    tips = _tips(rng, 10, 20)
    base = SegmentedFanSource(
        fan, _stack(tips[0]), xs, tips_neg=tips[..., 0], tips_pos=tips[..., 1]
    )
    kwargs = dict(n_histories_per_beamlet=2_000, n_batches=2)
    routes = {
        "per_edge": lambda t: _per_edge(fan, xs, t),
        "segmented": lambda t: _vectorized(base, t),
    }
    for build in routes.values():  # warm
        engine.run_beamlet_contraction(build(tips), weights, seed=SEED, **kwargs)
    timings: dict[str, list[float]] = {f"{k}_{p}": [] for k in routes for p in ("build", "run")}
    for r in range(ROUNDS):
        tips = _tips(rng, 10, 20)
        for name, build in routes.items():
            wp.synchronize_device(DEVICE)
            start = time.perf_counter()
            source = build(tips)
            middle = time.perf_counter()
            engine.run_beamlet_contraction(source, weights, seed=SEED + r, **kwargs)
            wp.synchronize_device(DEVICE)
            timings[f"{name}_build"].append(middle - start)
            timings[f"{name}_run"].append(time.perf_counter() - middle)
    best = {f"best_{k}_s": min(v) for k, v in timings.items()}
    benchmark.extra_info.update(best)
    benchmark.pedantic(
        lambda: engine.run_beamlet_contraction(
            _vectorized(base, tips), weights, seed=SEED, **kwargs
        ),
        rounds=1,
        iterations=1,
    )
