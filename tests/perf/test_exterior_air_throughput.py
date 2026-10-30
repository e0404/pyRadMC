"""Exterior-air path cost. Alerts, not pass/fail.

Current stepping against the exact-helix path on the synthetic body in 3 cm of air, recorded
via pytest-benchmark so a regression in either surfaces as baseline drift
(``pytest -m perf --benchmark-only``). The analytic source's water at air density keeps the
benchmark free of EPICS data; the body lies along y so the +z beam crosses the margin and the
body side on. The interleaved tabulated measurement behind any decision about the default is
in docs/decisions.md; check clocks before believing one (AGENTS.md section 8).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.exterior import ExteriorAirMap
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource
from tests.conftest import SEED
from tests.exterior_fixtures import body_grid

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.perf, pytest.mark.warp, pytest.mark.gpu]

N = 8_000_000


def _grid() -> VoxelGrid:
    """The body turned so that its long axis is y and z runs through air, body, air."""
    body = body_grid(3.0)
    density = np.swapaxes(np.asarray(body.density), 1, 2)
    return VoxelGrid(
        shape=density.shape,
        spacing=body.spacing,
        density=density,
        material=np.full(density.shape, WATER, dtype=np.int32),
    )


@pytest.mark.parametrize("path", ["current", "exterior"])
@pytest.mark.parametrize("b_tesla", [0.35, 1.5])
def test_body_in_air(benchmark, path: str, b_tesla: float) -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine = WarpEngine(
        grid=grid,
        cross_sections=xs,
        device="cuda:0",
        magnetic_field=MagneticField.uniform((b_tesla, 0.0, 0.0)),
        exterior_air=ExteriorAirMap.from_grid(grid) if path == "exterior" else None,
    )
    cx, cy = grid.shape[0] * grid.spacing[0] / 2.0, grid.shape[1] * grid.spacing[1] / 2.0
    beam = ParallelBeamSource(
        energy=6.0, z=-1.0, x_range=(cx - 5.0, cx + 5.0), y_range=(cy - 5.0, cy + 5.0)
    )
    engine.run(beam, n_histories=10_000, n_batches=1, seed=SEED)  # compile + warm
    benchmark.pedantic(
        lambda: engine.run(beam, n_histories=N, n_batches=4, seed=SEED),
        rounds=5,
        iterations=1,
    )
    engine.release_device_memory()
