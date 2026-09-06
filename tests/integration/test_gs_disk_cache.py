"""Cache persistence leaves reference dose and its batch uncertainty unchanged."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = """
import math
import sys
from pathlib import Path
import numpy as np
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
import pyradmc.data.goudsmit_saunderson as gs
from pyradmc.data.interface import _GS_BINS_PER_LOG, _GS_TABLE_NODES
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PencilBeamSource
from pyradmc.rng.host import HostRNG

gs._GS_GRID_CACHE_DIR = Path(sys.argv[1])
# Test instrument: cheap node construction isolates persistence, not GS accuracy.
# Its separate construction key cannot read or pollute a production cache.
gs._RAW_SAMPLES = 512
if sys.argv[2] == 'uncached':
    def row(i, j):
        return gs.gs_scaled_deflection_table(
            math.exp(i / _GS_BINS_PER_LOG), math.exp(j / _GS_BINS_PER_LOG), n_u=_GS_TABLE_NODES
        )
    gs.gs_grid_nodes = lambda i, j: (row(i, j), row(i+1, j), row(i, j+1), row(i+1, j+1))
elif sys.argv[2] == 'read':
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError('a warm reference dose must not rebuild any node')
    gs._gs_raw_cosines = refuse
grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1., 1., 1.))
xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
result = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
    PencilBeamSource(0.8, (8., 8., 0.), (0., 0., 1.)),
    n_histories=24, n_batches=4, seed=20260711,
)
assert np.any(result.dose > 0)
assert np.any(result.dose_sigma > 0)
assert xs._gs_table_cache, 'the transport must actually sample GS nodes'
np.savez(sys.argv[3], dose=result.dose, sigma=result.dose_sigma)
"""


def test_fresh_process_cache_preserves_dose_bytes(tmp_path: Path) -> None:
    """Uncached, cold-disk and warm-disk transport agree within the same target."""
    cache = tmp_path / "cache"
    for mode in ("uncached", "build", "read"):
        completed = subprocess.run(
            [sys.executable, "-c", SCRIPT, str(cache), mode, str(tmp_path / f"{mode}.npz")],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        if mode == "uncached":
            assert not list(cache.glob("*.npz")), "the baseline must bypass disk persistence"
        else:
            assert list(cache.glob("*.npz"))
    with np.load(tmp_path / "uncached.npz") as before:
        for mode in ("build", "read"):
            with np.load(tmp_path / f"{mode}.npz") as after:
                for name in ("dose", "sigma"):
                    assert before[name].tobytes() == after[name].tobytes()
