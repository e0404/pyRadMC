"""Fano in a field through the long exact-helix steps of the exterior-air path.

The oracle and recipe are those of :mod:`tests.validation.test_fano_magnetic` (Bouchard et
al. 2015, doi:10.1088/0031-9155/60/17/6639; Malkov & Rogers 2016,
doi:10.1118/1.4954318): an isotropic source uniform per unit mass, one medium, walls and a
gap at a thousandth of the density, the field parallel to the interfaces. A uniform
per-mass source gives the same dose per unit mass everywhere for any field, so any bias of
the long steps in where or how much energy lands reads as a gap-to-wall ratio off unity.

The gap here is sized so that the long steps are actually taken: 8 x 8 x 6 cm against the
2 cm gap of the base gates, so its core has several cm of safety. A cavity sealed by water
is not exterior by :func:`~pyradmc.geometry.exterior.exterior_region`, and it must stay
sealed for Fano to hold, so the region is declared with
:meth:`~pyradmc.geometry.exterior.ExteriorAirMap.from_region`, the same bound through the
same kernel path. One case bores a single-voxel chimney to the boundary instead, so that
:meth:`~pyradmc.geometry.exterior.ExteriorAirMap.from_grid` itself classifies the cavity.

**Gates, each with the base file's numbers.**

1. The field-free ratio is inside the base file's air-density gate (the phantom is new).
2. The map moves the ratio by less than ``FIELD_GATE`` (1 %) against the current stepping in
   the same field, over the whole cavity and over the voxels that start long steps. This is
   what isolates the new path: it is the only difference between the two runs.
3. **Power**: a safety map inflated by half lets chords run into the wall, and fails gate 2
   on the same statistics. That proves the path is taken on Warp and that this gate sees a
   safety defect; a gate that cannot fail is not a gate. Measured with the small-step
   chord put into the long steps by a scratch copy of the package, which no test can do to a
   compiled kernel: cavity -7 to -18 %, long-step voxels -10 to -47 %, so this gate also
   guards the chord on Warp, which the realistic comparison measurably does not.

**Field against field-free is reported and deliberately not gated here.** The base file's
gate 1 asks that of a 2 cm gap. On this 6 cm cavity the *current stepping* itself moves the
ratio by -0.8 % at 1 MeV and -1.3 to -1.4 % at 0.5 MeV (see docs/decisions.md), so the bar
cannot separate the two paths: the map moves it by -0.8 to -1.0 % and -1.4 %. That residual
of the current stepping is recorded as an open finding, not explained away here.

ECUT 0.05 MeV is the base file's test instrument, for the same variance reason.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pyradmc.backends.results import TransportResult
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.exterior import ExteriorAirMap, exterior_region
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from tests.conftest import SEED
from tests.fano_fixtures import FanoElectronSource, fano_phantom

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.validation, pytest.mark.warp, pytest.mark.gpu]

FIELD_GATE = 0.01  # same as test_fano_magnetic
FIELD_FREE_GATE = 0.03  # same as test_fano_magnetic at air density
AIR_LIKE = 1.0e-3
ECUT_INSTRUMENT = 0.05
GAP_CM = 6.0
NXY = 20
MARGIN_CM = 1.0
N_HISTORIES = 60_000_000
INFLATION = 1.5


def _phantom(*, chimney: bool = False) -> tuple[VoxelGrid, slice, slice, slice, slice, np.ndarray]:
    grid, core, upstream, gap, downstream = fano_phantom(
        GAP_CM, AIR_LIKE, nxy=NXY, margin_cm=MARGIN_CM
    )
    n_margin = round(MARGIN_CM / grid.spacing[0])
    region = np.zeros(grid.shape, dtype=bool)
    lateral = slice(n_margin, NXY - n_margin)
    region[lateral, lateral, gap] = True
    if not chimney:
        return grid, core, upstream, gap, downstream, region
    density = np.array(grid.density)
    mid_z = (gap.start + gap.stop) // 2
    density[:n_margin, NXY // 2, mid_z] = AIR_LIKE
    bored = VoxelGrid(
        shape=grid.shape,
        spacing=grid.spacing,
        density=density,
        material=np.full(grid.shape, WATER, dtype=np.int32),
    )
    return bored, core, upstream, gap, downstream, exterior_region(bored)


def _dose_and_sigma(result: TransportResult, mask: np.ndarray) -> tuple[float, float]:
    d = result.dose[mask]
    s = result.dose_sigma[mask]
    return float(d.mean()), float(np.sqrt((s * s).sum()) / d.size)


def _ratios(
    grid: VoxelGrid,
    core: slice,
    upstream: slice,
    gap: slice,
    downstream: slice,
    cavity_inner: np.ndarray,
    long_voxels: np.ndarray,
    energy: float,
    field: MagneticField | None,
    air: ExteriorAirMap | None,
) -> dict[str, float]:
    from pyradmc.backends.warp.engine import WarpEngine

    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine = WarpEngine(
        grid=grid, cross_sections=xs, device="cuda:0", magnetic_field=field, exterior_air=air
    )
    result = engine.run(
        FanoElectronSource(grid, energy),
        n_histories=N_HISTORIES,
        n_batches=10,
        seed=SEED,
        ecut=ECUT_INSTRUMENT,
    )
    engine.release_device_memory()
    wall_up = result.dose[core, core, upstream]
    wall_down = result.dose[core, core, downstream]
    s_up = float(np.sqrt((result.dose_sigma[core, core, upstream] ** 2).sum()) / wall_up.size)
    s_down = float(np.sqrt((result.dose_sigma[core, core, downstream] ** 2).sum()) / wall_down.size)
    wall = 0.5 * float(wall_up.mean() + wall_down.mean())
    s_wall = 0.5 * float(np.hypot(s_up, s_down))
    out: dict[str, float] = {"walls": float(wall_up.mean() / wall_down.mean())}
    for name, mask in (("cavity", cavity_inner), ("long", long_voxels)):
        dose, sigma = _dose_and_sigma(result, mask)
        out[name] = dose / wall
        out[f"{name}_sigma"] = (dose / wall) * float(np.hypot(sigma / dose, s_wall / wall))
    return out


def _report(label: str, r: dict[str, float]) -> str:
    return (
        f"{label}: cavity {r['cavity']:.4f} +- {r['cavity_sigma']:.4f}, "
        f"long-step voxels {r['long']:.4f} +- {r['long_sigma']:.4f}, walls {r['walls']:.4f}"
    )


def _case(chimney: bool = False):
    grid, core, upstream, gap, downstream, region = _phantom(chimney=chimney)
    air = ExteriorAirMap.from_grid(grid) if chimney else ExteriorAirMap.from_region(grid, region)
    n_margin = round(MARGIN_CM / grid.spacing[0])
    inner = np.zeros(grid.shape, dtype=bool)
    inner[n_margin + 1 : NXY - n_margin - 1, n_margin + 1 : NXY - n_margin - 1, gap] = True
    long_voxels = air.safety > 0.0
    assert long_voxels.sum() > 0.2 * inner.sum(), "too little of the cavity takes long steps"
    return grid, core, upstream, gap, downstream, inner, long_voxels, air


@pytest.fixture(scope="module", autouse=True)
def _needs_cuda() -> None:
    if not wp.is_cuda_available():
        pytest.skip("the Fano gates need ~1e8 electron histories per case; CUDA only")


@pytest.mark.parametrize("b_tesla", [0.35, 1.5])
@pytest.mark.parametrize("energy", [0.5, 1.0])
def test_long_steps_keep_the_fano_dose(energy: float, b_tesla: float) -> None:
    """Gates 1 and 2: the map moves the ratio by less than 1 % against current stepping."""
    grid, core, upstream, gap, downstream, inner, long_voxels, air = _case()
    args = (grid, core, upstream, gap, downstream, inner, long_voxels, energy)
    field = MagneticField.uniform((b_tesla, 0.0, 0.0))
    free = _ratios(*args, None, None)
    current = _ratios(*args, field, None)
    helix = _ratios(*args, field, air)
    text = " | ".join(
        (_report("0 T", free), _report(f"{b_tesla} T", current), _report("map", helix))
    )
    print(f"\n{energy} MeV: {text}")
    assert abs(free["cavity"] - 1.0) < FIELD_FREE_GATE, f"field-free: {text}"
    for r in (free, current, helix):
        assert abs(r["walls"] - 1.0) < FIELD_GATE, f"walls differ: {text}"
        assert r["cavity_sigma"] < FIELD_GATE / 3.0, f"statistics too poor: {text}"
        assert r["long_sigma"] < FIELD_GATE / 3.0, f"statistics too poor: {text}"
    for region in ("cavity", "long"):
        moved = helix[region] / current[region] - 1.0
        assert abs(moved) < FIELD_GATE, f"map moved {region} by {moved:+.4f}: {text}"


def test_the_grid_derived_region_keeps_the_fano_dose() -> None:
    """The same gate with the region classified by ``from_grid`` through a chimney."""
    grid, core, upstream, gap, downstream, inner, long_voxels, air = _case(chimney=True)
    args = (grid, core, upstream, gap, downstream, inner, long_voxels, 1.0)
    field = MagneticField.uniform((1.5, 0.0, 0.0))
    current = _ratios(*args, field, None)
    helix = _ratios(*args, field, air)
    text = f"{_report('1.5 T', current)} | {_report('map', helix)}"
    print(f"\nchimney: {text}")
    for region in ("cavity", "long"):
        moved = helix[region] / current[region] - 1.0
        assert abs(moved) < FIELD_GATE, f"map moved {region} by {moved:+.4f}: {text}"


def test_the_gate_catches_an_inflated_safety() -> None:
    """Gate 3, power: a map that lets chords run into the walls must fail on these numbers."""
    grid, core, upstream, gap, downstream, inner, long_voxels, air = _case()
    args = (grid, core, upstream, gap, downstream, inner, long_voxels, 1.0)
    field = MagneticField.uniform((1.5, 0.0, 0.0))
    current = _ratios(*args, field, None)
    inflated = dataclasses.replace(air, safety=air.safety * INFLATION)
    mutant = _ratios(*args, field, inflated)
    text = f"{_report('1.5 T', current)} | {_report('inflated', mutant)}"
    print(f"\nmutant: {text}")
    moved = max(abs(mutant[r] / current[r] - 1.0) for r in ("cavity", "long"))
    assert moved > FIELD_GATE, f"the gate did not see a {INFLATION}x safety map: {text}"
