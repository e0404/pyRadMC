"""The exterior-air path as the default: what ``auto`` does, and the caveat it carries.

Under a uniform non-zero field, an engine given no explicit choice derives
:meth:`~pyradmc.geometry.exterior.ExteriorAirMap.from_grid` itself. Pinned per target and never
across targets (AGENTS.md 2.3):

* ``auto`` is the same run as handing in the derived map, byte for byte.
* ``exterior_air=None`` is the explicit way to current stepping, and it warns nothing.
* Field-free, zero-field and non-uniform-field engines build no map, warn nothing, and are byte
  identical to ``exterior_air=None``.
* A grid with no usable exterior air warns nothing and records nothing: nothing ran.
* The caveat fires once per engine, however many runs it does, and is a result caveat
  (``warnings.warn``), so these tests expect it rather than filter it.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.exterior import EXTERIOR_AUTO, ExteriorAirMap
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

AIR = 1.205e-3
CAVEAT = "exterior-air"
ZERO = MagneticField.uniform((0.0, 0.0, 0.0))
UNIFORM = MagneticField.uniform((1.5, 0.0, 0.0))


def _grid() -> VoxelGrid:
    shape = (20, 20, 20)
    density = np.full(shape, AIR)
    density[5:15, 5:15, 5:15] = 1.0
    return VoxelGrid(
        shape=shape,
        spacing=(0.5, 0.5, 0.5),
        density=density,
        material=np.full(shape, WATER, dtype=np.int32),
    )


def _beam() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(3.0, 7.0), y_range=(3.0, 7.0))


def _nonuniform(grid: VoxelGrid) -> MagneticField:
    values = np.zeros((*grid.shape, 3))
    values[..., 0] = np.linspace(1.4, 1.6, grid.shape[2])
    return MagneticField.on_grid(grid, values)


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _ref(grid: VoxelGrid, field: MagneticField | None, **kwargs: Any) -> ReferenceEngine:
    return ReferenceEngine(
        grid=grid, cross_sections=_xs(grid), rng=HostRNG(), magnetic_field=field, **kwargs
    )


def _warp_factory(device: str):
    def build(grid: VoxelGrid, field: MagneticField | None, **kwargs: Any):
        from pyradmc.backends.warp.engine import WarpEngine

        return WarpEngine(
            grid=grid, cross_sections=_xs(grid), device=device, magnetic_field=field, **kwargs
        )

    return build


try:
    import warp as wp
except ModuleNotFoundError as exc:
    if exc.name != "warp":
        raise
    wp = None


@pytest.fixture(
    params=["ref", "cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)],
    ids=["ref", "warp-cpu", "warp-cuda"],
)
def target(request):
    """(engine builder, histories): the reference engine and Warp on each device."""
    if request.param == "ref":
        return _ref, 100
    if wp is None:
        pytest.skip("warp-lang optional dependency not installed")
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return _warp_factory(request.param), 4000


def _run(engine, n: int):
    return engine.run(_beam(), n_histories=n, n_batches=2, seed=SEED)


def test_auto_is_the_default_and_warns(target) -> None:
    build, _ = target
    with pytest.warns(UserWarning, match=CAVEAT):
        engine = build(_grid(), UNIFORM)
    assert engine.exterior_air is EXTERIOR_AUTO


def test_auto_is_the_derived_map_byte_for_byte(target) -> None:
    build, n = target
    grid = _grid()
    air = ExteriorAirMap.from_grid(grid)
    with pytest.warns(UserWarning, match=CAVEAT):
        auto = build(grid, UNIFORM)
    with pytest.warns(UserWarning, match=CAVEAT):
        explicit = build(grid, UNIFORM, exterior_air=air)
    a, b = _run(auto, n), _run(explicit, n)
    np.testing.assert_array_equal(a.dose, b.dose)
    assert a.energy_escaped == b.energy_escaped
    assert a.provenance is not None and a.provenance.exterior_air == air.provenance


def test_none_is_current_stepping_and_silent(target) -> None:
    build, n = target
    grid = _grid()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        off = build(grid, UNIFORM, exterior_air=None)
        result = _run(off, n)
    assert result.provenance is not None and result.provenance.exterior_air is None
    with pytest.warns(UserWarning, match=CAVEAT):
        on = build(grid, UNIFORM)
    assert not np.array_equal(result.dose, _run(on, n).dose), "auto changed nothing"


def test_the_caveat_fires_once_per_engine_not_per_run(target) -> None:
    build, n = target
    with pytest.warns(UserWarning, match=CAVEAT) as record:
        engine = build(_grid(), UNIFORM)
    assert len([w for w in record if CAVEAT in str(w.message)]) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(3):
            _run(engine, n)


@pytest.mark.parametrize("kind", ["none", "zero", "nonuniform"])
def test_no_uniform_field_means_no_map_no_warning_same_bytes(target, kind, monkeypatch) -> None:
    build, n = target
    grid = _grid()
    field = {"none": None, "zero": ZERO, "nonuniform": _nonuniform(grid)}[kind]
    baseline = _run(build(grid, field, exterior_air=None), n)

    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("a map was built for a run that cannot use it")

    monkeypatch.setattr(ExteriorAirMap, "from_grid", refuse)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        auto = _run(build(grid, field), n)
    np.testing.assert_array_equal(baseline.dose, auto.dose)
    assert baseline.energy_escaped == auto.energy_escaped
    assert auto.provenance is not None and auto.provenance.exterior_air is None


def test_a_grid_with_no_exterior_air_warns_nothing(target) -> None:
    """Nothing to step through, so no caveat and no provenance claim: nothing ran."""
    build, n = target
    grid = VoxelGrid.uniform_water(shape=(12, 12, 12), spacing=(0.5, 0.5, 0.5))
    baseline = _run(build(grid, UNIFORM, exterior_air=None), n)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        auto = _run(build(grid, UNIFORM), n)
    np.testing.assert_array_equal(baseline.dose, auto.dose)
    assert auto.provenance is not None and auto.provenance.exterior_air is None


def test_the_caveat_says_what_the_maintainer_asked() -> None:
    with pytest.warns(UserWarning, match=CAVEAT) as record:
        _ref(_grid(), UNIFORM)
    text = str(record[0].message)
    for needle in ("named approximation", "Fano", "not established", "exterior_air=None"):
        assert needle in text, f"caveat lost {needle!r}"
    assert "spatially resolved" in text


def test_the_caveat_points_at_the_caller() -> None:
    with pytest.warns(UserWarning, match=CAVEAT) as record:
        _ref(_grid(), UNIFORM)
    assert record[0].filename == __file__ or record[0].filename.endswith(
        "test_exterior_air_default.py"
    )
