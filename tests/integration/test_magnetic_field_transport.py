"""The magnetic field as engine data: plumbing, provenance and the field-free pin.

A :class:`~pyradmc.geometry.field.MagneticField` is world data handed to an engine at
construction, next to the grid and the cross-sections (AGENTS.md 2.10: physics is
data-driven, never flag-driven — there is no switch, only the presence or absence of
a field). Two guarantees are pinned here, per target and never across targets
(AGENTS.md 2.3):

* **No field is the archived path.** Every result produced before the field existed
  came from ``magnetic_field=None``; it must stay byte-identical. The cheapest
  standing witness is that an explicit *zero* field reproduces ``None`` bit for bit:
  the zero guard in the step loops skips every field operation and draws no random
  number, so the two runs share one stream.
* **The record says what ran.** ``RunProvenance.magnetic_field`` carries the field's
  provenance string, or ``None``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

ZERO = MagneticField.uniform((0.0, 0.0, 0.0))
TRANSVERSE = MagneticField.uniform((1.5, 0.0, 0.0))


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(6, 6, 12), spacing=(1.0, 1.0, 0.5))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _photons() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(1.0, 5.0), y_range=(1.0, 5.0))


def _electrons() -> ParallelBeamSource:
    return ParallelBeamSource(energy=4.0, z=-1.0, x_range=(2.0, 4.0), y_range=(2.0, 4.0))


def _lattice() -> BeamletGridSource:
    return BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(1.0, 5.0), y_range=(1.0, 5.0), n_x=2, n_y=1
    )


def _ref(field: MagneticField | None) -> ReferenceEngine:
    grid = _grid()
    return ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG(), magnetic_field=field)


def _too_small_map(grid: VoxelGrid) -> MagneticField:
    """A gridded map whose extent stops short of the transport grid's far face."""
    nx, ny, nz = grid.shape
    values = np.zeros((nx, ny, nz - 2, 3))
    values[..., 2] = 1.5
    return MagneticField(values=values, spacing=grid.spacing, origin=grid.origin)


# --- reference engine -------------------------------------------------------------


class TestReference:
    def test_zero_field_is_bit_identical_to_no_field_on_run(self) -> None:
        kwargs = dict(n_histories=400, n_batches=4, seed=SEED)
        none = _ref(None).run(_photons(), **kwargs)
        zero = _ref(ZERO).run(_photons(), **kwargs)
        np.testing.assert_array_equal(none.dose, zero.dose)
        np.testing.assert_array_equal(none.dose_sigma, zero.dose_sigma)
        assert none.energy_deposited == zero.energy_deposited
        assert none.energy_escaped == zero.energy_escaped
        assert none.energy_unscored == zero.energy_unscored

    def test_zero_field_is_bit_identical_for_electron_primaries(self) -> None:
        kwargs = dict(n_histories=200, n_batches=4, seed=SEED, primary_kind="electron")
        none = _ref(None).run(_electrons(), **kwargs)
        zero = _ref(ZERO).run(_electrons(), **kwargs)
        np.testing.assert_array_equal(none.dose, zero.dose)
        assert none.energy_escaped == zero.energy_escaped

    def test_zero_field_is_bit_identical_on_run_dij(self) -> None:
        kwargs = dict(n_histories_per_beamlet=200, n_batches=2, seed=SEED)
        none = _ref(None).run_dij(_lattice(), **kwargs)
        zero = _ref(ZERO).run_dij(_lattice(), **kwargs)
        np.testing.assert_array_equal(none.dose_csc().toarray(), zero.dose_csc().toarray())
        np.testing.assert_array_equal(none.sigma_csc().toarray(), zero.sigma_csc().toarray())

    def test_provenance_records_the_field(self) -> None:
        kwargs = dict(n_histories=8, n_batches=2, seed=SEED)
        none = _ref(None).run(_photons(), **kwargs)
        on = _ref(TRANSVERSE).run(_photons(), **kwargs)
        assert none.provenance is not None and none.provenance.magnetic_field is None
        assert on.provenance is not None
        assert on.provenance.magnetic_field == TRANSVERSE.provenance
        assert "1.5" in on.provenance.summary()
        dij = _ref(TRANSVERSE).run_dij(
            _lattice(), n_histories_per_beamlet=4, n_batches=2, seed=SEED
        )
        assert dij.provenance is not None
        assert dij.provenance.magnetic_field == TRANSVERSE.provenance

    def test_a_map_that_does_not_cover_the_grid_is_refused(self) -> None:
        grid = _grid()
        with pytest.raises(ValueError, match="cover"):
            ReferenceEngine(
                grid=grid,
                cross_sections=_xs(grid),
                rng=HostRNG(),
                magnetic_field=_too_small_map(grid),
            )


# --- Warp backend -------------------------------------------------------------------

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _warp(device: str, field: MagneticField | None):
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device, magnetic_field=field)


@pytest.mark.warp
class TestWarp:
    def test_zero_field_is_bit_identical_to_no_field_on_run(self, device: str) -> None:
        kwargs = dict(n_histories=4000, n_batches=4, seed=SEED)
        none = _warp(device, None).run(_photons(), **kwargs)
        zero = _warp(device, ZERO).run(_photons(), **kwargs)
        np.testing.assert_array_equal(none.dose, zero.dose)
        np.testing.assert_array_equal(none.dose_sigma, zero.dose_sigma)
        assert none.energy_deposited == zero.energy_deposited
        assert none.energy_escaped == zero.energy_escaped

    def test_zero_field_is_bit_identical_for_electron_primaries(self, device: str) -> None:
        kwargs = dict(n_histories=2000, n_batches=4, seed=SEED, primary_kind="electron")
        none = _warp(device, None).run(_electrons(), **kwargs)
        zero = _warp(device, ZERO).run(_electrons(), **kwargs)
        np.testing.assert_array_equal(none.dose, zero.dose)
        assert none.energy_escaped == zero.energy_escaped

    def test_zero_field_is_bit_identical_on_run_dij(self, device: str) -> None:
        kwargs = dict(n_histories_per_beamlet=2000, n_batches=2, seed=SEED)
        none = _warp(device, None).run_dij(_lattice(), **kwargs)
        zero = _warp(device, ZERO).run_dij(_lattice(), **kwargs)
        np.testing.assert_array_equal(none.dose_csc().toarray(), zero.dose_csc().toarray())
        np.testing.assert_array_equal(none.sigma_csc().toarray(), zero.sigma_csc().toarray())

    def test_provenance_records_the_field(self, device: str) -> None:
        kwargs = dict(n_histories=64, n_batches=2, seed=SEED)
        none = _warp(device, None).run(_photons(), **kwargs)
        on = _warp(device, TRANSVERSE).run(_photons(), **kwargs)
        assert none.provenance is not None and none.provenance.magnetic_field is None
        assert on.provenance is not None
        assert on.provenance.magnetic_field == TRANSVERSE.provenance

    def test_a_map_that_does_not_cover_the_grid_is_refused(self, device: str) -> None:
        from pyradmc.backends.warp.engine import WarpEngine

        grid = _grid()
        with pytest.raises(ValueError, match="cover"):
            WarpEngine(
                grid=grid,
                cross_sections=_xs(grid),
                device=device,
                magnetic_field=_too_small_map(grid),
            )
