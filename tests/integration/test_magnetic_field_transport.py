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
from tests.conftest import SEED, assert_chi2_consistent_batched

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


# --- Warp against the reference oracle, field on ------------------------------------


def _slab_grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 30), spacing=(2.0, 2.0, 0.2))


def _slab_photons() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0))


def _mask(ref, other) -> np.ndarray:
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (other.dose_sigma > 0.0)
    assert np.count_nonzero(mask) > 300, "mask too small to detect anything"
    return mask


@pytest.fixture(scope="module")
def ref_in_field() -> dict:
    """Reference runs in a transverse 1.5 T field, shared across devices."""
    grid = _slab_grid()
    engine = ReferenceEngine(
        grid=grid, cross_sections=_xs(grid), rng=HostRNG(), magnetic_field=TRANSVERSE
    )
    electrons = ParallelBeamSource(energy=5.0, z=-0.5, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
    fine = VoxelGrid.uniform_water(shape=(8, 8, 40), spacing=(2.0, 2.0, 0.1))
    fine_engine = ReferenceEngine(
        grid=fine, cross_sections=_xs(fine), rng=HostRNG(), magnetic_field=TRANSVERSE
    )
    return {
        "photons": engine.run(_slab_photons(), n_histories=4_000, n_batches=8, seed=SEED),
        "electrons": fine_engine.run(
            electrons, n_histories=1_200, n_batches=8, seed=SEED, primary_kind="electron"
        ),
        "electron_source": electrons,
        "fine": fine,
    }


@pytest.mark.warp
class TestWarpMatchesReferenceInField:
    """Chi-squared detection oracle (AGENTS.md 4) with the Lorentz step active."""

    def test_photon_beam_in_transverse_field(self, device: str, ref_in_field: dict) -> None:
        from pyradmc.backends.warp.engine import WarpEngine

        grid = _slab_grid()
        engine = WarpEngine(
            grid=grid, cross_sections=_xs(grid), device=device, magnetic_field=TRANSVERSE
        )
        result = engine.run(_slab_photons(), n_histories=16_000, n_batches=8, seed=SEED)
        ref = ref_in_field["photons"]
        assert_chi2_consistent_batched(
            ref.dose,
            ref.dose_sigma,
            ref.n_batches,
            result.dose,
            result.dose_sigma,
            result.n_batches,
            mask=_mask(ref, result),
        )

    def test_electron_beam_in_transverse_field(self, device: str, ref_in_field: dict) -> None:
        """Electron primaries exercise the bent loop directly, including positron-free exit."""
        from pyradmc.backends.warp.engine import WarpEngine

        fine = ref_in_field["fine"]
        engine = WarpEngine(
            grid=fine, cross_sections=_xs(fine), device=device, magnetic_field=TRANSVERSE
        )
        result = engine.run(
            ref_in_field["electron_source"],
            n_histories=4_800,
            n_batches=8,
            seed=SEED,
            primary_kind="electron",
        )
        ref = ref_in_field["electrons"]
        assert_chi2_consistent_batched(
            ref.dose,
            ref.dose_sigma,
            ref.n_batches,
            result.dose,
            result.dose_sigma,
            result.n_batches,
            mask=_mask(ref, result),
        )

    def test_field_moves_the_electron_beam_centroid(self, device: str, ref_in_field: dict) -> None:
        """The sign pin at engine level: B along +x bends a +z electron beam toward -y.

        F = q v x B with q = -e: v = +z, B = +x gives v x B = +y, so the electron force
        is toward -y. Checked on the dose centroid of a lateral profile, per device.
        """
        from pyradmc.backends.warp.engine import WarpEngine

        grid = VoxelGrid.uniform_water(shape=(4, 40, 24), spacing=(2.0, 0.2, 0.1))
        xs = _xs(grid)
        beam = ParallelBeamSource(energy=3.0, z=-0.5, x_range=(0.0, 8.0), y_range=(3.8, 4.2))
        off = WarpEngine(grid=grid, cross_sections=xs, device=device).run(
            beam, n_histories=4_000, n_batches=4, seed=SEED, primary_kind="electron"
        )
        on = WarpEngine(
            grid=grid,
            cross_sections=xs,
            device=device,
            magnetic_field=MagneticField.uniform((1.5, 0.0, 0.0)),
        ).run(beam, n_histories=4_000, n_batches=4, seed=SEED, primary_kind="electron")
        y = (np.arange(40) + 0.5) * 0.2
        deep = slice(8, 16)  # 0.8-1.6 cm depth, where the bend has had room to act

        def centroid(result) -> float:
            profile = result.dose[:, :, deep].sum(axis=(0, 2))
            return float((profile * y).sum() / profile.sum())

        assert abs(centroid(off) - 4.0) < 0.05
        assert centroid(on) < centroid(off) - 0.1


@pytest.mark.warp
class TestDijSchedulingStaysBitInertInField:
    def test_group_size_and_chunking_do_not_change_the_dij(self, device: str) -> None:
        from pyradmc.backends.warp.engine import WarpEngine

        grid = _grid()
        kwargs = dict(n_histories_per_beamlet=1_000, n_batches=2, seed=SEED)

        def dij(**engine_kwargs):
            engine = WarpEngine(
                grid=grid,
                cross_sections=_xs(grid),
                device=device,
                magnetic_field=TRANSVERSE,
                **engine_kwargs,
            )
            return engine.run_dij(_lattice(), **kwargs)

        base = dij().dose_csc().toarray()
        np.testing.assert_array_equal(base, dij(chunk_size=128).dose_csc().toarray())
        grouped = WarpEngine(
            grid=grid, cross_sections=_xs(grid), device=device, magnetic_field=TRANSVERSE
        ).run_dij(_lattice(), beamlet_group_size=1, **kwargs)
        np.testing.assert_array_equal(base, grouped.dose_csc().toarray())


# --- the electron return effect, qualitatively -------------------------------------
#
# Water / low-density gap / water, 6 MeV photons, B transverse. Electrons leaving the
# upstream water into the gap gyrate back (gyroradius ~3 mm at 1 MeV, 1.5 T) and
# redeposit on the water side: the last water voxels before the gap gain dose, the
# downstream water loses the electrons that used to cross and must rebuild from its
# own surface. The gap is water at air density because the analytic source is
# water-only; the effect is a density effect and the sign test does not care. The
# force on an electron with v along +z in B along +x is toward -y, so the -y column
# gains over the +y column.

GAP = slice(10, 20)  # z-voxels 2-4 cm
BEFORE = slice(7, 10)  # last 0.6 cm of upstream water
AFTER = slice(20, 23)  # first 0.6 cm of downstream water


def _ere_grid() -> VoxelGrid:
    shape = (3, 3, 30)
    density = np.ones(shape)
    density[:, :, GAP] = 1.205e-3
    from pyradmc.data.materials import WATER

    return VoxelGrid(
        shape=shape,
        spacing=(4.0, 4.0, 0.2),
        density=density,
        material=np.full(shape, WATER, dtype=np.int32),
    )


def _ere_beam() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(0.0, 12.0), y_range=(0.0, 12.0))


def _column(result, sl: slice) -> tuple[float, float]:
    """Central-axis dose summed over a depth slice, with its sigma."""
    d = result.dose[1, 1, sl]
    s = result.dose_sigma[1, 1, sl]
    return float(d.sum()), float(np.sqrt((s * s).sum()))


def _assert_electron_return_effect(off, on, *, lateral: bool) -> None:
    before_off, s_bo = _column(off, BEFORE)
    before_on, s_bn = _column(on, BEFORE)
    k = (before_on - before_off) / np.hypot(s_bo, s_bn)
    assert before_on > 1.3 * before_off, f"no ERE gain before the gap: {before_on / before_off:.2f}"
    assert k > 3.0, f"ERE gain only {k:.1f} sigma"
    after_off, _ = _column(off, AFTER)
    after_on, _ = _column(on, AFTER)
    assert after_on < 0.6 * after_off, (
        f"downstream water kept its electrons: {after_on / after_off:.2f}"
    )
    if not lateral:
        return
    # Lateral: the -y column gains over the +y column only with the field on. The
    # 4 cm columns dilute the push to a ~10 % asymmetry on Warp statistics, so the
    # pin is the field-on asymmetry relative to the field-off one.
    lo_on = on.dose[1, 0, :10].sum()
    hi_on = on.dose[1, 2, :10].sum()
    lo_off = off.dose[1, 0, :10].sum()
    hi_off = off.dose[1, 2, :10].sum()
    asymmetry_on = lo_on / hi_on
    asymmetry_off = lo_off / hi_off
    assert asymmetry_on / asymmetry_off > 1.15, (
        f"no lateral push toward -y: on {asymmetry_on:.3f} vs off {asymmetry_off:.3f}"
    )
    assert abs(asymmetry_off - 1.0) < 0.15, f"field-free asymmetry {asymmetry_off:.3f}"


def test_electron_return_effect_on_reference() -> None:
    grid = _ere_grid()
    xs = _xs(grid)
    kwargs = dict(n_histories=12_000, n_batches=8, seed=SEED)
    off = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(_ere_beam(), **kwargs)
    on = ReferenceEngine(
        grid=grid, cross_sections=xs, rng=HostRNG(), magnetic_field=TRANSVERSE
    ).run(_ere_beam(), **kwargs)
    assert on.energy_emitted == off.energy_emitted
    # The lateral pin needs Warp statistics: at 12k histories the field-free column
    # ratio itself wanders by tens of percent on the 4 cm columns.
    _assert_electron_return_effect(off, on, lateral=False)


@pytest.mark.warp
def test_electron_return_effect_on_warp(device: str) -> None:
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _ere_grid()
    xs = _xs(grid)
    kwargs = dict(n_histories=100_000, n_batches=8, seed=SEED)
    off = WarpEngine(grid=grid, cross_sections=xs, device=device).run(_ere_beam(), **kwargs)
    on = WarpEngine(grid=grid, cross_sections=xs, device=device, magnetic_field=TRANSVERSE).run(
        _ere_beam(), **kwargs
    )
    _assert_electron_return_effect(off, on, lateral=True)
