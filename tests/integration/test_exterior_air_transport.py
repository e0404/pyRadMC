"""The exterior-air path as engine data: contracts on both loops, and a fast-tier twin check.

An :class:`~pyradmc.geometry.exterior.ExteriorAirMap` handed to an engine lets a charged
particle in a uniform field take exact-helix steps through air far from anything else.
Pinned here, per target and never across targets (AGENTS.md 2.3):

* **Field-free is untouched.** With no field, or an explicit zero field, the map is inert:
  byte-identical to a run without it, on ``run``, electron primaries and ``run_dij``.
* **A non-uniform field keeps current stepping**, byte for byte.
* **The path is taken**, proven by a counter on the reference loop and by changed bytes on
  Warp; a run through it is still bit-reproducible within a target.
* **The energy ledger stays exact** at each backend's documented tolerance.
* **Energy from a long step never lands in the body**, on the reference loop's deposits.
* **Warp and the reference agree** through the new path, at equal history counts.

The physical gates (Fano, the realistic thorax-like comparison and its mutant) are in the
validation tier; this file is contract and plumbing.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.exterior import ExteriorAirMap, exterior_region, resolve_exterior_air
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, ParallelBeamSource
from pyradmc.rng.host import HostRNG
from pyradmc.transport import electron as electron_module
from pyradmc.transport.electron import electron_steps
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.fano_fixtures import FanoElectronSource

AIR = 1.205e-3
ZERO = MagneticField.uniform((0.0, 0.0, 0.0))
TRANSVERSE = MagneticField.uniform((1.5, 0.0, 0.0))
LOW = MagneticField.uniform((0.35, 0.0, 0.0))
# The ledger tolerances the engines document: exact float64 on the reference, float32
# transport arithmetic plus scoring quantization on Warp.
LEDGER_REF = 1e-9
LEDGER_WARP = 1e-4


def _grid() -> VoxelGrid:
    """A water box in a 3 cm air margin, open to the grid on its z faces too."""
    shape = (24, 24, 24)
    density = np.full(shape, AIR)
    density[6:18, 6:18, 6:18] = 1.0
    density[10:14, 10:14, 10:14] = 0.26
    return VoxelGrid(
        shape=shape,
        spacing=(0.5, 0.5, 0.5),
        density=density,
        material=np.full(shape, WATER, dtype=np.int32),
    )


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _photons() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(3.0, 9.0), y_range=(3.0, 9.0))


def _air_electrons() -> FanoElectronSource:
    """Isotropic electrons born in the mid-margin, which the safety map covers.

    A beam from outside will not do at 1.5 T: it curls back out of its entry face within
    a gyroradius, in the band next to the grid boundary where the current stepping runs.
    The source's own grid only says where to emit; the transport grid is :func:`_grid`.
    """
    grid = _grid()
    weights = np.full(grid.shape, 1.0e-9)
    weights[np.asarray(ExteriorAirMap.from_grid(grid).safety) > 0.0] = 1.0
    emit = VoxelGrid(
        shape=grid.shape, spacing=grid.spacing, density=weights, material=grid.material
    )
    return FanoElectronSource(emit, 2.0)


def _lattice() -> BeamletGridSource:
    return BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(3.0, 9.0), y_range=(3.0, 9.0), n_x=2, n_y=1
    )


def _nonuniform(grid: VoxelGrid) -> MagneticField:
    values = np.zeros((*grid.shape, 3))
    values[..., 0] = np.linspace(1.4, 1.6, grid.shape[2])
    return MagneticField.on_grid(grid, values)


def _expecting_caveat(build, grid: VoxelGrid, field, air):
    """Build an engine; one that will use the path must warn, one that will not, must not."""
    if resolve_exterior_air(air, grid, field) is None:
        return build()
    with pytest.warns(UserWarning, match="exterior-air"):
        return build()


def _ref(grid: VoxelGrid, field: MagneticField | None, air: ExteriorAirMap | None):
    return _expecting_caveat(
        lambda: ReferenceEngine(
            grid=grid,
            cross_sections=_xs(grid),
            rng=HostRNG(),
            magnetic_field=field,
            exterior_air=air,
        ),
        grid,
        field,
        air,
    )


def _balance(result) -> float:
    books = result.energy_deposited + result.energy_unscored + result.energy_escaped
    return abs(result.energy_emitted - books) / result.energy_emitted


class _Counted:
    """Count the long-step chords and every substep of the reference loop."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.long_chords = 0
        real = electron_module.helix_chord

        def spy(*args: float) -> tuple[float, float, float]:
            self.long_chords += 1
            return real(*args)

        monkeypatch.setattr(electron_module, "helix_chord", spy)


# --- reference engine ---------------------------------------------------------------


class TestReference:
    def test_the_path_is_taken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid = _grid()
        count = _Counted(monkeypatch)
        _ref(grid, TRANSVERSE, ExteriorAirMap.from_grid(grid)).run(
            _air_electrons(), n_histories=100, n_batches=2, seed=SEED, primary_kind="electron"
        )
        assert count.long_chords > 200

    def test_no_map_never_takes_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid = _grid()
        count = _Counted(monkeypatch)
        _ref(grid, TRANSVERSE, None).run(
            _air_electrons(), n_histories=50, n_batches=2, seed=SEED, primary_kind="electron"
        )
        assert count.long_chords == 0

    def test_it_changes_the_transport(self) -> None:
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        kwargs: dict[str, Any] = dict(n_histories=100, n_batches=2, seed=SEED)
        kwargs["primary_kind"] = "electron"
        off = _ref(grid, TRANSVERSE, None).run(_air_electrons(), **kwargs)
        on = _ref(grid, TRANSVERSE, air).run(_air_electrons(), **kwargs)
        assert on.energy_escaped != off.energy_escaped

    @pytest.mark.parametrize("field", [None, ZERO], ids=["none", "zero"])
    def test_field_free_is_byte_identical_on_run(self, field) -> None:
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        kwargs: dict[str, Any] = dict(n_histories=200, n_batches=4, seed=SEED)
        plain = _ref(grid, field, None).run(_photons(), **kwargs)
        mapped = _ref(grid, field, air).run(_photons(), **kwargs)
        np.testing.assert_array_equal(plain.dose, mapped.dose)
        np.testing.assert_array_equal(plain.dose_sigma, mapped.dose_sigma)
        assert plain.energy_escaped == mapped.energy_escaped

    @pytest.mark.parametrize("field", [None, ZERO], ids=["none", "zero"])
    def test_field_free_is_byte_identical_for_electron_primaries(self, field) -> None:
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        kwargs: dict[str, Any] = dict(
            n_histories=100, n_batches=4, seed=SEED, primary_kind="electron"
        )
        plain = _ref(grid, field, None).run(_air_electrons(), **kwargs)
        mapped = _ref(grid, field, air).run(_air_electrons(), **kwargs)
        np.testing.assert_array_equal(plain.dose, mapped.dose)
        assert plain.energy_escaped == mapped.energy_escaped

    def test_field_free_is_byte_identical_on_run_dij(self) -> None:
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        kwargs: dict[str, Any] = dict(n_histories_per_beamlet=100, n_batches=2, seed=SEED)
        plain = _ref(grid, None, None).run_dij(_lattice(), **kwargs)
        mapped = _ref(grid, None, air).run_dij(_lattice(), **kwargs)
        np.testing.assert_array_equal(plain.dose_csc().toarray(), mapped.dose_csc().toarray())

    def test_a_nonuniform_field_keeps_current_stepping(self, monkeypatch) -> None:
        grid = _grid()
        field = _nonuniform(grid)
        count = _Counted(monkeypatch)
        kwargs: dict[str, Any] = dict(
            n_histories=50, n_batches=2, seed=SEED, primary_kind="electron"
        )
        plain = _ref(grid, field, None).run(_air_electrons(), **kwargs)
        mapped = _ref(grid, field, ExteriorAirMap.from_grid(grid)).run(_air_electrons(), **kwargs)
        np.testing.assert_array_equal(plain.dose, mapped.dose)
        assert count.long_chords == 0
        assert mapped.provenance is not None
        assert mapped.provenance.exterior_air is None

    def test_the_ledger_is_exact_through_the_path(self) -> None:
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        for source, kind in ((_photons(), "photon"), (_air_electrons(), "electron")):
            result = _ref(grid, TRANSVERSE, air).run(
                source, n_histories=150, n_batches=3, seed=SEED, primary_kind=kind
            )
            assert _balance(result) < LEDGER_REF, f"{kind}: ledger off by {_balance(result):.2e}"

    def test_provenance_records_what_ran(self) -> None:
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        kwargs: dict[str, Any] = dict(n_histories=8, n_batches=2, seed=SEED)
        on = _ref(grid, TRANSVERSE, air).run(_photons(), **kwargs)
        assert on.provenance is not None and on.provenance.exterior_air == air.provenance
        assert air.digest[:12] in on.provenance.summary()
        off = _ref(grid, TRANSVERSE, None).run(_photons(), **kwargs)
        assert off.provenance is not None and off.provenance.exterior_air is None
        # Field-free: the map is inert, so the record must not claim it ran.
        inert = _ref(grid, None, air).run(_photons(), **kwargs)
        assert inert.provenance is not None and inert.provenance.exterior_air is None
        dij = _ref(grid, TRANSVERSE, air).run_dij(
            _lattice(), n_histories_per_beamlet=4, n_batches=2, seed=SEED
        )
        assert dij.provenance is not None and dij.provenance.exterior_air == air.provenance

    def test_a_map_of_another_grid_is_refused(self) -> None:
        grid = _grid()
        other = VoxelGrid.uniform_water(shape=(24, 24, 24), spacing=(0.5, 0.5, 0.5))
        with pytest.raises(ValueError, match="exterior-air map"):
            _ref(grid, TRANSVERSE, ExteriorAirMap.from_grid(other))


def test_long_step_energy_lands_in_exterior_voxels_only() -> None:
    """The stated placement: a long step's every deposit is in the exterior region.

    Walks the loop directly with a deposit spy. ``csda_range`` is read exactly once per
    substep, so it opens a substep; the spied chord marks it long; a deposit filed while
    the mark stands must lie in an exterior voxel. The deposit after the second chord and
    a local brems photon's deposit are covered the same way.
    """
    grid = _grid()
    air = ExteriorAirMap.from_grid(grid)
    region = exterior_region(grid)
    xs = _xs(grid)
    long_step = {"on": False, "deposits": 0}
    real_chord = electron_module.helix_chord

    class Proxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(xs, name)

        def csda_range(self, e: float, material: int) -> float:
            long_step["on"] = False
            return xs.csda_range(e, material)

    def chord(*args: float) -> tuple[float, float, float]:
        long_step["on"] = True
        return real_chord(*args)

    def deposit(x: float, y: float, z: float, amount: float, scored: float) -> None:
        if long_step["on"]:
            long_step["deposits"] += 1
            assert region[grid.voxel_index(x, y, z)], f"long step deposited in body at {x, y, z}"

    rng = HostRNG()
    generator = np.random.default_rng(SEED)
    original = electron_module.helix_chord
    electron_module.helix_chord = chord  # type: ignore[assignment]
    try:
        for history in range(400):
            # Aim from the margin at the body: the awkward case for this claim.
            start = np.array([1.5 + generator.random(), *generator.uniform(1.0, 11.0, 2)])
            direction = generator.normal(size=3)
            direction[0] = abs(direction[0])
            direction /= np.linalg.norm(direction)
            stack: list[tuple] = []
            electron_steps(
                False,
                float(generator.uniform(0.3, 4.0)),
                1.0,
                *start,
                *direction,
                grid,
                Proxy(),  # type: ignore[arg-type]
                rng.init_state(SEED, history),
                deposit,
                stack.append,
                0.05,
                0.2,
                field=LOW if history % 2 else TRANSVERSE,
                exterior_safety=air.safety,
            )
    finally:
        electron_module.helix_chord = original  # type: ignore[assignment]
    assert long_step["deposits"] > 400


def test_deposit_resolution_spreads_a_long_step_along_its_chord() -> None:
    """``deposit_resolution_cm`` still only moves energy, and the ledger is untouched."""
    grid = _grid()
    air = ExteriorAirMap.from_grid(grid)
    kwargs: dict[str, Any] = dict(n_histories=100, n_batches=2, seed=SEED, primary_kind="electron")
    plain = _ref(grid, TRANSVERSE, air).run(_air_electrons(), **kwargs)
    fine = _ref(grid, TRANSVERSE, air).run(_air_electrons(), deposit_resolution_cm=0.1, **kwargs)
    assert fine.energy_escaped == plain.energy_escaped
    assert fine.energy_deposited == pytest.approx(plain.energy_deposited, rel=1e-12)
    assert _balance(fine) < LEDGER_REF


# --- Warp backend -------------------------------------------------------------------

try:
    import warp as wp
except ModuleNotFoundError as exc:
    if exc.name != "warp":
        raise
    wp = None

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if wp is None:
        pytest.skip("warp-lang optional dependency not installed")
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _warp(device: str, grid: VoxelGrid, field: MagneticField | None, air: ExteriorAirMap | None):
    from pyradmc.backends.warp.engine import WarpEngine

    return _expecting_caveat(
        lambda: WarpEngine(
            grid=grid,
            cross_sections=_xs(grid),
            device=device,
            magnetic_field=field,
            exterior_air=air,
        ),
        grid,
        field,
        air,
    )


@pytest.mark.warp
class TestWarp:
    @pytest.mark.parametrize("field", [None, ZERO], ids=["none", "zero"])
    def test_field_free_is_byte_identical_on_run(self, device: str, field) -> None:
        grid = _grid()
        kwargs: dict[str, Any] = dict(n_histories=4000, n_batches=4, seed=SEED)
        plain = _warp(device, grid, field, None).run(_photons(), **kwargs)
        mapped = _warp(device, grid, field, ExteriorAirMap.from_grid(grid)).run(
            _photons(), **kwargs
        )
        np.testing.assert_array_equal(plain.dose, mapped.dose)
        np.testing.assert_array_equal(plain.dose_sigma, mapped.dose_sigma)
        assert plain.energy_escaped == mapped.energy_escaped

    @pytest.mark.parametrize("field", [None, ZERO], ids=["none", "zero"])
    def test_field_free_is_byte_identical_for_electrons(self, device: str, field) -> None:
        grid = _grid()
        kwargs: dict[str, Any] = dict(
            n_histories=2000, n_batches=4, seed=SEED, primary_kind="electron"
        )
        plain = _warp(device, grid, field, None).run(_air_electrons(), **kwargs)
        mapped = _warp(device, grid, field, ExteriorAirMap.from_grid(grid)).run(
            _air_electrons(), **kwargs
        )
        np.testing.assert_array_equal(plain.dose, mapped.dose)

    def test_field_free_is_byte_identical_on_run_dij(self, device: str) -> None:
        grid = _grid()
        kwargs: dict[str, Any] = dict(n_histories_per_beamlet=2000, n_batches=2, seed=SEED)
        plain = _warp(device, grid, None, None).run_dij(_lattice(), **kwargs)
        mapped = _warp(device, grid, None, ExteriorAirMap.from_grid(grid)).run_dij(
            _lattice(), **kwargs
        )
        np.testing.assert_array_equal(plain.dose_csc().toarray(), mapped.dose_csc().toarray())

    def test_a_nonuniform_field_keeps_current_stepping(self, device: str) -> None:
        grid = _grid()
        field = _nonuniform(grid)
        kwargs: dict[str, Any] = dict(
            n_histories=1000, n_batches=2, seed=SEED, primary_kind="electron"
        )
        plain = _warp(device, grid, field, None).run(_air_electrons(), **kwargs)
        mapped = _warp(device, grid, field, ExteriorAirMap.from_grid(grid)).run(
            _air_electrons(), **kwargs
        )
        np.testing.assert_array_equal(plain.dose, mapped.dose)
        assert mapped.provenance is not None and mapped.provenance.exterior_air is None

    def test_the_path_is_taken_and_reproducible(self, device: str) -> None:
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        kwargs: dict[str, Any] = dict(
            n_histories=2000, n_batches=2, seed=SEED, primary_kind="electron"
        )
        off = _warp(device, grid, TRANSVERSE, None).run(_air_electrons(), **kwargs)
        on = _warp(device, grid, TRANSVERSE, air).run(_air_electrons(), **kwargs)
        again = _warp(device, grid, TRANSVERSE, air).run(_air_electrons(), **kwargs)
        assert not np.array_equal(off.dose, on.dose), "the map changed nothing on Warp"
        np.testing.assert_array_equal(on.dose, again.dose)

    def test_the_ledger_holds_through_the_path(self, device: str) -> None:
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        for source, kind in ((_photons(), "photon"), (_air_electrons(), "electron")):
            result = _warp(device, grid, TRANSVERSE, air).run(
                source, n_histories=8000, n_batches=4, seed=SEED, primary_kind=kind
            )
            assert _balance(result) < LEDGER_WARP, f"{kind}: ledger off by {_balance(result):.2e}"

    def test_provenance_records_what_ran(self, device: str) -> None:
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        kwargs: dict[str, Any] = dict(n_histories=64, n_batches=2, seed=SEED)
        on = _warp(device, grid, TRANSVERSE, air).run(_photons(), **kwargs)
        assert on.provenance is not None and on.provenance.exterior_air == air.provenance
        inert = _warp(device, grid, None, air).run(_photons(), **kwargs)
        assert inert.provenance is not None and inert.provenance.exterior_air is None

    def test_a_map_of_another_grid_is_refused(self, device: str) -> None:
        grid = _grid()
        other = VoxelGrid.uniform_water(shape=(24, 24, 24), spacing=(0.5, 0.5, 0.5))
        with pytest.raises(ValueError, match="exterior-air map"):
            _warp(device, grid, TRANSVERSE, ExteriorAirMap.from_grid(other))

    def test_a_dij_runs_through_the_path(self, device: str) -> None:
        """The Dij launches the same kernel: it takes the map and says so, reproducibly."""
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        kwargs: dict[str, Any] = dict(n_histories_per_beamlet=2000, n_batches=2, seed=SEED)
        first = _warp(device, grid, TRANSVERSE, air).run_dij(_lattice(), **kwargs)
        again = _warp(device, grid, TRANSVERSE, air).run_dij(_lattice(), **kwargs)
        np.testing.assert_array_equal(first.dose_csc().toarray(), again.dose_csc().toarray())
        assert first.provenance is not None and first.provenance.exterior_air == air.provenance

    def test_scheduling_stays_bit_inert_through_the_path(self, device: str) -> None:
        grid = _grid()
        air = ExteriorAirMap.from_grid(grid)
        from pyradmc.backends.warp.engine import WarpEngine

        kwargs: dict[str, Any] = dict(
            n_histories=4000, n_batches=2, seed=SEED, primary_kind="electron"
        )
        base = _warp(device, grid, TRANSVERSE, air).run(_air_electrons(), **kwargs)
        chunked = _expecting_caveat(
            lambda: WarpEngine(
                grid=grid,
                cross_sections=_xs(grid),
                device=device,
                magnetic_field=TRANSVERSE,
                exterior_air=air,
                chunk_size=512,
            ),
            grid,
            TRANSVERSE,
            air,
        ).run(_air_electrons(), **kwargs)
        np.testing.assert_array_equal(base.dose, chunked.dose)


# --- Warp against the reference oracle through the new path -------------------------
#
# Equal history counts on both sides: unequal counts miscalibrate the batched oracle on
# sparse voxels. The compared quantity is rebinned: energy per coarse block over the body
# and the margin, at replicate level, because air voxels are sparse and voxelwise batched
# chi^2 has no power there.

N_SEEDS = 6
N_HISTORIES = 1500
REBIN = 4


def _block_energy(result, grid: VoxelGrid) -> np.ndarray:
    energy = np.asarray(result.dose) * grid.density * grid.voxel_volume
    nx, ny, nz = energy.shape
    return energy.reshape(nx // REBIN, REBIN, ny // REBIN, REBIN, nz // REBIN, REBIN).sum(
        axis=(1, 3, 5)
    )


def _electron_replicates(run_one, grid: VoxelGrid, offset: int) -> np.ndarray:
    return np.array(
        [
            _block_energy(
                run_one(
                    _air_electrons(),
                    n_histories=N_HISTORIES,
                    n_batches=1,
                    seed=SEED + offset + k,
                    primary_kind="electron",
                ),
                grid,
            )
            for k in range(N_SEEDS)
        ]
    )


_REFERENCE_REPLICATES: dict[float, np.ndarray] = {}


def _reference_replicates(field: MagneticField, grid: VoxelGrid, air: ExteriorAirMap) -> np.ndarray:
    """The reference arm, shared by the devices: the oracle does not depend on them."""
    key = field.max_magnitude
    if key not in _REFERENCE_REPLICATES:
        _REFERENCE_REPLICATES[key] = _electron_replicates(_ref(grid, field, air).run, grid, 0)
    return _REFERENCE_REPLICATES[key]


@pytest.mark.warp
@pytest.mark.parametrize("field", [TRANSVERSE, LOW], ids=["1.5T", "0.35T"])
def test_warp_matches_the_reference_through_the_path(device: str, field) -> None:
    grid = _grid()
    air = ExteriorAirMap.from_grid(grid)
    ref = _reference_replicates(field, grid, air)
    warp = _electron_replicates(_warp(device, grid, field, air).run, grid, 1000)
    a, b = ref.mean(axis=0), warp.mean(axis=0)
    sa = ref.std(axis=0, ddof=1) / np.sqrt(N_SEEDS)
    sb = warp.std(axis=0, ddof=1) / np.sqrt(N_SEEDS)
    mask = (0.5 * (a + b) > 0.01 * (0.5 * (a + b)).max()) & (sa > 0.0) & (sb > 0.0)
    assert int(mask.sum()) >= 20, f"only {int(mask.sum())} blocks to compare"
    assert_chi2_consistent_batched(a, sa, N_SEEDS, b, sb, N_SEEDS, mask=mask)


def test_inflated_map_is_a_data_mutant() -> None:
    """The instrument the validation tier's power proof uses: a tampered map still loads."""
    grid = _grid()
    air = ExteriorAirMap.from_grid(grid)
    inflated = dataclasses.replace(air, safety=air.safety * 4.0)
    engine = _ref(grid, TRANSVERSE, inflated)
    assert engine.exterior_air is inflated
    assert inflated.matches(grid)
