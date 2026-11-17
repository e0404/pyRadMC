"""The ``"bhabha-in-flight"`` positron model on the Warp backend: kernel wiring.

The compiled samplers are pinned at the distribution level
(``tests/physics/test_warp_positron.py``) and the reference loop against interface
expectations (``test_positron_in_flight_ref.py``); these tests pin the mechanism:
the model's tables reach the device, the electron kernel compiled for the model is
the one launched, the energy books close on every run path with a positron's rest
mass booked once however it ends, a run is bit-reproducible within a device, and
the default path is the shared module's kernel with no model data. The
cross-backend dose comparison of the model, with its measured null and its
mutants, is ``test_pair_positron_cross_backend.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import pyradmc.data.positron as positron_data
from pyradmc import ELECTRON_MASS_MEV, PHOTON_ROULETTE_WEIGHT_CAP, PhysicsOptions
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, PencilBeamSource, Primary, Source
from pyradmc.physics_options import CLINICAL_PHYSICS
from pyradmc.rng import RNGState
from tests.conftest import SEED
from tests.integration.test_bethe_heitler_ref import _WideAnalyticWater

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]
IN_FLIGHT = PhysicsOptions(positron="bhabha-in-flight")
RESOLVED = dataclasses.replace(CLINICAL_PHYSICS, positron="bhabha-in-flight")
ENERGY_MEV = 18.0


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return str(request.param)


def _phantom() -> VoxelGrid:
    shape = (8, 8, 14)
    density = np.ones(shape, dtype=np.float64)
    density[:, :, 5:8] = 0.26
    return VoxelGrid(
        shape=shape,
        spacing=(1.0, 1.0, 1.0),
        density=density,
        material=np.zeros(shape, dtype=np.int32),
    )


def _engine(grid: VoxelGrid, device: str = "cpu", wide: bool = False):  # type: ignore[no-untyped-def]
    from pyradmc.backends.warp.engine import WarpEngine

    xs_type = _WideAnalyticWater if wide else AnalyticCrossSections
    return WarpEngine(
        grid=grid,
        cross_sections=xs_type(geometry_densities=grid.max_density_by_material()),
        device=device,
    )


def _beam(energy: float = ENERGY_MEV) -> PencilBeamSource:
    return PencilBeamSource(energy=energy, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))


def _beamlets() -> BeamletGridSource:
    return BeamletGridSource(
        energy=ENERGY_MEV, z=-1.0, x_range=(2.0, 6.0), y_range=(3.0, 5.0), n_x=2, n_y=1
    )


class _PositronPencil(Source):
    """A positron pencil along +z into the phantom, every primary at one weight."""

    particle_kinds = frozenset({"positron"})

    def __init__(self, energy: float = 12.0, weight: float = 1.0) -> None:
        self._energy = energy
        self._weight = weight

    @property
    def max_energy(self) -> float:
        return self._energy

    def emit(self, rng_state: RNGState) -> Primary:
        return Primary(
            self._energy, 4.0, 4.0, -1.0, 0.0, 0.0, 1.0, weight=self._weight, kind="positron"
        )


def _closes(result) -> None:  # type: ignore[no-untyped-def]
    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=1e-6)


# -- energy books and reproducibility ------------------------------------------------


@pytest.mark.parametrize("primaries", ["photon", "positron"])
def test_energy_ledger_closes(device: str, primaries: str) -> None:
    """Positrons from pair production, and positrons the source starts.

    A positron primary is booked as emitted with its kinetic energy and its
    2 m_e c^2; whether it then annihilates in flight, stops, or leaves the grid,
    that rest mass must come out exactly once for the books to close.
    """
    source = _beam() if primaries == "photon" else _PositronPencil()
    result = _engine(_phantom(), device).run(
        source, n_histories=8_000, n_batches=4, seed=SEED, physics=IN_FLIGHT
    )
    assert result.provenance is not None
    assert result.provenance.physics == RESOLVED
    _closes(result)
    if primaries == "positron":
        per_history = result.energy_emitted / 8_000
        assert per_history == pytest.approx(12.0 + 2.0 * ELECTRON_MASS_MEV, rel=1e-6)


def test_dose_to_water_ledger_closes(device: str) -> None:
    result = _engine(_phantom(), device).run(
        _PositronPencil(),
        n_histories=8_000,
        n_batches=4,
        seed=SEED,
        physics=IN_FLIGHT,
        scoring_mode="dose_to_water",
    )
    _closes(result)


@pytest.mark.parametrize("correlated", [True, False])
def test_dij_ledger_closes_and_records_the_model(device: str, correlated: bool) -> None:
    result = _engine(_phantom(), device).run_dij(
        _beamlets(),
        n_histories_per_beamlet=2_000,
        n_batches=4,
        seed=SEED,
        physics=IN_FLIGHT,
        correlated=correlated,
    )
    assert result.provenance is not None
    assert result.provenance.physics == RESOLVED
    assert result.dose_csc().nnz > 0
    _closes(result)


def test_device_dij_runs_the_model(device: str) -> None:
    """``run_dij_device`` launches the same model kernel as ``run_dij``: same bits."""
    engine = _engine(_phantom(), device)
    common = dict(n_histories_per_beamlet=2_000, n_batches=4, seed=SEED)
    host = engine.run_dij(_beamlets(), **common, physics=IN_FLIGHT)
    on_device = engine.run_dij_device(_beamlets(), **common, physics=IN_FLIGHT).to_host()
    default = engine.run_dij_device(_beamlets(), **common).to_host()

    dose = on_device.dose_csc()
    assert dose.nnz > 0
    assert on_device.provenance is not None
    assert on_device.provenance.physics == RESOLVED
    assert (dose != host.dose_csc()).nnz == 0
    assert (dose != default.dose_csc()).nnz > 0
    _closes(on_device)


def test_is_bit_reproducible_within_a_device(device: str) -> None:
    engine = _engine(_phantom(), device)
    runs = [
        engine.run(_PositronPencil(), n_histories=4_000, n_batches=4, seed=SEED, physics=IN_FLIGHT)
        for _ in range(2)
    ]
    assert runs[0].dose.max() > 0.0
    assert runs[0].dose.tobytes() == runs[1].dose.tobytes()
    assert runs[0].energy_escaped == runs[1].energy_escaped


def test_the_model_changes_the_transported_histories(device: str) -> None:
    engine = _engine(_phantom(), device)
    common = dict(n_histories=4_000, n_batches=4, seed=SEED)
    default = engine.run(_beam(), **common)
    named = engine.run(_beam(), **common, physics=PhysicsOptions(positron="moller-at-rest"))
    own = engine.run(_beam(), **common, physics=IN_FLIGHT)
    assert named.dose.tobytes() == default.dose.tobytes()  # "auto" is moller-at-rest
    assert own.dose.tobytes() != default.dose.tobytes()


def test_electrons_take_the_electron_arms_in_the_models_kernel(device: str) -> None:
    """Below the pair threshold no positron exists: the model's kernel gives the default's bits.

    The kernel compiled for the model carries the positron arm behind
    ``is_positron``; an electron in it must draw the same numbers and read the
    same tables as in the default kernel.
    """
    engine = _engine(_phantom(), device)
    beam = _beam(1.0)
    common = dict(n_histories=4_000, n_batches=4, seed=SEED, primary_kind="electron")
    default = engine.run(beam, **common)
    own = engine.run(beam, **common, physics=IN_FLIGHT)
    assert default.dose.max() > 0.0
    assert own.dose.tobytes() == default.dose.tobytes()


# -- one compiled electron kernel per model combination --------------------------------


class TestOneKernelPerModel:
    def test_default_kernel_is_the_shared_modules(self) -> None:
        from pyradmc.backends.warp import kernels

        default = kernels.make_electron_kernel(
            kernels.BREMS_THIN_TARGET, kernels.POSITRON_MOLLER_AT_REST
        )
        assert default is kernels.electron_kernel
        assert default is kernels.make_electron_kernel(kernels.BREMS_THIN_TARGET)
        assert default.module is wp.get_module(kernels.__name__)

    def test_every_other_combination_has_a_module_of_its_own(self) -> None:
        from pyradmc.backends.warp import kernels

        combos = [
            (kernels.BREMS_THIN_TARGET, kernels.POSITRON_BHABHA_IN_FLIGHT),
            (kernels.BREMS_BETHE_HEITLER, kernels.POSITRON_MOLLER_AT_REST),
            (kernels.BREMS_BETHE_HEITLER, kernels.POSITRON_BHABHA_IN_FLIGHT),
        ]
        compiled = [kernels.make_electron_kernel(*combo) for combo in combos]
        modules = {kernel.module for kernel in compiled} | {kernels.electron_kernel.module}
        assert len(modules) == 4
        for combo, kernel in zip(combos, compiled, strict=True):
            assert kernel is kernels.make_electron_kernel(*combo)
            assert kernel.module.options["fast_math"] is True
            assert kernel.module.options["enable_backward"] is False
            model_type = next(arg.type for arg in kernel.adj.args if arg.label == "model")
            assert model_type is kernels.ElectronModelTables

    def test_an_unknown_code_has_no_kernel(self) -> None:
        from pyradmc.backends.warp import kernels

        with pytest.raises(ValueError, match="no electron kernel for positron"):
            kernels.make_electron_kernel(kernels.BREMS_THIN_TARGET, 99)

    def test_the_tables_select_the_kernel(self) -> None:
        from pyradmc.backends.warp import kernels

        engine = _engine(_phantom())
        default = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)
        model = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)
        assert default.positron_model == kernels.POSITRON_MOLLER_AT_REST
        assert model.positron_model == kernels.POSITRON_BHABHA_IN_FLIGHT
        assert kernels.electron_kernel_for(default) is kernels.electron_kernel
        assert kernels.electron_kernel_for(model) is kernels.make_electron_kernel(
            kernels.BREMS_THIN_TARGET, kernels.POSITRON_BHABHA_IN_FLIGHT
        )
        # The positron model does not touch the photon kernel.
        assert kernels.photon_kernel_for(model) is kernels.photon_kernel

    def test_concurrent_lanes_run_the_model(self, device: str) -> None:
        engine = _engine(_phantom(), device)
        common = dict(n_histories=4_000, n_batches=4, seed=SEED, physics=IN_FLIGHT)
        serial = engine.run(_beam(), **common)
        lanes = engine.run(_beam(), **common, concurrent_batches=2)
        assert serial.dose.max() > 0.0
        assert lanes.dose.tobytes() == serial.dose.tobytes()
        dij = engine.run_dij(
            _beamlets(),
            n_histories_per_beamlet=1_000,
            n_batches=4,
            seed=SEED,
            physics=IN_FLIGHT,
            concurrent_batches=2,
        )
        _closes(dij)


# -- the model's tables --------------------------------------------------------------


class TestTables:
    def test_device_tables_are_the_host_queries(self) -> None:
        engine = _engine(_phantom())
        xs = engine.cross_sections
        tab = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)
        model = engine._model_tables_for(tab, "cpu")
        bhabha = model.bhabha.numpy()[0]
        annihilation = model.annihilation.numpy()[0]
        positron_range = model.positron_restricted_range.numpy()[0]
        assert model.positron_energy_of_restricted_range.shape == (1, tab.n_points)

        for j in (700, 1100, tab.n_points - 1):
            energy = float(np.exp(tab.e_log_e_min + j / tab.e_inv_dlog))
            assert energy > 0.25
            assert bhabha[j] == pytest.approx(xs.bhabha_cross_section(energy, 0, 0.2), rel=1e-5)
            assert annihilation[j] == pytest.approx(
                xs.annihilation_cross_section(energy, 0), rel=1e-5
            )
            assert positron_range[j] == pytest.approx(
                xs.positron_restricted_range(energy, 0, 0.2), rel=1e-5
            )
        # Its bremsstrahlung fields are placeholders: that model was not selected.
        assert model.brems_cross_section.shape == (1, 1)

    def test_a_default_run_builds_and_uploads_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[float] = []
        build = positron_data.build_positron_tables

        def counting(source, ecut, e_max, n_points=None):  # type: ignore[no-untyped-def]
            calls.append(e_max)
            return build(source, ecut, e_max, n_points)

        monkeypatch.setattr(positron_data, "build_positron_tables", counting)
        engine = _engine(_phantom())

        tab = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)
        assert calls == []
        assert engine._model_tables_for(tab, "cpu") == 0  # the default kernel's placeholder

        engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)
        engine._upload_tables(ENERGY_MEV, 0.02, 0.2, "cpu", physics=RESOLVED)
        assert len(calls) == 1  # independent of PCUT: one build per (ecut, e_max)

    def test_the_tables_struct_carries_no_model_arrays(self) -> None:
        from pyradmc.backends.warp import kernels

        assert not {
            "bhabha",
            "annihilation",
            "positron_restricted_range",
            "positron_energy_of_restricted_range",
        } & set(kernels.Tables.vars)

    def test_the_other_models_fields_are_placeholders(self) -> None:
        """A struct needs an array on every field; an unselected model's is one element."""
        engine = _engine(_phantom())
        brems_only = dataclasses.replace(CLINICAL_PHYSICS, bremsstrahlung="bethe-heitler")
        tab = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=brems_only)
        model = engine._model_tables_for(tab, "cpu")
        assert model.brems_cross_section.shape == (1, tab.n_points)
        assert model.bhabha.shape == (1, 1)
        assert model.positron_energy_of_restricted_range.shape == (1, 1)


# -- "auto" above the high-energy threshold ------------------------------------------


@pytest.mark.parametrize("energy", [40.0, 80.0, 200.0])
def test_auto_above_the_threshold_runs_the_model(device: str, energy: float) -> None:
    """Mechanics on the wide-domain instrument; nothing here is a validated dose."""
    grid = VoxelGrid.uniform_water(shape=(1, 1, 40), spacing=(60.0, 60.0, 2.0))
    beam = PencilBeamSource(energy=energy, position=(30.0, 30.0, -0.5), direction=(0, 0, 1.0))
    auto = _engine(grid, device, wide=True).run(beam, n_histories=4_000, n_batches=4, seed=SEED)

    assert auto.provenance is not None
    assert auto.provenance.physics.positron == "bhabha-in-flight"
    assert auto.provenance.physics.pair == "bethe-heitler"
    assert auto.provenance.physics.bremsstrahlung == "bethe-heitler"
    _closes(auto)


# -- signed weights ------------------------------------------------------------------


@pytest.mark.parametrize("weight", [0.5, PHOTON_ROULETTE_WEIGHT_CAP])
@pytest.mark.parametrize("bremsstrahlung", ["thin-target", "bethe-heitler"])
def test_a_negative_weight_history_mirrors_the_positive_one(
    device: str, weight: float, bremsstrahlung: str
) -> None:
    """Weight -w against +w on one stream: equal and opposite up to the quanta.

    As in ``test_bethe_heitler_warp.py``: only the fixed-point rounding differs, by
    at most one 1e-9 MeV quantum per deposit pair. An annihilation photon or a
    deposit dropped for its sign breaks the bound by orders of magnitude and
    opens the books.
    """
    n = 4_000
    engine = _engine(_phantom(), device)
    physics = PhysicsOptions(positron="bhabha-in-flight", bremsstrahlung=bremsstrahlung)
    common = dict(n_histories=n, n_batches=4, seed=SEED, physics=physics)
    plus = engine.run(_PositronPencil(weight=weight), **common)
    minus = engine.run(_PositronPencil(weight=-weight), **common)

    assert np.any(plus.dose > 0.0)
    mass = np.asarray(_phantom().density) * 1.0  # 1 cm^3 voxels
    residual_quanta = np.abs(minus.dose + plus.dose) * mass * n / 1.0e-9
    assert residual_quanta.sum() <= 10_000 * n, residual_quanta.sum() / n
    assert minus.energy_emitted < 0.0
    books = minus.energy_deposited + minus.energy_escaped + minus.energy_unscored
    assert books == pytest.approx(minus.energy_emitted, rel=1.0e-4)
