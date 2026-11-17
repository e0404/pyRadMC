"""The ``"bethe-heitler"`` pair model on the Warp backend: kernel wiring.

The compiled samplers are pinned at the distribution level
(``tests/physics/test_warp_pair.py``) and the reference loop against interface
expectations (``test_pair_bethe_heitler_ref.py``); these tests pin the mechanism:
the model's screening scalars reach the device, the photon kernel compiled for the
model is the one launched, the energy books close on every run path, a run is
bit-reproducible within a device, and the default path is the shared module's
kernel with no model data. The cross-backend dose comparison of the model, with
its measured null and its mutants, is ``test_pair_positron_cross_backend.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pyradmc import PHOTON_ROULETTE_WEIGHT_CAP, PhysicsOptions
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
BETHE_HEITLER = PhysicsOptions(pair="bethe-heitler")
RESOLVED = dataclasses.replace(CLINICAL_PHYSICS, pair="bethe-heitler")
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


def _closes(result) -> None:  # type: ignore[no-untyped-def]
    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=1e-6)


# -- energy books and reproducibility ------------------------------------------------


def test_energy_ledger_closes(device: str) -> None:
    result = _engine(_phantom(), device).run(
        _beam(), n_histories=8_000, n_batches=4, seed=SEED, physics=BETHE_HEITLER
    )
    assert result.provenance is not None
    assert result.provenance.physics == RESOLVED
    _closes(result)


def test_dose_to_water_ledger_closes(device: str) -> None:
    result = _engine(_phantom(), device).run(
        _beam(),
        n_histories=8_000,
        n_batches=4,
        seed=SEED,
        physics=BETHE_HEITLER,
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
        physics=BETHE_HEITLER,
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
    host = engine.run_dij(_beamlets(), **common, physics=BETHE_HEITLER)
    on_device = engine.run_dij_device(_beamlets(), **common, physics=BETHE_HEITLER).to_host()
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
        engine.run(_beam(), n_histories=4_000, n_batches=4, seed=SEED, physics=BETHE_HEITLER)
        for _ in range(2)
    ]
    assert runs[0].dose.max() > 0.0
    assert runs[0].dose.tobytes() == runs[1].dose.tobytes()
    assert runs[0].energy_escaped == runs[1].energy_escaped


def test_the_model_changes_the_transported_histories(device: str) -> None:
    engine = _engine(_phantom(), device)
    common = dict(n_histories=4_000, n_batches=4, seed=SEED)
    default = engine.run(_beam(), **common)
    split = engine.run(_beam(), **common, physics=PhysicsOptions(pair="uniform-split"))
    shared = engine.run(_beam(), **common, physics=BETHE_HEITLER)
    assert split.dose.tobytes() == default.dose.tobytes()  # "auto" is the uniform split
    assert shared.dose.tobytes() != default.dose.tobytes()


def test_kerma_mode_does_not_read_the_model(device: str) -> None:
    engine = _engine(_phantom(), device)
    common = dict(n_histories=4_000, n_batches=4, seed=SEED, transport_electrons=False)
    split = engine.run(_beam(), **common)
    shared = engine.run(_beam(), **common, physics=BETHE_HEITLER)
    assert split.dose.tobytes() == shared.dose.tobytes()


# -- one compiled photon kernel per model --------------------------------------------


class TestOneKernelPerModel:
    """The pair model is a compile-time constant of the photon kernel (AGENTS.md 8)."""

    def test_default_kernel_is_the_shared_modules(self) -> None:
        from pyradmc.backends.warp import kernels

        default = kernels.make_photon_kernel(kernels.PAIR_UNIFORM_SPLIT)
        assert default is kernels.photon_kernel
        assert default.module is wp.get_module(kernels.__name__)

    def test_the_model_compiles_in_its_own_module(self) -> None:
        from pyradmc.backends.warp import kernels

        model = kernels.make_photon_kernel(kernels.PAIR_BETHE_HEITLER)
        assert model is kernels.make_photon_kernel(kernels.PAIR_BETHE_HEITLER)
        assert model is not kernels.photon_kernel
        assert model.module is not kernels.photon_kernel.module
        assert model.module.options["fast_math"] is True
        assert model.module.options["enable_backward"] is False

    def test_only_the_model_kernel_takes_model_data(self) -> None:
        from pyradmc.backends.warp import kernels

        def model_type(kernel):  # type: ignore[no-untyped-def]
            return next(arg.type for arg in kernel.adj.args if arg.label == "model")

        assert model_type(kernels.photon_kernel) is wp.int32
        model = kernels.make_photon_kernel(kernels.PAIR_BETHE_HEITLER)
        assert model_type(model) is kernels.PhotonModelTables

    def test_an_unknown_code_has_no_kernel(self) -> None:
        from pyradmc.backends.warp import kernels

        with pytest.raises(ValueError, match="no photon kernel"):
            kernels.make_photon_kernel(99)

    def test_the_tables_select_the_kernel(self) -> None:
        from pyradmc.backends.warp import kernels

        engine = _engine(_phantom())
        default = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)
        model = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)
        assert default.pair_model == kernels.PAIR_UNIFORM_SPLIT
        assert model.pair_model == kernels.PAIR_BETHE_HEITLER
        assert kernels.photon_kernel_for(default) is kernels.photon_kernel
        assert kernels.photon_kernel_for(model) is kernels.make_photon_kernel(
            kernels.PAIR_BETHE_HEITLER
        )
        # The pair model does not touch the electron kernel.
        assert kernels.electron_kernel_for(model) is kernels.electron_kernel

    def test_concurrent_lanes_run_the_model(self, device: str) -> None:
        """Lanes launch from host threads; the model's module is loaded before them."""
        engine = _engine(_phantom(), device)
        common = dict(n_histories=4_000, n_batches=4, seed=SEED, physics=BETHE_HEITLER)
        serial = engine.run(_beam(), **common)
        lanes = engine.run(_beam(), **common, concurrent_batches=2)
        assert serial.dose.max() > 0.0
        assert lanes.dose.tobytes() == serial.dose.tobytes()
        dij = engine.run_dij(
            _beamlets(),
            n_histories_per_beamlet=1_000,
            n_batches=4,
            seed=SEED,
            physics=BETHE_HEITLER,
            concurrent_batches=2,
        )
        _closes(dij)


class TestModelData:
    def test_the_screening_scalars_are_the_interface_values(self) -> None:
        engine = _engine(_phantom())
        tab = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)
        model = engine._photon_model_tables_for(tab, "cpu")
        screening = model.pair_screening.numpy()
        assert screening.shape == (engine.cross_sections.n_materials, 3)
        for material in range(engine.cross_sections.n_materials):
            assert screening[material] == pytest.approx(
                np.array(engine.cross_sections.pair_screening(material)), rel=1e-6
            )

    def test_the_default_path_carries_no_model_data(self) -> None:
        from pyradmc.backends.warp import kernels

        assert "pair_screening" not in kernels.Tables.vars
        engine = _engine(_phantom())
        tab = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)
        assert engine._photon_model_tables_for(tab, "cpu") == 0
        with pytest.raises(RuntimeError, match="without model data"):
            engine._photon_model_tables_for(kernels.Tables(), "cpu")


# -- "auto" above the high-energy threshold ------------------------------------------


@pytest.mark.parametrize("energy", [40.0, 80.0, 200.0])
def test_auto_above_the_threshold_runs_the_model(device: str, energy: float) -> None:
    """Mechanics on the wide-domain instrument; nothing here is a validated dose."""
    grid = VoxelGrid.uniform_water(shape=(1, 1, 40), spacing=(60.0, 60.0, 2.0))
    beam = PencilBeamSource(energy=energy, position=(30.0, 30.0, -0.5), direction=(0, 0, 1.0))
    common = dict(n_histories=4_000, n_batches=4, seed=SEED)
    auto = _engine(grid, device, wide=True).run(beam, **common)

    assert auto.provenance is not None
    assert auto.provenance.physics.pair == "bethe-heitler"
    _closes(auto)
    with pytest.raises(ValueError, match="energy domain"):
        _engine(grid, device).run(beam, **common)


# -- signed weights ------------------------------------------------------------------


class _SignedPhotonPencil(Source):
    """An 18 MeV photon pencil along +z, every primary at one signed weight."""

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def max_energy(self) -> float:
        return ENERGY_MEV

    def emit(self, rng_state: RNGState) -> Primary:
        return Primary(ENERGY_MEV, 4.0, 4.0, -1.0, 0.0, 0.0, 1.0, weight=self._weight)


@pytest.mark.parametrize("weight", [0.5, PHOTON_ROULETTE_WEIGHT_CAP])
def test_a_negative_weight_history_mirrors_the_positive_one(device: str, weight: float) -> None:
    """Weight -w against +w on one stream: equal and opposite up to the quanta.

    As in ``test_bethe_heitler_warp.py``: every amount of the negative run is the
    exact negative of its twin's, and only the fixed-point rounding differs, by at
    most one 1e-9 MeV quantum per deposit pair (bounded here by 10^4 deposits per
    18 MeV history). A lepton or deposit dropped for its sign breaks the bound by
    orders of magnitude and opens the books.
    """
    n = 4_000
    engine = _engine(_phantom(), device)
    common = dict(n_histories=n, n_batches=4, seed=SEED, physics=BETHE_HEITLER)
    plus = engine.run(_SignedPhotonPencil(weight), **common)
    minus = engine.run(_SignedPhotonPencil(-weight), **common)

    assert np.any(plus.dose > 0.0)
    mass = np.asarray(_phantom().density) * 1.0  # 1 cm^3 voxels
    residual_quanta = np.abs(minus.dose + plus.dose) * mass * n / 1.0e-9
    assert residual_quanta.sum() <= 10_000 * n, residual_quanta.sum() / n
    assert minus.energy_emitted < 0.0
    books = minus.energy_deposited + minus.energy_escaped + minus.energy_unscored
    assert books == pytest.approx(minus.energy_emitted, rel=1.0e-4)
