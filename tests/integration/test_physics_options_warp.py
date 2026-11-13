"""The ``physics`` argument on the Warp engine (AGENTS.md 2.10).

The companion of ``test_physics_options_ref.py``. Byte identity is asserted
*within* one device only (AGENTS.md 2.3): naming today's models explicitly gives
the same dose, bit for bit, as leaving every process on ``"auto"`` — on ``run``
and ``run_dij``, on every available device. The 10 MeV beam makes the pair,
bremsstrahlung and positron paths part of the transported histories.

The plumbing tests pin what the next model relies on: the resolved choice
reaches the transport kernels as model codes on the ``Tables`` struct, the
kernels' code table covers exactly the registered models, and a model that is
registered without kernel code is refused before any launch — a kernel cannot
raise, so the refusal has to be the engine's.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pyradmc import PhysicsOptions, physics_options
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, PencilBeamSource
from pyradmc.physics_options import CLINICAL_PHYSICS, PHYSICS_MODELS, ProcessModels
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]
EXPLICIT = PhysicsOptions(
    bremsstrahlung="thin-target", pair="uniform-split", positron="moller-at-rest"
)
ENERGY_MEV = 10.0
LOGGER = "pyradmc.backends.warp.engine"


def _engine(device: str = "cpu"):
    from pyradmc.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(8, 8, 12), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device=device)


def _beam() -> PencilBeamSource:
    return PencilBeamSource(energy=ENERGY_MEV, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))


def _beamlets() -> BeamletGridSource:
    return BeamletGridSource(
        energy=ENERGY_MEV, z=-1.0, x_range=(2.0, 6.0), y_range=(3.0, 5.0), n_x=2, n_y=1
    )


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return str(request.param)


class TestByteIdentityWithinADevice:
    @pytest.mark.parametrize("physics", ["auto", EXPLICIT])
    def test_run_matches_the_default(self, device: str, physics) -> None:
        engine = _engine(device)
        common = dict(n_histories=4_000, n_batches=4, seed=SEED)
        default = engine.run(_beam(), **common)
        result = engine.run(_beam(), **common, physics=physics)

        assert default.dose.max() > 0.0
        assert result.dose.tobytes() == default.dose.tobytes()
        assert result.dose_sigma.tobytes() == default.dose_sigma.tobytes()
        assert result.energy_deposited == default.energy_deposited
        assert result.energy_escaped == default.energy_escaped

    def test_run_dij_matches_the_default(self, device: str) -> None:
        engine = _engine(device)
        common = dict(n_histories_per_beamlet=2_000, n_batches=4, seed=SEED)
        default = engine.run_dij(_beamlets(), **common)
        explicit = engine.run_dij(_beamlets(), **common, physics=EXPLICIT)

        a, b = default.dose_csc(), explicit.dose_csc()
        assert a.nnz > 0
        assert a.data.tobytes() == b.data.tobytes()
        assert np.array_equal(a.indices, b.indices)
        assert np.array_equal(a.indptr, b.indptr)
        assert default.energy_escaped == explicit.energy_escaped

    def test_run_dij_device_matches_the_default(self, device: str) -> None:
        """The device hand-over is a transport path too: it takes ``physics``."""
        engine = _engine(device)
        common = dict(n_histories_per_beamlet=2_000, n_batches=4, seed=SEED)
        default = engine.run_dij_device(_beamlets(), **common).to_host()
        explicit = engine.run_dij_device(_beamlets(), **common, physics=EXPLICIT).to_host()

        assert default.dose.size > 0
        assert default.dose.tobytes() == explicit.dose.tobytes()
        assert default.sigma.tobytes() == explicit.sigma.tobytes()
        assert np.array_equal(default.indices, explicit.indices)
        assert np.array_equal(default.indptr, explicit.indptr)
        assert default.energy_escaped == explicit.energy_escaped
        assert explicit.provenance is not None
        assert explicit.provenance.physics == CLINICAL_PHYSICS


class TestResolutionIsRecorded:
    def test_run_logs_once_and_records_the_models(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            result = _engine().run(_beam(), n_histories=64, n_batches=2, seed=SEED)

        records = [r for r in caplog.records if "physics" in r.getMessage()]
        assert len(records) == 1
        assert "pair=uniform-split" in records[0].getMessage()
        assert "photon primaries up to 10 MeV" in records[0].getMessage()
        assert result.provenance is not None
        assert result.provenance.physics == CLINICAL_PHYSICS

    def test_run_dij_logs_once_and_records_the_models(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            result = _engine().run_dij(
                _beamlets(), n_histories_per_beamlet=32, n_batches=2, seed=SEED
            )

        assert sum("physics" in r.getMessage() for r in caplog.records) == 1
        assert result.provenance is not None
        assert result.provenance.physics == CLINICAL_PHYSICS

    def test_an_in_kernel_electron_beam_is_named(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            _engine().run(_beam(), n_histories=16, n_batches=2, seed=SEED, primary_kind="electron")

        assert "electron primaries" in caplog.text


class TestTheKernelsReceiveTheModels:
    def test_kernel_codes_cover_exactly_the_registered_models(self) -> None:
        """Registering a model without giving the kernels a code for it fails here."""
        from pyradmc.backends.warp.kernels import KERNEL_PHYSICS_CODES

        assert set(KERNEL_PHYSICS_CODES) == set(PHYSICS_MODELS)
        for process, row in PHYSICS_MODELS.items():
            assert set(KERNEL_PHYSICS_CODES[process]) == set(row.models)
            codes = list(KERNEL_PHYSICS_CODES[process].values())
            assert len(set(codes)) == len(codes)

    def test_tables_carry_the_resolved_model_codes(self) -> None:
        from pyradmc.backends.warp.kernels import KERNEL_PHYSICS_CODES

        tab = _engine()._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)

        assert tab.brems_model == KERNEL_PHYSICS_CODES["bremsstrahlung"]["thin-target"]
        assert tab.pair_model == KERNEL_PHYSICS_CODES["pair"]["uniform-split"]
        assert tab.positron_model == KERNEL_PHYSICS_CODES["positron"]["moller-at-rest"]

    def test_resident_tables_are_keyed_by_the_model_codes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The struct is resident and shared between runs; the codes are in its key."""
        from pyradmc.backends.warp import kernels

        engine = _engine()
        first = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)
        again = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)
        assert again is first

        codes = {process: dict(row) for process, row in kernels.KERNEL_PHYSICS_CODES.items()}
        codes["bremsstrahlung"]["thin-target"] = 7
        monkeypatch.setattr(kernels, "KERNEL_PHYSICS_CODES", codes)
        other = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)

        assert other is not first
        assert other.brems_model == 7
        assert first.brems_model == kernels.BREMS_THIN_TARGET
        assert other.mu_compton is first.mu_compton


class TestRefusals:
    def test_unknown_preset_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown physics preset"):
            _engine().run(_beam(), n_histories=4, n_batches=2, seed=SEED, physics="fancy")
        with pytest.raises(ValueError, match="unknown physics preset"):
            _engine().run_dij(
                _beamlets(), n_histories_per_beamlet=4, n_batches=2, seed=SEED, physics="fancy"
            )
        with pytest.raises(ValueError, match="unknown physics preset"):
            _engine().run_dij_device(
                _beamlets(), n_histories_per_beamlet=4, n_batches=2, seed=SEED, physics="fancy"
            )

    @pytest.mark.parametrize("process", ["bremsstrahlung", "pair", "positron"])
    def test_a_model_without_kernel_code_is_refused_before_launch(
        self, monkeypatch: pytest.MonkeyPatch, process: str
    ) -> None:
        table = dict(PHYSICS_MODELS)
        row = table[process]
        table[process] = ProcessModels(
            models=(*row.models, "host-only"),
            auto_clinical=row.auto_clinical,
            auto_high_energy=row.auto_high_energy,
        )
        monkeypatch.setattr(physics_options, "PHYSICS_MODELS", table)
        physics = PhysicsOptions(**{process: "host-only"})

        match = f"{process} model 'host-only'.*Warp kernels"
        with pytest.raises(NotImplementedError, match=match):
            _engine().run(_beam(), n_histories=4, n_batches=2, seed=SEED, physics=physics)
        with pytest.raises(NotImplementedError, match=match):
            _engine().run_dij(
                _beamlets(), n_histories_per_beamlet=4, n_batches=2, seed=SEED, physics=physics
            )
