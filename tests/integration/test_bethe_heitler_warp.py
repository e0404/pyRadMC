"""The ``"bethe-heitler"`` bremsstrahlung model on the Warp backend, against the oracle.

The compiled samplers are pinned at the distribution level
(``tests/physics/test_warp_bethe_heitler.py``) and the reference loop against
analytic expectations (``test_bethe_heitler_ref.py``); these tests pin the kernel
*wiring*: the model's tables reach the device, the energy books close, a run is
bit-reproducible within a device, and the dose agrees with the reference backend.

The cross-backend comparison (AGENTS.md 2.3: chi-squared, never equality) is set up
so that the oracle's null is measured before its verdict is trusted:

- **Equal history counts** on both sides.
- **A geometry whose bins are independent enough for a chi-squared.** A 15 MeV
  electron pencil beam into a single lateral column, binned in depth, with a
  lung-density slab so boundary-truncated substeps are part of the path. Every
  history crosses every bin, so the bins carry about one percent each — no sparse
  voxels — and there is no primary-fluence fluctuation to correlate them.
- **A measured null.** With both sides the *same* backend (1000 pairs of
  independent Warp runs at this size, geometry and mask; develop b858f86 plus
  this model, 2026-11-14) the statistic has the right mean (chi2 over its null
  mean: 1.008, sd 0.44) but a heavier upper tail than the oracle's normal
  approximation assumes at 15 bins: 5.1 % of pairs fall below a nominal p of
  0.01 and 9.6 % below 0.05; the measured 95 % and 99 % quantiles of the ratio
  are 1.90 and 2.45. The comparison below sits at a ratio of 1.28 (nominal
  p = 0.23, empirical p = 0.20), and the two mutants at 8.5 and 79, beyond every
  one of the 1000 null pairs. So the oracle is anticonservative here by a known
  factor, and the verdicts are not near its threshold. On a voxel lattice the
  same measurement is worse (100 pairs: 6 % rejections at nominal 0.01 for a
  broad beam, 16 % for a pencil beam): voxels of one column rise and fall
  together with the number of primaries that entered it.
- **A mutant the oracle must reject.** The thin-target model on one side is *not*
  that mutant — measured, the two models' depth doses differ by less than this
  sample resolves (ratio 1.31, empirical p = 0.18) — so the mutants are the two
  wiring errors the split of the radiative loss invites: charging the total
  radiative stopping power on top of the discrete photons, and dropping the
  discrete channel.

Both backends run the Gaussian hinge here: the reference backend builds its
Goudsmit-Saunderson tables lazily, which at 15 MeV costs minutes, and the
bremsstrahlung arm does not depend on the angular law.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import pyradmc.data.bethe_heitler as bethe_heitler_data
from pyradmc import PHOTON_ROULETTE_WEIGHT_CAP, PhysicsOptions
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.bethe_heitler import BetheHeitlerTables, build_bethe_heitler_tables
from pyradmc.data.interface import CrossSectionSource
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, PencilBeamSource, Primary, Source
from pyradmc.physics_options import CLINICAL_PHYSICS
from pyradmc.rng import RNGState
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]
BETHE_HEITLER = PhysicsOptions(bremsstrahlung="bethe-heitler")
RESOLVED = dataclasses.replace(CLINICAL_PHYSICS, bremsstrahlung="bethe-heitler")

COLUMN_SHAPE = (1, 1, 24)
COLUMN_SPACING = (10.0, 10.0, 0.5)
LUNG = slice(6, 10)  # 3 to 5 cm depth at lung-like density
ENERGY_MEV = 15.0
N_HISTORIES = 3_000
N_BATCHES = 20


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return str(request.param)


def _column() -> VoxelGrid:
    density = np.ones(COLUMN_SHAPE, dtype=np.float64)
    density[:, :, LUNG] = 0.26
    return VoxelGrid(
        shape=COLUMN_SHAPE,
        spacing=COLUMN_SPACING,
        density=density,
        material=np.zeros(COLUMN_SHAPE, dtype=np.int32),
    )


def _pencil() -> PencilBeamSource:
    return PencilBeamSource(energy=ENERGY_MEV, position=(5.0, 5.0, -0.5), direction=(0, 0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _warp_engine(grid: VoxelGrid, device: str = "cpu", wide: bool = False):  # type: ignore[no-untyped-def]
    from pyradmc.backends.warp.engine import WarpEngine

    xs = (
        _WideAnalyticWater(geometry_densities=grid.max_density_by_material()) if wide else _xs(grid)
    )
    return WarpEngine(grid=grid, cross_sections=xs, device=device)


class _WideAnalyticWater(AnalyticCrossSections):
    """Test instrument: analytic water declaring a 250 MeV energy domain.

    The analytic source ends at 20 MeV, so no shipped analytic run reaches the
    30 MeV threshold of ``"auto"``. This instrument declares the wider domain the
    way ``tests/unit/test_energy_domain.py`` does, to exercise the *mechanics*
    above the threshold without the EPICS libraries: range, table and
    bremsstrahlung grids extend to the declared domain; the collision stopping
    power is Berger-Seltzer and the radiative stopping power follows ESTAR to
    1 GeV, both valid there. Its photon data is the analytic fit extrapolated, so
    nothing run with it is a validated dose; the validated route is a tabulated
    compile (``tests/validation/test_bethe_heitler_high_energy.py``).
    """

    @property
    def energy_max(self) -> float:
        return 250.0


def _column_run(engine, seed: int):  # type: ignore[no-untyped-def]
    return engine.run(
        _pencil(),
        n_histories=N_HISTORIES,
        n_batches=N_BATCHES,
        seed=seed,
        primary_kind="electron",
        msc_model="gaussian",
        physics=BETHE_HEITLER,
    )


@pytest.fixture(scope="module")
def ref_column():  # type: ignore[no-untyped-def]
    """One reference run shared by every device comparison and every mutant."""
    grid = _column()
    return _column_run(ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG()), SEED)


def _assert_consistent(ref, other) -> None:  # type: ignore[no-untyped-def]
    mask = ref.dose > 0.5 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (other.dose_sigma > 0.0)
    assert np.count_nonzero(mask) >= 12
    # No sparse bins: the oracle has no power where the batch means are not normal.
    assert np.all(ref.dose_sigma[mask] < 0.05 * ref.dose[mask])
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        other.dose,
        other.dose_sigma,
        other.n_batches,
        mask=mask,
    )


# -- cross-backend agreement, and the oracle's power ---------------------------------


def test_dose_agrees_across_backends(device: str, ref_column) -> None:  # type: ignore[no-untyped-def]
    """Reference and Warp agree under the model, at equal history counts."""
    warp_result = _column_run(_warp_engine(_column(), device), SEED + 1)
    assert warp_result.provenance is not None
    assert warp_result.provenance.physics == RESOLVED
    _assert_consistent(ref_column, warp_result)


def _total_radiative_as_restricted(
    source: CrossSectionSource, ecut: float, pcut: float, e_max: float, n_points: int | None = None
) -> BetheHeitlerTables:
    """Mutant: the continuous arm charges the *total* radiative stopping power.

    Every photon above PCUT is then paid for twice — once continuously, once as
    the discrete event.
    """
    tables = build_bethe_heitler_tables(source, ecut, pcut, e_max, n_points)
    energies = np.geomspace(0.5 * ecut, e_max, tables.n_points)
    total = np.array(
        [
            [source.radiative_stopping_power(float(e), material) for e in energies]
            for material in range(source.n_materials)
        ]
    )
    return dataclasses.replace(tables, stopping_radiative_restricted=total)


def _no_discrete_channel(
    source: CrossSectionSource, ecut: float, pcut: float, e_max: float, n_points: int | None = None
) -> BetheHeitlerTables:
    """Mutant: the discrete cross-section is zero, so no photon above PCUT is emitted."""
    tables = build_bethe_heitler_tables(source, ecut, pcut, e_max, n_points)
    return dataclasses.replace(tables, cross_section=np.zeros_like(tables.cross_section))


@pytest.mark.parametrize("mutant", [_total_radiative_as_restricted, _no_discrete_channel])
def test_the_oracle_rejects_a_broken_radiative_split(  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch, ref_column, mutant
) -> None:
    """Power: the same comparison fails when the kernel is fed a wrong split.

    The mutant replaces the tables the engine uploads; the kernel and the
    reference run are untouched. Energy stays conserved under both mutants (the
    books close), so only the dose comparison can see them.
    """
    monkeypatch.setattr(bethe_heitler_data, "build_bethe_heitler_tables", mutant)
    broken = _column_run(_warp_engine(_column()), SEED + 1)

    balance = broken.energy_deposited + broken.energy_escaped
    assert balance == pytest.approx(broken.energy_emitted, rel=1e-6)
    with pytest.raises(AssertionError, match="not statistically consistent"):
        _assert_consistent(ref_column, broken)


# -- energy books and reproducibility ------------------------------------------------


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


@pytest.mark.parametrize("primary_kind", ["photon", "electron"])
def test_energy_ledger_closes(device: str, primary_kind: str) -> None:
    """emitted == deposited + escaped under the model, roulette included (default GS)."""
    result = _warp_engine(_phantom(), device).run(
        PencilBeamSource(energy=18.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0)),
        n_histories=8_000,
        n_batches=4,
        seed=SEED,
        primary_kind=primary_kind,
        physics=BETHE_HEITLER,
    )
    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=1e-6)


def test_dij_ledger_closes_and_records_the_model(device: str) -> None:
    """The Dij route carries the model to the same kernel."""
    result = _warp_engine(_phantom(), device).run_dij(
        BeamletGridSource(
            energy=18.0, z=-1.0, x_range=(2.0, 6.0), y_range=(3.0, 5.0), n_x=2, n_y=1
        ),
        n_histories_per_beamlet=2_000,
        n_batches=4,
        seed=SEED,
        physics=BETHE_HEITLER,
    )
    assert result.provenance is not None
    assert result.provenance.physics == RESOLVED
    assert result.dose_csc().nnz > 0
    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=1e-6)


def test_is_bit_reproducible_within_a_device(device: str) -> None:
    """Same device, same seed: identical dose (AGENTS.md 2.3)."""
    engine = _warp_engine(_phantom(), device)
    beam = PencilBeamSource(energy=18.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
    runs = [
        engine.run(
            beam,
            n_histories=4_000,
            n_batches=4,
            seed=SEED,
            primary_kind="electron",
            physics=BETHE_HEITLER,
        )
        for _ in range(2)
    ]
    assert runs[0].dose.max() > 0.0
    assert runs[0].dose.tobytes() == runs[1].dose.tobytes()
    assert runs[0].energy_escaped == runs[1].energy_escaped


def test_the_model_changes_the_transported_histories(device: str) -> None:
    """Same device and seed, the other model: a different dose, not an alias."""
    engine = _warp_engine(_phantom(), device)
    beam = PencilBeamSource(energy=18.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
    common = dict(n_histories=4_000, n_batches=4, seed=SEED, primary_kind="electron")
    thin = engine.run(beam, **common, physics=PhysicsOptions(bremsstrahlung="thin-target"))
    screened = engine.run(beam, **common, physics=BETHE_HEITLER)
    assert thin.dose.tobytes() != screened.dose.tobytes()


# -- one compiled electron kernel per model ------------------------------------------


class TestOneKernelPerModel:
    """The model is a compile-time constant of the kernel, not a run-time branch.

    A run-time branch on ``tab.brems_model`` cost the thin-target path 6.5 to 8 %
    on CUDA although it was never taken (kernels.make_electron_kernel). These pin
    the mechanism that replaced it: the default kernel stays in the shared module,
    any other model gets a module of its own, and the engine picks by the codes on
    the tables.
    """

    def test_default_kernel_is_the_shared_modules(self) -> None:
        from pyradmc.backends.warp import kernels

        default = kernels.make_electron_kernel(kernels.BREMS_THIN_TARGET)
        assert default is kernels.electron_kernel
        assert default.module is wp.get_module(kernels.__name__)

    def test_another_model_compiles_in_its_own_module(self) -> None:
        from pyradmc.backends.warp import kernels

        screened = kernels.make_electron_kernel(kernels.BREMS_BETHE_HEITLER)
        assert screened is kernels.make_electron_kernel(kernels.BREMS_BETHE_HEITLER)
        assert screened is not kernels.electron_kernel
        assert screened.module is not kernels.electron_kernel.module
        assert screened.module.options["fast_math"] is True
        assert screened.module.options["enable_backward"] is False

    def test_only_a_non_default_kernel_takes_model_data(self) -> None:
        from pyradmc.backends.warp import kernels

        def model_type(kernel):  # type: ignore[no-untyped-def]
            return next(arg.type for arg in kernel.adj.args if arg.label == "model")

        assert model_type(kernels.electron_kernel) is wp.int32
        screened = kernels.make_electron_kernel(kernels.BREMS_BETHE_HEITLER)
        assert model_type(screened) is kernels.ElectronModelTables

    def test_an_unknown_code_has_no_kernel(self) -> None:
        from pyradmc.backends.warp import kernels

        with pytest.raises(ValueError, match="no electron kernel"):
            kernels.make_electron_kernel(99)

    def test_the_tables_select_the_kernel(self) -> None:
        from pyradmc.backends.warp import kernels

        engine = _warp_engine(_phantom())
        thin = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)
        screened = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)
        assert kernels.electron_kernel_for(thin) is kernels.electron_kernel
        assert kernels.electron_kernel_for(screened) is kernels.make_electron_kernel(
            kernels.BREMS_BETHE_HEITLER
        )

    def test_concurrent_lanes_run_the_model(self, device: str) -> None:
        """Lanes launch from host threads; the model's module is loaded before them."""
        engine = _warp_engine(_phantom(), device)
        beam = PencilBeamSource(energy=18.0, position=(4.0, 4.0, -1.0), direction=(0, 0, 1.0))
        common = dict(n_histories=4_000, n_batches=4, seed=SEED, physics=BETHE_HEITLER)
        serial = engine.run(beam, **common)
        lanes = engine.run(beam, **common, concurrent_batches=2)
        assert serial.dose.max() > 0.0
        assert lanes.dose.tobytes() == serial.dose.tobytes()
        dij = engine.run_dij(
            BeamletGridSource(
                energy=18.0, z=-1.0, x_range=(2.0, 6.0), y_range=(3.0, 5.0), n_x=2, n_y=1
            ),
            n_histories_per_beamlet=1_000,
            n_batches=4,
            seed=SEED,
            physics=BETHE_HEITLER,
            concurrent_batches=2,
        )
        balance = dij.energy_deposited + dij.energy_escaped
        assert balance == pytest.approx(dij.energy_emitted, rel=1e-6)


# -- the model's tables --------------------------------------------------------------


class TestTables:
    def test_selected_model_uploads_its_tables(self) -> None:
        from pyradmc.backends.warp.kernels import KERNEL_PHYSICS_CODES

        engine = _warp_engine(_phantom())
        tab = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)

        model = engine._model_tables_for(tab, "cpu")

        assert tab.brems_model == KERNEL_PHYSICS_CODES["bremsstrahlung"]["bethe-heitler"]
        assert model.brems_cross_section.shape == (1, tab.n_points)
        assert model.stopping_radiative_restricted.shape == (1, tab.n_points)
        assert model.brems_screening.shape == (1, 5)

    def test_device_tables_are_the_host_queries(self) -> None:
        """On the kernel's electron grid: node j is the interface's answer at E_j."""
        grid = _phantom()
        engine = _warp_engine(grid)
        xs = engine.cross_sections
        tab = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)
        model = engine._model_tables_for(tab, "cpu")
        sigma = model.brems_cross_section.numpy()[0]
        restricted = model.stopping_radiative_restricted.numpy()[0]

        for j in (0, 300, 1100, tab.n_points - 1):
            energy = float(np.exp(tab.e_log_e_min + j / tab.e_inv_dlog))
            assert sigma[j] == pytest.approx(
                xs.bremsstrahlung_cross_section(energy, 0, 0.05), rel=1e-5, abs=1e-12
            )
            assert restricted[j] == pytest.approx(
                xs.restricted_radiative_stopping_power(energy, 0, 0.05), rel=1e-5
            )
        assert model.brems_screening.numpy()[0] == pytest.approx(
            np.array(xs.bremsstrahlung_screening(0)[:5]), rel=1e-6
        )

    def test_thin_target_run_builds_and_uploads_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that does not select the model does not pay for its integrals."""
        calls: list[float] = []

        def counting(source, ecut, pcut, e_max, n_points=None):  # type: ignore[no-untyped-def]
            calls.append(e_max)
            return build_bethe_heitler_tables(source, ecut, pcut, e_max, n_points)

        monkeypatch.setattr(bethe_heitler_data, "build_bethe_heitler_tables", counting)
        engine = _warp_engine(_phantom())

        tab = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)
        assert calls == []
        assert engine._model_tables_for(tab, "cpu") == 0  # the default kernel's placeholder

        engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)
        engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)
        assert len(calls) == 1  # cached per (ecut, pcut, e_max), like the other tables
        engine._upload_tables(ENERGY_MEV, 0.02, 0.2, "cpu", physics=RESOLVED)
        assert len(calls) == 2  # PCUT is the model's photon threshold: a new table

    def test_the_tables_struct_carries_no_model_arrays(self) -> None:
        """Model arrays on ``Tables`` cost the default path 2 to 2.7 % on CUDA (measured).

        They ride in ``ElectronModelTables``, the ``model`` argument of a kernel
        compiled for a non-default model, paired with the resident ``Tables`` they
        were built for; the default kernel's ``model`` is an ``int`` placeholder.
        """
        from pyradmc.backends.warp import kernels

        assert not {
            "brems_cross_section",
            "stopping_radiative_restricted",
            "brems_screening",
        } & set(kernels.Tables.vars)
        engine = _warp_engine(_phantom())
        thin = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=CLINICAL_PHYSICS)
        screened = engine._upload_tables(ENERGY_MEV, 0.05, 0.2, "cpu", physics=RESOLVED)
        assert engine._model_tables_for(thin, "cpu") == 0
        model = engine._model_tables_for(screened, "cpu")
        assert model.brems_screening.shape == (1, 5)
        with pytest.raises(RuntimeError, match="without model data"):
            engine._model_tables_for(kernels.Tables(), "cpu")

    def test_model_data_follows_the_resident_struct_out(self) -> None:
        """Evicting a ``Tables`` struct evicts its model data: nothing accumulates."""
        from pyradmc.backends.warp.engine import _TABLES_RESIDENT

        engine = _warp_engine(_phantom())
        for i in range(_TABLES_RESIDENT + 3):
            engine._upload_tables(ENERGY_MEV, 0.05 + 0.001 * i, 0.2, "cpu")
        state = engine._state("cpu")
        assert len(state.tables) == _TABLES_RESIDENT
        assert set(state.model_tables) == {id(t) for t in state.tables.values()}


# -- "auto" above the high-energy threshold ------------------------------------------


class TestAutoAboveTheThreshold:
    """A source above 30 MeV resolves to the model and runs end to end.

    This pins the *mechanics* of the ``"auto"`` path (resolution, table build to
    the source energy, transport, books) on the wide-domain instrument, at 40 MeV
    and, now that the range grid follows the declared domain, at 80 and 200 MeV,
    where the stale 64 MeV clamp used to dump the excess energy in the first
    substep. Nothing here is a validated dose. The shipped analytic source
    refuses all three energies before any transport.
    """

    @pytest.fixture(params=[40.0, 80.0, 200.0])
    def energy(self, request: pytest.FixtureRequest) -> float:
        return float(request.param)

    def _run(self, device: str, physics, energy: float, wide: bool = True):  # type: ignore[no-untyped-def]
        n_z = 60
        dz = 0.5 * energy / 40.0  # the column scales with the range
        grid = VoxelGrid.uniform_water(shape=(1, 1, n_z), spacing=(60.0, 60.0, dz))
        beam = PencilBeamSource(energy=energy, position=(30.0, 30.0, -0.5), direction=(0, 0, 1.0))
        return _warp_engine(grid, device, wide=wide).run(
            beam,
            n_histories=4_000,
            n_batches=4,
            seed=SEED,
            primary_kind="electron",
            physics=physics,
        )

    def test_the_analytic_source_refuses_the_run(self, device: str, energy: float) -> None:
        for physics in ("auto", BETHE_HEITLER):
            with pytest.raises(ValueError, match="energy domain"):
                self._run(device, physics, energy, wide=False)

    def test_auto_runs_the_model_and_the_books_close(self, device: str, energy: float) -> None:
        result = self._run(device, "auto", energy)

        assert result.provenance is not None
        assert result.provenance.physics == RESOLVED
        assert "above the 30 MeV" in result.provenance.physics.reason
        balance = result.energy_deposited + result.energy_escaped
        assert balance == pytest.approx(result.energy_emitted, rel=1e-6)

    def test_the_surface_voxel_takes_no_excess_energy(self, device: str, energy: float) -> None:
        """The entrance bin holds a collision loss, not the energy above a clamp.

        With the range map clamped at 64 MeV an 80 MeV electron dropped 16 MeV at
        the surface. The entrance bin's deposit per electron is bounded here by
        twice the unrestricted collision loss across it (about 2 MeV/cm in water).
        """
        result = self._run(device, "auto", energy)
        dz = 0.5 * energy / 40.0
        mass_g = 60.0 * 60.0 * dz
        per_electron = float(result.dose[0, 0, 0]) * mass_g  # dose is MeV/g per history
        assert 0.5 * 2.0 * dz < per_electron < 2.0 * 2.0 * dz

    def test_auto_is_the_explicit_model_bit_for_bit(self, device: str) -> None:
        auto = self._run(device, "auto", 40.0)
        explicit = self._run(device, BETHE_HEITLER, 40.0)
        thin = self._run(device, PhysicsOptions(bremsstrahlung="thin-target"), 40.0)

        assert auto.dose.max() > 0.0
        assert auto.dose.tobytes() == explicit.dose.tobytes()
        assert auto.energy_escaped == explicit.energy_escaped
        assert auto.dose.tobytes() != thin.dose.tobytes()


class TestDosimetricEffectOfTheSwitch:
    """What changing ``"auto"`` above the threshold does to a dose (AGENTS.md 2.10).

    The two models radiate the same energy; the screened shape spends it in more,
    softer photons, so fewer electrons lose a large fraction of their energy in
    one event and the *typical* electron radiates closer to the mean. The depth of
    half-maximum dose therefore moves shallower: at 40 MeV by 2.4 +- 0.4 mm in
    this sample (eight replicates of 8000 histories per model; the example
    ``examples/bremsstrahlung_model_demo.py`` measures 2.80 +- 0.02 mm at eight
    million). This pins the sign and the significance, not the value.

    Run on the wide-domain instrument (the shipped analytic source ends at
    20 MeV): read the number as the difference between the models, not as a
    validated 40 MeV range.
    """

    N_Z = 60
    DZ_CM = 0.5
    REPLICATES = 8

    def _r50(self, dose: np.ndarray) -> float:
        depth = (np.arange(self.N_Z) + 0.5) * self.DZ_CM
        peak = int(np.argmax(dose))
        half = 0.5 * dose[peak]
        below = peak + int(np.argmax(dose[peak:] < half))
        d0, d1 = dose[below - 1], dose[below]
        return float(depth[below - 1] + (d0 - half) / (d0 - d1) * self.DZ_CM)

    def _r50_replicates(self, engine, model: str, first_seed: int) -> np.ndarray:  # type: ignore[no-untyped-def]
        beam = PencilBeamSource(energy=40.0, position=(30.0, 30.0, -0.5), direction=(0, 0, 1.0))
        return np.array(
            [
                self._r50(
                    engine.run(
                        beam,
                        n_histories=8_000,
                        n_batches=4,
                        seed=first_seed + r,
                        primary_kind="electron",
                        physics=PhysicsOptions(bremsstrahlung=model),
                    ).dose[0, 0]
                )
                for r in range(self.REPLICATES)
            ]
        )

    def test_the_electron_range_moves_shallower(self, device: str) -> None:
        grid = VoxelGrid.uniform_water(shape=(1, 1, self.N_Z), spacing=(60.0, 60.0, self.DZ_CM))
        engine = _warp_engine(grid, device, wide=True)
        thin = self._r50_replicates(engine, "thin-target", SEED)
        screened = self._r50_replicates(engine, "bethe-heitler", SEED + 1_000)

        shift = float(screened.mean() - thin.mean())
        error = float(np.hypot(screened.std(ddof=1), thin.std(ddof=1)) / np.sqrt(self.REPLICATES))
        assert shift / error < -3.0, f"R50 shift {10 * shift:+.2f} +- {10 * error:.2f} mm"
        assert -0.6 < shift < 0.0  # millimetres, not centimetres: a shape effect


class _SignedElectronPencil(Source):
    """An 18 MeV pencil along +z, every primary at one signed statistical weight."""

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def max_energy(self) -> float:
        return 18.0

    def emit(self, rng_state: RNGState) -> Primary:
        return Primary(18.0, 4.0, 4.0, -1.0, 0.0, 0.0, 1.0, weight=self._weight)


@pytest.mark.parametrize("weight", [0.5, PHOTON_ROULETTE_WEIGHT_CAP])
def test_a_negative_weight_history_mirrors_the_positive_one(device: str, weight: float) -> None:
    """Weight -w against +w on one stream: equal and opposite up to the quanta.

    Mirror of ``tests/integration/test_signed_weights.py`` under the model: every
    amount of the negative run is the exact negative of its twin's, and only the
    fixed-point rounding differs, by at most one 1e-9 MeV quantum per deposit pair
    (bounded here by 10^4 deposits per 18 MeV history). A photon or deposit dropped
    for its sign breaks the bound by orders of magnitude and opens the books.
    """
    n = 4_000
    engine = _warp_engine(_phantom(), device)
    common = dict(
        n_histories=n, n_batches=4, seed=SEED, primary_kind="electron", physics=BETHE_HEITLER
    )
    plus = engine.run(_SignedElectronPencil(weight), **common)
    minus = engine.run(_SignedElectronPencil(-weight), **common)

    assert np.any(plus.dose > 0.0)
    mass = np.asarray(_phantom().density) * 1.0  # 1 cm^3 voxels
    residual_quanta = np.abs(minus.dose + plus.dose) * mass * n / 1.0e-9
    assert residual_quanta.sum() <= 10_000 * n, residual_quanta.sum() / n
    assert minus.energy_emitted < 0.0
    books = minus.energy_deposited + minus.energy_escaped + minus.energy_unscored
    assert books == pytest.approx(minus.energy_emitted, rel=1.0e-4)
