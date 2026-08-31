"""``step_energy_fraction`` is a run-time knob on both engines (perf item 2).

The maximum fraction of CSDA range per electron substep is adjustable per run
(the STEP_ENERGY_FRACTION bias study needed it, and on the Warp backend a
*parameter* compiles once while a constant would recompile per value). Since
the Goudsmit-Saunderson adoption the default is **per model**: ``None``
resolves to the shipped GS configuration's 0.20, and to the Gaussian
instrument's validated 0.05 — the resolution is pinned bit-for-bit below.

Also pinned: a non-default value actually reaches the substep loop on every
backend (the dose moves); the energy ledger closes regardless of the value;
and the knob means the same physics on both backends (cross-backend
chi-squared at 0.10). Changing a *default* remains a maintainer decision with
a validation rerun, not a knob turn.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

ENERGY_BALANCE_RTOL = 1.0e-4


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _source() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(3.0, 13.0), y_range=(3.0, 13.0))


def _ref(grid: VoxelGrid) -> ReferenceEngine:
    return ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())


def test_default_resolves_to_the_shipped_gs_fraction() -> None:
    """``None`` equals the GS configuration's constant on the reference oracle."""
    from pyradmc.transport.electron import GS_STEP_ENERGY_FRACTION

    grid = _grid()
    kwargs = dict(n_histories=2_000, n_batches=4, seed=SEED)
    baseline = _ref(grid).run(_source(), **kwargs)
    explicit = _ref(grid).run(_source(), step_energy_fraction=GS_STEP_ENERGY_FRACTION, **kwargs)
    np.testing.assert_array_equal(baseline.dose, explicit.dose)


def test_default_resolves_to_the_instrument_fraction_under_gaussian() -> None:
    """``None`` equals 0.05 when the Gaussian instrument is selected."""
    from pyradmc.transport.electron import STEP_ENERGY_FRACTION

    grid = _grid()
    kwargs = dict(n_histories=2_000, n_batches=4, seed=SEED, msc_model="gaussian")
    baseline = _ref(grid).run(_source(), **kwargs)
    explicit = _ref(grid).run(_source(), step_energy_fraction=STEP_ENERGY_FRACTION, **kwargs)
    np.testing.assert_array_equal(baseline.dose, explicit.dose)


def test_nondefault_fraction_reaches_the_substep_loop_on_ref() -> None:
    """0.10 changes substep lengths, so the sampled cascade must differ."""
    grid = _grid()
    kwargs = dict(n_histories=2_000, n_batches=4, seed=SEED)
    default = _ref(grid).run(_source(), **kwargs)
    coarse = _ref(grid).run(_source(), step_energy_fraction=0.10, **kwargs)
    assert not np.array_equal(default.dose, coarse.dose)
    assert coarse.energy_emitted == pytest.approx(
        coarse.energy_deposited + coarse.energy_escaped, rel=ENERGY_BALANCE_RTOL
    )


@pytest.mark.warp
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)],
)
def test_nondefault_fraction_reaches_the_substep_loop_on_warp(device: str) -> None:
    """Same pin on the Warp backend: the knob must thread into the kernel."""
    wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    engine = WarpEngine(grid=grid, cross_sections=_xs(grid), device=device)
    kwargs = dict(n_histories=8_000, n_batches=4, seed=SEED)
    default = engine.run(_source(), **kwargs)
    coarse = engine.run(_source(), step_energy_fraction=0.10, **kwargs)
    assert not np.array_equal(default.dose, coarse.dose)
    assert coarse.energy_emitted == pytest.approx(
        coarse.energy_deposited + coarse.energy_escaped, rel=ENERGY_BALANCE_RTOL
    )


@pytest.mark.warp
def test_knob_means_the_same_physics_on_both_backends() -> None:
    """ref and warp at 0.10 stay chi-squared consistent — one knob, one meaning."""
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    kwargs = dict(n_batches=12, seed=SEED, step_energy_fraction=0.10)
    ref = _ref(grid).run(_source(), n_histories=6_000, **kwargs)
    warp_res = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu").run(
        _source(), n_histories=24_000, **kwargs
    )
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp_res.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        warp_res.dose,
        warp_res.dose_sigma,
        warp_res.n_batches,
        mask=mask,
    )


@pytest.mark.warp
def test_knob_is_plumbed_through_the_dij() -> None:
    """run_dij carries the knob too; ledger closes at a non-default value."""
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp.engine import WarpEngine
    from pyradmc.geometry.source import BeamletGridSource

    grid = _grid()
    lattice = BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(3.0, 13.0), y_range=(3.0, 13.0), n_x=2, n_y=2
    )
    engine = WarpEngine(grid=grid, cross_sections=_xs(grid), device="cpu")
    kwargs = dict(n_histories_per_beamlet=1_000, n_batches=2, seed=SEED)
    default = engine.run_dij(lattice, **kwargs)
    coarse = engine.run_dij(lattice, step_energy_fraction=0.10, **kwargs)
    assert not np.array_equal(default.column_dense(0), coarse.column_dense(0))
    assert coarse.energy_emitted == pytest.approx(
        coarse.energy_deposited + coarse.energy_escaped, rel=ENERGY_BALANCE_RTOL
    )
