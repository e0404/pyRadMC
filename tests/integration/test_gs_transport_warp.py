"""Goudsmit-Saunderson transport on the Warp backend, against the reference oracle.

The GS sampler itself is validated at the distribution level
(``tests/physics/test_warp_gs_sampler.py``); these tests pin the *transport
wiring*: the eager grid reaches the kernel, the angular cap is conditional on
the model exactly as in the reference loop, the energy ledger stays exact, and
the run is bit-reproducible within a device. The cross-backend comparison is
chi-squared (AGENTS.md 2.3) on an electron beam through a density slab — an
electron primary isolates the electron kernel, and the low-density slab makes
boundary-truncated substeps (the small-Lambda table regime) part of the path.

The GS configuration under test is the validated one: ``msc_model="gs"`` with
``step_energy_fraction=0.20``, where the angular cap is lifted and the substep
schedule genuinely differs from the shipped default.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletGridSource, ParallelBeamSource
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

pytestmark = pytest.mark.warp

SHAPE = (10, 10, 18)
SPACING = (1.0, 1.0, 0.5)
SLAB = slice(6, 10)  # z-voxels at lung-like density
ENERGY_MEV = 6.0
GS_FRACTION = 0.20


def _slab_grid() -> VoxelGrid:
    density = np.ones(SHAPE, dtype=np.float64)
    density[:, :, SLAB] = 0.26
    return VoxelGrid(
        shape=SHAPE,
        spacing=SPACING,
        density=density,
        material=np.zeros(SHAPE, dtype=np.int32),
    )


def _beam() -> ParallelBeamSource:
    return ParallelBeamSource(energy=ENERGY_MEV, z=-0.5, x_range=(2.0, 8.0), y_range=(2.0, 8.0))


def _warp_engine(grid: VoxelGrid, device: str = "cpu"):
    from pyRadMC.backends.warp.engine import WarpEngine

    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device=device)


DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]


@pytest.fixture(scope="module")
def ref_gs_result():
    """One reference GS run shared by every device comparison.

    The reference arm is the expensive one (single-history Python, plus its
    per-source lazy GS table builds); the device under test varies, the oracle
    does not.
    """
    grid = _slab_grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
        _beam(),
        n_histories=1_500,
        n_batches=12,
        seed=SEED,
        primary_kind="electron",
        msc_model="gs",
        step_energy_fraction=GS_FRACTION,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_gs_dose_agrees_across_backends(device: str, ref_gs_result) -> None:
    """Ref and Warp agree under GS f=0.20 (chi-squared, high-dose mask).

    Pins the whole device-side GS path at once: log_eta lookup, window offsets,
    grid upload and cast, the conditional cap, and the hinge branch — any of
    them wrong moves dose in a way the batched chi-squared detects.
    """
    import warp as wp

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    grid = _slab_grid()
    ref = ref_gs_result
    warp_result = _warp_engine(grid, device).run(
        _beam(),
        n_histories=12_000,
        n_batches=12,
        seed=SEED,
        primary_kind="electron",
        msc_model="gs",
        step_energy_fraction=GS_FRACTION,
    )
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp_result.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        warp_result.dose,
        warp_result.dose_sigma,
        warp_result.n_batches,
        mask=mask,
    )


def test_gs_energy_ledger_closes_on_warp() -> None:
    """emitted == deposited + escaped, exactly as under the Gaussian model."""
    result = _warp_engine(_slab_grid()).run(
        _beam(),
        n_histories=4_000,
        n_batches=8,
        seed=SEED,
        primary_kind="electron",
        msc_model="gs",
        step_energy_fraction=GS_FRACTION,
    )
    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=1e-6)


def test_gs_is_bit_reproducible_within_device() -> None:
    """Same device, same seed: identical dose under GS (AGENTS.md 2.3)."""
    grid = _slab_grid()
    runs = [
        _warp_engine(grid).run(
            _beam(),
            n_histories=2_000,
            n_batches=4,
            seed=SEED,
            primary_kind="electron",
            msc_model="gs",
            step_energy_fraction=GS_FRACTION,
        )
        for _ in range(2)
    ]
    np.testing.assert_array_equal(runs[0].dose, runs[1].dose)


def test_gaussian_default_is_the_explicit_gaussian(  # the shipped path must not move
) -> None:
    """Omitting msc_model equals passing "gaussian", to the bit.

    Guards the default routing: the new parameter must be inert unless asked
    for, so every existing Warp result is unchanged by this change.
    """
    grid = _slab_grid()
    implicit = _warp_engine(grid).run(
        _beam(), n_histories=2_000, n_batches=4, seed=SEED, primary_kind="electron"
    )
    explicit = _warp_engine(grid).run(
        _beam(),
        n_histories=2_000,
        n_batches=4,
        seed=SEED,
        primary_kind="electron",
        msc_model="gaussian",
    )
    np.testing.assert_array_equal(implicit.dose, explicit.dose)


def test_rejects_unknown_msc_model() -> None:
    """Same refusal (and wording) as the reference loop: fail before launching."""
    with pytest.raises(ValueError, match="msc_model"):
        _warp_engine(_slab_grid()).run(
            _beam(), n_histories=8, n_batches=1, seed=SEED, msc_model="moliere"
        )


def test_gs_dij_grouping_stays_bit_inert() -> None:
    """The Dij route carries msc_model too, and grouping stays bit-inert under it.

    The group-size invariance is the strongest cheap pin that the new argument
    reaches every launch wrapper consistently — a path that dropped or reordered
    it would key different tables in different groups and break bit-equality.
    """
    grid = _slab_grid()
    engine = _warp_engine(grid)
    source = BeamletGridSource(
        energy=ENERGY_MEV, z=-0.5, x_range=(2.0, 8.0), y_range=(2.0, 8.0), n_x=2, n_y=2
    )
    results = [
        engine.run_dij(
            source,
            n_histories_per_beamlet=400,
            n_batches=2,
            seed=SEED,
            msc_model="gs",
            step_energy_fraction=GS_FRACTION,
            beamlet_group_size=size,
        )
        for size in (2, 4)
    ]
    a, b = (r.dose_csc() for r in results)
    assert (a != b).nnz == 0
