"""Dose-to-water on the Warp backend, per device.

Same exact identities as the reference tests (transport is invariant to the
scoring mode; the ledger books physical energy), at float32/quantization
precision where the SPR factor itself is computed in-kernel. The physical
``energy_deposited`` is *exactly* equal across modes even on the device: the
per-deposit physical quanta are identical, only the tallied (scored) quanta
change.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletGridSource, ParallelBeamSource
from tests.conftest import SEED
from tests.unit.test_dose_to_water_spr import DELTA_CUT, _two_material_tables

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyRadMC.data.analytic import AnalyticCrossSections  # noqa: E402
from pyRadMC.data.tabulated.source import TabulatedCrossSections  # noqa: E402

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

ENERGY_BALANCE_RTOL = 1.0e-4  # float32 transport + 1e-9 MeV scoring quanta
SPR = 1.1


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _source() -> ParallelBeamSource:
    return ParallelBeamSource(energy=6.0, z=-1.0, x_range=(2.0, 14.0), y_range=(2.0, 14.0))


def _water_engine(device: str):
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(2.0, 2.0, 0.5))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device=device)


def _medium_engine(device: str):
    from pyRadMC.backends.warp.engine import WarpEngine

    shape = (8, 8, 16)
    grid = VoxelGrid(
        shape=shape,
        spacing=(2.0, 2.0, 0.5),
        density=np.ones(shape),
        material=np.full(shape, 1, dtype=np.int32),
    )
    xs = TabulatedCrossSections(
        _two_material_tables(spr=SPR), geometry_densities=grid.max_density_by_material()
    )
    return WarpEngine(grid=grid, cross_sections=xs, device=device)


def _run(engine, mode: str, n: int = 4_000):
    return engine.run(
        _source(), n_histories=n, n_batches=4, seed=SEED, ecut=DELTA_CUT, scoring_mode=mode
    )


def test_water_dose_to_water_is_bit_identical_to_dose_to_medium(device: str) -> None:
    engine = _water_engine(device)
    dm = _run(engine, "dose_to_medium")
    dw = _run(engine, "dose_to_water")
    np.testing.assert_array_equal(dw.dose, dm.dose)
    np.testing.assert_array_equal(dw.dose_sigma, dm.dose_sigma)
    # Same physical quanta; the two modes aggregate them differently (per-voxel
    # float64 sum vs the exact int64 counter), so equality is to float64 rounding.
    assert dw.energy_deposited == pytest.approx(dm.energy_deposited, rel=1e-12)
    assert dw.energy_escaped == dm.energy_escaped


def test_uniform_medium_scales_by_the_constant_spr(device: str) -> None:
    engine = _medium_engine(device)
    dm = _run(engine, "dose_to_medium")
    dw = _run(engine, "dose_to_water")

    touched = dm.dose > 0.0
    assert np.count_nonzero(touched) > 100
    # The factor is evaluated in float32 per deposit; 1e-5 covers its rounding.
    np.testing.assert_allclose(dw.dose[touched], SPR * dm.dose[touched], rtol=1e-5)

    # Physical books: per-deposit physical quanta are identical across modes
    # (aggregation differs at float64 rounding, as in the water test above).
    assert dw.energy_deposited == pytest.approx(dm.energy_deposited, rel=1e-12)
    assert dw.energy_escaped == dm.energy_escaped
    assert dw.energy_deposited + dw.energy_unscored + dw.energy_escaped == pytest.approx(
        dw.energy_emitted, rel=ENERGY_BALANCE_RTOL
    )


def test_dij_column_scales_by_the_constant_spr(device: str) -> None:
    engine = _medium_engine(device)
    lattice = BeamletGridSource(
        energy=6.0, z=-1.0, x_range=(2.0, 14.0), y_range=(2.0, 14.0), n_x=2, n_y=1
    )
    kwargs = dict(
        n_histories_per_beamlet=800,
        n_batches=4,
        seed=SEED,
        ecut=DELTA_CUT,
        truncation=0.0,
    )
    dm = engine.run_dij(lattice, scoring_mode="dose_to_medium", **kwargs)
    dw = engine.run_dij(lattice, scoring_mode="dose_to_water", **kwargs)
    for j in range(lattice.n_beamlets):
        a, b = dm.column_dense(j), dw.column_dense(j)
        touched = a > 0.0
        np.testing.assert_allclose(b[touched], SPR * a[touched], rtol=1e-5)
    assert dw.energy_deposited == dm.energy_deposited
    assert dw.energy_deposited + dw.energy_unscored + dw.energy_escaped == pytest.approx(
        dw.energy_emitted, rel=ENERGY_BALANCE_RTOL
    )
    assert dw.scoring_mode == "dose_to_water"


def test_kerma_mode_rejects_dose_to_water(device: str) -> None:
    engine = _water_engine(device)
    with pytest.raises(ValueError, match="dose_to_water"):
        engine.run(
            _source(),
            n_histories=4,
            n_batches=1,
            seed=SEED,
            transport_electrons=False,
            scoring_mode="dose_to_water",
        )
