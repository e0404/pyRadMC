"""The device-resident pre-solve buffer transports like the host-handoff phase space.

``presolve_head_device(..., return_device_source=True)`` leaves the exit population
on the device; ``WarpEngine.run`` then seeds the transport queues straight from it,
skipping the copy back to host and the re-upload the ``InMemoryPhaseSpaceSource``
path incurs. Both paths draw records (with replacement) from the *same* pre-solved
population, so with independent transport seeds they are two unbiased estimators of
one dose — chi-squared-consistent on a small water phantom — and the no-copy run's
energy ledger closes like any other. Analytic water devices keep the tier EPDL-free;
runs on cpu and, when present, cuda.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import GaussianSpotBeamSource
from pyradmc.geometry.spectrum import Spectrum
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [
    pytest.mark.warp,
    pytest.mark.filterwarnings("ignore:.*latent variance:UserWarning"),
]

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
FRAME = BeamFrame(origin=FOCAL)
N_PRESOLVE = 60_000
N_HISTORIES = 40_000


def _source() -> GaussianSpotBeamSource:
    return GaussianSpotBeamSource(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        center=(8.0, 8.0, -70.0),
        width_u=5.0,
        width_v=5.0,
        sigma_u=0.1,
        sigma_v=0.1,
    )


def _stack() -> BeamLimitingStack:
    return BeamLimitingStack(
        frame=FRAME,
        devices=(
            JawPair(
                axis="u",
                z_top=40.0,
                z_bottom=45.0,
                edge_neg=-1000.0,
                edge_pos=0.5,
                material=WATER,
                density=1.0,
            ),
            MLC(
                z_top=46.0,
                z_bottom=50.0,
                leaf_edges_v=(-3.0, -1.0, 1.0, 3.0),
                tips_neg=(-1.5, 0.2, -1.5),
                tips_pos=(1.5, 0.2, 1.5),
                tip_radius=8.0,
                material=WATER,
                density=1.0,
            ),
        ),
    )


def _grid() -> VoxelGrid:
    # Beam axis at (8, 8); the exit plane (local w = 60 => engine z = -40) feeds the top.
    return VoxelGrid.uniform_water(
        shape=(16, 16, 20), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, -40.0)
    )


def _require(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")


@pytest.mark.parametrize("device", DEVICES)
def test_device_resident_transport_matches_host_handoff(device: str) -> None:
    _require(device)
    from pyradmc.backends.warp.engine import WarpEngine
    from pyradmc.backends.warp.presolve import presolve_head_device

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine = WarpEngine(grid=grid, cross_sections=xs, device=device)
    presolve_kwargs = dict(
        cross_sections=AnalyticCrossSections(),
        n_histories=N_PRESOLVE,
        seed=SEED,
        exit_z=60.0,
        stack=_stack(),
        mode="first_compton",
        device=device,
    )

    handoff = presolve_head_device(_source(), **presolve_kwargs)
    resident = presolve_head_device(_source(), return_device_source=True, **presolve_kwargs)

    run_kwargs = dict(n_histories=N_HISTORIES, n_batches=8, transport_electrons=False)
    # Independent transport seeds: two unbiased estimators of one population's dose.
    a = engine.run(handoff, seed=SEED + 1, **run_kwargs)
    b = engine.run(resident, seed=SEED + 2, **run_kwargs)

    mask = (a.dose > 0.1 * a.dose.max()) & (a.dose_sigma > 0.0) & (b.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        a.dose, a.dose_sigma, a.n_batches, b.dose, b.dose_sigma, b.n_batches, mask=mask
    )
    # The no-copy run's energy ledger closes (emitted booked in the seeding kernel).
    assert b.energy_emitted == pytest.approx(
        b.energy_deposited + b.energy_escaped + b.energy_unscored, rel=1e-6
    )


@pytest.mark.parametrize("device", DEVICES)
def test_device_source_round_trips_to_the_host_phase_space(device: str) -> None:
    """``to_phase_space`` materializes the same population the buffer holds."""
    _require(device)
    from pyradmc.backends.warp.presolve import presolve_head_device

    resident = presolve_head_device(
        _source(),
        cross_sections=AnalyticCrossSections(),
        n_histories=N_PRESOLVE,
        seed=SEED,
        exit_z=60.0,
        stack=_stack(),
        mode="first_compton",
        device=device,
        return_device_source=True,
    )
    phase_space = resident.to_phase_space()
    assert len(phase_space) == resident.count == len(resident)
    cols = phase_space.columns()
    w_buf = resident.buffer.weight.numpy()[: resident.count].astype(np.float64)
    np.testing.assert_allclose(w_buf.sum(), cols["weight"].sum(), rtol=1e-6)


def test_engine_rejects_a_buffer_from_another_device() -> None:
    if not wp.is_cuda_available():
        pytest.skip("needs two devices to cross them")
    from pyradmc.backends.warp.engine import WarpEngine
    from pyradmc.backends.warp.presolve import presolve_head_device

    grid = _grid()
    resident = presolve_head_device(
        _source(),
        cross_sections=AnalyticCrossSections(),
        n_histories=10_000,
        seed=SEED,
        exit_z=60.0,
        stack=_stack(),
        mode="attenuation",
        device="cuda:0",
        return_device_source=True,
    )
    engine = WarpEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        device="cpu",
    )
    with pytest.raises(ValueError, match="device"):
        engine.run(resident, n_histories=8_000, n_batches=8, seed=SEED + 1)
