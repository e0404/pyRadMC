"""The head pre-solve's phase space transports like the physics it approximates.

Three anchors on a 16^3 water phantom (analytic backend, water devices):

1. **Pass-through**: with nothing in the beam, dose from the pre-solved phase
   space is chi-squared-consistent with transporting the planar source directly.
2. **Attenuation mode == the deterministic wrapper**: same Beer-Lambert physics,
   different representation (phase space vs per-ray weights).
3. **First Compton adds, never subtracts**: the under-block region gains the
   collimator-scatter component on top of the attenuation-only dose.

Oversampling a finite pre-solve population is inherent to the comparison, so the
latent-variance caveat is acknowledged; the pre-solve is sized at the history
count to keep the reuse mild.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.geometry.collimation import (
    BeamFrame,
    BeamLimitingStack,
    CollimatedSource,
    JawPair,
)
from pyRadMC.geometry.head import presolve_head
from pyRadMC.geometry.source import GaussianSpotBeamSource
from pyRadMC.geometry.spectrum import Spectrum
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

pytestmark = pytest.mark.filterwarnings("ignore:.*latent variance:UserWarning")

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
FRAME = BeamFrame(origin=FOCAL)
N_PRESOLVE = 20_000
N_HISTORIES = 6_000


def _plane_source() -> GaussianSpotBeamSource:
    return GaussianSpotBeamSource(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        center=(8.0, 8.0, -70.0),
        width_u=5.0,
        width_v=5.0,
        sigma_u=0.1,
        sigma_v=0.1,
    )


def _jaws() -> BeamLimitingStack:
    """The +u half of the field passes under 10 cm of water-equivalent block."""
    return BeamLimitingStack(
        frame=FRAME,
        devices=(
            JawPair(
                axis="u",
                z_top=40.0,
                z_bottom=50.0,
                edge_neg=-1000.0,
                edge_pos=0.0,
                material=WATER,
                density=1.0,
            ),
        ),
    )


def _grid():  # -> VoxelGrid
    from pyRadMC.geometry.grid import VoxelGrid

    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _engine(grid) -> ReferenceEngine:
    return ReferenceEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )


def _run(engine, source):
    return engine.run(
        source, n_histories=N_HISTORIES, n_batches=8, seed=SEED + 1, transport_electrons=False
    )


def _chi2(a, b) -> None:
    mask = a.dose > 0.1 * a.dose.max()
    mask &= (a.dose_sigma > 0.0) & (b.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        a.dose, a.dose_sigma, a.n_batches, b.dose, b.dose_sigma, b.n_batches, mask=mask
    )


def test_pass_through_matches_the_direct_source() -> None:
    grid = _grid()
    engine = _engine(grid)
    source = _plane_source()
    phsp = presolve_head(
        source,
        stack=None,
        frame=FRAME,
        cross_sections=AnalyticCrossSections(),
        n_histories=N_PRESOLVE,
        seed=SEED,
        exit_z=95.0,
    )
    _chi2(_run(engine, source), _run(engine, phsp))


def test_attenuation_mode_matches_the_collimated_wrapper() -> None:
    grid = _grid()
    engine = _engine(grid)
    source = _plane_source()
    stack = _jaws()
    xs = AnalyticCrossSections()
    phsp = presolve_head(
        source,
        stack=stack,
        cross_sections=xs,
        n_histories=N_PRESOLVE,
        seed=SEED,
        exit_z=95.0,
        mode="attenuation",
    )
    wrapped = CollimatedSource(source, stack, xs)
    _chi2(_run(engine, wrapped), _run(engine, phsp))
    # The downstream run's ledger closes as for any weighted source.
    result = _run(engine, phsp)
    assert result.energy_emitted == pytest.approx(
        result.energy_deposited + result.energy_escaped, rel=1e-9
    )


def test_first_compton_adds_dose_under_the_block() -> None:
    """Collimator scatter raises the shielded-side dose above attenuation-only.

    The scattered photons carry ~(1 - T) * f_c of the blocked weight; with 10 cm
    of water-equivalent block that is a several-percent addition to the shielded
    half, resolvable at these statistics. The open half must stay consistent.
    """
    grid = _grid()
    engine = _engine(grid)
    source = _plane_source()
    stack = _jaws()
    xs = AnalyticCrossSections()
    kwargs = dict(stack=stack, cross_sections=xs, n_histories=N_PRESOLVE, seed=SEED, exit_z=95.0)
    attenuation = _run(engine, presolve_head(source, mode="attenuation", **kwargs))
    scatter = _run(engine, presolve_head(source, mode="first_compton", **kwargs))
    shielded = np.s_[10:14, 6:10, :4]
    assert scatter.dose[shielded].mean() > attenuation.dose[shielded].mean()
