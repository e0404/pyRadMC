"""Transporting a phase-space source through the reference engine.

The invariant that proves per-record *weight* and *kind* are threaded end to end is
energy conservation: emitted = deposited + escaped, exactly (bookkeeping, not
statistics). It holds only if the emitted-energy book weights each primary by its
record weight the same way the transport and scorer do. A file mixing photons,
electrons and positrons at non-unit weights exercises every kind mapping at once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.phasespace import PhaseSpaceSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED
from tests.phsp_fixtures import Rec, write_phsp

# These tests deliberately oversample tiny synthetic phase spaces; the finite-
# reuse latent-variance caveat is expected and acknowledged (pinned explicitly
# in the tripwire tests).
pytestmark = pytest.mark.filterwarnings("ignore:.*latent variance:UserWarning")


def _engine() -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())


def _mixed_source(tmp_path: Path) -> PhaseSpaceSource:
    # Directed into the phantom (+z) from just outside the front face at z = -1.
    common = dict(x=8.0, y=8.0, z=-1.0, u=0.0, v=0.0)
    recs = [
        Rec(1, 6.0, weight=1.0, **common),  # photon, full weight
        Rec(1, 2.0, weight=0.5, **common),  # photon, down-weighted
        Rec(2, 1.5, weight=0.25, **common),  # electron contaminant
        Rec(3, 0.8, weight=0.75, **common),  # positron contaminant
    ]
    return PhaseSpaceSource(write_phsp(tmp_path / "beam", recs))


def test_energy_conserved_with_weights_and_mixed_kinds(tmp_path: Path) -> None:
    src = _mixed_source(tmp_path)
    result = _engine().run(src, n_histories=400, n_batches=4, seed=SEED)

    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=1e-9)
    assert result.energy_emitted > 0.0
    assert result.energy_deposited > 0.0


def test_emitted_energy_is_weighted(tmp_path: Path) -> None:
    """Emitted energy weights each primary by its record weight, not raw energy.

    Both records carry the same weight*energy = 3.0 MeV but different raw energy, so
    the emitted book is 400 * 3.0 = 1200 MeV whichever records the histories sample.
    Were weight ignored, the sum of raw energies (6.0 or 3.0 per history) could never
    equal 1200 for all seeds. This isolates the weight factor from the sampling.
    """
    common = dict(x=8.0, y=8.0, z=-1.0, u=0.0, v=0.0)
    recs = [
        Rec(1, 6.0, weight=0.5, **common),  # weighted energy 3.0
        Rec(1, 3.0, weight=1.0, **common),  # weighted energy 3.0
    ]
    src = PhaseSpaceSource(write_phsp(tmp_path / "beam", recs))
    result = _engine().run(src, n_histories=400, n_batches=4, seed=SEED)
    assert result.energy_emitted == pytest.approx(400 * 3.0, rel=1e-9)


def test_positron_primary_emitted_energy_includes_rest_mass(tmp_path: Path) -> None:
    """A positron primary emits kinetic + 2*m_e c^2: its annihilation photons come
    from rest mass, and the emitted book must count them or the ledger cannot close.

    A pure-positron file makes the emitted energy exact regardless of sampling:
    400 * weight * (E + 1.022). The conservation check then confirms the deposited +
    escaped side (which receives those annihilation photons) matches.

    ``ke`` and ``weight`` are chosen exactly representable in float32, because phsp
    records store energy as float32; a value like 0.9 would round on write and the
    analytic oracle below would miss the engine's stored energy by ~1e-8 relative.
    """
    weight, ke = 0.5, 0.75
    recs = [Rec(3, ke, x=8.0, y=8.0, z=-1.0, weight=weight)]  # positron
    src = PhaseSpaceSource(write_phsp(tmp_path / "pos", recs))
    result = _engine().run(src, n_histories=400, n_batches=4, seed=SEED)

    expected = 400 * weight * (ke + 2.0 * ELECTRON_MASS_MEV)
    assert result.energy_emitted == pytest.approx(expected, rel=1e-9)
    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=1e-9)
