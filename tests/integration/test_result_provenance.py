"""Every engine result carries the configuration that produced it.

:class:`~pyRadMC.backends.results.RunProvenance` is optional on the result
dataclasses — a hand-assembled or reloaded result need not fabricate one — so the
guarantee that *engine* results always carry it cannot live in the type. It lives
here instead, across both backends and both entry points (``run`` and ``run_dij``),
because that is the guarantee downstream consumers actually depend on: a dose array
handed to an optimizer, archived, and read back a year later must still say which
cross-sections, cutoffs and seed produced it.

The values are checked against what the call actually requested, not merely for
presence: a record that reports defaults regardless of arguments is worse than none,
because it looks authoritative.
"""

from __future__ import annotations

import pytest

import pyRadMC
from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletGridSource, PencilBeamSource
from pyRadMC.rng.host import HostRNG

SEED = 20260726


@pytest.fixture(scope="module")
def engine() -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(8, 8, 8), spacing=(0.5, 0.5, 0.5))
    return ReferenceEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        rng=HostRNG(),
    )


def _beam() -> PencilBeamSource:
    return PencilBeamSource(energy=2.0, position=(2.0, 2.0, -1.0), direction=(0.0, 0.0, 1.0))


def test_run_records_what_the_call_requested(engine: ReferenceEngine) -> None:
    """Non-default cutoffs and model must appear in the record, not the defaults."""
    result = engine.run(
        _beam(),
        n_histories=16,
        n_batches=2,
        seed=SEED,
        pcut=0.06,
        ecut=0.25,
        msc_model="gaussian",
        step_energy_fraction=0.04,
    )

    provenance = result.provenance
    assert provenance is not None
    assert provenance.version == pyRadMC.__version__
    assert provenance.backend == "ref"
    assert provenance.device == "cpu"
    assert provenance.seed == SEED
    assert provenance.pcut_mev == 0.06
    assert provenance.ecut_mev == 0.25
    assert provenance.msc_model == "gaussian"
    assert provenance.step_energy_fraction == 0.04
    assert "analytic" in provenance.cross_sections


def test_unspecified_step_fraction_is_recorded_resolved(engine: ReferenceEngine) -> None:
    """``None`` means 'follow the model', and the record must say which value ran.

    Storing the caller's ``None`` would make the record useless for exactly the
    runs that use the shipped defaults, i.e. almost all of them.
    """
    from pyRadMC.transport.electron import default_step_energy_fraction

    result = engine.run(_beam(), n_histories=16, n_batches=2, seed=SEED, msc_model="gs")

    assert result.provenance is not None
    assert result.provenance.step_energy_fraction == default_step_energy_fraction("gs")
    assert result.provenance.step_energy_fraction is not None


def test_run_dij_records_provenance(engine: ReferenceEngine) -> None:
    """The Dij is the archived product; it is the one that most needs the record."""
    source = BeamletGridSource(
        energy=2.0,
        z=-1.0,
        x_range=(1.0, 3.0),
        y_range=(1.0, 3.0),
        n_x=2,
        n_y=1,
    )
    result = engine.run_dij(source, n_histories_per_beamlet=16, n_batches=2, seed=SEED, ecut=0.3)

    assert result.provenance is not None
    assert result.provenance.backend == "ref"
    assert result.provenance.seed == SEED
    assert result.provenance.ecut_mev == 0.3


def test_summary_names_the_run(engine: ReferenceEngine) -> None:
    """The one-line digest is what ends up in a log or a file header."""
    result = engine.run(_beam(), n_histories=16, n_batches=2, seed=SEED)

    assert result.provenance is not None
    summary = result.provenance.summary()
    assert pyRadMC.__version__ in summary
    assert "ref/cpu" in summary
    assert str(SEED) in summary


@pytest.mark.warp
def test_warp_backend_records_its_device() -> None:
    """The production backend records ``warp`` and the device it actually ran on.

    Bit-reproducibility holds per device, never across them (AGENTS.md 2.3), so the
    device string is the part of the record that makes a warp result reproducible
    at all.
    """
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = VoxelGrid.uniform_water(shape=(8, 8, 8), spacing=(0.5, 0.5, 0.5))
    result = WarpEngine(
        grid=grid,
        cross_sections=AnalyticCrossSections(geometry_densities=grid.max_density_by_material()),
        device="cpu",
    ).run(_beam(), n_histories=64, n_batches=2, seed=SEED)

    assert result.provenance is not None
    assert result.provenance.backend == "warp"
    assert result.provenance.device == "cpu"
    assert result.provenance.seed == SEED
    assert result.provenance.version == pyRadMC.__version__


def test_provenance_does_not_perturb_the_dose(engine: ReferenceEngine) -> None:
    """A record is a record: attaching it must not touch the transport.

    Two runs at the same seed stay bit-identical, which they would not be if the
    provenance construction had consumed a random draw or reordered anything.
    """
    common = dict(n_histories=16, n_batches=2, seed=SEED)
    first = engine.run(_beam(), **common)
    second = engine.run(_beam(), **common)

    assert (first.dose == second.dose).all()
    assert first.provenance == second.provenance
