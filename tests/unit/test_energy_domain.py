"""Data sources declare the energy range they are valid for, and it is enforced.

A query, a table or a run above a source's ``energy_max`` used to clamp silently
to whatever the data's top edge happened to be (the restricted-range grid at
64 MeV, a compiled table at its ``e_max``, the analytic fits by extrapolation).
That is a silent transport bias of exactly the kind AGENTS.md 2.6 exists to rule
out, so the declared domain is now checked fail-closed at every entry point:
the range lookup, the table build, and the engines' run entries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from pyradmc import ECUT_MEV, PCUT_MEV
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.grids import extendable_log_grid
from pyradmc.data.interface import DOMAIN_EDGE_RTOL, require_energy_domain
from pyradmc.data.materials import WATER
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, PencilBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED
from tests.unit.test_tabulated_format import synthetic_tables

# The pre-domain restricted-range grid, reproduced verbatim: the extended grid
# must keep these nodes so every result below 64 MeV is unchanged bit for bit.
_LEGACY_RANGE_TOP = 64.0
_LEGACY_RANGE_POINTS = 4096


class _WideAnalytic(AnalyticCrossSections):
    """Analytic water claiming a VHEE-sized domain: a grid-extension instrument.

    The restricted collision stopping it integrates is Berger-Seltzer, which holds
    at these energies; nothing here asserts the photon data is valid up there.
    """

    @property
    def energy_max(self) -> float:
        return 250.0


# -- the declaration ---------------------------------------------------------


def test_analytic_domain_is_its_calibration_range() -> None:
    """Pair and radiative anchors stop at 20 MeV; so does the analytic source."""
    assert AnalyticCrossSections().energy_max == 20.0


def test_tabulated_domain_is_the_compiled_grid_top() -> None:
    """A compiled table answers up to the lower of its two grid tops, no further."""
    data = synthetic_tables()  # both grids end at 10 MeV
    assert TabulatedCrossSections(data).energy_max == pytest.approx(10.0, rel=1e-12)


def test_require_energy_domain_refuses_above_and_admits_the_edge() -> None:
    xs = AnalyticCrossSections()
    require_energy_domain(xs, 20.0)
    require_energy_domain(xs, 20.0 * (1.0 + 0.5 * DOMAIN_EDGE_RTOL))
    with pytest.raises(ValueError, match="energy domain"):
        require_energy_domain(xs, 20.5)


# -- the table build ---------------------------------------------------------


def test_tables_above_the_domain_are_refused() -> None:
    with pytest.raises(ValueError, match="energy domain"):
        AnalyticCrossSections().build_tables(ecut=ECUT_MEV, pcut=PCUT_MEV, e_max=25.0, n_points=16)


def test_tables_at_the_engines_margin_are_admitted() -> None:
    """The Warp engine builds at ``max_energy * (1 + 1e-6)``; that must pass at 20 MeV."""
    tables = AnalyticCrossSections().build_tables(
        ecut=ECUT_MEV, pcut=PCUT_MEV, e_max=20.0 * (1.0 + 1.0e-6), n_points=16
    )
    assert tables.e_max > 20.0


def test_tabulated_tables_above_the_compiled_grid_are_refused() -> None:
    """Previously this clamped to the 10 MeV values without a word."""
    xs = TabulatedCrossSections(synthetic_tables())
    with pytest.raises(ValueError, match="energy domain"):
        xs.build_tables(ecut=0.2, pcut=0.05, e_max=12.0, n_points=16)


# -- the restricted-range lookup ----------------------------------------------


def test_range_above_the_domain_raises_instead_of_clamping() -> None:
    xs = AnalyticCrossSections()
    with pytest.raises(ValueError, match="energy domain"):
        xs.restricted_range(30.0, WATER, ECUT_MEV)


def test_range_grid_extends_to_a_wide_domain() -> None:
    """At 200 MeV the range is the integral of 1/S, not the 64 MeV value."""
    xs = _WideAnalytic()

    def integrand(e: float) -> float:
        return 1.0 / xs.restricted_stopping_power(e, WATER, ECUT_MEV)

    for energy in (100.0, 200.0, 250.0):
        expected, _ = quad(integrand, ECUT_MEV, energy, limit=200)
        assert xs.restricted_range(energy, WATER, ECUT_MEV) == pytest.approx(expected, rel=1e-4)


def test_range_grid_extension_keeps_every_legacy_result() -> None:
    """Below 64 MeV a wide-domain source answers bit-identically to the old grid."""
    narrow, wide = AnalyticCrossSections(), _WideAnalytic()
    energies = np.geomspace(ECUT_MEV, 20.0, 37)
    for e in energies:
        assert wide.restricted_range(float(e), WATER, ECUT_MEV) == narrow.restricted_range(
            float(e), WATER, ECUT_MEV
        )
        for path in (1.0e-3, 0.3, 2.0):
            assert wide.energy_after_mass_path(
                float(e), WATER, ECUT_MEV, path
            ) == narrow.energy_after_mass_path(float(e), WATER, ECUT_MEV, path)


# -- the shared grid rule -----------------------------------------------------


def test_extendable_grid_reproduces_the_reference_span_exactly() -> None:
    nodes = extendable_log_grid(0.2, 64.0, 4096, e_top=20.0)
    np.testing.assert_array_equal(nodes, np.geomspace(0.2, 64.0, 4096))


def test_extendable_grid_appends_at_the_same_spacing() -> None:
    base = np.geomspace(0.025, 30.0, 2048)
    wide = extendable_log_grid(0.025, 30.0, 2048, e_top=250.0)
    np.testing.assert_array_equal(wide[:2048], base)
    assert wide[-1] >= 250.0 and wide[-2] < 250.0
    dlog = np.diff(np.log(wide))
    np.testing.assert_allclose(dlog, dlog[0], rtol=1e-9)


def test_extendable_grid_rejects_a_degenerate_reference() -> None:
    with pytest.raises(ValueError):
        extendable_log_grid(1.0, 1.0, 16, e_top=2.0)
    with pytest.raises(ValueError):
        extendable_log_grid(1.0, 2.0, 1, e_top=2.0)


# -- the engines ---------------------------------------------------------------


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))


def test_reference_run_refuses_a_source_above_the_domain() -> None:
    engine = ReferenceEngine(grid=_grid(), cross_sections=AnalyticCrossSections(), rng=HostRNG())
    source = PencilBeamSource(50.0, (2.0, 2.0, -1.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="energy domain"):
        engine.run(source, n_histories=2, n_batches=2, seed=SEED)


def test_reference_dij_refuses_a_source_above_the_domain() -> None:
    engine = ReferenceEngine(grid=_grid(), cross_sections=AnalyticCrossSections(), rng=HostRNG())
    source = BeamletGridSource(50.0, -1.0, (0.0, 4.0), (0.0, 4.0), 2, 2)
    with pytest.raises(ValueError, match="energy domain"):
        engine.run_dij(source, n_histories_per_beamlet=2, n_batches=2, seed=SEED)


def test_warp_run_refuses_a_source_above_the_domain() -> None:
    pytest.importorskip("warp")
    from pyradmc.backends.warp.engine import WarpEngine

    engine = WarpEngine(grid=_grid(), cross_sections=AnalyticCrossSections(), device="cpu")
    source = PencilBeamSource(50.0, (2.0, 2.0, -1.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="energy domain"):
        engine.run(source, n_histories=2, n_batches=2, seed=SEED)
    beamlets = BeamletGridSource(50.0, -1.0, (0.0, 4.0), (0.0, 4.0), 2, 2)
    with pytest.raises(ValueError, match="energy domain"):
        engine.run_dij(beamlets, n_histories_per_beamlet=2, n_batches=2, seed=SEED)


def test_default_domain_is_unbounded_for_instruments() -> None:
    """Material-independent test instruments keep answering, as with n_materials."""
    from pyradmc.data.interface import CrossSectionSource

    assert CrossSectionSource.energy_max.fget is not None  # type: ignore[attr-defined]
    assert math.isinf(CrossSectionSource.energy_max.fget(object()))  # type: ignore[attr-defined]
