"""Precompiler guards.

Fast-tier checks that need no data: argument validation and the strategy enum. The full
real-library compilation of both strategies is exercised in the validation tier
(``tests/validation/test_tabulated_water_compile.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc import RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV
from pyradmc.data.materials import MATERIALS
from pyradmc.data.tabulated.precompile import (
    COMPILED_GRID_E_REF,
    ElectronStoppingStrategy,
    compile_materials,
    compile_water,
    compiled_grids,
)


def test_default_grids_are_the_historical_ones() -> None:
    """At the default 30 MeV top the compiled nodes are exactly what they always were."""
    photon, electron, coherent_x = compiled_grids(pcut=0.05, ecut=0.2, e_max=30.0, n_points=2048)
    np.testing.assert_array_equal(photon, np.geomspace(0.025, 30.0, 2048))
    np.testing.assert_array_equal(electron, np.geomspace(0.1, 30.0, 2048))
    x_max = RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV * 30.0
    np.testing.assert_array_equal(coherent_x, np.geomspace(x_max * 1.0e-7, x_max, 512))


def test_a_wider_table_only_appends_nodes() -> None:
    """Compiling for 250 MeV leaves every node of the 30 MeV table where it was.

    Widening a table for a higher-energy source must not move a single node a
    lower-energy run interpolates on.
    """
    narrow = compiled_grids(pcut=0.05, ecut=0.2, e_max=COMPILED_GRID_E_REF, n_points=2048)
    wide = compiled_grids(pcut=0.05, ecut=0.2, e_max=250.0, n_points=2048)
    for small, large in zip(narrow, wide, strict=True):
        assert large.size > small.size
        np.testing.assert_array_equal(large[: small.size], small)
    assert wide[0][-1] >= 250.0 and wide[1][-1] >= 250.0


def test_a_narrower_request_keeps_the_reference_span() -> None:
    """Below the reference top the grid does not shrink (shrinking would move nodes)."""
    photon, electron, _ = compiled_grids(pcut=0.05, ecut=0.2, e_max=10.0, n_points=64)
    np.testing.assert_array_equal(photon, np.geomspace(0.025, COMPILED_GRID_E_REF, 64))
    np.testing.assert_array_equal(electron, np.geomspace(0.1, COMPILED_GRID_E_REF, 64))


def test_e_max_must_cover_the_transport_range() -> None:
    """A grid top below the cutoffs is rejected before any parsing."""
    with pytest.raises(ValueError, match="does not cover"):
        compile_water("", "", e_max=0.01)


def test_n_materials_must_be_a_registry_prefix() -> None:
    """Compiled rows are registry-index-aligned; out-of-range counts are rejected."""
    with pytest.raises(ValueError, match="outside the registry"):
        compile_materials("", "", n_materials=0)
    with pytest.raises(ValueError, match="outside the registry"):
        compile_materials("", "", n_materials=len(MATERIALS) + 1)


def test_strategy_values() -> None:
    """Both stopping strategies are selectable by their string values."""
    assert ElectronStoppingStrategy("berger-seltzer") is ElectronStoppingStrategy.BERGER_SELTZER
    assert ElectronStoppingStrategy("eedl") is ElectronStoppingStrategy.EEDL
