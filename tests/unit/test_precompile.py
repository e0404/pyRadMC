"""Precompiler guards.

Fast-tier checks that need no data: argument validation and the strategy enum. The full
real-library compilation of both strategies is exercised in the validation tier
(``tests/validation/test_tabulated_water_compile.py``).
"""

from __future__ import annotations

import pytest

from pyradmc.data.materials import MATERIALS
from pyradmc.data.tabulated.precompile import (
    ElectronStoppingStrategy,
    compile_materials,
    compile_water,
)


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
