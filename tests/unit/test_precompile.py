"""Precompiler guards (Phase 5, tabulated, slice C).

Fast-tier checks that need no data: argument validation and the strategy enum. The full
real-library compilation of both strategies is exercised in the validation tier
(``tests/validation/test_tabulated_water_compile.py``).
"""

from __future__ import annotations

import pytest

from pyRadMC.data.tabulated.precompile import ElectronStoppingStrategy, compile_water


def test_e_max_must_cover_the_transport_range() -> None:
    """A grid top below the cutoffs is rejected before any parsing."""
    with pytest.raises(ValueError, match="does not cover"):
        compile_water("", "", e_max=0.01)


def test_strategy_values() -> None:
    """Both stopping strategies are selectable by their string values."""
    assert ElectronStoppingStrategy("berger-seltzer") is ElectronStoppingStrategy.BERGER_SELTZER
    assert ElectronStoppingStrategy("eedl") is ElectronStoppingStrategy.EEDL
