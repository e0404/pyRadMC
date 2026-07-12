"""Precompiler guards (Phase 5, tabulated, slice C).

Fast-tier checks that need no data: the strategy dispatch and argument validation. The
full real-library compilation is exercised in the validation tier
(``tests/validation/test_tabulated_water_compile.py``).
"""

from __future__ import annotations

import pytest

from pyRadMC.data.tabulated.precompile import ElectronStoppingStrategy, compile_water


def test_eedl_strategy_is_not_yet_implemented() -> None:
    """The 'eedl' stopping strategy raises until restricted collision stopping exists."""
    with pytest.raises(NotImplementedError, match="eedl"):
        compile_water("", "", strategy=ElectronStoppingStrategy.EEDL)


def test_e_max_must_cover_the_transport_range() -> None:
    """A grid top below the cutoffs is rejected before any parsing."""
    with pytest.raises(ValueError, match="does not cover"):
        compile_water("", "", e_max=0.01)
