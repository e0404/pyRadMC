"""Dij tier scaffold. Dij scoring is Phase 3; earlier phases must not begin it (AGENTS.md 7).

When this tier lands, it owns: Dij column consistency against single-beamlet dose
runs, truncation tested against DVH endpoints (never a matrix norm, AGENTS.md 2.8),
and the fluence-sum identity (sum over beamlets at unit weights equals the open-field
dose).
"""

from __future__ import annotations

import pytest


def test_dij_tier_pending() -> None:
    """Placeholder keeping the tier collected and visibly skipped until Phase 3."""
    pytest.skip("Phase 3: Dij scoring waits for electron transport and the Warp backend")
