"""Perf tier scaffold: benchmarks with recorded baselines. Alerts, not pass/fail.

Runs only with ``-m perf``. The reference backend is deliberately never optimized
(AGENTS.md section 2.2), so it gets no benchmark; baselines start with the Warp
backend (Phase 2).
"""

from __future__ import annotations

import pytest


@pytest.mark.perf
def test_perf_tier_pending() -> None:
    """Placeholder keeping the tier collected and visibly skipped until Phase 2."""
    pytest.skip("Phase 2: benchmarks start with the Warp backend")
