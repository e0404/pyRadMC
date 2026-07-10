"""Validation tier scaffold: PDD and profiles against published data, nightly/tagged.

Runs only with ``-m validation``. Gamma index comparison belongs here and only here
(AGENTS.md section 4). Meaningful validation of the KERMA-mode Phase 0 engine against
published depth-dose data is limited (no electron buildup region); full PDD and
heterogeneous-slab comparisons land with electron transport (Phase 1).
"""

from __future__ import annotations

import pytest


@pytest.mark.validation
def test_validation_tier_pending() -> None:
    """Placeholder keeping the tier collected and visibly skipped until Phase 1."""
    pytest.skip("Phase 1: PDD validation requires electron transport")
