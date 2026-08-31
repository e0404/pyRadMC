"""The pure-NumPy reference backend: the oracle. Never optimized (AGENTS.md 2.2)."""

from pyradmc.backends.ref.engine import ReferenceEngine, TransportResult

__all__ = ["ReferenceEngine", "TransportResult"]
