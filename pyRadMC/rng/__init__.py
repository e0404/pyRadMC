"""RNG interface and per-target shims.

Physics routines import :func:`uniform` from here and call ``uniform(state)`` — nothing
else (AGENTS.md section 2.6). The binding below is the host implementation used by the
``ref`` backend; kernel targets (Phase 2+) bind their own ``uniform`` at kernel compile
time and never route through this module.
"""

from pyRadMC.rng.host import HostRNG, uniform
from pyRadMC.rng.interface import RNG, RNGState

__all__ = ["RNG", "HostRNG", "RNGState", "uniform"]
