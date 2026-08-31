"""RNG interface and per-target shims.

Physics routines import :func:`uniform` from here and call ``uniform(state)`` — nothing
else (AGENTS.md section 2.6). The binding below is the host implementation used by the
``ref`` backend; kernel targets bind their own ``uniform`` (see
:mod:`pyradmc.rng.warp_shim`) at kernel compile time and never route through this module.
"""

from pyradmc.rng.host import HostRNG, uniform
from pyradmc.rng.interface import RNG, RNGState

__all__ = ["RNG", "HostRNG", "RNGState", "uniform"]
