"""Optional adapters that bridge external inputs to the engine's core types.

Adapters live outside the core (AGENTS.md 6): each may carry its own optional
dependency, declared as an extra in ``pyproject.toml``, and the core never imports one.
The CT adapter (:mod:`pyradmc.adapters.ct`) turns a patient CT into a
:class:`~pyradmc.geometry.grid.VoxelGrid`; the pyRadPlan adapter is planned to follow.
"""

from __future__ import annotations
