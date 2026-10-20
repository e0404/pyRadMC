"""Array-handle type aliases for single-source, kernel-compilable data access.

The table lookups in :mod:`pyradmc.data.tables` are pure scalar functions of scalars
and *array handles* (AGENTS.md 2.5). What an array handle is differs per target: a
NumPy ``ndarray`` on the host, a ``wp.array`` in Warp kernels. These aliases are the
indirection point — the same trick :data:`pyradmc.rng.interface.RNGState` plays for
RNG state.

Host code (and mypy) sees NumPy arrays. The Warp physics loader substitutes this
module at re-import time with one whose aliases are Warp array types, so the
annotations on the lookup functions resolve to what the codegen needs — without the
lookup source changing by a character.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = ["Table1D", "Table2D", "Table3D", "Vector3Field"]

Table1D: TypeAlias = npt.NDArray[np.floating]
"""One quantity on an energy grid, indexed ``[energy_node]``."""

Table2D: TypeAlias = npt.NDArray[np.floating]
"""One quantity per material and energy, indexed ``[material, energy_node]``."""

Table3D: TypeAlias = npt.NDArray[np.floating]
"""A rectangle of tables, indexed ``[node_i, node_j, entry]``; partial indexing
``table[i, j]`` yields a :data:`Table1D` row on both targets (NumPy views on the
host, Warp array views in kernels), which is what lets a row-consuming sampler
run unchanged over a flat uploaded grid."""

Vector3Field: TypeAlias = npt.NDArray[np.floating]
"""A 3-vector per cell of a rectilinear grid, indexed ``[i, j, k, component]`` on the
host (``(nx, ny, nz, 3)``) and ``wp.array3d(dtype=wp.vec3)`` in kernels. Partial
indexing ``values[i, j, k]`` yields the vector on both targets (a length-3 NumPy
view on the host, a ``vec3`` in kernels); components are then read as ``v[0]``,
``v[1]``, ``v[2]`` — never with a fourth subscript, which Warp rejects. Used by
:func:`pyradmc.geometry.field.field_at`."""
