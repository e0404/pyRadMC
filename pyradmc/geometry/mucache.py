"""Host-side memo of attenuation curves queried through a ``CrossSectionSource``.

The collimator pre-solve and the stack transmission model both flatten
``mu_over_rho`` on a log-energy grid by calling the scalar query API once per node
(AGENTS.md 2.6 forbids anything but that interface). For a tabulated source that is
several milliseconds of Python per material per call, and an aperture optimizer
repeats it with the *same* materials on every iteration. The curves are pure
functions of ``(source, material, energy grid)``, and a source is immutable after
construction, so they are memoised here per source object.

Each curve is exactly what the removed per-call loop produced, element for element
(the same scalar query on the same ``float(e)`` grid node), so a cache hit and a
cold build are bit-identical; the arrays are returned read-only because they are
shared. Entries die with their source (``weakref`` callback), and the key is the
object's identity, never its address alone, so a recycled ``id`` cannot alias.
Densities are *not* part of the key: a caller scales the curve itself.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyradmc.data.interface import CrossSectionSource

_F64 = NDArray[np.float64]

_MAX_ASSEMBLED = 32  # bound on memoised assembled tables per source

_lock = threading.RLock()
_entries: dict[int, _Entry] = {}


class _Entry:
    """Per-source memo: energy grids, mu curves, and caller-assembled objects."""

    def __init__(self, source: CrossSectionSource) -> None:
        key = id(source)

        def on_death(ref: weakref.ref[CrossSectionSource]) -> None:
            _evict(key, ref)

        self.ref = weakref.ref(source, on_death)
        self.grids: dict[tuple[float, float, int], _F64] = {}
        self.curves: dict[tuple[Any, ...], _F64] = {}
        self.assembled: dict[tuple[Any, ...], Any] = {}


def _evict(key: int, ref: weakref.ref[CrossSectionSource]) -> None:
    with _lock:
        entry = _entries.get(key)
        if entry is not None and entry.ref is ref:
            del _entries[key]


def _entry(source: CrossSectionSource) -> _Entry | None:
    """Return the memo for ``source``, or ``None`` if it cannot be weakly referenced."""
    with _lock:
        entry = _entries.get(id(source))
        if entry is not None and entry.ref() is source:
            return entry
        try:
            entry = _Entry(source)
        except TypeError:  # not weak-referenceable: fall back to uncached
            return None
        _entries[id(source)] = entry
        return entry


def clear() -> None:
    """Drop every memoised curve (test instrument and escape hatch)."""
    with _lock:
        _entries.clear()


def energy_grid(e_lo: float, e_hi: float, n: int) -> _F64:
    """Geometric grid ``np.geomspace(e_lo, e_hi, n)``, read-only."""
    grid = np.geomspace(e_lo, e_hi, n)
    grid.setflags(write=False)
    return grid


def mu_curve(
    source: CrossSectionSource,
    material: int,
    e_lo: float,
    e_hi: float,
    n: int,
    process: int | None,
) -> tuple[_F64, _F64]:
    """Mass attenuation coefficient (cm^2/g) of ``material`` on a geometric grid.

    Returns ``(energies, values)``. ``process=None`` queries ``mu_over_rho_total``;
    otherwise ``mu_over_rho`` for that :class:`~pyradmc.data.interface.PhotonProcess`.
    The scalar query is evaluated at ``float(e)`` for each grid node, as the
    uncached loops did. Both arrays are read-only and shared between callers.
    """
    entry = _entry(source)
    grid_key = (float(e_lo), float(e_hi), int(n))
    if entry is not None:
        with _lock:
            energies = entry.grids.get(grid_key)
            if energies is not None:
                cached = entry.curves.get((grid_key, int(material), process))
                if cached is not None:
                    return energies, cached
    energies = energy_grid(e_lo, e_hi, n)
    if process is None:
        values = np.array([source.mu_over_rho_total(float(e), material) for e in energies])
    else:
        values = np.array([source.mu_over_rho(float(e), material, process) for e in energies])
    values.setflags(write=False)
    if entry is not None:
        with _lock:
            energies = entry.grids.setdefault(grid_key, energies)
            entry.curves[(grid_key, int(material), process)] = values
    return energies, values


def assembled(source: CrossSectionSource, key: tuple[Any, ...]) -> Any | None:
    """Return a previously stored assembled object for ``(source, key)``, if any."""
    entry = _entry(source)
    if entry is None:
        return None
    with _lock:
        return entry.assembled.get(key)


def store_assembled(source: CrossSectionSource, key: tuple[Any, ...], value: Any) -> None:
    """Remember an assembled object (bounded, oldest evicted first)."""
    entry = _entry(source)
    if entry is None:
        return
    with _lock:
        if len(entry.assembled) >= _MAX_ASSEMBLED:
            del entry.assembled[next(iter(entry.assembled))]
        entry.assembled[key] = value
