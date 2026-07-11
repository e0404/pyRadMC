"""Single-source physics, compiled to ``@wp.func`` for the Warp targets.

The physics modules under :mod:`pyRadMC.physics` are pure, scalar, and allocation-free
precisely so that one source serves every target (AGENTS.md 2.5). This module is the
Warp side of that bargain: it re-imports the **unmodified** physics source files with
two import-time substitutions, then wraps each public function with ``wp.func``:

``pyRadMC.rng``
    Replaced by a shim exposing the kernel-side ``uniform`` and the Warp RNG state
    type (:mod:`pyRadMC.rng.warp_shim`), so ``uniform(state)`` binds to the device
    generator — the binding the RNG interface documents for kernel targets.
``math``
    Replaced by a shim forwarding every attribute of the real module (Warp resolves
    genuine ``math`` functions to its intrinsics) except ``log1p``, which Warp lacks
    and which becomes ``wp.log(1.0 + x)``. In float32 the precision distinction is
    void; the float64 reference keeps the true ``log1p``.

Warp rejects ``exec``-generated sources, so substitution happens at import machinery
level: the physics files on disk are what compiles, and the reference backend and the
Warp kernels can never drift apart. ``sys.modules`` and the ``pyRadMC.physics``
package attributes are restored afterwards; the host modules are contract-tested to
stay untouched (``tests/physics/test_warp_physics.py``).

Loading is done once and cached: every kernel must reference the same compiled
function objects. Not thread-safe — first call happens at backend construction.

This module is Warp kernel plumbing: exempt from ``mypy --strict`` (AGENTS.md 5).
"""

from __future__ import annotations

import importlib
import math as _host_math
import sys
import types

import warp as wp

from pyRadMC.rng import warp_shim

__all__ = ["warp_physics"]

# Modules to re-import under the shims, mapped to the functions to wrap with
# wp.func. None means the module's whole __all__ (the physics modules export
# nothing but kernel-compilable functions); data modules name their kernel-side
# lookups explicitly, since their __all__ also carries host-side builders.
_KERNEL_MODULES: dict[str, tuple[str, ...] | None] = {
    "pyRadMC.physics.path": None,
    "pyRadMC.physics.channel": None,
    "pyRadMC.physics.direction": None,
    "pyRadMC.physics.compton": None,
    "pyRadMC.physics.moller": None,
    "pyRadMC.physics.msc": None,
    "pyRadMC.physics.brems": None,
    "pyRadMC.physics.rayleigh": None,
    "pyRadMC.physics.roulette": None,
    "pyRadMC.data.tables": ("lookup_loglinear_1d", "lookup_loglinear_2d"),
    "pyRadMC.geometry.grid": ("point_inside", "point_axis_index", "slab_entry_distance"),
}

_MISSING = object()


@wp.func
def _log1p(x: float) -> float:
    """log(1 + x); Warp has no log1p intrinsic. Exact enough for float32 kernels."""
    return wp.log(1.0 + x)


def _make_math_shim() -> types.ModuleType:
    """Build the real math module with ``log1p`` swapped for the Warp form."""
    shim = types.ModuleType("math")
    for name in dir(_host_math):
        if not name.startswith("_"):
            setattr(shim, name, getattr(_host_math, name))
    shim.log1p = _log1p
    return shim


def _make_rng_shim() -> types.ModuleType:
    """Build a stand-in for ``pyRadMC.rng`` with the kernel-side uniform and state."""
    shim = types.ModuleType("pyRadMC.rng")
    shim.uniform = warp_shim.uniform
    shim.RNGState = warp_shim.WarpRNGState
    return shim


def _make_handles_shim() -> types.ModuleType:
    """Build a stand-in for ``pyRadMC.data.handles`` with Warp array types."""
    shim = types.ModuleType("pyRadMC.data.handles")
    shim.Table1D = wp.array(dtype=float)
    shim.Table2D = wp.array2d(dtype=float)
    return shim


def _load() -> types.SimpleNamespace:
    """Re-import the kernel-function sources under the shims and wrap with wp.func."""
    patched = {
        "math": _make_math_shim(),
        "pyRadMC.rng": _make_rng_shim(),
        "pyRadMC.data.handles": _make_handles_shim(),
    }

    saved_modules = {name: sys.modules.pop(name, None) for name in (*patched, *_KERNEL_MODULES)}
    # Re-importing a submodule also rebinds it as an attribute of its parent
    # package; save those bindings so the host packages are restored exactly.
    parents = {name: importlib.import_module(name.rsplit(".", 1)[0]) for name in _KERNEL_MODULES}
    saved_attrs = {
        name: getattr(parents[name], name.rsplit(".", 1)[1], _MISSING) for name in _KERNEL_MODULES
    }

    namespace = types.SimpleNamespace()
    try:
        sys.modules.update(patched)
        for name, wrap_names in _KERNEL_MODULES.items():
            module = importlib.import_module(name)
            for public_name in wrap_names if wrap_names is not None else module.__all__:
                fn = getattr(module, public_name)
                wrapped = wp.func(fn)
                # Rebinding inside the module makes intra-module calls (e.g.
                # compton_electron_cos_theta -> compton_cos_theta) resolve to the
                # wrapped function at Warp codegen time.
                setattr(module, public_name, wrapped)
                setattr(namespace, public_name, wrapped)
    finally:
        for name, saved in saved_modules.items():
            if saved is not None:
                sys.modules[name] = saved
            else:
                sys.modules.pop(name, None)
        for name, saved in saved_attrs.items():
            short_name = name.rsplit(".", 1)[1]
            if saved is _MISSING:
                if hasattr(parents[name], short_name):
                    delattr(parents[name], short_name)
            else:
                setattr(parents[name], short_name, saved)
    return namespace


_cache: types.SimpleNamespace | None = None


def warp_physics() -> types.SimpleNamespace:
    """Return the ``@wp.func``-compiled physics functions, loaded once and shared.

    Attributes are the public functions of every :mod:`pyRadMC.physics` module
    (their names are globally unique), each callable from Warp kernels on any
    device. RNG-consuming functions take a :class:`pyRadMC.rng.warp_shim.WarpRNGState`.
    """
    global _cache
    if _cache is None:
        _cache = _load()
    return _cache
