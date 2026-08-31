"""Single-source physics, compiled to ``@wp.func`` for the Warp targets.

The physics modules under :mod:`pyradmc.physics` are pure, scalar, and allocation-free
precisely so that one source serves every target (AGENTS.md 2.5). This module is the
Warp side of that bargain: it re-imports the **unmodified** physics source files with
two import-time substitutions, then wraps each public function with ``wp.func``:

``pyradmc.rng``
    Replaced by a shim exposing the kernel-side ``uniform`` and the Warp RNG state
    type (:mod:`pyradmc.rng.warp_shim`), so ``uniform(state)`` binds to the device
    generator — the binding the RNG interface documents for kernel targets.
``math``
    Replaced by a shim forwarding every attribute of the real module (Warp resolves
    genuine ``math`` functions to its intrinsics) except ``log1p``, which Warp lacks
    and which becomes ``wp.log(1.0 + x)``. In float32 the precision distinction is
    void; the float64 reference keeps the true ``log1p``.

Warp rejects ``exec``-generated sources, so substitution happens at import machinery
level: the physics files on disk are what compiles, and the reference backend and the
Warp kernels can never drift apart. ``sys.modules`` and the ``pyradmc.physics``
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

from pyradmc.rng import warp_shim

__all__ = ["warp_physics"]

# Modules to re-import under the shims, mapped to the functions to wrap with
# wp.func. None means the module's whole __all__ (the physics modules export
# nothing but kernel-compilable functions); data modules name their kernel-side
# lookups explicitly, since their __all__ also carries host-side builders.
_KERNEL_MODULES: dict[str, tuple[str, ...] | None] = {
    "pyradmc.physics.path": None,
    "pyradmc.physics.channel": None,
    "pyradmc.physics.direction": None,
    "pyradmc.physics.compton": None,
    "pyradmc.physics.moller": None,
    "pyradmc.physics.msc": None,
    "pyradmc.physics.gs": None,
    "pyradmc.physics.brems": None,
    "pyradmc.physics.rayleigh": None,
    "pyradmc.physics.roulette": None,
    "pyradmc.data.tables": ("lookup_loglinear_1d", "lookup_loglinear_2d"),
    "pyradmc.geometry.grid": (
        "point_inside",
        "point_axis_index",
        "slab_entry_distance",
        "distance_to_voxel_boundary",
    ),
    "pyradmc.geometry.cylinder": (
        "cylinder_contains",
        "cylinder_radius_squared",
        "edge_bin_index",
    ),
    "pyradmc.transport.electron": ("substep_pieces",),
    "pyradmc.geometry.collimation": (
        "_clip_len",
        "_affine_interval_length",
        "_mlc_tip_length",
        "jaw_path_length",
        "mlc_path_length",
    ),
}

_MISSING = object()


@wp.func
def _log1p(x: float) -> float:
    """log(1 + x); Warp has no log1p intrinsic. Exact enough for float32 kernels."""
    return wp.log(1.0 + x)


@wp.func
def _expm1(x: float) -> float:
    """exp(x) - 1; Warp has no expm1 intrinsic.

    The plain subtraction cancels to zero for |x| below float32 epsilon, which
    is exactly where the GS sampler's anchor ``1 - e^(-theta2/2)`` lives on the
    shortest substeps; the second-order Taylor branch keeps those anchors exact
    to float32 (its truncation error, |x|^3/6 at the switch point, is ~1e-13 —
    far below one ulp of the result).
    """
    if wp.abs(x) < 1.0e-4:
        return x * (1.0 + 0.5 * x)
    return wp.exp(x) - 1.0


def _make_math_shim() -> types.ModuleType:
    """Build the real math module with ``log1p``/``expm1`` swapped for Warp forms."""
    shim = types.ModuleType("math")
    for name in dir(_host_math):
        if not name.startswith("_"):
            setattr(shim, name, getattr(_host_math, name))
    shim.log1p = _log1p
    shim.expm1 = _expm1
    return shim


def _make_rng_shim() -> types.ModuleType:
    """Build a stand-in for ``pyradmc.rng`` with the kernel-side uniform and state."""
    shim = types.ModuleType("pyradmc.rng")
    shim.uniform = warp_shim.uniform
    shim.RNGState = warp_shim.WarpRNGState
    return shim


def _make_handles_shim() -> types.ModuleType:
    """Build a stand-in for ``pyradmc.data.handles`` with Warp array types."""
    shim = types.ModuleType("pyradmc.data.handles")
    shim.Table1D = wp.array(dtype=float)
    shim.Table2D = wp.array2d(dtype=float)
    shim.Table3D = wp.array3d(dtype=float)
    return shim


def _load() -> types.SimpleNamespace:
    """Re-import the kernel-function sources under the shims and wrap with wp.func."""
    patched = {
        "math": _make_math_shim(),
        "pyradmc.rng": _make_rng_shim(),
        "pyradmc.data.handles": _make_handles_shim(),
    }

    # Everything importable before the patch window opens. Any pyradmc module that
    # first arrives *inside* the window was imported under the shims and may have
    # bound the kernel-side rng at module level; it must not survive the restore
    # (the leak recorded in AGENTS.md: geometry.spectrum kept the shim ``uniform``
    # and broke the reference backend's spectral sources in-process).
    modules_before = set(sys.modules)

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
        # Evict pyradmc modules first imported inside the window (transitive
        # dependencies of the kernel modules that are not themselves in
        # _KERNEL_MODULES): they bound the shims and would poison later host
        # imports. Dropping both the sys.modules entry and the parent-package
        # attribute forces a clean re-import — `from package import submodule`
        # returns the stale attribute if only the sys.modules entry goes.
        for name in set(sys.modules) - modules_before - set(saved_modules):
            if not name.startswith("pyradmc"):
                continue  # third-party imports bind no pyradmc shims
            module = sys.modules.pop(name)
            parent_name, _, short_name = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, short_name, None) is module:
                delattr(parent, short_name)
    return namespace


_cache: types.SimpleNamespace | None = None


def warp_physics() -> types.SimpleNamespace:
    """Return the ``@wp.func``-compiled physics functions, loaded once and shared.

    Attributes are the public functions of every :mod:`pyradmc.physics` module
    (their names are globally unique), each callable from Warp kernels on any
    device. RNG-consuming functions take a :class:`pyradmc.rng.warp_shim.WarpRNGState`.
    """
    global _cache
    if _cache is None:
        _cache = _load()
    return _cache
