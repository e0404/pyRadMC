"""Per-call setup of the device pre-solve is cached, and caching changes no number.

The device pre-solve rebuilt its attenuation tables, flattened and uploaded the
stack, and re-uploaded the tables on every call. These tests pin that a repeat
call with the same stack and cross-section source does none of that, and that the
exit phase space is **bit-identical** to a cold call (per target: cpu and cuda
separately, AGENTS.md 2.3). The device append order is not deterministic (atomic
counter), so populations are compared as lexicographically sorted record tables.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry import mucache
from pyradmc.geometry.head import AirColumn
from tests.physics.test_warp_presolve import DEVICES, N, _require, _source, _stack

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.warp, pytest.mark.filterwarnings("ignore:.*latent variance:UserWarning")]

KEYS = ("particle_type", "energy", "x", "y", "z", "ux", "uy", "uz", "weight")


def _records(phase_space) -> np.ndarray:
    columns = phase_space.columns()
    table = np.column_stack([np.asarray(columns[k], dtype=np.float64) for k in KEYS])
    return table[np.lexsort(table.T[::-1])]


def _run(device: str, stack, xs, mode: str = "attenuation", air=None):
    from pyradmc.backends.warp.presolve import presolve_head_device

    return presolve_head_device(
        _source(),
        cross_sections=xs,
        n_histories=N,
        seed=11,
        exit_z=60.0,
        stack=stack,
        air=air,
        mode=mode,
        device=device,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["attenuation", "first_compton"])
def test_warm_call_is_bit_identical_to_cold_call(device: str, mode: str) -> None:
    _require(device)
    air = (
        AirColumn(z_start=51.0, z_end=60.0, material=WATER, density=1.0e-3)
        if mode == "first_compton"
        else None
    )
    xs = AnalyticCrossSections()
    stack = _stack()
    mucache.clear()
    cold = _records(_run(device, stack, xs, mode, air))
    warm = _records(_run(device, stack, xs, mode, air))  # all caches hit
    np.testing.assert_array_equal(cold, warm)
    # A fresh-but-equal stack object and source: a cold path again, same bytes.
    fresh = _records(_run(device, _stack(), AnalyticCrossSections(), mode, air))
    np.testing.assert_array_equal(cold, fresh)


@pytest.mark.parametrize("device", DEVICES)
def test_repeat_call_skips_table_build_and_uploads(device: str, monkeypatch) -> None:
    _require(device)
    from pyradmc.backends.warp import presolve

    xs = AnalyticCrossSections()
    stack = _stack()
    _run(device, stack, xs)  # warm everything
    calls = {"upload_stack": 0, "upload_mu": 0}
    real_stack, real_mu = presolve.upload_stack, presolve._upload_mu

    def counting_stack(*args, **kwargs):
        calls["upload_stack"] += 1
        return real_stack(*args, **kwargs)

    def counting_mu(*args, **kwargs):
        calls["upload_mu"] += 1
        return real_mu(*args, **kwargs)

    monkeypatch.setattr(presolve, "upload_stack", counting_stack)
    monkeypatch.setattr(presolve, "_upload_mu", counting_mu)
    monkeypatch.setattr(
        type(xs),
        "mu_over_rho",
        lambda *a, **k: pytest.fail("a warm pre-solve must not query cross-sections"),
    )
    _run(device, stack, xs)
    assert calls == {"upload_stack": 0, "upload_mu": 0}


def test_as_float32_does_not_copy_a_matching_array() -> None:
    from pyradmc.backends.warp.presolve import _as_float32

    f32 = np.arange(8, dtype=np.float32)
    assert _as_float32(f32) is f32
    f64 = np.arange(8, dtype=np.float64)
    converted = _as_float32(f64)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, f64.astype(np.float32))


def _tweaked_stack(shift: float):
    """The test stack with its MLC tips moved: same shapes, different leaf geometry."""
    import dataclasses

    base = _stack()
    jaw, mlc = base.devices
    moved = dataclasses.replace(
        mlc,
        tips_neg=tuple(t - shift for t in mlc.tips_neg),
        tips_pos=tuple(t + shift for t in mlc.tips_pos),
    )
    return type(base)(frame=base.frame, devices=(jaw, moved))


def _drop_caches() -> None:
    from pyradmc.backends.warp import presolve

    mucache.clear()
    presolve._stack_cache.clear()
    presolve._mu_device_cache.clear()


@pytest.mark.parametrize("device", DEVICES)
def test_a_different_stack_is_never_served_the_cached_geometry(device: str) -> None:
    _require(device)
    xs = AnalyticCrossSections()
    stack_a, stack_b = _tweaked_stack(0.0), _tweaked_stack(1.4)
    _drop_caches()
    a = _records(_run(device, stack_a, xs))
    b_warm = _records(_run(device, stack_b, xs))  # caches hold A
    _drop_caches()
    b_cold = _records(_run(device, _tweaked_stack(1.4), xs))
    assert b_warm.shape == b_cold.shape
    np.testing.assert_array_equal(b_warm, b_cold)
    assert a.shape != b_warm.shape or not np.array_equal(a, b_warm)


@pytest.mark.parametrize("device", DEVICES)
def test_a_retuned_density_is_never_served_the_cached_table(device: str) -> None:
    _require(device)
    xs = AnalyticCrossSections()
    stack_a, stack_b = _density_stack(1.0), _density_stack(0.6)
    _drop_caches()
    _run(device, stack_a, xs)
    warm = _records(_run(device, stack_b, xs))
    _drop_caches()
    cold = _records(_run(device, _density_stack(0.6), xs))
    np.testing.assert_array_equal(warm, cold)
    assert not np.array_equal(warm, _records(_run(device, stack_a, xs)))


def _density_stack(density: float):
    import dataclasses

    base = _stack()
    return type(base)(
        frame=base.frame,
        devices=tuple(dataclasses.replace(d, density=density) for d in base.devices),
    )


def test_cpu_and_cuda_uploads_are_kept_apart() -> None:
    _require("cuda:0")
    xs = AnalyticCrossSections()
    stack = _stack()
    _drop_caches()
    cpu_cold = _records(_run("cpu", stack, xs))
    cuda_cold = _records(_run("cuda:0", stack, xs))  # same stack and table objects
    np.testing.assert_array_equal(cpu_cold, _records(_run("cpu", stack, xs)))
    np.testing.assert_array_equal(cuda_cold, _records(_run("cuda:0", stack, xs)))
    _drop_caches()
    np.testing.assert_array_equal(cuda_cold, _records(_run("cuda:0", _stack(), xs)))


def test_numpy_scalar_densities_are_coerced_to_float() -> None:
    """A float32 density used to build float32 tables; it is now a Python float."""
    xs = AnalyticCrossSections()
    base = _stack()
    import dataclasses

    stack32 = type(base)(
        frame=base.frame,
        devices=tuple(dataclasses.replace(d, density=np.float32(0.7)) for d in base.devices),
    )
    stack64 = type(base)(
        frame=base.frame,
        devices=tuple(dataclasses.replace(d, density=float(np.float32(0.7))) for d in base.devices),
    )
    _drop_caches()
    a = _records(_run("cpu", stack32, xs))
    _drop_caches()
    b = _records(_run("cpu", stack64, xs))
    np.testing.assert_array_equal(a, b)


_GC_SCRIPT = """
import gc
gc.set_threshold(1)  # collect on nearly every allocation, including inside the lock
from pyradmc.backends.warp import presolve
from tests.physics.test_warp_presolve import _stack
presolve.upload_stack = lambda compiled, device: object()  # isolate the cache locking
for _ in range(20000):
    stack = _stack()
    holder = {"stack": stack}
    holder["me"] = holder  # a reference cycle: only the cyclic GC can free the stack
    presolve._device_stack(stack, "cpu")
    del stack, holder
gc.collect()
print("ok")
"""


def test_gc_freeing_a_cached_stack_never_deadlocks_the_cache_lock() -> None:
    """Weakref callbacks fire inside cyclic GC, possibly while the lock is held."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root)}
    done = subprocess.run(
        [sys.executable, "-c", _GC_SCRIPT],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("ok")
