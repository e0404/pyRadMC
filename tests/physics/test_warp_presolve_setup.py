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
