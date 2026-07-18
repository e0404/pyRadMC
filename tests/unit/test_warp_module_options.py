"""Compile options of the Warp kernel modules are deliberate; pin them.

``enable_backward=False``: warp compiles an adjoint version of every kernel by
default; this engine never differentiates, and skipping the adjoints cuts the
cold kernel compile ~3x (measured 6.4 s -> 2.2 s). No runtime semantics.

The *pre-solve* module must stay precise-math: its attenuation mode is
test-pinned float32-exact against the host, which fast div/log would break.
"""

from __future__ import annotations

import pytest

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

import pyRadMC.backends.warp.kernels  # noqa: E402
import pyRadMC.backends.warp.presolve  # noqa: E402, F401

pytestmark = pytest.mark.warp


def _options(module_name: str) -> dict:
    return wp.get_module(module_name).options


def test_transport_kernels_skip_backward_compilation() -> None:
    assert _options("pyRadMC.backends.warp.kernels")["enable_backward"] is False


def test_presolve_skips_backward_compilation() -> None:
    assert _options("pyRadMC.backends.warp.presolve")["enable_backward"] is False


def test_transport_kernels_use_fast_math() -> None:
    # Approximate transcendentals, measured 14-23% on the latency-bound transport
    # (2026-07-18); within-device bit reproducibility is unaffected (one binary),
    # and cross-target equivalence is statistical by contract (AGENTS.md 2.3).
    assert _options("pyRadMC.backends.warp.kernels")["fast_math"] is True


def test_presolve_keeps_precise_math() -> None:
    # The device pre-solve's attenuation mode is pinned float32-exact against the
    # host (test_device_presolve); approximate division would break that parity.
    assert _options("pyRadMC.backends.warp.presolve")["fast_math"] is False
