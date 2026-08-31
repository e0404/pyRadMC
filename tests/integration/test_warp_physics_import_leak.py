"""``warp_physics()`` must not leak kernel shims into host modules.

The loader re-imports the physics sources with ``pyradmc.rng`` (and ``math``)
patched in ``sys.modules``. A module imported *transitively* inside that window
which is not itself in ``_KERNEL_MODULES`` — ``pyradmc.geometry.spectrum`` is the
one that bites — would bind the kernel-side ``uniform`` at module level and keep
it after the restore, breaking the *reference* backend's spectral sources for the
rest of the process (AGENTS.md: the open bug recorded 2026-07-17). The loader
therefore evicts every pyradmc module first imported inside the window, forcing a
clean host re-import later.

Each check runs in a subprocess: the leak only manifests when the victim module
is first imported inside the patch window, which an in-process test cannot
guarantee once the suite has imported it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

_LEAK_CHECK = """
import pyradmc.backends.warp.kernels  # runs warp_physics() at import
from pyradmc import rng
from pyradmc.geometry import spectrum

assert spectrum.uniform is rng.uniform, (
    f"leak: spectrum.uniform is {spectrum.uniform!r}, host uniform is {rng.uniform!r}"
)
"""

_REF_AFTER_WARP_CHECK = """
import numpy as np

import pyradmc.backends.warp.kernels  # constructing any WarpEngine triggers this
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG

spectrum = Spectrum(edges=np.array([1.0, 2.0, 3.0]), weights=np.array([1.0, 1.0]))
spectrum.sample_energy(HostRNG().init_state(1, 0))  # raised through the leaked shim
"""


def _run(snippet: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr


def test_spectrum_keeps_the_host_uniform_after_warp_physics() -> None:
    _run(_LEAK_CHECK)


def test_reference_spectral_sampling_survives_warp_import() -> None:
    _run(_REF_AFTER_WARP_CHECK)
