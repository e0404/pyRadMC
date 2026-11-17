"""The Bhabha and annihilation samplers, compiled under Warp, against their oracles.

The companion of ``test_positron_sampling.py`` for the ``@wp.func`` build of
:mod:`pyradmc.physics.bhabha` and :mod:`pyradmc.physics.annihilation`: the same
differential cross-sections, applied to samples drawn in float32 on each Warp
device, plus the conservation identities evaluated on the compiled direction
cosines. A codegen defect, or single precision losing the physics, shows as a
shifted distribution or an open momentum balance.

Three positron energies: 1 MeV, 20 MeV (the top of the clinical range) and
200 MeV. The cosines are float32: at 200 MeV the hard annihilation photon leaves
within a few milliradians of the positron and its cosine has a few dozen
representable values there, which bounds how well the transverse momentum can
close (the tolerance below); the energies, which are what the dose sees, are
unaffected.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data.positron import annihilation_dcs_per_electron, bhabha_dcs_per_electron
from tests.conftest import SEED
from tests.physics.test_positron_sampling import _spectrum_p_value

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

N = 40_000
CUT = 0.2
TWO_M = 2.0 * ELECTRON_MASS_MEV
ENERGIES = (1.0, 20.0, 200.0)
# Per energy: delta energy, cos delta, cos positron, photon energy, cos first, cos second.
COLUMNS = 6


@pytest.fixture(scope="module", params=DEVICES)
def device_samples(request: pytest.FixtureRequest) -> np.ndarray:
    device = request.param
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    from pyradmc.backends.warp.physics import warp_physics
    from pyradmc.rng.warp_shim import WarpRNGState, init_slot

    p = warp_physics()
    delta_energy = p.sample_bhabha_delta_energy
    bhabha_cosines = p.bhabha_direction_cosines
    photon_energy = p.sample_annihilation_photon_energy
    annihilation_cosines = p.annihilation_direction_cosines

    @wp.kernel
    def sample_all(
        seed: int,
        energies: wp.array(dtype=float),
        slots: wp.array(dtype=wp.uint32),
        out: wp.array2d(dtype=float),
    ):
        tid = wp.tid()
        slots[tid] = init_slot(seed, tid)
        state = WarpRNGState()
        state.slots = slots
        state.idx = tid

        for j in range(energies.shape[0]):
            energy = energies[j]
            base = j * COLUMNS
            w = delta_energy(energy, CUT, state)
            cos_delta, cos_positron = bhabha_cosines(energy, w)
            out[tid, base] = w
            out[tid, base + 1] = cos_delta
            out[tid, base + 2] = cos_positron
            k = photon_energy(energy, state)
            cos_first, cos_second = annihilation_cosines(energy, k)
            out[tid, base + 3] = k
            out[tid, base + 4] = cos_first
            out[tid, base + 5] = cos_second

    energies = wp.array(np.array(ENERGIES, dtype=np.float32), dtype=float, device=device)
    slots = wp.zeros(N, dtype=wp.uint32, device=device)
    out = wp.zeros((N, COLUMNS * len(ENERGIES)), dtype=float, device=device)
    wp.launch(sample_all, dim=N, inputs=[SEED, energies, slots], outputs=[out], device=device)
    wp.synchronize_device(device)
    return np.asarray(out.numpy(), dtype=np.float64)


def _momentum(kinetic: np.ndarray | float) -> np.ndarray | float:
    return np.sqrt(kinetic * (kinetic + TWO_M))


@pytest.mark.parametrize(("column", "energy"), list(enumerate(ENERGIES)))
class TestCompiledBhabha:
    def test_delta_spectrum_follows_the_bhabha_dcs(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        w = device_samples[:, column * COLUMNS]
        assert np.all(w >= CUT * (1.0 - 1.0e-6))
        assert np.all(w <= energy * (1.0 + 1.0e-6))
        edges = np.geomspace(CUT, energy, 13)  # as the host test: the hardest bin keeps > 20
        _, p_value = _spectrum_p_value(w, lambda x: bhabha_dcs_per_electron(energy, x), edges)
        assert p_value > 0.01, f"compiled Bhabha spectrum off at T={energy} MeV: p={p_value:.2e}"

    def test_oracle_rejects_the_bare_envelope(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        """Power: the same samples against pure 1/W^2 (no Bhabha bracket) fail."""
        w = device_samples[:, column * COLUMNS]
        edges = np.geomspace(CUT, energy, 13)  # as the host test: the hardest bin keeps > 20
        _, p_value = _spectrum_p_value(w, lambda x: 1.0 / (x * x), edges)
        assert p_value < 1.0e-6

    def test_direction_cosines_conserve_momentum(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        w = device_samples[:, column * COLUMNS]
        cos_delta = device_samples[:, column * COLUMNS + 1]
        cos_positron = device_samples[:, column * COLUMNS + 2]
        p_in = _momentum(energy)
        longitudinal = _momentum(w) * cos_delta + _momentum(np.maximum(energy - w, 0.0)) * (
            cos_positron
        )
        np.testing.assert_allclose(longitudinal, p_in, rtol=2.0e-5)


@pytest.mark.parametrize(("column", "energy"), list(enumerate(ENERGIES)))
class TestCompiledAnnihilation:
    @staticmethod
    def _limits(energy: float) -> tuple[float, float]:
        total = energy + TWO_M
        half_width = 0.5 * math.sqrt(energy / total)
        return total * (0.5 - half_width), total * (0.5 + half_width)

    def test_photon_spectrum_follows_the_heitler_dcs(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        k = device_samples[:, column * COLUMNS + 3]
        k_min, k_max = self._limits(energy)
        assert np.all(k >= k_min * (1.0 - 1.0e-5))
        assert np.all(k <= k_max * (1.0 + 1.0e-5))
        edges = np.geomspace(k_min, k_max, 21)
        _, p_value = _spectrum_p_value(k, lambda x: annihilation_dcs_per_electron(energy, x), edges)
        assert p_value > 0.01, f"compiled Heitler spectrum off at T={energy} MeV: p={p_value:.2e}"

    def test_oracle_rejects_the_bare_envelope(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        k = device_samples[:, column * COLUMNS + 3]
        k_min, k_max = self._limits(energy)
        edges = np.geomspace(k_min, k_max, 21)
        _, p_value = _spectrum_p_value(k, lambda x: 1.0 / x, edges)
        assert p_value < 1.0e-6

    def test_direction_cosines_conserve_momentum(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        """Longitudinal momenta add to the positron's; transverse ones cancel.

        The transverse residual is bounded by the float32 cosine of the forward
        photon: an error of one float32 step (6e-8) in a cosine near one is an
        angle error of up to sqrt(1.2e-7) = 3.5e-4 rad.
        """
        k_first = device_samples[:, column * COLUMNS + 3]
        cos_first = np.clip(device_samples[:, column * COLUMNS + 4], -1.0, 1.0)
        cos_second = np.clip(device_samples[:, column * COLUMNS + 5], -1.0, 1.0)
        k_second = energy + TWO_M - k_first
        p_in = _momentum(energy)
        np.testing.assert_allclose(k_first * cos_first + k_second * cos_second, p_in, rtol=2.0e-5)
        transverse = k_first * np.sqrt(1.0 - cos_first**2) - k_second * np.sqrt(1.0 - cos_second**2)
        assert np.all(np.abs(transverse) < 2.0e-3 * (energy + TWO_M))
