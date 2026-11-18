"""The screened Bethe-Heitler samplers, compiled under Warp, against their oracles.

The companion of ``test_bethe_heitler_sampling.py`` for the ``@wp.func`` build of
:mod:`pyradmc.physics.brems_bh`: the same analytic oracles, applied to samples
drawn in float32 on each Warp device. Not a comparison against the host samplers
(cross-target streams share nothing) but against the same ground truth, so a
codegen defect — or single precision losing the physics — shows as a shifted
distribution.

Two energies: 20 MeV, the top of the clinical range, and 300 MeV, the top of the
model's stated range, where the emission angle is of order 1.7 mrad and its
cosine differs from one by 1.4e-6 — about twenty float32 steps. That is why the
kernel carries ``1 - cos(theta)`` into
:func:`~pyradmc.physics.direction.rotate_direction_one_minus_cos`, and the last
tests here pin that the angle survives the rotation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data.materials import WATER
from tests.conftest import SEED
from tests.physics.test_bethe_heitler_sampling import K_CUT, _shape_args, _spectrum_p_value

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

N = 30_000
ENERGIES = (20.0, 300.0)
AXIS = (0.6, -0.48, 0.64)  # deliberately not axis-aligned
# Columns per energy: photon energy, 1 - cos, chord^2 by 1 - cos form, chord^2 by cosine form.
COLUMNS = 4


@pytest.fixture(scope="module", params=DEVICES)
def device_samples(request: pytest.FixtureRequest) -> np.ndarray:
    """One launch per device: ``N`` draws of every quantity at each energy."""
    device = request.param
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    from pyradmc.backends.warp.physics import warp_physics
    from pyradmc.rng.warp_shim import WarpRNGState, init_slot, uniform

    p = warp_physics()
    photon_energy = p.sample_bethe_heitler_photon_energy
    one_minus_cos = p.sample_bremsstrahlung_one_minus_cos_theta
    rotate = p.rotate_direction
    rotate_small = p.rotate_direction_one_minus_cos
    z_ratio, c_nuc, c_el, g_nuc, g_el = _shape_args(WATER)
    ax, ay, az = AXIS

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
            out[tid, base] = photon_energy(energy, K_CUT, z_ratio, c_nuc, c_el, g_nuc, g_el, state)
            x = one_minus_cos(energy, state)
            out[tid, base + 1] = x
            phi = 2.0 * 3.14159265 * uniform(state)
            vx, vy, vz = rotate_small(ax, ay, az, x, phi)
            out[tid, base + 2] = (
                (vx - ax) * (vx - ax) + (vy - ay) * (vy - ay) + (vz - az) * (vz - az)
            )
            cx, cy, cz = rotate(ax, ay, az, 1.0 - x, phi)
            out[tid, base + 3] = (
                (cx - ax) * (cx - ax) + (cy - ay) * (cy - ay) + (cz - az) * (cz - az)
            )

    energies = wp.array(np.array(ENERGIES, dtype=np.float32), dtype=float, device=device)
    slots = wp.zeros(N, dtype=wp.uint32, device=device)
    out = wp.zeros((N, COLUMNS * len(ENERGIES)), dtype=float, device=device)
    wp.launch(sample_all, dim=N, inputs=[SEED, energies, slots], outputs=[out], device=device)
    wp.synchronize_device(device)
    return np.asarray(out.numpy(), dtype=np.float64)


def _beta(energy: float) -> float:
    total = energy + ELECTRON_MASS_MEV
    return math.sqrt(energy * (energy + 2.0 * ELECTRON_MASS_MEV)) / total


def _angle_p_value(x: np.ndarray, energy: float) -> float:
    """Chi-squared of ``1 - cos`` in 40 equal-probability bins of the angular law.

    From the closed-form inverse ``x(u) = 2 (1 - beta)(1 - u) / (1 - beta + 2 beta u)``;
    binned in ``x`` itself so the oracle does not round the angle the way a cosine
    would.
    """
    beta = _beta(energy)
    one_minus_beta = (ELECTRON_MASS_MEV / (energy + ELECTRON_MASS_MEV)) ** 2 / (1.0 + beta)
    u = np.linspace(1.0, 0.0, 41)
    edges = 2.0 * one_minus_beta * (1.0 - u) / (one_minus_beta + 2.0 * beta * u)
    edges[0], edges[-1] = 0.0, 2.0 + 1.0e-6
    observed, _ = np.histogram(x, bins=edges)
    assert observed.sum() == x.size, "a sample fell outside [0, 2]"
    return float(stats.chisquare(observed).pvalue)


@pytest.mark.parametrize(("column", "energy"), list(enumerate(ENERGIES)))
class TestCompiledSamplers:
    def test_photon_energy_follows_the_screened_shape(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        k = device_samples[:, column * COLUMNS]
        # float32: the energy the kernel saw, and a sample may round onto an edge.
        energy32 = float(np.float32(energy))
        assert np.all(k >= K_CUT * (1.0 - 1.0e-6))
        assert np.all(k <= energy32)
        k = np.clip(k, K_CUT, energy)
        _, p_value, _ = _spectrum_p_value(k, energy, _shape_args(WATER))
        assert p_value > 0.01, f"compiled spectrum off at T={energy} MeV: p={p_value:.2e}"

    def test_emission_angle_follows_the_boosted_law(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        x = device_samples[:, column * COLUMNS + 1]
        p_value = _angle_p_value(x, energy)
        assert p_value > 0.01, f"compiled emission angle off at T={energy} MeV: p={p_value:.2e}"

    def test_angle_oracle_rejects_a_wrong_velocity(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        """Power: the same samples against the law at 1.2 times the energy must fail."""
        x = device_samples[:, column * COLUMNS + 1]
        assert _angle_p_value(x, 1.2 * energy) < 1.0e-6

    def test_rotation_keeps_the_sampled_angle(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        """|v - u|^2 = 2 (1 - cos theta), in float32, photon by photon.

        The tolerance is float32 rounding of the unit-vector components
        (6e-8 each) against a chord of order 1/gamma, plus a relative term.
        """
        x = device_samples[:, column * COLUMNS + 1]
        chord2 = device_samples[:, column * COLUMNS + 2]
        chord = np.sqrt(chord2)
        expected = np.sqrt(2.0 * x)
        assert np.all(np.abs(chord - expected) < 5.0e-7 + 1.0e-3 * expected)
        assert _angle_p_value(0.5 * chord2, energy) > 0.01


def test_a_float32_cosine_loses_the_angle_at_300_mev(device_samples: np.ndarray) -> None:
    """Why the kernel does not pass a cosine: the same draws, rotated through one.

    ``1 - x`` in float32 is a multiple of 6e-8, so ``sqrt(1 - cos^2)`` quantizes a
    1.4e-6 deflection to a handful of values and the angular law fails its own
    oracle; the ``1 - cos`` form (test above) passes it.
    """
    column = ENERGIES.index(300.0)
    chord2 = device_samples[:, column * COLUMNS + 3]
    assert _angle_p_value(0.5 * chord2, 300.0) < 1.0e-6
