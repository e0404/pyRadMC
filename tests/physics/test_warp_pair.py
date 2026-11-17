"""The pair samplers, compiled under Warp, against their oracles.

The companion of ``test_pair_sampling.py`` for the ``@wp.func`` build of
:mod:`pyradmc.physics.pair`: the same analytic oracles, applied to samples drawn
in float32 on each Warp device. Not a comparison against the host samplers
(cross-target streams share nothing) but against the same ground truth, so a
codegen defect, or single precision losing the physics, shows as a shifted
distribution.

Three photon energies: 6 MeV, 20 MeV (the top of the clinical range) and 200 MeV,
on either side of the 50 MeV Coulomb-correction switch. The lepton angle is drawn
at the same three values as a kinetic energy; at 200 MeV its cosine differs from
one by 3e-6, about fifty float32 steps, which is why the kernel carries
``1 - cos(theta)`` into
:func:`~pyradmc.physics.direction.rotate_direction_one_minus_cos`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import integrate, stats

from pyradmc import ELECTRON_MASS_MEV
from pyradmc.data.materials import MATERIALS, WATER
from pyradmc.data.pair import pair_screening_parameters, pair_sharing_dcs
from tests.conftest import SEED
from tests.physics.test_warp_bethe_heitler import _angle_p_value

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

N = 40_000
N_BINS = 24
ENERGIES = (6.0, 20.0, 200.0)
AXIS = (0.6, -0.48, 0.64)  # deliberately not axis-aligned
COLUMNS = 3  # per energy: electron kinetic energy, 1 - cos, chord^2 after the rotation
WATER_PARAMS = pair_screening_parameters(MATERIALS[WATER].composition)


@pytest.fixture(scope="module", params=DEVICES)
def device_samples(request: pytest.FixtureRequest) -> np.ndarray:
    """One launch per device: ``N`` draws of every quantity at each energy."""
    device = request.param
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    from pyradmc.backends.warp.physics import warp_physics
    from pyradmc.rng.warp_shim import WarpRNGState, init_slot, uniform

    p = warp_physics()
    electron_energy = p.sample_pair_electron_energy
    one_minus_cos = p.sample_pair_one_minus_cos_theta
    rotate_small = p.rotate_direction_one_minus_cos
    screening_constant, coulomb_low, coulomb_high = WATER_PARAMS
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
            out[tid, base] = electron_energy(
                energy, screening_constant, coulomb_low, coulomb_high, state
            )
            x = one_minus_cos(energy, state)
            out[tid, base + 1] = x
            phi = 2.0 * 3.14159265 * uniform(state)
            vx, vy, vz = rotate_small(ax, ay, az, x, phi)
            out[tid, base + 2] = (
                (vx - ax) * (vx - ax) + (vy - ay) * (vy - ay) + (vz - az) * (vz - az)
            )

    energies = wp.array(np.array(ENERGIES, dtype=np.float32), dtype=float, device=device)
    slots = wp.zeros(N, dtype=wp.uint32, device=device)
    out = wp.zeros((N, COLUMNS * len(ENERGIES)), dtype=float, device=device)
    wp.launch(sample_all, dim=N, inputs=[SEED, energies, slots], outputs=[out], device=device)
    wp.synchronize_device(device)
    return np.asarray(out.numpy(), dtype=np.float64)


def _sharing_p_value(kinetic: np.ndarray, energy: float, uniform_law: bool = False) -> float:
    """Chi-squared of electron kinetic energies against the screened sharing DCS.

    The binning of ``test_pair_sampling.py``: equal bins in the total-energy
    fraction over the support of the DCS, strays counted into the edge bins.
    ``uniform_law`` tests against the uniform split instead (the power check).
    """
    eps = (kinetic + ELECTRON_MASS_MEV) / energy
    eps0 = ELECTRON_MASS_MEV / energy
    grid = np.linspace(eps0, 1.0 - eps0, 4001)
    support = grid[[pair_sharing_dcs(energy, float(e), WATER_PARAMS) > 0.0 for e in grid]]
    edges = np.linspace(support[0], support[-1], N_BINS + 1)
    observed, _ = np.histogram(np.clip(eps, edges[0], edges[-1]), bins=edges)
    if uniform_law:
        expected = np.diff(np.clip(edges, eps0, 1.0 - eps0))
    else:
        expected = np.array(
            [
                integrate.quad(lambda e: pair_sharing_dcs(energy, e, WATER_PARAMS), lo, hi)[0]
                for lo, hi in itertools.pairwise(edges)
            ]
        )
    expected = expected * (kinetic.size / expected.sum())
    assert np.all(expected > 20.0), "rebin: chi-squared needs counts"
    return float(stats.chisquare(observed, expected).pvalue)


@pytest.mark.parametrize(("column", "energy"), list(enumerate(ENERGIES)))
class TestCompiledSamplers:
    def test_sharing_follows_the_screened_dcs(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        t_minus = device_samples[:, column * COLUMNS]
        energy32 = float(np.float32(energy))
        kinetic = energy32 - 2.0 * ELECTRON_MASS_MEV
        assert np.all(t_minus >= 0.0)
        assert np.all(t_minus <= kinetic * (1.0 + 1.0e-6))
        p_value = _sharing_p_value(t_minus, energy)
        assert p_value > 0.01, f"compiled pair sharing off at k={energy} MeV: p={p_value:.2e}"

    def test_sharing_oracle_rejects_the_uniform_split(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        """Power: the compiled samples are not the uniform split."""
        t_minus = device_samples[:, column * COLUMNS]
        assert _sharing_p_value(t_minus, energy, uniform_law=True) < 1.0e-6

    def test_angle_follows_the_leading_order_law(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        x = device_samples[:, column * COLUMNS + 1]
        p_value = _angle_p_value(x, energy)
        assert p_value > 0.01, f"compiled pair angle off at T={energy} MeV: p={p_value:.2e}"

    def test_angle_oracle_rejects_a_wrong_velocity(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        x = device_samples[:, column * COLUMNS + 1]
        assert _angle_p_value(x, 1.2 * energy) < 1.0e-6

    def test_rotation_keeps_the_sampled_angle(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        """|v - u|^2 = 2 (1 - cos theta), in float32, lepton by lepton."""
        x = device_samples[:, column * COLUMNS + 1]
        chord = np.sqrt(device_samples[:, column * COLUMNS + 2])
        expected = np.sqrt(2.0 * x)
        assert np.all(np.abs(chord - expected) < 5.0e-7 + 1.0e-3 * expected)
        assert _angle_p_value(0.5 * chord**2, energy) > 0.01
