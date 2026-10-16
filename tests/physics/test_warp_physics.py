"""The single-source physics functions, compiled under Warp, against analytic oracles.

Same source, different compiler: these tests re-run the host-tier oracles (analytic
pdfs, kinematic identities, exact inversions) on the ``@wp.func``-compiled physics on
each Warp device. They are *not* comparisons against the host samplers — cross-target
streams share nothing — but against the same analytic ground truth, so a codegen or
shim defect shows up as a distribution shift or a broken identity.

Deterministic functions are additionally checked against their float64 host values at
float32-appropriate tolerance; that is a same-formula precision check, not the
cross-target dose equality AGENTS.md 2.3 forbids.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import integrate, stats

from pyradmc import ELECTRON_MASS_MEV
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

N = 30_000
COMPTON_ENERGY = 1.25
COMPTON_ENERGY_HIGH = 6.0
MOLLER_ENERGY = 2.0
MOLLER_CUT = 0.2
HINGE_MSQ = 0.01
BREMS_ENERGY = 2.0
BREMS_PCUT = 0.05
PATH_MU = 0.05
CHANNEL_MU_COMPTON = 0.55
CHANNEL_MU_PHOTO = 0.25
CHANNEL_MU_PAIR = 0.12
CHANNEL_MU_RAYLEIGH = 0.08


@pytest.fixture(scope="module", params=DEVICES)
def device_samples(request):
    """One launch per device: every physics function sampled N times."""
    device = request.param
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    from pyradmc.backends.warp.physics import warp_physics
    from pyradmc.rng.warp_shim import WarpRNGState, init_slot

    p = warp_physics()

    sample_path = p.sample_path_length
    select_process = p.select_photon_process
    compton_ratio = p.sample_compton_energy_ratio
    compton_cos = p.compton_cos_theta
    compton_ecos = p.compton_electron_cos_theta
    isotropic = p.sample_isotropic_direction
    rotate = p.rotate_direction
    moller_delta = p.sample_moller_delta_energy
    moller_cos = p.moller_direction_cosines
    hinge = p.sample_hinge_cos_theta
    brems_params = p.bremsstrahlung_step_parameters
    brems_energy = p.sample_bremsstrahlung_energy
    rayleigh_cos = p.sample_rayleigh_cos_theta

    @wp.kernel
    def sample_all(seed: int, slots: wp.array(dtype=wp.uint32), out: wp.array2d(dtype=float)):
        tid = wp.tid()
        slots[tid] = init_slot(seed, tid)
        state = WarpRNGState()
        state.slots = slots
        state.idx = tid

        out[tid, 0] = sample_path(PATH_MU, state)
        out[tid, 1] = float(
            select_process(
                CHANNEL_MU_COMPTON, CHANNEL_MU_PHOTO, CHANNEL_MU_PAIR, CHANNEL_MU_RAYLEIGH, state
            )
        )

        r = compton_ratio(COMPTON_ENERGY, state)
        out[tid, 2] = r
        out[tid, 3] = compton_cos(COMPTON_ENERGY, r)
        out[tid, 4] = compton_ecos(COMPTON_ENERGY, r)
        out[tid, 5] = compton_ratio(COMPTON_ENERGY_HIGH, state)

        ax, ay, az = isotropic(state)
        out[tid, 6] = ax * ax + ay * ay + az * az
        out[tid, 7] = az
        vx, vy, vz = rotate(ax, ay, az, out[tid, 3], 2.0 * 3.14159265 * 0.25)
        out[tid, 8] = vx * vx + vy * vy + vz * vz

        w = moller_delta(MOLLER_ENERGY, MOLLER_CUT, state)
        out[tid, 9] = w
        cd, cp = moller_cos(MOLLER_ENERGY, w)
        out[tid, 10] = cd
        out[tid, 11] = cp

        out[tid, 12] = hinge(HINGE_MSQ, state)
        prob, local = brems_params(BREMS_ENERGY, 0.02, 1.0, 0.05, BREMS_PCUT)
        out[tid, 13] = prob
        out[tid, 14] = local
        out[tid, 15] = brems_energy(BREMS_ENERGY, BREMS_PCUT, state)
        out[tid, 16] = rayleigh_cos(state)

    slots = wp.zeros(N, dtype=wp.uint32, device=device)
    out = wp.zeros((N, 17), dtype=float, device=device)
    wp.launch(sample_all, dim=N, inputs=[SEED, slots], outputs=[out], device=device)
    wp.synchronize_device(device)
    return out.numpy()


def _kn_pdf_unnormalized(r: np.ndarray, alpha: float) -> np.ndarray:
    """Klein-Nishina dsigma/dr up to a constant; see test_compton_sampling.py."""
    cos_theta = 1.0 + 1.0 / alpha - 1.0 / (alpha * r)
    return r + 1.0 / r - (1.0 - cos_theta**2)


def _assert_histogram_matches_pdf(
    samples: np.ndarray, pdf, lo: float, hi: float, n_bins: int = 32
) -> None:
    """Chi-squared of a sample histogram against an analytic (unnormalized) pdf."""
    edges = np.linspace(lo, hi, n_bins + 1)
    observed, _ = np.histogram(samples, bins=edges)
    norm, _ = integrate.quad(pdf, lo, hi)
    expected = (
        np.array([integrate.quad(pdf, a, b)[0] for a, b in itertools.pairwise(edges)])
        / norm
        * samples.size
    )
    assert np.all(expected > 20.0), "bin layout leaves low-count bins; rebin"
    chi2, p_value = stats.chisquare(observed, expected * observed.sum() / expected.sum())
    assert p_value > 0.01, f"chi2={chi2:.1f} on {n_bins - 1} dof, p={p_value:.2e}"


def test_host_physics_modules_stay_untouched() -> None:
    """Loading the warp physics must not replace or mutate the host modules.

    The loader re-imports the physics sources under shimmed ``pyradmc.rng`` and
    ``math`` entries. If the host modules or their parent-package attributes end up
    pointing at the warp copies, the reference backend silently starts calling
    ``@wp.func`` objects — the exact single-source failure mode this design avoids.
    """
    import pyradmc.physics
    from pyradmc.backends.warp.physics import warp_physics
    from pyradmc.physics import compton as host_compton
    from pyradmc.rng.host import HostRNG

    p = warp_physics()

    import pyradmc.physics.compton

    assert pyradmc.physics.compton is host_compton
    assert getattr(pyradmc.physics, "compton") is host_compton  # noqa: B009
    assert p.sample_compton_energy_ratio is not host_compton.sample_compton_energy_ratio

    # The host function still runs on a host RNG state (a wp.func would not).
    state = HostRNG().init_state(SEED, 0)
    r = host_compton.sample_compton_energy_ratio(1.25, state)
    assert 0.0 < r <= 1.0


def test_warp_physics_is_a_singleton() -> None:
    """Kernels must all reference one compiled copy of each function."""
    from pyradmc.backends.warp.physics import warp_physics

    assert warp_physics() is warp_physics()


class TestSampledDistributions:
    """Warp-compiled samplers against the same analytic pdfs the host tier uses."""

    def test_path_length_is_exponential(self, device_samples: np.ndarray) -> None:
        s = device_samples[:, 0]
        assert np.all(s > 0.0)
        ks = stats.kstest(s, "expon", args=(0.0, 1.0 / PATH_MU))
        assert ks.pvalue > 0.01, f"KS p={ks.pvalue:.2e}"

    def test_channel_selection_frequencies(self, device_samples: np.ndarray) -> None:
        c = device_samples[:, 1].astype(int)
        observed = np.bincount(c, minlength=4)
        mu = np.array([CHANNEL_MU_COMPTON, CHANNEL_MU_PHOTO, CHANNEL_MU_PAIR, CHANNEL_MU_RAYLEIGH])
        expected = mu / mu.sum() * c.size
        _chi2, p_value = stats.chisquare(observed, expected)
        assert p_value > 0.01, f"channel frequencies {observed / c.size} vs {mu / mu.sum()}"

    @pytest.mark.parametrize("column, energy", [(2, COMPTON_ENERGY), (5, COMPTON_ENERGY_HIGH)])
    def test_compton_spectrum_matches_klein_nishina(
        self, device_samples: np.ndarray, column: int, energy: float
    ) -> None:
        alpha = energy / ELECTRON_MASS_MEV
        r = device_samples[:, column]
        r_min = 1.0 / (1.0 + 2.0 * alpha)
        assert np.all(r >= r_min - 1e-6)
        assert np.all(r <= 1.0 + 1e-6)
        _assert_histogram_matches_pdf(
            r, lambda x: float(_kn_pdf_unnormalized(np.asarray(x), alpha)), r_min, 1.0
        )

    def test_moller_spectrum_matches_dcs(self, device_samples: np.ndarray) -> None:
        from pyradmc.data.analytic import moller_dcs_per_electron

        w = device_samples[:, 9]
        assert np.all(w >= MOLLER_CUT - 1e-6)
        assert np.all(w <= MOLLER_ENERGY / 2.0 + 1e-6)
        _assert_histogram_matches_pdf(
            w,
            lambda x: moller_dcs_per_electron(MOLLER_ENERGY, float(x)),
            MOLLER_CUT,
            MOLLER_ENERGY / 2.0,
        )

    def test_hinge_angle_squared_is_exponential(self, device_samples: np.ndarray) -> None:
        cos_h = device_samples[:, 12]
        theta_sq = np.arccos(np.clip(cos_h, -1.0, 1.0)) ** 2
        ks = stats.kstest(theta_sq, "expon", args=(0.0, HINGE_MSQ))
        assert ks.pvalue > 0.01, f"KS p={ks.pvalue:.2e}"

    def test_brems_energy_is_log_uniform(self, device_samples: np.ndarray) -> None:
        k = device_samples[:, 15]
        assert np.all(k >= BREMS_PCUT * (1.0 - 1e-6))
        assert np.all(k <= BREMS_ENERGY * (1.0 + 1e-6))
        u = np.log(k / BREMS_PCUT) / np.log(BREMS_ENERGY / BREMS_PCUT)
        ks = stats.kstest(u, "uniform")
        assert ks.pvalue > 0.01, f"KS p={ks.pvalue:.2e}"

    def test_rayleigh_cosine_matches_thomson(self, device_samples: np.ndarray) -> None:
        mu = device_samples[:, 16]
        assert np.all(np.abs(mu) <= 1.0)
        _assert_histogram_matches_pdf(mu, lambda x: 1.0 + x * x, -1.0, 1.0)

    def test_isotropic_direction_is_unit_and_uniform(self, device_samples: np.ndarray) -> None:
        norm_sq = device_samples[:, 6]
        uz = device_samples[:, 7]
        assert np.all(np.abs(norm_sq - 1.0) < 1e-5)
        ks = stats.kstest(uz, "uniform", args=(-1.0, 2.0))
        assert ks.pvalue > 0.01, f"KS p={ks.pvalue:.2e}"


class TestDeterministicParity:
    """float32 warp values of the closed-form functions against float64 host values."""

    RTOL = 3e-5  # float32 arithmetic on well-conditioned closed forms

    def test_compton_angle_identity(self, device_samples: np.ndarray) -> None:
        """cos theta from the warp function round-trips the Compton relation."""
        alpha = COMPTON_ENERGY / ELECTRON_MASS_MEV
        r = device_samples[:, 2].astype(np.float64)
        cos_theta = device_samples[:, 3].astype(np.float64)
        round_trip = 1.0 / (1.0 + alpha * (1.0 - cos_theta))
        np.testing.assert_allclose(round_trip, r, rtol=1e-4)

    def test_compton_electron_cosine_matches_host(self, device_samples: np.ndarray) -> None:
        """Parity where the function is actually used: recoil above a spawn threshold.

        As r -> 1 the recoil momentum vanishes and cos_theta_e becomes ill-conditioned
        in the float32 value of r itself (relative error ~eps32 / (1 - r)); the
        transport loop never evaluates it there, because sub-ECUT recoils deposit
        locally instead of spawning. 50 keV is far below any ECUT this engine allows.
        """
        from pyradmc.physics.compton import compton_electron_cos_theta

        r = device_samples[:, 2]
        used = r < 1.0 - 0.050 / COMPTON_ENERGY
        assert np.count_nonzero(used) > 20_000  # the cut must not hollow out the test
        got = device_samples[used, 4]
        want = np.array([compton_electron_cos_theta(COMPTON_ENERGY, float(ri)) for ri in r[used]])
        np.testing.assert_allclose(got, want, rtol=self.RTOL, atol=1e-6)

    def test_rotation_preserves_unit_norm(self, device_samples: np.ndarray) -> None:
        norm_sq = device_samples[:, 8]
        assert np.all(np.abs(norm_sq - 1.0) < 1e-5)

    def test_moller_cosines_match_host(self, device_samples: np.ndarray) -> None:
        from pyradmc.physics.moller import moller_direction_cosines

        w = device_samples[:, 9]
        want = np.array([moller_direction_cosines(MOLLER_ENERGY, float(wi)) for wi in w])
        np.testing.assert_allclose(device_samples[:, 10], want[:, 0], rtol=self.RTOL)
        np.testing.assert_allclose(device_samples[:, 11], want[:, 1], rtol=self.RTOL)

    def test_moller_transverse_momentum_cancels(self, device_samples: np.ndarray) -> None:
        """The test-pinned kinematic identity, evaluated on warp outputs."""
        two_m = 2.0 * ELECTRON_MASS_MEV
        w = device_samples[:, 9].astype(np.float64)
        cd = device_samples[:, 10].astype(np.float64)
        cp = device_samples[:, 11].astype(np.float64)
        p_in = math.sqrt(MOLLER_ENERGY * (MOLLER_ENERGY + two_m))
        p_delta = np.sqrt(w * (w + two_m))
        p_prim = np.sqrt((MOLLER_ENERGY - w) * (MOLLER_ENERGY - w + two_m))
        longitudinal = p_delta * cd + p_prim * cp
        np.testing.assert_allclose(longitudinal, p_in, rtol=1e-4)
        transverse = p_delta * np.sqrt(1.0 - cd**2) - p_prim * np.sqrt(1.0 - cp**2)
        np.testing.assert_allclose(transverse, 0.0, atol=2e-3 * p_in)

    def test_brems_step_parameters_match_host(self, device_samples: np.ndarray) -> None:
        from pyradmc.physics.brems import bremsstrahlung_step_parameters

        want_prob, want_local = bremsstrahlung_step_parameters(
            BREMS_ENERGY, 0.02, 1.0, 0.05, BREMS_PCUT
        )
        np.testing.assert_allclose(device_samples[:, 13], want_prob, rtol=self.RTOL)
        np.testing.assert_allclose(device_samples[:, 14], want_local, rtol=self.RTOL)


# --- Lorentz-force primitives --------------------------------------------------
# Deterministic closed forms, so the twin is host float64 on the *same float32
# inputs* the device saw, at float32 tolerance. Inputs are generated on the host
# from the fixed seed: no RNG is involved in the functions themselves.

N_LORENTZ = 4_096
LORENTZ_B_TESLA = 1.5


@pytest.fixture(scope="module", params=DEVICES)
def lorentz_device_values(request):
    """The Lorentz primitives evaluated on each device over seeded float32 inputs."""
    device = request.param
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    from pyradmc.backends.warp.physics import warp_physics

    p = warp_physics()
    momentum = p.momentum_mev_c
    gyroradius = p.gyroradius_cm
    rotate = p.rotate_about_axis
    angle = p.lorentz_rotation_angle

    rng = np.random.default_rng(SEED)
    energy = np.exp(rng.uniform(np.log(0.05), np.log(20.0), N_LORENTZ)).astype(np.float32)
    u = rng.normal(size=(N_LORENTZ, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    a = rng.normal(size=(N_LORENTZ, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    theta = rng.uniform(-np.pi, np.pi, N_LORENTZ)
    theta[::8] = 0.0  # every eighth lane exercises the exact no-op guard
    u = u.astype(np.float32)
    a = a.astype(np.float32)
    theta = theta.astype(np.float32)
    chord = rng.uniform(1e-3, 0.5, N_LORENTZ).astype(np.float32)

    @wp.kernel
    def probe(
        e_kin: wp.array(dtype=float),
        ux: wp.array(dtype=float),
        uy: wp.array(dtype=float),
        uz: wp.array(dtype=float),
        ax: wp.array(dtype=float),
        ay: wp.array(dtype=float),
        az: wp.array(dtype=float),
        th: wp.array(dtype=float),
        s: wp.array(dtype=float),
        out: wp.array2d(dtype=float),
    ):
        tid = wp.tid()
        out[tid, 0] = momentum(e_kin[tid])
        r_g = gyroradius(e_kin[tid], LORENTZ_B_TESLA)
        out[tid, 1] = r_g
        vx, vy, vz = rotate(ux[tid], uy[tid], uz[tid], ax[tid], ay[tid], az[tid], th[tid])
        out[tid, 2] = vx
        out[tid, 3] = vy
        out[tid, 4] = vz
        # Rotate back: the inverse must undo the rotation in float32 too.
        bx, by, bz = rotate(vx, vy, vz, ax[tid], ay[tid], az[tid], -th[tid])
        out[tid, 5] = bx
        out[tid, 6] = by
        out[tid, 7] = bz
        out[tid, 8] = angle(s[tid], r_g, False)
        out[tid, 9] = angle(s[tid], r_g, tid >= 0)  # a runtime bool, as in the kernel

    columns = (energy, *u.T, *a.T, theta, chord)
    arrays = [wp.array(np.ascontiguousarray(c), dtype=float, device=device) for c in columns]
    out = wp.zeros((N_LORENTZ, 10), dtype=float, device=device)
    wp.launch(probe, dim=N_LORENTZ, inputs=arrays, outputs=[out], device=device)
    wp.synchronize_device(device)
    return {
        "energy": energy.astype(np.float64),
        "u": u.astype(np.float64),
        "a": a.astype(np.float64),
        "theta": theta.astype(np.float64),
        "chord": chord.astype(np.float64),
        "out": out.numpy().astype(np.float64),
    }


class TestLorentzParity:
    """float32 device values of the Lorentz primitives against float64 host values."""

    RTOL = 3e-5  # float32 arithmetic on well-conditioned closed forms
    ATOL_DIRECTION = 4e-6  # a few float32 ulps of a unit-vector component

    def test_momentum_matches_host(self, lorentz_device_values: dict) -> None:
        from pyradmc.physics.lorentz import momentum_mev_c

        e = lorentz_device_values["energy"]
        want = np.array([momentum_mev_c(float(ei)) for ei in e])
        np.testing.assert_allclose(lorentz_device_values["out"][:, 0], want, rtol=self.RTOL)

    def test_gyroradius_matches_host(self, lorentz_device_values: dict) -> None:
        from pyradmc.physics.lorentz import gyroradius_cm

        e = lorentz_device_values["energy"]
        want = np.array([gyroradius_cm(float(ei), LORENTZ_B_TESLA) for ei in e])
        np.testing.assert_allclose(lorentz_device_values["out"][:, 1], want, rtol=self.RTOL)

    def test_rotation_matches_host(self, lorentz_device_values: dict) -> None:
        from pyradmc.physics.lorentz import rotate_about_axis

        u = lorentz_device_values["u"]
        a = lorentz_device_values["a"]
        theta = lorentz_device_values["theta"]
        want = np.array(
            [rotate_about_axis(*ui, *ai, float(ti)) for ui, ai, ti in zip(u, a, theta, strict=True)]
        )
        got = lorentz_device_values["out"][:, 2:5]
        np.testing.assert_allclose(got, want, rtol=self.RTOL, atol=self.ATOL_DIRECTION)

    def test_rotation_preserves_unit_norm(self, lorentz_device_values: dict) -> None:
        v = lorentz_device_values["out"][:, 2:5]
        assert np.all(np.abs(np.einsum("ij,ij->i", v, v) - 1.0) < 1e-5)

    def test_rotation_preserves_axial_component(self, lorentz_device_values: dict) -> None:
        u = lorentz_device_values["u"]
        a = lorentz_device_values["a"]
        v = lorentz_device_values["out"][:, 2:5]
        np.testing.assert_allclose(
            np.einsum("ij,ij->i", a, v), np.einsum("ij,ij->i", a, u), atol=1e-5
        )

    def test_zero_angle_is_bit_exact_on_device(self, lorentz_device_values: dict) -> None:
        """The B = 0 guard must return the input floats untouched in float32 too."""
        zero = lorentz_device_values["theta"] == 0.0
        assert np.count_nonzero(zero) == N_LORENTZ // 8
        u = lorentz_device_values["u"][zero]
        v = lorentz_device_values["out"][zero, 2:5]
        assert np.array_equal(u, v)

    def test_inverse_rotation_undoes_in_float32(self, lorentz_device_values: dict) -> None:
        u = lorentz_device_values["u"]
        back = lorentz_device_values["out"][:, 5:8]
        np.testing.assert_allclose(back, u, atol=1e-5)

    def test_rotation_angle_matches_host_and_sign(self, lorentz_device_values: dict) -> None:
        from pyradmc.physics.lorentz import lorentz_rotation_angle

        s = lorentz_device_values["chord"]
        r_g = lorentz_device_values["out"][:, 1]
        electron = lorentz_device_values["out"][:, 8]
        positron = lorentz_device_values["out"][:, 9]
        want = np.array(
            [
                lorentz_rotation_angle(float(si), float(ri), False)
                for si, ri in zip(s, r_g, strict=True)
            ]
        )
        np.testing.assert_allclose(electron, want, rtol=self.RTOL)
        np.testing.assert_allclose(positron, -want, rtol=self.RTOL)
        assert np.all(electron > 0.0)
        assert np.all(positron < 0.0)
