"""The device treatment-head pre-solve against the host oracle (cpu and cuda).

``presolve_head`` (host NumPy) is the ``ref`` oracle; ``presolve_head_device`` runs
the same head MC as a Warp kernel. Because both consume the *same* host-sampled
primaries, the pre-solve splits into a deterministic part and a stochastic part:

* **Attenuation mode** is pure Beer-Lambert geometry — no RNG — so device and host
  must agree to float32 on the kept count and the summed weight and weight-energy.
  This is the strong geometry/mu/attenuation check.
* **First-Compton mode** adds forced-interaction secondaries sampled from the
  device's own per-primary stream — a *different* generator from the host's, so
  AGENTS.md 2.3 allows statistical agreement only: the totals and per-kind counts
  match within a tight relative band, never bit-wise.

Water devices on the analytic backend keep the tier EPDL-free (as the host pre-solve
tests do). Runs on cpu and, when present, cuda.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyRadMC.geometry.head import AirColumn, presolve_head
from pyRadMC.geometry.source import GaussianSpotBeamSource
from pyRadMC.geometry.spectrum import Spectrum
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [
    pytest.mark.warp,
    pytest.mark.filterwarnings("ignore:.*latent variance:UserWarning"),
]

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

N = 120_000
SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
FRAME = BeamFrame(origin=FOCAL)


def _source() -> GaussianSpotBeamSource:
    return GaussianSpotBeamSource(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        center=(8.0, 8.0, -70.0),
        width_u=6.0,
        width_v=6.0,
        sigma_u=0.1,
        sigma_v=0.1,
    )


def _stack() -> BeamLimitingStack:
    """A half-blocking jaw plus a staircase MLC (exercises both device kinds)."""
    return BeamLimitingStack(
        frame=FRAME,
        devices=(
            JawPair(
                axis="u",
                z_top=40.0,
                z_bottom=45.0,
                edge_neg=-1000.0,
                edge_pos=0.5,
                material=WATER,
                density=1.0,
            ),
            MLC(
                z_top=46.0,
                z_bottom=50.0,
                leaf_edges_v=(-3.0, -1.0, 1.0, 3.0),
                tips_neg=(-1.5, 0.2, -1.5),
                tips_pos=(1.5, 0.2, 1.5),
                tip_radius=8.0,
                material=WATER,
                density=1.0,
            ),
        ),
    )


def _require(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")


def _totals(phase_space) -> dict[str, float]:
    c = phase_space.columns()
    w = c["weight"].astype(np.float64)
    e = c["energy"].astype(np.float64)
    return {
        "n": float(len(phase_space)),
        "weight": float(w.sum()),
        "weight_energy": float((w * e).sum()),
        "n_photon": float(np.count_nonzero(c["particle_type"] == 1)),
        "n_electron": float(np.count_nonzero(c["particle_type"] == 2)),
        "mean_energy": float((w * e).sum() / w.sum()),
    }


@pytest.mark.parametrize("device", DEVICES)
def test_attenuation_mode_matches_host_to_float32(device: str) -> None:
    _require(device)
    from pyRadMC.backends.warp.presolve import presolve_head_device

    kwargs = dict(
        cross_sections=AnalyticCrossSections(),
        n_histories=N,
        seed=SEED,
        exit_z=60.0,
        stack=_stack(),
        mode="attenuation",
    )
    host = _totals(presolve_head(_source(), **kwargs))
    got = _totals(presolve_head_device(_source(), device=device, **kwargs))
    # No RNG here: same primaries, same geometry, same mu — float32 only.
    assert got["n"] == host["n"]
    assert got["n_electron"] == 0.0
    np.testing.assert_allclose(got["weight"], host["weight"], rtol=2e-3)
    np.testing.assert_allclose(got["weight_energy"], host["weight_energy"], rtol=2e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_first_compton_totals_match_host_statistically(device: str) -> None:
    _require(device)
    from pyRadMC.backends.warp.presolve import presolve_head_device

    kwargs = dict(
        cross_sections=AnalyticCrossSections(),
        n_histories=N,
        seed=SEED,
        exit_z=60.0,
        stack=_stack(),
        mode="first_compton",
    )
    host = _totals(presolve_head(_source(), **kwargs))
    got = _totals(presolve_head_device(_source(), device=device, **kwargs))
    assert got["n_electron"] == 0.0  # no air => no contaminant electrons
    # The transmitted backbone is identical; only the forced-Compton secondaries
    # differ by RNG, a small fraction of the total — 0.5% brackets it generously.
    np.testing.assert_allclose(got["n"], host["n"], rtol=5e-3)
    np.testing.assert_allclose(got["weight"], host["weight"], rtol=5e-3)
    np.testing.assert_allclose(got["weight_energy"], host["weight_energy"], rtol=5e-3)
    np.testing.assert_allclose(got["mean_energy"], host["mean_energy"], rtol=2e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_air_column_adds_contaminant_electrons_matching_host(device: str) -> None:
    _require(device)
    from pyRadMC.backends.warp.presolve import presolve_head_device

    kwargs = dict(
        cross_sections=AnalyticCrossSections(),
        n_histories=N,
        seed=SEED,
        exit_z=95.0,
        stack=_stack(),
        air=AirColumn(z_start=51.0, z_end=95.0, material=WATER, density=1.2e-3),
        mode="first_compton",
    )
    host = _totals(presolve_head(_source(), **kwargs))
    got = _totals(presolve_head_device(_source(), device=device, **kwargs))
    assert got["n_electron"] > 0.0
    np.testing.assert_allclose(got["n_electron"], host["n_electron"], rtol=2e-2)
    np.testing.assert_allclose(got["weight"], host["weight"], rtol=5e-3)
    np.testing.assert_allclose(got["weight_energy"], host["weight_energy"], rtol=5e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_return_buffer_exposes_the_same_normalized_population(device: str) -> None:
    """The no-copy hook: the on-device buffer holds the copied-back phase space.

    ``return_buffer=True`` hands back the device :class:`ExitBuffer` and its filled
    ``count`` for a future transport path that skips the round-trip. Its first
    ``count`` rows must be the *normalized* population the phase space carries — same
    count, same summed weight and weight-energy — so the two paths are interchangeable.
    """
    _require(device)
    from pyRadMC.backends.warp.presolve import presolve_head_device

    phase_space, buf, count = presolve_head_device(
        _source(),
        cross_sections=AnalyticCrossSections(),
        n_histories=N,
        seed=SEED,
        exit_z=60.0,
        stack=_stack(),
        mode="first_compton",
        device=device,
        return_buffer=True,
    )
    assert count == len(phase_space)
    cols = phase_space.columns()
    w_buf = buf.weight.numpy()[:count].astype(np.float64)
    e_buf = buf.energy.numpy()[:count].astype(np.float64)
    # Weights are already normalized on-device, so these match to float32 copy noise.
    np.testing.assert_allclose(w_buf.sum(), cols["weight"].sum(), rtol=1e-6)
    np.testing.assert_allclose(
        (w_buf * e_buf).sum(), (cols["weight"] * cols["energy"]).sum(), rtol=1e-5
    )
    np.testing.assert_array_equal(
        np.sort(buf.particle_type.numpy()[:count]), np.sort(cols["particle_type"])
    )


def test_device_presolve_requires_a_stack() -> None:
    from pyRadMC.backends.warp.presolve import presolve_head_device

    with pytest.raises(ValueError, match="requires a stack"):
        presolve_head_device(
            _source(),
            cross_sections=AnalyticCrossSections(),
            n_histories=8,
            seed=SEED,
            exit_z=60.0,
            stack=None,
            device="cpu",
        )
