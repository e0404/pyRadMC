"""Per-call host setup of the collimator head pre-solve and of collimator objects.

An aperture optimizer rebuilds collimator geometry and re-runs the head pre-solve
hundreds of times, with the same device materials each time. These benchmarks track
the setup that is *not* the kernel or the primary sampling: the attenuation-table
build (cold, from scalar cross-section queries, versus warm, from the per-source
memo), construction of an ``MLC``/``BeamLimitingStack``, and the whole setup of a
tiny device pre-solve (2000 histories, so the kernel and sampling are negligible).

Alerts, not pass/fail. The tabulated source is synthetic (no EPICS data needed): the
cost being measured is the Python query loop, not the table contents. The GPU row is
timed on whatever else shares the card; compare only best-of-N and interleaved.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.interface import PhotonProcess
from pyradmc.data.tabulated.model import TabulatedData
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry import mucache
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.head import _MuTables, mu_tables

pytestmark = pytest.mark.perf

PAIRS = ((1, 19.3), (1, 19.3), (0, 1.205e-3))
N_PAIRS = 60


def _source() -> TabulatedCrossSections:
    pe = np.geomspace(0.01, 10.0, 64)
    base = {
        PhotonProcess.COMPTON: 0.15 * np.ones_like(pe),
        PhotonProcess.PHOTOELECTRIC: 1.0e-3 / pe**3,
        PhotonProcess.PAIR: np.maximum(pe - 1.022, 0.0) * 1.0e-3,
        PhotonProcess.RAYLEIGH: 0.01 / pe,
    }
    two = lambda a, f: np.stack([a, f * a])  # noqa: E731
    return TabulatedCrossSections(
        TabulatedData(
            photon_energies=pe,
            mu_over_rho={p: two(v, 3.0) for p, v in base.items()},
            electron_energies=pe,
            restricted_stopping=two(2.0 / pe, 0.5),
            radiative_stopping=two(0.01 * pe, 0.5),
            moller=two(0.1 / pe, 0.5),
            csda_range=two(0.5 * pe, 2.0),
            scattering_power=two(1.0 / pe, 0.5),
            delta_cut=0.2,
            materials=("water", "air"),
            provenance="synthetic",
        )
    )


def _mlc() -> MLC:
    edges = tuple(np.linspace(-10.0, 10.0, N_PAIRS + 1).tolist())
    return MLC(46.0, 50.0, edges, (-1.5,) * N_PAIRS, (1.5,) * N_PAIRS, 8.0, 1, 19.3)


def test_mu_tables_cold_build(benchmark) -> None:
    xs = _source()

    def cold() -> None:
        mucache.clear()
        _MuTables(xs, PAIRS, 0.010, 6.0, 512)

    benchmark(cold)


def test_mu_tables_warm_lookup(benchmark) -> None:
    xs = _source()
    mu_tables(xs, PAIRS, 0.010, 6.0, 512)
    benchmark(lambda: mu_tables(xs, PAIRS, 0.010, 6.0, 512))


def test_mlc_construction(benchmark) -> None:
    benchmark(_mlc)


def test_beam_frame_construction(benchmark) -> None:
    benchmark(lambda: BeamFrame(origin=(8.0, 8.0, -100.0)))


@pytest.mark.gpu
def test_cuda_presolve_setup(benchmark) -> None:
    wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.presolve import presolve_head_device
    from pyradmc.geometry.source import GaussianSpotBeamSource
    from pyradmc.geometry.spectrum import Spectrum

    focal = (8.0, 8.0, -100.0)
    source = GaussianSpotBeamSource(
        spectrum=Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0)),
        focal_point=focal,
        center=(8.0, 8.0, -70.0),
        width_u=6.0,
        width_v=6.0,
        sigma_u=0.1,
        sigma_v=0.1,
    )
    jaw = JawPair("u", 40.0, 45.0, -1000.0, 0.5, material=1, density=19.3)
    stack = BeamLimitingStack(frame=BeamFrame(origin=focal), devices=(jaw, _mlc()))
    xs = _source()

    def run() -> object:
        return presolve_head_device(
            source,
            cross_sections=xs,
            n_histories=2000,
            seed=1,
            exit_z=60.0,
            stack=stack,
            mode="attenuation",
            device="cuda:0",
            return_device_source=True,
        )

    run()  # compile + warm every cache
    benchmark.pedantic(run, rounds=20, iterations=5)
