"""CT-grade throughput benchmarks: synthetic thorax, spectral beam, tabulated media.

The 64^3 water benchmarks measure the engine where a global Woodcock majorant is
nearly free; a patient CT is the regime it is not. This workload is a synthetic
thorax HU volume through the CT adapter's default calibration — an adipose-shelled
water body with two lungs, a spine and a sternum in an air margin (48 percent air,
24 percent lung by voxel; majorant density 1.42 at the spine) — with an Ali-Rogers
6 MV divergent beam entering anteriorly and the EPICS tabulated multi-material
backend. Measured at adoption (2026-07-19, laptop RTX 4070): ~57 percent of
Woodcock trials along the primaries are virtual, the electron kernel holds ~74
percent of GPU kernel time and the photon kernel ~16, and the in-kernel spectral
generation added the same day runs the open field at ~7.5e7 histories/s and the
100-beamlet Dij at ~2.5e6.

Alerts, not pass/fail: baselines drift, no floor. Requires the EPICS libraries in
the default cache (``python -m pyRadMC.data.tabulated.build`` or any tabulated
demo fetches them); skips rather than downloading ~120 MB from a perf tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyRadMC.adapters.ct import grid_from_hu  # noqa: E402
from pyRadMC.data.tabulated import build  # noqa: E402
from pyRadMC.data.tabulated.precompile import compile_materials  # noqa: E402
from pyRadMC.data.tabulated.source import TabulatedCrossSections  # noqa: E402
from pyRadMC.geometry.source import SpectralBeamletSource, SpectralBeamSource  # noqa: E402
from pyRadMC.geometry.spectrum import ali_rogers_mv  # noqa: E402

pytestmark = [pytest.mark.perf, pytest.mark.warp]

SHAPE = (128, 64, 80)  # (x, y, z) at 3 mm: 38.4 x 19.2 x 24.0 cm; z anterior->posterior
SPACING = (0.3, 0.3, 0.3)


def thorax_hu() -> np.ndarray:
    """Synthetic thorax HU volume in engine axis order (x, y, z)."""
    nx, _ny, nz = SHAPE
    x = (np.arange(nx) + 0.5) * SPACING[0]
    z = (np.arange(nz) + 0.5) * SPACING[2]
    xg, zg = np.meshgrid(x, z, indexing="ij")  # (nx, nz); uniform in y

    cx, cz = nx * SPACING[0] / 2.0, 12.0  # body centre
    body = ((xg - cx) / 16.0) ** 2 + ((zg - cz) / 9.5) ** 2  # ellipse, semi-axes cm

    hu2d = np.full((nx, nz), -1000.0)  # air outside the body
    hu2d[body <= 1.0] = -80.0  # adipose shell
    hu2d[body <= 0.81] = 0.0  # water-like interior (0.9^2)

    # Two lungs: ellipses in (x, z), mirrored about the midline.
    for sx in (-1.0, 1.0):
        lung = ((xg - (cx + sx * 7.5)) / 5.5) ** 2 + ((zg - 10.5) / 6.5) ** 2
        hu2d[lung <= 1.0] = -750.0

    # Spine: posterior midline cylinder; sternum: small anterior bone block.
    spine = ((xg - cx) / 1.5) ** 2 + ((zg - 19.5) / 1.5) ** 2
    hu2d[spine <= 1.0] = 700.0
    hu2d[(np.abs(xg - cx) < 1.5) & (np.abs(zg - 3.6) < 0.5)] = 700.0

    return np.broadcast_to(hu2d[:, None, :], SHAPE).astype(np.float64).copy()


@pytest.fixture(scope="module")
def thorax_engine():
    """The thorax grid with compiled EPICS media on a CUDA Warp engine."""
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    epdl, eedl = build.library_path("epdl", None), build.library_path("eedl", None)
    if not (epdl.is_file() and eedl.is_file()):
        pytest.skip("EPICS libraries not cached; run python -m pyRadMC.data.tabulated.build")
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = grid_from_hu(thorax_hu(), SPACING)
    data = compile_materials(
        epdl.read_text(encoding="latin-1"), eedl.read_text(encoding="latin-1"), e_max=8.0
    )
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device="cuda:0", chunk_size=262_144)


def _center() -> tuple[float, float]:
    return SHAPE[0] * SPACING[0] / 2.0, SHAPE[1] * SPACING[1] / 2.0


@pytest.mark.gpu
def test_cuda_ct_open_field_throughput(thorax_engine, benchmark) -> None:
    """6 MV spectral open field on the thorax; in-kernel spectral generation."""
    cx, cy = _center()
    source = SpectralBeamSource(
        spectrum=ali_rogers_mv("varian-6mv"),
        focal_point=(cx, cy, -100.0),
        center=(cx, cy, 0.0),  # 10 x 10 aperture at the phantom face, SSD 100
        width_u=10.0,
        width_v=10.0,
    )
    thorax_engine.run(source, n_histories=100_000, n_batches=1, seed=SEED)  # compile + warm

    n = 4_000_000
    benchmark.pedantic(
        lambda: thorax_engine.run(source, n_histories=n, n_batches=10, seed=SEED),
        rounds=3,
        iterations=1,
    )


@pytest.mark.gpu
def test_cuda_ct_spectral_dij_throughput(thorax_engine, benchmark) -> None:
    """100-beamlet spectral Dij on the thorax; the pyRadPlan-adapter shape."""
    cx, cy = _center()
    centers = [
        (cx - 4.5 + i, cy - 4.5 + j, 0.0)
        for j in np.arange(0.0, 10.0)
        for i in np.arange(0.0, 10.0)
    ]
    source = SpectralBeamletSource(
        spectrum=ali_rogers_mv("varian-6mv"),
        focal_point=(cx, cy, -100.0),
        centers=centers,
        width_u=1.0,
        width_v=1.0,
    )
    thorax_engine.run_dij(source, n_histories_per_beamlet=1_000, n_batches=1, seed=SEED)  # warm

    benchmark.pedantic(
        lambda: thorax_engine.run_dij(
            source, n_histories_per_beamlet=20_000, n_batches=10, seed=SEED
        ),
        rounds=3,
        iterations=1,
    )
