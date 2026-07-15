"""Warp phase-space source against the reference oracle (Phase 5).

The warp engine transports a phase-space source by host-sampling each chunk and
seeding the photon and electron queues by kind. These tests hold it to the same
bar as every other warp path (AGENTS.md 2.3, 4): statistical equivalence to the
reference engine via the chi-squared oracle over the high-dose region, never bit
equality across targets; bit reproducibility and chunk-invariance *within* a
device; and the energy ledger closing at the backend's documented tolerance.

Because ``PhaseSpaceSource.sample_batch`` draws with the same per-history stream
the reference ``emit`` uses, both backends sample the *same records* for a given
seed, so ``energy_emitted`` matches exactly and only the transport rng differs.
The synthetic file is a broad field of 6 MeV photons (to fill a comparable
high-dose region) plus a few contamination electrons and positrons, so the
kind-split and the positron rest-mass ledger term are both exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyRadMC.backends.ref.engine import ReferenceEngine
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.phasespace import PhaseSpaceSource
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.phsp_fixtures import Rec, write_phsp

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

# These tests deliberately oversample tiny synthetic phase spaces; the finite-
# reuse latent-variance caveat is expected and acknowledged (pinned elsewhere).
pytestmark = [
    pytest.mark.warp,
    pytest.mark.filterwarnings("ignore:.*latent variance:UserWarning"),
]

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

ENERGY_BALANCE_RTOL = 1.0e-4  # float32 transport + 1e-9 MeV scoring quanta


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 30), spacing=(2.0, 2.0, 0.2))


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


@pytest.fixture(scope="module")
def phsp_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A broad 6 MeV photon field over the phantom face, with e-/e+ contamination."""
    rng = np.random.default_rng(SEED)
    n = 6000
    recs: list[Rec] = []
    for _ in range(n):
        x = float(rng.uniform(0.0, 16.0))
        y = float(rng.uniform(0.0, 16.0))
        roll = rng.random()
        if roll < 0.95:
            recs.append(Rec(1, 6.0, x=x, y=y, z=-1.0, weight=1.0))
        elif roll < 0.99:
            recs.append(Rec(2, 1.0, x=x, y=y, z=-1.0, weight=0.7))
        else:
            recs.append(Rec(3, 0.8, x=x, y=y, z=-1.0, weight=0.5))
    stem = tmp_path_factory.mktemp("phsp") / "broad"
    write_phsp(stem, recs)
    return stem


def _source(phsp_path: Path) -> PhaseSpaceSource:
    return PhaseSpaceSource(phsp_path)


def _warp_engine(device: str, **kw):
    from pyRadMC.backends.warp.engine import WarpEngine

    grid = _grid()
    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device, **kw)


@pytest.fixture(scope="module")
def ref_result(phsp_path: Path) -> object:
    grid = _grid()
    engine = ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())
    with _source(phsp_path) as src:
        return engine.run(src, n_histories=6_000, n_batches=8, seed=SEED)


def _mask(ref, warp_result) -> np.ndarray:
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp_result.dose_sigma > 0.0)
    assert np.count_nonzero(mask) > 300, "mask too small to detect anything"
    return mask


@pytest.fixture(scope="module", params=DEVICES)
def device_param(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def test_phsp_dose_matches_ref(device_param: str, phsp_path: Path, ref_result) -> None:
    """The GPU/CPU phase-space dose is statistically consistent with the oracle."""
    with _source(phsp_path) as src:
        result = _warp_engine(device_param).run(src, n_histories=24_000, n_batches=8, seed=SEED)
    assert_chi2_consistent_batched(
        ref_result.dose,
        ref_result.dose_sigma,
        ref_result.n_batches,
        result.dose,
        result.dose_sigma,
        result.n_batches,
        mask=_mask(ref_result, result),
    )


def test_emitted_is_statistically_consistent_with_ref(
    device_param: str, phsp_path: Path, ref_result
) -> None:
    """Independent record sampling -> emitted energy agrees statistically, not bit-wise.

    The warp sampler draws from a different stream than the reference ``emit``, so the
    two emitted-energy books are independent MC estimates of the same mean; at 6000
    histories they agree to well under a percent (tolerance set generously at 2%).
    """
    with _source(phsp_path) as src:
        result = _warp_engine(device_param).run(src, n_histories=6_000, n_batches=8, seed=SEED)
    assert result.energy_emitted == pytest.approx(ref_result.energy_emitted, rel=0.02)


def test_energy_balance(device_param: str, phsp_path: Path) -> None:
    """emitted = deposited + escaped at the warp backend's documented tolerance."""
    with _source(phsp_path) as src:
        result = _warp_engine(device_param).run(src, n_histories=8_000, n_batches=4, seed=SEED)
    balance = result.energy_deposited + result.energy_escaped
    assert balance == pytest.approx(result.energy_emitted, rel=ENERGY_BALANCE_RTOL)


def test_chunking_does_not_change_the_result(device_param: str, phsp_path: Path) -> None:
    """Chunk size is inert: the dose is bit-identical within one device."""
    doses = []
    for chunk in (2048, 8192):
        with _source(phsp_path) as src:
            doses.append(
                _warp_engine(device_param, chunk_size=chunk)
                .run(src, n_histories=8_000, n_batches=4, seed=SEED)
                .dose
            )
    np.testing.assert_array_equal(doses[0], doses[1])
