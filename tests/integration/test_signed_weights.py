"""A primary of weight -w deposits, in expectation, exactly minus the dose of +w.

Transport is linear in the statistical weight: no physics decision may depend on its
sign. Signed weights are not exotic here — every derivative estimator of
:mod:`pyradmc.geometry.aperture_gradient` emits them (a closing leaf edge is a
negative derivative), and the edge gradients of a direct aperture optimization are
contracted from exactly such runs. Pinned per engine on a 2 MeV, 2 x 2 cm parallel
beam in water, two independent seeds, batched chi-squared over the dose region.

**Known defect, pinned as a strict xfail:** with electrons transported, both engines
drop a negative-weight electron's continuous-loss deposits (the deposit guard
``amount <= 0`` tests the *weighted* amount: ``deposit_or_escape`` in
``transport/electron.py``, ``_deposit_or_escape`` in the Warp kernels), so ``-D(-w)``
comes out at about 0.22 ``D(w)``; KERMA mode is unaffected. The same guard on the
bremsstrahlung photon weight (``photon_w > 0``) drops negative-weight photons.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import Primary, Source
from pyradmc.rng import RNGState, uniform
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

N_BATCHES = 8
ENGINES = [
    pytest.param("ref", id="ref"),
    pytest.param("cpu", marks=pytest.mark.warp, id="warp-cpu"),
    pytest.param("cuda:0", marks=[pytest.mark.warp, pytest.mark.gpu], id="warp-cuda"),
]
HISTORIES = {"ref": 1_600, "cpu": 16_000, "cuda:0": 160_000}
FIELD = (1.0, 3.0)  # cm, in x and in y


class _SignedBeam(Source):
    """A 2 MeV parallel beam along +z over 2 x 2 cm, every primary at one signed weight."""

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def max_energy(self) -> float:
        """The beam energy."""
        return 2.0

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one primary at a uniform field position; consumes two uniforms."""
        x = FIELD[0] + (FIELD[1] - FIELD[0]) * uniform(rng_state)
        y = FIELD[0] + (FIELD[1] - FIELD[0]) * uniform(rng_state)
        return Primary(2.0, x, y, 0.0, 0.0, 0.0, 1.0, weight=self._weight)

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, Any]:
        """Vectorized host batch (chunk-invariant ``PCG64(seed)`` stream, two draws each)."""
        bitgen = np.random.PCG64(seed)
        bitgen.advance(2 * history_offset)
        u = np.random.Generator(bitgen).random((n, 2))
        xy = FIELD[0] + (FIELD[1] - FIELD[0]) * u
        f32 = np.float32
        return {
            "particle_type": np.ones(n, dtype=np.int32),
            "energy": np.full(n, 2.0, dtype=f32),
            "x": xy[:, 0].astype(f32),
            "y": xy[:, 1].astype(f32),
            "z": np.zeros(n, dtype=f32),
            "ux": np.zeros(n, dtype=f32),
            "uy": np.zeros(n, dtype=f32),
            "uz": np.ones(n, dtype=f32),
            "weight": np.full(n, self._weight, dtype=f32),
        }


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(0.5, 0.5, 0.5))


def _engine(name: str) -> Any:
    xs = AnalyticCrossSections(geometry_densities=_grid().max_density_by_material())
    if name == "ref":
        return ReferenceEngine(grid=_grid(), cross_sections=xs, rng=HostRNG())
    wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    if name.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    return WarpEngine(grid=_grid(), cross_sections=xs, device=name)


def _check(name: str, transport_electrons: bool) -> None:
    engine = _engine(name)
    kwargs = dict(
        n_histories=HISTORIES[name], n_batches=N_BATCHES, transport_electrons=transport_electrons
    )
    plus = engine.run(_SignedBeam(0.5), seed=SEED, **kwargs)
    minus = engine.run(_SignedBeam(-0.5), seed=SEED + 1, **kwargs)
    # The pooled dose, so the mask follows neither side's fluctuations.
    pooled = plus.dose - minus.dose
    mask = pooled > 0.1 * pooled.max()
    mask &= (plus.dose_sigma > 0.0) & (minus.dose_sigma > 0.0)
    if mask.sum() < 50:  # not an AssertionError: the xfail below accepts only the oracle's
        raise RuntimeError(f"only {int(mask.sum())} voxels in the compared region")
    assert_chi2_consistent_batched(
        plus.dose,
        plus.dose_sigma,
        N_BATCHES,
        -minus.dose,
        minus.dose_sigma,
        N_BATCHES,
        mask=mask,
    )


@pytest.mark.parametrize("name", ENGINES)
def test_negative_weight_mirrors_positive_in_kerma_mode(name: str) -> None:
    _check(name, transport_electrons=False)


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="negative-weight electrons lose their continuous-loss deposits "
    "(deposit guard `amount <= 0` on the weighted amount)",
)
@pytest.mark.parametrize("name", ENGINES)
def test_negative_weight_mirrors_positive_with_electrons(name: str) -> None:
    _check(name, transport_electrons=True)
