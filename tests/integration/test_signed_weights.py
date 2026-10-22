"""A primary of weight -w deposits, in expectation, exactly minus the dose of +w.

Transport is linear in the statistical weight: no physics decision may depend on its
sign. Signed weights are not exotic here — every derivative estimator of
:mod:`pyradmc.geometry.aperture_gradient` emits them (a closing leaf edge is a
negative derivative), and the edge gradients of a direct aperture optimization are
contracted from exactly such runs. Pinned per engine on a 2 MeV, 2 x 2 cm parallel
beam in water, two independent seeds, batched chi-squared over the dose region.

Before the fix of 2026-10-23 both engines dropped a negative-weight electron's
continuous-loss deposits (the deposit guard tested the *weighted* amount, ``amount
<= 0``) and its bremsstrahlung photons (``photon_w > 0``), so ``-D(-w)`` came out at
0.22 ``D(w)``; the guards now decide on the unweighted energy and on a nonzero
photon weight, and the photon roulette cap compares ``abs(w)``. Positive-weight
results are bit-identical to before on ref, Warp cpu and Warp cuda
(docs/decisions.md, "DAO gradient consistency"). Pinned here: the mirror within
each engine (statistically; exactly on the reference engine, whose every decision
is now independent of the weight's sign and whose scoring is float64), and a
signed-weight beam on Warp against the reference engine, electrons on.
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
    """A 2 MeV parallel beam along +z over a square field, every primary at one signed weight.

    Default: 2 x 2 cm entering at the surface ``z = 0``.
    """

    def __init__(self, weight: float, field: tuple[float, float] = FIELD, z: float = 0.0) -> None:
        self._weight = weight
        self._field = field
        self._z = z

    @property
    def max_energy(self) -> float:
        """The beam energy."""
        return 2.0

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one primary at a uniform field position; consumes two uniforms."""
        lo, hi = self._field
        x = lo + (hi - lo) * uniform(rng_state)
        y = lo + (hi - lo) * uniform(rng_state)
        return Primary(2.0, x, y, self._z, 0.0, 0.0, 1.0, weight=self._weight)

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, Any]:
        """Vectorized host batch (chunk-invariant ``PCG64(seed)`` stream, two draws each)."""
        bitgen = np.random.PCG64(seed)
        bitgen.advance(2 * history_offset)
        u = np.random.Generator(bitgen).random((n, 2))
        xy = self._field[0] + (self._field[1] - self._field[0]) * u
        f32 = np.float32
        return {
            "particle_type": np.ones(n, dtype=np.int32),
            "energy": np.full(n, 2.0, dtype=f32),
            "x": xy[:, 0].astype(f32),
            "y": xy[:, 1].astype(f32),
            "z": np.full(n, self._z, dtype=f32),
            "ux": np.zeros(n, dtype=f32),
            "uy": np.zeros(n, dtype=f32),
            "uz": np.ones(n, dtype=f32),
            "weight": np.full(n, self._weight, dtype=f32),
        }


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(0.5, 0.5, 0.5))


def _engine(name: str, grid: VoxelGrid | None = None) -> Any:
    grid = _grid() if grid is None else grid
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    if name == "ref":
        return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    if name.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    return WarpEngine(grid=grid, cross_sections=xs, device=name)


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
    if mask.sum() < 50:
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


@pytest.mark.parametrize("name", ENGINES)
def test_negative_weight_mirrors_positive_with_electrons(name: str) -> None:
    _check(name, transport_electrons=True)


def test_reference_mirror_is_exact_on_one_stream() -> None:
    """Same seed, weight -w against +w: the reference dose negates bit for bit.

    No transport decision may read the weight's sign (deposit guards, photon
    guard, roulette cap), and float64 negation is exact, so every deposit of the
    negative run is the exact negative of its twin's.
    """
    engine = _engine("ref")
    kwargs = dict(n_histories=400, n_batches=N_BATCHES, seed=SEED, transport_electrons=True)
    plus = engine.run(_SignedBeam(0.5), **kwargs)
    minus = engine.run(_SignedBeam(-0.5), **kwargs)
    assert np.any(plus.dose > 0.0)
    np.testing.assert_array_equal(minus.dose, -plus.dose)
    assert minus.energy_deposited == -plus.energy_deposited


@pytest.mark.warp
def test_signed_weight_beam_matches_the_reference_engine() -> None:
    """A negative-weight beam, electrons on: Warp cpu against the reference engine.

    A broad beam entering from vacuum, scored as a depth-dose curve in 16 x 16 cm
    by 5 mm slabs, so every compared bin holds many deposits per batch (on fine
    voxels at these history counts the batched chi-squared sits below its null and
    has no power). Power: either engine alone keeping the old weighted-amount
    deposit guard is rejected (docs/decisions.md, "DAO gradient consistency").
    """
    grid = VoxelGrid.uniform_water(shape=(1, 1, 20), spacing=(16.0, 16.0, 0.5))
    beam = _SignedBeam(-0.5, field=(0.0, 16.0), z=-1.0)
    kwargs = dict(n_batches=16, transport_electrons=True)
    ref = _engine("ref", grid).run(beam, n_histories=3_200, seed=SEED, **kwargs)
    warp = _engine("cpu", grid).run(beam, n_histories=32_000, seed=SEED, **kwargs)
    pooled = -(ref.dose + warp.dose)
    mask = pooled > 0.1 * pooled.max()
    mask &= (ref.dose_sigma > 0.0) & (warp.dose_sigma > 0.0)
    if mask.sum() < 15:
        raise RuntimeError(f"only {int(mask.sum())} depth bins in the compared region")
    assert_chi2_consistent_batched(
        ref.dose, ref.dose_sigma, 16, warp.dose, warp.dose_sigma, 16, mask=mask
    )
