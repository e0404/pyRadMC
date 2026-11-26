"""Stochastic Dij truncation on the reference engine: the bias the mode removes.

The hard rule drops every entry below ``truncation * column_max``. At low
statistics a single hit in a tail voxel exceeds the threshold and survives; at high
statistics the tail converges below it and is dropped whole, so the dose the hard
rule removes *grows with the histories per beamlet* (measured on a TG-119 arc: body
dose kept 0.992 / 0.979 / 0.947 / 0.894 at 20k / 40k / 80k / 160k). The stochastic
mode plays Russian roulette on the tail instead (Thieke et al., Med. Phys. 29 (2002)
676, doi:10.1118/1.1469633): unbiased for every history count.

Every comparison here is between runs with the same seed and history count, so
the dense columns are bit-identical before sparsification and the Monte Carlo
noise cancels exactly. What remains is the roulette's own, known variance
``sum over tail entries of m (t - m)``, and its known support (a stored tail value
is 0 or ``t``), from which every bound below is derived.

Geometry and counts: one 1 x 1 cm beamlet of 6 MeV photons centred on a
16 x 16 x 16 cm water cube in 1 cm voxels, KERMA mode (the scatter-photon tail is
what is truncated; electron transport costs about 20x the time and changes nothing
the assembler is tested for). ``N = 2000`` is the smallest count of a probe at this seed
with more than 50 sub-threshold entries in the column (61), so the normal
approximation of the roulette's z-score is reasonable; ``4 N = 8000`` is the factor
the bias is documented over. The probe (hard rule: removed fraction of the column
sum, in roulette sigmas, and tail entries): N = 1000: 4.0e-4 (7 sigma, 22);
2000: 7.3e-4 (9 sigma, 61); 4000: 1.7e-3 (15 sigma, 181); 16000: 8.6e-3 (32
sigma, 1081). The counts were fixed from the runtime budget and the tail population
before the assertions were written, not tuned to them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from pyradmc import DIJ_TRUNCATION_RELATIVE
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource
from pyradmc.rng.host import HostRNG
from pyradmc.scoring.dij import DijResult
from tests.conftest import SEED

N = 2_000
# Two-sided normal false-alarm probability of 6.3e-5 per z-score assertion.
K_SIGMA = 4.0


def _engine() -> ReferenceEngine:
    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())


def _beamlet(n_x: int = 1, n_y: int = 1) -> BeamletGridSource:
    half_x, half_y = 0.5 * n_x, 0.5 * n_y
    return BeamletGridSource(
        energy=6.0,
        z=-1.0,
        x_range=(8.0 - half_x, 8.0 + half_x),
        y_range=(8.0 - half_y, 8.0 + half_y),
        n_x=n_x,
        n_y=n_y,
    )


def _run(n: int, source: BeamletGridSource | None = None, **kwargs: object) -> DijResult:
    return _engine().run_dij(
        _beamlet() if source is None else source,
        n_histories_per_beamlet=n,
        n_batches=4,
        seed=SEED,
        transport_electrons=False,
        **kwargs,  # type: ignore[arg-type]
    )


def _roulette_variance(full: DijResult, j: int) -> tuple[float, int]:
    """Variance of column ``j``'s stochastic sum about the full sum, and the tail size."""
    m = full.column_dense(j).ravel()
    t = DIJ_TRUNCATION_RELATIVE * m.max()
    tail = m[(m > 0.0) & (m < t)]
    return float(np.sum(tail * (t - tail))), int(tail.size)


@pytest.fixture(scope="module")
def runs() -> dict[tuple[int, str], DijResult]:
    out: dict[tuple[int, str], DijResult] = {}
    for n in (N, 4 * N):
        out[n, "full"] = _run(n, truncation=0.0)
        out[n, "stochastic"] = _run(n, truncation_mode="stochastic")
        out[n, "hard"] = _run(n)
    return out


class TestColumnSum:
    @pytest.mark.parametrize("n", [N, 4 * N])
    def test_stochastic_column_sum_is_the_untruncated_one(
        self, runs: dict[tuple[int, str], DijResult], n: int
    ) -> None:
        full, sto = runs[n, "full"], runs[n, "stochastic"]
        assert sto.truncation_mode == "stochastic"
        assert sto.truncation == DIJ_TRUNCATION_RELATIVE
        variance, n_tail = _roulette_variance(full, 0)
        assert n_tail > 50, "the tail must hold sub-threshold entries"
        assert sto.indices.size < full.indices.size, "the mode must sparsify"
        z = (sto.dose.sum() - full.dose.sum()) / math.sqrt(variance)
        assert abs(z) < K_SIGMA, f"stochastic column sum off by z = {z:.2f} at N = {n}"

    def test_hard_mode_removes_more_than_the_roulette_sigma(
        self, runs: dict[tuple[int, str], DijResult]
    ) -> None:
        """Characterization of the defect: at 4N the hard cut is a many-sigma bias."""
        full, hard = runs[4 * N, "full"], runs[4 * N, "hard"]
        assert hard.truncation_mode == "hard"
        variance, _ = _roulette_variance(full, 0)
        kept = hard.dose.sum() / full.dose.sum()
        sigma_rel = math.sqrt(variance) / full.dose.sum()
        assert 1.0 - kept > K_SIGMA * sigma_rel, (kept, sigma_rel)

    def test_hard_bias_grows_with_the_histories(
        self, runs: dict[tuple[int, str], DijResult]
    ) -> None:
        """The removed fraction is larger at 4N than at N: the defect is N-dependent."""

        def removed(n: int) -> float:
            return 1.0 - float(runs[n, "hard"].dose.sum() / runs[n, "full"].dose.sum())

        assert removed(4 * N) > removed(N) > 0.0

    def test_stochastic_run_is_reproducible(self, runs: dict[tuple[int, str], DijResult]) -> None:
        again = _run(N, truncation_mode="stochastic")
        first = runs[N, "stochastic"]
        for name in ("indptr", "indices", "dose", "sigma"):
            assert np.array_equal(getattr(again, name), getattr(first, name)), name

    def test_the_roulette_is_keyed_on_the_run_seed(
        self, runs: dict[tuple[int, str], DijResult]
    ) -> None:
        """The engine's stochastic matrix is the assembler's roulette, under the run seed.

        Rebuilt from the engine's own untruncated matrix (bit-identical dense columns:
        same seed and histories) with ``seed=provenance.seed`` it matches byte for
        byte; under another seed it does not. An engine that handed the assembler any
        other seed would pass every statistical test and fail this one.
        """
        from pyradmc.scoring.dij import DijAssembler

        full, sto = runs[4 * N, "full"], runs[4 * N, "stochastic"]
        assert sto.provenance is not None and sto.provenance.seed == SEED

        def roulette(seed: int) -> DijResult:
            asm = DijAssembler(
                grid_shape=full.grid_shape,
                n_beamlets=full.n_beamlets,
                n_histories_per_beamlet=full.n_histories_per_beamlet,
                n_batches=full.n_batches,
                truncation=DIJ_TRUNCATION_RELATIVE,
                truncation_mode="stochastic",
                seed=seed,
            )
            dose = np.stack([full.column_dense(j).ravel() for j in range(full.n_beamlets)])
            sigma = np.stack([full.sigma_dense(j).ravel() for j in range(full.n_beamlets)])
            asm.add_block(0, dose, sigma)
            return asm.finalize(energy_emitted=1.0, energy_deposited=0.0, energy_escaped=0.0)

        rebuilt = roulette(sto.provenance.seed)
        for name in ("indptr", "indices", "dose", "sigma"):
            assert np.array_equal(getattr(rebuilt, name), getattr(sto, name)), name
        other = roulette(sto.provenance.seed + 1)
        assert not np.array_equal(other.indices, sto.indices)

    def test_engine_rejects_an_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="truncation_mode"):
            _run(4, truncation_mode="soft")


@dataclass(frozen=True)
class _Plans:
    """Unit-weight plan doses of one lattice in three modes, and the roulette bounds."""

    full: np.ndarray
    sto: np.ndarray
    hard: np.ndarray
    variance: np.ndarray
    """Per voxel, ``sum_j w_j^2 m_ij (t_j - m_ij)`` over the tail entries."""
    reach: np.ndarray
    """Per voxel, ``sum_j w_j max(m_ij, t_j - m_ij)`` over the tail entries: the
    furthest the roulette can move that voxel's plan dose, whatever it draws."""
    nnz_full: int
    nnz_sto: int


@pytest.fixture(scope="module")
def plans() -> _Plans:
    source = _beamlet(2, 2)
    full = _run(4 * N, source=source, truncation=0.0)
    sto = _run(4 * N, source=source, truncation_mode="stochastic")
    hard = _run(4 * N, source=source)
    w = np.ones(source.n_beamlets)
    variance = np.zeros(full.n_voxels)
    reach = np.zeros(full.n_voxels)
    for j in range(source.n_beamlets):
        m = full.column_dense(j).ravel()
        t = DIJ_TRUNCATION_RELATIVE * m.max()
        tail = (m > 0.0) & (m < t)
        variance += np.where(tail, w[j] ** 2 * m * (t - m), 0.0)
        reach += np.where(tail, w[j] * np.maximum(m, t - m), 0.0)
    return _Plans(
        full=full.dose_for_weights(w).ravel(),
        sto=sto.dose_for_weights(w).ravel(),
        hard=hard.dose_for_weights(w).ravel(),
        variance=variance,
        reach=reach,
        nnz_full=int(full.indices.size),
        nnz_sto=int(sto.indices.size),
    )


def _rois(full: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "high": full > 0.1 * full.max(),
        "low": (full > 0.0) & (full < 0.01 * full.max()),
    }


def _dvh_d(dose_in_roi: np.ndarray, volume_percent: float) -> float:
    """Dose received by at least ``volume_percent`` of the ROI volume."""
    return float(np.percentile(dose_in_roi, 100.0 - volume_percent))


class TestStochasticTruncationAgainstDVH:
    """AGENTS.md 2.8: truncation is tested on DVH endpoints.

    A 2 x 2 lattice of 1 cm beamlets, unit weights, ``4 N`` histories each, stochastic
    against untruncated from the same seed. The plan-dose difference in voxel ``i``,
    ``Delta_i = sum_j w_j (X_ij - m_ij)``, is a sum of independent zero-mean terms with
    variance ``sigma_i^2 = sum_j w_j^2 m_ij (t_j - m_ij)`` over the tail entries, and
    since a stored tail value is 0 or ``t_j``, ``|Delta_i| <= R_i = sum_j w_j
    max(m_ij, t_j - m_ij)`` for every draw.

    - **Mean dose** of an ROI (the NTCP endpoint) is linear in the matrix, so the
      roulette leaves it unbiased: its shift is a z-score against
      ``sqrt(sum_i sigma_i^2) / n``, two-sided at ``K_SIGMA``.
    - **D2, D50, D98** are order statistics, which a zero-mean per-voxel noise does
      not leave unbiased. A percentile moves by at most ``max_i |Delta_i|`` over the
      ROI (it is 1-Lipschitz in the sup norm), hence by at most ``max_i R_i``: a
      bound with no false alarm. It is asserted in the high-dose region, where it
      is below 0.5 % of each endpoint (checked, so the assertion has teeth at the
      Warp truncation DVH test's level).

    Two ROIs: the high-dose region (dose above 10 % of the maximum, as in the Warp
    engine's truncation DVH test) and the low-dose region (above 0 and below 1 % of
    the maximum), where truncation acts and the hard rule's bias shows. In the low
    region of this four-beamlet plan the dose is itself of the order of ``t``, so
    ``max_i R_i`` exceeds the low-region D50 and its order statistics are dominated
    by the roulette's noise (measured at this seed: D50 5.2e-5 moved by 5.7e-5 under
    the stochastic mode and by -5.2e-5 under the hard rule). Only its linear
    endpoints are asserted there.
    """

    def test_mean_dose_is_unbiased(self, plans: _Plans) -> None:
        assert plans.nnz_sto < 0.95 * plans.nnz_full, "non-vacuity: the mode must sparsify"
        for name, roi in _rois(plans.full).items():
            n = int(np.count_nonzero(roi))
            assert n >= 64, name  # the high ROI is the 2 x 2 x 16 field core
            sigma_mean = math.sqrt(float(plans.variance[roi].sum())) / n
            shift = float(plans.sto[roi].mean() - plans.full[roi].mean())
            if sigma_mean == 0.0:
                # No tail entry reaches this ROI: the roulette cannot touch it.
                assert shift == 0.0, name
                continue
            z = shift / sigma_mean
            assert abs(z) < K_SIGMA, f"{name}-dose ROI mean moved by z = {z:.2f}"

    def test_hard_rule_biases_the_low_dose_mean(self, plans: _Plans) -> None:
        """Characterization: the hard rule's low-dose mean shift is a many-sigma bias."""
        roi = _rois(plans.full)["low"]
        n = int(np.count_nonzero(roi))
        sigma_mean = math.sqrt(float(plans.variance[roi].sum())) / n
        shift = float(plans.hard[roi].mean() - plans.full[roi].mean())
        assert shift < -K_SIGMA * sigma_mean, (shift, sigma_mean)

    @pytest.mark.parametrize("volume_percent", [2.0, 50.0, 98.0])
    def test_high_dose_dvh_endpoints_within_the_roulette_reach(
        self, plans: _Plans, volume_percent: float
    ) -> None:
        roi = _rois(plans.full)["high"]
        reach = float(plans.reach[roi].max())
        d_full = _dvh_d(plans.full[roi], volume_percent)
        d_sto = _dvh_d(plans.sto[roi], volume_percent)
        assert reach < 5.0e-3 * d_full, "non-vacuity: the bound must be tight"
        assert abs(d_sto - d_full) <= reach, (
            f"D{volume_percent:g} moved by {abs(d_sto - d_full):.3e} > reach {reach:.3e}"
        )
