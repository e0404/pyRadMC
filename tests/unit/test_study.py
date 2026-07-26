"""The study instrumentation: toy plan optimization and DVH endpoints.

The noise/bias study's conclusions are only as trustworthy as its harness, so
the harness is tested like everything else: the optimizer must solve problems
with known answers exactly, its gradient must match finite differences, and
the DVH endpoint must match its percentile definition on hand-computable
input. All deterministic — the stochastics live in the study's Dij inputs,
never in the harness.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_array

from pyRadMC.study import (
    ToyPlanProblem,
    dose_at_volume,
    objective_and_gradient,
    optimize_weights,
)


def _diagonal_problem(n: int = 5, prescription: float = 2.0) -> ToyPlanProblem:
    """Each beamlet deposits unit dose in exactly its own PTV voxel.

    The objective is then separable with the exact minimum w_j = prescription,
    reachable within the w >= 0 bounds: an analytic anchor for the optimizer.
    """
    dij = csc_array(np.eye(n))
    ptv = np.ones(n, dtype=bool)
    oar = np.zeros(n, dtype=bool)
    return ToyPlanProblem(dij=dij, ptv=ptv, oar=oar, prescription=prescription)


class TestObjective:
    def test_perfect_plan_has_zero_objective_and_gradient(self) -> None:
        problem = _diagonal_problem(prescription=2.0)
        f, g = objective_and_gradient(problem, np.full(5, 2.0))
        assert f == pytest.approx(0.0, abs=1.0e-30)
        np.testing.assert_allclose(g, 0.0, atol=1.0e-15)

    def test_gradient_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(20260711)
        n_vox, n_beam = 40, 6
        dij = csc_array(rng.random((n_vox, n_beam)))
        ptv = np.zeros(n_vox, dtype=bool)
        ptv[:15] = True
        oar = np.zeros(n_vox, dtype=bool)
        oar[15:30] = True
        problem = ToyPlanProblem(dij=dij, ptv=ptv, oar=oar, prescription=1.0)
        w = rng.random(n_beam)
        _, g = objective_and_gradient(problem, w)
        eps = 1.0e-7
        for j in range(n_beam):
            dw = np.zeros(n_beam)
            dw[j] = eps
            f_plus, _ = objective_and_gradient(problem, w + dw)
            f_minus, _ = objective_and_gradient(problem, w - dw)
            assert g[j] == pytest.approx((f_plus - f_minus) / (2.0 * eps), rel=1.0e-5)

    def test_oar_penalty_is_one_sided(self) -> None:
        """Dose below the OAR limit must contribute nothing, in value or gradient."""
        n = 4
        dij = csc_array(np.eye(n))
        ptv = np.array([True, True, False, False])
        oar = np.array([False, False, True, True])
        problem = ToyPlanProblem(
            dij=dij, ptv=ptv, oar=oar, prescription=1.0, oar_max=0.5, oar_penalty=10.0
        )
        # OAR voxels get 0.2 < 0.5: only the PTV term remains.
        f, g = objective_and_gradient(problem, np.array([1.0, 1.0, 0.2, 0.2]))
        assert f == pytest.approx(0.0, abs=1.0e-30)
        np.testing.assert_allclose(g[2:], 0.0, atol=1.0e-15)


class TestOptimizer:
    def test_recovers_the_separable_exact_solution(self) -> None:
        problem = _diagonal_problem(prescription=2.0)
        w = optimize_weights(problem)
        np.testing.assert_allclose(w, 2.0, rtol=1.0e-6)

    def test_weights_respect_the_nonnegativity_bound(self) -> None:
        """A beamlet that only harms the objective must pin to zero, not go negative."""
        # Beamlet 0 hits only the OAR; beamlet 1 hits only the PTV.
        dij = csc_array(np.array([[0.0, 1.0], [1.0, 0.0]]))
        ptv = np.array([True, False])
        oar = np.array([False, True])
        problem = ToyPlanProblem(
            dij=dij, ptv=ptv, oar=oar, prescription=1.0, oar_max=0.0, oar_penalty=10.0
        )
        w = optimize_weights(problem)
        assert w[0] == pytest.approx(0.0, abs=1.0e-8)
        assert w[1] == pytest.approx(1.0, rel=1.0e-5)

    def test_is_deterministic(self) -> None:
        rng = np.random.default_rng(20260711)
        dij = csc_array(rng.random((30, 5)))
        ptv = np.zeros(30, dtype=bool)
        ptv[:10] = True
        problem = ToyPlanProblem(dij=dij, ptv=ptv, oar=np.zeros(30, dtype=bool))
        w1 = optimize_weights(problem)
        w2 = optimize_weights(problem)
        np.testing.assert_array_equal(w1, w2)


class TestDoseAtVolume:
    def test_matches_the_percentile_definition(self) -> None:
        """D(v%): the dose received by at least v percent of the ROI volume."""
        dose = np.linspace(1.0, 100.0, 100)
        assert dose_at_volume(dose, 98.0) == pytest.approx(np.percentile(dose, 2.0))
        assert dose_at_volume(dose, 50.0) == pytest.approx(np.percentile(dose, 50.0))
        assert dose_at_volume(dose, 2.0) == pytest.approx(np.percentile(dose, 98.0))

    def test_rejects_empty_roi(self) -> None:
        with pytest.raises(ValueError):
            dose_at_volume(np.zeros(0), 50.0)
