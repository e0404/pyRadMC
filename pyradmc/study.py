"""Study instrumentation: a toy fluence optimizer and DVH endpoints.

This module exists to measure one thing: how statistical noise in the Dij
biases the plans an optimizer produces from it (docs/decisions.md). It is
**not** a treatment planning system and must not grow into one — no clinical
objective vocabulary, no beam models, no constraints beyond what the study
needs. The engine's product remains the Dij; anything richer belongs in a
planning system on the other side of an adapter (AGENTS.md 6).

The optimization problem is deliberately minimal and fully deterministic:

    minimize_{w >= 0}   mean_{v in PTV} (d_v(w) - Rx)^2
                      + lambda * mean_{v in OAR} max(0, d_v(w) - c * Rx)^2

with ``d(w) = Dij @ w``. A one-sided quadratic OAR penalty is the simplest
objective that gives the optimizer an exploitable trade-off, which is all the
noise/bias study requires. Solved with L-BFGS-B under nonnegativity bounds;
every stochastic element of the study lives in the Dij realizations passed
in, never in this harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csc_array, sparray

__all__ = [
    "ToyPlanProblem",
    "dose_at_volume",
    "objective_and_gradient",
    "optimize_weights",
]


@dataclass(frozen=True)
class ToyPlanProblem:
    """One instance of the study's fluence optimization problem.

    Parameters
    ----------
    dij
        Dose influence matrix, ``(n_voxels, n_beamlets)``, dose per unit
        beamlet weight (any sparse scipy array works; CSC is what
        :meth:`pyradmc.scoring.dij.DijResult.dose_csc` produces).
    ptv
        Flat boolean mask of target voxels; must select at least one voxel.
    oar
        Flat boolean mask of organ-at-risk voxels; may be empty, in which
        case the penalty term vanishes.
    prescription
        Target dose ``Rx`` in the PTV, in the Dij's dose unit.
    oar_max
        OAR dose limit as a fraction ``c`` of the prescription.
    oar_penalty
        Penalty weight ``lambda`` of the one-sided OAR term.
    """

    dij: sparray
    ptv: np.ndarray
    oar: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    prescription: float = 1.0
    oar_max: float = 0.3
    oar_penalty: float = 1.0

    def __post_init__(self) -> None:
        """Validate mask shapes against the matrix."""
        n_voxels = self.dij.shape[0]
        if self.ptv.shape != (n_voxels,) or not bool(self.ptv.any()):
            raise ValueError(f"ptv must be a nonempty flat mask of {n_voxels} voxels")
        if self.oar.size and self.oar.shape != (n_voxels,):
            raise ValueError(f"oar mask shape {self.oar.shape} != ({n_voxels},)")

    @property
    def n_beamlets(self) -> int:
        """Number of optimization variables (columns of the Dij)."""
        return int(self.dij.shape[1])

    @cached_property
    def dij_ptv(self) -> csc_array:
        """The Dij restricted to PTV voxels, cached: the optimizer's hot path."""
        return csc_array(self.dij)[self.ptv, :]

    @cached_property
    def dij_oar(self) -> csc_array:
        """The Dij restricted to OAR voxels, cached alongside :attr:`dij_ptv`."""
        return csc_array(self.dij)[self.oar, :]


def objective_and_gradient(
    problem: ToyPlanProblem, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Objective value and its analytic gradient at ``weights``.

    The gradient of the mean-squared PTV term is ``2/n * D_ptv.T @ (d - Rx)``;
    the one-sided OAR term contributes only where its residual is positive.
    Pinned against central finite differences in the unit tier.
    """
    w = np.asarray(weights, dtype=np.float64)
    r_ptv = problem.dij_ptv @ w - problem.prescription
    n_ptv = r_ptv.size
    f = float(r_ptv @ r_ptv) / n_ptv
    g = 2.0 / n_ptv * (problem.dij_ptv.T @ r_ptv)

    if problem.oar.size and bool(problem.oar.any()):
        d_oar = problem.dij_oar @ w
        r_oar = np.maximum(0.0, d_oar - problem.oar_max * problem.prescription)
        n_oar = r_oar.size
        f += problem.oar_penalty * float(r_oar @ r_oar) / n_oar
        g += 2.0 * problem.oar_penalty / n_oar * (problem.dij_oar.T @ r_oar)
    return f, np.asarray(g, dtype=np.float64)


def optimize_weights(problem: ToyPlanProblem, w0: np.ndarray | None = None) -> np.ndarray:
    """Solve the toy problem with L-BFGS-B under ``w >= 0``.

    Deterministic: fixed start (uniform weights scaled so the mean PTV dose
    matches the prescription, unless ``w0`` is given), fixed tolerances, no
    randomized components — repeat calls return identical weights, so any
    spread in study results is attributable to the Dij realizations alone.
    """
    if w0 is None:
        uniform = np.ones(problem.n_beamlets)
        mean_dose = float(np.mean(problem.dij_ptv @ uniform))
        scale = problem.prescription / mean_dose if mean_dose > 0.0 else 1.0
        w0 = uniform * scale
    result = minimize(
        lambda w: objective_and_gradient(problem, w),
        w0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * problem.n_beamlets,
        options={"maxiter": 500, "ftol": 1.0e-12, "gtol": 1.0e-10},
    )
    return np.asarray(result.x, dtype=np.float64)


def dose_at_volume(dose_in_roi: np.ndarray, volume_percent: float) -> float:
    """DVH endpoint D(v%): the dose received by at least ``v`` percent of the ROI.

    The (100 - v)-th percentile of the ROI's voxel doses — D98 is a coverage
    (near-minimum) endpoint, D2 a hot-spot (near-maximum) endpoint. Matches
    the convention of the truncation certification.
    """
    if dose_in_roi.size == 0:
        raise ValueError("empty ROI has no DVH")
    return float(np.percentile(dose_in_roi, 100.0 - volume_percent))
