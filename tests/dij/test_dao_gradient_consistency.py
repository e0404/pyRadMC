"""The DAO gradient is the derivative of the DAO forward dose, end to end.

A Monte Carlo direct aperture optimization computes its forward dose and its leaf
gradient through two code paths that share no code for the transmission
derivative:

- **forward** — :class:`~pyradmc.geometry.segments.SegmentedFanSource` columns from
  ``run_dij``, each primary weighted in-kernel by its segment's stack transmission;
- **gradient** — :class:`~pyradmc.geometry.aperture_gradient.EdgeGradientBeamletSource`
  of :class:`~pyradmc.geometry.aperture_gradient.FanEdgeDerivativeSource` edges,
  generated in-kernel and contracted with the adjoint ``a = dL/dD`` by
  ``run_beamlet_contraction``.

The oracle is the common-random-number central difference of the forward model in
the contracted quantity, ``g_FD = a . (D(x + h) - D(x - h)) / 2h`` (one-sided,
second order, at a closed pair, whose tips may not cross): both displaced tips are
columns of one correlated ``run_dij``, so the fan rays and every transport draw
coincide and only the transmission weights differ. Its sigma comes from
independent replicate seeds (a batch sigma per column cannot see the covariance of
the two columns). Electrons are transported: the identity is transport-independent
by construction, and a transport defect that only electrons expose is exactly what
this end-to-end check is for (see below). ``h = 0.005`` cm at the mid-plane; the
O(h^2) bias is measured negligible there (docs/decisions.md, "DAO gradient
consistency"). The per-edge statistic is Welch's t with its Welch-Satterthwaite
degrees of freedom, mapped to a normal score, so the chi-squared over a handful of
edges is calibrated at the replicate and batch counts used here.

Both sides of every pair are pinned. The ``neg``-side edges (all of whose weights
are negative) were the ones that exposed the transport's negative-weight defect,
fixed 2026-10-23 (``tests/integration/test_signed_weights.py``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy import stats

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.aperture_gradient import (
    EdgeGradientBeamletSource,
    EdgeParameter,
    FanEdgeDerivativeSource,
)
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

FOCAL = (5.0, 2.0, -100.0)
H = 0.005  # cm at the MLC mid-plane
ALPHA = 0.01
# Per segment: the jaw's (neg, pos) edges at its mid-plane, then (neg, pos) tips per
# leaf pair at the MLC mid-plane. A: an open pair and a nearly closed one (0.8 mm
# gap); B: a closed pair and an open one. The jaw's + edge (at the angle of 0.95 cm on
# the MLC mid-plane) cuts every + tip's penumbra, so the rest-of-stack transmission
# T_rest of a tip edge is far from 1 there, and the MLC's is for the jaw edge.
JAW_POS = 0.95 * 33.9 / 53.0
SEGMENTS = {
    "A": ((-5.0, JAW_POS), ((0.35, 0.75), (0.50, 0.58))),
    "B": ((-5.0, JAW_POS), ((0.53, 0.53), (0.30, 0.80))),
}
# (segment, device, pair, side, stencil); device 0 is the jaw, 1 the MLC; the
# stencil is "central", or "one-sided" in the opening direction (a closed pair).
EDGES = (
    ("A", 1, 0, "pos", "central"),
    ("A", 1, 1, "pos", "central"),
    ("B", 1, 0, "pos", "one-sided"),
    ("B", 1, 1, "pos", "central"),
    ("A", 0, None, "pos", "central"),
    ("A", 1, 0, "neg", "central"),
    ("A", 1, 1, "neg", "central"),
    ("B", 1, 0, "neg", "one-sided"),
    ("B", 1, 1, "neg", "central"),
)
# Per device: (edge histories, edge batches, FD replicates, FD histories per column).
BUDGET = {"cpu": (8_000, 16, 8, 10_000), "cuda:0": (64_000, 16, 16, 200_000)}
DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 10, 10), spacing=(0.5, 0.5, 1.0))


# One cross-section source for every edge: an in-kernel edge set requires it.
XS = AnalyticCrossSections(geometry_densities=_grid().max_density_by_material())


def _fan() -> SpectralBeamSource:
    """A 2 MeV +- 10 % fan through 2 x 4 cm of surface: both strips, both blocked sides."""
    return SpectralBeamSource(
        Spectrum(edges=[1.8, 2.2], weights=[1.0]),
        focal_point=FOCAL,
        center=(6.0, 2.0, 0.0),
        width_u=2.0,
        width_v=4.0,
    )


def _stack(segment: str) -> BeamLimitingStack:
    """A focused jaw over an MLC; dense water (leaf T_min 0.23 at 2 MeV), 8 cm tips."""
    jaw, tips = SEGMENTS[segment]
    return BeamLimitingStack(
        frame=BeamFrame(origin=FOCAL),
        devices=(
            JawPair(
                axis="u",
                z_top=30.0,
                z_bottom=37.8,
                edge_neg=jaw[0],
                edge_pos=jaw[1],
                material=WATER,
                density=5.0,
                focused=True,
            ),
            MLC(
                z_top=50.0,
                z_bottom=56.0,
                leaf_edges_v=(-1.06, 0.0, 1.06),
                tips_neg=tuple(float(t[0]) for t in tips),
                tips_pos=tuple(float(t[1]) for t in tips),
                tip_radius=8.0,
                material=WATER,
                density=5.0,
                focused_sides=True,
            ),
        ),
    )


def _adjoints() -> dict[str, np.ndarray]:
    """``ones``: integral dose, the sign obvious; ``signed``: a mixed-sign adjoint."""
    x, y, z = np.meshgrid(*(np.arange(n) for n in _grid().shape), indexing="ij")
    return {"ones": np.ones(_grid().shape), "signed": np.sin(0.7 * x) + 0.3 * y - 0.1 * z}


def _stencil(kind: str, side: str) -> tuple[tuple[float, ...], np.ndarray]:
    """Tip offsets and weights of the difference quotient (an O(h^2) stencil)."""
    if kind == "central":
        return (-H, H), np.array([-1.0, 1.0]) / (2.0 * H)
    opening = 1.0 if side == "pos" else -1.0
    return (0.0, opening * H, 2.0 * opening * H), opening * np.array([-3.0, 4.0, -1.0]) / (2 * H)


def _finite_difference(engine: Any, edge: int, n_replicates: int, n_fd: int) -> dict[str, Any]:
    """Replicate CRN difference quotients of the forward model, per adjoint."""
    segment, device, pair, side, kind = EDGES[edge]
    jaw, tips = SEGMENTS[segment]
    offsets, weights = _stencil(kind, side)
    n = len(offsets)
    settings = {
        "jaw_neg": np.full((n, 1), jaw[0]),
        "jaw_pos": np.full((n, 1), jaw[1]),
        "tips_neg": np.tile([t[0] for t in tips], (n, 1)),
        "tips_pos": np.tile([t[1] for t in tips], (n, 1)),
    }
    name = ("jaw_" if device == 0 else "tips_") + side
    settings[name][:, 0 if pair is None else pair] += np.array(offsets)
    source = SegmentedFanSource(_fan(), _stack(segment), XS, **settings)
    out: dict[str, list[float]] = {name: [] for name in _adjoints()}
    for r in range(n_replicates):
        dij = engine.run_dij(
            source,
            n_histories_per_beamlet=n_fd,
            n_batches=1,
            seed=SEED + 1000 * (edge + 1) + r,
            truncation=0.0,
            correlated=True,
        )
        columns = np.array([dij.column_dense(c).reshape(-1) for c in range(n)])
        for adjoint, a in _adjoints().items():
            out[adjoint].append(float(weights @ (columns @ a.reshape(-1))))
    return {name: np.array(values) for name, values in out.items()}


@pytest.fixture(scope="module", params=DEVICES)
def results(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Edge contractions and replicate finite differences for every edge and adjoint."""
    device = request.param
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    n_edge, n_batches, n_replicates, n_fd = BUDGET[device]
    engine = WarpEngine(grid=_grid(), cross_sections=XS, device=device)
    sources = [
        FanEdgeDerivativeSource(_fan(), _stack(s), XS, EdgeParameter(device, side, pair))
        for s, device, pair, side, _ in EDGES
    ]
    edge = {
        name: engine.run_beamlet_contraction(
            EdgeGradientBeamletSource(sources),
            a,
            n_histories_per_beamlet=n_edge,
            n_batches=n_batches,
            seed=SEED,
            correlated=False,  # independent columns: the edges enter one chi-squared
        )
        for name, a in _adjoints().items()
    }
    fd = [_finite_difference(engine, j, n_replicates, n_fd) for j in range(len(EDGES))]
    return {"edge": edge, "fd": fd, "n_batches": n_batches}


def _p_value(results: dict[str, Any], adjoint: str, picks: list[int], values: np.ndarray) -> float:
    """Calibrated chi-squared p of edge values ``values`` against the FD of edges ``picks``."""
    sigma = results["edge"][adjoint].sigma
    k = results["n_batches"]
    scores = []
    for j, value in zip(picks, values, strict=True):
        replicates = results["fd"][j][adjoint]
        r = replicates.size
        v_fd = replicates.var(ddof=1) / r
        v_edge = sigma[j] ** 2
        t = (value - replicates.mean()) / math.sqrt(v_fd + v_edge)
        nu = (v_fd + v_edge) ** 2 / (v_fd**2 / (r - 1) + v_edge**2 / (k - 1))
        # Welch t -> normal score through the CDF (two tails kept apart).
        scores.append(stats.norm.isf(stats.t.sf(t, nu)))
    chi2 = float(np.sum(np.square(scores)))
    return float(stats.chi2.sf(chi2, len(scores)))


def _side(side: str) -> list[int]:
    return [j for j, e in enumerate(EDGES) if e[3] == side]


@pytest.mark.parametrize("adjoint", ["ones", "signed"])
def test_positive_side_gradients_match_the_forward_model(results, adjoint: str) -> None:
    picks = _side("pos")
    values = results["edge"][adjoint].values[picks]
    p = _p_value(results, adjoint, picks, values)
    assert p > ALPHA, (adjoint, p)


@pytest.mark.parametrize("adjoint", ["ones", "signed"])
def test_negative_side_gradients_match_the_forward_model(results, adjoint: str) -> None:
    picks = _side("neg")
    values = results["edge"][adjoint].values[picks]
    p = _p_value(results, adjoint, picks, values)
    assert p > ALPHA, (adjoint, p)


def test_integral_dose_gradients_have_the_obvious_sign(results) -> None:
    """Opening a leaf raises the integral dose: + for a pos tip, - for a neg tip."""
    for j, (_, _, _, side, _) in enumerate(EDGES):
        sign = 1.0 if side == "pos" else -1.0
        assert sign * results["fd"][j]["ones"].mean() > 0.0
        assert sign * results["edge"]["ones"].values[j] > 0.0


@pytest.mark.parametrize(
    "mutant",
    ["sign-flip", "wrong-pair", "wrong-segment"],
)
def test_the_oracle_rejects_a_wrong_gradient(results, mutant: str) -> None:
    """Power, at the same seeds: a gradient from the wrong place or of the wrong sign fails.

    ``wrong-pair`` reports pair 1's gradient for pair 0 (and back) within a segment;
    ``wrong-segment`` reports segment B's for segment A's (and back), same pair and
    side. On the signed adjoint, whose pattern varies across the strips.
    """
    picks = [j for j in _side("pos") if EDGES[j][1] == 1]  # the MLC tip edges
    lookup = {(EDGES[j][0], EDGES[j][2]): j for j in picks}
    values = results["edge"]["signed"].values
    if mutant == "sign-flip":
        mutated = -values[picks]
    elif mutant == "wrong-pair":
        mutated = np.array([values[lookup[(EDGES[j][0], 1 - EDGES[j][2])]] for j in picks])
    else:
        other = {"A": "B", "B": "A"}
        mutated = np.array([values[lookup[(other[EDGES[j][0]], EDGES[j][2])]] for j in picks])
    assert _p_value(results, "signed", picks, mutated) < 1.0e-6
