"""The segmented edge set on the Warp engine: the per-edge route's bits, without its objects.

:class:`SegmentedEdgeSource` builds the in-kernel edge set's columns vectorized; the
per-edge route builds them from one :class:`FanEdgeDerivativeSource` per edge. For the
same edges, seed and device the two must be indistinguishable: pinned here on the
uploaded device arrays (every ``EdgeSet`` column, the concatenated stack, the mu rows,
the device-built proposal tables) and on ``run_beamlet_contraction`` itself
(``np.array_equal`` of values, sigma and energy books), correlated and independent,
over the unit tier's fixture — focused and straight jaws, open, nearly closed and
closed leaf pairs, tips on the field limit and tips whose support start needs the
bisection fallback. Within one device only: CPU and GPU never agree bit for bit
(AGENTS.md 2.3). Then that the route is taken, that a subclass keeps the host route,
and that the reference engine agrees statistically.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.aperture_gradient import (
    EdgeGradientBeamletSource,
    EdgeParameter,
    FanEdgeDerivativeSource,
)
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segment_edges import SegmentedEdgeSource
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.unit.test_segmented_edge_source import (
    XS,
    _arc_parameters,
    _arc_segments,
    _expected_parameters,
    _per_edge,
    _segments,
)

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

from pyradmc.backends.warp import aperture  # noqa: E402  (needs warp)

pytestmark = pytest.mark.warp

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


def _grid() -> VoxelGrid:
    """Water under the fixture's field (engine x 3..11, y -5..1 at the surface)."""
    return VoxelGrid.uniform_water(
        shape=(16, 3, 8), spacing=(0.5, 2.0, 1.0), origin=(3.0, -5.0, 0.0)
    )


def _weights() -> np.ndarray:
    x, y, z = np.meshgrid(*(np.arange(n) for n in _grid().shape), indexing="ij")
    return np.exp(-(((x - 7.0) / 4.0) ** 2)) * (1.0 + 0.2 * y) - 0.05 * z


def _engine(device: str):
    from pyradmc.backends.warp.engine import WarpEngine

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return WarpEngine(grid=_grid(), cross_sections=XS, device=device)


def _pair() -> tuple[SegmentedEdgeSource, EdgeGradientBeamletSource]:
    """The same 40 edges both ways: vectorized, and one per-edge source each."""
    segments = _segments()
    return (
        SegmentedEdgeSource(segments),
        EdgeGradientBeamletSource(_per_edge(segments, _expected_parameters(4))),
    )


def _device_arrays(edge_set) -> dict[str, np.ndarray]:
    es, (sp_edges, sp_cdf) = edge_set
    out = {"sp_edges": sp_edges.numpy(), "sp_cdf": sp_cdf.numpy()}
    for name in dir(es):
        value = getattr(es, name)
        if isinstance(value, wp.array):
            out[name] = value.numpy()
        elif name == "stack":
            for sub in dir(value):
                if isinstance(getattr(value, sub), wp.array):
                    out[f"stack.{sub}"] = getattr(value, sub).numpy()
        elif isinstance(value, int | float) and not name.startswith("_"):
            out[name] = np.array(value)
    # What each edge reads of the s grids (rows may be shared between edges).
    s_row = out.pop("s_row")
    out["s_grids"] = out.pop("s").reshape(-1, int(out["n_table"]))[s_row[s_row >= 0]]
    out["smooth"] = s_row >= 0
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_uploaded_edge_set_equals_the_per_edge_upload(device: str) -> None:
    from pyradmc.backends.warp import aperture

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    vectorized, per_edge = _pair()
    got = _device_arrays(aperture.upload_edge_set(vectorized, device))
    want = _device_arrays(aperture.upload_edge_set(per_edge, device))
    assert got.keys() == want.keys()
    assert {"cdf", "cell", "log_mu", "stack.tips_pos", "q0"} <= got.keys()
    for name in want:
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)


@pytest.mark.parametrize("correlated", [True, False], ids=["correlated", "independent"])
@pytest.mark.parametrize("device", DEVICES)
def test_contraction_is_bit_identical_to_the_per_edge_route(device: str, correlated: bool) -> None:
    engine = _engine(device)
    vectorized, per_edge = _pair()
    kwargs = dict(
        n_histories_per_beamlet=400 if device == "cpu" else 4_000,
        n_batches=2,
        seed=SEED,
        correlated=correlated,
    )
    got = engine.run_beamlet_contraction(vectorized, _weights(), **kwargs)
    want = engine.run_beamlet_contraction(per_edge, _weights(), **kwargs)
    assert np.count_nonzero(want.values) > 20  # most edges see the phantom
    np.testing.assert_array_equal(got.values, want.values)
    np.testing.assert_array_equal(got.sigma, want.sigma)
    assert got.energy_emitted == want.energy_emitted
    assert got.energy_deposited == want.energy_deposited
    assert got.energy_escaped == want.energy_escaped


# -- an arc: each segment's edges on its own fan and frame ---------------------------------


def _arc_grid() -> VoxelGrid:
    """Water around the arc's isocentre (8, 8, 8)."""
    return VoxelGrid.uniform_water(shape=(8, 4, 8), spacing=(2.0, 2.0, 2.0), origin=(0.0, 4.0, 0.0))


def _arc_weights() -> np.ndarray:
    x, y, z = np.meshgrid(*(np.arange(n) for n in _arc_grid().shape), indexing="ij")
    return np.exp(-(((x - 3.5) / 2.5) ** 2) - ((z - 3.0) / 3.0) ** 2) * (1.0 + 0.3 * y) - 0.05


def _arc_pair() -> tuple[SegmentedEdgeSource, EdgeGradientBeamletSource]:
    segments = _arc_segments()
    return (
        SegmentedEdgeSource(segments),
        EdgeGradientBeamletSource(_per_edge(segments, _arc_parameters())),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_arc_uploaded_edge_set_equals_the_per_edge_upload(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    vectorized, per_edge = _arc_pair()
    got = _device_arrays(aperture.upload_edge_set(vectorized, device))
    want = _device_arrays(aperture.upload_edge_set(per_edge, device))
    assert got.keys() == want.keys()
    for name in want:
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)


@pytest.mark.parametrize("correlated", [True, False], ids=["correlated", "independent"])
@pytest.mark.parametrize("device", DEVICES)
def test_arc_contraction_is_bit_identical_to_the_per_edge_route(
    device: str, correlated: bool
) -> None:
    from pyradmc.backends.warp.engine import WarpEngine

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    engine = WarpEngine(grid=_arc_grid(), cross_sections=XS, device=device)
    vectorized, per_edge = _arc_pair()
    kwargs = dict(
        n_histories_per_beamlet=400 if device == "cpu" else 4_000,
        n_batches=2,
        seed=SEED,
        correlated=correlated,
    )
    got = engine.run_beamlet_contraction(vectorized, _arc_weights(), **kwargs)
    want = engine.run_beamlet_contraction(per_edge, _arc_weights(), **kwargs)
    assert np.count_nonzero(want.values) > 15
    np.testing.assert_array_equal(got.values, want.values)
    np.testing.assert_array_equal(got.sigma, want.sigma)
    assert got.energy_emitted == want.energy_emitted
    assert got.energy_deposited == want.energy_deposited


def test_arc_in_kernel_route_agrees_with_the_host_sampler() -> None:
    """Every arc edge's contraction: the in-kernel rays against the per-edge host sampler.

    The host side is the Warp host route of a subclass, i.e. each edge's
    :class:`FanEdgeDerivativeSource` sampling the rays the reference engine emits,
    then the same transport; so this isolates the in-kernel ray generation in each
    segment's frame, at statistics the reference engine cannot reach here (at 800
    histories per edge its heavy-tailed batch means made the batched chi-squared
    fail an unmutated device run on 2 of 3 seeds). Independent sampling, KERMA mode,
    8 batches; edges that see none of the phantom (closed pairs, a jaw outside the
    field) are exactly zero on both sides and left out. Power, measured: a 0.25 deg
    tilt of one segment's frame, a 2 mm focal-spot shift, segment 0's frame in every
    segment and two segments' frames swapped all fail (decisions.md, "Arc edge sets").
    """
    from pyradmc.backends.warp.engine import WarpEngine

    engine = WarpEngine(grid=_arc_grid(), cross_sections=XS, device="cpu")
    kwargs = dict(n_batches=8, transport_electrons=False, correlated=False)
    device = engine.run_beamlet_contraction(
        SegmentedEdgeSource(_arc_segments()), _arc_weights(), n_histories_per_beamlet=8_000,
        seed=SEED, **kwargs,
    )  # fmt: skip
    host = engine.run_beamlet_contraction(
        _HostEdges(_arc_segments()), _arc_weights(), n_histories_per_beamlet=8_000,
        seed=SEED + 1, **kwargs,
    )  # fmt: skip
    seen = (device.sigma > 0.0) | (host.sigma > 0.0)
    np.testing.assert_array_equal(device.values[~seen], 0.0)
    np.testing.assert_array_equal(host.values[~seen], 0.0)
    assert np.all(np.abs(device.values[seen]) > 3.0 * device.sigma[seen])  # every edge resolved
    assert seen.sum() >= 15, seen.sum()
    assert_chi2_consistent_batched(
        host.values, host.sigma, 8, device.values, device.sigma, 8, mask=seen
    )


def test_the_in_kernel_route_is_taken(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("no host sampling and no per-edge object on the in-kernel route")

    monkeypatch.setattr(SegmentedEdgeSource, "sample_beamlet_batch", refuse)
    monkeypatch.setattr(SegmentedEdgeSource, "edge", refuse)
    monkeypatch.setattr(FanEdgeDerivativeSource, "__init__", refuse)
    result = _engine("cpu").run_beamlet_contraction(
        SegmentedEdgeSource(_segments()),
        _weights(),
        n_histories_per_beamlet=200,
        n_batches=2,
        seed=SEED,
        transport_electrons=False,
    )
    assert np.all(np.isfinite(result.values)) and np.any(result.values != 0.0)


class _HostEdges(SegmentedEdgeSource):
    """A subclass: the engine cannot know it samples as its parent does."""


class _HostList(EdgeGradientBeamletSource):
    """The per-edge list's host route, for comparison."""


class _HostSegments(SegmentedFanSource):
    """A segmented-fan subclass under an exact edge source: host route too."""


def test_subclasses_take_the_host_route_and_match_the_per_edge_host_route() -> None:
    """The host route delegates to the per-edge sources: equal to their host route, bitwise."""
    engine = _engine("cpu")
    segments = _segments()
    kwargs = dict(n_histories_per_beamlet=60, n_batches=2, seed=SEED, transport_electrons=False)
    want = engine.run_beamlet_contraction(
        _HostList(_per_edge(segments, _expected_parameters(4))), _weights(), **kwargs
    )
    got = engine.run_beamlet_contraction(_HostEdges(segments), _weights(), **kwargs)
    np.testing.assert_array_equal(got.values, want.values)
    np.testing.assert_array_equal(got.sigma, want.sigma)
    inner = _HostSegments(
        segments.fan,
        segments.template,
        XS,
        jaw_neg=segments.jaw_neg[:, :2],
        jaw_pos=segments.jaw_pos[:, :2],
        tips_neg=segments.tips_neg,
        tips_pos=segments.tips_pos,
    )
    got = engine.run_beamlet_contraction(SegmentedEdgeSource(inner), _weights(), **kwargs)
    np.testing.assert_array_equal(got.values, want.values)


# -- statistics, in the regime the in-kernel near-tangent chord is built for ---------------

FOCAL = (5.0, 2.0, -100.0)
SPECTRUM = Spectrum(edges=[1.8, 2.2], weights=[1.0])


def _demo_segments() -> SegmentedFanSource:
    """Two segments of a focused jaw over a 6 cm, R = 8 cm two-pair MLC (dense water).

    The DAO example's regime: thick slab, tips a few millimetres off axis, every
    support start in closed form. Every edge sits in the field and moves the dose;
    segment 1 holds a closed pair.
    """
    fan = SpectralBeamSource(
        SPECTRUM, focal_point=FOCAL, center=(5.5, 2.0, 0.0), width_u=1.0, width_v=4.0
    )
    template = BeamLimitingStack(
        frame=BeamFrame(origin=FOCAL),
        devices=(
            JawPair(
                axis="u", z_top=30.0, z_bottom=37.8, edge_neg=-5.0, edge_pos=5.0,
                material=WATER, density=5.0, focused=True,
            ),
            MLC(
                z_top=50.0, z_bottom=56.0, leaf_edges_v=(-1.06, 0.0, 1.06),
                tips_neg=(-5.0, -5.0), tips_pos=(5.0, 5.0), tip_radius=8.0,
                material=WATER, density=5.0, focused_sides=True,
            ),
        ),
    )  # fmt: skip
    return SegmentedFanSource(
        fan,
        template,
        XS_DEMO,
        jaw_neg=[[0.03], [0.01]],
        jaw_pos=[[0.30], [0.18]],
        tips_neg=[[0.05, 0.10], [0.15, 0.25]],
        tips_pos=[[0.27, 0.35], [0.40, 0.25]],
    )


def _demo_grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(20, 2, 8), spacing=(0.5, 2.0, 1.0))


XS_DEMO = AnalyticCrossSections(geometry_densities=_demo_grid().max_density_by_material())


def _demo_weights() -> np.ndarray:
    x, y, z = np.meshgrid(*(np.arange(n) for n in _demo_grid().shape), indexing="ij")
    return np.exp(-(((x - 11.0) / 3.0) ** 2)) * (1.0 + 0.2 * y) - 0.05 * z


STAT_KWARGS = dict(n_batches=8, transport_electrons=False, correlated=False)


def test_reference_engine_agrees_statistically() -> None:
    """The reference engine (the per-edge oracle through ``emit``) against the device.

    Independent sampling (correlated columns share their fluctuations, AGENTS.md 8),
    KERMA mode, 8 batches. Each compared value is one edge's whole contraction, a sum
    over every voxel of hundreds of histories' deposits per batch, so the batched
    chi-squared is at its null here, not bounded below it as on sparse voxels (the
    segmented-fan entry of docs/decisions.md). Calibration and power were measured
    in a probe (decisions.md, "Vectorized edge sets").
    """
    from pyradmc.backends.warp.engine import WarpEngine

    source = SegmentedEdgeSource(_demo_segments())
    assert source.n_beamlets == 12
    engine = WarpEngine(grid=_demo_grid(), cross_sections=XS_DEMO, device="cpu")
    device = engine.run_beamlet_contraction(
        source, _demo_weights(), n_histories_per_beamlet=16_000, seed=SEED, **STAT_KWARGS
    )
    reference = ReferenceEngine(grid=_demo_grid(), cross_sections=XS_DEMO, rng=HostRNG())
    ref = reference.run_beamlet_contraction(
        SegmentedEdgeSource(_demo_segments()),
        _demo_weights(),
        n_histories_per_beamlet=1_000,
        seed=SEED,
        **STAT_KWARGS,
    )
    assert np.all(np.abs(device.values) > 3.0 * device.sigma)  # every edge resolved
    assert_chi2_consistent_batched(ref.values, ref.sigma, 8, device.values, device.sigma, 8)


@pytest.mark.parametrize("device", DEVICES)
def test_in_kernel_route_is_unbiased_where_the_support_start_is_bisected(device: str) -> None:
    """Device against host route on two fixture edges: a bisected start, a thin-slab tangent.

    The regression test of the near-tangent fix (decisions.md, "Vectorized edge
    sets"): before it, the closed-form near-tangent chord was taken on both edges
    and they came out at z = +14 and -7.4 at these statistics (cpu).
    """
    engine = _engine(device)
    segments = _segments()
    picks = [
        (2, EdgeParameter(2, "pos", 1)),  # x = 3.3: support start by bisection
        (1, EdgeParameter(2, "neg", 1)),  # x = 2.5: tangent point 2.5 mm inside a face
    ]
    kwargs = dict(n_histories_per_beamlet=16_000, n_batches=8, transport_electrons=False)
    kwargs["correlated"] = False
    device = engine.run_beamlet_contraction(
        EdgeGradientBeamletSource(_per_edge(segments, picks)), _weights(), seed=SEED, **kwargs
    )
    host = engine.run_beamlet_contraction(
        _HostList(_per_edge(segments, picks)), _weights(), seed=SEED + 1, **kwargs
    )
    assert_chi2_consistent_batched(device.values, device.sigma, 8, host.values, host.sigma, 8)


@pytest.mark.gpu
def test_proposal_routes_give_the_same_tables(monkeypatch) -> None:
    """Node densities split out (small sets on a GPU) or evaluated inline: same bits.

    The split is a scheduling choice by edge count (``_SPLIT_EDGES_PER_SM``); the
    proposal tables, and so every ray, must not depend on it.
    """
    from pyradmc.backends.warp import aperture

    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    source = SegmentedEdgeSource(_segments())
    tables = {}
    for per_sm in (0, 1_000_000):  # never split, always split
        monkeypatch.setattr(aperture, "_SPLIT_EDGES_PER_SM", per_sm)
        es, _ = aperture.upload_edge_set(source, "cuda:0")
        tables[per_sm] = (es.cdf.numpy(), es.cell.numpy())
    np.testing.assert_array_equal(tables[0][0], tables[1_000_000][0])
    np.testing.assert_array_equal(tables[0][1], tables[1_000_000][1])
    assert np.all(np.diff(tables[0][0].reshape(-1, source.device_columns().n_table - 1)) >= 0.0)


# -- the near-tangent chord, ray by ray -------------------------------------------------

# The float32 chord tolerances of tests/physics/test_warp_compiled_stack.py (a chord
# enters the weight only through exp(-mu l); its edge rate enters linearly).
CHORD_RTOL = 2.0e-3
CHORD_ATOL = 2.0e-3  # cm
# The twin's rate on a ray through a tip's circle away from the tangent band: its
# float32 ray-circle quadratic has a discriminant D = b^2 - 4ac = 4 h^2 (h the bare-disc
# half chord, b ~ 2 t_c with t_c the distance to the tip centre), so h, n.d = h / R and
# the Hadamard rate n_u / (n.d) carry a relative error k eps t_c^2 / (2 h^2), with k the
# discriminant's absolute error in units of eps b^2. To first order in the worst case:
# 2 eps |b|^2 from b's own rounding (three products, two sums) squared, eps b^2 from the
# square, about 2 eps b^2 from 4ac (|c|^2 ~ b^2 / 4 rounded over its products and sums,
# then the R^2 subtraction) and eps b^2 from the final subtraction: k ~ 6. Fixed at 8.
# Measured up to 2.7 (cpu and cuda); the pre-fix error was >= 840x this bound.
TWIN_ROUNDINGS = 8.0


@wp.kernel
def _block_terms_probe(
    es: aperture.EdgeSet, s_in: wp.array(dtype=float), out: wp.array2d(dtype=float)
):
    """Edge 0's ray at proposal node ``s`` built as the generator builds it: q, chord, rate."""
    i = wp.tid()
    s = s_in[i]
    qq = es.q0[0] + es.sign[0] * s * s
    zm = es.z_mid[0]
    inv = 1.0 / wp.sqrt(qq * qq + es.r_c[0] * es.r_c[0] + zm * zm)
    l_block, rate, _l_rest = aperture._block_terms(
        es, 0, qq, s, qq * inv, es.r_c[0] * inv, zm * inv
    )
    out[i, 0] = qq
    out[i, 1] = l_block
    out[i, 2] = rate


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "pick",
    [
        (2, EdgeParameter(2, "pos", 1)),  # x = 3.3: bisected, start 1.9e-3 past tangent
        (3, EdgeParameter(2, "neg", 2)),  # x = 2.6: bisected, tangent just outside a face
        (1, EdgeParameter(2, "pos", 2)),  # x = 4.9: bisected far from tangent: the twin
        (1, EdgeParameter(2, "neg", 1)),  # x = 2.5: tangent point 2.5 mm inside a face
        (0, EdgeParameter(2, "neg", 0)),  # x = 1.5: tangent chord well inside the slab
    ],
    ids=["bisected", "bisected-at-face", "bisected-far", "near-face", "clear"],
)
def test_block_terms_follow_the_host_chord_ray_by_ray(device: str, pick) -> None:
    """Chord and edge rate of each near-tangent ray against the float64 host geometry.

    Nodes ``s^2`` in [1e-4, 2e-3); below that the float32 ``qq`` itself no longer
    resolves ``q - q0`` to a percent on any edge, a statement about the ray, not the
    chord. The host is evaluated at ``q = qq``. Each ray is classified as the kernel
    classifies it: within ``s^2 + t_off < 2e-3`` of the tangent ray the closed form
    answers (whole or clipped at a face) and is held to the float32 chord tolerance;
    beyond it the twin answers, held to its derived rounding bound.
    """
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    edge = _per_edge(_segments(), [pick])[0]
    es, _ = aperture.upload_edge_set(EdgeGradientBeamletSource([edge]), device)
    s = np.sqrt(np.geomspace(1.0e-4, 1.9e-3, 25)).astype(np.float32)
    out = wp.zeros((len(s), 3), dtype=float, device=device)
    wp.launch(
        _block_terms_probe, dim=len(s), inputs=[es, wp.array(s, dtype=float, device=device), out],
        device=device,
    )  # fmt: skip
    q, chord, rate = out.numpy().astype(np.float64).T
    r = np.full(len(q), edge._r_c)
    local, _ = edge._directions(q, r)
    np.testing.assert_allclose(chord, edge._block_chords(q, r), rtol=CHORD_RTOL, atol=CHORD_ATOL)
    host_rate = edge._chord_rates(local)
    closed = s.astype(np.float64) ** 2 + edge._sign * (edge._q0 - edge._q_tangent) < 2.0e-3
    np.testing.assert_allclose(rate[closed], host_rate[closed], rtol=CHORD_RTOL)
    z, radius = edge._z_mid, edge._block.tip_radius
    c = edge._x + edge._sign * radius
    cu, cw = q / np.hypot(q, z), z / np.hypot(q, z)
    t_c = c * cu + z * cw
    half = np.sqrt(radius**2 - (c * cw - z * cu) ** 2)
    bound = TWIN_ROUNDINGS * 2.0**-24 * t_c**2 / (2.0 * half**2)
    error = np.abs(rate / host_rate - 1.0)
    assert np.all(error[~closed] <= bound[~closed]), (error / bound)[~closed]


# -- integrated derivative mass, kernel against float64 host (deterministic) -------------

MU_PROBE = 0.5  # 1/cm: one fixed attenuation, so the integrand tests chord and rate only
MASS_RTOL = 1.0e-3  # see test_derivative_mass_matches_the_host_quadrature
N_NODES = 3_000


@wp.kernel
def _block_terms_sweep(
    es: aperture.EdgeSet, s_in: wp.array2d(dtype=float), out: wp.array3d(dtype=float)
):
    """Edge ``j``'s ray at its node ``s_in[j, i]``, built as the generator builds it."""
    j, i = wp.tid()
    s = s_in[j, i]
    qq = es.q0[j] + es.sign[j] * s * s
    lu = qq
    lv = es.r_c[j]
    if es.q_axis[j] == 1:
        lu = es.r_c[j]
        lv = qq
    zm = es.z_mid[j]
    inv = 1.0 / wp.sqrt(lu * lu + lv * lv + zm * zm)
    l_block, rate, _l_rest = aperture._block_terms(es, j, qq, s, lu * inv, lv * inv, zm * inv)
    out[j, i, 0] = l_block
    out[j, i, 1] = rate


def _mass_ratios(edges, device: str) -> np.ndarray:
    """``int mu exp(-mu l) |dl/dx| 2s ds`` over the proposal's s range, kernel / host.

    The same float32 nodes on both sides (dense toward the support start), the host
    in float64 at ``q = q0 + sign s^2``: quadrature error is common to both, so the
    ratio measures the kernel's chord and rate alone.
    """
    es, _ = aperture.upload_edge_set(EdgeGradientBeamletSource(list(edges)), device)
    s32 = np.array(
        [np.concatenate([[0.0], np.geomspace(1e-7, np.sqrt(e._span), N_NODES)]) for e in edges],
        dtype=np.float32,
    )
    out = wp.zeros((len(edges), N_NODES + 1, 2), dtype=float, device=device)
    wp.launch(
        _block_terms_sweep, dim=s32.shape,
        inputs=[es, wp.array(s32, dtype=float, device=device), out], device=device,
    )  # fmt: skip
    device_terms = out.numpy().astype(np.float64)
    ratios = []
    for j, edge in enumerate(edges):
        s = s32[j].astype(np.float64)
        q = edge._q0 + edge._sign * s**2
        r = np.full(len(q), edge._r_c)
        local, _ = edge._directions(q, r)

        def mass(l, rate, s=s):
            return np.trapezoid(MU_PROBE * np.exp(-MU_PROBE * l) * np.abs(rate) * 2.0 * s, s)

        host = mass(edge._block_chords(q, r), edge._chord_rates(local))
        ratios.append(mass(device_terms[j, :, 0], device_terms[j, :, 1]) / host)
    return np.array(ratios)


def _tangent_w(x: float, sign: float, z: float, radius: float) -> float:
    """Depth of a rounded tip's tangent point on the focal ray (the closed form)."""
    c = x + sign * radius
    root = math.sqrt(z * z + c * c - radius * radius)
    q0 = (c * z * z - sign * z * radius * root) / (z * z - radius * radius)
    norm = math.hypot(q0, z)
    return (c * q0 + z * z) / norm * z / norm


def _face_edge(side: str, outside: float) -> FanEdgeDerivativeSource:
    """A 20 cm tip whose tangent point sits ``outside`` cm above the slab's top face."""
    radius, z_bottom = 20.0, 52.0
    x = 2.0 if side == "neg" else -2.0
    sign = 1.0 if side == "pos" else -1.0
    lo, hi = z_bottom - 5.0, z_bottom - 0.01
    for _ in range(200):  # z_top with the tangent point at z_top - outside
        mid = 0.5 * (lo + hi)
        if _tangent_w(x, sign, 0.5 * (mid + z_bottom), radius) > mid - outside:
            lo = mid
        else:
            hi = mid
    focal = (0.0, 0.0, -100.0)
    fan = SpectralBeamSource(
        Spectrum(edges=[1.0, 2.2], weights=[1.0]), focal_point=focal, center=(x, 0.0, 0.0),
        width_u=8.0, width_v=4.0,
    )  # fmt: skip
    mlc = MLC(
        z_top=0.5 * (lo + hi), z_bottom=z_bottom, leaf_edges_v=(-1.0, 1.0),
        tips_neg=(x if side == "neg" else -50.0,), tips_pos=(50.0 if side == "neg" else x,),
        tip_radius=radius, material=WATER, density=5.0, focused_sides=True,
    )  # fmt: skip
    stack = BeamLimitingStack(frame=BeamFrame(origin=focal), devices=(mlc,))
    return FanEdgeDerivativeSource(fan, stack, XS, EdgeParameter(0, side, 0))


@pytest.mark.parametrize("device", DEVICES)
def test_derivative_mass_matches_the_host_quadrature(device: str) -> None:
    """Every smooth fixture edge, and a tangent point swept across a slab face.

    Deterministic: one quadrature, kernel chord and rate against float64 on the same
    nodes. With the fix every ratio is within 2.5e-4 of 1 on cpu and cuda (measured
    2026-10-23). A near-tangent ray handed to the twin drops or skews the sliver that
    carries a quarter of the mass: the tangent-flag gate (the 2026-10-23 review
    finding, tangent point 1e-5..1e-2 cm outside a face) left the ratio 0.2-3.1 %
    off, the original ``s^2``-only gate 21-35 %. MASS_RTOL sits between, 4x above
    the fixed route and 2x below the smallest defect.
    """
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    edges = [e for e in _per_edge(_segments(), _expected_parameters(4)) if not e._jump]
    sweep = [
        _face_edge(side, outside)
        for side in ("neg", "pos")
        for outside in (-1e-3, 0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    ]
    for label, group in (("fixture", edges), ("face sweep", sweep)):
        ratios = _mass_ratios(group, device)
        bad = np.flatnonzero(np.abs(ratios - 1.0) > MASS_RTOL)
        assert bad.size == 0, (label, bad.tolist(), ratios[bad].tolist())
