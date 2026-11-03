"""In-kernel fan edge rays: the device route of ``EdgeGradientBeamletSource``.

The host pre-sampling route seeds and drains the transport queues once per edge,
which dominated the wall clock of a 400-edge gradient (~1.4 ms per edge). A set of
:class:`FanEdgeDerivativeSource` edges therefore gets a built-in generator: every
segment's stack in one device table, the per-edge proposal tables, and the exact
per-ray weight ``-mu T_block dl/dx / p(q) * T_rest`` computed on the device. Pinned
here: the route is taken (the host sampler is never called), and its contraction
agrees statistically with the host route and with the reference engine, over two
segments with different stacks, a smooth tip edge and a jump jaw edge.
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
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

FOCAL = (5.0, 2.0, -100.0)
SPECTRUM = Spectrum(edges=[1.8, 2.2], weights=[1.0])
# Independent streams per edge: correlated columns share their fluctuations, so a
# chi-squared across columns would be inflated (AGENTS.md 8; measured here first).
KWARGS = dict(n_batches=8, transport_electrons=False, correlated=False)


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(20, 2, 8), spacing=(0.5, 2.0, 1.0))


def _xs() -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=_grid().max_density_by_material())


def _beam() -> SpectralBeamSource:
    return SpectralBeamSource(
        SPECTRUM, focal_point=FOCAL, center=(5.5, 2.0, 0.0), width_u=1.0, width_v=4.0
    )


def _stack(tips_pos: tuple[float, float], jaw_edge: float) -> BeamLimitingStack:
    return BeamLimitingStack(
        frame=BeamFrame(origin=FOCAL),
        devices=(
            JawPair(
                axis="u",
                z_top=30.0,
                z_bottom=37.8,
                edge_neg=-5.0,
                edge_pos=jaw_edge,
                material=WATER,
                density=5.0,
                focused=True,
            ),
            MLC(
                z_top=50.0,
                z_bottom=56.0,
                leaf_edges_v=(-1.06, 0.0, 1.06),
                tips_neg=(-5.0, -5.0),
                tips_pos=tips_pos,
                tip_radius=8.0,
                material=WATER,
                density=5.0,
                focused_sides=True,
            ),
        ),
    )


def _edges() -> list[FanEdgeDerivativeSource]:
    """Two segments: tip edges of both pairs under an open jaw, then a jaw edge alone."""
    xs = _xs()
    segment_a = _stack((0.27, 0.35), 5.0)
    segment_b = _stack((8.0, 8.0), 0.18)
    return [
        FanEdgeDerivativeSource(_beam(), segment_a, xs, EdgeParameter(1, "pos", 0)),
        FanEdgeDerivativeSource(_beam(), segment_a, xs, EdgeParameter(1, "pos", 1)),
        FanEdgeDerivativeSource(_beam(), segment_b, xs, EdgeParameter(0, "pos")),
    ]


def _weights() -> np.ndarray:
    x, y, z = np.meshgrid(*(np.arange(n) for n in _grid().shape), indexing="ij")
    return np.exp(-(((x - 11.0) / 3.0) ** 2)) * (1.0 + 0.2 * y) - 0.05 * z


class _HostRouted(EdgeGradientBeamletSource):
    """A subclass keeps the host pre-sampling route (exact-type routing)."""


def _warp():
    from pyradmc.backends.warp.engine import WarpEngine

    return WarpEngine(grid=_grid(), cross_sections=_xs(), device="cpu")


def test_the_in_kernel_route_is_taken(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("the host sampler must not run on the in-kernel route")

    monkeypatch.setattr(FanEdgeDerivativeSource, "sample_batch", refuse)
    result = _warp().run_beamlet_contraction(
        EdgeGradientBeamletSource(_edges()),
        _weights(),
        n_histories_per_beamlet=800,
        seed=SEED,
        **KWARGS,
    )
    assert np.all(np.isfinite(result.values)) and np.any(result.values != 0.0)


def test_in_kernel_agrees_with_the_host_route() -> None:
    engine = _warp()
    device = engine.run_beamlet_contraction(
        EdgeGradientBeamletSource(_edges()),
        _weights(),
        n_histories_per_beamlet=16_000,
        seed=SEED,
        **KWARGS,
    )
    host = engine.run_beamlet_contraction(
        _HostRouted(_edges()), _weights(), n_histories_per_beamlet=16_000, seed=SEED + 1, **KWARGS
    )
    assert_chi2_consistent_batched(device.values, device.sigma, 8, host.values, host.sigma, 8)
    assert np.all(np.abs(device.values) > 3.0 * device.sigma)  # each edge resolved


def test_in_kernel_agrees_with_the_reference_engine() -> None:
    device = _warp().run_beamlet_contraction(
        EdgeGradientBeamletSource(_edges()),
        _weights(),
        n_histories_per_beamlet=16_000,
        seed=SEED,
        **KWARGS,
    )
    ref = ReferenceEngine(grid=_grid(), cross_sections=_xs(), rng=HostRNG())
    reference = ref.run_beamlet_contraction(
        EdgeGradientBeamletSource(_edges()),
        _weights(),
        n_histories_per_beamlet=1_600,
        seed=SEED,
        **KWARGS,
    )
    assert_chi2_consistent_batched(
        reference.values, reference.sigma, 8, device.values, device.sigma, 8
    )


def _device_rays(edges, n_per: int):
    """Run the generator alone: per-edge weights, energies, engine directions."""
    from pyradmc.backends.warp import aperture
    from pyradmc.backends.warp.engine import _upload_queue

    edge_set, (sp_edges, sp_cdf) = aperture.upload_edge_set(EdgeGradientBeamletSource(edges), "cpu")
    total = len(edges) * n_per
    queue = _upload_queue(total, "cpu")
    slots = wp.zeros(total, dtype=wp.uint32, device="cpu")
    emitted = wp.zeros(1, dtype=wp.int64, device="cpu")
    wp.launch(
        aperture.generate_fan_edges,
        dim=total,
        inputs=[
            *(SEED, 0, n_per, n_per, 0, 0, 0, sp_edges, sp_cdf, int(sp_cdf.shape[0])),
            *(edge_set, slots, emitted, queue),
        ],
        device="cpu",
    )
    weights = queue.weight.numpy()[:total].astype(np.float64).reshape(len(edges), n_per)
    energies = queue.energy.numpy()[:total].astype(np.float64).reshape(len(edges), n_per)
    directions = np.stack([queue.ux.numpy(), queue.uy.numpy(), queue.uz.numpy()], axis=1)
    directions = directions[:total].astype(np.float64).reshape(len(edges), n_per, 3)
    return weights, energies, directions, edge_set


def _response(edge, directions: np.ndarray) -> np.ndarray:
    """A smooth test response of the ray's mid-plane crossing (no transport)."""
    local = directions @ edge._axes.T
    q = local[:, 0] / local[:, 2] * edge._z_mid
    return np.exp(-(((q - edge._x) / 0.3) ** 2))


def test_device_weights_follow_the_host_arithmetic_ray_by_ray() -> None:
    """Away from tangency, each kernel weight is the host formula on the same ray.

    The ray's (q, r, E) is recovered from the float32 direction (q to ~5e-6 cm),
    so the comparison is restricted to q - q0 > 1e-2 cm, where that recovery is
    accurate; the near-tangent rays are covered by the source-level means below.
    """
    edges = _edges()
    weights, energies, directions, edge_set = _device_rays(edges, 20_000)
    n_table = edge_set.n_table
    s_tables = edge_set.s.numpy().astype(np.float64).reshape(-1, n_table)
    cell_tables = edge_set.cell.numpy().astype(np.float64).reshape(-1, n_table - 1)
    for j, edge in enumerate(edges[:2]):  # the smooth tip edges: tables 0 and 1
        local = directions[j] @ edge._axes.T
        q = local[:, 0] / local[:, 2] * edge._z_mid
        r = local[:, 1] / local[:, 2] * edge._z_mid
        far = edge._sign * (q - edge._q0) > 1.0e-2
        s = np.sqrt(np.abs(q - edge._q0))
        # The device's own (float32) proposal tables: the density it weights by.
        cell = np.clip(np.searchsorted(s_tables[j], s, side="right") - 1, 0, n_table - 2)
        density = cell_tables[j][cell] / (2.0 * s)
        _, engine = edge._directions(q, r)
        inside = (
            (q >= edge._field[0][0]) & (q <= edge._field[0][1])
            & (r >= edge._field[1][0]) & (r <= edge._field[1][1])
        )  # fmt: skip
        rest = edge._other.factors(energies[j], np.tile(edge._origin, (len(q), 1)), engine)
        host = np.where(
            inside, edge._scale * edge._derivative(q, r, energies[j]) / density * rest, 0.0
        )
        agree = np.isclose(weights[j][far], host[far], rtol=5e-3, atol=1e-6)
        assert far.sum() > 5_000
        assert agree.mean() > 0.999, (j, int((~agree).sum()))


@pytest.mark.parametrize("j", [0, 1, 2], ids=["tip-0", "tip-1", "jaw"])
def test_device_rays_are_unbiased_at_the_source(j: int) -> None:
    """Mean weight and mean weight x response, kernel against host batch, 200k rays each.

    Source-level and transport-free, so a percent-level bias is a ten-sigma event:
    this is the check that catches the near-tangent rays, which carry the largest
    weights and are where a float32 circle test cancels.
    """
    edges = _edges()
    weights, _, directions, _ = _device_rays(edges, 200_000)
    host = edges[j].sample_batch(SEED + 7, 0, 200_000)
    host_dirs = np.stack([host["ux"], host["uy"], host["uz"]], axis=1).astype(np.float64)
    host_w = host["weight"].astype(np.float64)
    for device_values, host_values in (
        (weights[j], host_w),
        (weights[j] * _response(edges[j], directions[j]), host_w * _response(edges[j], host_dirs)),
    ):
        sem = math.hypot(device_values.std(), host_values.std()) / math.sqrt(200_000)
        assert abs(device_values.mean() - host_values.mean()) < 4.0 * sem, (
            device_values.mean(),
            host_values.mean(),
            sem,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("lanes", [2, 4])
def test_batch_lanes_are_bit_inert_on_the_edge_route(lanes: int) -> None:
    """Concurrent batch lanes drain with their own pooled workspace and change nothing.

    The sequential lane resources carry no pinned staging element, so a drain that is
    handed one in place of the workspace falls back to a throwaway workspace and
    passes unnoticed; a lane's staging element is a real array, which that drain
    cannot use. Pinning bit-identity with lanes exercises the lane route end to end.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    engine = WarpEngine(grid=_grid(), cross_sections=_xs(), device="cuda:0")
    kwargs = dict(n_histories_per_beamlet=2_000, seed=SEED, **KWARGS)
    source = EdgeGradientBeamletSource(_edges())
    want = engine.run_beamlet_contraction(source, _weights(), **kwargs)
    got = engine.run_beamlet_contraction(source, _weights(), concurrent_batches=lanes, **kwargs)
    np.testing.assert_array_equal(got.values, want.values)
    np.testing.assert_array_equal(got.sigma, want.sigma)
