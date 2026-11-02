"""The Warp engine's device-resident Dij and device-resident contraction inputs.

``WarpEngine.run_dij_device`` runs the same computation as ``run_dij`` on one device
and stops before the host: the truncated, compacted CSC blocks stay where the kernels
wrote them and are concatenated there. The contract is therefore not statistical but
exact (AGENTS.md 2.3 *within* one target): on the same device, with the same
arguments, ``run_dij_device(...).to_host()`` equals ``run_dij(...)`` field for field,
bit for bit. Pinned here for an in-kernel lattice and an in-kernel segmented fan,
with and without truncation, correlated and independent, on ``cpu`` and ``cuda:0``,
over several beamlet groups so the device concatenation is what is being tested.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pyradmc import DIJ_TRUNCATION_RELATIVE
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segments import SegmentedFanSource
from pyradmc.geometry.source import BeamletGridSource, SpectralBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.scoring.dij import DijResult
from pyradmc.scoring.grid import ScoringGrid
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

ARRAY_FIELDS = ("indptr", "indices", "dose", "sigma")


@pytest.fixture(scope="module", params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


# -- the two sources --------------------------------------------------------------


def _lattice_grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(8, 8, 16), spacing=(2.0, 2.0, 0.5))


def _lattice(x_range=(2.0, 14.0)) -> BeamletGridSource:
    return BeamletGridSource(energy=6.0, z=-1.0, x_range=x_range, y_range=(2.0, 14.0), n_x=3, n_y=2)


SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
FRAME = BeamFrame(origin=FOCAL)
SETTINGS = (
    (-2.0, 2.0, (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)),
    (-0.5, 2.5, (-0.5, 0.0, -0.5), (2.0, 2.0, 2.0)),
    (-2.5, 0.5, (-0.3, -0.6, -0.3), (0.3, 0.6, 0.3)),
)


def _stack(jaw_neg, jaw_pos, tips_neg, tips_pos) -> BeamLimitingStack:
    return BeamLimitingStack(
        frame=FRAME,
        devices=(
            JawPair("u", 30.0, 36.0, jaw_neg, jaw_pos, material=WATER, density=4.0),
            MLC(
                z_top=40.0,
                z_bottom=46.0,
                leaf_edges_v=(-2.0, -0.6, 0.6, 2.0),
                tips_neg=tuple(tips_neg),
                tips_pos=tuple(tips_pos),
                tip_radius=8.0,
                material=WATER,
                density=4.0,
                focused_sides=True,
            ),
        ),
    )


def _segmented() -> SegmentedFanSource:
    fan = SpectralBeamSource(
        SPECTRUM, focal_point=FOCAL, center=(8.0, 8.0, 0.0), width_u=10.0, width_v=8.0
    )
    return SegmentedFanSource.from_stacks(
        fan, [_stack(*s) for s in SETTINGS], AnalyticCrossSections()
    )


def _fan_grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def _engine(grid: VoxelGrid, device: str):
    from pyradmc.backends.warp.engine import WarpEngine

    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    return WarpEngine(grid=grid, cross_sections=xs, device=device)


def _case(kind: str):
    """(grid, source, extra kwargs) for one source kind; small and fast on cpu."""
    if kind == "lattice":
        return _lattice_grid(), _lattice(), dict(n_histories_per_beamlet=300, n_batches=3)
    grid = _fan_grid()
    scoring = ScoringGrid.rebin(grid, shape=(8, 8, 8), spacing=(2.0, 2.0, 2.0), origin=grid.origin)
    return (
        grid,
        _segmented(),
        dict(
            n_histories_per_beamlet=600,
            n_batches=3,
            transport_electrons=False,
            scoring_grid=scoring,
        ),
    )


def _assert_identical(got: DijResult, want: DijResult) -> None:
    """Every field of two DijResults, arrays with dtype and bit-exact values."""
    for f in dataclasses.fields(DijResult):
        a, b = getattr(got, f.name), getattr(want, f.name)
        if f.name in ARRAY_FIELDS:
            assert a.dtype == b.dtype, f.name
            assert np.array_equal(a, b), f.name
        else:
            assert a == b, f.name


# -- acceptance: bit-identical to the host path ------------------------------------


@pytest.mark.parametrize("correlated", [True, False])
@pytest.mark.parametrize("truncation", [0.0, DIJ_TRUNCATION_RELATIVE])
@pytest.mark.parametrize("kind", ["lattice", "segmented"])
def test_device_dij_equals_host_dij(device: str, kind: str, truncation, correlated) -> None:
    grid, source, kwargs = _case(kind)
    engine = _engine(grid, device)
    # Group size 2 forces several groups (3 or 6 columns), so the device-side
    # concatenation and indptr are exercised, not just one block.
    common = dict(
        seed=SEED,
        truncation=truncation,
        correlated=correlated,
        beamlet_group_size=2,
        **kwargs,
    )
    want = engine.run_dij(source, **common)
    got = engine.run_dij_device(source, **common)
    assert got.device == wp.get_device(device)
    _assert_identical(got.to_host(), want)
    assert want.indices.size > 0  # the comparison is not vacuous


def test_empty_columns_and_empty_groups_concatenate(device: str) -> None:
    """Beamlets that miss the phantom give zero-entry columns (and, at group size 1,
    zero-entry blocks); the device indptr must carry them exactly like the host's."""
    engine = _engine(_lattice_grid(), device)
    source = _lattice(x_range=(-16.0, 14.0))  # first column entirely off the grid
    common = dict(n_histories_per_beamlet=300, n_batches=3, seed=SEED, beamlet_group_size=1)
    want = engine.run_dij(source, **common)
    counts = np.diff(want.indptr)
    assert (counts == 0).any() and (counts > 0).any()
    _assert_identical(engine.run_dij_device(source, **common).to_host(), want)


def test_device_arrays_are_warp_arrays_on_the_engine_device(device: str) -> None:
    grid, source, kwargs = _case("lattice")
    got = _engine(grid, device).run_dij_device(source, seed=SEED, **kwargs)
    assert got.indptr.dtype == wp.int64 and got.indptr.shape == (source.n_beamlets + 1,)
    assert got.indices.dtype == wp.int64
    assert got.dose.dtype == wp.float64 and got.sigma.dtype == wp.float64
    nnz = got.nnz
    assert got.indices.shape == got.dose.shape == got.sigma.shape == (nnz,)
    for a in (got.indptr, got.indices, got.dose, got.sigma):
        assert a.device == wp.get_device(device)
    assert got.n_voxels == int(np.prod(got.grid_shape))


def test_to_host_owns_its_arrays() -> None:
    """On the cpu device ``numpy()`` is a view; the host result must not alias it."""
    grid, source, kwargs = _case("lattice")
    got = _engine(grid, "cpu").run_dij_device(source, seed=SEED, **kwargs)
    host = got.to_host()
    before = host.dose.copy()
    got.dose.fill_(-1.0)
    assert np.array_equal(host.dose, before)


def test_device_path_does_no_host_assembly(device: str, monkeypatch) -> None:
    """No DijAssembler: the blocks never leave the device (only the per-group counts
    that size the compaction are read back, as on the host path)."""
    from pyradmc.backends.warp import engine as engine_module

    class _Refuse:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("host assembly on the device path")

    grid, source, kwargs = _case("lattice")
    engine = _engine(grid, device)
    monkeypatch.setattr(engine_module, "DijAssembler", _Refuse)
    engine.run_dij_device(source, seed=SEED, beamlet_group_size=2, **kwargs)


def test_more_than_one_device_is_refused() -> None:
    grid, source, kwargs = _case("lattice")
    with pytest.raises(ValueError, match="one device"):
        _engine(grid, "cpu").run_dij_device(source, seed=SEED, devices=["cpu", "cuda:0"], **kwargs)


def test_an_explicit_single_device_is_honoured(device: str) -> None:
    """``devices=[d]`` computes on ``d`` even if the engine's own device differs."""
    grid, source, kwargs = _case("lattice")
    engine = _engine(grid, "cpu")
    got = engine.run_dij_device(source, seed=SEED, devices=[device], **kwargs)
    assert got.device == wp.get_device(device)
    want = engine.run_dij(source, seed=SEED, devices=[device], **kwargs)
    _assert_identical(got.to_host(), want)
