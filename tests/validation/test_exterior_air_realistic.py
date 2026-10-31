"""The exterior-air path on realistic geometry: tabulated, spectral, heterogeneous.

The measurement AGENTS.md section 8 asks of any low-density fast path: the elliptical body of
:mod:`tests.exterior_fixtures`, 6 MV from a focal spot, at 1.5 and 0.35 T with the field along
and across the body, old path against new over several seeds.

The statistics are replicate-level. Doses are rebinned into 1.2 cm blocks and each arm's error
is its seed spread, then fed to the calibrated batched chi-squared, because voxelwise batched
chi-squared has no power on sparse voxels; the masks are defined on the mean of both arms. The
quantities: body blocks, blocks of the outer shell (two voxels next to the air, where electrons
have to curve back), mean shell dose and mean body dose.

**Power is part of the gate.** The same oracle must reject a safety map inflated four times.
Measured 2026-10-31 at the 0.25 cm minimum and these statistics, mutant against current
stepping: shell -1.13 to -7.1 %, block chi-squared p at most 1e-210, against -0.06 to +0.02 % and
p above 0.1 for the true map. The same statistic misses a 1.5x map except at 0.35 T. A wrong
*chord* is **not** visible here (shell at most +0.11 +- 0.07 %): body dose does not resolve what
happens in the air far from it. The chord is pinned by the analytic orbit in
``tests/physics/test_exterior_helix.py`` on the reference loop and through the compiled form on
Warp, and that is where its protection is, not here.

The shell's own z-squared sums read somewhat below expectation (0.79 to 0.93), which on sparse
data is a warning rather than a pass; the mutant's 12.7 to 474 on the same statistic is what
shows the oracle is not blind.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.tabulated import build
from pyradmc.data.tabulated.precompile import compile_materials
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry.exterior import ExteriorAirMap
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched
from tests.exterior_fixtures import block_sum, body_grid, body_masks

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.validation, pytest.mark.warp, pytest.mark.gpu]

MARGIN_CM = 3.0
N_SEEDS = 8
N_HISTORIES = 40_000_000
BLOCK = 4  # voxels of 3 mm
MASK_FRACTION = 0.05
SHELL_GATE = 0.0025  # measured at most 0.0006 +- 0.0005; the 4x mutant at least 0.0113
BODY_GATE = 0.001  # measured at most 0.0001 +- 0.0002
INFLATION = 4.0
TWIN_SEEDS = 6
TWIN_HISTORIES = 20_000

CASES = [
    ((1.5, 0.0, 0.0), "AP"),
    ((0.0, 0.0, 1.5), "AP"),
    ((0.0, 0.0, 1.5), "LAT"),
    ((0.0, 0.0, 0.35), "AP"),
]
IDS = ["1.5T-across-AP", "1.5T-along-AP", "1.5T-along-LAT", "0.35T-along-AP"]


@pytest.fixture(scope="module")
def world():
    """The body with compiled EPICS media, or skip."""
    if not wp.is_cuda_available():
        pytest.skip("the realistic gate needs ~1e9 histories per case; CUDA only")
    epdl, eedl = build.library_path("epdl", None), build.library_path("eedl", None)
    if not (epdl.is_file() and eedl.is_file()):
        pytest.skip("EPICS libraries not cached; run python -m pyradmc.data.tabulated.build")
    grid = body_grid(MARGIN_CM)
    data = compile_materials(
        epdl.read_text(encoding="latin-1"), eedl.read_text(encoding="latin-1"), e_max=8.0
    )
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    return grid, xs


def _source(grid, geometry: str) -> SpectralBeamSource:
    c = tuple(n * s / 2.0 for n, s in zip(grid.shape, grid.spacing, strict=True))
    if geometry == "AP":
        focal = (c[0], c[1] - 100.0, c[2])
        u_axis = (1.0, 0.0, 0.0)
    else:
        focal = (c[0] - 100.0, c[1], c[2])
        u_axis = (0.0, 1.0, 0.0)
    return SpectralBeamSource(
        spectrum=ali_rogers_mv("varian-6mv"),
        focal_point=focal,
        center=c,
        width_u=10.0,
        width_v=10.0,
        u_axis=u_axis,
        v_axis=(0.0, 0.0, 1.0),
    )


def _arm(world, b, geometry: str, air: ExteriorAirMap | None, seed_offset: int) -> dict[str, Any]:
    from pyradmc.backends.warp.engine import WarpEngine

    grid, xs = world
    masks = body_masks(grid)

    def build() -> WarpEngine:
        return WarpEngine(
            grid=grid,
            cross_sections=xs,
            device="cuda:0",
            magnetic_field=MagneticField.uniform(b),
            exterior_air=air,
        )

    if air is None:
        engine = build()
    else:
        with pytest.warns(UserWarning, match="exterior-air"):
            engine = build()
    source = _source(grid, geometry)
    blocks, shells, shell_mean, body_mean = [], [], [], []
    for k in range(N_SEEDS):
        result = engine.run(
            source, n_histories=N_HISTORIES, n_batches=1, seed=SEED + seed_offset + k
        )
        books = result.energy_deposited + result.energy_unscored + result.energy_escaped
        assert abs(result.energy_emitted - books) < 1e-4 * result.energy_emitted
        dose = np.asarray(result.dose)
        blocks.append(block_sum(dose * masks.body, BLOCK))
        shells.append(block_sum(dose * masks.shell, BLOCK))
        shell_mean.append(dose[masks.shell].mean())
        body_mean.append(dose[masks.body].mean())
    engine.release_device_memory()
    return {
        "blocks": np.stack(blocks),
        "shells": np.stack(shells),
        "shell": np.asarray(shell_mean),
        "body": np.asarray(body_mean),
    }


def _ratio(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    sa = a.std(ddof=1) / np.sqrt(N_SEEDS) / a.mean()
    sb = b.std(ddof=1) / np.sqrt(N_SEEDS) / b.mean()
    r = float(b.mean() / a.mean())
    return r - 1.0, float(r * np.hypot(sa, sb))


def _assert_blocks_agree(a: np.ndarray, b: np.ndarray) -> None:
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    sa = a.std(axis=0, ddof=1) / np.sqrt(N_SEEDS)
    sb = b.std(axis=0, ddof=1) / np.sqrt(N_SEEDS)
    fair = 0.5 * (ma + mb)
    mask = (fair > MASK_FRACTION * fair.max()) & (sa > 0.0) & (sb > 0.0)
    assert int(mask.sum()) >= 100, f"only {int(mask.sum())} blocks to compare"
    assert_chi2_consistent_batched(ma, sa, N_SEEDS, mb, sb, N_SEEDS, mask=mask)


@pytest.mark.parametrize(("b", "geometry"), CASES, ids=IDS)
def test_new_path_matches_current_stepping(world, b, geometry: str) -> None:
    grid, _ = world
    old = _arm(world, b, geometry, None, 0)
    new = _arm(world, b, geometry, ExteriorAirMap.from_grid(grid), 1000)
    shell, s_shell = _ratio(old["shell"], new["shell"])
    body, s_body = _ratio(old["body"], new["body"])
    print(f"\nshell {shell:+.5f} +- {s_shell:.5f}, body {body:+.5f} +- {s_body:.5f}")
    assert abs(shell) < SHELL_GATE, f"shell moved {shell:+.5f} +- {s_shell:.5f}"
    assert abs(body) < BODY_GATE, f"body moved {body:+.5f} +- {s_body:.5f}"
    _assert_blocks_agree(old["blocks"], new["blocks"])
    _assert_blocks_agree(old["shells"], new["shells"])


@pytest.mark.parametrize(("b", "geometry"), CASES, ids=IDS)
def test_the_comparison_catches_an_inflated_safety(world, b, geometry: str) -> None:
    grid, _ = world
    air = ExteriorAirMap.from_grid(grid)
    old = _arm(world, b, geometry, None, 0)
    bad = dataclasses.replace(air, safety=air.safety * INFLATION)
    mutant = _arm(world, b, geometry, bad, 1000)
    shell, s_shell = _ratio(old["shell"], mutant["shell"])
    print(f"\nmutant shell {shell:+.5f} +- {s_shell:.5f}")
    assert abs(shell) > SHELL_GATE, f"a {INFLATION}x safety map moved the shell only {shell:+.5f}"
    with pytest.raises(AssertionError):
        _assert_blocks_agree(old["shells"], mutant["shells"])


@pytest.mark.parametrize("b", [(1.5, 0.0, 0.0), (0.0, 0.0, 0.35)], ids=["1.5T", "0.35T"])
def test_warp_matches_the_reference_on_the_body(world, b) -> None:
    """Twin check through the new path, at equal history counts, energy per slab along the beam.

    Energy rather than dose so near-vacuum voxels inject no 1/mass speckle; laterally integrated
    so the statistic has power; the seed spread is the uncertainty (as in
    ``test_gs_thorax_cross_backend``).
    """
    from pyradmc.backends.warp.engine import WarpEngine

    grid, xs = world
    air = ExteriorAirMap.from_grid(grid)
    field = MagneticField.uniform(b)
    source = _source(grid, "AP")
    volume = float(np.prod(grid.spacing))
    density = np.asarray(grid.density)

    def profile(result) -> np.ndarray:
        return np.sum(np.asarray(result.dose) * density * volume, axis=(0, 2))

    with pytest.warns(UserWarning, match="exterior-air"):
        ref_engine = ReferenceEngine(
            grid=grid, cross_sections=xs, rng=HostRNG(), magnetic_field=field, exterior_air=air
        )
    with pytest.warns(UserWarning, match="exterior-air"):
        warp_engine = WarpEngine(
            grid=grid, cross_sections=xs, device="cuda:0", magnetic_field=field, exterior_air=air
        )
    kwargs: dict[str, Any] = {"n_histories": TWIN_HISTORIES, "n_batches": 1}
    ref = np.stack(
        [profile(ref_engine.run(source, seed=SEED + k, **kwargs)) for k in range(TWIN_SEEDS)]
    )
    warp = np.stack(
        [profile(warp_engine.run(source, seed=SEED + 100 + k, **kwargs)) for k in range(TWIN_SEEDS)]
    )
    warp_engine.release_device_memory()
    a, b_ = ref.mean(axis=0), warp.mean(axis=0)
    sa = ref.std(axis=0, ddof=1) / np.sqrt(TWIN_SEEDS)
    sb = warp.std(axis=0, ddof=1) / np.sqrt(TWIN_SEEDS)
    fair = 0.5 * (a + b_)
    mask = (fair > 0.01 * fair.max()) & (sa > 0.0) & (sb > 0.0)
    assert int(mask.sum()) >= 20, "body profile mask unexpectedly small"
    assert_chi2_consistent_batched(a, sa, TWIN_SEEDS, b_, sb, TWIN_SEEDS, mask=mask)
