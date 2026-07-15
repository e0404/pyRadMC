"""Nightly sanity gate: the 80-20 penumbra of a pre-solved 6 MV tungsten field.

A Varian-type 6 MV beam (Ali & Rogers analytic spectrum) through real tungsten
jaws (compiled EPDL table), 1 mm Gaussian spot, 10 x 10 cm field at SSD 100:
the crossline 80-20 penumbra at 10 cm depth is gated against the width the v1
**straight-edge (unfocused) jaw model predicts**, not against clinical focused-
jaw values. A straight edge of height ``H`` at distance ``z_mid`` throws a
partial-transmission band whose projection at measurement distance ``D`` is
``u_edge * D * (1/z_top - 1/z_bottom)`` — here ~8.9 mm — on top of which sit the
spot size (~1.7 mm projected) and the in-phantom transport blur (~4-6 mm at
depth for 6 MV). Measured 2026-07-15: 15.8 mm, consistent with that stack-up;
a future focused-edge option should collapse the geometric band and bring the
width toward the clinical 4-8 mm. The gate brackets the model so a projection,
frame, or pre-solve regression is caught; it needs the EPICS libraries (skips
without the env paths).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyRadMC.data.materials import TUNGSTEN
from pyRadMC.geometry.collimation import BeamFrame, BeamLimitingStack, JawPair
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.head import presolve_head
from pyRadMC.geometry.source import GaussianSpotBeamSource
from pyRadMC.geometry.spectrum import ali_rogers_mv
from tests.conftest import SEED

pytestmark = [
    pytest.mark.validation,
    pytest.mark.filterwarnings("ignore:.*latent variance:UserWarning"),
]

SAD = 100.0
FIELD_HALF = 5.0  # 10 x 10 cm at the isocenter plane
JAW_TOP, JAW_BOTTOM = 40.0, 47.0
DEPTH_CM = 10.0


def _compiled_data():
    epdl = os.environ.get("PYRADMC_EPDL_PATH")
    eedl = os.environ.get("PYRADMC_EEDL_PATH")
    if not (epdl and Path(epdl).is_file() and eedl and Path(eedl).is_file()):
        pytest.skip("set PYRADMC_EPDL_PATH and PYRADMC_EEDL_PATH to run this validation")
    from pyRadMC.data.tabulated.precompile import compile_materials

    return compile_materials(
        Path(epdl).read_text(encoding="latin-1"),
        Path(eedl).read_text(encoding="latin-1"),
        e_max=7.0,
    )


def test_80_20_penumbra_width_is_physical() -> None:
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyRadMC.backends.warp.engine import WarpEngine

    # Phantom: 24 cm crossline at 2 mm, 10 cm inline at 1 cm, 16 cm depth at 5 mm;
    # surface at z = 0, beam axis through its center.
    nx, ny, nz = 120, 10, 32
    grid = VoxelGrid.uniform_water(shape=(nx, ny, nz), spacing=(0.2, 1.0, 0.5))
    center_x, center_y = nx * 0.2 / 2.0, ny * 1.0 / 2.0
    focal = (center_x, center_y, -SAD)
    frame = BeamFrame(origin=focal)

    mid = (JAW_TOP + JAW_BOTTOM) / 2.0
    edge = FIELD_HALF * mid / SAD  # project the isocenter setting to the jaw plane
    stack = BeamLimitingStack(
        frame=frame,
        devices=(
            JawPair(
                axis="u",
                z_top=JAW_TOP,
                z_bottom=JAW_BOTTOM,
                edge_neg=-edge,
                edge_pos=edge,
                material=TUNGSTEN,
                density=19.30,
            ),
        ),
    )
    # The emission plane sits above the jaws and generously covers the open fan.
    source = GaussianSpotBeamSource(
        spectrum=ali_rogers_mv("varian-6mv"),
        focal_point=focal,
        center=(center_x, center_y, -SAD + 30.0),
        width_u=6.0,
        width_v=6.0,
        sigma_u=0.1,  # 1 mm focal spot
        sigma_v=0.1,
    )
    from pyRadMC.data.tabulated.source import TabulatedCrossSections

    data = _compiled_data()
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    n_histories = 2_000_000
    phsp = presolve_head(
        source,
        stack=stack,
        cross_sections=xs,
        n_histories=n_histories,
        seed=SEED,
        exit_z=SAD - 5.0,  # just above the surface; the gap to the grid is vacuum
    )
    result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        phsp, n_histories=n_histories, n_batches=8, seed=SEED + 1
    )

    depth_index = int(DEPTH_CM / 0.5)
    profile = result.dose[:, ny // 2, depth_index].astype(np.float64)
    x = (np.arange(nx) + 0.5) * 0.2
    # Normalize to the central plateau and find the 80/20 crossings on the +x side.
    plateau = profile[np.abs(x - center_x) < 2.0].mean()
    relative = profile / plateau
    right = x > center_x
    x_right, p_right = x[right], relative[right]
    x80 = float(np.interp(0.8, p_right[::-1], x_right[::-1]))
    x20 = float(np.interp(0.2, p_right[::-1], x_right[::-1]))
    width_mm = 10.0 * (x20 - x80)

    # The straight-edge model: the edge face's partial-transmission band projected
    # to the measurement plane (SAD + depth), plus room for spot size and the
    # in-phantom blur above it — and no less than half the band below it.
    band_mm = 10.0 * edge * (SAD + DEPTH_CM) * (1.0 / JAW_TOP - 1.0 / JAW_BOTTOM)
    low, high = 0.5 * band_mm, band_mm + 10.0
    assert low < width_mm < high, (
        f"80-20 penumbra {width_mm:.1f} mm outside [{low:.1f}, {high:.1f}] mm "
        f"(straight-edge band {band_mm:.1f} mm)"
    )
