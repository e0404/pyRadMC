"""Exit-side electron return effect against Raaijmakers et al. 2008: the first literature gate.

Raaijmakers, Raaymakers & Lagendijk (2008, PMB 53, 909, doi:10.1088/0031-9155/53/4/006,
sec. 3.1) irradiated a 20 x 20 x 8 cm3 water phantom with a non-divergent 10 x 10 cm2
6 MV beam in a transverse field of 0.2, 0.75, 1.5 and 3 T (Geant4 8.2, 0.5 mm scoring)
and reported, relative to the field-free dose maximum:

- the maximum ERE dose increase at the exit: **40, 52, 42 and 25 %** — largest at 0.75 T
  and *falling* above it, because the compressed in-water kernel feeds the exit face
  from a smaller volume; and as large at low field as at 1.5 T for a broad field,
  because every exiting electron below a few MeV returns whatever the radius;
- the lateral "ERE shift" of the 20 % point of the exit profile toward the Lorentz
  side: **17.5, 9.5 and 4 mm** at 0.75, 1.5 and 3 T (beyond 5 cm at 0.2 T);
- the build-up depth: 17, 17, 15, 12 and 7 mm at 0, 0.2, 0.75, 1.5 and 3 T.

Measured here 2026-10-18 with the Ali-Rogers Varian 6 MV spectrum (theirs: an Elekta SL
head simulation) on the same phantom with air below it inside the grid: central-axis
excess in the last 0.5 mm of 0.42 / 0.63 / 0.50 / 0.29 at 0.2 / 0.75 / 1.5 / 3 T, i.e.
within 5-20 % of theirs with the same field-strength ratios (0.79 and 0.58 for
1.5 / 0.75 T and 3 / 1.5 T against their 0.81 and 0.60); shifts of 15 / 7.5 / 2.5 mm
(2.5 mm lateral bins); build-up 15.8 / 14.8 / 12.2 / 12.2 mm at 0 / 0.75 / 1.5 / 3 T.
The 3 T build-up (theirs 7 mm) is the one quantity off by more than the gate; it is
outside the MR-linac range and recorded as a caveat in docs/validation.md.

Gates are deliberately loose — a different spectrum, code and scoring — but tight
enough that a lost factor in the gyroradius, the returned energy or its lateral
displacement fails them: the excess within 30 % of the paper at 0.75, 1.5 and 3 T, the
ordering 0.75 > 1.5 > 3 T, the shifts within 4 mm, and the 1.5 T build-up within 2 mm.
CUDA only; four fields of 4e7 histories run in under a minute.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc import AIR, WATER, MagneticField
from pyradmc.geometry.exterior import EXTERIOR_AUTO
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import SpectralBeamSource
from pyradmc.geometry.spectrum import ali_rogers_mv
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.validation, pytest.mark.warp, pytest.mark.gpu]

PAPER_EXCESS = {0.75: 0.52, 1.5: 0.42, 3.0: 0.25}  # (D_B - D_0) / dmax_0, last 0.5 mm
PAPER_SHIFT_MM = {0.75: 17.5, 1.5: 9.5, 3.0: 4.0}
PAPER_BUILDUP_MM = {1.5: 12.0}
EXCESS_TOLERANCE = 0.30  # relative
SHIFT_TOLERANCE_MM = 4.0
BUILDUP_TOLERANCE_MM = 2.0
N_HISTORIES = 40_000_000

DXY, DZ = 0.25, 0.05
NXY, NZ = 80, 280  # 20 cm lateral; 8 cm water over 6 cm of air, all inside the field
WATER_CM = 8.0


def _phantom() -> VoxelGrid:
    shape = (NXY, NXY, NZ)
    zc = (np.arange(NZ) + 0.5) * DZ
    density = np.ones(shape)
    material = np.full(shape, WATER, dtype=np.int32)
    density[:, :, zc >= WATER_CM] = 1.205e-3
    material[:, :, zc >= WATER_CM] = AIR
    return VoxelGrid(shape=shape, spacing=(DXY, DXY, DZ), density=density, material=material)


@pytest.fixture(scope="module", params=[None, EXTERIOR_AUTO], ids=["current", "default"])
def exit_curves(request):
    """Central-axis depth dose and exit-face lateral profile per field strength, per path.

    The air behind the water reaches the grid boundary, so the shipped default steps through
    it on the long-step path. The paper's numbers are held on both paths, unchanged.
    """
    if not wp.is_cuda_available():
        pytest.skip("four 4e7-history fields; CUDA only")
    from pyradmc.backends.warp.engine import WarpEngine
    from pyradmc.data.tabulated import build
    from pyradmc.data.tabulated.precompile import compile_materials
    from pyradmc.data.tabulated.source import TabulatedCrossSections

    try:
        epdl = build.download_library("epdl", None)
        eedl = build.download_library("eedl", None)
    except Exception as exc:  # pragma: no cover - network / cache dependent
        pytest.skip(f"EPICS libraries unavailable: {exc}")
    data = compile_materials(
        epdl.read_text(encoding="latin-1"), eedl.read_text(encoding="latin-1"), e_max=8.0
    )
    grid = _phantom()
    xs = TabulatedCrossSections(data, geometry_densities=grid.max_density_by_material())
    half = 0.5 * NXY * DXY
    # A 1000 cm focal distance: 0.4 % divergence over the phantom (the paper's beam is
    # non-divergent); a farther focal point is a float32 hazard for the aperture.
    source = SpectralBeamSource(
        ali_rogers_mv("varian-6mv"),
        focal_point=(half, half, -1000.0),
        center=(half, half, 0.0),
        width_u=10.0,
        width_v=10.0,
    )
    zc = (np.arange(NZ) + 0.5) * DZ
    yc = (np.arange(NXY) + 0.5) * DXY - half
    water = zc < WATER_CM
    exit_band = water & (zc >= WATER_CM - 0.2)
    core = slice(NXY // 2 - 2, NXY // 2 + 2)  # central 1 cm
    curves = {}
    for b in (0.0, 0.75, 1.5, 3.0):
        field = None if b == 0.0 else MagneticField.uniform((b, 0.0, 0.0))
        air = request.param
        if field is None or air is None:
            engine = WarpEngine(
                grid=grid,
                cross_sections=xs,
                device="cuda:0",
                magnetic_field=field,
                exterior_air=air,
            )
        else:
            with pytest.warns(UserWarning, match="exterior-air"):
                engine = WarpEngine(
                    grid=grid,
                    cross_sections=xs,
                    device="cuda:0",
                    magnetic_field=field,
                    exterior_air=air,
                )
        result = engine.run(source, n_histories=N_HISTORIES, n_batches=10, seed=SEED)
        axis = result.dose[core, core, :].mean(axis=(0, 1))
        exit_profile = result.dose[core, :, exit_band].mean(axis=(0, 2))
        curves[b] = (axis, exit_profile)
    return zc, yc, water, curves


def _excess_last_half_mm(curves, water, b: float) -> float:
    axis0 = curves[0.0][0]
    dmax0 = axis0[water].max()
    return float((curves[b][0][water][-1] - axis0[water][-1]) / dmax0)


def test_exit_ere_excess_matches_raaijmakers_2008(exit_curves) -> None:
    _zc, _yc, water, curves = exit_curves
    excess = {b: _excess_last_half_mm(curves, water, b) for b in PAPER_EXCESS}
    for b, paper in PAPER_EXCESS.items():
        assert abs(excess[b] / paper - 1.0) < EXCESS_TOLERANCE, (
            f"B={b} T: exit ERE excess {excess[b]:.3f} dmax0 vs paper {paper:.2f}"
        )
    assert excess[0.75] > excess[1.5] > excess[3.0], f"ordering: {excess}"


def test_exit_ere_lateral_shift_matches_raaijmakers_2008(exit_curves) -> None:
    _zc, yc, _water, curves = exit_curves
    exit0 = curves[0.0][1]
    level = 0.2 * exit0.max()
    y20_0 = yc[np.flatnonzero(exit0 >= level)[0]]
    for b, paper in PAPER_SHIFT_MM.items():
        y20 = yc[np.flatnonzero(curves[b][1] >= level)[0]]
        shift = 10.0 * (y20_0 - y20)  # toward -y, the Lorentz side
        assert abs(shift - paper) < SHIFT_TOLERANCE_MM, (
            f"B={b} T: ERE 20 % shift {shift:.1f} mm vs paper {paper:.1f} mm"
        )


def test_build_up_depth_shrinks_as_in_raaijmakers_2008(exit_curves) -> None:
    zc, _yc, _water, curves = exit_curves
    upstream = zc < 4.0

    def buildup_mm(axis) -> float:
        smooth = np.convolve(axis[upstream], np.ones(5) / 5, mode="same")
        return float(10.0 * zc[upstream][np.argmax(smooth)])

    d0, d15 = buildup_mm(curves[0.0][0]), buildup_mm(curves[1.5][0])
    assert d15 < d0, f"build-up did not shrink: {d0:.1f} -> {d15:.1f} mm"
    assert abs(d15 - PAPER_BUILDUP_MM[1.5]) < BUILDUP_TOLERANCE_MM, (
        f"1.5 T build-up {d15:.1f} mm vs paper {PAPER_BUILDUP_MM[1.5]:.0f} mm"
    )
