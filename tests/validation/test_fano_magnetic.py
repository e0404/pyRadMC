"""Fano cavity test in a static magnetic field.

Fano's theorem fails in a magnetic field in general — the Lorentz force in
mass-thickness coordinates depends on the density, so the charged-particle fluence
is no longer density independent. Bouchard, de Pooter, Bielajew and Duane (2015,
PMB 60, 6639, doi:10.1088/0031-9155/60/17/6639) prove the one condition under which
it still holds for *any* field: a spatially uniform **and isotropic** charged-particle
source, per unit mass, in a medium of uniform atomic properties. Under it the fluence
is uniform and isotropic for any B, and the dose per unit mass is the same everywhere,
field or not. The recipe is the one Malkov & Rogers (2016, Med. Phys. 43, 4447,
doi:10.1118/1.4954318, sec. 2.G) used to validate EGSnrc: two dense walls with a gap
of the same material at a thousandth of the density, the field parallel to the
interfaces (the harder orientation), monoenergetic electrons emitted isotropically and
uniformly per unit mass, and the dose per unit mass compared between gap and walls.
Their cylinder is sized to the electron range *in the gas*; here the light slab is a
cavity embedded laterally in water instead (see ``_phantom``), which keeps the
statistics on a small grid.

This is the test that bites where the stepping approximations do: electrons entering
the gap gyrate back on a radius far below the gap's electron range, the gyration cap
binds on every substep there, and any bias in the chord geometry, the step cap or the
boundary handling shows up as a gap-to-wall dose ratio off unity (Simiele & DeWerd 2018,
PMB 63, 235012, found 7-11 % under-response in exactly this configuration before fixing
Geant4's step limits).

The gate here is **1 %**, not EGSnrc's 0.1 %: bremsstrahlung photons are transported
analogously and leave the phantom, lowering every region's dose by a fraction of the
radiative yield, and the yield's spatial distribution is not perfectly uniform near the
walls' outer faces. The field-free run is held to the same gate, which pins the test
itself. Tightening it needs a photon-killing test instrument (AGENTS.md 2.10), which
is a maintainer decision.

**Gap density.** The gated case is a 2 cm gap at 0.1 of the wall density (lung-like),
not EGSnrc's 0.001. At a thousandfold contrast the engine fails Fano *without any
field* — measured 2026-10-17: gap-to-wall ratio 5.1 and 2.9 for 0.1 and 0.2 cm
air-density gaps at 0.5 MeV — because the random hinge's second chord may cross a
voxel face after the deflection, so a wall substep's energy loss is filed into the
light voxel: a fixed energy per unit interface area, hence a dose excess scaling as
one over (gap density x thickness) and with the substep fraction (four times smaller
under the Gaussian hinge's 0.05 than under GS 0.20). At 0.01 of the wall density and
2 cm the same artifact is +1.0 % (0.5 MeV) and +3.0 % (1 MeV), field-free, so that
ratio is not gated either; at 0.1 it is below 0.3 %. The artifact is the residual of
the interface spike docs/decisions.md records as reduced from 1.95x to under 1.4x at
CT contrast, and it predates the field: a stepping limitation, not a field one, so the
air-density cases stay here as *strict expected failures* and must be promoted when
the boundary crossing is made exact. At 0.1 density the gyroradius (1.9 mm at
0.5 MeV, 1.5 T) is still far below the gap's electron range (1.8 cm), and the gyration
cap binds on every gap substep, so the field physics is fully exercised. Measured at
adoption: 0.994 / 0.994 / 0.998 (0.5 MeV) and 0.994 / 0.995 / 1.002 (1 MeV) at
0 / 0.35 / 1.5 T, each +- 0.002 — the field moves the ratio by at most 0.7 %.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.field import MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import _KIND_TO_IAEA, _UPLOAD_COLUMNS, Primary, Source
from pyradmc.physics.direction import sample_isotropic_direction
from pyradmc.rng import RNGState, uniform
from tests.conftest import SEED

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.validation, pytest.mark.warp]

GATE = 0.01  # gap-to-wall dose-per-mass ratio, |ratio - 1|
AIR_LIKE = 1.0e-3  # fails field-free (see module docstring); kept as strict xfail
WALL_CM = 1.5  # > 1.4 CSDA ranges at 1 MeV (0.44 cm)
BUFFER_CM = 0.7  # wall depth excluded from the ROI: within range of the open z faces
NXY, DXY = 16, 0.5  # 8 x 8 cm laterally; the ROI is the central 2 x 2 cm
MARGIN_CM = 1.0  # water around the light cavity, > 2 wall ranges at 1 MeV
DZ = 0.1
FIELDS_TESLA = (0.0, 0.35, 1.5)


@dataclass(frozen=True)
class FanoElectronSource(Source):
    """Monoenergetic electrons, isotropic, uniform per unit mass over a voxel grid.

    The Bouchard et al. condition I source. Voxels are chosen with probability
    proportional to their mass, the position uniformly inside the voxel, the direction
    isotropically. ``sample_batch`` is vectorised because the device route would
    otherwise call ``emit`` once per history from Python.
    """

    grid: VoxelGrid
    energy: float
    _cdf: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mass = np.asarray(self.grid.density, dtype=np.float64).ravel()
        cdf = np.cumsum(mass)
        object.__setattr__(self, "_cdf", cdf / cdf[-1])

    @property
    def max_energy(self) -> float:
        return self.energy

    def _voxel(self, u: float) -> tuple[int, int, int]:
        flat = min(bisect.bisect_right(self._cdf, u), self._cdf.size - 1)
        ix, iy, iz = (int(i) for i in np.unravel_index(flat, self.grid.shape))
        return ix, iy, iz

    def emit(self, rng_state: RNGState) -> Primary:
        ix, iy, iz = self._voxel(uniform(rng_state))
        ox, oy, oz = self.grid.origin
        sx, sy, sz = self.grid.spacing
        x = ox + (ix + uniform(rng_state)) * sx
        y = oy + (iy + uniform(rng_state)) * sy
        z = oz + (iz + uniform(rng_state)) * sz
        ux, uy, uz = sample_isotropic_direction(rng_state)
        return Primary(self.energy, x, y, z, ux, uy, uz, kind="electron")

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, npt.NDArray[Any]]:
        rng = np.random.default_rng([seed, history_offset])
        flat = np.minimum(
            np.searchsorted(self._cdf, rng.random(n), side="right"), self._cdf.size - 1
        )
        ix, iy, iz = np.unravel_index(flat, self.grid.shape)
        ox, oy, oz = self.grid.origin
        sx, sy, sz = self.grid.spacing
        cos_t = 1.0 - 2.0 * rng.random(n)
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
        phi = 2.0 * np.pi * rng.random(n)
        values = {
            "energy": np.full(n, self.energy),
            "x": ox + (ix + rng.random(n)) * sx,
            "y": oy + (iy + rng.random(n)) * sy,
            "z": oz + (iz + rng.random(n)) * sz,
            "ux": sin_t * np.cos(phi),
            "uy": sin_t * np.sin(phi),
            "uz": cos_t,
            "weight": np.ones(n),
        }
        columns = {name: values[name].astype(np.float32) for name in _UPLOAD_COLUMNS}
        particle_type = np.full(n, _KIND_TO_IAEA["electron"], dtype=np.int32)
        return {"particle_type": particle_type, **columns}


def _phantom(gap_cm: float, gap_density: float) -> tuple[VoxelGrid, slice, slice, slice]:
    """Walls, a light slab between them, and water around the slab laterally.

    The light slab is a cavity, not a full layer: it stops ``MARGIN_CM`` short of the
    phantom's lateral faces and water fills the rest. In a light medium the electron
    range is tens of centimetres, so a slab open to the lateral vacuum faces loses
    electrons sideways with nothing replacing them — a 17-23 % field-free deficit at a
    0.01 density ratio in an 8 cm wide phantom, measured before this change. Embedded
    in water, every electron leaving the cavity sideways is replaced by one from the
    water, exactly as Fano's uniform-per-mass source requires, and only the outer
    vacuum faces (all more than a wall range from the ROI) break the symmetry.
    """
    n_wall = round(WALL_CM / DZ)
    n_gap = round(gap_cm / DZ)
    nz = 2 * n_wall + n_gap
    shape = (NXY, NXY, nz)
    density = np.ones(shape)
    n_margin = round(MARGIN_CM / DXY)
    lateral = slice(n_margin, NXY - n_margin)
    density[lateral, lateral, n_wall : n_wall + n_gap] = gap_density
    grid = VoxelGrid(
        shape=shape,
        spacing=(DXY, DXY, DZ),
        density=density,
        material=np.full(shape, WATER, dtype=np.int32),
    )
    n_buffer = round(BUFFER_CM / DZ)
    upstream = slice(n_buffer, n_wall)
    gap = slice(n_wall, n_wall + n_gap)
    downstream = slice(n_wall + n_gap, nz - n_buffer)
    return grid, upstream, gap, downstream


def _device() -> str:
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _region_dose(result, lateral: slice, depth: slice) -> tuple[float, float]:
    """Mean dose per unit mass over a region and its (quadrature) sigma."""
    d = result.dose[lateral, lateral, depth]
    s = result.dose_sigma[lateral, lateral, depth]
    return float(d.mean()), float(np.sqrt((s * s).sum()) / d.size)


AIR_LIKE_FAILS = pytest.mark.xfail(
    strict=True,
    reason="field-free Fano violation at 1e-3 density contrast from the post-hinge face "
    "crossing (module docstring); promote when the boundary crossing is made exact",
)


@pytest.mark.parametrize(
    ("gap_cm", "gap_density"),
    [
        (2.0, 0.1),
        pytest.param(0.2, AIR_LIKE, marks=AIR_LIKE_FAILS),
        pytest.param(2.0, AIR_LIKE, marks=AIR_LIKE_FAILS),
    ],
    ids=["2cm-rho0.1", "0.2cm-air", "2cm-air"],
)
@pytest.mark.parametrize("energy", [0.5, 1.0])
def test_dose_per_mass_is_uniform_across_the_gap(
    gap_cm: float, gap_density: float, energy: float
) -> None:
    from pyradmc.backends.warp.engine import WarpEngine

    grid, upstream, gap, downstream = _phantom(gap_cm, gap_density)
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = FanoElectronSource(grid, energy)
    core = slice(NXY // 2 - 2, NXY // 2 + 2)
    device = _device()
    if not device.startswith("cuda"):
        pytest.skip("the Fano-in-field gate needs ~1e8 electron histories per field; CUDA only")
    n_histories = 40_000_000 if gap_density >= 0.01 else 120_000_000
    report = []
    for b in FIELDS_TESLA:
        fld = None if b == 0.0 else MagneticField.uniform((b, 0.0, 0.0))
        engine = WarpEngine(grid=grid, cross_sections=xs, device=device, magnetic_field=fld)
        result = engine.run(source, n_histories=n_histories, n_batches=10, seed=SEED)
        wall_up, s_up = _region_dose(result, core, upstream)
        in_gap, s_gap = _region_dose(result, core, gap)
        wall_down, s_down = _region_dose(result, core, downstream)
        wall = 0.5 * (wall_up + wall_down)
        ratio = in_gap / wall
        sigma = ratio * np.hypot(s_gap / in_gap, 0.5 * np.hypot(s_up, s_down) / wall)
        walls = wall_up / wall_down
        report.append(f"B={b} T: gap/wall={ratio:.4f} +- {sigma:.4f}, up/down walls={walls:.4f}")
        assert sigma < GATE / 3.0, f"statistics too poor to gate at {GATE}: {report[-1]}"
        assert abs(ratio - 1.0) < GATE, (
            f"Fano violated: {report[-1]} (gap {gap_cm} cm, {energy} MeV)"
        )
        assert abs(walls - 1.0) < GATE, (
            f"walls differ: {report[-1]} (gap {gap_cm} cm, {energy} MeV)"
        )
    print("\n".join(report))
