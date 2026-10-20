"""Shared pieces of the Fano cavity tests: the per-mass isotropic source and the phantom.

Fano's theorem (and its magnetic-field form, Bouchard et al. 2015, PMB 60, 6639,
doi:10.1088/0031-9155/60/17/6639, condition I) needs a charged-particle source that is
spatially uniform **per unit mass** and isotropic in a medium of uniform atomic
properties. :class:`FanoElectronSource` is that source over a voxel grid; the phantom is
two dense walls with a light slab between them, the slab embedded laterally in wall
material so that electrons leaving it sideways over their long range in the light
medium are replaced by wall electrons instead of being lost to the vacuum outside.
Used by the validation-tier Fano-in-field gate and by the integration-tier interface
check at air contrast.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from pyradmc.data.materials import WATER
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import _KIND_TO_IAEA, _UPLOAD_COLUMNS, Primary, Source
from pyradmc.physics.direction import sample_isotropic_direction
from pyradmc.rng import RNGState, uniform

__all__ = ["FanoElectronSource", "fano_phantom"]


@dataclass(frozen=True)
class FanoElectronSource(Source):
    """Monoenergetic electrons, isotropic, uniform per unit mass over a voxel grid.

    Voxels are chosen with probability proportional to their mass, the position
    uniformly inside the voxel, the direction isotropically. ``sample_batch`` is
    vectorised because the device route would otherwise call ``emit`` once per
    history from Python.
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
        # Six draws per history, with a fixed stream position independent of the
        # engine's chunk size (the same contract as the spectral source sampler).
        bitgen = np.random.PCG64(seed)
        bitgen.advance(6 * history_offset)
        u = np.random.Generator(bitgen).random((n, 6))
        flat = np.minimum(np.searchsorted(self._cdf, u[:, 0], side="right"), self._cdf.size - 1)
        ix, iy, iz = np.unravel_index(flat, self.grid.shape)
        ox, oy, oz = self.grid.origin
        sx, sy, sz = self.grid.spacing
        cos_t = 1.0 - 2.0 * u[:, 1]
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
        phi = 2.0 * np.pi * u[:, 2]
        values = {
            "energy": np.full(n, self.energy),
            "x": ox + (ix + u[:, 3]) * sx,
            "y": oy + (iy + u[:, 4]) * sy,
            "z": oz + (iz + u[:, 5]) * sz,
            "ux": sin_t * np.cos(phi),
            "uy": sin_t * np.sin(phi),
            "uz": cos_t,
            "weight": np.ones(n),
        }
        columns = {name: values[name].astype(np.float32) for name in _UPLOAD_COLUMNS}
        particle_type = np.full(n, _KIND_TO_IAEA["electron"], dtype=np.int32)
        return {"particle_type": particle_type, **columns}


def fano_phantom(
    gap_cm: float,
    gap_density: float,
    *,
    wall_cm: float = 1.5,
    buffer_cm: float = 0.7,
    margin_cm: float = 1.0,
    nxy: int = 16,
    dxy: float = 0.5,
    dz: float = 0.1,
) -> tuple[VoxelGrid, slice, slice, slice, slice]:
    """Walls, a light cavity between them, water around the cavity laterally.

    Returns the grid and the slices ``(core, upstream, gap, downstream)``: the central
    lateral ROI, the upstream-wall depth range (excluding ``buffer_cm`` nearest the
    open face), the cavity, and the downstream-wall range. ``wall_cm`` exceeds 1.4
    CSDA ranges at 1 MeV (0.44 cm); ``margin_cm`` of water around the cavity exceeds
    two wall ranges, so lateral leakage from the cavity is replaced, not lost.
    """
    n_wall = round(wall_cm / dz)
    n_gap = round(gap_cm / dz)
    nz = 2 * n_wall + n_gap
    shape = (nxy, nxy, nz)
    density = np.ones(shape)
    n_margin = round(margin_cm / dxy)
    lateral = slice(n_margin, nxy - n_margin)
    density[lateral, lateral, n_wall : n_wall + n_gap] = gap_density
    grid = VoxelGrid(
        shape=shape,
        spacing=(dxy, dxy, dz),
        density=density,
        material=np.full(shape, WATER, dtype=np.int32),
    )
    n_buffer = round(buffer_cm / dz)
    core = slice(nxy // 2 - 2, nxy // 2 + 2)
    upstream = slice(n_buffer, n_wall)
    gap = slice(n_wall, n_wall + n_gap)
    downstream = slice(n_wall + n_gap, nz - n_buffer)
    return grid, core, upstream, gap, downstream
