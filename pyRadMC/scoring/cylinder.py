"""Cylindrical dose scoring: depth bins by radial shells, for pencil-beam kernels.

The geometry a mono-energetic pencil-beam kernel is defined on: a narrow beam along
+z entering a homogeneous medium, dose binned by depth ``z`` and by radius about the
beam axis (Mackie et al., Med. Phys. 12(2):188-196, 1985, doi:10.1118/1.595774;
Ahnesjo, Med. Phys. 16(4):577-592, 1989, doi:10.1118/1.596360).

**This is a scoring geometry only.** Transport runs on the rectilinear
:class:`~pyRadMC.geometry.grid.VoxelGrid` exactly as always — the RNG streams,
Woodcock tracking and stepping never see this class, so the physics is invariant to
choosing it (the same contract :class:`~pyRadMC.scoring.grid.ScoringGrid` carries,
and it is test-pinned the same way). The rectilinear-geometry rule of AGENTS.md 6 is
about what the engine *transports* through, and that is untouched.

The phantom is therefore a water **box** that surrounds the binned cylinder, not a
cylinder with vacuum outside: the kernel is defined in a semi-infinite medium, and a
box that circumscribes the scored region keeps the outermost shell under full
lateral scatter conditions instead of making it scatter-deficient. Deposits that
land in the box but outside the shells go to the engine's *unscored* energy bucket,
never into an edge bin, so ``emitted == deposited + unscored + escaped`` stays exact.

Why shells rather than lateral voxels: the distribution is axially symmetric, so a
shell averages over the whole azimuth. That is a variance reduction of the *estimator*,
not of the transport — it changes no expectation — and it is what makes a kernel
resolvable at history counts a Cartesian grid would need orders more for.

**Per-shell mass is analytic and exact:** ``m = rho * pi * (r_out^2 - r_in^2) * dz``.
It is not an overlap rebin of the transport grid, because an annulus is not separable
over Cartesian axes and no exact separable rebin exists. That is why
:meth:`CylindricalScoringGrid.for_grid` *refuses* a medium that is not uniform over
the binned region rather than shipping a silent approximation; a heterogeneous
phantom wants :class:`~pyRadMC.scoring.grid.ScoringGrid`.

Lengths are in cm, masses in g.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from pyRadMC.geometry.cylinder import (
    cylinder_contains,
    cylinder_radius_squared,
    edge_bin_index,
)
from pyRadMC.geometry.grid import VoxelGrid

__all__ = [
    "CylindricalScoringGrid",
    "common_bin_divisor",
    "geometric_edges",
    "graded_edges",
    "uniform_edges",
]


def uniform_edges(r_max: float, n_shells: int) -> np.ndarray:
    """Equal-thickness shell edges from 0 to ``r_max``: ``n_shells + 1`` radii in cm.

    Simple, and adequate when the quantity of interest is integral rather than the
    near-axis gradient. For a pencil kernel prefer :func:`geometric_edges`, which
    spends its bins where the dose actually varies.
    """
    if n_shells < 1:
        raise ValueError(f"a cylindrical scorer needs at least one shell, got {n_shells}")
    if r_max <= 0.0:
        raise ValueError(f"r_max must be positive, got {r_max}")
    return np.linspace(0.0, r_max, n_shells + 1, dtype=np.float64)


def geometric_edges(r_max: float, n_shells: int, r_min: float) -> np.ndarray:
    """Shell edges that resolve the core: a central disc, then equal ratios to ``r_max``.

    A pencil-beam kernel spans several decades in dose between the axis and the
    scatter tail, and essentially all of the structure sits in the first few
    millimetres. Equal-ratio shells put a constant *relative* resolution everywhere,
    which is the natural binning for a quantity that falls roughly as a power law.

    Geometric spacing cannot start at zero, so the innermost bin is the full disc
    ``[0, r_min)`` and the remaining ``n_shells - 1`` bins are geometric from
    ``r_min`` to ``r_max``. Returns ``n_shells + 1`` radii in cm.

    Bin choice is deliberately the caller's: it is a readout resolution, not an
    accuracy-defining default (AGENTS.md 2.8), and no value of it changes transport.
    """
    if n_shells < 2:
        raise ValueError(
            f"geometric edges need at least two shells (a central disc plus one "
            f"geometric shell), got {n_shells}"
        )
    if r_max <= 0.0:
        raise ValueError(f"r_max must be positive, got {r_max}")
    if not 0.0 < r_min < r_max:
        raise ValueError(f"need 0 < r_min < r_max, got r_min={r_min}, r_max={r_max}")
    return np.concatenate(
        ([0.0], np.geomspace(r_min, r_max, n_shells, dtype=np.float64)),
    )


def graded_edges(segments: Sequence[tuple[float, float, float]]) -> np.ndarray:
    """Edges from contiguous ``(start, stop, step)`` segments of differing resolution.

    The depth binning a pencil-beam kernel database wants: fine through the build-up
    region, where the curve has all its structure, and coarse in the slowly varying
    tail, so that bins are spent where the gradient is rather than uniformly. For
    example, the classic 0-32 cm water schedule

    ``[(0.0, 0.5, 0.010), (0.5, 2.0, 0.025), (2.0, 6.0, 0.25), (6.0, 32.0, 1.0)]``

    yields 152 bins where a uniform 0.010 cm grid would need 3200.

    Every segment must start where the previous one stopped and span a whole number
    of steps. A segment that does not is refused rather than rounded: absorbing the
    remainder into a short last bin would misplace every edge downstream of it, and
    the bin a dose lands in is not a detail that should be decided silently.
    """
    if not segments:
        raise ValueError("need at least one segment")
    edges: list[float] = [float(segments[0][0])]
    for start, stop, step in segments:
        if step <= 0.0:
            raise ValueError(f"segment step must be positive, got {step}")
        if abs(start - edges[-1]) > 1e-12 * max(1.0, abs(start)):
            raise ValueError(
                f"segments must be contiguous: segment starting at {start} does not "
                f"continue from {edges[-1]}"
            )
        if stop <= start:
            raise ValueError(f"segment stop {stop} must exceed start {start}")
        n = round((stop - start) / step)
        if abs(start + n * step - stop) > 1e-9 * max(1.0, abs(stop)):
            raise ValueError(f"segment {start} to {stop} is not a whole number of steps of {step}")
        edges.extend(start + step * k for k in range(1, n + 1))
    edges[-1] = float(segments[-1][1])  # exact endpoint, not an accumulated sum
    return np.asarray(edges, dtype=np.float64)


def common_bin_divisor(widths: np.ndarray, floor: float) -> float:
    """Largest length that divides every width in ``widths``, or ``floor`` if none does.

    Why divisibility matters: sub-substep deposits are laid at a *fixed* spacing from
    the step start, and step starts are pinned to transport voxel faces, so the whole
    point set is locked to the voxel lattice. If the spacing does not divide the
    scoring bin width, the two beat against each other and the fixed phase turns that
    beat into a standing ripple rather than noise. Measured on a 6 / 15 MeV pencil
    beam over 0.025 cm bins: a 0.010 cm spacing (ratio 2.5) leaves 5.7 / 12.7 percent
    peak-to-trough, while 0.005 cm (ratio 5) leaves 1.3 / 1.7 percent.

    Euclid on floats, with a relative tolerance, because bin schedules are built from
    decimal step sizes and their exact float representations are not commensurate.
    Genuinely incommensurable widths drive the divisor toward zero; ``floor`` bounds
    that, at the cost of leaving some ripple, since arbitrarily fine spacing is
    arbitrarily expensive.
    """
    distinct = np.unique(np.round(np.asarray(widths, dtype=np.float64) / floor)) * floor
    g = float(distinct[0])
    for value in distinct[1:]:
        a, c = g, float(value)
        while c > floor:
            a, c = c, a - c * math.floor(a / c + 1.0e-9)
        g = a
        if g < floor:
            return floor
    return max(g, floor)


@dataclass(frozen=True)
class CylindricalScoringGrid:
    """Depth-by-radial-shell dose bins about a beam axis parallel to z.

    Construct through :meth:`for_grid` (bins spanning a transport phantom, density
    read from it) or :meth:`uniform` (bins and density stated outright). Hand the
    engine a scorer built from the same :class:`~pyRadMC.geometry.grid.VoxelGrid`
    it transports on; a scorer reaching outside that grid would divide real energy
    by grams that are not there, which :meth:`for_grid` refuses up front.

    Bins follow the transport grid's half-open convention on both axes (the
    entrance plane and inner shell surface are inside, the exit plane and outer
    surface are outside), delegated to the shared scalar primitives in
    :mod:`pyRadMC.geometry.cylinder` so host and kernel cannot disagree.

    Attributes
    ----------
    axis
        ``(x, y)`` of the cylinder axis in cm; the axis runs parallel to z.
    depth_edges
        Increasing depth boundaries in cm, ``n_depth + 1`` of them. Bins need not
        be uniform: :func:`graded_edges` builds the fine-through-build-up,
        coarse-in-the-tail schedule a kernel database wants.
    radial_edges
        Increasing shell radii in cm, ``n_shells + 1`` of them, starting at or
        above zero.
    voxel_mass
        Mass per bin in g, shape ``(n_depth, n_shells)`` — the annulus volume
        times the medium density. Named to match
        :class:`~pyRadMC.scoring.grid.ScoringGrid` so the batched scorer and both
        engines consume either geometry through one code path.
    """

    axis: tuple[float, float]
    depth_edges: np.ndarray
    radial_edges: np.ndarray
    voxel_mass: np.ndarray
    _edges_squared: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the binning and precompute the squared edges the lookups use."""
        depth = np.asarray(self.depth_edges, dtype=np.float64)
        if depth.ndim != 1 or depth.size < 2:
            raise ValueError("depth_edges needs at least one depth bin, i.e. two depths")
        if np.any(np.diff(depth) <= 0.0):
            raise ValueError("depth_edges must be strictly increasing")
        edges = np.asarray(self.radial_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("radial_edges needs at least one shell, i.e. two radii")
        if np.any(edges < 0.0):
            raise ValueError("negative radial edge")
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("radial_edges must be strictly increasing")
        if self.voxel_mass.shape != (depth.size - 1, edges.size - 1):
            raise ValueError(
                f"voxel_mass shape {self.voxel_mass.shape} != "
                f"{(depth.size - 1, edges.size - 1)} (n_depth, n_shells)"
            )
        if np.any(self.voxel_mass < 0.0):
            raise ValueError("negative shell mass")
        object.__setattr__(self, "depth_edges", depth)
        object.__setattr__(self, "radial_edges", edges)
        object.__setattr__(self, "_edges_squared", edges**2)

    # --- constructors ------------------------------------------------------

    @classmethod
    def uniform(
        cls,
        density: float,
        axis: tuple[float, float],
        depth_edges: np.ndarray,
        radial_edges: np.ndarray,
    ) -> CylindricalScoringGrid:
        """Bins in a medium of stated uniform ``density`` (g/cm^3); mass analytic.

        The direct constructor, for a phantom whose density the caller already knows.
        :meth:`for_grid` is the same thing with the density taken from the transport
        grid and the coverage checked.
        """
        if density <= 0.0:
            raise ValueError(f"non-positive density {density}")
        edges = np.asarray(radial_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("radial_edges needs at least one shell, i.e. two radii")
        if np.any(edges < 0.0):
            raise ValueError("negative radial edge")
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("radial_edges must be strictly increasing")
        depth = np.asarray(depth_edges, dtype=np.float64)
        if depth.ndim != 1 or depth.size < 2:
            raise ValueError("depth_edges needs at least one depth bin, i.e. two depths")
        if np.any(np.diff(depth) <= 0.0):
            raise ValueError("depth_edges must be strictly increasing")
        # m = rho * pi * (r_out^2 - r_in^2) * dz, with each bin's own dz: the outer
        # product of the annulus areas with the depth thicknesses.
        shell_area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        mass = density * np.outer(np.diff(depth), shell_area)
        return cls(
            axis=axis,
            depth_edges=depth,
            radial_edges=edges,
            voxel_mass=mass,
        )

    @classmethod
    def for_grid(
        cls,
        grid: VoxelGrid,
        radial_edges: np.ndarray,
        axis: tuple[float, float] | None = None,
        depth_edges: np.ndarray | None = None,
    ) -> CylindricalScoringGrid:
        """Bins spanning a transport phantom, density read from it and checked.

        Defaults place the cylinder where a pencil-beam kernel run wants it: the
        axis on the phantom's lateral centre, and depth bins covering the phantom's
        full z extent at its z spacing. Either may be overridden — pass
        :func:`graded_edges` for a depth schedule that follows the build-up.

        Raises
        ------
        ValueError
            If the binned region reaches outside the transport grid (its mass would
            be fictitious), or if the medium it overlays is not uniform in density
            and material (the analytic annulus mass would then be wrong, and no
            exact separable rebin of an annulus exists — use
            :class:`~pyRadMC.scoring.grid.ScoringGrid` for a heterogeneous phantom).
        """
        edges = np.asarray(radial_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("radial_edges needs at least one shell, i.e. two radii")
        hi = grid.upper_corner
        if axis is None:
            axis = (
                0.5 * (grid.origin[0] + hi[0]),
                0.5 * (grid.origin[1] + hi[1]),
            )
        if depth_edges is None:
            n_depth = max(1, math.floor((hi[2] - grid.origin[2]) / grid.spacing[2]))
            depth_edges = grid.origin[2] + grid.spacing[2] * np.arange(
                n_depth + 1, dtype=np.float64
            )
        depth = np.asarray(depth_edges, dtype=np.float64)
        if depth.ndim != 1 or depth.size < 2:
            raise ValueError("depth_edges needs at least one depth bin, i.e. two depths")

        r_max = float(edges[-1])
        depth_origin = float(depth[0])
        depth_hi = float(depth[-1])
        if (
            axis[0] - r_max < grid.origin[0]
            or axis[0] + r_max > hi[0]
            or axis[1] - r_max < grid.origin[1]
            or axis[1] + r_max > hi[1]
            or depth_origin < grid.origin[2]
            or depth_hi > hi[2]
        ):
            raise ValueError(
                f"the scoring cylinder (axis {axis}, r_max {r_max} cm, depth "
                f"[{depth_origin}, {depth_hi}) cm) reaches outside the transport grid "
                f"{grid.origin} to {hi}; its shells would claim mass the phantom does "
                "not have"
            )

        density = _uniform_medium_density(grid, axis, r_max, depth_origin, depth_hi)
        return cls.uniform(
            density=density,
            axis=axis,
            depth_edges=depth,
            radial_edges=edges,
        )

    # --- geometry ----------------------------------------------------------

    @property
    def n_shells(self) -> int:
        """Number of radial shells."""
        return self.radial_edges.size - 1

    @property
    def n_depth(self) -> int:
        """Number of depth bins."""
        return self.depth_edges.size - 1

    @property
    def depth_origin(self) -> float:
        """Position of the entrance plane of the first depth bin, in cm."""
        return float(self.depth_edges[0])

    @property
    def depth_thickness(self) -> np.ndarray:
        """Thickness of each depth bin in cm, ``n_depth`` of them.

        The per-bin ``dz``. Divide an energy-per-bin by this to compare graded bins
        against each other on a per-centimetre footing.
        """
        thickness: np.ndarray = np.diff(self.depth_edges)
        return thickness

    @property
    def shape(self) -> tuple[int, int]:
        """Bin counts as ``(n_depth, n_shells)`` — the shape of ``dose``."""
        return (self.n_depth, self.n_shells)

    @property
    def n_voxels(self) -> int:
        """Number of bins; the length of the flat buffer a device scores into."""
        return self.n_depth * self.n_shells

    @property
    def deposit_resolution_cm(self) -> float:
        """Suggested ``deposit_resolution_cm`` for an engine run on this binning.

        A pencil kernel bins far below the transport voxel scale, which is exactly
        where a half-substep filed as one midpoint deposit prints the voxel lattice
        onto the dose profile. Passing this asks the transport loop to file energy
        finely enough for these bins to be meaningful.

        **The depth axis, not the radial one.** Spreading subdivides a substep
        *along its own direction*, and for the beam this geometry describes that
        direction is the depth axis. The radial profile is resolved by the
        transverse spread of many histories, not by subdividing one step — so the
        innermost geometric shell, which can be micrometres wide, would demand
        hundreds of deposits per step and buy nothing.

        **Half the largest length that divides every depth bin width**, not simply
        the finest bin. A spacing that does not divide the bin width aliases against
        it, and because both are locked to the voxel lattice the beat stands still
        instead of averaging out — see :func:`common_bin_divisor` for the measured
        sizes. Halving it puts at least two deposits in the narrowest bin while
        staying commensurate with all of them.
        """
        widths = self.depth_thickness
        floor = float(widths.min()) / 16.0
        return 0.5 * common_bin_divisor(widths, floor)

    @property
    def depth_upper(self) -> float:
        """Position of the exit plane of the last depth bin, in cm (outside)."""
        return float(self.depth_edges[-1])

    @property
    def depth_centers(self) -> np.ndarray:
        """Depth bin midpoints in cm — the abscissa a depth-dose curve is plotted on."""
        centers: np.ndarray = 0.5 * (self.depth_edges[:-1] + self.depth_edges[1:])
        return centers

    @property
    def radial_centers(self) -> np.ndarray:
        """Area-weighted shell radii in cm: ``sqrt((r_in^2 + r_out^2) / 2)``.

        The radius that halves each shell's area, which is where a smoothly varying
        radial quantity averages to its shell mean — not the arithmetic midpoint,
        which biases outward-falling profiles inward on wide shells.
        """
        centers: np.ndarray = np.sqrt(0.5 * (self._edges_squared[:-1] + self._edges_squared[1:]))
        return centers

    @property
    def shell_volume(self) -> np.ndarray:
        """Volume of each bin in cm^3, shape ``(n_depth, n_shells)``."""
        area = math.pi * (self._edges_squared[1:] - self._edges_squared[:-1])
        return np.outer(self.depth_thickness, area)

    def contains(self, x: float, y: float, z: float) -> bool:
        """Whether the position lies inside the binned region (upper faces excluded)."""
        return cylinder_contains(
            x,
            y,
            z,
            self.axis[0],
            self.axis[1],
            self.depth_origin,
            self.depth_upper,
            float(self._edges_squared[0]),
            float(self._edges_squared[-1]),
        )

    def voxel_index(self, x: float, y: float, z: float) -> tuple[int, int]:
        """``(depth bin, shell)`` containing the position; caller guarantees ``contains``."""
        iz = edge_bin_index(z, self.depth_edges, self.n_depth)
        ir = edge_bin_index(
            cylinder_radius_squared(x, y, self.axis[0], self.axis[1]),
            self._edges_squared,
            self.n_shells,
        )
        return (iz, ir)

    def flat_index(self, x: float, y: float, z: float) -> int:
        """Flat bin index ``iz * n_shells + ir``, or ``-1`` when outside.

        C order, matching ``voxel_mass.reshape(-1)`` and the flat buffer the Warp
        kernels score into. Outside is reported in band so that a caller needs one
        query per deposit rather than a separate containment test.
        """
        if not self.contains(x, y, z):
            return -1
        iz, ir = self.voxel_index(x, y, z)
        return iz * self.n_shells + ir


def _uniform_medium_density(
    grid: VoxelGrid,
    axis: tuple[float, float],
    r_max: float,
    depth_lo: float,
    depth_hi: float,
) -> float:
    """Return the single density of the medium the cylinder overlays, or raise.

    Every transport voxel whose box intersects the cylinder must carry the same
    density and material. The test is the exact box-circle one — the closest point
    of the voxel's lateral footprint to the axis lies within ``r_max`` — so voxels
    that only touch the cylinder's bounding square, but not the cylinder, do not
    make a phantom look heterogeneous when it is not.
    """
    nx, ny, nz = grid.shape
    ox, oy, oz = grid.origin
    sx, sy, sz = grid.spacing

    x_lo = ox + sx * np.arange(nx, dtype=np.float64)
    y_lo = oy + sy * np.arange(ny, dtype=np.float64)
    z_lo = oz + sz * np.arange(nz, dtype=np.float64)

    # Closest point of each voxel's extent to the axis, per lateral axis.
    dx = np.maximum(0.0, np.maximum(x_lo - axis[0], axis[0] - (x_lo + sx)))
    dy = np.maximum(0.0, np.maximum(y_lo - axis[1], axis[1] - (y_lo + sy)))
    lateral = (dx[:, None] ** 2 + dy[None, :] ** 2) < r_max**2
    depth = (z_lo + sz > depth_lo) & (z_lo < depth_hi)
    overlaps = lateral[:, :, None] & depth[None, None, :]

    densities = np.unique(grid.density[overlaps])
    materials = np.unique(grid.material[overlaps])
    if densities.size != 1 or materials.size != 1:
        raise ValueError(
            "cylindrical scoring needs a uniform medium over the binned region "
            f"(found {densities.size} densities and {materials.size} materials there): "
            "the analytic annulus mass is exact only in a uniform medium, and an "
            "annulus admits no exact separable rebin of the transport grid. Use "
            "pyRadMC.scoring.grid.ScoringGrid for a heterogeneous phantom."
        )
    return float(densities[0])
