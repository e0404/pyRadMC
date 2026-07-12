"""Primary photon sources.

Host-side source models; each ``emit`` draws from the per-history RNG state so that a
history's primary is part of its reproducible stream. Monoenergetic only in Phase 0 —
spectra are a later, data-driven addition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from pyRadMC.rng import RNGState, uniform

__all__ = ["BeamletGridSource", "ParallelBeamSource", "PencilBeamSource", "Primary"]


class Primary(NamedTuple):
    """One emitted primary particle.

    ``kind`` and ``weight`` are additive with defaults so the monoenergetic beam
    sources — which emit unit-weight photons and construct ``Primary`` positionally
    with the first seven fields — are unchanged. ``kind`` is ``None`` for those
    sources, meaning "defer to the engine's ``primary_kind`` argument"; a
    phase-space source sets it per record ("photon", "electron", "positron"). A
    ``weight`` other than 1.0 is the statistical weight a phase-space record carries.
    """

    energy: float
    x: float
    y: float
    z: float
    ux: float
    uy: float
    uz: float
    kind: str | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class PencilBeamSource:
    """Zero-width monoenergetic beam from a fixed point along a fixed direction.

    The direction is normalized at construction; a non-unit direction here would
    silently stretch every sampled path length.
    """

    energy: float
    position: tuple[float, float, float]
    direction: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Validate energy and normalize the direction."""
        if self.energy <= 0.0:
            raise ValueError(f"non-positive energy {self.energy} MeV")
        norm = math.sqrt(sum(c * c for c in self.direction))
        if norm == 0.0:
            raise ValueError("zero direction vector")
        object.__setattr__(self, "direction", tuple(c / norm for c in self.direction))

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit the (deterministic) primary; consumes no random numbers."""
        return Primary(
            self.energy,
            self.position[0],
            self.position[1],
            self.position[2],
            self.direction[0],
            self.direction[1],
            self.direction[2],
        )


@dataclass(frozen=True)
class ParallelBeamSource:
    """Broad parallel beam along +z, uniform over a rectangular field at plane z.

    The broad-beam geometry of the Phase 0 buildup test: uniform fluence over
    ``x_range`` x ``y_range``, all photons travelling in +z.
    """

    energy: float
    z: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate energy and field extents."""
        if self.energy <= 0.0:
            raise ValueError(f"non-positive energy {self.energy} MeV")
        if self.x_range[1] <= self.x_range[0] or self.y_range[1] <= self.y_range[0]:
            raise ValueError("empty field")

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one primary at a uniform position in the field; consumes two uniforms."""
        x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * uniform(rng_state)
        y = self.y_range[0] + (self.y_range[1] - self.y_range[0]) * uniform(rng_state)
        return Primary(self.energy, x, y, self.z, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class BeamletGridSource:
    """Parallel beamlet lattice along +z: an ``n_x`` x ``n_y`` tiling of the field.

    The Phase 3 Dij source. Each beamlet is one rectangle of the tiling, indexed
    x-major: ``j = jx * n_y + jy``. Which beamlet a history feeds is the *caller's*
    decision — the engines derive it deterministically from the history index
    (stratified sampling), so per-beamlet history counts are exact rather than
    multinomial. ``emit`` then places the primary uniformly *within* that beamlet,
    consuming exactly the two uniforms :class:`ParallelBeamSource` consumes for the
    whole field; a 1x1 lattice is therefore bit-identical to the open field on a
    given target (test-pinned).

    Beamlets partition the primary fluence and transport is linear in the source,
    so scoring each history's whole family into its beamlet's column decomposes
    the open-field dose exactly — no crosstalk approximation. (A phase-space
    source would break unique beamlet ownership; that is a Phase 5 concern.)
    """

    energy: float
    z: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    n_x: int
    n_y: int

    def __post_init__(self) -> None:
        """Validate energy, field extents, and lattice shape."""
        if self.energy <= 0.0:
            raise ValueError(f"non-positive energy {self.energy} MeV")
        if self.x_range[1] <= self.x_range[0] or self.y_range[1] <= self.y_range[0]:
            raise ValueError("empty field")
        if self.n_x < 1 or self.n_y < 1:
            raise ValueError(f"lattice must be at least 1x1, got {self.n_x}x{self.n_y}")

    @property
    def n_beamlets(self) -> int:
        """Number of beamlets in the lattice."""
        return self.n_x * self.n_y

    def beamlet_bounds(self, beamlet: int) -> tuple[float, float, float, float]:
        """Rectangle ``(x_lo, x_hi, y_lo, y_hi)`` of one beamlet.

        Edges are computed by linear interpolation between the field bounds (never
        by accumulating widths), so the outer edges of the lattice are exactly the
        field bounds and shared edges are exactly equal between neighbours.
        """
        if not 0 <= beamlet < self.n_beamlets:
            raise IndexError(f"beamlet {beamlet} outside lattice of {self.n_beamlets}")
        jx, jy = divmod(beamlet, self.n_y)
        return (
            self._edge(self.x_range, jx, self.n_x),
            self._edge(self.x_range, jx + 1, self.n_x),
            self._edge(self.y_range, jy, self.n_y),
            self._edge(self.y_range, jy + 1, self.n_y),
        )

    @staticmethod
    def _edge(bounds: tuple[float, float], i: int, n: int) -> float:
        lo, hi = bounds
        if i == 0:
            return lo
        if i == n:
            return hi
        return lo + (hi - lo) * (i / n)

    def emit(self, beamlet: int, rng_state: RNGState) -> Primary:
        """Emit one primary uniformly within ``beamlet``; consumes two uniforms.

        The draw order (x, then y) and count match :class:`ParallelBeamSource.emit`
        so the 1x1 lattice bit-equivalence holds.
        """
        x_lo, x_hi, y_lo, y_hi = self.beamlet_bounds(beamlet)
        x = x_lo + (x_hi - x_lo) * uniform(rng_state)
        y = y_lo + (y_hi - y_lo) * uniform(rng_state)
        return Primary(self.energy, x, y, self.z, 0.0, 0.0, 1.0)
