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

__all__ = ["ParallelBeamSource", "PencilBeamSource", "Primary"]


class Primary(NamedTuple):
    """One emitted primary photon."""

    energy: float
    x: float
    y: float
    z: float
    ux: float
    uy: float
    uz: float


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
        object.__setattr__(
            self, "direction", tuple(c / norm for c in self.direction)
        )

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
