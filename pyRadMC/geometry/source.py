"""Primary sources and the interface a user implements to define their own.

Host-side source models; each ``emit`` draws from the per-history RNG state so that a
history's primary is part of its reproducible stream. Two ABCs define the contract:
:class:`Source` (open field, ``run``) and :class:`BeamletSource` (Dij, ``run_dij``);
they are disjoint because their ``emit`` signatures differ. Both give a source **two
ways onto a device backend** (see :mod:`pyRadMC.backends.warp.engine`):

- the *simple* route — the default ``sample_batch`` host-samples the primaries into
  column arrays a backend uploads; it works for any source with an ``emit``;
- the *advanced* route — the source exposes an optional ``@wp.func`` sampler
  (``warp_sampler`` / ``warp_beamlet_sampler``) the Warp engine wraps into a generator
  kernel, so the primaries are generated in-kernel with no host round trip.

Monoenergetic beam sources here draw from neither route's data — they are analytic.
Spectra enter via the phase-space source (per-record energy) or a user source.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

import numpy as np
import numpy.typing as npt

from pyRadMC.rng import RNGState, uniform

__all__ = [
    "BeamletGridSource",
    "BeamletSource",
    "ParallelBeamSource",
    "PencilBeamSource",
    "Primary",
    "Source",
]

# Primary.kind -> IAEA particle code (photon 1, electron 2, positron 3); the pre-sampled
# upload columns carry this so a device backend can split a mixed batch into its
# per-kind transport queues. ``None`` means "an unqualified primary" -> photon.
_KIND_TO_IAEA: dict[str | None, int] = {None: 1, "photon": 1, "electron": 2, "positron": 3}

# The upload columns a device backend consumes (all float32 but ``particle_type``);
# the shared contract between ``sample_batch`` and ``generate_from_upload``.
_UPLOAD_COLUMNS: tuple[str, ...] = ("energy", "x", "y", "z", "ux", "uy", "uz", "weight")


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


def _presample(
    emit_one: Callable[[RNGState], Primary], seed: int, history_offset: int, n: int
) -> dict[str, npt.NDArray[Any]]:
    """Host-sample ``n`` primaries into the device upload columns.

    Calls ``emit_one`` once per history on the same per-history ``HostRNG`` stream the
    reference backend uses (``init_state(seed, history_index)``), so the pre-sampled
    primaries are exactly the ones ``emit`` produces. The shared body of the default
    ``sample_batch``/``sample_beamlet_batch``; a source with a vectorized sampler
    (phase space) overrides those for speed.
    """
    from pyRadMC.rng.host import HostRNG

    rng = HostRNG()
    columns = {name: np.empty(n, dtype=np.float32) for name in _UPLOAD_COLUMNS}
    particle_type = np.empty(n, dtype=np.int32)
    for i in range(n):
        p = emit_one(rng.init_state(seed, history_offset + i))
        columns["energy"][i] = p.energy
        columns["x"][i] = p.x
        columns["y"][i] = p.y
        columns["z"][i] = p.z
        columns["ux"][i] = p.ux
        columns["uy"][i] = p.uy
        columns["uz"][i] = p.uz
        columns["weight"][i] = p.weight
        particle_type[i] = _KIND_TO_IAEA[p.kind]
    return {"particle_type": particle_type, **columns}


class Source(ABC):
    """Interface for an open-field primary source (consumed by ``Engine.run``).

    Implement :meth:`emit` and :attr:`max_energy` and the reference backend transports
    it. For a device backend, the default :meth:`sample_batch` provides the *simple*
    host-pre-sampling route for free; override :attr:`warp_sampler` with a ``@wp.func``
    for the *advanced* in-kernel route. See the module docstring.
    """

    #: Optional ``@wp.func`` for the advanced Warp route; ``None`` selects pre-sampling.
    #: Signature ``(history_index: int, state) -> (kind, energy, x, y, z, ux, uy, uz,
    #: weight)``. Left untyped because it is a Warp object the core never imports.
    warp_sampler: ClassVar[Any] = None

    @property
    @abstractmethod
    def max_energy(self) -> float:
        """Highest primary energy in MeV, for cross-section table sizing."""

    @abstractmethod
    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one primary, drawing from the per-history RNG ``state``."""

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, npt.NDArray[Any]]:
        """Host-sample histories ``[history_offset, history_offset + n)`` into columns.

        Returns the device upload columns (``particle_type`` plus
        :data:`_UPLOAD_COLUMNS`). The default calls :meth:`emit` per history; override
        for a vectorized sampler.
        """
        return _presample(self.emit, seed, history_offset, n)


class BeamletSource(ABC):
    """Interface for a beamlet-resolved source (consumed by ``Engine.run_dij``).

    Like :class:`Source` but every emission is tagged by a beamlet index: the Dij
    assembles one dose column per beamlet. Implement :meth:`emit`, :meth:`n_beamlets`
    and :attr:`max_energy`; :meth:`sample_beamlet_batch` is the default simple route
    and :attr:`warp_beamlet_sampler` the advanced one. (Beamlet *geometry* — a
    rectangle, a lattice, an arbitrary aperture — is the source's private business; the
    engine only ever asks for the count and per-beamlet emissions.)
    """

    #: Optional ``@wp.func`` for the advanced Warp Dij route; ``None`` selects
    #: pre-sampling. Signature ``(beamlet: int, within_index: int, state) -> (...)``.
    warp_beamlet_sampler: ClassVar[Any] = None

    @property
    @abstractmethod
    def max_energy(self) -> float:
        """Highest primary energy in MeV, for cross-section table sizing."""

    @property
    @abstractmethod
    def n_beamlets(self) -> int:
        """Number of beamlets whose columns the Dij will hold."""

    @abstractmethod
    def emit(self, beamlet: int, rng_state: RNGState) -> Primary:
        """Emit one primary for ``beamlet``, drawing from the per-history RNG ``state``."""

    def sample_beamlet_batch(
        self, seed: int, history_offset: int, n: int, beamlet: int
    ) -> dict[str, npt.NDArray[Any]]:
        """Host-sample ``n`` primaries of one ``beamlet`` into device upload columns.

        The caller (the Dij engine) chooses ``seed``/``history_offset`` to realize the
        correlated-sampling history mapping; this just emits that beamlet's primaries.
        The default calls :meth:`emit`; override for a vectorized sampler.
        """
        return _presample(lambda state: self.emit(beamlet, state), seed, history_offset, n)


@dataclass(frozen=True)
class PencilBeamSource(Source):
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

    @property
    def max_energy(self) -> float:
        """The single beam energy."""
        return self.energy

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
class ParallelBeamSource(Source):
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

    @property
    def max_energy(self) -> float:
        """The single beam energy."""
        return self.energy

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one primary at a uniform position in the field; consumes two uniforms."""
        x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * uniform(rng_state)
        y = self.y_range[0] + (self.y_range[1] - self.y_range[0]) * uniform(rng_state)
        return Primary(self.energy, x, y, self.z, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class BeamletGridSource(BeamletSource):
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
    def max_energy(self) -> float:
        """The single beam energy."""
        return self.energy

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
