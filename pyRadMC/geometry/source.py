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
Spectra enter via the phase-space source (per-record energy), the divergent
:class:`SpectralBeamSource`/:class:`SpectralBeamletSource` (a histogram
:class:`~pyRadMC.geometry.spectrum.Spectrum` sampled per history), or a user source.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

import numpy as np
import numpy.typing as npt

from pyRadMC.geometry.spectrum import Spectrum
from pyRadMC.rng import RNGState, uniform

__all__ = [
    "BeamletGridSource",
    "BeamletSource",
    "CompositeBeamletSource",
    "CompositeSource",
    "GaussianSpotBeamSource",
    "GaussianSpotBeamletSource",
    "ParallelBeamSource",
    "PencilBeamSource",
    "Primary",
    "Source",
    "SpectralBeamSource",
    "SpectralBeamletSource",
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
    #: pre-sampling. Signature ``(beamlet: int, within_index: int, state) ->
    #: (energy, x, y, z, ux, uy, uz, weight)`` — a photon with a statistical
    #: weight (1.0 for analog sources; the collimated wrappers attenuate by it),
    #: mirroring what :meth:`emit` returns for the same stream.
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


def _selection_cdf(weights: Sequence[float]) -> npt.NDArray[np.float64]:
    """Build the normalized cumulative weights for choosing a mixture component."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        raise ValueError("a composite source needs at least one component")
    if np.any(w < 0.0):
        raise ValueError("component weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("component weights must sum to a positive value")
    return np.cumsum(w) / total


class CompositeSource(Source):
    """A mixture of open-field sources — the virtual-source-model building block.

    Each history is emitted by one component, chosen with probability proportional to
    its weight (a single uniform draw), then that component's ``emit`` runs. So a beam
    modelled as, e.g., a narrow Gaussian core plus a broad scatter tail is
    ``CompositeSource([(core, 0.85), (tail, 0.15)])``. ``weight`` is the *selection*
    probability, which equals the fluence fraction for unit-weight components; a
    component that itself carries a per-primary weight (a phase space) has that weight
    multiplied on top.

    Composites transport on both backends through the pre-sampling route (they carry no
    ``warp_sampler``); a mixture wanting the in-kernel route writes a single
    ``warp_sampler`` that branches internally.
    """

    def __init__(self, components: Sequence[tuple[Source, float]]) -> None:
        if not components:
            raise ValueError("a composite source needs at least one component")
        self._sources = tuple(source for source, _ in components)
        self._cdf = _selection_cdf([weight for _, weight in components])
        self._max_energy = max(source.max_energy for source in self._sources)

    @property
    def max_energy(self) -> float:
        """Highest energy any component can emit, for cross-section table sizing."""
        return self._max_energy

    def emit(self, rng_state: RNGState) -> Primary:
        """Choose a component by weight (one uniform), then emit from it."""
        index = min(int(np.searchsorted(self._cdf, uniform(rng_state))), len(self._sources) - 1)
        return self._sources[index].emit(rng_state)


class CompositeBeamletSource(BeamletSource):
    """A per-beamlet mixture of beamlet sources — a VSM for beamlet-resolved dose.

    Every component describes the *same* beamlets (identical ``n_beamlets``), so beamlet
    ``j`` is a mixture: :meth:`emit` chooses a component by weight (one uniform) and
    emits that component's beamlet ``j``. Assembling the Dij then gives each beamlet's
    column as the virtual-source-model dose. Like :class:`CompositeSource`, it runs on
    both backends through the pre-sampling Dij route; ``weight`` is the selection
    probability. (The Dij transports each beamlet primary as a unit-weight photon, so
    components should be photon beamlet sources.)
    """

    def __init__(self, components: Sequence[tuple[BeamletSource, float]]) -> None:
        if not components:
            raise ValueError("a composite beamlet source needs at least one component")
        self._sources = tuple(source for source, _ in components)
        counts = {source.n_beamlets for source in self._sources}
        if len(counts) != 1:
            raise ValueError(f"all components must share n_beamlets, got {sorted(counts)}")
        self._n_beamlets = counts.pop()
        self._cdf = _selection_cdf([weight for _, weight in components])
        self._max_energy = max(source.max_energy for source in self._sources)

    @property
    def max_energy(self) -> float:
        """Highest energy any component can emit, for cross-section table sizing."""
        return self._max_energy

    @property
    def n_beamlets(self) -> int:
        """The shared beamlet count of every component."""
        return self._n_beamlets

    def emit(self, beamlet: int, rng_state: RNGState) -> Primary:
        """Choose a component by weight (one uniform), then emit its ``beamlet``."""
        index = min(int(np.searchsorted(self._cdf, uniform(rng_state))), len(self._sources) - 1)
        return self._sources[index].emit(beamlet, rng_state)


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


@dataclass(frozen=True)
class _DivergentFan:
    """Validated fan geometry shared by the spectral sources: focal spot + apertures.

    Everything is **in the engine frame** ((x, y, z) cm, lower-corner origin): a
    caller with gantry/couch angles (the pyRadPlan adapter) rotates the focal point,
    aperture centres and axes into this frame itself, exactly as the CT adapter maps
    image coordinates — no rotation logic lives in the core. Each aperture is the
    rectangle ``center + u*u_axis + v*v_axis`` with ``|u| <= width_u / 2``,
    ``|v| <= width_v / 2``; the axes are normalized here and may be non-orthogonal
    (a parallelogram aperture), but not parallel.
    """

    focal_point: tuple[float, float, float]
    centers: tuple[tuple[float, float, float], ...]
    width_u: float
    width_v: float
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Normalize the axes and validate the fan geometry."""
        if len(self.centers) == 0:
            raise ValueError("a spectral source needs at least one aperture centre")
        if self.width_u < 0.0 or self.width_v < 0.0:
            raise ValueError(
                f"aperture widths must be non-negative, got ({self.width_u}, {self.width_v})"
            )
        u = _normalized(self.u_axis, "u_axis")
        v = _normalized(self.v_axis, "v_axis")
        cross = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        if math.sqrt(sum(c * c for c in cross)) < 1.0e-12:
            raise ValueError("aperture axes are parallel; they must span a plane")
        object.__setattr__(self, "u_axis", u)
        object.__setattr__(self, "v_axis", v)
        object.__setattr__(self, "centers", tuple(tuple(c) for c in self.centers))
        for center in self.centers:
            if all(a == b for a, b in zip(center, self.focal_point, strict=True)):
                raise ValueError(
                    f"aperture centre {center} coincides with the focal point; "
                    "the fan direction is undefined"
                )

    def emit_through(self, spectrum: Spectrum, aperture: int, rng_state: RNGState) -> Primary:
        """Emit one primary from the focal spot through one aperture.

        Consumes exactly four uniforms in a fixed order — two for the energy
        (:meth:`Spectrum.sample_energy`), then the in-aperture u and v offsets —
        so under correlated Dij sampling corresponding histories of every beamlet
        replay the same energy and the same within-aperture offset, and only the
        aperture centre differs. The primary starts *at* the focal point (the
        transport loops fly the vacuum up to the grid), and uniform aperture
        sampling from a point source carries the 1/r^2 divergence for free.
        """
        energy = spectrum.sample_energy(rng_state)
        du = (uniform(rng_state) - 0.5) * self.width_u
        dv = (uniform(rng_state) - 0.5) * self.width_v
        center = self.centers[aperture]
        dx = center[0] + du * self.u_axis[0] + dv * self.v_axis[0] - self.focal_point[0]
        dy = center[1] + du * self.u_axis[1] + dv * self.v_axis[1] - self.focal_point[1]
        dz = center[2] + du * self.u_axis[2] + dv * self.v_axis[2] - self.focal_point[2]
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        return Primary(
            energy,
            self.focal_point[0],
            self.focal_point[1],
            self.focal_point[2],
            dx / norm,
            dy / norm,
            dz / norm,
        )

    def sample_through_batch(
        self, spectrum: Spectrum, aperture: int, seed: int, history_offset: int, n: int
    ) -> dict[str, npt.NDArray[Any]]:
        """Vectorized pre-sampling of ``n`` primaries through one aperture.

        The vectorized sibling of :meth:`emit_through` (the phase-space precedent,
        AGENTS.md 7.2): the four per-history uniforms come from a single
        ``PCG64(seed)`` stream advanced to ``4 * history_offset``, so history ``h``
        always consumes draws ``4h .. 4h + 3`` regardless of chunking
        (chunk-invariant) and independently of the aperture (correlated Dij
        sampling replays the same energy and in-aperture offset in every beamlet).
        This is a *different* stream from ``emit_through``'s per-history spawn, so
        the two backends draw independent primaries — both unbiased estimators of
        the same dose, compared statistically, never bit-wise.
        """
        bitgen = np.random.PCG64(seed)
        bitgen.advance(4 * history_offset)
        u = np.random.Generator(bitgen).random((n, 4))

        energy = spectrum.sample_energies(u[:, 0], u[:, 1])
        du = (u[:, 2] - 0.5) * self.width_u
        dv = (u[:, 3] - 0.5) * self.width_v
        focal = np.asarray(self.focal_point, dtype=np.float64)
        target = (
            np.asarray(self.centers[aperture], dtype=np.float64)
            + du[:, None] * np.asarray(self.u_axis, dtype=np.float64)
            + dv[:, None] * np.asarray(self.v_axis, dtype=np.float64)
        )
        direction = target - focal
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)

        return {
            "particle_type": np.ones(n, dtype=np.int32),  # photons, unit weight
            "energy": energy.astype(np.float32),
            "x": np.full(n, focal[0], dtype=np.float32),
            "y": np.full(n, focal[1], dtype=np.float32),
            "z": np.full(n, focal[2], dtype=np.float32),
            "ux": direction[:, 0].astype(np.float32),
            "uy": direction[:, 1].astype(np.float32),
            "uz": direction[:, 2].astype(np.float32),
            "weight": np.ones(n, dtype=np.float32),
        }


def _normalized(vector: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    """Return the unit vector, rejecting a zero input by name."""
    norm = math.sqrt(sum(c * c for c in vector))
    if norm == 0.0:
        raise ValueError(f"zero {name} vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


class SpectralBeamSource(Source):
    """Divergent polyenergetic open field: a focal spot fanning through one aperture.

    The photon energy is sampled from a histogram
    :class:`~pyRadMC.geometry.spectrum.Spectrum` by CDF inversion; the geometry is a
    point source at ``focal_point`` emitting toward points sampled uniformly in the
    rectangular aperture at the reference plane (see :class:`_DivergentFan` for the
    engine-frame convention and axis semantics). Transports on both backends via the
    default pre-sampling route (no ``warp_sampler``).
    """

    def __init__(
        self,
        spectrum: Spectrum,
        focal_point: tuple[float, float, float],
        center: tuple[float, float, float],
        width_u: float,
        width_v: float,
        u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        self._spectrum = spectrum
        self._fan = _DivergentFan(focal_point, (center,), width_u, width_v, u_axis, v_axis)

    @property
    def max_energy(self) -> float:
        """The spectrum's top bin edge, for cross-section table sizing."""
        return self._spectrum.max_energy

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one primary through the aperture; consumes exactly four uniforms."""
        return self._fan.emit_through(self._spectrum, 0, rng_state)

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, npt.NDArray[Any]]:
        """Vectorized simple-route batch; see :meth:`_DivergentFan.sample_through_batch`."""
        return self._fan.sample_through_batch(self._spectrum, 0, seed, history_offset, n)


class SpectralBeamletSource(BeamletSource):
    """Divergent polyenergetic beamlet fan — the pyRadPlan adapter's Dij source.

    Beamlet ``j`` is the fan from ``focal_point`` through the rectangular aperture
    centred at ``centers[j]`` (all engine-frame; see :class:`_DivergentFan`). The
    beamlet order is the caller's: pyRadPlan hands the centres in its own
    bixel-index order and the Dij columns come back in the same order. All bixels
    share one aperture size, the width at the reference plane the centres lie on.
    Transports on both backends via the default per-beamlet pre-sampling route.
    """

    def __init__(
        self,
        spectrum: Spectrum,
        focal_point: tuple[float, float, float],
        centers: Sequence[tuple[float, float, float]],
        width_u: float,
        width_v: float,
        u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        self._spectrum = spectrum
        self._fan = _DivergentFan(focal_point, tuple(centers), width_u, width_v, u_axis, v_axis)

    @property
    def max_energy(self) -> float:
        """The spectrum's top bin edge, for cross-section table sizing."""
        return self._spectrum.max_energy

    @property
    def n_beamlets(self) -> int:
        """One beamlet per aperture centre."""
        return len(self._fan.centers)

    def emit(self, beamlet: int, rng_state: RNGState) -> Primary:
        """Emit one primary for ``beamlet``; consumes exactly four uniforms."""
        if not 0 <= beamlet < self.n_beamlets:
            raise IndexError(f"beamlet {beamlet} outside fan of {self.n_beamlets}")
        return self._fan.emit_through(self._spectrum, beamlet, rng_state)

    def sample_beamlet_batch(
        self, seed: int, history_offset: int, n: int, beamlet: int
    ) -> dict[str, npt.NDArray[Any]]:
        """Vectorized per-beamlet batch; see :meth:`_DivergentFan.sample_through_batch`.

        The caller keys ``history_offset`` for the correlated/independent mapping;
        the draw stream never sees the beamlet, so correlated sampling replays the
        same energy and in-aperture offset in every beamlet's column (test-pinned).
        """
        if not 0 <= beamlet < self.n_beamlets:
            raise IndexError(f"beamlet {beamlet} outside fan of {self.n_beamlets}")
        return self._fan.sample_through_batch(self._spectrum, beamlet, seed, history_offset, n)


# A uniform draw of exactly zero would send the Box-Muller radius to infinity; the
# clamp changes the spot distribution only below the 1e-12 quantile.
_BOX_MULLER_FLOOR = 1.0e-12


@dataclass(frozen=True)
class _GaussianSpotFan:
    """The divergent fan generalized to a finite 2D-Gaussian focal spot.

    Composes a :class:`_DivergentFan` (whose validation and engine-frame
    conventions apply unchanged) with per-axis spot sigmas at the focal point.
    Primaries start **on** the rectangle at the reference plane — e.g. directly
    upstream of the beam-limiting devices — travelling as if they came in a
    straight line from a point sampled from the Gaussian spot (the standard
    finite-source-size model): downstream collimation then acquires a geometric
    penumbra with no extra machinery, and the transport loops fly whatever vacuum
    remains to the grid. The spot spreads along the *aperture axes* translated to
    the focal point; the two Gaussian components are independent (Box-Muller).
    """

    fan: _DivergentFan
    sigma_u: float
    sigma_v: float

    def __post_init__(self) -> None:
        """Reject negative spot widths (zero is the point-source degeneracy)."""
        if self.sigma_u < 0.0 or self.sigma_v < 0.0:
            raise ValueError(
                f"spot sigma must be non-negative, got ({self.sigma_u}, {self.sigma_v})"
            )

    def emit_from_plane(self, spectrum: Spectrum, aperture: int, rng_state: RNGState) -> Primary:
        """Emit one primary on the plane, aimed from a Gaussian-sampled spot point.

        Consumes **exactly six uniforms in fixed order** — two for the energy,
        two for the in-rectangle offset (identical to
        :meth:`_DivergentFan.emit_through`, so the zero-sigma source replays the
        fan's lines on the same stream), and two Box-Muller draws for the spot,
        consumed even at zero sigma so the draw count never varies (the
        vectorized-batch requirement). Under correlated Dij sampling every
        beamlet replays the same energy, rectangle offset *and* spot point.
        """
        energy = spectrum.sample_energy(rng_state)
        du = (uniform(rng_state) - 0.5) * self.fan.width_u
        dv = (uniform(rng_state) - 0.5) * self.fan.width_v
        radius = math.sqrt(-2.0 * math.log(max(uniform(rng_state), _BOX_MULLER_FLOOR)))
        angle = 2.0 * math.pi * uniform(rng_state)
        spot_u = self.sigma_u * radius * math.cos(angle)
        spot_v = self.sigma_v * radius * math.sin(angle)

        center = self.fan.centers[aperture]
        u_axis, v_axis = self.fan.u_axis, self.fan.v_axis
        px = center[0] + du * u_axis[0] + dv * v_axis[0]
        py = center[1] + du * u_axis[1] + dv * v_axis[1]
        pz = center[2] + du * u_axis[2] + dv * v_axis[2]
        focal = self.fan.focal_point
        sx = focal[0] + spot_u * u_axis[0] + spot_v * v_axis[0]
        sy = focal[1] + spot_u * u_axis[1] + spot_v * v_axis[1]
        sz = focal[2] + spot_u * u_axis[2] + spot_v * v_axis[2]
        dx, dy, dz = px - sx, py - sy, pz - sz
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        return Primary(energy, px, py, pz, dx / norm, dy / norm, dz / norm)

    def sample_from_plane_batch(
        self, spectrum: Spectrum, aperture: int, seed: int, history_offset: int, n: int
    ) -> dict[str, npt.NDArray[Any]]:
        """Vectorized pre-sampling sibling of :meth:`emit_from_plane`.

        One ``PCG64(seed)`` stream advanced to ``6 * history_offset``: history
        ``h`` always consumes draws ``6h .. 6h + 5`` regardless of chunking and
        of the aperture (chunk-invariant, beamlet-blind) — a *different* stream
        from ``emit_from_plane``'s per-history spawn, so the backends draw
        independent primaries and agree statistically, never bit-wise (the
        spectral-source precedent).
        """
        bitgen = np.random.PCG64(seed)
        bitgen.advance(6 * history_offset)
        u = np.random.Generator(bitgen).random((n, 6))

        energy = spectrum.sample_energies(u[:, 0], u[:, 1])
        du = (u[:, 2] - 0.5) * self.fan.width_u
        dv = (u[:, 3] - 0.5) * self.fan.width_v
        radius = np.sqrt(-2.0 * np.log(np.maximum(u[:, 4], _BOX_MULLER_FLOOR)))
        angle = 2.0 * np.pi * u[:, 5]
        spot_u = self.sigma_u * radius * np.cos(angle)
        spot_v = self.sigma_v * radius * np.sin(angle)

        u_axis = np.asarray(self.fan.u_axis, dtype=np.float64)
        v_axis = np.asarray(self.fan.v_axis, dtype=np.float64)
        center = np.asarray(self.fan.centers[aperture], dtype=np.float64)
        focal = np.asarray(self.fan.focal_point, dtype=np.float64)
        plane_point = center + du[:, None] * u_axis + dv[:, None] * v_axis
        spot_point = focal + spot_u[:, None] * u_axis + spot_v[:, None] * v_axis
        direction = plane_point - spot_point
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)

        return {
            "particle_type": np.ones(n, dtype=np.int32),  # photons, unit weight
            "energy": energy.astype(np.float32),
            "x": plane_point[:, 0].astype(np.float32),
            "y": plane_point[:, 1].astype(np.float32),
            "z": plane_point[:, 2].astype(np.float32),
            "ux": direction[:, 0].astype(np.float32),
            "uy": direction[:, 1].astype(np.float32),
            "uz": direction[:, 2].astype(np.float32),
            "weight": np.ones(n, dtype=np.float32),
        }


class GaussianSpotBeamSource(Source):
    """Photons born on a plane rectangle, aimed from a 2D-Gaussian focal spot.

    The simplified head-input source of the BLD workstream: emission happens on
    the rectangle at the reference plane (e.g. directly upstream of the limiting
    devices), each photon travelling as if it originated at the Gaussian spot —
    compose with :class:`~pyRadMC.geometry.collimation.CollimatedSource` (whose
    full-line convention handles the devices downstream of this plane) or feed
    the head pre-solve. ``sigma_u = sigma_v = 0`` degenerates to
    :class:`SpectralBeamSource`'s fan lines, started on the plane. Transports on
    both backends via the vectorized pre-sampling route.
    """

    def __init__(
        self,
        spectrum: Spectrum,
        focal_point: tuple[float, float, float],
        center: tuple[float, float, float],
        width_u: float,
        width_v: float,
        sigma_u: float,
        sigma_v: float,
        u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        self._spectrum = spectrum
        self._spot_fan = _GaussianSpotFan(
            _DivergentFan(focal_point, (center,), width_u, width_v, u_axis, v_axis),
            sigma_u,
            sigma_v,
        )

    @property
    def max_energy(self) -> float:
        """The spectrum's top bin edge, for cross-section table sizing."""
        return self._spectrum.max_energy

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one primary on the plane; consumes exactly six uniforms."""
        return self._spot_fan.emit_from_plane(self._spectrum, 0, rng_state)

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, npt.NDArray[Any]]:
        """Vectorized simple-route batch; see :meth:`_GaussianSpotFan.sample_from_plane_batch`."""
        return self._spot_fan.sample_from_plane_batch(self._spectrum, 0, seed, history_offset, n)


class GaussianSpotBeamletSource(BeamletSource):
    """The beamlet-resolved planar Gaussian-spot source (one rectangle per bixel).

    Beamlet ``j`` emits on the rectangle centred at ``centers[j]``; the draw
    stream never sees the beamlet, so correlated Dij sampling replays the same
    energy, in-rectangle offset and spot point in every column. See
    :class:`GaussianSpotBeamSource` for the geometry and conventions.
    """

    def __init__(
        self,
        spectrum: Spectrum,
        focal_point: tuple[float, float, float],
        centers: Sequence[tuple[float, float, float]],
        width_u: float,
        width_v: float,
        sigma_u: float,
        sigma_v: float,
        u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        self._spectrum = spectrum
        self._spot_fan = _GaussianSpotFan(
            _DivergentFan(focal_point, tuple(centers), width_u, width_v, u_axis, v_axis),
            sigma_u,
            sigma_v,
        )

    @property
    def max_energy(self) -> float:
        """The spectrum's top bin edge, for cross-section table sizing."""
        return self._spectrum.max_energy

    @property
    def n_beamlets(self) -> int:
        """One beamlet per plane rectangle centre."""
        return len(self._spot_fan.fan.centers)

    def emit(self, beamlet: int, rng_state: RNGState) -> Primary:
        """Emit one primary for ``beamlet``; consumes exactly six uniforms."""
        if not 0 <= beamlet < self.n_beamlets:
            raise IndexError(f"beamlet {beamlet} outside fan of {self.n_beamlets}")
        return self._spot_fan.emit_from_plane(self._spectrum, beamlet, rng_state)

    def sample_beamlet_batch(
        self, seed: int, history_offset: int, n: int, beamlet: int
    ) -> dict[str, npt.NDArray[Any]]:
        """Vectorized per-beamlet batch; chunk-invariant and beamlet-blind."""
        if not 0 <= beamlet < self.n_beamlets:
            raise IndexError(f"beamlet {beamlet} outside fan of {self.n_beamlets}")
        return self._spot_fan.sample_from_plane_batch(
            self._spectrum, beamlet, seed, history_offset, n
        )
