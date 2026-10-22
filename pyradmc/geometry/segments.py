"""The collimated segments (apertures) of one photon fan as the beamlets of a source.

A step-and-shoot or VMAT plan delivers one fan through many collimator settings.
:class:`SegmentedFanSource` makes each setting a *beamlet*: ``run_dij`` then returns
one dose column per segment (the Dij of a segmented plan) from a single engine
launch, instead of one head pre-solve plus one transport run per segment.

Physics — exactly the head pre-solve's ``mode="attenuation"``
(:func:`pyradmc.geometry.head.presolve_head`) and of
:class:`~pyradmc.geometry.collimation.CollimatedSource`: a history samples the fan
of a :class:`~pyradmc.geometry.source.SpectralBeamSource` (energy, then the ray
from the focal point through the field) and, as a primary of segment ``s``,
carries the statistical weight

    w_s = exp(-sum_j mu_j(E) L_{s,j}(ray)),

the narrow-beam Beer-Lambert transmission through segment ``s``'s stack, with
``mu_j`` the total macroscopic attenuation coefficient of device ``j`` (from the
:class:`~pyradmc.data.interface.CrossSectionSource`, coherent included) and
``L_{s,j}`` the device chord the ray cuts (full-line convention of
:mod:`pyradmc.geometry.collimation`). Stated approximations, inherited unchanged
from the attenuation mode: photons scattered in the devices are removed and none
are transported onward (no head scatter, no leaf-scattered photons), and no
contaminant electrons. Column ``s`` of the Dij is the dose per fan history
through segment ``s``; a plan's dose is ``sum_s MU_s * column_s``.

Every segment shares the fan, the beam frame, and the stack's *structure* —
device kinds, slabs, materials, densities, leaf strips, tip radius, focusing —
and differs only in jaw edges and leaf tips. That is the device layout too: one
uploaded stack plus per-segment columns of edges and tips. Building one
:class:`~pyradmc.geometry.collimation.MLC` per segment per optimization iteration
costs its validation; the array constructor takes the tips directly
(:meth:`SegmentedFanSource.with_settings` reuses the attenuation tables as well),
and the host route builds the stacks lazily, only when it needs them.

The fan draws are segment-blind (four uniforms, the spectral fan's own contract),
so under correlated Dij sampling every column replays the same ray and only the
weight differs. The Warp engine generates this exact type in-kernel, with the
transmission evaluated on the device by the same collimation twins the device
pre-solve uses; a subclass, or a subclass of the fan, keeps the host
pre-sampling route.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pyradmc.data.interface import CrossSectionSource
from pyradmc.geometry.collimation import MLC, BeamLimitingStack, JawPair
from pyradmc.geometry.head import _MuTables
from pyradmc.geometry.source import BeamletSource, Primary, Source, SpectralBeamSource
from pyradmc.rng import RNGState, uniform

__all__ = ["SegmentedFanSource", "SegmentedPlanSource"]

_F64 = npt.NDArray[np.float64]

# How far the stack frame's origin may sit from the fan's focal point (cm): the
# devices are positioned relative to the focal spot, so a frame anchored elsewhere
# (the isocentre, say) is a setup error, not a geometry; anything above rounding is.
_FOCAL_TOLERANCE_CM = 1.0e-6

# Salt of the plan view's vectorized segment-selection stream, so it is independent
# of the fan's own PCG64(seed) batch stream (which it must not shift: the fan
# columns of a plan batch are the fan's batch, test-pinned).
_PLAN_SELECTION_SALT = 0x504C414E


def _jaws(stack: BeamLimitingStack) -> list[int]:
    return [j for j, device in enumerate(stack.devices) if isinstance(device, JawPair)]


def _mlcs(stack: BeamLimitingStack) -> list[int]:
    return [j for j, device in enumerate(stack.devices) if isinstance(device, MLC)]


def _structure_mismatch(template: JawPair | MLC, device: JawPair | MLC) -> str | None:
    """Name the first structural field in which ``device`` differs from ``template``."""
    if type(device) is not type(template):
        return f"type {type(device).__name__} != {type(template).__name__}"
    shared = ("z_top", "z_bottom", "material", "density")
    if isinstance(template, JawPair):
        fields: tuple[str, ...] = (*shared, "axis", "focused")
    else:
        fields = (*shared, "leaf_edges_v", "tip_radius", "focused_sides")
    for name in fields:
        if getattr(device, name) != getattr(template, name):
            return f"{name} {getattr(device, name)!r} != {getattr(template, name)!r}"
    return None


def _settings_array(
    values: npt.ArrayLike | None, name: str, width: int
) -> npt.NDArray[np.float64] | None:
    """Validate one per-segment settings array: ``(n_segments, width)``, finite."""
    if values is None:
        return None
    array = np.array(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (n_segments, {width}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


class SegmentedFanSource(BeamletSource):
    """One photon fan through several collimator settings; segment ``s`` is beamlet ``s``.

    Physics, conventions and stated approximations: the module docstring.

    Parameters
    ----------
    fan
        The open fan: spectrum, focal point, field rectangle. The stack frame's
        origin must be its focal point, and the field must lie downstream.
    template
        A stack giving every segment's structure (devices, slabs, materials, leaf
        strips, tip radius, focusing) and the default settings of any device whose
        per-segment settings are not given.
    cross_sections
        Source of the device attenuation coefficients, tabulated once per source.
    jaw_neg, jaw_pos
        Per-segment jaw edges, shape ``(n_segments, n_jaws)``: column ``k`` is the
        ``k``-th :class:`~pyradmc.geometry.collimation.JawPair` in stack order, at
        its mid-plane (cm). ``None`` keeps the template's edges in every segment.
    tips_neg, tips_pos
        Per-segment leaf tips, shape ``(n_segments, n_tips)``, the leaf pairs of
        every :class:`~pyradmc.geometry.collimation.MLC` in stack order,
        concatenated (the :meth:`~BeamLimitingStack.flatten` layout), at the
        mid-plane (cm). ``None`` keeps the template's tips in every segment.
    e_min, n_table
        The attenuation table's log-energy grid, as in the head pre-solve (whose
        defaults these are, so both routes interpolate one table).

    At least one per-segment array is required; they fix ``n_segments``.
    :meth:`from_stacks` builds the source from one stack per segment instead.
    """

    def __init__(
        self,
        fan: SpectralBeamSource,
        template: BeamLimitingStack,
        cross_sections: CrossSectionSource,
        *,
        jaw_neg: npt.ArrayLike | None = None,
        jaw_pos: npt.ArrayLike | None = None,
        tips_neg: npt.ArrayLike | None = None,
        tips_pos: npt.ArrayLike | None = None,
        e_min: float = 0.010,
        n_table: int = 512,
    ) -> None:
        pairs = tuple(zip(template.materials, template.densities, strict=True))
        tables = _MuTables(cross_sections, pairs, e_min, fan.max_energy, n_table)
        self._setup(fan, template, tables, jaw_neg, jaw_pos, tips_neg, tips_pos)

    @classmethod
    def from_stacks(
        cls,
        fan: SpectralBeamSource,
        stacks: Sequence[BeamLimitingStack],
        cross_sections: CrossSectionSource,
        *,
        e_min: float = 0.010,
        n_table: int = 512,
    ) -> SegmentedFanSource:
        """Build the source from one stack per segment (the first is the template).

        Every stack must share the template's frame and structure; only jaw edges
        and leaf tips may differ.
        """
        stacks = tuple(stacks)
        if not stacks:
            raise ValueError("a segmented fan needs at least one segment")
        template = stacks[0]
        for s, stack in enumerate(stacks[1:], start=1):
            if stack.frame != template.frame:
                raise ValueError(f"segment {s}: beam frame differs from segment 0's frame")
            if len(stack.devices) != len(template.devices):
                raise ValueError(
                    f"segment {s}: {len(stack.devices)} devices, segment 0 has "
                    f"{len(template.devices)}"
                )
            for j, (device, reference) in enumerate(
                zip(stack.devices, template.devices, strict=True)
            ):
                mismatch = _structure_mismatch(reference, device)
                if mismatch is not None:
                    raise ValueError(f"segment {s}, device {j}: {mismatch}")
        jaws = _jaws(template)
        mlcs = _mlcs(template)

        def jaw_column(name: str) -> _F64:
            return np.array(
                [[getattr(stack.devices[j], name) for j in jaws] for stack in stacks],
                dtype=np.float64,
            ).reshape(len(stacks), len(jaws))

        def tip_column(name: str) -> _F64:
            rows = [[t for j in mlcs for t in getattr(stack.devices[j], name)] for stack in stacks]
            return np.array(rows, dtype=np.float64).reshape(len(stacks), -1)

        source = cls(
            fan,
            template,
            cross_sections,
            jaw_neg=jaw_column("edge_neg"),
            jaw_pos=jaw_column("edge_pos"),
            tips_neg=tip_column("tips_neg"),
            tips_pos=tip_column("tips_pos"),
            e_min=e_min,
            n_table=n_table,
        )
        source._stacks.update(enumerate(stacks))
        return source

    def with_settings(
        self,
        *,
        jaw_neg: npt.ArrayLike | None = None,
        jaw_pos: npt.ArrayLike | None = None,
        tips_neg: npt.ArrayLike | None = None,
        tips_pos: npt.ArrayLike | None = None,
    ) -> SegmentedFanSource:
        """Return the same fan and structure with new per-segment settings.

        The cheap per-iteration path of an aperture optimization: no device
        objects are built and the attenuation tables are shared, not rebuilt.
        Settings not given default to the *template's* (not to this source's).
        """
        source = object.__new__(type(self))
        source._setup(self._fan, self._template, self._tables, jaw_neg, jaw_pos, tips_neg, tips_pos)
        return source

    def _setup(
        self,
        fan: SpectralBeamSource,
        template: BeamLimitingStack,
        tables: _MuTables,
        jaw_neg: npt.ArrayLike | None,
        jaw_pos: npt.ArrayLike | None,
        tips_neg: npt.ArrayLike | None,
        tips_pos: npt.ArrayLike | None,
    ) -> None:
        """Validate and store the fan, structure, tables and per-segment settings."""
        frame = template.frame
        offset = math.dist(frame.origin, fan.focal_point)
        if offset > _FOCAL_TOLERANCE_CM:
            raise ValueError(
                f"the stack frame's origin {frame.origin} must be the fan's focal point "
                f"{fan.focal_point} (off by {offset:.3g} cm)"
            )
        axial = sum(
            (c - f) * w for c, f, w in zip(fan.center, fan.focal_point, frame.w_axis, strict=True)
        )
        if axial <= 0.0:
            raise ValueError("the fan's field must lie downstream of the focal point (local w > 0)")

        jaws = _jaws(template)
        mlcs = _mlcs(template)
        compiled = template.flatten()
        n_devices = compiled.n_devices
        n_tips = int(compiled.mlc_n_pairs.sum())
        given = {
            "jaw_neg": _settings_array(jaw_neg, "jaw_neg", len(jaws)),
            "jaw_pos": _settings_array(jaw_pos, "jaw_pos", len(jaws)),
            "tips_neg": _settings_array(tips_neg, "tips_neg", n_tips),
            "tips_pos": _settings_array(tips_pos, "tips_pos", n_tips),
        }
        counts = {array.shape[0] for array in given.values() if array is not None}
        if not counts:
            raise ValueError("give at least one per-segment setting array (n_segments rows)")
        if len(counts) != 1:
            raise ValueError(f"settings arrays disagree on the number of segments: {counts}")
        n_segments = counts.pop()
        if n_segments < 1:
            raise ValueError("a segmented fan needs at least one segment")

        def full(array: _F64 | None, default: _F64) -> _F64:
            if array is None:
                return np.tile(default, (n_segments, 1))
            return array

        jaw_lo = full(given["jaw_neg"], compiled.jaw_edge_neg[jaws])
        jaw_hi = full(given["jaw_pos"], compiled.jaw_edge_pos[jaws])
        tip_lo = full(given["tips_neg"], compiled.tips_neg[:n_tips])
        tip_hi = full(given["tips_pos"], compiled.tips_pos[:n_tips])
        if np.any(jaw_lo > jaw_hi):
            s, k = np.argwhere(jaw_lo > jaw_hi)[0]
            raise ValueError(f"segment {s}, jaw {k}: edges cross ({jaw_lo[s, k]} > {jaw_hi[s, k]})")
        if np.any(tip_lo > tip_hi):
            s, k = np.argwhere(tip_lo > tip_hi)[0]
            raise ValueError(
                f"segment {s}, leaf pair {k}: tips cross ({tip_lo[s, k]} > {tip_hi[s, k]})"
            )

        # Device layout: jaw edges per (segment, device) with MLC rows unused (zero),
        # so a kernel indexes them with the device index it already has.
        self._jaw_neg = np.zeros((n_segments, n_devices), dtype=np.float64)
        self._jaw_pos = np.zeros((n_segments, n_devices), dtype=np.float64)
        self._jaw_neg[:, jaws] = jaw_lo
        self._jaw_pos[:, jaws] = jaw_hi
        self._tips_neg = tip_lo
        self._tips_pos = tip_hi
        self._fan = fan
        self._template = template
        self._tables = tables
        self._jaw_devices = tuple(jaws)
        self._mlc_devices = tuple(mlcs)
        self._stacks: dict[int, BeamLimitingStack] = {}

    # --- the BeamletSource contract ----------------------------------------------------

    @property
    def max_energy(self) -> float:
        """The fan spectrum's top edge, for cross-section table sizing."""
        return self._fan.max_energy

    @property
    def n_beamlets(self) -> int:
        """One beamlet (Dij column) per segment."""
        return self.n_segments

    def emit(self, beamlet: int, rng_state: RNGState) -> Primary:
        """Emit the fan's primary weighted by segment ``beamlet``'s transmission.

        Consumes exactly the fan's four uniforms; the segment never touches the
        stream, so correlated sampling replays one ray in every column.
        """
        self._check(beamlet)
        primary = self._fan.emit(rng_state)
        factor = self.transmission(
            beamlet,
            np.array([primary.energy]),
            np.array([[primary.x, primary.y, primary.z]]),
            np.array([[primary.ux, primary.uy, primary.uz]]),
        )
        return primary._replace(weight=primary.weight * float(factor[0]))

    def sample_beamlet_batch(
        self, seed: int, history_offset: int, n: int, beamlet: int
    ) -> dict[str, npt.NDArray[Any]]:
        """Vectorized host batch: the fan's batch, weights times the transmission.

        Chunk-invariant and segment-blind, exactly as the fan's own batch is (a
        *different* stream from :meth:`emit`'s per-history spawn, the spectral
        sources' precedent: the backends agree statistically, never bit-wise).
        """
        self._check(beamlet)
        columns = self._fan.sample_batch(seed, history_offset, n)
        origins = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
        directions = np.stack([columns["ux"], columns["uy"], columns["uz"]], axis=1).astype(
            np.float64
        )
        factors = self.transmission(
            beamlet, columns["energy"].astype(np.float64), origins, directions
        )
        out = dict(columns)
        out["weight"] = (columns["weight"].astype(np.float64) * factors).astype(np.float32)
        return out

    # --- geometry and attenuation --------------------------------------------------------

    def transmission(self, segment: int, energies: _F64, origins: _F64, directions: _F64) -> _F64:
        """Narrow-beam transmission of engine-frame rays through one segment's stack.

        ``exp(-sum_j mu_j(E) L_j)`` over ``(n,)`` energies and ``(n, 3)`` rays, with
        the head pre-solve's attenuation table and arithmetic.
        """
        self._check(segment)
        stack = self.segment_stack(segment)
        thickness = stack.path_lengths(origins, directions)
        tau = np.zeros(thickness.shape[0], dtype=np.float64)
        for j in range(thickness.shape[1]):
            tau += self._tables.total(j, energies) * thickness[:, j]
        return np.asarray(np.exp(-tau), dtype=np.float64)

    def segment_stack(self, segment: int) -> BeamLimitingStack:
        """Segment ``segment``'s stack as device objects (built once, on demand)."""
        self._check(segment)
        stack = self._stacks.get(segment)
        if stack is None:
            devices = list(self._template.devices)
            for j in self._jaw_devices:
                jaw = devices[j]
                assert isinstance(jaw, JawPair)
                devices[j] = dataclasses.replace(
                    jaw,
                    edge_neg=float(self._jaw_neg[segment, j]),
                    edge_pos=float(self._jaw_pos[segment, j]),
                )
            start = 0
            for j in self._mlc_devices:
                device = devices[j]
                assert isinstance(device, MLC)
                stop = start + device.n_pairs
                devices[j] = dataclasses.replace(
                    device,
                    tips_neg=tuple(float(t) for t in self._tips_neg[segment, start:stop]),
                    tips_pos=tuple(float(t) for t in self._tips_pos[segment, start:stop]),
                )
                start = stop
            stack = BeamLimitingStack(frame=self._template.frame, devices=tuple(devices))
            self._stacks[segment] = stack
        return stack

    def plan(self, weights: npt.ArrayLike) -> SegmentedPlanSource:
        """Return the open-field view of the segments summed with ``weights`` (e.g. MU).

        ``run`` on the result estimates ``sum_s weights[s] * column_s`` of this
        source's Dij in one launch, with histories allocated to segments in
        proportion to their weight; see :class:`SegmentedPlanSource`.
        """
        return SegmentedPlanSource(self, weights)

    def _check(self, segment: int) -> None:
        if not 0 <= segment < self.n_segments:
            raise IndexError(f"segment {segment} outside fan of {self.n_segments}")

    # --- what a device backend uploads -------------------------------------------------

    @property
    def n_segments(self) -> int:
        """Number of segments (collimator settings)."""
        return int(self._tips_neg.shape[0])

    @property
    def fan(self) -> SpectralBeamSource:
        """The open fan every segment collimates."""
        return self._fan

    @property
    def template(self) -> BeamLimitingStack:
        """The stack giving the shared structure (and default settings)."""
        return self._template

    @property
    def jaw_neg(self) -> _F64:
        """Jaw ``-`` edges, ``(n_segments, n_devices)``; non-jaw columns are unused."""
        return self._jaw_neg

    @property
    def jaw_pos(self) -> _F64:
        """Jaw ``+`` edges, ``(n_segments, n_devices)``; non-jaw columns are unused."""
        return self._jaw_pos

    @property
    def tips_neg(self) -> _F64:
        """``-u`` leaf tips, ``(n_segments, n_tips)``, in the flattened-stack layout."""
        return self._tips_neg

    @property
    def tips_pos(self) -> _F64:
        """``+u`` leaf tips, ``(n_segments, n_tips)``, in the flattened-stack layout."""
        return self._tips_pos

    @property
    def attenuation_tables(self) -> _MuTables:
        """The per-device macroscopic attenuation tables (shared with the pre-solve)."""
        return self._tables


class SegmentedPlanSource(Source):
    """A segmented fan's plan dose as an open-field source: segments summed by weight.

    Each history first chooses a segment with probability ``p_s = w_s / W``
    (``W = sum_s w_s``; one uniform), then emits the fan exactly as
    :meth:`SegmentedFanSource.emit` does for that segment, at weight
    ``W * exp(-tau_s)``. ``run`` therefore estimates

        sum_s p_s * W * column_s = sum_s w_s * column_s,

    the weighted sum of the segments' Dij columns (each per fan history through
    its segment), with no Dij, no per-segment dose maps and no truncation — the
    plan-dose route of an optimization loop. Histories go to segments in
    proportion to their weight, so a segment with more monitor units gets more of
    them, multinomially (the history index is not visible to ``emit``, so the
    allocation is sampled rather than stratified); a zero-weight segment is never
    chosen (selection takes the first ``s`` whose cumulative share exceeds the
    uniform). Build it with :meth:`SegmentedFanSource.plan`. The Warp engine
    generates this exact type in-kernel when its segments and fan are exact types
    too; anything else is host pre-sampled.
    """

    def __init__(self, segments: SegmentedFanSource, weights: npt.ArrayLike) -> None:
        w = np.array(weights, dtype=np.float64).reshape(-1)
        if w.size != segments.n_segments:
            raise ValueError(f"need one weight per segment ({segments.n_segments}), got {w.size}")
        if not np.all(np.isfinite(w)):
            raise ValueError("segment weights must be finite")
        if np.any(w < 0.0):
            raise ValueError("segment weights must be non-negative")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("segment weights must sum to a positive value")
        cdf = np.cumsum(w) / total
        cdf[-1] = 1.0  # exact: a uniform in [0, 1) always finds a segment
        self._segments = segments
        self._weights = w
        self._total = total
        self._cdf = cdf

    @property
    def max_energy(self) -> float:
        """The fan spectrum's top edge."""
        return self._segments.max_energy

    @property
    def segments(self) -> SegmentedFanSource:
        """The segmented fan whose columns this plan sums."""
        return self._segments

    @property
    def weights(self) -> _F64:
        """The per-segment weights (e.g. MU), as given."""
        return self._weights

    @property
    def total_weight(self) -> float:
        """``W = sum_s w_s``, the factor every primary's transmission is scaled by."""
        return self._total

    @property
    def selection_cdf(self) -> _F64:
        """Cumulative selection shares ``sum_{k<=s} w_k / W`` (last entry exactly 1)."""
        return self._cdf

    def _select(self, u: npt.ArrayLike) -> npt.NDArray[np.int64]:
        index = np.searchsorted(self._cdf, u, side="right")
        return np.minimum(index, self._cdf.size - 1).astype(np.int64)

    def emit(self, rng_state: RNGState) -> Primary:
        """Choose a segment by weight (one uniform), then emit its weighted primary."""
        segment = int(self._select(uniform(rng_state)))
        primary = self._segments.emit(segment, rng_state)
        return primary._replace(weight=self._total * primary.weight)

    def segment_of(self, seed: int, history_offset: int, n: int) -> npt.NDArray[np.int64]:
        """Return the segment the vectorized batch gives each of ``n`` histories.

        One uniform per history from a salted ``PCG64`` stream advanced to
        ``history_offset``: chunk-invariant, and independent of the fan's batch.
        """
        bitgen = np.random.PCG64(np.random.SeedSequence((seed, _PLAN_SELECTION_SALT)))
        bitgen.advance(history_offset)
        return self._select(np.random.Generator(bitgen).random(n))

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, npt.NDArray[Any]]:
        """Vectorized host batch: the fan's batch, each row weighted by its segment."""
        columns = self._segments.fan.sample_batch(seed, history_offset, n)
        segment = self.segment_of(seed, history_offset, n)
        energies = columns["energy"].astype(np.float64)
        origins = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
        directions = np.stack([columns["ux"], columns["uy"], columns["uz"]], axis=1).astype(
            np.float64
        )
        factors = np.zeros(n, dtype=np.float64)
        for s in np.unique(segment):
            rows = segment == s
            factors[rows] = self._segments.transmission(
                int(s), energies[rows], origins[rows], directions[rows]
            )
        out = dict(columns)
        out["weight"] = (columns["weight"].astype(np.float64) * self._total * factors).astype(
            np.float32
        )
        return out
