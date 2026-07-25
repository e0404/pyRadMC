"""The Warp engine: launch and memory management for the transport kernels.

Production backend, ``cpu`` and ``cuda`` from one kernel source (AGENTS.md section 3:
nothing physical lives here). The run loop is:

1. Flatten the cross-sections (``build_tables``) and upload everything as float32.
2. Per batch, per chunk of histories: generate primaries into a queue, then
   ping-pong the photon and electron kernels until every queue drains. Photons
   spawn charged secondaries into the electron queue; electrons spawn
   bremsstrahlung and annihilation photons back, and delta rays into the next
   electron round.
3. Reduce the int64 fixed-point energy maps to per-voxel dose mean and sigma **on
   the device**: one ``kernels.accumulate_run_batch`` / ``kernels.accumulate_dij_batch``
   launch per statistical batch folds that batch's map into running float64 sums, and
   ``kernels.finalize_run`` turns the sums into mean and sigma — so a batch's map is
   reused, never stored, and only the reduced maps read back. The reduction mirrors
   the reference ``BatchedDoseScorer`` batch statistics (AGENTS.md 2.4) and is shared
   by the open-field and Dij paths, so a 1x1 Dij column reduces to the open-field
   dose bit for bit on one device.

Reproducibility (AGENTS.md 2.3): streams are pure functions of
``(seed, history_index)`` with child streams derived from parent draws, and scoring
is associative integer arithmetic — so one device, one seed is bit-reproducible
regardless of scheduling, chunking, or queue ordering, and *nothing* is reproducible
across targets. Energy conservation holds to float32 transport arithmetic plus the
scoring quantum; the equivalence tests assert it at 1e-4 relative.

This module is Warp backend code: exempt from ``mypy --strict`` (AGENTS.md 5).
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from pyRadMC import DIJ_TRUNCATION_RELATIVE, ECUT_MEV, ELECTRON_MASS_MEV, PCUT_MEV
from pyRadMC.backends.results import TransportResult
from pyRadMC.backends.warp import kernels
from pyRadMC.backends.warp.kernels import ENERGY_QUANTUM_MEV, GridInfo, Queue, Tables
from pyRadMC.data.interface import CrossSectionSource
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import (
    BeamletGridSource,
    BeamletSource,
    ParallelBeamSource,
    PencilBeamSource,
    Source,
    SpectralBeamletSource,
    SpectralBeamSource,
)
from pyRadMC.progress import ProgressCallback, ProgressEmitter
from pyRadMC.scoring.dij import DijAssembler, DijResult
from pyRadMC.scoring.dose_to_water import validate_scoring_mode
from pyRadMC.scoring.grid import ScoringGrid
from pyRadMC.transport.electron import default_step_energy_fraction
from pyRadMC.transport.particles import ELECTRON, PHOTON, POSITRON

__all__ = ["WarpEngine"]

_E_MAX_MARGIN = 1.0 + 1.0e-6  # table upper edge strictly above the primary energy

# Salt so the device pre-solve buffer's record-sampling stream is independent of the
# transport stream, which is seeded from the bare (seed, history) — a within-device
# concern only; cross-target agreement is statistical (AGENTS.md 2.3).
_PRESOLVE_SAMPLING_SALT = 0x50524553

# IAEA particle code (from a source's sample_batch columns) -> transport constant,
# as a lookup indexed by the code (1/2/3); slot 0 is unused.
_IAEA_TO_TRANSPORT = np.array([-1, PHOTON, ELECTRON, POSITRON], dtype=np.int32)


def _scoring_info(scoring: ScoringGrid) -> GridInfo:
    """Scalar metadata of the scoring grid, in the same struct as the transport grid.

    The kernels' ``_deposit`` routes positions through this exactly as the host
    scorer's ``deposit_at`` does; only the scalars travel — the mass map stays on
    the host, where the dose normalization happens.
    """
    si = GridInfo()
    si.x_lo, si.y_lo, si.z_lo = scoring.origin
    si.x_hi, si.y_hi, si.z_hi = scoring.upper_corner
    si.sx, si.sy, si.sz = scoring.spacing
    si.nx, si.ny, si.nz = scoring.shape
    si.n_voxels = scoring.n_voxels
    si.min_spacing = min(scoring.spacing)  # unused for scoring; kept coherent
    return si


def _validate_msc_model(msc_model: str) -> int:
    """Map the ``msc_model`` name to the kernel's launch flag; refuse unknowns.

    Same names and refusal wording as the reference loop
    (:func:`pyRadMC.transport.electron.electron_steps`), but checked *before*
    any launch — a kernel cannot raise. The int is what the kernels branch on:
    0 = Gaussian hinge (the retained test instrument),
    1 = Goudsmit-Saunderson (the shipped default).
    """
    if msc_model not in ("gaussian", "gs"):
        raise ValueError(f"unknown msc_model {msc_model!r}; expected 'gaussian' or 'gs'")
    return 1 if msc_model == "gs" else 0


_GROUP_SIZE_FALLBACK = 128
"""Beamlet group size when free device memory is unknown (cpu, or no query)."""

_GROUP_SIZE_CAP = 1024
"""Upper bound on an auto-sized group: past this the launches are already coarse
and the marginal drain amortization is nil, while the dense maps keep growing."""

_GROUP_MEMORY_FRACTION = 0.5
"""Fraction of reported-free device memory budgeted for the dense group maps; the
rest stays for geometry, tables, queues, and the sparse compaction output."""

_CHUNK_HISTORIES_PER_SM = 8192
"""Concurrent source histories per SM targeted by the auto chunk size.

Measured 2026-07-25 on a 36-SM RTX 4070 Laptop, 6 MV spectral open field into a
100^3 water phantom, interleaved best-of-3 (round-to-round spread under 0.5
percent). Wall time falls monotonically with chunk size — 32k 7.16 s, 64k 5.59,
128k 4.72, 256k 4.18, 512k 3.97, 1M 3.94 — then *regresses* at 2M (4.20 s) as the
queues crowd the device. The transport is latency-bound, so the lever is simply
how much independent work is resident per launch; the knee is at roughly 7-8k
histories per SM and the curve is flat past ~16k. This coefficient sits at the
knee rather than the flat top: the last 6 percent costs 3x the queue memory,
which on the Dij is memory taken straight out of the beamlet group."""

_CHUNK_SIZE_FALLBACK = 32_768
"""Chunk size where the device reports neither SM count nor free memory (cpu)."""

_CHUNK_SIZE_CAP = 1_048_576
"""Upper bound on an auto chunk size: measured saturation, and 2M regressed."""

_CHUNK_MEMORY_FRACTION = 0.25
"""Fraction of reported-free device memory the auto chunk size may spend on queues."""


_QUEUE_BYTES_PER_SLOT = 4 * 48 + 4
"""Device bytes a lane holds per queue slot: four queues of twelve 4-byte columns,
plus the lane's uint32 RNG-slot array. Pinned in ``tests/unit/test_warp_sizing.py``
against an actual allocation, so a column added to ``Queue`` cannot silently
invalidate the sizing arithmetic."""


def _queue_bytes(chunk: int, queue_factor: int, lanes: int) -> int:
    """Device memory the transport queues occupy for a given chunk size."""
    return chunk * queue_factor * _QUEUE_BYTES_PER_SLOT * lanes


def _auto_group_size(
    free_bytes: int, n_voxels: int, n_beamlets: int, lanes: int, queue_bytes: int = 0
) -> int:
    """Largest beamlet group whose dense device maps fit the memory budget.

    Group size is bit-inert (test-pinned), so this is pure scheduling policy.
    Dense per-beamlet cost: one int64 quanta map per batch lane plus the two
    float64 running sums — ``(8 * lanes + 16) * n_voxels`` bytes. The result is
    clamped to ``[1, min(n_beamlets, cap)]``; unknown free memory falls back to
    the documented default.

    ``queue_bytes`` is the transport queues' footprint (:func:`_queue_bytes`),
    taken off the top before the dense budget is computed. The queues scale with
    the chunk size, which this function does not otherwise see: budgeting a fixed
    fraction and *assuming* the remainder covered them was safe only while the
    chunk size was small. At the auto chunk size they are of the same order as
    the dense maps, not a rounding error.
    """
    if free_bytes <= 0:
        return min(_GROUP_SIZE_FALLBACK, n_beamlets)
    per_beamlet = (8 * lanes + 16) * n_voxels
    budget = max(0, free_bytes - queue_bytes) * _GROUP_MEMORY_FRACTION
    sized = int(budget) // per_beamlet
    return max(1, min(sized, n_beamlets, _GROUP_SIZE_CAP))


def _auto_chunk_size(device: str, queue_factor: int) -> int:
    """Concurrent histories per launch, sized from the device's width and memory.

    Bit-inert like every other scheduling knob, so this is pure policy: enough
    resident work to keep a latency-bound transport busy
    (:data:`_CHUNK_HISTORIES_PER_SM` per SM), capped by
    :data:`_CHUNK_SIZE_CAP` and by what the queues may occupy
    (:data:`_CHUNK_MEMORY_FRACTION` of reported-free memory). A device that
    reports neither width nor free memory — cpu — keeps the historic default.
    """
    d = wp.get_device(device)
    sm_count = int(getattr(d, "sm_count", 0) or 0)
    if not d.is_cuda or sm_count <= 0:
        return _CHUNK_SIZE_FALLBACK
    target = min(_CHUNK_HISTORIES_PER_SM * sm_count, _CHUNK_SIZE_CAP)
    free = _device_free_bytes(device)
    if free > 0:
        affordable = int(free * _CHUNK_MEMORY_FRACTION) // (queue_factor * _QUEUE_BYTES_PER_SLOT)
        target = min(target, affordable)
    return max(1, target)


def _device_free_bytes(device: str) -> int:
    """Return a device's reported free memory; 0 where warp has nothing to report."""
    d = wp.get_device(device)
    if d.is_cuda:
        return int(d.free_memory)
    return 0  # cpu: warp needs psutil to report, and RAM is not the binding budget


def _upload_queue(capacity: int, device: str) -> Queue:
    q = Queue()
    q.kind = wp.zeros(capacity, dtype=wp.int32, device=device)
    q.beamlet = wp.zeros(capacity, dtype=wp.int32, device=device)
    q.primary = wp.zeros(capacity, dtype=wp.int32, device=device)
    q.energy = wp.zeros(capacity, dtype=float, device=device)
    q.weight = wp.zeros(capacity, dtype=float, device=device)
    q.x = wp.zeros(capacity, dtype=float, device=device)
    q.y = wp.zeros(capacity, dtype=float, device=device)
    q.z = wp.zeros(capacity, dtype=float, device=device)
    q.ux = wp.zeros(capacity, dtype=float, device=device)
    q.uy = wp.zeros(capacity, dtype=float, device=device)
    q.uz = wp.zeros(capacity, dtype=float, device=device)
    q.rng = wp.zeros(capacity, dtype=wp.uint32, device=device)
    q.count = wp.zeros(1, dtype=wp.int32, device=device)
    q.capacity = capacity
    return q


@dataclass
class _DijShard:
    """One device's share of a Dij: its sparse blocks and its energy books.

    The books stay in exact int64 quanta so the caller can sum shards without the
    total depending on which device finished first.
    """

    blocks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    deposited_quanta: int = 0
    escaped_quanta: int = 0
    unscored_quanta: int = 0
    emitted_quanta: int = 0
    emitted_energy: float = 0.0


@dataclass
class _LaneResources:
    """One batch lane's private device state.

    A lane transports whole batches independently of its peers, so everything a
    batch's transport writes — queues, RNG slots, the quanta map — is per lane;
    the lane's ``stream`` orders its work and ``staging`` is the pinned element
    its count readbacks go through. The sequential path is lane 0 with no stream.
    """

    stream: object
    staging: object
    queues: list
    slots: object
    edep: object


def _launch(kernel, dim, inputs, device: str, stream) -> None:
    """Launch on the device's current stream, or on an explicit lane stream.

    Warp takes *either* ``device`` or ``stream`` (a stream names its device);
    every launch on a Dij code path routes through here so that a batch lane's
    work is ordered on the lane's stream and nothing leaks onto the default one.
    """
    if stream is None:
        wp.launch(kernel, dim=dim, inputs=inputs, device=device)
    else:
        wp.launch(kernel, dim=dim, inputs=inputs, stream=stream)


def _queue_count(q: Queue, stream=None, staging=None) -> int:
    """Read a queue's count back to the host.

    Without a stream this is the plain (default-stream) readback. On a lane
    stream, ``numpy()`` would sync only the *default* stream and could read a
    count the lane has not finished writing, so the value goes through a pinned
    staging element copied and synchronized on the lane's own stream.
    """
    if stream is None:
        count = int(q.count.numpy()[0])
    else:
        wp.copy(staging, q.count, stream=stream)
        wp.synchronize_stream(stream)
        count = int(staging.numpy()[0])
    if count > q.capacity:
        raise RuntimeError(
            f"particle queue overflow: {count} > capacity {q.capacity}. "
            "Lower chunk_size or raise queue_factor on the WarpEngine."
        )
    return count


def _reset_count(q: Queue, device: str, stream=None) -> None:
    if stream is None:
        q.count.zero_()
    else:
        _launch(kernels.fill_int32, 1, [q.count, 0], device, stream)


def _set_count(q: Queue, value: int, device: str, stream=None) -> None:
    if stream is None:
        wp.copy(q.count, wp.array(np.array([value], dtype=np.int32), device=device))
    else:
        _launch(kernels.fill_int32, 1, [q.count, value], device, stream)


@dataclass(frozen=True)
class WarpEngine:
    """Dual-target production engine over a voxel grid.

    Parameters
    ----------
    grid, cross_sections
        As in the reference engine.
    device
        Warp device string: ``"cpu"`` or ``"cuda:N"``.
    chunk_size
        Histories transported concurrently. Statistically and bit-wise inert
        (test-pinned); it only trades memory against launch count. Default
        ``None`` auto-sizes it per device from the SM count and reported-free
        memory (:func:`_auto_chunk_size`); an explicit integer is used as given.
        The transport is latency-bound, so this is the knob that decides how much
        independent work is resident: measured on a 36-SM 4070, the historic
        fixed 32768 ran a 6 MV open field at **0.55x** the auto size's throughput.
    queue_factor
        Queue capacity per chunk history. Overflow raises rather than dropping
        secondaries.
    """

    grid: VoxelGrid
    cross_sections: CrossSectionSource
    device: str = "cpu"
    chunk_size: int | None = None
    queue_factor: int = 16
    _auto_chunk: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    # Host-table cache: build_tables flattens the source by looping the Python query
    # API over every grid node — seconds of fixed overhead for a tabulated source —
    # while its result is frozen after construction, so one build per
    # (ecut, pcut, e_max) serves every run and every device shard. Only the host
    # flatten is cached; the device upload stays per call. The lock keeps concurrent
    # Dij shards (one host thread per device) from building the same key twice.
    _table_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _table_cache_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def _chunk_histories(self, device: str | None = None) -> int:
        """Resolve ``chunk_size``, auto-sizing per device when it was left unset.

        Memoized per device string: a ``devices=[...]`` Dij sizes each shard from
        the device that will run it, so a mixed set does not inherit one card's
        width. Bit-inert, so shards may legitimately differ here.
        """
        if self.chunk_size is not None:
            return self.chunk_size
        key = device or self.device
        cached = self._auto_chunk.get(key)
        if cached is None:
            cached = _auto_chunk_size(key, self.queue_factor)
            self._auto_chunk[key] = cached
        return int(cached)

    def run(
        self,
        source: Source,
        n_histories: int,
        n_batches: int,
        seed: int,
        pcut: float = PCUT_MEV,
        ecut: float = ECUT_MEV,
        transport_electrons: bool = True,
        primary_kind: str = "photon",
        scoring_grid: ScoringGrid | None = None,
        scoring_mode: str = "dose_to_medium",
        step_energy_fraction: float | None = None,
        msc_model: str = "gs",
        progress: ProgressCallback | None = None,
    ) -> TransportResult:
        """Transport ``n_histories`` primaries; same contract as the reference engine.

        See :meth:`pyRadMC.backends.ref.engine.ReferenceEngine.run` for parameter
        semantics — the two signatures are deliberately identical (``scoring_grid``
        included: the dose grid deposits accumulate on, default the transport grid,
        with off-grid deposits booked to ``energy_unscored``; and ``scoring_mode``:
        dose-to-water weights each deposit in-kernel by the stopping-power ratio
        from the flattened tables while the books stay physical; ``progress``: one
        tick per completed batch, the identical cadence to the reference engine —
        see :mod:`pyRadMC.progress`). A source with an in-kernel generator is
        generated on-device: the built-in mono beams from analytic parameters (all
        of one ``primary_kind``) and the exact
        :class:`~pyRadMC.geometry.source.SpectralBeamSource` type from its uploaded
        spectrum tables (photons; ``primary_kind`` ignored). Any other source (a
        phase space, a spectral *subclass*, or a user
        :class:`~pyRadMC.geometry.source.Source`) is transported by host-sampling
        each chunk via ``sample_batch`` and seeding the photon and electron queues
        by the per-record kind; ``primary_kind`` is then ignored and
        ``energy_emitted`` is booked from the sampled records.

        ``msc_model`` selects the multiple-scattering law exactly as in the
        reference loop (see
        :func:`pyRadMC.transport.electron.electron_steps`): ``"gs"`` — the
        shipped default — samples Goudsmit-Saunderson deflections from an
        eagerly precomputed table grid (built host-side on the first GS run
        for this ``(ecut, e_max)``, persisted to the user cache, uploaded per
        run) and does not apply the Gaussian-validity angular cap; the
        ``"gaussian"`` hinge survives as the paired-comparison test
        instrument. ``step_energy_fraction=None`` resolves to the selected
        model's validated fraction.
        """
        if n_histories < 1:
            raise ValueError(f"need at least one history, got {n_histories}")
        if n_histories % n_batches != 0:
            raise ValueError(
                f"n_histories={n_histories} not divisible by n_batches={n_batches}; "
                "unequal batches would weight batch means inconsistently"
            )
        from pyRadMC.backends.warp.presolve import DevicePhaseSpace

        in_kernel = isinstance(source, PencilBeamSource | ParallelBeamSource)
        is_device_ps = isinstance(source, DevicePhaseSpace)
        # Exact type, not isinstance: a subclass may override emit/sample_batch, and
        # the built-in generator would silently bypass the override; a subclass keeps
        # the host pre-sampling route (test-pinned).
        is_spectral = type(source) is SpectralBeamSource
        if is_device_ps and source.device != self.device:
            raise ValueError(
                f"the device pre-solve buffer is on {source.device!r} but this engine "
                f"runs on {self.device!r}; pre-solve on the engine's device"
            )
        if in_kernel:
            if primary_kind not in ("photon", "electron"):
                raise ValueError(f"unknown primary_kind {primary_kind!r}")
            kind = PHOTON if primary_kind == "photon" else ELECTRON

        dose_to_water = 1 if validate_scoring_mode(scoring_mode, transport_electrons) else 0
        msc_model_gs = _validate_msc_model(msc_model)
        if step_energy_fraction is None:
            step_energy_fraction = default_step_energy_fraction(msc_model)

        device = self.device
        gi, density, material = self._upload_grid(device)
        scoring = scoring_grid if scoring_grid is not None else ScoringGrid.for_grid(self.grid)
        si = _scoring_info(scoring)
        table_energy = source.max_energy
        tab = self._upload_tables(table_energy, pcut, ecut, device, with_gs=msc_model_gs == 1)

        chunk = min(self._chunk_histories(device), n_histories)
        capacity = chunk * self.queue_factor
        queues = [_upload_queue(capacity, device) for _ in range(4)]
        slots = wp.zeros(capacity, dtype=wp.uint32, device=device)
        edep = wp.zeros(scoring.n_voxels, dtype=wp.int64, device=device)
        escaped = wp.zeros(1, dtype=wp.int64, device=device)
        unscored = wp.zeros(1, dtype=wp.int64, device=device)
        deposited = wp.zeros(1, dtype=wp.int64, device=device)
        violations = wp.zeros(1, dtype=wp.int32, device=device)

        # Advanced route: a source exposing a warp_sampler is generated in-kernel by a
        # wrapper kernel (built once per sampler), which books emitted weight-energy
        # into this cumulative counter (mono beams book analytically; a pre-sampled
        # source books per chunk).
        generator = None
        emitted = None
        wraps = not in_kernel and not is_device_ps and not is_spectral
        if wraps and source.warp_sampler is not None:
            generator = kernels.make_generator_kernel(source.warp_sampler)
            emitted = wp.zeros(1, dtype=wp.int64, device=device)
        if is_device_ps:
            # The seeding kernel books emitted weight-energy into this counter,
            # like the wrapped-generator route, read once after the batches.
            emitted = wp.zeros(1, dtype=wp.int64, device=device)
        spectral_tables = None
        if is_spectral:
            # Built-in in-kernel generation for the exact spectral type: the spectrum
            # inversion tables upload once per run and each chunk samples on-device
            # (the host pre-sampling was measured wall-dominant on CT-grade runs).
            # Emitted energy is booked per primary, like the wrapped-generator route.
            spectrum = source.spectrum
            spectral_tables = (
                wp.array(spectrum.edges.astype(np.float32), dtype=float, device=device),
                wp.array(spectrum.cdf.astype(np.float32), dtype=float, device=device),
            )
            emitted = wp.zeros(1, dtype=wp.int64, device=device)

        n_voxels = scoring.n_voxels
        voxel_mass = wp.array(scoring.voxel_mass.reshape(n_voxels), dtype=wp.float64, device=device)
        # Dose finalize on the device: per-voxel batch sums of dose and dose^2, folded
        # in batch order so the open-field reduction matches a 1x1 Dij column bitwise
        # on one device (kernels.accumulate_run_batch). Only the reduced maps read
        # back, not the dense per-batch fixed-point buffer.
        s1 = wp.zeros(n_voxels, dtype=wp.float64, device=device)
        s2 = wp.zeros(n_voxels, dtype=wp.float64, device=device)
        dep_total = wp.zeros(1, dtype=wp.int64, device=device)  # dose-to-medium quanta
        per_batch = n_histories // n_batches
        energy_escaped = 0.0
        energy_unscored = 0.0
        energy_deposited = 0.0  # dose-to-water physical book (summed per-batch counter)
        energy_emitted = 0.0
        emitter = ProgressEmitter(progress, n_histories)

        history = 0
        for _ in range(n_batches):
            edep.zero_()
            escaped.zero_()
            unscored.zero_()
            deposited.zero_()
            remaining = per_batch
            while remaining > 0:
                n_chunk = min(chunk, remaining)
                if in_kernel:
                    self._transport_chunk(
                        source,
                        kind,
                        seed,
                        history,
                        n_chunk,
                        gi,
                        si,
                        density,
                        material,
                        tab,
                        queues,
                        slots,
                        edep,
                        escaped,
                        unscored,
                        deposited,
                        violations,
                        pcut,
                        ecut,
                        step_energy_fraction,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        device,
                    )
                elif is_device_ps:
                    self._transport_chunk_device_buffer(
                        source,
                        seed,
                        history,
                        n_chunk,
                        gi,
                        si,
                        density,
                        material,
                        tab,
                        queues,
                        slots,
                        edep,
                        escaped,
                        unscored,
                        deposited,
                        violations,
                        pcut,
                        ecut,
                        step_energy_fraction,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        emitted,
                        device,
                    )
                elif is_spectral:
                    self._transport_chunk_spectral(
                        source,
                        spectral_tables,
                        seed,
                        history,
                        n_chunk,
                        gi,
                        si,
                        density,
                        material,
                        tab,
                        queues,
                        slots,
                        edep,
                        escaped,
                        unscored,
                        deposited,
                        violations,
                        pcut,
                        ecut,
                        step_energy_fraction,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        emitted,
                        device,
                    )
                elif generator is not None:
                    self._transport_chunk_wrapped(
                        generator,
                        seed,
                        history,
                        n_chunk,
                        gi,
                        si,
                        density,
                        material,
                        tab,
                        queues,
                        slots,
                        edep,
                        escaped,
                        unscored,
                        deposited,
                        violations,
                        pcut,
                        ecut,
                        step_energy_fraction,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        emitted,
                        device,
                    )
                else:
                    energy_emitted += self._transport_chunk_presampled(
                        source,
                        seed,
                        history,
                        n_chunk,
                        gi,
                        si,
                        density,
                        material,
                        tab,
                        queues,
                        slots,
                        edep,
                        escaped,
                        unscored,
                        deposited,
                        violations,
                        pcut,
                        ecut,
                        step_energy_fraction,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        device,
                    )
                history += n_chunk
                remaining -= n_chunk
            wp.synchronize_device(device)
            if int(violations.numpy()[0]) != 0:
                raise RuntimeError(
                    "Woodcock majorant violated in the kernel despite table headroom. "
                    "The geometry contains material or density the majorant "
                    "declaration did not cover."
                )
            # Fold this batch into the running dose sums on the device; the dense
            # fixed-point map never leaves the GPU. dep_total sums the physical
            # quanta (dose-to-medium book); dose-to-water reads its own counter.
            wp.launch(
                kernels.accumulate_run_batch,
                dim=n_voxels,
                inputs=[
                    edep,
                    voxel_mass,
                    float(per_batch),
                    float(ENERGY_QUANTUM_MEV),
                    s1,
                    s2,
                    dep_total,
                ],
                device=device,
            )
            energy_escaped += float(escaped.numpy()[0]) * ENERGY_QUANTUM_MEV
            energy_unscored += float(unscored.numpy()[0]) * ENERGY_QUANTUM_MEV
            if dose_to_water != 0:
                energy_deposited += float(deposited.numpy()[0]) * ENERGY_QUANTUM_MEV
            emitter.tick(per_batch)

        if in_kernel:
            # Mono beams: every primary is one unit-weight photon at the beam energy
            # (max_energy == energy), so the device need not report emitted energy.
            energy_emitted = n_histories * source.max_energy
        elif emitted is not None:
            energy_emitted = float(emitted.numpy()[0]) * ENERGY_QUANTUM_MEV

        mean_dev = wp.zeros(n_voxels, dtype=wp.float64, device=device)
        sigma_dev = wp.zeros(n_voxels, dtype=wp.float64, device=device)
        wp.launch(
            kernels.finalize_run,
            dim=n_voxels,
            inputs=[s1, s2, n_batches, mean_dev, sigma_dev],
            device=device,
        )
        wp.synchronize_device(device)
        if dose_to_water == 0:
            energy_deposited = float(dep_total.numpy()[0]) * ENERGY_QUANTUM_MEV
        return TransportResult(
            dose=mean_dev.numpy().reshape(scoring.shape),
            dose_sigma=sigma_dev.numpy().reshape(scoring.shape),
            energy_emitted=energy_emitted,
            energy_deposited=energy_deposited,
            energy_escaped=energy_escaped,
            energy_unscored=energy_unscored,
            n_histories=n_histories,
            n_batches=n_batches,
            scoring_mode=scoring_mode,
        )

    def run_dij(
        self,
        source: BeamletSource,
        n_histories_per_beamlet: int,
        n_batches: int,
        seed: int,
        pcut: float = PCUT_MEV,
        ecut: float = ECUT_MEV,
        transport_electrons: bool = True,
        truncation: float = DIJ_TRUNCATION_RELATIVE,
        correlated: bool = True,
        beamlet_group_size: int | None = None,
        scoring_grid: ScoringGrid | None = None,
        scoring_mode: str = "dose_to_medium",
        step_energy_fraction: float | None = None,
        msc_model: str = "gs",
        devices: Sequence[str] | None = None,
        concurrent_batches: int = 1,
        progress: ProgressCallback | None = None,
    ) -> DijResult:
        """Compute the Dij over the lattice; same contract as the reference engine.

        See :meth:`pyRadMC.backends.ref.engine.ReferenceEngine.run_dij` for the
        history-to-beamlet mapping and parameter semantics, ``correlated``
        (the shipped sampling configuration, default True; ``False`` is the
        test instrument) included — the signatures are deliberately identical
        up to the one scheduling knob:

        Parameters
        ----------
        beamlet_group_size
            Beamlets scored concurrently into one dense device buffer. Purely a
            memory/occupancy trade-off: streams are pure functions of
            ``(seed, history)`` and scoring is associative, so the result is
            bit-identical for any value (test-pinned), exactly like ``chunk_size``.

            Default ``None`` auto-sizes it from the *reported free* memory of the
            (each) device — the largest group whose dense maps fit half of it,
            clamped to ``[1, min(n_beamlets, 1024)]``, the minimum over a
            multi-device set, and 128 where free memory is unknown (cpu). An
            explicit integer is used exactly as given.

            This is the *only* dial for the dense device cost, which is
            ``group * n_voxels * 24`` bytes — one int64 quanta map plus the two
            float64 sums — and is **independent of** ``n_batches``: batches are
            streamed one drain at a time and folded into the sums, not stored
            along a device axis. It is also what sets the launch width, since a
            batch's block is ``group * n_histories_per_beamlet / n_batches``
            histories; raise it until that block covers ``chunk_size``, memory
            permitting. On a CT-resolution scoring grid the auto-size shrinks
            the group to fit (pass an explicit value to override) — or coarsen
            ``scoring_grid``, which shrinks the dense cost cubically.

        devices
            Devices to shard the beamlet groups over, one host thread each; default
            (``None``) runs everything on the engine's own ``device``. Groups are
            independent — no cross-device reduction — so this scales with the device
            count rather than trading anything away, and each device pays the full
            per-device footprint (geometry, tables, queues, and the dense
            ``group * n_voxels * 24`` maps) since nothing is shared.

            Scheduling is greedy from a shared ordered queue: a device pulls the
            next group the moment it is free, so an idle device always receives new
            work before any deeper concurrency (``concurrent_batches`` lanes) on a
            busy one, and a slow device in a mixed set self-limits to the groups it
            can finish instead of holding an equal share hostage.

            Reproducibility: every column is computed wholly on one device, so
            scheduling never changes *what* a device computes for a beamlet — only
            which device computes it (test-pinned). Over identical devices the Dij
            is therefore bit-identical however the pulls interleave. Over a
            *heterogeneous* set (e.g. ``["cuda:0", "cpu"]``) the column-to-device
            assignment is timing-dependent run to run — columns stay statistically
            equivalent, the cpu/cuda relationship AGENTS.md 2.3 defines, but the
            same run twice may place them differently; pass a single device where
            strict run-to-run bit reproducibility matters on mixed hardware. The
            energy books stay exact either way, being integer quanta summed in
            device-list order.
        concurrent_batches
            Batch lanes per CUDA device: up to this many batches of the current
            group transport concurrently, each lane on its own stream with its own
            queues, RNG slots, and quanta map. **Bitwise inert** (test-pinned): the
            float64 fold of batch sums is serialized in batch order across lanes,
            so any value reproduces the sequential Dij exactly — the knob only
            trades memory for overlap, like ``chunk_size``. Extra lanes cost their
            queues and one ``group * n_voxels`` int64 map each; a cpu device has no
            streams and ignores the setting. On this engine's development hardware
            (laptop RTX 4070) the drain gaps lanes can hide measured at 2-3% of
            Dij wall time — the knob exists for larger cards, where the balance
            may differ; measure before defaulting it on.

        ``scoring_grid`` (semantics as in :meth:`run`) is the memory lever here:
        ``n_voxels`` above is the *scoring* voxel count, so a coarser dose grid
        shrinks the per-group device buffer and the sparse Dij cubically while
        transport keeps the full CT resolution. ``scoring_mode`` weights the
        column tallies as in :meth:`run`; the energy books stay physical.
        ``msc_model`` as in :meth:`run`: the GS grid is built once, before any
        launch, under the same lock every device shard takes.
        progress
            Optional callback, as in :meth:`run` (see :mod:`pyRadMC.progress`).
            Ticks once per **beamlet group** *completed* (``ceil(n_beamlets /
            beamlet_group_size)`` ticks total), each covering that group's full
            ``group_size * n_histories_per_beamlet`` histories across all of its
            batches. A tick marks the group finished — transported, reduced,
            truncated and read back — not merely transported, so ``rate_hz``
            describes the group it names and the last tick coincides with the
            method returning — a different axis from the reference engine's per-batch
            ticks, since this engine schedules groups off a shared queue rather
            than iterating batches outermost. Both reach the same total; treat
            ``histories_done / histories_total`` as the portable signal, not tick
            count or spacing. One :class:`~pyRadMC.progress.ProgressEmitter` is
            shared across every device shard and, within a shard, is unaffected
            by ``concurrent_batches`` (only the shard's outer per-group point
            ticks): the emitter's own lock serializes ticks arriving from
            multiple device threads, so the callback is thread-safe without any
            extra care on the caller's part.
        """
        if n_histories_per_beamlet < 1:
            raise ValueError(
                f"need at least one history per beamlet, got {n_histories_per_beamlet}"
            )
        if n_histories_per_beamlet % n_batches != 0:
            raise ValueError(
                f"n_histories_per_beamlet={n_histories_per_beamlet} not divisible by "
                f"n_batches={n_batches}; unequal batches would weight batch means inconsistently"
            )
        if beamlet_group_size is not None and beamlet_group_size < 1:
            raise ValueError(f"need a positive beamlet group size, got {beamlet_group_size}")
        if concurrent_batches < 1:
            raise ValueError(f"need at least one lane, got concurrent_batches={concurrent_batches}")
        dose_to_water = 1 if validate_scoring_mode(scoring_mode, transport_electrons) else 0
        msc_model_gs = _validate_msc_model(msc_model)
        if step_energy_fraction is None:
            step_energy_fraction = default_step_energy_fraction(msc_model)

        scoring = scoring_grid if scoring_grid is not None else ScoringGrid.for_grid(self.grid)
        n_beamlets = source.n_beamlets
        per_batch = n_histories_per_beamlet // n_batches

        device_list = [self.device] if devices is None else list(dict.fromkeys(devices))
        if not device_list:
            raise ValueError("devices must name at least one device")

        if beamlet_group_size is None:
            # One shared size for every device: group_starts partitions the beamlet
            # axis once, so a multi-device set takes the tightest device's fit. The
            # queues are charged before the dense budget: they scale with the chunk
            # size and, at the auto chunk size, are the same order as the dense maps.
            # ``n_beamlets * per_batch`` bounds the block a group can ever launch, so
            # this over- rather than under-states the queue cost.
            def _fit(d: str) -> int:
                lanes_d = min(concurrent_batches, n_batches) if wp.get_device(d).is_cuda else 1
                chunk_d = min(self._chunk_histories(d), n_beamlets * per_batch)
                return _auto_group_size(
                    _device_free_bytes(d),
                    scoring.n_voxels,
                    n_beamlets,
                    lanes_d,
                    _queue_bytes(chunk_d, self.queue_factor, lanes_d),
                )

            beamlet_group_size = min(_fit(d) for d in device_list)

        # Beamlet groups are independent — no cross-group reduction — so sharding them
        # over devices needs no communication at all. Assignment is greedy from a
        # shared ordered queue: a device pulls the next group the moment it is free,
        # so an idle device always takes new work before any deeper concurrency
        # (batch lanes) on a busy one — the scheduling preference this engine
        # promises. Each column is computed wholly on one device, so scheduling can
        # never change a column's value on a given device, only which device
        # produces it (test-pinned). Over *identical* devices the whole Dij is
        # therefore bit-identical however the pulls interleave; over a mixed set
        # (e.g. cuda + cpu) the column-to-device assignment is timing-dependent run
        # to run, each column still bit-equal to a whole run of its computing device
        # — pass a single device where strict run-to-run reproducibility on mixed
        # hardware matters. The energy books are exact integer quanta either way.
        group_starts = list(range(0, n_beamlets, beamlet_group_size))
        pending = deque(group_starts)
        pending_lock = threading.Lock()

        def _next_group() -> int | None:
            with pending_lock:
                return pending.popleft() if pending else None

        assembler = DijAssembler(
            grid_shape=scoring.shape,
            n_beamlets=n_beamlets,
            n_histories_per_beamlet=n_histories_per_beamlet,
            n_batches=n_batches,
            truncation=truncation,
            correlated=correlated,
            scoring_mode=scoring_mode,
        )

        emitter = ProgressEmitter(progress, n_beamlets * n_histories_per_beamlet)
        args = dict(
            source=source,
            n_histories_per_beamlet=n_histories_per_beamlet,
            n_batches=n_batches,
            per_batch=per_batch,
            seed=seed,
            pcut=pcut,
            ecut=ecut,
            step_energy_fraction=step_energy_fraction,
            msc_model_gs=msc_model_gs,
            transport_electrons=transport_electrons,
            truncation=truncation,
            correlated=correlated,
            beamlet_group_size=beamlet_group_size,
            scoring=scoring,
            dose_to_water=dose_to_water,
            concurrent_batches=concurrent_batches,
            emitter=emitter,
        )
        # Kernel modules are loaded up front whenever any host thread beyond this
        # one will launch (device workers or batch lanes): warp's module loading is
        # not thread-safe. Everything after takes its device explicitly, so the
        # threads share no warp state.
        if len(device_list) > 1 or concurrent_batches > 1:
            for d in device_list:
                wp.load_module(kernels, device=d)
        if len(device_list) == 1:
            shard_results = [self._run_dij_shard(device_list[0], _next_group, **args)]
        else:
            # One host thread per device: warp launches are asynchronous but the drain
            # loop's count readbacks block, so a single thread would serialize the
            # devices on those readbacks.
            with ThreadPoolExecutor(max_workers=len(device_list)) as pool:
                futures = [
                    pool.submit(self._run_dij_shard, d, _next_group, **args) for d in device_list
                ]
                shard_results = [f.result() for f in futures]

        # Energy books in exact integer quanta (Python ints, unbounded), converted
        # to MeV once at the end: float accumulation order would otherwise make
        # the tallies — unlike the matrix — depend on the group size. Summed over
        # shards in device-list order, so the books do not depend on which device
        # finished first.
        deposited_quanta = sum(r.deposited_quanta for r in shard_results)
        escaped_quanta = sum(r.escaped_quanta for r in shard_results)
        unscored_quanta = sum(r.unscored_quanta for r in shard_results)
        emitted_quanta = sum(r.emitted_quanta for r in shard_results)
        emitted_energy = sum(r.emitted_energy for r in shard_results)

        # The assembler takes blocks in ascending, gap-free beamlet order; shards
        # finish in whatever order the devices happen to, so feed it from the merged
        # map rather than as the blocks arrive.
        blocks = {gs: block for r in shard_results for gs, block in r.blocks.items()}
        for group_start in group_starts:
            counts, indices, dose, sigma = blocks[group_start]
            assembler.add_sparse_block(group_start, counts, indices, dose, sigma)

        use_lattice = isinstance(source, BeamletGridSource)
        if use_lattice:
            emitted_energy = n_beamlets * n_histories_per_beamlet * source.max_energy
        elif emitted_quanta:
            emitted_energy = float(emitted_quanta) * ENERGY_QUANTUM_MEV
        return assembler.finalize(
            energy_emitted=emitted_energy,
            energy_deposited=deposited_quanta * ENERGY_QUANTUM_MEV,
            energy_escaped=escaped_quanta * ENERGY_QUANTUM_MEV,
            energy_unscored=unscored_quanta * ENERGY_QUANTUM_MEV,
        )

    def _run_dij_shard(
        self,
        device: str,
        next_group: Callable[[], int | None],
        *,
        source,
        n_histories_per_beamlet: int,
        n_batches: int,
        per_batch: int,
        seed: int,
        pcut: float,
        ecut: float,
        step_energy_fraction: float,
        msc_model_gs: int,
        transport_electrons: bool,
        truncation: float,
        correlated: bool,
        beamlet_group_size: int,
        scoring: ScoringGrid,
        dose_to_water: int,
        concurrent_batches: int,
        emitter: ProgressEmitter,
    ) -> _DijShard:
        """Pull beamlet groups from the shared queue and run them on one device.

        Owns every device resource it touches (geometry, tables, queues, dense maps,
        energy counters), so shards on different devices share nothing and may run
        concurrently on their own host threads. Returns the shard's sparse blocks
        keyed by group start, plus its energy books in exact quanta for the caller
        to merge in a device-order-independent way.

        ``concurrent_batches`` > 1 transports that many batches of the current group
        concurrently, each lane on its own CUDA stream with its own queues, RNG
        slots, and quanta map (``_run_dij_group_lanes``); a cpu device has no
        streams and always runs the sequential path.
        """
        gi, density, material = self._upload_grid(device)
        si = _scoring_info(scoring)
        tab = self._upload_tables(source.max_energy, pcut, ecut, device, with_gs=msc_model_gs == 1)
        n_beamlets = source.n_beamlets
        n_voxels = scoring.n_voxels
        # Route by capability: the built-in lattice generates in-kernel from analytic
        # bounds; the exact spectral beamlet type generates in-kernel from its
        # uploaded spectrum tables (a subclass keeps pre-sampling — exact-type check,
        # since an emit/sample override must not be bypassed; the host route was
        # measured wall-dominant on CT-grade Dij runs); a source with a
        # warp_beamlet_sampler is generated in-kernel by a wrapped kernel; any other
        # source is host pre-sampled per beamlet and uploaded. Beamlet primaries are
        # photons carrying the source's statistical weight (the built-in routes are
        # unit-weight by construction; the other two take the weight from the
        # sampler/emit, matching the reference Dij).
        use_lattice = isinstance(source, BeamletGridSource)
        use_spectral = type(source) is SpectralBeamletSource
        beamlet_generator = None
        emitted_counter = None
        spectral_tables = None
        if not use_lattice and not use_spectral and source.warp_beamlet_sampler is not None:
            beamlet_generator = kernels.make_beamlet_generator_kernel(source.warp_beamlet_sampler)
            emitted_counter = wp.zeros(1, dtype=wp.int64, device=device)
        if use_spectral:
            spectrum = source.spectrum
            spectral_tables = (
                wp.array(spectrum.edges.astype(np.float32), dtype=float, device=device),
                wp.array(spectrum.cdf.astype(np.float32), dtype=float, device=device),
            )
            emitted_counter = wp.zeros(1, dtype=wp.int64, device=device)

        voxel_mass = wp.array(scoring.voxel_mass.reshape(n_voxels), dtype=wp.float64, device=device)
        escaped = wp.zeros(1, dtype=wp.int64, device=device)
        unscored = wp.zeros(1, dtype=wp.int64, device=device)
        deposited = wp.zeros(1, dtype=wp.int64, device=device)
        violations = wp.zeros(1, dtype=wp.int32, device=device)
        shard = _DijShard()
        deposited_quanta = 0
        escaped_quanta = 0
        unscored_quanta = 0

        # Group-invariant device buffers, allocated once at the largest group's size
        # and reused per group. Allocating these *inside* the loop costs twice their
        # footprint at the peak: the successor is allocated while the predecessor is
        # still referenced. A trailing short group uses a prefix of each buffer, which
        # is why every launch below is dimensioned from ``group`` rather than from the
        # buffer length.
        #
        # The batch axis is streamed, not stored: ``edep`` holds one batch at a time
        # and each batch is folded into the running float64 sums s1/s2, so the dense
        # cost is group * n_voxels * (8 + 8 + 8) independent of n_batches, rather than
        # group * n_batches * n_voxels * 8. s1/s2 are then overwritten in place with
        # the mean/sigma they imply (see the finalize_run launch), so the group needs
        # no separate dense output maps.
        max_group = min(beamlet_group_size, n_beamlets)
        shard_chunk = self._chunk_histories(device)
        max_capacity = max(1, min(shard_chunk, max_group * per_batch)) * self.queue_factor
        # A cpu device has no streams; lanes beyond the first would serialize on the
        # single device queue anyway, so they collapse to the sequential path.
        lanes = 1
        if concurrent_batches > 1 and wp.get_device(device).is_cuda:
            lanes = min(concurrent_batches, n_batches)
        lane_resources = [
            _LaneResources(
                stream=wp.Stream(device) if lanes > 1 else None,
                staging=(
                    wp.zeros(1, dtype=wp.int32, device="cpu", pinned=True) if lanes > 1 else None
                ),
                queues=[_upload_queue(max_capacity, device) for _ in range(4)],
                slots=wp.zeros(max_capacity, dtype=wp.uint32, device=device),
                edep=wp.zeros(max_group * n_voxels, dtype=wp.int64, device=device),
            )
            for _ in range(lanes)
        ]
        fold_stream = wp.Stream(device) if lanes > 1 else None
        s1 = wp.zeros(max_group * n_voxels, dtype=wp.float64, device=device)
        s2 = wp.zeros(max_group * n_voxels, dtype=wp.float64, device=device)
        total_q = wp.zeros(1, dtype=wp.int64, device=device)
        # Truncation runs one thread per (column, chunk) rather than one per column;
        # the layout is scheduling only and the output is independent of it. Sized
        # from the widest group and this device's SM count, so the launch fills the
        # card: the chunk count a group needs falls as the group widens. A trailing
        # short group reuses this layout — it launches proportionally fewer threads
        # for proportionally less work, at unchanged per-thread cost.
        n_chunks, voxel_chunk = kernels.truncation_chunk_layout(
            n_voxels, max_group, int(getattr(wp.get_device(device), "sm_count", 0) or 0)
        )
        col_max_dev = wp.zeros(max_group, dtype=wp.float64, device=device)
        chunk_counts_dev = wp.zeros(max_group * n_chunks, dtype=wp.int32, device=device)

        route_args = dict(
            source=source,
            beamlet_generator=beamlet_generator,
            emitted_counter=emitted_counter,
            use_lattice=use_lattice,
            use_spectral=use_spectral,
            spectral_tables=spectral_tables,
            seed=seed,
            n_per=n_histories_per_beamlet,
            per_batch=per_batch,
            correlated=correlated,
            gi=gi,
            si=si,
            density=density,
            material=material,
            tab=tab,
            escaped=escaped,
            unscored=unscored,
            deposited=deposited,
            violations=violations,
            pcut=pcut,
            ecut=ecut,
            step_energy_fraction=step_energy_fraction,
            msc_model_gs=msc_model_gs,
            transport_electrons=transport_electrons,
            dose_to_water=dose_to_water,
            device=device,
        )
        fold_args = dict(
            voxel_mass=voxel_mass,
            n_voxels=n_voxels,
            per_batch=per_batch,
            s1=s1,
            s2=s2,
            total_q=total_q,
            device=device,
        )

        while (group_start := next_group()) is not None:
            group = min(beamlet_group_size, n_beamlets - group_start)
            block_histories = group * per_batch  # one batch's block
            chunk = min(shard_chunk, block_histories)
            # The per-group upload for whichever in-kernel route is active: analytic
            # bounds for the lattice, aperture centres for the spectral fan.
            group_arrays = None
            if use_lattice:
                group_arrays = self._upload_lattice_bounds(source, group_start, group, device)
            elif use_spectral:
                group_arrays = self._upload_spectral_centers(source, group_start, group, device)
            # Clear only what this group uses: the dense maps are allocated at
            # max_group, so zero_() would charge a trailing short group the full
            # group's clear. Every dense launch below is dimensioned from ``group``,
            # so the tail beyond it is never read.
            span = group * n_voxels
            wp.launch(kernels.fill_float64, dim=span, inputs=[s1, 0.0], device=device)
            wp.launch(kernels.fill_float64, dim=span, inputs=[s2, 0.0], device=device)
            # The fold resets every entry it consumes, so a quanta map only needs
            # clearing once per group rather than once per batch. It is still cleared
            # here rather than relying on the reset alone: a shorter group resets only
            # its own prefix, and the buffer should not depend on a short group always
            # being the last one. Each lane keeps its own map, so clear them all.
            for res in lane_resources:
                wp.launch(kernels.fill_int64, dim=span, inputs=[res.edep, 0], device=device)
            total_q.zero_()
            escaped.zero_()
            unscored.zero_()
            deposited.zero_()

            if lanes == 1:
                # One drain per batch, folded into s1/s2 before the next batch
                # reuses edep. Batch order is the fold order (accumulate_dij_batch).
                res = lane_resources[0]
                for batch in range(n_batches):
                    shard.emitted_energy += self._run_dij_batch(
                        res,
                        group_start,
                        group,
                        batch,
                        chunk,
                        block_histories,
                        group_arrays,
                        **route_args,
                    )
                    self._fold_dij_batch(res.edep, group, None, **fold_args)
            else:
                self._run_dij_group_lanes(
                    lane_resources,
                    fold_stream,
                    group_start,
                    group,
                    n_batches,
                    chunk,
                    block_histories,
                    group_arrays,
                    shard,
                    route_args,
                    fold_args,
                )

            wp.synchronize_device(device)
            if int(violations.numpy()[0]) != 0:
                raise RuntimeError(
                    "Woodcock majorant violated in the kernel despite table headroom. "
                    "The geometry contains material or density the majorant "
                    "declaration did not cover."
                )
            # Turn the group's accumulated sums into per-column dose mean and sigma,
            # in place: each thread reads s1[tid]/s2[tid] and writes only its own
            # element, so aliasing the outputs onto the inputs is safe and saves two
            # dense group * n_voxels float64 maps. Only these read back, never a
            # per-batch buffer. total_q summed the physical quanta over the batch
            # loop (dose-to-medium book); dose-to-water uses its counter.
            wp.launch(
                kernels.finalize_run,
                dim=group * n_voxels,
                inputs=[s1, s2, n_batches, s1, s2],
                device=device,
            )
            mean_dev, sigma_dev = s1, s2
            wp.synchronize_device(device)
            if dose_to_water != 0:
                deposited_quanta += int(deposited.numpy()[0])
            else:
                deposited_quanta += int(total_q.numpy()[0])
            escaped_quanta += int(escaped.numpy()[0])
            unscored_quanta += int(unscored.numpy()[0])

            # Truncate + compact each column on the device; only the surviving sparse
            # CSC entries read back, never the dense per-column dose maps. col_max is
            # a maximum and the keep test uses the same float64 threshold product as
            # DijAssembler.add_block, so this is byte-identical to host truncation on
            # these maps. Each column's sweep is split over ``n_chunks`` threads (a
            # single thread per column left the GPU essentially idle through the
            # dominant phase of the Dij); determinism survives because the maximum is
            # order-independent and each chunk writes from the exclusive prefix of the
            # chunks before it *in its own column*, so rows stay ascending.
            col_max_dev.zero_()
            wp.launch(
                kernels.column_max,
                dim=group * n_voxels,
                inputs=[mean_dev, n_voxels, col_max_dev],
                device=device,
            )
            wp.launch(
                kernels.count_kept_per_chunk,
                dim=group * n_chunks,
                inputs=[
                    mean_dev,
                    n_voxels,
                    n_chunks,
                    voxel_chunk,
                    float(truncation),
                    col_max_dev,
                    chunk_counts_dev,
                ],
                device=device,
            )
            wp.synchronize_device(device)
            per_chunk = chunk_counts_dev.numpy()[: group * n_chunks].reshape(group, n_chunks)
            counts = per_chunk.sum(axis=1).astype(np.int32)
            column_base = np.zeros(group, dtype=np.int64)
            np.cumsum(counts[:-1], out=column_base[1:])  # exclusive prefix over columns
            within_column = np.zeros((group, n_chunks), dtype=np.int64)
            np.cumsum(per_chunk[:, :-1], axis=1, out=within_column[:, 1:])  # ... and within one
            chunk_base = (column_base[:, None] + within_column).reshape(-1).astype(np.int32)
            nnz = int(counts.sum())
            out_indices = wp.zeros(nnz, dtype=wp.int64, device=device)
            out_dose = wp.zeros(nnz, dtype=wp.float64, device=device)
            out_sigma = wp.zeros(nnz, dtype=wp.float64, device=device)
            wp.launch(
                kernels.compact_column_chunked,
                dim=group * n_chunks,
                inputs=[
                    mean_dev,
                    sigma_dev,
                    n_voxels,
                    n_chunks,
                    voxel_chunk,
                    float(truncation),
                    col_max_dev,
                    wp.array(chunk_base, dtype=wp.int32, device=device),
                    out_indices,
                    out_dose,
                    out_sigma,
                ],
                device=device,
            )
            wp.synchronize_device(device)
            # Copy, do not alias: on a cpu device ``numpy()`` is a *view* of the warp
            # array, and these blocks outlive both the reused ``counts_dev`` and this
            # group's compaction buffers. (The blocks are the sparse result itself, so
            # owning them costs nothing beyond what the DijResult holds anyway.)
            shard.blocks[group_start] = (
                counts.copy(),
                out_indices.numpy().copy(),
                out_dose.numpy().copy(),
                out_sigma.numpy().copy(),
            )
            # Tick only now, with the group *complete*. Ticking right after transport
            # instead charged each group's reduction, truncation and readback to its
            # successor's interval: the first group then reported a rate no later one
            # could reach, and the final group's share landed after the last tick as
            # silence at 100 percent. The readback above is part of the group's cost,
            # so the tick follows it; ``numpy()`` has already synchronized, so this
            # still lands on an existing hard sync and adds none.
            emitter.tick(group * n_histories_per_beamlet)

        shard.deposited_quanta = deposited_quanta
        shard.escaped_quanta = escaped_quanta
        shard.unscored_quanta = unscored_quanta
        if emitted_counter is not None:
            shard.emitted_quanta = int(emitted_counter.numpy()[0])
        return shard

    def _run_dij_batch(
        self,
        res: _LaneResources,
        group_start,
        group,
        batch,
        chunk,
        block_histories,
        group_arrays,
        *,
        source,
        beamlet_generator,
        emitted_counter,
        use_lattice,
        use_spectral,
        spectral_tables,
        seed,
        n_per,
        per_batch,
        correlated,
        gi,
        si,
        density,
        material,
        tab,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
    ) -> float:
        """Generate and drain one batch of one group with a lane's resources.

        The single routing point for the four generation routes; sequential and
        lane execution differ only in which :class:`_LaneResources` they pass.
        ``group_arrays`` carries the active in-kernel route's per-group upload
        (lattice bounds or spectral aperture centres). Returns the batch's
        host-summed emitted energy (nonzero only on the pre-sampled route; the
        in-kernel routes book emitted energy on the device).
        """
        if use_lattice:
            self._generate_group_lattice(
                source,
                group_arrays,
                group_start,
                group,
                seed,
                n_per,
                per_batch,
                batch,
                correlated,
                chunk,
                block_histories,
                gi,
                si,
                density,
                material,
                tab,
                res.queues,
                res.slots,
                res.edep,
                escaped,
                unscored,
                deposited,
                violations,
                pcut,
                ecut,
                step_energy_fraction,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                device,
                res.stream,
                res.staging,
            )
            return 0.0
        if use_spectral:
            self._generate_group_spectral(
                source,
                spectral_tables,
                group_arrays,
                group_start,
                group,
                seed,
                n_per,
                per_batch,
                batch,
                correlated,
                chunk,
                block_histories,
                gi,
                si,
                density,
                material,
                tab,
                res.queues,
                res.slots,
                res.edep,
                escaped,
                unscored,
                deposited,
                violations,
                pcut,
                ecut,
                step_energy_fraction,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                emitted_counter,
                device,
                res.stream,
                res.staging,
            )
            return 0.0
        if beamlet_generator is not None:
            self._generate_group_beamlet_wrapped(
                beamlet_generator,
                group_start,
                group,
                seed,
                n_per,
                per_batch,
                batch,
                correlated,
                chunk,
                block_histories,
                gi,
                si,
                density,
                material,
                tab,
                res.queues,
                res.slots,
                res.edep,
                escaped,
                unscored,
                deposited,
                violations,
                pcut,
                ecut,
                step_energy_fraction,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                emitted_counter,
                device,
                res.stream,
                res.staging,
            )
            return 0.0
        return self._generate_group_presampled(
            source,
            group_start,
            group,
            seed,
            n_per,
            per_batch,
            batch,
            correlated,
            chunk,
            gi,
            si,
            density,
            material,
            tab,
            res.queues,
            res.slots,
            res.edep,
            escaped,
            unscored,
            deposited,
            violations,
            pcut,
            ecut,
            step_energy_fraction,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
            res.stream,
            res.staging,
        )

    def _fold_dij_batch(
        self,
        edep,
        group,
        stream,
        *,
        voxel_mass,
        n_voxels,
        per_batch,
        s1,
        s2,
        total_q,
        device,
    ) -> None:
        """Fold one batch's quanta map into the running sums (accumulate_dij_batch)."""
        _launch(
            kernels.accumulate_dij_batch,
            group * n_voxels,
            [
                edep,
                voxel_mass,
                n_voxels,
                float(per_batch),
                float(ENERGY_QUANTUM_MEV),
                s1,
                s2,
                total_q,
            ],
            device,
            stream,
        )

    def _run_dij_group_lanes(
        self,
        lane_resources,
        fold_stream,
        group_start,
        group,
        n_batches,
        chunk,
        block_histories,
        group_arrays,
        shard,
        route_args,
        fold_args,
    ) -> None:
        """Transport up to ``len(lane_resources)`` batches of one group concurrently.

        Each lane is a host thread driving its own CUDA stream: lane ``l`` takes
        batches ``l, l+lanes, ...``, so the batch-to-lane map is static and the
        transports are fully independent (private queues, slots, quanta map; the
        shared energy counters take int64 atomics, order-independent by integer
        associativity).

        The one float64 reduction — the fold of batch sums into ``s1``/``s2`` — is
        forced into batch order: a lane finishing batch ``b`` waits its turn on the
        shared counter, issues the fold on the single fold stream, and synchronizes
        it before releasing the next turn (and before reusing its own quanta map).
        The fold sequence is therefore the identical left fold the sequential path
        performs, which is what makes ``concurrent_batches`` bitwise inert
        (test-pinned). The wait costs little: folds are one cheap pass over
        ``group * n_voxels`` against a whole batch transport.

        The pre-sampled route's host-side emitted sums are collected per batch and
        folded into the shard in batch order after the join, so that float sum
        cannot depend on lane timing either.
        """
        device = route_args["device"]
        lanes = len(lane_resources)
        # The group-start zeroing and any bounds upload ran on the default stream;
        # order them before the first lane-stream launch.
        wp.synchronize_device(device)
        state = {"next_fold": 0, "failed": False}
        cond = threading.Condition()
        emitted_by_batch = [0.0] * n_batches

        def lane(lane_index: int) -> None:
            res = lane_resources[lane_index]
            try:
                for b in range(lane_index, n_batches, lanes):
                    # No per-batch clear: the group-start clear ran on the default
                    # stream (ordered by the synchronize above) and every fold since
                    # has reset the entries it consumed. A lane's own fold is awaited
                    # under ``cond`` below before it loops, so its map is clean here.
                    emitted_by_batch[b] = self._run_dij_batch(
                        res,
                        group_start,
                        group,
                        b,
                        chunk,
                        block_histories,
                        group_arrays,
                        **route_args,
                    )
                    # Transport is complete on the device here: the drain loop's
                    # final count readback synchronized the lane stream.
                    with cond:
                        while state["next_fold"] != b and not state["failed"]:
                            cond.wait()
                        if state["failed"]:
                            return
                        self._fold_dij_batch(res.edep, group, fold_stream, **fold_args)
                        wp.synchronize_stream(fold_stream)
                        state["next_fold"] = b + 1
                        cond.notify_all()
            except BaseException:
                with cond:
                    state["failed"] = True
                    cond.notify_all()
                raise

        with ThreadPoolExecutor(max_workers=lanes) as pool:
            futures = [pool.submit(lane, index) for index in range(lanes)]
            for f in futures:
                f.result()
        for emitted in emitted_by_batch:
            shard.emitted_energy += emitted

    # -- internals --------------------------------------------------------------

    def _transport_chunk(
        self,
        source,
        kind,
        seed,
        history_offset,
        n_chunk,
        gi,
        si,
        density,
        material,
        tab,
        queues,
        slots,
        edep,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
    ) -> None:
        """Generate one chunk of primaries and drain all queues."""
        q_particle, q_particle_alt, q_other, q_other_alt = queues
        for q in queues:
            _reset_count(q, device)

        if kind == PHOTON:
            q_photon, q_photon_alt = q_particle, q_particle_alt
            q_electron, q_electron_alt = q_other, q_other_alt
            target = q_photon
        else:
            q_electron, q_electron_alt = q_particle, q_particle_alt
            q_photon, q_photon_alt = q_other, q_other_alt
            target = q_electron

        self._generate(source, kind, seed, history_offset, n_chunk, target, slots, device)
        _set_count(target, n_chunk, device)

        self._drain_queues(
            q_photon,
            q_photon_alt,
            q_electron,
            q_electron_alt,
            gi,
            si,
            density,
            material,
            tab,
            slots,
            edep,
            escaped,
            unscored,
            deposited,
            violations,
            pcut,
            ecut,
            step_energy_fraction,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
        )

    def _transport_chunk_spectral(
        self,
        source,
        spectral_tables,
        seed,
        history_offset,
        n_chunk,
        gi,
        si,
        density,
        material,
        tab,
        queues,
        slots,
        edep,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
    ) -> None:
        """Generate one spectral-beam chunk in-kernel, then drain all queues.

        Photons only, at fixed thread slots into the photon queue (the mono-beam
        idiom: an explicit count write, no atomics); the kernel books each primary's
        energy into ``emitted``, since a polyenergetic ledger has no analytic total.
        """
        q_photon, q_photon_alt, q_electron, q_electron_alt = queues
        for q in queues:
            _reset_count(q, device)

        sp_edges, sp_cdf = spectral_tables
        focal = source.focal_point
        center = source.center
        u_axis = source.u_axis
        v_axis = source.v_axis
        wp.launch(
            kernels.generate_spectral_beam,
            dim=n_chunk,
            inputs=[
                seed,
                history_offset,
                sp_edges,
                sp_cdf,
                int(sp_cdf.shape[0]),
                focal[0],
                focal[1],
                focal[2],
                center[0],
                center[1],
                center[2],
                u_axis[0],
                u_axis[1],
                u_axis[2],
                v_axis[0],
                v_axis[1],
                v_axis[2],
                source.width_u,
                source.width_v,
                slots,
                q_photon,
                emitted,
            ],
            device=device,
        )
        _set_count(q_photon, n_chunk, device)

        self._drain_queues(
            q_photon,
            q_photon_alt,
            q_electron,
            q_electron_alt,
            gi,
            si,
            density,
            material,
            tab,
            slots,
            edep,
            escaped,
            unscored,
            deposited,
            violations,
            pcut,
            ecut,
            step_energy_fraction,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
        )

    def _transport_chunk_wrapped(
        self,
        generator,
        seed,
        history_offset,
        n_chunk,
        gi,
        si,
        density,
        material,
        tab,
        queues,
        slots,
        edep,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
    ) -> None:
        """Generate one chunk in-kernel via a wrapped user sampler, then drain.

        The wrapped generator (:func:`~pyRadMC.backends.warp.kernels.make_generator_kernel`)
        pushes each primary into the photon or electron queue by its kind at an atomic
        slot and books emitted weight-energy into ``emitted``; the queue counts are then
        whatever the pushes set, so — unlike the mono-beam path — no explicit count is
        written.
        """
        q_photon, q_photon_alt, q_electron, q_electron_alt = queues
        for q in queues:
            _reset_count(q, device)

        wp.launch(
            generator,
            dim=n_chunk,
            inputs=[seed, history_offset, q_photon, q_electron, slots, emitted],
            device=device,
        )

        self._drain_queues(
            q_photon,
            q_photon_alt,
            q_electron,
            q_electron_alt,
            gi,
            si,
            density,
            material,
            tab,
            slots,
            edep,
            escaped,
            unscored,
            deposited,
            violations,
            pcut,
            ecut,
            step_energy_fraction,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
        )

    def _transport_chunk_presampled(
        self,
        source,
        seed,
        history_offset,
        n_chunk,
        gi,
        si,
        density,
        material,
        tab,
        queues,
        slots,
        edep,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
    ) -> float:
        """Host-sample a chunk via ``source.sample_batch``, seed queues by kind, drain.

        The general (pre-sampling) route: works for any source with a ``sample_batch``
        — a phase space, or a user :class:`~pyRadMC.geometry.source.Source` using the
        emit-based default. Returns the chunk's emitted energy: sum of
        ``weight * energy`` over the sampled records, plus ``weight * 2 m_e c^2`` for
        each positron whose annihilation photons the device books into
        deposited/escaped (so the emitted = deposited + escaped ledger closes, as on
        the reference backend).
        """
        q_photon, q_photon_alt, q_electron, q_electron_alt = queues
        for q in queues:
            _reset_count(q, device)

        batch = source.sample_batch(seed, history_offset, n_chunk)
        pt = batch["particle_type"]
        hist = (history_offset + np.arange(n_chunk, dtype=np.int64)).astype(np.int32)
        transport_kind = _IAEA_TO_TRANSPORT[pt]

        photon = pt == 1  # IAEA photon; electrons (2) and positrons (3) share the e- queue
        charged = ~photon
        zero_tag = np.zeros(n_chunk, dtype=np.int32)  # a plain run scores one column
        self._seed_queue(
            q_photon,
            seed,
            hist[photon],
            zero_tag[photon],
            transport_kind[photon],
            batch,
            photon,
            device,
        )
        self._seed_queue(
            q_electron,
            seed,
            hist[charged],
            zero_tag[charged],
            transport_kind[charged],
            batch,
            charged,
            device,
        )

        self._drain_queues(
            q_photon,
            q_photon_alt,
            q_electron,
            q_electron_alt,
            gi,
            si,
            density,
            material,
            tab,
            slots,
            edep,
            escaped,
            unscored,
            deposited,
            violations,
            pcut,
            ecut,
            step_energy_fraction,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
        )

        latent = np.where(pt == 3, 2.0 * ELECTRON_MASS_MEV, 0.0)
        return float(
            np.sum(
                batch["weight"].astype(np.float64) * (batch["energy"].astype(np.float64) + latent)
            )
        )

    def _transport_chunk_device_buffer(
        self,
        source,
        seed,
        history_offset,
        n_chunk,
        gi,
        si,
        density,
        material,
        tab,
        queues,
        slots,
        edep,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
    ) -> None:
        """Seed the transport queues straight from a device pre-solve buffer, then drain.

        The no-copy path (``DevicePhaseSpace``): no host ``sample_batch`` and no
        re-upload — one kernel samples a record per history from the on-device
        population and atomic-appends it to the photon or electron queue as a source
        primary, booking the emitted weight-energy into ``emitted``. Transport is
        then identical to every other route.
        """
        q_photon, q_photon_alt, q_electron, q_electron_alt = queues
        for q in queues:
            _reset_count(q, device)
        buf = source.buffer
        wp.launch(
            kernels.generate_from_exit_buffer,
            dim=n_chunk,
            inputs=[
                seed,
                seed ^ _PRESOLVE_SAMPLING_SALT,
                history_offset,
                source.count,
                buf.particle_type,
                buf.energy,
                buf.x,
                buf.y,
                buf.z,
                buf.ux,
                buf.uy,
                buf.uz,
                buf.weight,
                q_photon,
                q_electron,
                slots,
                emitted,
            ],
            device=device,
        )
        self._drain_queues(
            q_photon,
            q_photon_alt,
            q_electron,
            q_electron_alt,
            gi,
            si,
            density,
            material,
            tab,
            slots,
            edep,
            escaped,
            unscored,
            deposited,
            violations,
            pcut,
            ecut,
            step_energy_fraction,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
        )

    def _seed_queue(
        self, queue, seed, hist_g, beamlet_g, kind_g, batch, mask, device, stream=None
    ) -> None:
        """Upload one kind-group's primaries (tagged by ``beamlet_g``) into ``queue``."""
        n = int(hist_g.shape[0])
        if n == 0:
            _set_count(queue, 0, device, stream)
            return
        uploads = {
            name: wp.array(batch[name][mask], dtype=float, device=device)
            for name in ("energy", "x", "y", "z", "ux", "uy", "uz", "weight")
        }
        uploads["hist"] = wp.array(hist_g, dtype=wp.int32, device=device)
        uploads["beamlet"] = wp.array(beamlet_g, dtype=wp.int32, device=device)
        uploads["kind"] = wp.array(kind_g, dtype=wp.int32, device=device)
        if stream is not None:
            # Host uploads run on the device's *default* stream; the launch below
            # runs on the lane's. Wait for the copies (only — other lanes' streams
            # are untouched) so the lane cannot read a half-arrived buffer.
            wp.synchronize_stream(wp.get_stream(device))
        _launch(
            kernels.generate_from_upload,
            n,
            [
                seed,
                uploads["hist"],
                uploads["beamlet"],
                uploads["kind"],
                uploads["energy"],
                uploads["x"],
                uploads["y"],
                uploads["z"],
                uploads["ux"],
                uploads["uy"],
                uploads["uz"],
                uploads["weight"],
                queue,
            ],
            device,
            stream,
        )
        _set_count(queue, n, device, stream)

    def _drain_queues(
        self,
        q_photon,
        q_photon_alt,
        q_electron,
        q_electron_alt,
        gi,
        si,
        density,
        material,
        tab,
        slots,
        edep,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
        stream=None,
        staging=None,
    ) -> None:
        """Ping-pong the photon and electron kernels until every queue is empty.

        With ``stream`` set (a Dij batch lane), every launch, count reset, and
        count readback is ordered on that stream, so lanes on the same device
        never observe each other's queues.
        """
        while True:
            n_photon = _queue_count(q_photon, stream, staging)
            if n_photon > 0:
                _launch(
                    kernels.photon_kernel,
                    n_photon,
                    [
                        gi,
                        si,
                        density,
                        material,
                        tab,
                        q_photon,
                        q_electron,
                        q_photon_alt,
                        slots,
                        edep,
                        escaped,
                        unscored,
                        deposited,
                        pcut,
                        ecut,
                        1 if transport_electrons else 0,
                        dose_to_water,
                        violations,
                    ],
                    device,
                    stream,
                )
                _reset_count(q_photon, device, stream)
                q_photon, q_photon_alt = q_photon_alt, q_photon

            n_electron = _queue_count(q_electron, stream, staging)
            if n_electron > 0:
                _launch(
                    kernels.electron_kernel,
                    n_electron,
                    [
                        gi,
                        si,
                        density,
                        material,
                        tab,
                        q_electron,
                        q_photon,
                        q_electron_alt,
                        slots,
                        edep,
                        escaped,
                        unscored,
                        deposited,
                        pcut,
                        ecut,
                        step_energy_fraction,
                        msc_model_gs,
                        dose_to_water,
                    ],
                    device,
                    stream,
                )
                _reset_count(q_electron, device, stream)
                q_electron, q_electron_alt = q_electron_alt, q_electron

            if n_photon == 0 and n_electron == 0:
                break

    def _generate(self, source, kind, seed, history_offset, n, queue, slots, device) -> None:
        if isinstance(source, PencilBeamSource):
            wp.launch(
                kernels.generate_pencil_beam,
                dim=n,
                inputs=[
                    seed,
                    history_offset,
                    kind,
                    source.energy,
                    source.position[0],
                    source.position[1],
                    source.position[2],
                    source.direction[0],
                    source.direction[1],
                    source.direction[2],
                    queue,
                ],
                device=device,
            )
        elif isinstance(source, ParallelBeamSource):
            wp.launch(
                kernels.generate_parallel_beam,
                dim=n,
                inputs=[
                    seed,
                    history_offset,
                    kind,
                    source.energy,
                    source.z,
                    source.x_range[0],
                    source.x_range[1] - source.x_range[0],
                    source.y_range[0],
                    source.y_range[1] - source.y_range[0],
                    slots,
                    queue,
                ],
                device=device,
            )
        else:
            raise ValueError(f"unsupported source type {type(source).__name__}")

    def _upload_lattice_bounds(self, source, group_start, group, device):
        """Upload one group's beamlet bounds once, shared by every batch and lane.

        Hoisted out of the per-batch generation for two reasons: the bounds do not
        change across batches, and host uploads run on the device's *default*
        stream — done here, before any lane stream launches, the single
        ``synchronize_device`` in the group loop orders them for every lane.
        """
        bounds = np.array(
            [source.beamlet_bounds(group_start + k) for k in range(group)], dtype=np.float64
        )
        return (
            wp.array(bounds[:, 0].astype(np.float32), dtype=float, device=device),
            wp.array((bounds[:, 1] - bounds[:, 0]).astype(np.float32), dtype=float, device=device),
            wp.array(bounds[:, 2].astype(np.float32), dtype=float, device=device),
            wp.array((bounds[:, 3] - bounds[:, 2]).astype(np.float32), dtype=float, device=device),
        )

    def _generate_group_lattice(
        self,
        source,
        lattice_bounds,
        group_start,
        group,
        seed,
        n_per,
        per_batch,
        batch,
        correlated,
        chunk,
        block_histories,
        gi,
        si,
        density,
        material,
        tab,
        queues,
        slots,
        edep,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
        stream=None,
        staging=None,
    ) -> None:
        """Built-in lattice: generate one batch of the group's block in-kernel, in chunks."""
        x_lo, x_extent, y_lo, y_extent = lattice_bounds
        t = 0
        while t < block_histories:
            n_chunk = min(chunk, block_histories - t)
            for q in queues:
                _reset_count(q, device, stream)
            _launch(
                kernels.generate_beamlet_lattice,
                n_chunk,
                [
                    seed,
                    group_start,
                    n_per,
                    per_batch,
                    batch,
                    t,
                    1 if correlated else 0,
                    source.energy,
                    source.z,
                    x_lo,
                    x_extent,
                    y_lo,
                    y_extent,
                    slots,
                    queues[0],
                ],
                device,
                stream,
            )
            _set_count(queues[0], n_chunk, device, stream)
            self._drain_queues(
                queues[0],
                queues[1],
                queues[2],
                queues[3],
                gi,
                si,
                density,
                material,
                tab,
                slots,
                edep,
                escaped,
                unscored,
                deposited,
                violations,
                pcut,
                ecut,
                step_energy_fraction,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                device,
                stream,
                staging,
            )
            t += n_chunk

    def _upload_spectral_centers(self, source, group_start, group, device):
        """Upload one group's aperture centres once, shared by every batch and lane.

        The spectral twin of :meth:`_upload_lattice_bounds`, with the same
        default-stream ordering rationale.
        """
        centers = np.array(
            [source.centers[group_start + k] for k in range(group)], dtype=np.float64
        )
        return (
            wp.array(centers[:, 0].astype(np.float32), dtype=float, device=device),
            wp.array(centers[:, 1].astype(np.float32), dtype=float, device=device),
            wp.array(centers[:, 2].astype(np.float32), dtype=float, device=device),
        )

    def _generate_group_spectral(
        self,
        source,
        spectral_tables,
        group_arrays,
        group_start,
        group,
        seed,
        n_per,
        per_batch,
        batch,
        correlated,
        chunk,
        block_histories,
        gi,
        si,
        density,
        material,
        tab,
        queues,
        slots,
        edep,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        stream=None,
        staging=None,
    ) -> None:
        """Built-in spectral Dij route: generate one batch of the group in-kernel.

        Same per-batch block chunking as the lattice, with the spectrum inversion
        and divergent-fan geometry inline
        (:func:`~pyRadMC.backends.warp.kernels.generate_spectral_beamlets`); the
        kernel books emitted energy into ``emitted`` per primary.
        """
        sp_edges, sp_cdf = spectral_tables
        cx, cy, cz = group_arrays
        focal = source.focal_point
        u_axis = source.u_axis
        v_axis = source.v_axis
        t = 0
        while t < block_histories:
            n_chunk = min(chunk, block_histories - t)
            for q in queues:
                _reset_count(q, device, stream)
            _launch(
                kernels.generate_spectral_beamlets,
                n_chunk,
                [
                    seed,
                    group_start,
                    n_per,
                    per_batch,
                    batch,
                    t,
                    1 if correlated else 0,
                    sp_edges,
                    sp_cdf,
                    int(sp_cdf.shape[0]),
                    focal[0],
                    focal[1],
                    focal[2],
                    u_axis[0],
                    u_axis[1],
                    u_axis[2],
                    v_axis[0],
                    v_axis[1],
                    v_axis[2],
                    source.width_u,
                    source.width_v,
                    cx,
                    cy,
                    cz,
                    slots,
                    emitted,
                    queues[0],
                ],
                device,
                stream,
            )
            _set_count(queues[0], n_chunk, device, stream)
            self._drain_queues(
                queues[0],
                queues[1],
                queues[2],
                queues[3],
                gi,
                si,
                density,
                material,
                tab,
                slots,
                edep,
                escaped,
                unscored,
                deposited,
                violations,
                pcut,
                ecut,
                step_energy_fraction,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                device,
                stream,
                staging,
            )
            t += n_chunk

    def _generate_group_beamlet_wrapped(
        self,
        generator,
        group_start,
        group,
        seed,
        n_per,
        per_batch,
        batch,
        correlated,
        chunk,
        block_histories,
        gi,
        si,
        density,
        material,
        tab,
        queues,
        slots,
        edep,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        stream=None,
        staging=None,
    ) -> None:
        """Advanced Dij route: generate one batch of the group in-kernel via a sampler.

        Same per-batch block chunking as the built-in lattice, but the per-primary
        position comes from the user's ``warp_beamlet_sampler`` (wrapped by
        :func:`~pyRadMC.backends.warp.kernels.make_beamlet_generator_kernel`) instead of
        analytic bounds; the wrapper books emitted energy into ``emitted``.
        """
        t = 0
        while t < block_histories:
            n_chunk = min(chunk, block_histories - t)
            for q in queues:
                _reset_count(q, device, stream)
            _launch(
                generator,
                n_chunk,
                [
                    seed,
                    group_start,
                    n_per,
                    per_batch,
                    batch,
                    t,
                    1 if correlated else 0,
                    slots,
                    emitted,
                    queues[0],
                ],
                device,
                stream,
            )
            _set_count(queues[0], n_chunk, device, stream)
            self._drain_queues(
                queues[0],
                queues[1],
                queues[2],
                queues[3],
                gi,
                si,
                density,
                material,
                tab,
                slots,
                edep,
                escaped,
                unscored,
                deposited,
                violations,
                pcut,
                ecut,
                step_energy_fraction,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                device,
                stream,
                staging,
            )
            t += n_chunk

    def _generate_group_presampled(
        self,
        source,
        group_start,
        group,
        seed,
        n_per,
        per_batch,
        batch,
        correlated,
        chunk,
        gi,
        si,
        density,
        material,
        tab,
        queues,
        slots,
        edep,
        escaped,
        unscored,
        deposited,
        violations,
        pcut,
        ecut,
        step_energy_fraction,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
        stream=None,
        staging=None,
    ) -> float:
        """Host pre-sample one batch of each beamlet's primaries and upload them.

        For beamlet ``j`` the within-beamlet index ``r = batch*per_batch + i`` keys the
        transport RNG on ``r`` (correlated) or ``h = j*n_per + r`` (independent) — the
        mapping ``ReferenceEngine.run_dij`` defines — and the column tag is ``local``,
        the batch being separated in time instead (see ``accumulate_dij_batch``), so the
        result is bit-invariant to grouping/chunking. Only this batch's ``per_batch``
        primaries are sampled, at history offset ``offset + batch*per_batch``:
        ``_presample`` keys each history on ``init_state(seed, history_offset + i)``, so
        a per-batch sub-range yields exactly the primaries the full-range sample would
        have put at those indices. Every beamlet primary is a photon at the sampled
        energy/position carrying the sampled statistical weight (the collimated sources
        attenuate by weight), and the emitted book sums ``weight * energy``, matching
        the reference Dij. Returns this batch's emitted energy for the group.
        """
        emitted = 0.0
        for local in range(group):
            beamlet = group_start + local
            offset = (0 if correlated else beamlet * n_per) + batch * per_batch
            cols = source.sample_beamlet_batch(seed, offset, per_batch, beamlet)
            emitted += float(
                np.sum(cols["weight"].astype(np.float64) * cols["energy"].astype(np.float64))
            )
            i = np.arange(per_batch, dtype=np.int64)
            key = (offset + i).astype(np.int32)
            tag = np.full(per_batch, local, dtype=np.int32)
            r0 = 0
            while r0 < per_batch:
                r1 = min(r0 + chunk, per_batch)
                nc = r1 - r0
                sub = slice(r0, r1)
                sub_batch = {
                    name: cols[name][sub]
                    for name in ("energy", "x", "y", "z", "ux", "uy", "uz", "weight")
                }
                for q in queues:
                    _reset_count(q, device, stream)
                self._seed_queue(
                    queues[0],
                    seed,
                    key[sub],
                    tag[sub],
                    np.full(nc, PHOTON, dtype=np.int32),
                    sub_batch,
                    np.ones(nc, dtype=bool),
                    device,
                    stream,
                )
                self._drain_queues(
                    queues[0],
                    queues[1],
                    queues[2],
                    queues[3],
                    gi,
                    si,
                    density,
                    material,
                    tab,
                    slots,
                    edep,
                    escaped,
                    unscored,
                    deposited,
                    violations,
                    pcut,
                    ecut,
                    step_energy_fraction,
                    msc_model_gs,
                    transport_electrons,
                    dose_to_water,
                    device,
                    stream,
                    staging,
                )
                r0 = r1
        return emitted

    def _upload_grid(self, device: str):
        gi = GridInfo()
        gi.x_lo, gi.y_lo, gi.z_lo = self.grid.origin
        gi.x_hi, gi.y_hi, gi.z_hi = self.grid.upper_corner
        gi.sx, gi.sy, gi.sz = self.grid.spacing
        gi.nx = self.grid.shape[0]
        gi.ny = self.grid.shape[1]
        gi.nz = self.grid.shape[2]
        gi.n_voxels = int(np.prod(self.grid.shape))
        gi.min_spacing = min(self.grid.spacing)
        density = wp.array(self.grid.density.astype(np.float32), dtype=float, device=device)
        # Material indices travel as uint8: 4x narrower than int32 on the transport
        # kernels' random per-step load, and lossless for any registry the tables can
        # hold. astype would silently *truncate* an index above 255 into a different
        # valid-looking material, so an out-of-range source is refused here, before
        # anything is transported.
        if self.cross_sections.n_materials > 256:
            raise ValueError(
                f"the Warp backend addresses at most 256 materials (uint8 voxel map); "
                f"this source declares {self.cross_sections.n_materials}"
            )
        material = wp.array(self.grid.material.astype(np.uint8), dtype=wp.uint8, device=device)
        return gi, density, material

    def _upload_tables(
        self, e_max: float, pcut: float, ecut: float, device: str, with_gs: bool = False
    ) -> Tables:
        key = (ecut, pcut, e_max)
        with self._table_cache_lock:
            host = self._table_cache.get(key)
            if host is None:
                host = self.cross_sections.build_tables(
                    ecut=ecut, pcut=pcut, e_max=e_max * _E_MAX_MARGIN
                )
                self._table_cache[key] = host
        tab = Tables()
        for table_name in (
            "mu_compton",
            "mu_photo",
            "mu_pair",
            "mu_rayleigh",
            "majorant",
            "stopping_restricted",
            "stopping_radiative",
            "moller",
            "csda_range",
            "scattering_power",
            "log_eta",
            "restricted_range",
            "energy_of_restricted_range",
        ):
            setattr(
                tab,
                table_name,
                wp.array(getattr(host, table_name).astype(np.float32), dtype=float, device=device),
            )
        tab.coherent_x = wp.array(host.coherent_x.astype(np.float32), dtype=float, device=device)
        tab.coherent_cumulative = wp.array(
            host.coherent_cumulative.astype(np.float32), dtype=float, device=device
        )
        tab.p_log_e_min = host.photon_log_e_min
        tab.p_inv_dlog = host.photon_inv_dlog
        tab.e_log_e_min = host.electron_log_e_min
        tab.e_inv_dlog = host.electron_inv_dlog
        tab.r_log_min = host.range_log_r_min
        tab.r_inv_dlog = host.range_inv_dlog
        tab.n_points = host.n_points
        tab.n_coherent = host.n_coherent
        if with_gs:
            gs_host = self._gs_grid_host(host, ecut, e_max)
            tab.gs_values = wp.array(gs_host.values.astype(np.float32), dtype=float, device=device)
            tab.gs_ix0 = gs_host.ix0
            tab.gs_iy0 = gs_host.iy0
            tab.gs_n_eta = gs_host.values.shape[0]
            tab.gs_n_theta2 = gs_host.values.shape[1]
            tab.gs_n_u = gs_host.n_u
        else:
            # Placeholder the kernel never reads: the branch is on the launch
            # flag, but a wp.struct must carry a valid array on every field.
            tab.gs_values = wp.zeros((1, 1, 1), dtype=float, device=device)
            tab.gs_ix0 = 0
            tab.gs_iy0 = 0
            tab.gs_n_eta = 1
            tab.gs_n_theta2 = 1
            tab.gs_n_u = 1
        return tab

    def _gs_grid_host(self, host, ecut: float, e_max: float):
        """Return the eager Goudsmit-Saunderson grid for these tables, built once.

        Built host-side **before any launch** — never lazily mid-transport (an
        86 s stall, measured) — and cached under the engine's table lock, so
        concurrent ``devices=[...]`` shard threads share one immutable build
        (the thread-safety hazard a lazily growing per-source dict had). The
        window covers exactly the keys the kernel's lookups can produce for
        the materials present in this engine's voxel map; the fixed build seed
        makes every node bit-identical to the reference backend's lazy cache.
        """
        from pyRadMC.data.goudsmit_saunderson import build_gs_grid, gs_window_from_tables

        key = ("gs", ecut, e_max)
        with self._table_cache_lock:
            gs_host = self._table_cache.get(key)
            if gs_host is None:
                present = np.unique(self.grid.material).astype(int).tolist()
                gs_host = build_gs_grid(*gs_window_from_tables(host, present))
                self._table_cache[key] = gs_host
        return gs_host
