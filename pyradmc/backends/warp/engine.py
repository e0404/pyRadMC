"""The Warp engine: launch and memory management for the transport kernels.

Production backend, ``cpu`` and ``cuda`` from one kernel source (AGENTS.md section 3:
nothing physical lives here). The run loop is:

1. Flatten the cross-sections (``build_tables``) and upload everything as float32 —
   once per engine and device: tables, voxel maps and the transport workspaces stay
   resident between calls (see :class:`WarpEngine`).
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

from pyradmc import (
    DIJ_TRUNCATION_RELATIVE,
    ECUT_MEV,
    ELECTRON_MASS_MEV,
    PCUT_MEV,
    __version__,
)
from pyradmc.backends.results import RunProvenance, TransportResult
from pyradmc.backends.warp import kernels
from pyradmc.backends.warp.kernels import ENERGY_QUANTUM_MEV, GridInfo, Queue, Tables
from pyradmc.data.interface import CrossSectionSource
from pyradmc.geometry.collimation import TransmissionMaskSource
from pyradmc.geometry.field import FIELD_NONE, MagneticField
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.segments import SegmentedFanSource, SegmentedPlanSource
from pyradmc.geometry.source import (
    BeamletGridSource,
    BeamletSource,
    ParallelBeamSource,
    PencilBeamSource,
    Source,
    SpectralBeamletSource,
    SpectralBeamSource,
)
from pyradmc.progress import ProgressCallback, ProgressEmitter
from pyradmc.scoring.cylinder import CylindricalScoringGrid
from pyradmc.scoring.dij import DijAssembler, DijResult
from pyradmc.scoring.dose import ScoringGeometry
from pyradmc.scoring.dose_to_water import validate_scoring_mode
from pyradmc.scoring.grid import ScoringGrid
from pyradmc.transport.electron import default_step_energy_fraction
from pyradmc.transport.particles import ELECTRON, PHOTON, POSITRON

__all__ = ["WarpEngine"]

_E_MAX_MARGIN = 1.0 + 1.0e-6  # table upper edge strictly above the primary energy

# Salt so the device pre-solve buffer's record-sampling stream is independent of the
# transport stream, which is seeded from the bare (seed, history) — a within-device
# concern only; cross-target agreement is statistical (AGENTS.md 2.3).
_PRESOLVE_SAMPLING_SALT = 0x50524553

# IAEA particle code (from a source's sample_batch columns) -> transport constant,
# as a lookup indexed by the code (1/2/3); slot 0 is unused.
_IAEA_TO_TRANSPORT = np.array([-1, PHOTON, ELECTRON, POSITRON], dtype=np.int32)


def _no_shells(device: str):
    """One-element placeholder for the bin-edge fields of a non-cylindrical grid.

    Warp struct array fields must reference real storage even where no kernel reads
    them; the rectilinear branch of ``kernels._scoring_index`` never touches these,
    exactly as the Gaussian-MSC path never touches the GS table placeholder.
    """
    return wp.zeros(1, dtype=float, device=device)


def _no_field(device: str):
    """One-cell placeholder for ``GridInfo.b_values`` where no map is read.

    Same contract as :func:`_no_shells`: a struct array field needs real storage even
    on the paths (scoring copies, the field-free world) that never index it.
    """
    return wp.zeros((1, 1, 1), dtype=wp.vec3, device=device)


def _fill_no_field(gi: GridInfo, device: str) -> None:
    """Set the magnetic-field members of ``gi`` to the inert field-free values."""
    gi.b_mode = FIELD_NONE
    gi.b0x = gi.b0y = gi.b0z = 0.0
    gi.b_ox = gi.b_oy = gi.b_oz = 0.0
    gi.b_sx = gi.b_sy = gi.b_sz = 1.0
    gi.b_nx = gi.b_ny = gi.b_nz = 1
    gi.b_values = _no_field(device)


def _scoring_info(scoring: ScoringGeometry, device: str) -> GridInfo:
    """Scalar metadata of the scoring geometry, in the same struct as the transport grid.

    The kernels' ``_deposit`` routes positions through this exactly as the host
    scorer's ``deposit_at`` does; only the scalars (and, for a cylinder, the small
    shell-edge array) travel — the mass map stays on the host, where the dose
    normalization happens.

    A cylindrical geometry keeps ``z_lo``/``z_hi``/``nz`` as its depth extent and bin
    count but uploads the depth boundaries themselves, because the bins are graded;
    the lateral fields carry the axis and the shell edges, uploaded already squared
    so no root is taken per deposit. ``sx``/``sy``/``sz``, ``nx`` and ``ny`` are never
    read in this mode.
    """
    si = GridInfo()
    _fill_no_field(si, device)  # scoring never reads the field
    if isinstance(scoring, CylindricalScoringGrid):
        si.mode = 1
        si.x_lo = si.y_lo = si.x_hi = si.y_hi = 0.0
        si.sx = si.sy = 1.0
        si.nx = scoring.n_depth
        si.ny = scoring.n_shells
        si.z_lo = scoring.depth_origin
        si.z_hi = scoring.depth_upper
        si.sz = 1.0
        si.nz = scoring.n_depth
        si.z_edges = wp.array(scoring.depth_edges.astype(np.float32), dtype=float, device=device)
        si.axis_x, si.axis_y = scoring.axis
        si.n_shells = scoring.n_shells
        edges_sq = scoring.radial_edges.astype(np.float32) ** 2
        si.r_inner_sq = float(edges_sq[0])
        si.r_outer_sq = float(edges_sq[-1])
        si.r_edges_sq = wp.array(edges_sq, dtype=float, device=device)
        si.n_voxels = scoring.n_voxels
        si.min_spacing = float(scoring.depth_thickness.min())  # unused; kept coherent
        return si
    si.mode = 0
    si.x_lo, si.y_lo, si.z_lo = scoring.origin
    si.x_hi, si.y_hi, si.z_hi = scoring.upper_corner
    si.sx, si.sy, si.sz = scoring.spacing
    si.nx, si.ny, si.nz = scoring.shape
    si.n_voxels = scoring.n_voxels
    si.min_spacing = min(scoring.spacing)  # unused for scoring; kept coherent
    si.axis_x = si.axis_y = 0.0
    si.n_shells = 0
    si.r_inner_sq = si.r_outer_sq = 0.0
    si.r_edges_sq = _no_shells(device)
    si.z_edges = _no_shells(device)
    return si


def _validate_msc_model(msc_model: str) -> int:
    """Map the ``msc_model`` name to the kernel's launch flag; refuse unknowns.

    Same names and refusal wording as the reference loop
    (:func:`pyradmc.transport.electron.electron_steps`), but checked *before*
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
    ``workspace`` is the pooled :class:`_Workspace` behind the lane; the drain loop
    takes its readback buffers from it.
    """

    stream: object
    staging: object
    queues: list
    slots: object
    edep: object
    workspace: object = None


class _Workspace:
    """One batch lane's transport scratch, kept resident between calls.

    Four queues (two ping-pong pairs), the RNG-slot array, the scalar energy and
    violation counters, and — created on first concurrent use — the lane's stream
    and pinned count-staging element. Pooled per engine and device
    (:meth:`WarpEngine._acquire_workspaces`), so a loop of calls stops paying a
    fresh allocation of every queue column per call.

    Every call refits it *exactly* to its own requirement, growing or shrinking:
    the pool holds what the last call needed, never the largest call so far. A
    steady workload (an optimizer's loop of equal calls) therefore reuses it
    untouched, while an alternating big/small workload reallocates on each change,
    as every call did before the pool existed. Holding the largest-ever size
    instead parked up to 1.7 GB behind tiny calls, and let a Dij's dense group maps
    ride on top of a big run's queues (peak 1.33 -> 2.05 GB, measured). An exact
    fit also makes each queue's ``capacity`` — the kernels' write guard and the
    engine's overflow bound — the one a fresh engine would allocate.

    Contents need no clearing between calls: every generator and every push
    writes all fields of a slot before a count publishes it, the transport reads
    only below the count, every RNG slot is written by its generator before it is
    read, and each counter is zeroed by the call that uses it.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.slots_allocated = 0
        self.queues: list = []
        self.slots = None
        self.stream = None
        self.staging = None
        self.fold = None
        self.drain_status = None
        self.drain_host = None
        self._book = None
        self._book_views: dict = {}

    @property
    def nbytes(self) -> int:
        """Per-slot bytes held, in the accounting of :data:`_QUEUE_BYTES_PER_SLOT`.

        The handful of scalar counters is left out, as in the sizing model.
        """
        return self.slots_allocated * _QUEUE_BYTES_PER_SLOT

    def fit(self, capacity: int) -> None:
        """Hold exactly ``capacity`` slots per queue, reallocating only on a change."""
        if capacity != self.slots_allocated:
            # Release before reallocating, so the peak is the new size, not old + new.
            self.queues = []
            self.slots = None
            self.slots_allocated = 0
            self.queues = [_upload_queue(capacity, self.device) for _ in range(4)]
            self.slots = wp.zeros(capacity, dtype=wp.uint32, device=self.device)
            self.slots_allocated = capacity
        for q in self.queues:
            q.capacity = capacity

    def fold_stream(self):
        """Return the stream a concurrent call folds its lanes' batches on (made once)."""
        if self.fold is None:
            self.fold = wp.Stream(self.device)
        return self.fold

    def drain_buffers(self, stream):
        """Return the drain's (device status, pinned host element) pair, made on demand.

        The status array is created on the first blind drain and the pinned element
        on the first one that reads back through a lane stream, like
        :meth:`fold_stream`; neither is tied to :meth:`fit`, so a capacity change
        keeps them. The drain zero-fills the status at its start, so nothing stale
        carries over between drains or calls.
        """
        if self.drain_status is None:
            self.drain_status = wp.zeros(
                kernels.DRAIN_STATUS_SLOTS, dtype=wp.int32, device=self.device
            )
            if stream is not None:
                # Allocated (and zeroed) on the device's default stream, then used
                # on the lane's: wait for the default stream once, as uploads do.
                wp.synchronize_stream(wp.get_stream(self.device))
        if stream is not None and self.drain_host is None:
            self.drain_host = wp.zeros(
                kernels.DRAIN_STATUS_SLOTS, dtype=wp.int32, device="cpu", pinned=True
            )
        return self.drain_status, self.drain_host

    def book(self, name: str) -> wp.array:
        """Return the one-element int64 energy counter ``name`` (a view of the book).

        The counters share one array so a call clears them in one memset and reads
        them in one readback instead of one synchronizing ``numpy()`` each. Each
        sits on its own 128-byte line, so the kernels' atomics on different
        counters never share a cache line — the layout the kernels had when
        every counter was its own allocation.
        """
        if self._book is None:
            book = wp.zeros(len(_BOOK) * _BOOK_STRIDE, dtype=wp.int64, device=self.device)
            views = {
                key: book[i * _BOOK_STRIDE : i * _BOOK_STRIDE + 1] for i, key in enumerate(_BOOK)
            }
            # The kernels count majorant violations in an int32; that counter is the
            # low half of a zeroed int64 slot (little-endian on every Warp target),
            # so the readback every pass already makes checks it for free. It only
            # grows and is never cleared per pass; any nonzero value is a violation.
            # The view does not own the memory: ``_book`` keeps it alive.
            views["violations"] = wp.array(
                ptr=book.ptr + _BOOK.index("violations") * _BOOK_STRIDE * 8,
                dtype=wp.int32,
                shape=(1,),
                device=self.device,
            )
            self._book, self._book_views = book, views
        return self._book_views[name]

    def clear_pass_books(self) -> None:
        """Zero the per-pass counters (the first :data:`_BOOK_PER_PASS` of the book)."""
        self.book(_BOOK[0])
        self._book[: _BOOK_PER_PASS * _BOOK_STRIDE].zero_()

    def clear_books(self) -> None:
        """Zero every counter in the book, the violation count included."""
        self.book(_BOOK[0])
        self._book.zero_()

    def read_books(self) -> dict[str, int]:
        """Read every counter back in one transfer (synchronizes the default stream)."""
        self.book(_BOOK[0])
        host = self._book.numpy()
        return {key: int(host[i * _BOOK_STRIDE]) for i, key in enumerate(_BOOK)}

    def lane(self, edep, concurrent: bool) -> _LaneResources:
        """View the workspace as a lane, on its own stream only when lanes overlap."""
        if not concurrent:
            return _LaneResources(
                stream=None,
                staging=None,
                queues=self.queues,
                slots=self.slots,
                edep=edep,
                workspace=self,
            )
        if self.stream is None:
            self.stream = wp.Stream(self.device)
            self.staging = wp.zeros(1, dtype=wp.int32, device="cpu", pinned=True)
        return _LaneResources(
            stream=self.stream,
            staging=self.staging,
            queues=self.queues,
            slots=self.slots,
            edep=edep,
            workspace=self,
        )


_BOOK = ("escaped", "unscored", "deposited", "total_q", "dep_total", "emitted", "violations")
"""The counters a workspace keeps in one int64 array. The first
:data:`_BOOK_PER_PASS` are cleared and read once per pass — a batch of ``run``, a
group of ``run_dij`` — so no int64 sum spans more than the pass it always did; the
rest accumulate over the whole call. ``violations`` is the kernels' int32
majorant-violation count, aliased onto its slot (see :meth:`_Workspace.book`), so
the per-pass readback raises on a violation in the pass that made it."""

_BOOK_PER_PASS = 4

_BOOK_STRIDE = 16
"""int64 elements between counters: 128 bytes, one cache line each."""


@dataclass
class _DeviceState:
    """What one engine keeps resident on one device between calls.

    The tables (per ``(ecut, pcut, e_max, with_gs)``, least recently used evicted
    past :data:`_TABLES_RESIDENT`) and the transport grid are functions of the
    engine's frozen fields plus the key. The scoring geometry and the spectra are
    per-call arguments instead, matched by identity with a strong reference held
    (so the identity cannot be recycled) — safe because their arrays are
    read-only. The idle transport workspaces are what the last call used.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    grid: tuple | None = None
    tables: dict = field(default_factory=dict)
    table_arrays: dict = field(default_factory=dict)
    scoring: tuple | None = None
    spectra: dict = field(default_factory=dict)
    idle: list = field(default_factory=list)


_SPECTRA_RESIDENT = 8
"""Spectrum tables kept per device; a few hundred bytes each, bounded anyway."""

_TABLES_RESIDENT = 4
"""Table structs kept per device (and twice as many base or GS array sets), least
recently used evicted. Each GS grid is about 12 MB of float32 on the device, so an
unbounded cache would park one per ECUT of a sweep; the host cache keeps the
flattened tables, so an evicted key costs only its re-upload."""


def _lru_get(cache: dict, key):
    """Look ``key`` up in an insertion-ordered dict, marking it most recently used."""
    value = cache.pop(key, None)
    if value is not None:
        cache[key] = value
    return value


def _lru_put(cache: dict, key, value, bound: int) -> None:
    """Insert ``key`` as most recently used, evicting the oldest beyond ``bound``."""
    cache[key] = value
    while len(cache) > bound:
        cache.pop(next(iter(cache)))


def _check_majorant(books: dict[str, int]) -> None:
    """Raise if the kernels counted a Woodcock majorant violation (book readback)."""
    if books["violations"] != 0:
        raise RuntimeError(
            "Woodcock majorant violated in the kernel despite table headroom. "
            "The geometry contains material or density the majorant "
            "declaration did not cover."
        )


def _launch(kernel, dim, inputs, device: str, stream, max_blocks: int = 0) -> None:
    """Launch on the device's current stream, or on an explicit lane stream.

    Warp takes *either* ``device`` or ``stream`` (a stream names its device);
    every launch on a Dij code path routes through here so that a batch lane's
    work is ordered on the lane's stream and nothing leaks onto the default one.
    ``max_blocks`` caps the CUDA grid (Warp's grid-stride loop covers the rest of
    ``dim``); 0 is Warp's default of one thread per index.
    """
    if stream is None:
        wp.launch(kernel, dim=dim, inputs=inputs, device=device, max_blocks=max_blocks)
    else:
        wp.launch(kernel, dim=dim, inputs=inputs, stream=stream, max_blocks=max_blocks)


_DRAIN_FIRST_BLOCK = 4
"""Blind rounds before the first count readback when no drain has been seen yet.

Measured 2026-10-22 on 6 MV-class photon chunks in water: a drain needs 3-5 rounds
(primary photons, their electrons, then bremsstrahlung and delta-ray generations
of a few hundred and a few dozen particles). Overshooting by a round costs four
near-empty launches; undershooting costs a readback."""

_DRAIN_FOLLOW_UP_BLOCK = 2
"""Blind rounds per block after the first came back non-empty: the tail is short."""

_DRAIN_MAX_BLOCK = 32
"""Cap on the adaptive block, so one pathological drain cannot set a huge one."""

_DRAIN_THREADS_PER_SM = 2048
"""Threads per SM a blind launch's grid is floored at.

Deliberately at or above every current architecture's resident-thread ceiling
(2048 on most; Ada caps at 1536), so the floor never leaves an SM idle."""

_WARP_BLOCK_DIM = 256
"""Warp's default CUDA block size, which every engine launch uses (``wp.launch``
without ``block_dim``); the blind grid is counted in these blocks."""


def _blind_max_blocks(device: str, capacity: int, queue_factor: int) -> int:
    """CUDA grid of a blind drain launch, which spans the whole queue capacity.

    A blind round is launched at ``capacity`` so that no count readback is needed,
    but a queue rarely holds more than about a chunk. One thread per capacity slot
    would schedule ``queue_factor`` times more blocks than the work needs; instead
    the grid is one chunk wide (at least enough threads to fill every SM), and
    Warp's grid-stride loop walks the remaining slots, where the kernels' count
    guard returns at once. A queue holding more than a chunk is still drained in
    full — its threads take a second slot each. Scheduling only: which thread
    transports a particle never changes its history (``slots[tid]`` is scratch
    for the particle's own stream). 0 (Warp's default) on cpu.
    """
    d = wp.get_device(device)
    if not d.is_cuda:
        return 0
    chunk_blocks = -(-max(1, capacity // max(1, queue_factor)) // _WARP_BLOCK_DIM)
    sm_count = int(getattr(d, "sm_count", 0) or 0)
    floor_blocks = sm_count * _DRAIN_THREADS_PER_SM // _WARP_BLOCK_DIM
    return max(1, chunk_blocks, floor_blocks)


class _DrainState:
    """Mutable drain-schedule state an engine carries across runs.

    ``last_rounds`` is the number of rounds the latest blind drain on each device
    needed — the adaptive schedule's next first block. The drain's readback buffers
    live in the lane's :class:`_Workspace` (:meth:`_Workspace.drain_buffers`).
    """

    def __init__(self) -> None:
        self.last_rounds: dict[str, int] = {}


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

    The engine keeps its device state resident between calls, per device: the
    cross-section tables (per ``(ecut, pcut, e_max, with_gs)``, the four most
    recently used), the voxel maps and field, the last scoring geometry and recent
    spectra (by identity), and one transport workspace (queues, RNG slots,
    counters) per batch lane, refitted exactly to each call by ``run`` and
    ``run_dij`` alike. A loop of equal calls — a plan optimization — therefore pays
    the uploads and allocations once. This is bitwise inert (test-pinned against
    fresh engines). The engine-level keys are frozen fields, so treat ``grid``,
    ``cross_sections`` and ``magnetic_field`` as immutable (a mutated array is not
    re-uploaded — build a new engine); the per-call scoring grid and spectrum are
    matched by identity, and their arrays are read-only for that reason.

    **Device memory stays held between calls** while the engine lives: the tables
    and voxel maps, plus the queues the *last* call needed — about 0.9 GB after a
    full-width ``run`` on a 36-SM card at the auto chunk size, times its
    ``concurrent_batches`` lanes; a smaller call shrinks it again. Call
    :meth:`release_device_memory` to hand it all back. The dense dose maps are not
    kept: their size is the scoring grid's (or, for a Dij, budgeted from free
    memory), so they are allocated per call as before.

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
    magnetic_field
        Static magnetic field the charged particles are bent by; ``None`` is the
        field-free world; an explicit zero field reproduces it byte for byte on
        each device. World data like ``grid``: uploaded once per device next to
        the density and material maps, and a gridded map must cover the transport
        grid (checked at construction).
    drain_rounds
        Advanced; leave at the default unless profiling says otherwise. How the
        photon/electron queues are drained between chunks — bit-inert
        scheduling, like ``chunk_size`` (test-pinned). ``0`` is the host-sized
        schedule: every launch is sized from a count read back to the host, one
        synchronization per launch. ``k >= 1`` is the blind schedule: rounds are
        launched at queue capacity (the kernels guard against the device-side
        count) and the counts read back once per ``k`` rounds, issuing further
        blocks until both queues are empty — the end check, never ``k``, decides
        termination. Default ``None``: blind with ``k`` adapted to the round count
        the previous drain on the device needed on CUDA, where each readback idles
        the GPU for a host round trip; host-sized on cpu, where a readback is a
        plain memory read and a capacity-wide launch would loop over every slot.
    """

    grid: VoxelGrid
    cross_sections: CrossSectionSource
    device: str = "cpu"
    chunk_size: int | None = None
    queue_factor: int = 16
    magnetic_field: MagneticField | None = None
    drain_rounds: int | None = None
    _auto_chunk: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _drain: _DrainState = field(default_factory=_DrainState, init=False, repr=False, compare=False)
    # Host-table cache: build_tables flattens the source by looping the Python query
    # API over every grid node — seconds of fixed overhead for a tabulated source —
    # while its result is frozen after construction, so one build per
    # (ecut, pcut, e_max) serves every run and every device shard. Only the host
    # flatten is cached here; the device upload is cached per device in _resident.
    # The lock keeps concurrent Dij shards (one host thread per device) from
    # building the same key twice.
    _table_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _table_cache_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )
    # Device-resident state, one _DeviceState per device string (see the class
    # docstring); the lock guards only the dict, each state carries its own.
    _resident: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _resident_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )
    _default_scoring: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Refuse a negative drain block or a field map that misses the grid."""
        if self.drain_rounds is not None and self.drain_rounds < 0:
            raise ValueError(f"drain_rounds must be None or >= 0, got {self.drain_rounds}")
        if self.magnetic_field is not None and not self.magnetic_field.covers(self.grid):
            raise ValueError(
                "the magnetic field map does not cover the transport grid: "
                f"field extent {self.magnetic_field.origin} to "
                f"{self.magnetic_field.upper_corner} cm, grid {self.grid.origin} to "
                f"{self.grid.upper_corner} cm"
            )

    def _provenance(
        self,
        seed: int,
        pcut: float,
        ecut: float,
        msc_model: str,
        step_energy_fraction: float,
        deposit_resolution_cm: float | None,
        devices: Sequence[str] | None = None,
    ) -> RunProvenance:
        """Record the configuration this run actually used.

        ``devices`` is the Dij sharding set: a matrix assembled across several cards
        did not come from one device, and recording only ``self.device`` would claim
        a reproducibility that does not hold. The joined string says what ran.
        """
        return RunProvenance(
            version=__version__,
            backend="warp",
            device="+".join(devices) if devices else self.device,
            seed=seed,
            pcut_mev=pcut,
            ecut_mev=ecut,
            msc_model=msc_model,
            step_energy_fraction=step_energy_fraction,
            cross_sections=self.cross_sections.provenance,
            deposit_resolution_cm=deposit_resolution_cm,
            magnetic_field=(
                None if self.magnetic_field is None else self.magnetic_field.provenance
            ),
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

    # -- device-resident state ----------------------------------------------------

    def release_device_memory(self) -> None:
        """Drop every device array this engine keeps between calls, on every device.

        Use it to hand the card to something else between phases of work: an engine
        otherwise holds its tables, voxel maps and the last call's queues for as
        long as it lives. Results are unaffected — the next call re-uploads and
        reallocates what it needs, bit-identically. Host-side caches (the flattened cross-section
        tables, the Goudsmit-Saunderson grid) are kept: they cost seconds to
        rebuild and no device memory. Not for use while a call on this engine is
        running on another thread.
        """
        with self._resident_lock:
            self._resident.clear()

    def _state(self, device: str) -> _DeviceState:
        with self._resident_lock:
            state = self._resident.get(device)
            if state is None:
                state = _DeviceState()
                self._resident[device] = state
            return state

    def _scoring_geometry(self, scoring_grid: ScoringGeometry | None) -> ScoringGeometry:
        """Resolve a call's scoring geometry; the transport-grid default is built once."""
        if scoring_grid is not None:
            return scoring_grid
        default = self._default_scoring.get("grid")
        if default is None:
            default = ScoringGrid.for_grid(self.grid)
            self._default_scoring["grid"] = default
        return default

    def _upload_scoring(self, scoring: ScoringGeometry, device: str):
        """Return scoring metadata and voxel masses on ``device``, kept for the last one.

        Keyed by identity with the geometry itself held, so the key cannot be
        recycled by a new object at the same address; one entry, because a loop
        scores on one geometry and a dense mass map per entry is not free.
        """
        state = self._state(device)
        with state.lock:
            cached = state.scoring
            if cached is not None and cached[0] is scoring:
                return cached[1], cached[2]
            state.scoring = None  # release the previous map before uploading this one
            si = _scoring_info(scoring, device)
            voxel_mass = wp.array(
                scoring.voxel_mass.reshape(scoring.n_voxels), dtype=wp.float64, device=device
            )
            state.scoring = (scoring, si, voxel_mass)
            return si, voxel_mass

    def _spectral_tables(self, spectrum, device: str):
        """Return the spectrum's float32 inversion tables on ``device``, by identity."""
        state = self._state(device)
        with state.lock:
            cached = state.spectra.get(id(spectrum))
            if cached is not None and cached[0] is spectrum:
                return cached[1]
            tables = (
                wp.array(spectrum.edges.astype(np.float32), dtype=float, device=device),
                wp.array(spectrum.cdf.astype(np.float32), dtype=float, device=device),
            )
            if len(state.spectra) >= _SPECTRA_RESIDENT:
                state.spectra.pop(next(iter(state.spectra)))
            state.spectra[id(spectrum)] = (spectrum, tables)
            return tables

    def _acquire_workspaces(self, device: str, n: int, capacity: int) -> list[_Workspace]:
        """Take ``n`` transport workspaces fitted to exactly ``capacity`` slots per queue.

        Idle workspaces are reused and refitted (see :class:`_Workspace`), any idle
        surplus beyond ``n`` is released before anything is allocated, and fresh ones
        make up a shortfall. The pool therefore never holds more than the last
        successful call used, and an acquirer may count every idle byte as free
        (each is either released or reused for its own queues). The caller
        hands them back with :meth:`_release_workspaces` on success only: after an
        exception a lane stream may still be writing into them, so they are left
        to the garbage collector instead of being reused.
        """
        state = self._state(device)
        with state.lock:
            # Dropping the idle list releases the surplus here, before anything below
            # allocates.
            taken, state.idle = state.idle[:n], []
        for ws in taken:
            ws.fit(capacity)
        while len(taken) < n:
            ws = _Workspace(device)
            ws.fit(capacity)
            taken.append(ws)
        return taken

    def _release_workspaces(self, device: str, workspaces: list[_Workspace]) -> None:
        state = self._state(device)
        with state.lock:
            state.idle.extend(workspaces)

    def _idle_workspace_bytes(self, device: str) -> int:
        state = self._state(device)
        with state.lock:
            return sum(ws.nbytes for ws in state.idle)

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
        scoring_grid: ScoringGeometry | None = None,
        scoring_mode: str = "dose_to_medium",
        step_energy_fraction: float | None = None,
        deposit_resolution_cm: float | None = None,
        msc_model: str = "gs",
        progress: ProgressCallback | None = None,
        concurrent_batches: int = 1,
    ) -> TransportResult:
        """Transport ``n_histories`` primaries; same contract as the reference engine.

        See :meth:`pyradmc.backends.ref.engine.ReferenceEngine.run` for parameter
        semantics — the two signatures are deliberately identical (``scoring_grid``
        included: the scoring geometry deposits accumulate on — a rectilinear
        :class:`~pyradmc.scoring.grid.ScoringGrid`, default the transport grid, or a
        :class:`~pyradmc.scoring.cylinder.CylindricalScoringGrid` for a pencil-beam
        kernel — with off-geometry deposits booked to ``energy_unscored``; and ``scoring_mode``:
        dose-to-water weights each deposit in-kernel by the stopping-power ratio
        from the flattened tables while the books stay physical; ``progress``: one
        tick per completed batch, the identical cadence to the reference engine —
        see :mod:`pyradmc.progress`). A source with an in-kernel generator is
        generated on-device: the built-in mono beams from analytic parameters (all
        of one ``primary_kind``) and the exact
        :class:`~pyradmc.geometry.source.SpectralBeamSource` type from its uploaded
        spectrum tables (photons; ``primary_kind`` ignored). Any other source (a
        phase space, a spectral *subclass*, or a user
        :class:`~pyradmc.geometry.source.Source`) is transported by host-sampling
        each chunk via ``sample_batch`` and seeding the photon and electron queues
        by the per-record kind; ``primary_kind`` is then ignored and
        ``energy_emitted`` is booked from the sampled records.

        ``concurrent_batches`` overlaps whole statistical batches on independent
        CUDA streams, with private queues, RNG slots, and fixed-point dose maps.
        The float64 batch-dose fold remains serialized in batch order, so changing
        the lane count is bitwise inert on one device. CPU accepts the same option
        but runs sequentially. Each lane requires another queue set and dose map;
        memory use therefore grows approximately linearly with the lane count.

        ``msc_model`` selects the multiple-scattering law exactly as in the
        reference loop (see
        :func:`pyradmc.transport.electron.electron_steps`): ``"gs"`` — the
        shipped default — samples Goudsmit-Saunderson deflections from an
        eagerly precomputed table grid (built host-side on the first GS run
        for this ``(ecut, e_max)``, persisted to the user cache, uploaded once
        per device and kept resident) and does not apply the Gaussian-validity angular cap; the
        ``"gaussian"`` hinge survives as the paired-comparison test
        instrument. ``step_energy_fraction=None`` resolves to the selected
        model's validated fraction.
        """
        if n_histories < 1:
            raise ValueError(f"need at least one history, got {n_histories}")
        if concurrent_batches < 1:
            raise ValueError(f"need at least one lane, got concurrent_batches={concurrent_batches}")
        if n_histories % n_batches != 0:
            raise ValueError(
                f"n_histories={n_histories} not divisible by n_batches={n_batches}; "
                "unequal batches would weight batch means inconsistently"
            )
        from pyradmc.backends.warp.presolve import DevicePhaseSpace

        # Exact type, not isinstance: a subclass may override emit/sample_batch, and
        # the built-in generator would silently bypass the override; a subclass keeps
        # the host pre-sampling route (test-pinned, for every built-in generator).
        in_kernel = type(source) in (PencilBeamSource, ParallelBeamSource)
        is_device_ps = isinstance(source, DevicePhaseSpace)
        is_spectral = type(source) is SpectralBeamSource
        is_spectral_mask = (
            type(source) is TransmissionMaskSource and type(source.inner) is SpectralBeamSource
        )
        # A segmented plan: exact plan, segments and fan types (an override of any
        # of the three keeps the host route).
        is_segment_plan = (
            type(source) is SegmentedPlanSource
            and type(source.segments) is SegmentedFanSource
            and type(source.segments.fan) is SpectralBeamSource
        )
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
        if deposit_resolution_cm is not None and deposit_resolution_cm <= 0.0:
            raise ValueError(f"deposit resolution must be positive, got {deposit_resolution_cm}")
        # One float travels to the kernels; non-positive is the single-midpoint
        # sentinel, matching pyradmc.transport.electron.substep_pieces.
        deposit_resolution = 0.0 if deposit_resolution_cm is None else deposit_resolution_cm

        device = self.device
        gi, density, material = self._upload_grid(device)
        scoring = self._scoring_geometry(scoring_grid)
        si, voxel_mass = self._upload_scoring(scoring, device)
        table_energy = source.max_energy
        tab = self._upload_tables(table_energy, pcut, ecut, device, with_gs=msc_model_gs == 1)

        chunk = min(self._chunk_histories(device), n_histories)
        capacity = chunk * self.queue_factor
        lanes = min(concurrent_batches, n_batches) if wp.get_device(device).is_cuda else 1
        # Queues, RNG slots and counters come from the engine's resident pool (one
        # workspace per lane); the batch dose map is per call, sized by the scoring
        # grid. Handed back to the pool only on success (see _acquire_workspaces).
        workspaces = self._acquire_workspaces(device, lanes, capacity)
        lane_resources = [
            ws.lane(wp.zeros(scoring.n_voxels, dtype=wp.int64, device=device), lanes > 1)
            for ws in workspaces
        ]
        queues = lane_resources[0].queues
        slots = lane_resources[0].slots
        edep = lane_resources[0].edep
        counters = workspaces[0]
        fold_stream = counters.fold_stream() if lanes > 1 else None
        counters.clear_books()
        escaped = counters.book("escaped")
        unscored = counters.book("unscored")
        deposited = counters.book("deposited")
        violations = counters.book("violations")  # zeroed with the book

        # Advanced route: a source exposing a warp_sampler is generated in-kernel by a
        # wrapper kernel (built once per sampler), which books emitted weight-energy
        # into this cumulative counter (mono beams book analytically; a pre-sampled
        # source books per chunk).
        generator = None
        emitted = None
        wraps = (
            not in_kernel
            and not is_device_ps
            and not is_spectral
            and not is_spectral_mask
            and not is_segment_plan
        )
        if wraps and source.warp_sampler is not None:
            generator = kernels.make_generator_kernel(source.warp_sampler)
            emitted = counters.book("emitted")
        if is_device_ps:
            # The seeding kernel books emitted weight-energy into this counter,
            # like the wrapped-generator route, read once after the batches.
            emitted = counters.book("emitted")
        spectral_tables = None
        mask_table = None
        if is_spectral or is_spectral_mask:
            # Built-in in-kernel generation for the exact spectral type: the spectrum
            # inversion tables stay resident per spectrum and each chunk samples
            # on-device (the host pre-sampling was measured wall-dominant on CT-grade
            # runs). Emitted energy is booked per primary, like the wrapped-generator
            # route.
            spectral_source = source.inner if is_spectral_mask else source
            spectral_tables = self._spectral_tables(spectral_source.spectrum, device)
            emitted = counters.book("emitted")
            if is_spectral_mask:
                mask_table = wp.array(
                    source.mask.reshape(-1).astype(np.float32), dtype=float, device=device
                )
        segment_plan = None
        if is_segment_plan:
            # The segmented fan's device layout plus the plan's selection shares;
            # each primary picks its segment and weight in-kernel.
            from pyradmc.backends.warp import segments as segments_module

            segment_plan = (
                segments_module.upload_segmented_fan(source.segments, device),
                wp.array(source.selection_cdf.astype(np.float32), dtype=float, device=device),
            )
            emitted = counters.book("emitted")
            if lanes > 1:  # warp module loading is not thread-safe
                wp.load_module(segments_module, device=device)

        n_voxels = scoring.n_voxels
        # Dose finalize on the device: per-voxel batch sums of dose and dose^2, folded
        # in batch order so the open-field reduction matches a 1x1 Dij column bitwise
        # on one device (kernels.accumulate_run_batch). Only the reduced maps read
        # back, not the dense per-batch fixed-point buffer.
        s1 = wp.zeros(n_voxels, dtype=wp.float64, device=device)
        s2 = wp.zeros(n_voxels, dtype=wp.float64, device=device)
        dep_total = counters.book("dep_total")  # dose-to-medium quanta
        per_batch = n_histories // n_batches
        energy_escaped = 0.0
        energy_unscored = 0.0
        energy_deposited = 0.0  # dose-to-water physical book (summed per-batch counter)
        energy_emitted = 0.0
        escaped_quanta = 0
        unscored_quanta = 0
        deposited_quanta = 0
        emitter = ProgressEmitter(progress, n_histories)

        if lanes > 1:
            # Shared counters use integer atomics and therefore need only one clear
            # for the whole concurrent run (done above). Each lane owns the state
            # whose writes are not commutative (queues, RNG slots, and its batch dose
            # map).
            energy_emitted += self._run_forward_batch_lanes(
                lane_resources,
                fold_stream,
                n_batches,
                per_batch,
                chunk,
                source=source,
                kind=kind if in_kernel else None,
                in_kernel=in_kernel,
                is_device_ps=is_device_ps,
                is_spectral=is_spectral,
                is_spectral_mask=is_spectral_mask,
                segment_plan=segment_plan,
                generator=generator,
                spectral_tables=spectral_tables,
                mask_table=mask_table,
                seed=seed,
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
                deposit_resolution=deposit_resolution,
                msc_model_gs=msc_model_gs,
                transport_electrons=transport_electrons,
                dose_to_water=dose_to_water,
                emitted=emitted,
                device=device,
                voxel_mass=voxel_mass,
                n_voxels=n_voxels,
                s1=s1,
                s2=s2,
                dep_total=dep_total,
                emitter=emitter,
            )
            wp.synchronize_device(device)
            books = counters.read_books()
            _check_majorant(books)
            escaped_quanta = books["escaped"]
            unscored_quanta = books["unscored"]
            if dose_to_water != 0:
                deposited_quanta = books["deposited"]

        history = 0
        for _ in range(n_batches if lanes == 1 else 0):
            edep.zero_()
            counters.clear_pass_books()
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
                        deposit_resolution,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        device,
                        workspace=counters,
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
                        deposit_resolution,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        emitted,
                        device,
                        workspace=counters,
                    )
                elif segment_plan is not None:
                    self._transport_chunk_segment_plan(
                        source,
                        segment_plan,
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
                        deposit_resolution,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        emitted,
                        device,
                        workspace=counters,
                    )
                elif is_spectral_mask:
                    self._transport_chunk_spectral_mask(
                        source,
                        spectral_tables,
                        mask_table,
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
                        deposit_resolution,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        emitted,
                        device,
                        workspace=counters,
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
                        deposit_resolution,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        emitted,
                        device,
                        workspace=counters,
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
                        deposit_resolution,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        emitted,
                        device,
                        workspace=counters,
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
                        deposit_resolution,
                        msc_model_gs,
                        transport_electrons,
                        dose_to_water,
                        device,
                        workspace=counters,
                    )
                history += n_chunk
                remaining -= n_chunk
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
            # One readback per batch for all three per-batch books; it synchronizes
            # the default stream, which carries every launch of this path, so the
            # tick still marks a finished batch. The int64 sums stay per batch, as
            # before (a whole-run int64 sum could overflow at ~9e9 MeV).
            books = counters.read_books()
            _check_majorant(books)
            escaped_quanta += books["escaped"]
            unscored_quanta += books["unscored"]
            if dose_to_water != 0:
                deposited_quanta += books["deposited"]
            emitter.tick(per_batch)

        # Convert the exact integer books once, after aggregation. Besides avoiding
        # batch-count-dependent float rounding, this matches the Dij energy fold and
        # keeps concurrent lane scheduling bitwise inert.
        energy_escaped = float(escaped_quanta) * ENERGY_QUANTUM_MEV
        energy_unscored = float(unscored_quanta) * ENERGY_QUANTUM_MEV
        if dose_to_water != 0:
            energy_deposited = float(deposited_quanta) * ENERGY_QUANTUM_MEV

        mean_dev = wp.zeros(n_voxels, dtype=wp.float64, device=device)
        sigma_dev = wp.zeros(n_voxels, dtype=wp.float64, device=device)
        wp.launch(
            kernels.finalize_run,
            dim=n_voxels,
            inputs=[s1, s2, n_batches, mean_dev, sigma_dev],
            device=device,
        )
        books = counters.read_books()
        if in_kernel:
            # Mono beams: every primary is one unit-weight photon at the beam energy
            # (max_energy == energy), so the device need not report emitted energy.
            energy_emitted = n_histories * source.max_energy
        elif emitted is not None:
            energy_emitted = float(books["emitted"]) * ENERGY_QUANTUM_MEV

        if dose_to_water == 0:
            energy_deposited = float(books["dep_total"]) * ENERGY_QUANTUM_MEV
        dose = mean_dev.numpy().reshape(scoring.shape)
        dose_sigma = sigma_dev.numpy().reshape(scoring.shape)
        self._release_workspaces(device, workspaces)
        return TransportResult(
            dose=dose,
            dose_sigma=dose_sigma,
            energy_emitted=energy_emitted,
            energy_deposited=energy_deposited,
            energy_escaped=energy_escaped,
            energy_unscored=energy_unscored,
            n_histories=n_histories,
            n_batches=n_batches,
            scoring_mode=scoring_mode,
            provenance=self._provenance(
                seed, pcut, ecut, msc_model, step_energy_fraction, deposit_resolution_cm
            ),
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
        deposit_resolution_cm: float | None = None,
        msc_model: str = "gs",
        devices: Sequence[str] | None = None,
        concurrent_batches: int = 1,
        progress: ProgressCallback | None = None,
    ) -> DijResult:
        """Compute the Dij over the lattice; same contract as the reference engine.

        See :meth:`pyradmc.backends.ref.engine.ReferenceEngine.run_dij` for the
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

        scoring_grid
            Semantics as in :meth:`run`, and the memory lever here: the
            ``n_voxels`` above is the *scoring* voxel count, so a coarser dose
            grid shrinks the per-group device buffer and the sparse Dij cubically
            while transport keeps the full CT resolution.
        scoring_mode
            Weights the column tallies as in :meth:`run`; the energy books stay
            physical in either mode.
        msc_model
            As in :meth:`run`. The GS grid is built once, before any launch, under
            the same lock every device shard takes.
        progress
            Optional callback, as in :meth:`run` (see :mod:`pyradmc.progress`).
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
            count or spacing. One :class:`~pyradmc.progress.ProgressEmitter` is
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
        if deposit_resolution_cm is not None and deposit_resolution_cm <= 0.0:
            raise ValueError(f"deposit resolution must be positive, got {deposit_resolution_cm}")
        # One float travels to the kernels; non-positive is the single-midpoint
        # sentinel, matching pyradmc.transport.electron.substep_pieces.
        deposit_resolution = 0.0 if deposit_resolution_cm is None else deposit_resolution_cm

        scoring = self._scoring_geometry(scoring_grid)
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
                free = _device_free_bytes(d)
                if free > 0:
                    # The shard refits the engine's idle pooled queues to its own
                    # size (releasing the rest) before allocating, so they budget
                    # as free: the group size a fresh engine gets.
                    free += self._idle_workspace_bytes(d)
                return _auto_group_size(
                    free,
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
            provenance=self._provenance(
                seed,
                pcut,
                ecut,
                msc_model,
                step_energy_fraction,
                deposit_resolution_cm,
                devices=device_list,
            ),
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
            deposit_resolution=deposit_resolution,
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
            if type(source) is SegmentedFanSource:
                from pyradmc.backends.warp import segments

                for d in device_list:
                    wp.load_module(segments, device=d)
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

        use_lattice = type(source) is BeamletGridSource  # exact type: see run()
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
        deposit_resolution: float,
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
        si, voxel_mass = self._upload_scoring(scoring, device)
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
        # sampler/emit, matching the reference Dij). The segmented fan weights each
        # primary by its segment's stack transmission, evaluated in-kernel.
        use_lattice = type(source) is BeamletGridSource  # exact type: see run()
        use_spectral = type(source) is SpectralBeamletSource
        # The segmented fan: exact type *and* exact fan type (the TransmissionMask
        # precedent), so an override of either keeps the host route.
        use_segments = type(source) is SegmentedFanSource and type(source.fan) is SpectralBeamSource
        beamlet_generator = None
        spectral_tables = None
        segment_upload = None
        in_kernel_builtin = use_lattice or use_spectral or use_segments
        if not in_kernel_builtin and source.warp_beamlet_sampler is not None:
            beamlet_generator = kernels.make_beamlet_generator_kernel(source.warp_beamlet_sampler)
        if use_segments:
            # One stack layout + per-segment settings + the pre-solve's attenuation
            # table, uploaded once per shard: every group indexes it by segment.
            from pyradmc.backends.warp.segments import upload_segmented_fan

            segment_upload = upload_segmented_fan(source, device)
        if use_spectral:
            spectral_tables = self._spectral_tables(source.spectrum, device)
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
        # Queues, RNG slots and counters come from the engine's resident pool, one
        # workspace per lane, shared with run(). The dense group maps stay per call:
        # their size is budgeted from free memory at the call, and parking up to half
        # the card between calls would starve whatever runs next.
        workspaces = self._acquire_workspaces(device, lanes, max_capacity)
        lane_resources = [
            ws.lane(wp.zeros(max_group * n_voxels, dtype=wp.int64, device=device), lanes > 1)
            for ws in workspaces
        ]
        counters = workspaces[0]
        fold_stream = counters.fold_stream() if lanes > 1 else None
        counters.clear_books()
        escaped = counters.book("escaped")
        unscored = counters.book("unscored")
        deposited = counters.book("deposited")
        violations = counters.book("violations")  # zeroed with the book
        emitted_counter = None
        if beamlet_generator is not None or use_spectral or use_segments:
            emitted_counter = counters.book("emitted")
        s1 = wp.zeros(max_group * n_voxels, dtype=wp.float64, device=device)
        s2 = wp.zeros(max_group * n_voxels, dtype=wp.float64, device=device)
        total_q = counters.book("total_q")
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
            use_segments=use_segments,
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
            deposit_resolution=deposit_resolution,
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
            elif use_segments:
                group_arrays = segment_upload  # shard-wide; segments index it directly
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
            counters.clear_pass_books()  # escaped, unscored, deposited, total_q

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

            if lanes > 1:
                # Order the lanes' streams before the default-stream reduction. The
                # sequential path needs nothing here: every launch it made is on the
                # default stream, and the readbacks below synchronize that stream.
                wp.synchronize_device(device)
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
            # One readback for the group's books (per group, so no int64 sum spans
            # more than it did); it synchronizes the default stream.
            books = counters.read_books()
            _check_majorant(books)
            if dose_to_water != 0:
                deposited_quanta += books["deposited"]
            else:
                deposited_quanta += books["total_q"]
            escaped_quanta += books["escaped"]
            unscored_quanta += books["unscored"]

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
            shard.emitted_quanta = counters.read_books()["emitted"]
        self._release_workspaces(device, workspaces)
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
        use_segments,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
    ) -> float:
        """Generate and drain one batch of one group with a lane's resources.

        The single routing point for the five generation routes; sequential and
        lane execution differ only in which :class:`_LaneResources` they pass.
        ``group_arrays`` carries the active in-kernel route's upload: lattice
        bounds or spectral aperture centres per group, or the segmented fan's
        shard-wide :class:`~pyradmc.backends.warp.segments.SegmentedFanUpload`.
        Returns the batch's host-summed emitted energy (nonzero only on the
        pre-sampled route; the in-kernel routes book emitted energy on the
        device).
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
                deposit_resolution,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                device,
                res.stream,
                res.workspace,
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
                deposit_resolution,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                emitted_counter,
                device,
                res.stream,
                res.workspace,
            )
            return 0.0
        if use_segments:
            self._generate_group_segments(
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
                deposit_resolution,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                emitted_counter,
                device,
                res.stream,
                res.workspace,
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
                deposit_resolution,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                emitted_counter,
                device,
                res.stream,
                res.workspace,
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
            deposit_resolution,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
            res.stream,
            res.workspace,
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

    def _run_forward_batch_lanes(
        self,
        lane_resources,
        fold_stream,
        n_batches,
        per_batch,
        chunk,
        **route_args,
    ) -> float:
        """Transport forward-dose batches concurrently and fold them in batch order.

        This is the open-field twin of :meth:`_run_dij_group_lanes`. Fixed-point
        scoring and the energy books are associative integer atomics; only the
        float64 batch-statistics reduction is order-sensitive, so one fold stream
        consumes completed lane maps in ascending batch order. A lane's map is
        cleared only after its fold has completed and before that lane reuses it.
        """
        device = route_args["device"]
        n_voxels = route_args["n_voxels"]
        voxel_mass = route_args["voxel_mass"]
        s1 = route_args["s1"]
        s2 = route_args["s2"]
        dep_total = route_args["dep_total"]
        emitter = route_args["emitter"]
        lanes = len(lane_resources)
        wp.synchronize_device(device)
        state = {"next_fold": 0, "failed": False}
        cond = threading.Condition()
        emitted_by_batch = [0.0] * n_batches

        def lane(lane_index: int) -> None:
            res = lane_resources[lane_index]
            try:
                for batch in range(lane_index, n_batches, lanes):
                    emitted_by_batch[batch] = self._run_forward_batch(
                        res,
                        batch,
                        per_batch,
                        chunk,
                        **route_args,
                    )
                    # The drain loop's final count readback has synchronized this
                    # lane stream. Wait until every earlier batch has been folded.
                    with cond:
                        while state["next_fold"] != batch and not state["failed"]:
                            cond.wait()
                        if state["failed"]:
                            return
                        _launch(
                            kernels.accumulate_run_batch,
                            n_voxels,
                            [
                                res.edep,
                                voxel_mass,
                                float(per_batch),
                                float(ENERGY_QUANTUM_MEV),
                                s1,
                                s2,
                                dep_total,
                            ],
                            device,
                            fold_stream,
                        )
                        wp.synchronize_stream(fold_stream)
                        # The fold does not reset the forward map (unlike the Dij
                        # fold), so clear it on this lane before its next transport.
                        _launch(kernels.fill_int64, n_voxels, [res.edep, 0], device, res.stream)
                        emitter.tick(per_batch)
                        state["next_fold"] = batch + 1
                        cond.notify_all()
            except BaseException:
                with cond:
                    state["failed"] = True
                    cond.notify_all()
                raise

        with ThreadPoolExecutor(max_workers=lanes) as pool:
            futures = [pool.submit(lane, index) for index in range(lanes)]
            for future in futures:
                future.result()
        return float(sum(emitted_by_batch))

    def _run_forward_batch(
        self,
        res: _LaneResources,
        batch,
        per_batch,
        chunk,
        *,
        source,
        kind,
        in_kernel,
        is_device_ps,
        is_spectral,
        is_spectral_mask,
        segment_plan,
        generator,
        spectral_tables,
        mask_table,
        seed,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        **_fold_args,
    ) -> float:
        """Transport one complete forward statistical batch on one lane."""
        history = batch * per_batch
        remaining = per_batch
        emitted_energy = 0.0
        while remaining > 0:
            n_chunk = min(chunk, remaining)
            common = (
                seed,
                history,
                n_chunk,
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
                deposit_resolution,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
            )
            if in_kernel:
                self._transport_chunk(
                    source,
                    kind,
                    *common,
                    device,
                    res.stream,
                    res.workspace,
                )
            elif is_device_ps:
                self._transport_chunk_device_buffer(
                    source,
                    *common,
                    emitted,
                    device,
                    res.stream,
                    res.workspace,
                )
            elif segment_plan is not None:
                self._transport_chunk_segment_plan(
                    source,
                    segment_plan,
                    *common,
                    emitted,
                    device,
                    res.stream,
                    res.workspace,
                )
            elif is_spectral_mask:
                self._transport_chunk_spectral_mask(
                    source,
                    spectral_tables,
                    mask_table,
                    *common,
                    emitted,
                    device,
                    res.stream,
                    res.workspace,
                )
            elif is_spectral:
                self._transport_chunk_spectral(
                    source,
                    spectral_tables,
                    *common,
                    emitted,
                    device,
                    res.stream,
                    res.workspace,
                )
            elif generator is not None:
                self._transport_chunk_wrapped(
                    generator,
                    *common,
                    emitted,
                    device,
                    res.stream,
                    res.workspace,
                )
            else:
                emitted_energy += self._transport_chunk_presampled(
                    source,
                    *common,
                    device,
                    res.stream,
                    res.workspace,
                )
            history += n_chunk
            remaining -= n_chunk
        return emitted_energy

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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
        stream=None,
        workspace=None,
    ) -> None:
        """Generate one chunk of primaries and drain all queues."""
        q_particle, q_particle_alt, q_other, q_other_alt = queues
        for q in queues:
            _reset_count(q, device, stream)

        if kind == PHOTON:
            q_photon, q_photon_alt = q_particle, q_particle_alt
            q_electron, q_electron_alt = q_other, q_other_alt
            target = q_photon
        else:
            q_electron, q_electron_alt = q_particle, q_particle_alt
            q_photon, q_photon_alt = q_other, q_other_alt
            target = q_electron

        self._generate(source, kind, seed, history_offset, n_chunk, target, slots, device, stream)
        _set_count(target, n_chunk, device, stream)

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
            deposit_resolution,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
            stream,
            workspace,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        stream=None,
        workspace=None,
    ) -> None:
        """Generate one spectral-beam chunk in-kernel, then drain all queues.

        Photons only, at fixed thread slots into the photon queue (the mono-beam
        idiom: an explicit count write, no atomics); the kernel books each primary's
        energy into ``emitted``, since a polyenergetic ledger has no analytic total.
        """
        q_photon, q_photon_alt, q_electron, q_electron_alt = queues
        for q in queues:
            _reset_count(q, device, stream)

        sp_edges, sp_cdf = spectral_tables
        focal = source.focal_point
        center = source.center
        u_axis = source.u_axis
        v_axis = source.v_axis
        _launch(
            kernels.generate_spectral_beam,
            n_chunk,
            [
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
            device,
            stream,
        )
        _set_count(q_photon, n_chunk, device, stream)

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
            deposit_resolution,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
            stream,
            workspace,
        )

    def _transport_chunk_spectral_mask(
        self,
        source,
        spectral_tables,
        mask_table,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        stream=None,
        workspace=None,
    ) -> None:
        """Generate a spectral fan and transmission-mask it entirely on-device."""
        q_photon, q_photon_alt, q_electron, q_electron_alt = queues
        for q in queues:
            _reset_count(q, device, stream)

        inner = source.inner
        sp_edges, sp_cdf = spectral_tables
        focal = inner.focal_point
        center = inner.center
        source_u = inner.u_axis
        source_v = inner.v_axis
        mask_center = source.plane_center
        mask_u = source.u_axis
        mask_v = source.v_axis
        mask_n_u, mask_n_v = source.mask.shape
        _launch(
            kernels.generate_spectral_masked_beam,
            n_chunk,
            [
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
                source_u[0],
                source_u[1],
                source_u[2],
                source_v[0],
                source_v[1],
                source_v[2],
                inner.width_u,
                inner.width_v,
                mask_table,
                mask_n_u,
                mask_n_v,
                mask_center[0],
                mask_center[1],
                mask_center[2],
                mask_u[0],
                mask_u[1],
                mask_u[2],
                mask_v[0],
                mask_v[1],
                mask_v[2],
                source.width_u,
                source.width_v,
                slots,
                q_photon,
                emitted,
            ],
            device,
            stream,
        )

        # The generator atomically compacts exactly-zero mask weights, so its queue
        # count is already the number of primaries that need transport.
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
            deposit_resolution,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
            stream,
            workspace,
        )

    def _transport_chunk_segment_plan(
        self,
        source,
        segment_plan,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        stream=None,
        workspace=None,
    ) -> None:
        """Generate one segmented-plan chunk in-kernel, then drain all queues.

        Each primary chooses its segment by its normalized share and carries that
        segment's transmission
        (:func:`~pyradmc.backends.warp.segments.generate_segmented_plan`); fixed
        thread slots into the photon queue, emitted energy booked per primary.
        """
        from pyradmc.backends.warp.segments import generate_segmented_plan

        q_photon, q_photon_alt, q_electron, q_electron_alt = queues
        for q in queues:
            _reset_count(q, device, stream)
        upload, selection_cdf = segment_plan
        fan = source.segments.fan
        focal = fan.focal_point
        center = fan.center
        u_axis = fan.u_axis
        v_axis = fan.v_axis
        _launch(
            generate_segmented_plan,
            n_chunk,
            [
                seed,
                history_offset,
                selection_cdf,
                int(selection_cdf.shape[0]),
                upload.edges,
                upload.cdf,
                int(upload.cdf.shape[0]),
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
                fan.width_u,
                fan.width_v,
                upload.stack,
                upload.segments,
                upload.mu,
                slots,
                emitted,
                q_photon,
            ],
            device,
            stream,
        )
        _set_count(q_photon, n_chunk, device, stream)
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
            deposit_resolution,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
            stream,
            workspace,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        stream=None,
        workspace=None,
    ) -> None:
        """Generate one chunk in-kernel via a wrapped user sampler, then drain.

        The wrapped generator (:func:`~pyradmc.backends.warp.kernels.make_generator_kernel`)
        pushes each primary into the photon or electron queue by its kind at an atomic
        slot and books emitted weight-energy into ``emitted``; the queue counts are then
        whatever the pushes set, so — unlike the mono-beam path — no explicit count is
        written.
        """
        q_photon, q_photon_alt, q_electron, q_electron_alt = queues
        for q in queues:
            _reset_count(q, device, stream)

        _launch(
            generator,
            n_chunk,
            [seed, history_offset, q_photon, q_electron, slots, emitted],
            device,
            stream,
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
            deposit_resolution,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
            stream,
            workspace,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
        stream=None,
        workspace=None,
    ) -> float:
        """Host-sample a chunk via ``source.sample_batch``, seed queues by kind, drain.

        The general (pre-sampling) route: works for any source with a ``sample_batch``
        — a phase space, or a user :class:`~pyradmc.geometry.source.Source` using the
        emit-based default. Returns the chunk's emitted energy: sum of
        ``weight * energy`` over the sampled records, plus ``weight * 2 m_e c^2`` for
        each positron whose annihilation photons the device books into
        deposited/escaped (so the emitted = deposited + escaped ledger closes, as on
        the reference backend).
        """
        q_photon, q_photon_alt, q_electron, q_electron_alt = queues
        for q in queues:
            _reset_count(q, device, stream)

        batch = source.sample_batch(seed, history_offset, n_chunk)
        pt = batch["particle_type"]
        hist = (history_offset + np.arange(n_chunk, dtype=np.int64)).astype(np.int32)
        transport_kind = _IAEA_TO_TRANSPORT[pt]

        # An exactly-zero statistical weight contributes nothing to dose or any
        # energy book. Do not spend a full transport history on it. This matters
        # especially for deterministic transmission masks, whose closed pixels
        # deliberately emit weight-zero records to preserve the source RNG stream.
        # The surviving records retain their global history keys, so compaction
        # changes neither their random streams nor same-device reproducibility.
        active = batch["weight"] != 0.0
        photon = active & (pt == 1)  # electrons (2) and positrons (3) share the e- queue
        charged = active & (pt != 1)
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
            stream,
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
            stream,
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
            deposit_resolution,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
            stream,
            workspace,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        stream=None,
        workspace=None,
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
            _reset_count(q, device, stream)
        buf = source.buffer
        _launch(
            kernels.generate_from_exit_buffer,
            n_chunk,
            [
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
            device,
            stream,
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
            deposit_resolution,
            msc_model_gs,
            transport_electrons,
            dose_to_water,
            device,
            stream,
            workspace,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
        stream=None,
        workspace=None,
    ) -> None:
        """Ping-pong the photon and electron kernels until every queue is empty.

        One *round* is a photon launch then an electron launch; each consumes its
        queue, pushing photons and electrons into the others. The schedule
        (``drain_rounds``, see the class docstring) decides only how often the
        counts come back to the host: a run is a function of the multiset of
        particles each launch consumes, and an empty launch consumes nothing, so
        every schedule gives the same bits.

        With ``stream`` set (a Dij batch lane), every launch, count reset, and
        count readback is ordered on that stream, so lanes on the same device
        never observe each other's queues.
        """

        def photon_round(dim, q_in, q_electron_out, q_photon_out, max_blocks=0):
            _launch(
                kernels.photon_kernel,
                dim,
                [
                    gi,
                    si,
                    density,
                    material,
                    tab,
                    q_in,
                    q_electron_out,
                    q_photon_out,
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
                max_blocks,
            )

        def electron_round(dim, q_in, q_photon_out, q_electron_out, max_blocks=0):
            _launch(
                kernels.electron_kernel,
                dim,
                [
                    gi,
                    si,
                    density,
                    material,
                    tab,
                    q_in,
                    q_photon_out,
                    q_electron_out,
                    slots,
                    edep,
                    escaped,
                    unscored,
                    deposited,
                    pcut,
                    ecut,
                    step_energy_fraction,
                    deposit_resolution,
                    msc_model_gs,
                    dose_to_water,
                ],
                device,
                stream,
                max_blocks,
            )

        if workspace is None:
            # Every engine route passes its lane's pooled workspace; a bare call
            # gets a throwaway one, whose buffers are made on demand below.
            workspace = _Workspace(device)
        rounds = self.drain_rounds
        if rounds is None:
            rounds = (
                self._drain.last_rounds.get(device, _DRAIN_FIRST_BLOCK)
                if wp.get_device(device).is_cuda
                else 0
            )
        if rounds == 0:
            # Host-sized: read each count back and launch exactly that many threads.
            staging = workspace.staging
            while True:
                n_photon = _queue_count(q_photon, stream, staging)
                if n_photon > 0:
                    photon_round(n_photon, q_photon, q_electron, q_photon_alt)
                    _reset_count(q_photon, device, stream)
                    q_photon, q_photon_alt = q_photon_alt, q_photon

                n_electron = _queue_count(q_electron, stream, staging)
                if n_electron > 0:
                    electron_round(n_electron, q_electron, q_photon, q_electron_alt)
                    _reset_count(q_electron, device, stream)
                    q_electron, q_electron_alt = q_electron_alt, q_electron

                if n_photon == 0 and n_electron == 0:
                    break
            return

        # Blind: launch whole blocks of rounds at capacity, read back once per block.
        # Every queue a launch consumes is retired by a one-thread kernel on the same
        # stream, which books overflow and the last busy round on the device; the
        # alternate queue a launch pushes into is always a retired (empty) one.
        status, host = workspace.drain_buffers(stream)
        _launch(kernels.fill_int32, kernels.DRAIN_STATUS_SLOTS, [status, 0], device, stream)
        capacity = q_photon.capacity
        # One launch dimension serves every queue: they must share one capacity.
        assert (
            q_photon_alt.capacity == q_electron.capacity == q_electron_alt.capacity == capacity
        ), "drain queues differ in capacity"
        max_blocks = _blind_max_blocks(device, capacity, self.queue_factor)
        issued = 0
        block = rounds
        while True:
            for _ in range(block):
                photon_round(capacity, q_photon, q_electron, q_photon_alt, max_blocks)
                _launch(
                    kernels.retire_queue,
                    1,
                    [q_photon.count, capacity, issued, status],
                    device,
                    stream,
                )
                q_photon, q_photon_alt = q_photon_alt, q_photon
                electron_round(capacity, q_electron, q_photon, q_electron_alt, max_blocks)
                _launch(
                    kernels.retire_queue,
                    1,
                    [q_electron.count, capacity, issued, status],
                    device,
                    stream,
                )
                q_electron, q_electron_alt = q_electron_alt, q_electron
                issued += 1
            _launch(
                kernels.snapshot_drain_status,
                1,
                [q_photon.count, q_electron.count, status],
                device,
                stream,
            )
            if stream is None:
                books = status.numpy()
            else:
                wp.copy(host, status, stream=stream)
                wp.synchronize_stream(stream)
                books = host.numpy()
            n_photon = int(books[kernels.DRAIN_PHOTON_COUNT])
            n_electron = int(books[kernels.DRAIN_ELECTRON_COUNT])
            overflow = max(int(books[kernels.DRAIN_OVERFLOW]), n_photon, n_electron)
            if overflow > capacity:
                raise RuntimeError(
                    f"particle queue overflow: {overflow} > capacity {capacity}. "
                    "Lower chunk_size or raise queue_factor on the WarpEngine."
                )
            if n_photon == 0 and n_electron == 0:
                break
            block = rounds if self.drain_rounds is not None else _DRAIN_FOLLOW_UP_BLOCK
        self._drain.last_rounds[device] = min(
            max(1, int(books[kernels.DRAIN_LAST_BUSY_ROUND])), _DRAIN_MAX_BLOCK
        )

    def _generate(
        self, source, kind, seed, history_offset, n, queue, slots, device, stream=None
    ) -> None:
        if isinstance(source, PencilBeamSource):
            _launch(
                kernels.generate_pencil_beam,
                n,
                [
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
                device,
                stream,
            )
        elif isinstance(source, ParallelBeamSource):
            _launch(
                kernels.generate_parallel_beam,
                n,
                [
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
                device,
                stream,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
        stream=None,
        workspace=None,
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
                deposit_resolution,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                device,
                stream,
                workspace,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        stream=None,
        workspace=None,
    ) -> None:
        """Built-in spectral Dij route: generate one batch of the group in-kernel.

        Same per-batch block chunking as the lattice, with the spectrum inversion
        and divergent-fan geometry inline
        (:func:`~pyradmc.backends.warp.kernels.generate_spectral_beamlets`); the
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
                deposit_resolution,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                device,
                stream,
                workspace,
            )
            t += n_chunk

    def _generate_group_segments(
        self,
        source,
        upload,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        stream=None,
        workspace=None,
    ) -> None:
        """Built-in segmented-fan Dij route: generate one batch of the group in-kernel.

        The spectral route's block chunking, with every primary weighted by its
        segment's stack transmission on the device
        (:func:`~pyradmc.backends.warp.segments.generate_segmented_fan`); the kernel
        books ``weight * energy`` into ``emitted`` per primary.
        """
        from pyradmc.backends.warp.segments import generate_segmented_fan

        fan = source.fan
        focal = fan.focal_point
        center = fan.center
        u_axis = fan.u_axis
        v_axis = fan.v_axis
        t = 0
        while t < block_histories:
            n_chunk = min(chunk, block_histories - t)
            for q in queues:
                _reset_count(q, device, stream)
            _launch(
                generate_segmented_fan,
                n_chunk,
                [
                    seed,
                    group_start,
                    n_per,
                    per_batch,
                    batch,
                    t,
                    1 if correlated else 0,
                    upload.edges,
                    upload.cdf,
                    int(upload.cdf.shape[0]),
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
                    fan.width_u,
                    fan.width_v,
                    upload.stack,
                    upload.segments,
                    upload.mu,
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
                deposit_resolution,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                device,
                stream,
                workspace,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        emitted,
        device,
        stream=None,
        workspace=None,
    ) -> None:
        """Advanced Dij route: generate one batch of the group in-kernel via a sampler.

        Same per-batch block chunking as the built-in lattice, but the per-primary
        position comes from the user's ``warp_beamlet_sampler`` (wrapped by
        :func:`~pyradmc.backends.warp.kernels.make_beamlet_generator_kernel`) instead of
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
                deposit_resolution,
                msc_model_gs,
                transport_electrons,
                dose_to_water,
                device,
                stream,
                workspace,
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
        deposit_resolution,
        msc_model_gs,
        transport_electrons,
        dose_to_water,
        device,
        stream=None,
        workspace=None,
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
                    deposit_resolution,
                    msc_model_gs,
                    transport_electrons,
                    dose_to_water,
                    device,
                    stream,
                    workspace,
                )
                r0 = r1
        return emitted

    def _upload_grid(self, device: str):
        """Return the transport grid (and field) on ``device``, uploaded once per device."""
        state = self._state(device)
        with state.lock:
            if state.grid is None:
                state.grid = self._build_grid(device)
            return state.grid

    def _build_grid(self, device: str):
        gi = GridInfo()
        gi.x_lo, gi.y_lo, gi.z_lo = self.grid.origin
        gi.x_hi, gi.y_hi, gi.z_hi = self.grid.upper_corner
        gi.sx, gi.sy, gi.sz = self.grid.spacing
        gi.nx = self.grid.shape[0]
        gi.ny = self.grid.shape[1]
        gi.nz = self.grid.shape[2]
        gi.n_voxels = int(np.prod(self.grid.shape))
        gi.min_spacing = min(self.grid.spacing)
        # The transport grid is always rectilinear (AGENTS.md 6); these fields exist
        # only because scoring shares the struct, and no kernel reads them at mode 0.
        gi.mode = 0
        gi.axis_x = gi.axis_y = 0.0
        gi.n_shells = 0
        gi.r_inner_sq = gi.r_outer_sq = 0.0
        gi.r_edges_sq = _no_shells(device)
        gi.z_edges = _no_shells(device)
        # The magnetic field rides in the same struct so that no kernel signature
        # changes: b_mode routes the electron kernel's per-substep lookup (uniform:
        # the three scalars; nearest/trilinear: the vec3 map with its own geometry).
        # The field-free world leaves b_mode at FIELD_NONE and the kernel executes
        # no field operation at all — that is what keeps it byte-identical.
        b = self.magnetic_field
        if b is None:
            _fill_no_field(gi, device)
        else:
            gi.b_mode = b.mode
            if b.is_uniform:
                gi.b0x, gi.b0y, gi.b0z = (float(v) for v in b.values[0, 0, 0])
            else:
                gi.b0x = gi.b0y = gi.b0z = 0.0
            gi.b_ox, gi.b_oy, gi.b_oz = (float(v) for v in b.origin)
            gi.b_sx, gi.b_sy, gi.b_sz = (float(v) for v in b.spacing)
            gi.b_nx, gi.b_ny, gi.b_nz = (int(n) for n in b.shape)
            gi.b_values = wp.array(b.values.astype(np.float32), dtype=wp.vec3, device=device)
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
        """Return the flattened tables on ``device``, uploaded once per key and device.

        Mirrors the host cache's ``(ecut, pcut, e_max)`` key, plus ``with_gs``
        because the struct differs; the Goudsmit-Saunderson grid — the one large
        table, independent of ``pcut`` — is uploaded once per ``(ecut, e_max)`` and
        shared by every struct that carries it. The struct is read-only to the
        kernels, so concurrent calls on one device may share it.
        """
        state = self._state(device)
        full_key = (ecut, pcut, e_max, with_gs)
        with state.lock:
            tab = _lru_get(state.tables, full_key)
        if tab is not None:
            return tab
        key = (ecut, pcut, e_max)
        with self._table_cache_lock:
            host = self._table_cache.get(key)
            if host is None:
                host = self.cross_sections.build_tables(
                    ecut=ecut, pcut=pcut, e_max=e_max * _E_MAX_MARGIN
                )
                self._table_cache[key] = host
        gs_host = self._gs_grid_host(host, ecut, e_max) if with_gs else None
        with state.lock:
            tab = _lru_get(state.tables, full_key)
            if tab is None:
                tab = self._build_tables(state, host, gs_host, ecut, pcut, e_max, device)
                _lru_put(state.tables, full_key, tab, _TABLES_RESIDENT)
            return tab

    def _build_tables(
        self, state: _DeviceState, host, gs_host, ecut, pcut, e_max, device
    ) -> Tables:
        """Assemble a ``Tables`` struct from device arrays shared across structs."""
        base_key = ("base", ecut, pcut, e_max)
        base = _lru_get(state.table_arrays, base_key)
        if base is None:
            base = {
                name: wp.array(getattr(host, name).astype(np.float32), dtype=float, device=device)
                for name in (
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
                    "coherent_x",
                    "coherent_cumulative",
                )
            }
            _lru_put(state.table_arrays, base_key, base, 2 * _TABLES_RESIDENT)
        tab = Tables()
        for name, arr in base.items():
            setattr(tab, name, arr)
        tab.p_log_e_min = host.photon_log_e_min
        tab.p_inv_dlog = host.photon_inv_dlog
        tab.e_log_e_min = host.electron_log_e_min
        tab.e_inv_dlog = host.electron_inv_dlog
        tab.r_log_min = host.range_log_r_min
        tab.r_inv_dlog = host.range_inv_dlog
        tab.n_points = host.n_points
        tab.n_coherent = host.n_coherent
        if gs_host is not None:
            gs_key = ("gs", ecut, e_max)
            gs_values = _lru_get(state.table_arrays, gs_key)
            if gs_values is None:
                gs_values = wp.array(gs_host.values.astype(np.float32), dtype=float, device=device)
                _lru_put(state.table_arrays, gs_key, gs_values, 2 * _TABLES_RESIDENT)
            tab.gs_values = gs_values
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
        from pyradmc.data.goudsmit_saunderson import build_gs_grid, gs_window_from_tables

        key = ("gs", ecut, e_max)
        with self._table_cache_lock:
            gs_host = self._table_cache.get(key)
            if gs_host is None:
                present = np.unique(self.grid.material).astype(int).tolist()
                gs_host = build_gs_grid(*gs_window_from_tables(host, present))
                self._table_cache[key] = gs_host
        return gs_host
