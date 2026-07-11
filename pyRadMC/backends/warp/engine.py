"""The Warp engine: launch and memory management for the transport kernels.

Production backend, ``cpu`` and ``cuda`` from one kernel source (AGENTS.md section 3:
nothing physical lives here). The run loop is:

1. Flatten the cross-sections (``build_tables``) and upload everything as float32.
2. Per batch, per chunk of histories: generate primaries into a queue, then
   ping-pong the photon and electron kernels until every queue drains. Photons
   spawn charged secondaries into the electron queue; electrons spawn
   bremsstrahlung and annihilation photons back, and delta rays into the next
   electron round.
3. Read back the int64 fixed-point energy map per batch and feed the shared
   ``BatchedDoseScorer``, so sigma comes from batch statistics exactly as in the
   reference backend (AGENTS.md 2.4).

Reproducibility (AGENTS.md 2.3): streams are pure functions of
``(seed, history_index)`` with child streams derived from parent draws, and scoring
is associative integer arithmetic — so one device, one seed is bit-reproducible
regardless of scheduling, chunking, or queue ordering, and *nothing* is reproducible
across targets. Energy conservation holds to float32 transport arithmetic plus the
scoring quantum; the equivalence tests assert it at 1e-4 relative.

This module is Warp backend code: exempt from ``mypy --strict`` (AGENTS.md 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from pyRadMC import ECUT_MEV, PCUT_MEV
from pyRadMC.backends.results import TransportResult
from pyRadMC.backends.warp import kernels
from pyRadMC.backends.warp.kernels import ENERGY_QUANTUM_MEV, GridInfo, Queue, Tables
from pyRadMC.data.interface import CrossSectionSource
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource, PencilBeamSource
from pyRadMC.scoring.dose import BatchedDoseScorer
from pyRadMC.transport.particles import ELECTRON, PHOTON

__all__ = ["WarpEngine"]

_E_MAX_MARGIN = 1.0 + 1.0e-6  # table upper edge strictly above the primary energy


def _upload_queue(capacity: int, device: str) -> Queue:
    q = Queue()
    q.kind = wp.zeros(capacity, dtype=wp.int32, device=device)
    q.energy = wp.zeros(capacity, dtype=float, device=device)
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


def _queue_count(q: Queue) -> int:
    count = int(q.count.numpy()[0])
    if count > q.capacity:
        raise RuntimeError(
            f"particle queue overflow: {count} > capacity {q.capacity}. "
            "Lower chunk_size or raise queue_factor on the WarpEngine."
        )
    return count


def _reset_count(q: Queue, device: str) -> None:
    q.count.zero_()


def _set_count(q: Queue, value: int, device: str) -> None:
    wp.copy(q.count, wp.array(np.array([value], dtype=np.int32), device=device))


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
        (test-pinned); it only trades memory against launch count.
    queue_factor
        Queue capacity per chunk history. Overflow raises rather than dropping
        secondaries.
    """

    grid: VoxelGrid
    cross_sections: CrossSectionSource
    device: str = "cpu"
    chunk_size: int = 32_768
    queue_factor: int = 16

    def run(
        self,
        source: PencilBeamSource | ParallelBeamSource,
        n_histories: int,
        n_batches: int,
        seed: int,
        pcut: float = PCUT_MEV,
        ecut: float = ECUT_MEV,
        transport_electrons: bool = True,
        primary_kind: str = "photon",
    ) -> TransportResult:
        """Transport ``n_histories`` primaries; same contract as the reference engine.

        See :meth:`pyRadMC.backends.ref.engine.ReferenceEngine.run` for parameter
        semantics — the two signatures are deliberately identical.
        """
        if n_histories < 1:
            raise ValueError(f"need at least one history, got {n_histories}")
        if n_histories % n_batches != 0:
            raise ValueError(
                f"n_histories={n_histories} not divisible by n_batches={n_batches}; "
                "unequal batches would weight batch means inconsistently"
            )
        if primary_kind not in ("photon", "electron"):
            raise ValueError(f"unknown primary_kind {primary_kind!r}")
        kind = PHOTON if primary_kind == "photon" else ELECTRON

        device = self.device
        gi, density, material = self._upload_grid(device)
        tab = self._upload_tables(source.energy, pcut, ecut, device)

        chunk = min(self.chunk_size, n_histories)
        capacity = chunk * self.queue_factor
        queues = [_upload_queue(capacity, device) for _ in range(4)]
        slots = wp.zeros(capacity, dtype=wp.uint32, device=device)
        n_voxels = int(np.prod(self.grid.shape))
        edep = wp.zeros(n_voxels, dtype=wp.int64, device=device)
        escaped = wp.zeros(1, dtype=wp.int64, device=device)
        violations = wp.zeros(1, dtype=wp.int32, device=device)

        scorer = BatchedDoseScorer(self.grid, n_batches)
        per_batch = n_histories // n_batches
        energy_escaped = 0.0

        history = 0
        for _ in range(n_batches):
            edep.zero_()
            escaped.zero_()
            remaining = per_batch
            while remaining > 0:
                n_chunk = min(chunk, remaining)
                self._transport_chunk(
                    source,
                    kind,
                    seed,
                    history,
                    n_chunk,
                    gi,
                    density,
                    material,
                    tab,
                    queues,
                    slots,
                    edep,
                    escaped,
                    violations,
                    pcut,
                    ecut,
                    transport_electrons,
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
            batch_energy = edep.numpy().astype(np.float64) * ENERGY_QUANTUM_MEV
            scorer.deposit_grid(batch_energy.reshape(self.grid.shape))
            scorer.end_batch(per_batch)
            energy_escaped += float(escaped.numpy()[0]) * ENERGY_QUANTUM_MEV

        dose = scorer.finalize()
        return TransportResult(
            dose=dose.dose,
            dose_sigma=dose.dose_sigma,
            energy_emitted=n_histories * source.energy,
            energy_deposited=dose.energy_deposited,
            energy_escaped=energy_escaped,
            n_histories=n_histories,
            n_batches=n_batches,
        )

    # -- internals --------------------------------------------------------------

    def _transport_chunk(
        self,
        source,
        kind,
        seed,
        history_offset,
        n_chunk,
        gi,
        density,
        material,
        tab,
        queues,
        slots,
        edep,
        escaped,
        violations,
        pcut,
        ecut,
        transport_electrons,
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

        while True:
            n_photon = _queue_count(q_photon)
            if n_photon > 0:
                wp.launch(
                    kernels.photon_kernel,
                    dim=n_photon,
                    inputs=[
                        gi,
                        density,
                        material,
                        tab,
                        q_photon,
                        q_electron,
                        q_photon_alt,
                        slots,
                        edep,
                        escaped,
                        pcut,
                        ecut,
                        1 if transport_electrons else 0,
                        violations,
                    ],
                    device=device,
                )
                _reset_count(q_photon, device)
                q_photon, q_photon_alt = q_photon_alt, q_photon

            n_electron = _queue_count(q_electron)
            if n_electron > 0:
                wp.launch(
                    kernels.electron_kernel,
                    dim=n_electron,
                    inputs=[
                        gi,
                        density,
                        material,
                        tab,
                        q_electron,
                        q_photon,
                        q_electron_alt,
                        slots,
                        edep,
                        escaped,
                        pcut,
                        ecut,
                    ],
                    device=device,
                )
                _reset_count(q_electron, device)
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

    def _upload_grid(self, device: str):
        gi = GridInfo()
        gi.x_lo, gi.y_lo, gi.z_lo = self.grid.origin
        gi.x_hi, gi.y_hi, gi.z_hi = self.grid.upper_corner
        gi.sx, gi.sy, gi.sz = self.grid.spacing
        gi.ny = self.grid.shape[1]
        gi.nz = self.grid.shape[2]
        gi.min_spacing = min(self.grid.spacing)
        density = wp.array(self.grid.density.astype(np.float32), dtype=float, device=device)
        material = wp.array(self.grid.material.astype(np.int32), dtype=wp.int32, device=device)
        return gi, density, material

    def _upload_tables(self, e_max: float, pcut: float, ecut: float, device: str) -> Tables:
        host = self.cross_sections.build_tables(ecut=ecut, pcut=pcut, e_max=e_max * _E_MAX_MARGIN)
        tab = Tables()
        for field in (
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
        ):
            setattr(
                tab,
                field,
                wp.array(getattr(host, field).astype(np.float32), dtype=float, device=device),
            )
        tab.p_log_e_min = host.photon_log_e_min
        tab.p_inv_dlog = host.photon_inv_dlog
        tab.e_log_e_min = host.electron_log_e_min
        tab.e_inv_dlog = host.electron_inv_dlog
        tab.n_points = host.n_points
        return tab
