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

from pyRadMC import DIJ_TRUNCATION_RELATIVE, ECUT_MEV, ELECTRON_MASS_MEV, PCUT_MEV
from pyRadMC.backends.results import TransportResult
from pyRadMC.backends.warp import kernels
from pyRadMC.backends.warp.kernels import ENERGY_QUANTUM_MEV, GridInfo, Queue, Tables
from pyRadMC.data.interface import CrossSectionSource
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.phasespace import PhaseSpaceSource
from pyRadMC.geometry.source import BeamletGridSource, ParallelBeamSource, PencilBeamSource
from pyRadMC.scoring.dij import BatchedBeamletScorer, DijAssembler, DijResult
from pyRadMC.scoring.dose import BatchedDoseScorer
from pyRadMC.transport.particles import ELECTRON, PHOTON, POSITRON

__all__ = ["WarpEngine"]

_E_MAX_MARGIN = 1.0 + 1.0e-6  # table upper edge strictly above the primary energy

# IAEA particle code (from PhaseSpaceSource.sample_batch) -> transport constant,
# as a lookup indexed by the code (1/2/3); slot 0 is unused.
_IAEA_TO_TRANSPORT = np.array([-1, PHOTON, ELECTRON, POSITRON], dtype=np.int32)


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
        source: PencilBeamSource | ParallelBeamSource | PhaseSpaceSource,
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
        semantics — the two signatures are deliberately identical. A
        :class:`~pyRadMC.geometry.phasespace.PhaseSpaceSource` is transported by
        host-sampling each chunk of primaries and seeding the photon and electron
        queues by kind; ``primary_kind`` is then ignored (each record carries its
        own kind), and ``energy_emitted`` is booked from the sampled records.
        """
        if n_histories < 1:
            raise ValueError(f"need at least one history, got {n_histories}")
        if n_histories % n_batches != 0:
            raise ValueError(
                f"n_histories={n_histories} not divisible by n_batches={n_batches}; "
                "unequal batches would weight batch means inconsistently"
            )
        is_phsp = isinstance(source, PhaseSpaceSource)
        if not is_phsp:
            if primary_kind not in ("photon", "electron"):
                raise ValueError(f"unknown primary_kind {primary_kind!r}")
            kind = PHOTON if primary_kind == "photon" else ELECTRON

        device = self.device
        gi, density, material = self._upload_grid(device)
        table_energy = source.max_energy if is_phsp else source.energy
        tab = self._upload_tables(table_energy, pcut, ecut, device)

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
        energy_emitted = 0.0

        history = 0
        for _ in range(n_batches):
            edep.zero_()
            escaped.zero_()
            remaining = per_batch
            while remaining > 0:
                n_chunk = min(chunk, remaining)
                if is_phsp:
                    energy_emitted += self._transport_chunk_phsp(
                        source,
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
                else:
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

        if not is_phsp:
            energy_emitted = n_histories * source.energy

        dose = scorer.finalize()
        return TransportResult(
            dose=dose.dose,
            dose_sigma=dose.dose_sigma,
            energy_emitted=energy_emitted,
            energy_deposited=dose.energy_deposited,
            energy_escaped=energy_escaped,
            n_histories=n_histories,
            n_batches=n_batches,
        )

    def run_dij(
        self,
        source: BeamletGridSource,
        n_histories_per_beamlet: int,
        n_batches: int,
        seed: int,
        pcut: float = PCUT_MEV,
        ecut: float = ECUT_MEV,
        transport_electrons: bool = True,
        truncation: float = DIJ_TRUNCATION_RELATIVE,
        correlated: bool = True,
        beamlet_group_size: int = 32,
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
            Beamlets scored concurrently into one dense device buffer of
            ``group * n_batches * n_voxels`` int64 quanta (the batch axis lives
            on the device too, so a group needs only one queue-drain sequence
            and one readback). Purely a memory/occupancy trade-off: streams are
            pure functions of ``(seed, history)`` and scoring is associative, so
            the result is bit-identical for any value (test-pinned), exactly
            like ``chunk_size``.
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
        if beamlet_group_size < 1:
            raise ValueError(f"need a positive beamlet group size, got {beamlet_group_size}")

        device = self.device
        gi, density, material = self._upload_grid(device)
        tab = self._upload_tables(source.energy, pcut, ecut, device)
        n_beamlets = source.n_beamlets
        per_batch = n_histories_per_beamlet // n_batches
        n_voxels = int(np.prod(self.grid.shape))

        escaped = wp.zeros(1, dtype=wp.int64, device=device)
        violations = wp.zeros(1, dtype=wp.int32, device=device)
        assembler = DijAssembler(
            grid_shape=self.grid.shape,
            n_beamlets=n_beamlets,
            n_histories_per_beamlet=n_histories_per_beamlet,
            n_batches=n_batches,
            truncation=truncation,
            correlated=correlated,
        )
        # Energy books in exact integer quanta (Python ints, unbounded), converted
        # to MeV once at the end: float accumulation order would otherwise make
        # the tallies — unlike the matrix — depend on the group size.
        deposited_quanta = 0
        escaped_quanta = 0

        for group_start in range(0, n_beamlets, beamlet_group_size):
            group = min(beamlet_group_size, n_beamlets - group_start)
            bounds = np.array(
                [source.beamlet_bounds(group_start + k) for k in range(group)],
                dtype=np.float64,
            )
            x_lo = wp.array(bounds[:, 0].astype(np.float32), dtype=float, device=device)
            x_extent = wp.array(
                (bounds[:, 1] - bounds[:, 0]).astype(np.float32), dtype=float, device=device
            )
            y_lo = wp.array(bounds[:, 2].astype(np.float32), dtype=float, device=device)
            y_extent = wp.array(
                (bounds[:, 3] - bounds[:, 2]).astype(np.float32), dtype=float, device=device
            )

            block_histories = group * n_histories_per_beamlet  # all batches at once
            chunk = min(self.chunk_size, block_histories)
            capacity = chunk * self.queue_factor
            queues = [_upload_queue(capacity, device) for _ in range(4)]
            slots = wp.zeros(capacity, dtype=wp.uint32, device=device)
            edep = wp.zeros(group * n_batches * n_voxels, dtype=wp.int64, device=device)
            scorer = BatchedBeamletScorer(self.grid, n_batches, group)

            escaped.zero_()
            t = 0
            while t < block_histories:
                n_chunk = min(chunk, block_histories - t)
                for q in queues:
                    _reset_count(q, device)
                wp.launch(
                    kernels.generate_beamlet_lattice,
                    dim=n_chunk,
                    inputs=[
                        seed,
                        group_start,
                        n_histories_per_beamlet,
                        per_batch,
                        n_batches,
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
                    device=device,
                )
                _set_count(queues[0], n_chunk, device)
                self._drain_queues(
                    queues[0],
                    queues[1],
                    queues[2],
                    queues[3],
                    gi,
                    density,
                    material,
                    tab,
                    slots,
                    edep,
                    escaped,
                    violations,
                    pcut,
                    ecut,
                    transport_electrons,
                    device,
                )
                t += n_chunk
            wp.synchronize_device(device)
            if int(violations.numpy()[0]) != 0:
                raise RuntimeError(
                    "Woodcock majorant violated in the kernel despite table headroom. "
                    "The geometry contains material or density the majorant "
                    "declaration did not cover."
                )
            group_quanta = edep.numpy()
            deposited_quanta += int(group_quanta.sum())
            escaped_quanta += int(escaped.numpy()[0])
            group_energy = group_quanta.astype(np.float64) * ENERGY_QUANTUM_MEV
            group_energy = group_energy.reshape(group, n_batches, n_voxels)
            for batch in range(n_batches):
                scorer.deposit_block(group_energy[:, batch, :])
                scorer.end_batch(per_batch)

            block = scorer.finalize()
            assembler.add_block(group_start, block.dose, block.sigma)

        return assembler.finalize(
            energy_emitted=n_beamlets * n_histories_per_beamlet * source.energy,
            energy_deposited=deposited_quanta * ENERGY_QUANTUM_MEV,
            energy_escaped=escaped_quanta * ENERGY_QUANTUM_MEV,
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

        self._drain_queues(
            q_photon,
            q_photon_alt,
            q_electron,
            q_electron_alt,
            gi,
            density,
            material,
            tab,
            slots,
            edep,
            escaped,
            violations,
            pcut,
            ecut,
            transport_electrons,
            device,
        )

    def _transport_chunk_phsp(
        self,
        source,
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
    ) -> float:
        """Host-sample a phase-space chunk, seed the queues by kind, drain.

        Returns the chunk's emitted energy: sum of ``weight * energy`` over the
        sampled records, plus ``weight * 2 m_e c^2`` for each positron whose
        annihilation photons the device books into deposited/escaped (so the
        emitted = deposited + escaped ledger closes, as on the reference backend).
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
        self._seed_queue(
            q_photon, seed, hist[photon], transport_kind[photon], batch, photon, device
        )
        self._seed_queue(
            q_electron, seed, hist[charged], transport_kind[charged], batch, charged, device
        )

        self._drain_queues(
            q_photon,
            q_photon_alt,
            q_electron,
            q_electron_alt,
            gi,
            density,
            material,
            tab,
            slots,
            edep,
            escaped,
            violations,
            pcut,
            ecut,
            transport_electrons,
            device,
        )

        latent = np.where(pt == 3, 2.0 * ELECTRON_MASS_MEV, 0.0)
        return float(
            np.sum(
                batch["weight"].astype(np.float64) * (batch["energy"].astype(np.float64) + latent)
            )
        )

    def _seed_queue(self, queue, seed, hist_g, kind_g, batch, mask, device) -> None:
        """Upload one kind-group's primaries and write them into ``queue``."""
        n = int(hist_g.shape[0])
        if n == 0:
            _set_count(queue, 0, device)
            return
        cols = {
            name: wp.array(batch[name][mask], dtype=float, device=device)
            for name in ("energy", "x", "y", "z", "ux", "uy", "uz", "weight")
        }
        wp.launch(
            kernels.generate_from_upload,
            dim=n,
            inputs=[
                seed,
                wp.array(hist_g, dtype=wp.int32, device=device),
                wp.array(kind_g, dtype=wp.int32, device=device),
                cols["energy"],
                cols["x"],
                cols["y"],
                cols["z"],
                cols["ux"],
                cols["uy"],
                cols["uz"],
                cols["weight"],
                queue,
            ],
            device=device,
        )
        _set_count(queue, n, device)

    def _drain_queues(
        self,
        q_photon,
        q_photon_alt,
        q_electron,
        q_electron_alt,
        gi,
        density,
        material,
        tab,
        slots,
        edep,
        escaped,
        violations,
        pcut,
        ecut,
        transport_electrons,
        device,
    ) -> None:
        """Ping-pong the photon and electron kernels until every queue is empty."""
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
        gi.nx = self.grid.shape[0]
        gi.ny = self.grid.shape[1]
        gi.nz = self.grid.shape[2]
        gi.n_voxels = int(np.prod(self.grid.shape))
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
        tab.coherent_x = wp.array(host.coherent_x.astype(np.float32), dtype=float, device=device)
        tab.coherent_cumulative = wp.array(
            host.coherent_cumulative.astype(np.float32), dtype=float, device=device
        )
        tab.p_log_e_min = host.photon_log_e_min
        tab.p_inv_dlog = host.photon_inv_dlog
        tab.e_log_e_min = host.electron_log_e_min
        tab.e_inv_dlog = host.electron_inv_dlog
        tab.n_points = host.n_points
        tab.n_coherent = host.n_coherent
        return tab
