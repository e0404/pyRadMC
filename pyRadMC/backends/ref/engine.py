"""The reference engine: pure NumPy, one history at a time, never optimized.

This is the oracle every other backend is validated against (AGENTS.md section 2.2).
Clarity beats speed here, always: no vectorization over histories, no caching beyond
what correctness requires, asserts left on. If this file is ever the bottleneck, the
answer is a faster *backend*, not a faster oracle.

The engine composes the pieces — source, per-history RNG, the transport loop of
:mod:`pyRadMC.transport.photon`, the batched scorer — and owns nothing physical
itself (launch and memory management only, AGENTS.md section 3).
"""

from __future__ import annotations

from dataclasses import dataclass

from pyRadMC import ECUT_MEV, PCUT_MEV
from pyRadMC.backends.results import TransportResult
from pyRadMC.data.interface import CrossSectionSource
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import ParallelBeamSource, PencilBeamSource
from pyRadMC.rng.interface import RNG
from pyRadMC.scoring.dose import BatchedDoseScorer
from pyRadMC.transport.history import transport_history
from pyRadMC.transport.particles import ELECTRON, PHOTON

__all__ = ["ReferenceEngine", "TransportResult"]


@dataclass(frozen=True)
class ReferenceEngine:
    """Single-threaded reference photon engine over a voxel grid."""

    grid: VoxelGrid
    cross_sections: CrossSectionSource
    rng: RNG

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
        """Transport ``n_histories`` primaries in ``n_batches`` equal batches.

        Parameters
        ----------
        source
            Primary source; its geometry is particle-agnostic (see ``primary_kind``).
        n_histories
            Total primaries; must be divisible by ``n_batches`` so every batch mean
            carries equal statistical weight.
        n_batches
            Batches for the sigma estimate (AGENTS.md section 2.4).
        seed
            Global seed; history ``i`` uses the stream ``(seed, i)``, so the result
            is bit-reproducible for a given target and seed regardless of batching.
        pcut, ecut
            Photon and electron cutoffs in MeV. Accuracy-defining (AGENTS.md
            section 2.8); the defaults are the project-wide values and changing one
            in a call is a visible, greppable decision.
        transport_electrons
            False selects the Phase 0 KERMA approximation (charged secondaries
            deposit at their creation voxel) — the explicit option AGENTS.md 7.2
            keeps for photon-only physics tests.
        primary_kind
            ``"photon"`` (default) or ``"electron"``: what the source emits. The
            electron option exists for validating electron transport against ranges;
            electron *beams* as a clinical modality remain out of scope (AGENTS.md 6).
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

        scorer = BatchedDoseScorer(self.grid, n_batches)
        per_batch = n_histories // n_batches
        energy_emitted = 0.0
        energy_escaped = 0.0

        history = 0
        for _ in range(n_batches):
            for _ in range(per_batch):
                state = self.rng.init_state(seed, history)
                history += 1
                primary = source.emit(state)
                energy_emitted += primary.energy
                energy_escaped += transport_history(
                    kind,
                    primary.energy,
                    primary.x,
                    primary.y,
                    primary.z,
                    primary.ux,
                    primary.uy,
                    primary.uz,
                    self.grid,
                    self.cross_sections,
                    state,
                    scorer.deposit,
                    pcut,
                    ecut,
                    transport_electrons,
                )
            scorer.end_batch(per_batch)

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
