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
from functools import partial

from pyRadMC import DIJ_TRUNCATION_RELATIVE, ECUT_MEV, ELECTRON_MASS_MEV, PCUT_MEV
from pyRadMC.backends.results import TransportResult
from pyRadMC.data.interface import CrossSectionSource
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.geometry.source import BeamletSource, Source
from pyRadMC.progress import ProgressCallback, ProgressEmitter
from pyRadMC.rng.interface import RNG
from pyRadMC.scoring.dij import BatchedBeamletScorer, DijAssembler, DijResult
from pyRadMC.scoring.dose import BatchedDoseScorer
from pyRadMC.scoring.dose_to_water import validate_scoring_mode, water_spr
from pyRadMC.scoring.grid import ScoringGrid
from pyRadMC.transport.electron import STEP_ENERGY_FRACTION
from pyRadMC.transport.history import transport_history
from pyRadMC.transport.particles import (
    ELECTRON,
    PHOTON,
    POSITRON,
    DepositWeightFn,
    unit_weight,
)

__all__ = ["ReferenceEngine", "TransportResult"]

# Maps a Primary's per-particle kind name to a transport particle constant. A
# phase-space source sets ``kind`` per record; beam sources leave it None and the
# run's ``primary_kind`` argument decides.
_KIND_TO_PARTICLE = {"photon": PHOTON, "electron": ELECTRON, "positron": POSITRON}


def _deposit_weight_for(
    scoring_mode: str, cross_sections: CrossSectionSource, ecut: float, transport_electrons: bool
) -> DepositWeightFn:
    """Validate the scoring mode and build the per-deposit weight it selects.

    ``scoring_mode`` is a scoring-OUTPUT selection, not a physics toggle
    (AGENTS.md 2.10): transport is identical in both modes — the RNG streams,
    interaction sampling and stepping never see it — only the tally weighting of
    each deposit differs, exactly like the choice of scoring grid. Dose-to-water
    is refused in KERMA mode: with no tracked electron the stopping-power ratio
    has nothing to be evaluated on (see :mod:`pyRadMC.scoring.dose_to_water`).
    """
    if not validate_scoring_mode(scoring_mode, transport_electrons):
        return unit_weight
    return partial(water_spr, cross_sections=cross_sections, ecut=ecut)


@dataclass(frozen=True)
class ReferenceEngine:
    """Single-threaded reference photon engine over a voxel grid."""

    grid: VoxelGrid
    cross_sections: CrossSectionSource
    rng: RNG

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
        step_energy_fraction: float = STEP_ENERGY_FRACTION,
        progress: ProgressCallback | None = None,
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
            ``"photon"`` (default) or ``"electron"``: the fallback kind for sources
            whose emitted :class:`~pyRadMC.geometry.source.Primary` leaves ``kind``
            unset (the monoenergetic beam sources). A phase-space source overrides
            it per record, so this argument is ignored for that source. The electron
            option exists for validating electron transport against ranges; electron
            *beams* as a clinical modality remain out of scope (AGENTS.md 6).
        scoring_grid
            Dose grid to accumulate on (Phase 5 decoupled scoring). ``None``
            (default) scores on the transport grid — byte-identical to the
            engine before scoring grids existed. Build a coarser, offset or
            subregion grid with :meth:`pyRadMC.scoring.grid.ScoringGrid.rebin`
            **from the same transport grid handed to this engine**; deposits it
            does not cover are booked to ``TransportResult.energy_unscored``, so
            ``emitted == deposited + unscored + escaped`` stays exact. Transport
            never sees this grid: the streams, and hence the physics, are
            invariant to it.
        scoring_mode
            ``"dose_to_medium"`` (default) or ``"dose_to_water"`` — a
            scoring-OUTPUT selection (see :func:`_deposit_weight_for` and
            :mod:`pyRadMC.scoring.dose_to_water`): transport is identical, only
            the per-deposit tally weighting differs, and the energy books stay
            physical in both modes. Requires ``transport_electrons=True``.
        step_energy_fraction
            Maximum fraction of CSDA range per electron substep
            (:data:`~pyRadMC.transport.electron.STEP_ENERGY_FRACTION`, the
            validated default). A measurement instrument for the substep
            resolution/bias trade-off; changing the *default* is a maintainer
            decision gated on the validation tier.
        progress
            Optional callback invoked with a :class:`~pyRadMC.progress.ProgressEvent`
            once per completed batch (``n_batches`` ticks total, each covering
            ``n_histories / n_batches`` histories). See
            :mod:`pyRadMC.progress` — the same tick cadence as
            :meth:`~pyRadMC.backends.warp.engine.WarpEngine.run`, so a callback
            written against one backend behaves identically against the other.
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
        deposit_weight = _deposit_weight_for(
            scoring_mode, self.cross_sections, ecut, transport_electrons
        )

        scorer = BatchedDoseScorer(
            scoring_grid if scoring_grid is not None else self.grid, n_batches
        )
        per_batch = n_histories // n_batches
        energy_emitted = 0.0
        energy_escaped = 0.0
        emitter = ProgressEmitter(progress, n_histories)

        history = 0
        for _ in range(n_batches):
            for _ in range(per_batch):
                state = self.rng.init_state(seed, history)
                history += 1
                primary = source.emit(state)
                kind_name = primary.kind if primary.kind is not None else primary_kind
                kind = _KIND_TO_PARTICLE[kind_name]
                # A positron primary will annihilate at rest, injecting 2*m_e c^2 of
                # photons from rest mass that its kinetic energy does not account for.
                # (For a photon that pair-produces, that 1.022 MeV is already inside
                # the photon's energy; a positron primary brings it as rest mass.)
                # Count it so the emitted = deposited + escaped ledger stays exact.
                rest_mass = 2.0 * ELECTRON_MASS_MEV if kind == POSITRON else 0.0
                energy_emitted += primary.weight * (primary.energy + rest_mass)
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
                    scorer.deposit_at,
                    pcut,
                    ecut,
                    transport_electrons,
                    weight=primary.weight,
                    deposit_weight=deposit_weight,
                    step_energy_fraction=step_energy_fraction,
                )
            scorer.end_batch(per_batch)
            emitter.tick(per_batch)

        dose = scorer.finalize()
        return TransportResult(
            dose=dose.dose,
            dose_sigma=dose.dose_sigma,
            energy_emitted=energy_emitted,
            energy_deposited=dose.energy_deposited,
            energy_escaped=energy_escaped,
            energy_unscored=dose.energy_unscored,
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
        scoring_grid: ScoringGrid | None = None,
        scoring_mode: str = "dose_to_medium",
        step_energy_fraction: float = STEP_ENERGY_FRACTION,
        progress: ProgressCallback | None = None,
    ) -> DijResult:
        """Compute the beamlet-resolved dose influence matrix over the lattice.

        History-to-beamlet mapping — the project-wide convention every backend
        follows: history ``h`` feeds beamlet ``j = h // n_histories_per_beamlet``,
        and within a beamlet, batch ``b`` owns the contiguous slice of
        ``n_histories_per_beamlet / n_batches`` histories starting at
        ``j * n_histories_per_beamlet + b * (that slice length)``. Streams are pure
        functions of ``(seed, h)``, so the Dij is bit-reproducible on one target
        regardless of how a backend schedules the transport, and a 1x1 lattice
        reproduces the open-field :meth:`run` bit for bit (test-pinned).

        Correlated sampling (Phase 4) changes the *stream key* only: with
        ``correlated=True``, the history at within-beamlet index
        ``rw = h - j * n_histories_per_beamlet`` draws the stream ``(seed, rw)``
        instead of ``(seed, h)``, so corresponding histories of every beamlet
        replay the same random sequence — same within-bixel entry offset, same
        interaction sequence — and only the beamlet's position differs. Beamlet
        assignment, batching, scoring and the energy books are untouched, and on
        a 1x1 lattice ``rw == h``, so the open-field anchor above holds in both
        modes (test-pinned).

        Every deposit of a history's whole secondary family scores into its
        beamlet's column: the columns partition the open-field dose exactly.
        Column doses are per emitted history *of that beamlet*, MeV/g.

        Parameters mirror :meth:`run`; the two Dij-specific ones:

        Parameters
        ----------
        n_histories_per_beamlet
            Histories per beamlet (equal by design — stratified, not sampled);
            must be divisible by ``n_batches``.
        truncation
            Per-column relative truncation threshold. Accuracy-defining
            (AGENTS.md 2.8): the default is :data:`pyRadMC.DIJ_TRUNCATION_RELATIVE`
            and a different value in a call is a visible, greppable decision.
        correlated
            Key streams on the within-beamlet index so columns share random
            sequences (correlated sampling). **This is the shipped
            configuration** (default True): the Phase 4 noise/bias study
            (``examples/phase4_noise_bias_study.py``) found it halves the
            renormalized plan-dose error at matched per-beamlet sigma, in water
            and through a heterogeneity, and never worse on raw plan quality.
            ``correlated=False`` selects the independent mapping and exists
            only as a **test instrument** (AGENTS.md 2.10): it isolates the
            column independence the fluence-sum identity's quadrature sigma
            needs. A correlated Dij's columns are statistically dependent —
            per-column sigmas stay valid, but never combine sigmas across
            columns in quadrature. The result records the mode in
            ``DijResult.correlated``.
        scoring_grid
            Dose grid the Dij columns live on; semantics as in :meth:`run`. The
            memory lever for plan-scale problems: the dense per-group buffers and
            the sparse Dij all scale with the *scoring* voxel count, so a coarser
            dose grid shrinks them cubically while transport keeps the full CT
            resolution.
        scoring_mode
            Tally weighting of the columns, as in :meth:`run`; recorded in
            ``DijResult.scoring_mode``.
        progress
            Optional callback, as in :meth:`run`. Ticks once per completed batch
            (``n_batches`` ticks total), each covering
            ``n_beamlets * n_histories_per_beamlet / n_batches`` histories — this
            engine iterates batch-outer, beamlet-inner, so a batch spans every
            beamlet. :meth:`~pyRadMC.backends.warp.engine.WarpEngine.run_dij`
            ticks on a different axis (per beamlet group, not per batch): both
            reach the same total, but tick count and spacing differ between
            backends. Treat ``histories_done / histories_total`` as the portable
            signal (see :mod:`pyRadMC.progress`).
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

        deposit_weight = _deposit_weight_for(
            scoring_mode, self.cross_sections, ecut, transport_electrons
        )
        n_beamlets = source.n_beamlets
        per_batch = n_histories_per_beamlet // n_batches
        scoring = scoring_grid if scoring_grid is not None else ScoringGrid.for_grid(self.grid)
        scorer = BatchedBeamletScorer(scoring, n_batches, n_beamlets)
        energy_emitted = 0.0
        energy_escaped = 0.0
        emitter = ProgressEmitter(progress, n_beamlets * n_histories_per_beamlet)

        for batch in range(n_batches):
            for beamlet in range(n_beamlets):
                deposit = partial(scorer.deposit_at, beamlet)
                for r in range(per_batch):
                    rw = batch * per_batch + r
                    h = beamlet * n_histories_per_beamlet + rw
                    state = self.rng.init_state(seed, rw if correlated else h)
                    primary = source.emit(beamlet, state)
                    energy_emitted += primary.weight * primary.energy
                    energy_escaped += transport_history(
                        PHOTON,
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
                        deposit,
                        pcut,
                        ecut,
                        transport_electrons,
                        weight=primary.weight,
                        deposit_weight=deposit_weight,
                        step_energy_fraction=step_energy_fraction,
                    )
            scorer.end_batch(per_batch)
            emitter.tick(n_beamlets * per_batch)

        block = scorer.finalize()
        assembler = DijAssembler(
            grid_shape=scoring.shape,
            n_beamlets=n_beamlets,
            n_histories_per_beamlet=n_histories_per_beamlet,
            n_batches=n_batches,
            truncation=truncation,
            correlated=correlated,
            scoring_mode=scoring_mode,
        )
        assembler.add_block(0, block.dose, block.sigma)
        return assembler.finalize(
            energy_emitted=energy_emitted,
            energy_deposited=block.energy_deposited,
            energy_escaped=energy_escaped,
            energy_unscored=block.energy_unscored,
        )
