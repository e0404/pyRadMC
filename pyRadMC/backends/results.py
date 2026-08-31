"""Backend-agnostic transport results.

Every backend returns the same result type from ``run``, so tests and callers compare
backends without caring which engine produced what. Defined here, above the backend
subpackages, because a result is not backend code (AGENTS.md section 3: backends hold
launch and memory management only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RunProvenance", "TransportResult"]


@dataclass(frozen=True)
class RunProvenance:
    """The configuration a result was produced under, carried with the result.

    A dose array outlives the process that made it: it is archived, handed to an
    optimizer, attached to a plan, compared against a run from six months ago. Every
    field here changes the numbers, and none of them is recoverable from the array
    afterwards — so a result that does not carry them is not reproducible, however
    carefully the run was scripted.

    This is a record, not a control surface: constructing one does not configure
    anything, and the engines fill it in from the arguments they were actually
    called with (a resolved ``step_energy_fraction``, not the ``None`` the caller
    may have passed).
    """

    version: str
    """``pyRadMC.__version__`` of the engine that produced the result."""
    backend: str
    """``"ref"`` or ``"warp"``."""
    device: str
    """``"cpu"`` or a CUDA device such as ``"cuda:0"``. Results are bit-reproducible
    for a given seed *on one device*, never across devices (AGENTS.md section 2.3)."""
    seed: int
    """Global seed; history ``i`` used the stream ``(seed, i)``."""
    pcut_mev: float
    """Photon transport cutoff in MeV."""
    ecut_mev: float
    """Electron transport and production cutoff, kinetic energy in MeV."""
    msc_model: str
    """Multiple-scattering model: ``"gs"`` (shipped) or ``"gaussian"``."""
    step_energy_fraction: float
    """Resolved electron substep energy-loss fraction, never ``None``."""
    cross_sections: str
    """Cross-section provenance, from
    :attr:`~pyRadMC.data.interface.CrossSectionSource.provenance` — the compiled
    library citation for a tabulated source, the parameterization for the analytic
    one. This is the field that distinguishes two otherwise identical runs."""
    deposit_resolution_cm: float | None = None
    """Longest piece a half-substep's continuous energy loss was filed as, in cm.

    ``None`` is the single midpoint deposit — the default, and what every result
    produced before this option existed used. A value moves dose *within* a
    transport voxel (never between voxels, and never any total), so two runs that
    differ only here agree on the energy books and on any dose scored at voxel
    resolution, and can differ below it. That is exactly why it is recorded: it is
    not recoverable from the dose array."""

    def summary(self) -> str:
        """One-line human-readable digest, for logs and file headers."""
        return (
            f"pyRadMC {self.version} {self.backend}/{self.device} seed={self.seed} "
            f"pcut={self.pcut_mev} ecut={self.ecut_mev} msc={self.msc_model} "
            f"step={self.step_energy_fraction} deposit_res={self.deposit_resolution_cm} "
            f"xs=[{self.cross_sections}]"
        )


@dataclass(frozen=True)
class TransportResult:
    """One engine run: batched dose estimate plus exact energy bookkeeping.

    ``energy_emitted == energy_deposited + energy_unscored + energy_escaped`` holds
    to accumulation precision of the producing backend — float64 exact for ``ref``,
    float32 transport arithmetic plus scoring quantization for ``warp`` — and is
    asserted in the integration tier at each backend's documented tolerance.

    ``energy_escaped`` is a *ledger*, not purely physical escape: it
    also carries the net weight-energy Russian roulette removes from the transported
    population (kills positive, survivor boosts negative), which is exactly what
    keeps the identity above exact per run under variance reduction.

    ``energy_unscored`` is deposit energy that landed inside the transport grid but
    outside the scoring grid (decoupled dose grid). It is exactly zero when
    the scoring grid covers the transport grid — in particular for the default
    score-on-the-transport-grid configuration.
    """

    dose: np.ndarray
    """Per-voxel dose on the scoring grid, MeV/g per emitted history."""
    dose_sigma: np.ndarray
    """Per-voxel 1-sigma standard error from batch statistics."""
    energy_emitted: float
    energy_deposited: float
    energy_escaped: float
    n_histories: int
    n_batches: int
    energy_unscored: float = 0.0
    scoring_mode: str = "dose_to_medium"
    """Tally weighting the dose was produced under: ``"dose_to_medium"``
    or ``"dose_to_water"``. The energy books are physical in both modes."""
    provenance: RunProvenance | None = None
    """How this result was produced; see :class:`RunProvenance`.

    Optional on the dataclass so that a hand-assembled result (a test instrument, a
    reload from disk) need not fabricate one, but **every engine run populates it** —
    that contract is test-pinned rather than expressed in the type, because it is a
    property of the engines, not of the container."""
