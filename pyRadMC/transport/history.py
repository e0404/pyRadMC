"""One full coupled photon-electron history.

The stack orchestration over :func:`pyRadMC.transport.photon.photon_steps` and
:func:`pyRadMC.transport.electron.electron_steps`: photons spawn electrons and
positrons, electrons spawn bremsstrahlung photons and delta rays, positrons add
annihilation photons — this module just keeps popping until the family is exhausted.

Energy accounting is exact across the whole family: everything emitted by the source
is either deposited through the callback or returned as escaped.
"""

from __future__ import annotations

from pyRadMC.data.interface import CrossSectionSource
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.rng import RNGState
from pyRadMC.transport.electron import electron_steps
from pyRadMC.transport.particles import (
    PHOTON,
    POSITRON,
    DepositFn,
    DepositWeightFn,
    StackEntry,
    unit_weight,
)
from pyRadMC.transport.photon import photon_steps

__all__ = ["transport_history"]


def transport_history(
    kind: int,
    energy: float,
    x: float,
    y: float,
    z: float,
    ux: float,
    uy: float,
    uz: float,
    grid: VoxelGrid,
    cross_sections: CrossSectionSource,
    rng_state: RNGState,
    deposit: DepositFn,
    pcut: float,
    ecut: float,
    transport_electrons: bool = True,
    weight: float = 1.0,
    deposit_weight: DepositWeightFn = unit_weight,
    step_energy_fraction: float | None = None,
    msc_model: str = "gs",
) -> float:
    """Transport one primary and all its descendants; returns the escaped energy.

    Parameters
    ----------
    kind
        ``PHOTON``, ``ELECTRON`` or ``POSITRON`` from
        :mod:`pyRadMC.transport.particles`.
    energy
        Primary energy in MeV (kinetic, for charged particles).
    x, y, z, ux, uy, uz
        Start position (cm) and unit direction; outside the grid is vacuum.
    grid, cross_sections, rng_state, deposit
        As in the step loops.
    pcut, ecut
        Transport/production cutoffs in MeV. Accuracy-defining (AGENTS.md 2.8);
        always explicit here, defaulted only at the engine surface.
    transport_electrons
        False selects the KERMA approximation: charged secondaries deposit
        at their creation voxel. Explicit engine option per docs/decisions.md.
    weight
        Statistical weight of the primary (1.0 for analog beam sources; a
        phase-space record supplies its own). Descendants inherit it and the scorer
        sees weight-scaled (expected) energy throughout, so the returned escaped
        energy is weight-scaled too — callers must weight the emitted-energy book
        the same way for the energy ledger to balance.
    deposit_weight
        Scoring-output weight per deposit (dose-to-water SPR; the default books
        dose-to-medium). Applied by the step loops; see
        :mod:`pyRadMC.scoring.dose_to_water`.
    step_energy_fraction
        Maximum fraction of CSDA range per electron substep; ``None`` resolves
        to the selected ``msc_model``'s validated fraction
        (:func:`~pyRadMC.transport.electron.default_step_energy_fraction`).
        See :func:`~pyRadMC.transport.electron.electron_steps`, also for
        ``msc_model`` (default ``"gs"``, the shipped configuration).
    """
    stack: list[StackEntry] = [(kind, energy, weight, x, y, z, ux, uy, uz)]
    escaped = 0.0
    first = True
    while stack:
        particle_kind, e, w, px, py, pz, dx, dy, dz = stack.pop()
        # Only the source photon is primary — the one particle that may split at
        # its first Compton (transport.photon). It is the initial stack entry,
        # popped first; everything after descends from an interaction. A charged
        # primary (electron-beam tests) is never primary in this sense, and its
        # bremsstrahlung photons must not split, so the flag is consumed here on
        # the first pop regardless of kind.
        is_primary = first and particle_kind == PHOTON
        first = False
        if particle_kind == PHOTON:
            escaped += photon_steps(
                e,
                w,
                px,
                py,
                pz,
                dx,
                dy,
                dz,
                grid,
                cross_sections,
                rng_state,
                deposit,
                stack.append,
                pcut,
                ecut,
                transport_electrons,
                is_primary=is_primary,
                deposit_weight=deposit_weight,
            )
        else:
            escaped += electron_steps(
                particle_kind == POSITRON,
                e,
                w,
                px,
                py,
                pz,
                dx,
                dy,
                dz,
                grid,
                cross_sections,
                rng_state,
                deposit,
                stack.append,
                pcut,
                ecut,
                deposit_weight=deposit_weight,
                step_energy_fraction=step_energy_fraction,
                msc_model=msc_model,
            )
    return escaped
