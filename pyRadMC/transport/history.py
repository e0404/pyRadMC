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
from pyRadMC.transport.particles import PHOTON, POSITRON, DepositFn, StackEntry
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
        False selects the Phase 0 KERMA approximation: charged secondaries deposit
        at their creation voxel. Explicit engine option per AGENTS.md 7.2.
    """
    stack: list[StackEntry] = [(kind, energy, 1.0, x, y, z, ux, uy, uz)]
    escaped = 0.0
    while stack:
        particle_kind, e, w, px, py, pz, dx, dy, dz = stack.pop()
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
            )
    return escaped
