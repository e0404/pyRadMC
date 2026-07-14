"""Electron substeps stay within one voxel across a density interface.

An electron condensed-history substep evaluates its energy loss, scattering and
interaction sampling with its *start* voxel's density and material. Without capping
the substep at the next voxel face, a step that crosses a sharp density interface
plows one medium's stopping power across the boundary and deposits it into the
neighbour's (much smaller) mass, producing a single-voxel dose spike right at the
interface (the artifact the Phase 5 materials demo surfaced).

This drives the fix directly through ``transport.electron.electron_steps``: transport
many electrons across a pure density step (same material, so the mass stopping power is
identical on both sides and the only thing that changes is density). The mass-scaled
deposit ``edep / rho`` is proportional to dose-to-medium, which is continuous across a
pure density change (electron fluence and mass stopping are both continuous), so a
smooth profile is the correct answer and a spike at the interface voxel is the bug.
"""

from __future__ import annotations

import numpy as np

from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.materials import WATER
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.rng.host import HostRNG
from pyRadMC.transport.electron import electron_steps
from tests.conftest import SEED

NZ = 80
DZ = 0.2  # cm; half-voxel (0.1) is the substep limiter at 6 MeV, maximizing any plow-through
INTERFACE = 40  # z-voxel where density drops
LOW_DENSITY = 0.2
ENERGY = 6.0
PCUT = 0.05
ECUT = 0.2
N_ELECTRONS = 30_000


def _phantom() -> VoxelGrid:
    density = np.ones((5, 5, NZ), dtype=np.float64)
    density[:, :, INTERFACE:] = LOW_DENSITY
    return VoxelGrid(
        shape=(5, 5, NZ),
        spacing=(2.0, 2.0, DZ),
        density=density,
        material=np.full((5, 5, NZ), WATER, dtype=np.int32),
    )


def _transport_profile() -> np.ndarray:
    """Mass-scaled deposit per z-slab from many electrons crossing the interface."""
    grid = _phantom()
    xs = AnalyticCrossSections()
    rng = HostRNG()
    edep = np.zeros(NZ)

    def deposit(x: float, y: float, z: float, amount: float, scored: float) -> None:
        # Position-keyed deposits; bin by the z-voxel exactly as the grid would.
        edep[min(int(z / DZ), NZ - 1)] += amount

    def spawn(_entry: object) -> None:
        # Secondaries carry the same substep logic; ignoring them isolates the
        # primary's continuous-loss deposit, which is where the artifact lives.
        pass

    # Start just upstream of the interface so every electron crosses it; 6 MeV range in
    # water is ~3 cm, so they penetrate well into the low-density region beyond.
    z0 = (INTERFACE - 5) * DZ
    for i in range(N_ELECTRONS):
        state = rng.init_state(SEED, i)
        electron_steps(
            False,
            ENERGY,
            1.0,
            5.0,
            5.0,
            z0,
            0.0,
            0.0,
            1.0,
            grid,
            xs,
            state,
            deposit,
            spawn,
            PCUT,
            ECUT,
        )

    rho = np.where(np.arange(NZ) >= INTERFACE, LOW_DENSITY, 1.0)
    return edep / rho


def test_interface_voxel_is_not_spiked() -> None:
    """The mass-scaled deposit is smooth across the interface, not spiked at it.

    Dose-to-medium is continuous across a pure density change, so the interface
    voxel must sit close to its neighbours. Before the substep boundary cap the
    interface voxel overshoots them by well over 2x (the ~1/LOW_DENSITY plow-through);
    the cap brings it into line. The 1.4x threshold sits clear of both regimes.
    """
    profile = _transport_profile()
    interface = profile[INTERFACE]
    downstream = profile[INTERFACE + 1 : INTERFACE + 6].mean()
    upstream = profile[INTERFACE - 6 : INTERFACE - 1].mean()
    reference = max(downstream, upstream)
    assert interface < 1.4 * reference, (
        f"interface voxel mass-scaled deposit {interface:.4e} spikes above "
        f"neighbours {reference:.4e} (ratio {interface / reference:.2f})"
    )
