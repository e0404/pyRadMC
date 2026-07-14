"""Channel-completeness of the photon transport loop (AGENTS.md 2.10).

The loop must sample every interaction channel the ``CrossSectionSource`` reports as
nonzero — enabling a channel is a data change, never a transport change. Exercised
with a synthetic coherent-only source: photons must random-walk without ever losing
energy, so nothing deposits and everything eventually escapes with full energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.data.interface import CrossSectionSource, PhotonProcess
from pyRadMC.geometry.grid import VoxelGrid
from pyRadMC.rng.host import HostRNG
from pyRadMC.transport.photon import transport_photon
from tests.conftest import SEED


class CoherentOnlySource(CrossSectionSource):
    """A data source whose only nonzero channel is Rayleigh scattering.

    Test instrument: no such medium exists; it isolates the coherent branch of the
    transport loop, which no physical data source currently reaches.
    """

    MU_RHO = 0.08  # cm^2/g

    def mu_over_rho(self, energy: float, material: int, process: int) -> float:
        return self.MU_RHO if process == PhotonProcess.RAYLEIGH else 0.0

    def mu_over_rho_total(self, energy: float, material: int) -> float:
        return self.MU_RHO

    def majorant(self, energy: float) -> float:
        return self.MU_RHO * 1.0  # unit-density water phantom below

    def restricted_stopping_power(self, energy: float, material: int, delta_cut: float) -> float:
        raise AssertionError("no charged particles exist in a coherent-only medium")

    def radiative_stopping_power(self, energy: float, material: int) -> float:
        raise AssertionError("no charged particles exist in a coherent-only medium")

    def moller_cross_section(self, energy: float, material: int, delta_cut: float) -> float:
        raise AssertionError("no charged particles exist in a coherent-only medium")

    def csda_range(self, energy: float, material: int) -> float:
        raise AssertionError("no charged particles exist in a coherent-only medium")

    def scattering_power(self, energy: float, material: int) -> float:
        raise AssertionError("no charged particles exist in a coherent-only medium")


class _CountingState:
    """Wraps a NumPy generator, counting draws; duck-types the HostRNG state."""

    def __init__(self, generator: np.random.Generator) -> None:
        self._generator = generator
        self.count = 0

    def random(self) -> float:
        self.count += 1
        return float(self._generator.random())


def test_coherent_channel_is_sampled_and_conserves_energy() -> None:
    """Nonzero coherent data must reach the Rayleigh branch, not crash or deposit.

    Every history ends with the photon escaping at its full energy — coherent
    scattering changes direction only — and none may deposit. That the branch is
    genuinely exercised is pinned through the RNG draw count: the majorant here
    equals the real cross-section, so there is no delta scattering, and any photon
    whose first flight ends inside the phantom *must* interact. A never-interacting
    history consumes exactly one draw (its escape flight); an interacting one
    consumes at least five (path, channel selection, two-plus rejection draws for
    the cosine, azimuth). At mu = 0.08/cm over ~17 cm the interacting fraction is
    ~74 percent; requiring 200/400 sits 13 sigma below it.
    """
    grid = VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    xs = CoherentOnlySource()
    rng = HostRNG()
    energy = 1.0

    deposits: list[float] = []

    def deposit(x: float, y: float, z: float, e: float, scored: float) -> None:
        deposits.append(e)

    n = 400
    escaped_total = 0.0
    interacted = 0
    for history in range(n):
        state = _CountingState(rng.init_state(SEED, history))
        escaped = transport_photon(
            energy, 8.0, 8.0, -1.0, 0.0, 0.0, 1.0, grid, xs, state, deposit, pcut=0.05
        )
        assert escaped == pytest.approx(energy, rel=1e-12), "coherent event lost energy"
        assert deposits == [], "coherent-only medium must never deposit"
        escaped_total += escaped
        if state.count >= 5:
            interacted += 1

    assert escaped_total == pytest.approx(n * energy, rel=1e-12)
    assert interacted > 200, f"only {interacted}/{n} histories reached the coherent branch"
