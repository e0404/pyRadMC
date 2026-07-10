"""Interaction channel selection: branching ratios and degenerate cases."""

from __future__ import annotations

import numpy as np
from scipy import stats

from pyRadMC.data.interface import PhotonProcess
from pyRadMC.physics.channel import select_photon_process
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED


def test_channel_frequencies_match_branching_ratios() -> None:
    """Selection frequencies reproduce mu_i / mu_total (chi-squared, fixed seed).

    Water-at-4-MeV-like numbers: Compton-dominated with a real pair fraction, so all
    three branches are exercised.
    """
    mu = {
        PhotonProcess.COMPTON: 0.0303,
        PhotonProcess.PHOTOELECTRIC: 0.0002,
        PhotonProcess.PAIR: 0.0035,
    }
    n = 50_000
    state = HostRNG().init_state(SEED, 30)
    counts = dict.fromkeys(mu, 0)
    for _ in range(n):
        process = select_photon_process(
            mu[PhotonProcess.COMPTON],
            mu[PhotonProcess.PHOTOELECTRIC],
            mu[PhotonProcess.PAIR],
            state,
        )
        counts[process] += 1

    total = sum(mu.values())
    observed = np.array([counts[p] for p in mu])
    expected = np.array([mu[p] / total * n for p in mu])
    chi2, p_value = stats.chisquare(observed, expected)
    assert p_value > 0.01, f"branching ratios off: chi2={chi2:.2f}, p={p_value:.2e}"


def test_zero_channels_are_never_selected() -> None:
    """A channel with zero cross-section must have exactly zero probability.

    Guards the cumulative-sum edge: u drawn exactly at a bin boundary must not fall
    into an empty channel.
    """
    state = HostRNG().init_state(SEED, 31)
    for _ in range(10_000):
        process = select_photon_process(0.05, 0.0, 0.0, state)
        assert process == PhotonProcess.COMPTON

    for _ in range(10_000):
        process = select_photon_process(0.0, 0.0, 0.004, state)
        assert process == PhotonProcess.PAIR


def test_selection_covers_all_channels_and_nothing_else() -> None:
    """Every draw lands in one of the three enabled processes."""
    state = HostRNG().init_state(SEED, 32)
    valid = {PhotonProcess.COMPTON, PhotonProcess.PHOTOELECTRIC, PhotonProcess.PAIR}
    seen = set()
    for _ in range(20_000):
        process = select_photon_process(0.02, 0.01, 0.01, state)
        assert process in valid
        seen.add(process)
    assert seen == valid
