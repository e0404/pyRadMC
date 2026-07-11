"""Interaction channel selection.

Pure scalar functions (AGENTS.md section 2.5). The macroscopic per-channel coefficients
arrive as plain floats, already looked up through
:class:`pyRadMC.data.interface.CrossSectionSource` by the transport loop — this module
never sees where they came from (AGENTS.md section 2.6).
"""

from __future__ import annotations

from pyRadMC.data.interface import PhotonProcess
from pyRadMC.rng import RNGState, uniform

__all__ = ["select_photon_process"]


def select_photon_process(
    mu_compton: float,
    mu_photoelectric: float,
    mu_pair: float,
    mu_rayleigh: float,
    rng_state: RNGState,
) -> int:
    """Select the interaction channel with probability mu_i / mu_total.

    Cumulative inversion on a single uniform: u mu_total is compared against running
    partial sums. Because u < 1 (RNG contract), a channel with zero coefficient can
    never be selected, including at the bin boundaries.

    Channel-complete (AGENTS.md section 2.10): all four ``PhotonProcess`` channels
    are offered unconditionally; a disabled channel is a zero coefficient supplied
    by the data source, never a missing branch. With the default (analytic) data
    the coherent coefficient is exactly zero and Rayleigh is unreachable.

    Parameters
    ----------
    mu_compton, mu_photoelectric, mu_pair, mu_rayleigh
        Per-channel macroscopic attenuation coefficients, in consistent units (their
        ratios are all that matters).
    rng_state
        Per-history RNG state; consumes one uniform.

    Returns
    -------
    int
        One of the :class:`pyRadMC.data.interface.PhotonProcess` constants.
    """
    threshold = uniform(rng_state) * (mu_compton + mu_photoelectric + mu_pair + mu_rayleigh)
    if threshold < mu_compton:
        return PhotonProcess.COMPTON
    if threshold < mu_compton + mu_photoelectric:
        return PhotonProcess.PHOTOELECTRIC
    if threshold < mu_compton + mu_photoelectric + mu_pair:
        return PhotonProcess.PAIR
    return PhotonProcess.RAYLEIGH
