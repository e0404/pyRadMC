"""Discrete thin-target bremsstrahlung.

Pure scalar functions (AGENTS.md section 2.5). The model, named here where it
is implemented (maintainer-approved, docs/decisions.md):

- Photon energies follow a thin-target ``1/k`` spectrum between PCUT and the electron
  kinetic energy — the leading behaviour of the Bethe-Heitler cross-section (Bethe &
  Heitler (1934), doi:10.1098/rspa.1934.0140; Koch & Motz, Rev. Mod. Phys. 31, 920
  (1959), doi:10.1103/RevModPhys.31.920). Energy radiated per unit k is then uniform
  in k, so the fraction of the radiative loss carried by photons above PCUT is simply
  ``1 - PCUT/E``; the remainder is deposited locally.
- At most one photon is emitted per condensed-history substep, with Bernoulli
  probability chosen so the *expected* emitted energy matches the radiative stopping
  power exactly (identity test-pinned). Substeps are short enough that the Bernoulli
  approximation to the Poisson count is sub-percent; the transport loop enforces it.
- Photons are emitted along the electron direction: the characteristic emission angle
  is ~1/gamma, well below the multiple-scattering width at these energies (stated
  approximation).
"""

from __future__ import annotations

import math

from pyradmc.rng import RNGState, uniform

__all__ = ["bremsstrahlung_step_parameters", "sample_bremsstrahlung_energy"]


def bremsstrahlung_step_parameters(
    energy: float,
    radiative_stopping_power: float,
    density: float,
    step_cm: float,
    pcut: float,
) -> tuple[float, float]:
    """Emission probability and local sub-cutoff deposit for one substep.

    Returns ``(probability, local_deposit_mev)`` such that

    ``probability * <k> + local_deposit == radiative_stopping_power * density * step``

    with ``<k> = (E - PCUT) / ln(E / PCUT)``, the mean of the 1/k spectrum. For an
    electron at or below PCUT no transportable photon exists and the whole radiative
    loss is local.

    Parameters
    ----------
    energy
        Electron kinetic energy in MeV.
    radiative_stopping_power
        Mass radiative stopping power in MeV cm^2/g at ``energy``.
    density
        Local mass density in g/cm^3.
    step_cm
        Substep length in cm.
    pcut
        Photon transport cutoff in MeV.
    """
    radiative_loss = radiative_stopping_power * density * step_cm
    if energy <= pcut:
        return (0.0, radiative_loss)
    emitted_fraction = 1.0 - pcut / energy
    mean_k = (energy - pcut) / math.log(energy / pcut)
    probability = radiative_loss * emitted_fraction / mean_k
    return (probability, radiative_loss * (1.0 - emitted_fraction))


def sample_bremsstrahlung_energy(energy: float, pcut: float, rng_state: RNGState) -> float:
    """Sample a photon energy from the 1/k spectrum on [PCUT, E].

    Exact inversion: ``k = PCUT * (E / PCUT)^u`` — uniform in ln k.

    Parameters
    ----------
    energy
        Electron kinetic energy in MeV; must exceed ``pcut``.
    pcut
        Photon transport cutoff in MeV.
    rng_state
        Per-history RNG state; consumes one uniform.
    """
    return pcut * math.exp(uniform(rng_state) * math.log(energy / pcut))
