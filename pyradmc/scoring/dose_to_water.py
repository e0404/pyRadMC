"""On-the-fly dose-to-water weighting: the restricted stopping-power ratio.

Dose-to-medium (the engines' default) divides deposited energy by the medium
mass. Dose-to-water instead asks what a small water cavity at the same point
would absorb: under Bragg-Gray conditions each electron energy deposit ``dE``
converts with the **mass collision stopping-power ratio** water/medium at the
electron's energy (Siebers, Keall, Nahum & Mohan, Phys. Med. Biol. 45 (2000)
983, doi:10.1088/0031-9155/45/4/983). The engines apply the ratio per deposit,
*before* tallying — spectrum-correct for the condensed-history deposits that
carry almost all the energy — while the energy ledger keeps booking the
physical (unweighted) energy, so ``emitted == deposited + unscored + escaped``
is mode-invariant.

Both stopping powers come from the engine's
:class:`~pyradmc.data.interface.CrossSectionSource` (AGENTS.md 2.6): the
analytic backend supplies Berger-Seltzer values, the tabulated backend its
compiled tables, and nothing here hardcodes a material property. The restricted
(not unrestricted) collision stopping power is used deliberately: it is the
same quantity transport charged the deposit with, so the conversion undoes
exactly the weighting the medium applied.

Stated approximations, named here because this is where they are implemented
(AGENTS.md 2.9):

- **Sub-cutoff deposits freeze the ratio at ECUT.** Local deposits of untracked
  particles (sub-ECUT recoils and delta rays, terminal residues, sub-PCUT
  photons) carry no spectrum; their weight is the SPR at ``max(E, ecut)``,
  which also keeps the evaluation inside the validated stopping-power domain.
  Sub-PCUT *photon* deposits would strictly need mass-energy-absorption ratios;
  at 1-20 MeV in low-Z media they carry a sub-percent energy share and the
  frozen electron SPR is applied instead.
- **KERMA mode is refused.** With ``transport_electrons=False`` there is no
  electron to evaluate the ratio on at all; the engines raise rather than
  silently produce mass-energy-absorption-weighted-in-name-only numbers.

The Warp kernels mirror this arithmetic in-kernel from the flattened stopping
tables (``_spr_factor`` in :mod:`pyradmc.backends.warp.kernels`).
"""

from __future__ import annotations

from pyradmc.data.interface import CrossSectionSource
from pyradmc.data.materials import WATER

__all__ = ["validate_scoring_mode", "water_spr"]


def validate_scoring_mode(scoring_mode: str, transport_electrons: bool) -> bool:
    """Validate an engine ``scoring_mode``; True selects dose-to-water.

    Shared by both engines so the accepted modes and the KERMA refusal (module
    docstring) have exactly one definition.
    """
    if scoring_mode not in ("dose_to_medium", "dose_to_water"):
        raise ValueError(
            f"unknown scoring_mode {scoring_mode!r}; expected 'dose_to_medium' or 'dose_to_water'"
        )
    if scoring_mode == "dose_to_water" and not transport_electrons:
        raise ValueError(
            "dose_to_water requires electron transport: KERMA mode deposits carry no "
            "electron to evaluate the stopping-power ratio on (it would need "
            "mass-energy-absorption ratios instead)"
        )
    return scoring_mode == "dose_to_water"


def water_spr(
    energy: float,
    material: int,
    *,
    cross_sections: CrossSectionSource,
    ecut: float,
) -> float:
    """Restricted collision stopping-power ratio water/medium at ``energy``.

    ``energy`` is clamped to ``ecut`` from below (module docstring). Exactly 1.0
    when ``material`` is water: the ratio of two identical lookups. Keyword-only
    data arguments so the engines can bind them with ``functools.partial`` into
    the transport loops' ``deposit_weight`` callback.
    """
    e = max(energy, ecut)
    return cross_sections.restricted_stopping_power(
        e, WATER, ecut
    ) / cross_sections.restricted_stopping_power(e, material, ecut)
