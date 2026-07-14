"""The dose-to-water deposit weight: restricted stopping-power ratio water/medium.

The weight is pure data plumbing (AGENTS.md 2.6): both stopping powers come from
the engine's CrossSectionSource, so the analytic and tabulated backends supply
their own numbers and nothing is hardcoded. Pinned here: the water identity
(SPR == 1 exactly when the medium IS water), the sub-cutoff clamp, the exact
constant on a synthetic proportional table, and the physical sign+magnitude for
cortical bone from the Berger-Seltzer data (water/bone ~ 1.1 at MV energies).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC import ECUT_MEV
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.interface import PhotonProcess
from pyRadMC.data.materials import CORTICAL_BONE, MATERIALS, WATER
from pyRadMC.data.tabulated.model import TabulatedData
from pyRadMC.data.tabulated.source import TabulatedCrossSections

DELTA_CUT = ECUT_MEV  # the compiled cut must match the transport ecut


def _two_material_tables(spr: float = 1.1) -> TabulatedData:
    """Water row + a second row whose restricted stopping is water's / spr.

    Log-linear interpolation of a proportional row is the proportional
    interpolant, so the water/medium ratio is exactly ``spr`` at *every* energy —
    the instrument that turns the on-the-fly weighting into an exact identity.
    Rayleigh is zero so transport needs no coherent form factors. The row is
    labeled 'air' only because compiled tables must be a registry prefix; its
    numbers are synthetic.
    """
    pe = np.geomspace(0.01, 10.0, 48)
    ee = np.geomspace(0.01, 10.0, 48)
    water = {
        PhotonProcess.COMPTON: 0.15 * np.ones_like(pe),
        PhotonProcess.PHOTOELECTRIC: 1.0e-3 / pe**3,
        PhotonProcess.PAIR: np.maximum(pe - 1.022, 0.0) * 1.0e-3,
        PhotonProcess.RAYLEIGH: np.zeros_like(pe),
    }
    stopping = 2.0 / ee
    return TabulatedData(
        photon_energies=pe,
        mu_over_rho={p: np.stack([t, t]) for p, t in water.items()},
        electron_energies=ee,
        restricted_stopping=np.stack([stopping, stopping / spr]),
        radiative_stopping=np.stack([0.01 * ee, 0.01 * ee]),
        moller=np.stack([0.1 / ee, 0.1 / ee]),
        csda_range=np.stack([0.5 * ee, 0.5 * ee]),
        scattering_power=np.stack([1.0 / ee, 1.0 / ee]),
        delta_cut=DELTA_CUT,
        materials=("water", "air"),
        provenance="synthetic dose-to-water instrument",
    )


def test_water_medium_weight_is_exactly_one() -> None:
    """SPR of water against itself is the ratio of identical numbers: exactly 1."""
    from pyRadMC.scoring.dose_to_water import water_spr

    xs = AnalyticCrossSections()
    for e in (0.05, 0.3, 1.0, 6.0):
        assert water_spr(e, WATER, cross_sections=xs, ecut=ECUT_MEV) == 1.0


def test_synthetic_proportional_table_gives_the_exact_constant() -> None:
    from pyRadMC.scoring.dose_to_water import water_spr

    xs = TabulatedCrossSections(_two_material_tables(spr=1.1))
    for e in (0.25, 1.0, 3.7, 9.0):
        assert water_spr(e, 1, cross_sections=xs, ecut=DELTA_CUT) == pytest.approx(1.1, rel=1e-9)


def test_sub_cutoff_deposits_clamp_to_the_cutoff() -> None:
    """Below ecut the electron spectrum is not tracked; the weight freezes at ecut."""
    from pyRadMC.scoring.dose_to_water import water_spr

    xs = TabulatedCrossSections(_two_material_tables(spr=1.1))
    at_cut = water_spr(DELTA_CUT, 1, cross_sections=xs, ecut=DELTA_CUT)
    assert water_spr(0.01, 1, cross_sections=xs, ecut=DELTA_CUT) == at_cut
    assert water_spr(0.0, 1, cross_sections=xs, ecut=DELTA_CUT) == at_cut


def test_bone_spr_sign_and_magnitude_from_berger_seltzer() -> None:
    """Water/bone restricted collision SPR at MV energies: above 1, near 1.1.

    Cortical bone carries fewer electrons per gram and a higher I-value than
    water, so its mass collision stopping power is lower and the dose-to-water
    conversion must *raise* bone dose by roughly 10 percent (Siebers et al.,
    Phys. Med. Biol. 45 (2000) 983, doi:10.1088/0031-9155/45/4/983).
    """
    from pyRadMC.data.berger_seltzer import restricted_collision_stopping

    water = MATERIALS[WATER]
    bone = MATERIALS[CORTICAL_BONE]
    for e in (0.3, 1.0, 3.0, 6.0):
        ratio = restricted_collision_stopping(e, water, ECUT_MEV) / (
            restricted_collision_stopping(e, bone, ECUT_MEV)
        )
        assert 1.02 < ratio < 1.20, f"water/bone SPR {ratio:.4f} at {e} MeV out of range"
