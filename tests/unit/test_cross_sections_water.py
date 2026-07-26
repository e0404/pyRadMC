"""The first red test.

Nothing in ``pyRadMC.data.analytic`` exists yet. That is the point: this test defines
what must exist and what it must produce, before a line of it is written.

Reference values are NIST XCOM total mass attenuation coefficients for liquid water,
with coherent scattering included. Verify against
https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html before trusting the numbers
below; they are transcribed here to two significant figures of tolerance, not as
authoritative data.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "pyRadMC.data.analytic",
    reason="analytic cross-section backend not implemented yet",
)

# Energy (MeV) -> mu/rho (cm^2/g), liquid water, NIST XCOM, with coherent scattering.
NIST_WATER_MU_OVER_RHO: dict[float, float] = {
    1.0: 0.0707,
    2.0: 0.0493,
    6.0: 0.0277,
    10.0: 0.0222,
    15.0: 0.0194,
}

# The analytic backend is a parameterization, not a table. It is not expected to
# reproduce XCOM to better than a few percent. This tolerance is a *statement of what
# the analytic backend is for*: standing up the transport loop. The
# sub-percent reproduction landed with the tabulated backend, in
# tests/validation/test_epdl_water_nist.py (EPDL-derived water vs NIST XCOM, gate 1%,
# measured ~0.3%); this analytic gate stays at 5% because loosening the oracle to the
# parameterization is exactly what AGENTS.md 2.2 forbids.
ANALYTIC_TOLERANCE_RELATIVE = 0.05


@pytest.mark.parametrize(("energy", "expected"), sorted(NIST_WATER_MU_OVER_RHO.items()))
def test_total_attenuation_water_matches_nist(energy: float, expected: float) -> None:
    """The analytic backend reproduces NIST mu/rho for water within 5 percent."""
    from pyRadMC.data.analytic import AnalyticCrossSections
    from pyRadMC.data.materials import WATER

    xs = AnalyticCrossSections()
    actual = xs.mu_over_rho_total(energy, WATER)

    relative_error = abs(actual - expected) / expected
    assert relative_error < ANALYTIC_TOLERANCE_RELATIVE, (
        f"mu/rho(water, {energy} MeV) = {actual:.5f} cm^2/g, "
        f"NIST = {expected:.5f}, relative error {relative_error:.1%}"
    )


@pytest.mark.parametrize("energy", sorted(NIST_WATER_MU_OVER_RHO))
def test_total_equals_sum_of_processes(energy: float) -> None:
    """The total is the sum over enabled channels.

    A contract test on :class:`CrossSectionSource`. The redundancy between
    ``mu_over_rho_total`` and the per-process accessors is deliberate: a mismatch here
    means the Woodcock majorant will be computed from a different cross-section than
    the one used to select the interaction, which biases the transport in a way that is
    invisible in a depth-dose curve.
    """
    from pyRadMC.data.analytic import AnalyticCrossSections
    from pyRadMC.data.interface import PhotonProcess
    from pyRadMC.data.materials import WATER

    xs = AnalyticCrossSections()
    processes = [
        PhotonProcess.COMPTON,
        PhotonProcess.PHOTOELECTRIC,
        PhotonProcess.PAIR,
    ]
    total = xs.mu_over_rho_total(energy, WATER)
    summed = sum(xs.mu_over_rho(energy, WATER, p) for p in processes)

    assert total == pytest.approx(summed, rel=1e-12), (
        "total attenuation is not the sum of its enabled channels"
    )
