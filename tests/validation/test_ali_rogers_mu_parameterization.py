"""The Ali-Rogers mu(E) parameterizations against EPICS 2023, the independent oracle.

`pyradmc.geometry.spectrum` carries closed-form fits to the tungsten and aluminium
mass attenuation coefficients (Ali and Rogers 2012, table 3) rather than going through
`CrossSectionSource`. That is deliberate and allowed — a spectrum is the *source's own*
data, not a material property, and the fits are what define the published C1/C2
coefficients, so replacing them with library values would change the meaning of the
fitted parameters. But it leaves the fits themselves unvalidated by the fast tiers,
which is how an unfiltered 511 keV annihilation line survived review once already.

This tier closes that hole: EPDL 2023 is a genuinely independent source for the same
quantity, so a transcription error, a wrong coefficient sign, or a units slip in either
parameterization shows up here as a percent-level divergence.

What this test resolves, and what it does not. The paper states a typical local error
near 0.5 percent against NIST. Measured against EPDL 2023 the fits agree to better than
0.9 percent everywhere on the grid below except tungsten at 1 MeV, which is off by
2.5 percent — a real fit deviation, not a resampling artifact, since EPDL carries a
native grid point at exactly 1 MeV. That outlier sets the floor: the gate cannot go
below about 3 percent without failing on the fit's own honest error, so it is set at 4.

The consequence is that this tier catches *structural* corruption — a sign flip on a
coefficient (measured: 131 percent), a units slip, the wrong channel sum, the
below-K-edge branch — and does not resolve individual digits. A one-digit typo in the
mu_W denominator was measured at 2.3 percent, i.e. invisible here. Digit-level fidelity
to Ali and Rogers table 3 is a proofreading obligation on the source, not something any
comparison against a different library can certify.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.tabulated import build
from pyradmc.data.tabulated.epdl import element_photon_channels, material_mu_over_rho
from pyradmc.geometry.spectrum import (
    _E_ANNIHILATION_MEV,
    _mu_over_rho_aluminium,
    _mu_over_rho_tungsten,
)

pytestmark = [pytest.mark.validation]

TOLERANCE = 4e-2
"""Agreement gate; see the module docstring for how it was derived."""

# Above the tungsten K-edge (69.5 keV) and inside both parameterizations' stated range.
# The edge itself is excluded deliberately: mu_W jumps by a factor of ~4.6 across it, the
# fit represents the above-edge branch, and a log-log resample of the EPDL grid at exactly
# the edge energy lands on whichever side the interpolation happens to pick. Spectra built
# at the _MU_W_FLOOR_MEV floor sit on that knife edge; the shipped e_min default of
# 0.15 MeV is far above it.
GRID_MEV = np.array([0.1, 0.2, 0.3, _E_ANNIHILATION_MEV, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0])


@pytest.fixture(scope="module")
def epdl_mu_over_rho():
    """``{Z: mu/rho on GRID_MEV}`` for tungsten and aluminium from EPDL 2023, or skip.

    Summed over *all* photon channels, coherent scattering included. That is not a free
    choice: Ali and Rogers fit the NIST narrow-beam mass attenuation coefficient, which
    carries coherent scattering. Dropping it moves tungsten by +6.8 percent at 511 keV —
    five times the tolerance here — so the coherent-free total is measurably the wrong
    comparand. (The same one-consistent-decomposition rule as AGENTS.md section 8's
    constraint on the analytic pair channel.)
    """
    epdl = build.library_path("epdl", None)
    if not epdl.is_file():
        pytest.skip("EPICS libraries not cached; run python -m pyradmc.data.tabulated.build")
    elements = element_photon_channels(epdl.read_text(encoding="latin-1"), elements=(13, 74))
    return {z: sum(material_mu_over_rho(elements, {z: 1.0}, GRID_MEV).values()) for z in (13, 74)}


@pytest.mark.parametrize(
    ("z", "parameterization", "name"),
    [
        (74, _mu_over_rho_tungsten, "tungsten"),
        (13, _mu_over_rho_aluminium, "aluminium"),
    ],
)
def test_ali_rogers_mu_matches_epdl(epdl_mu_over_rho, z, parameterization, name) -> None:
    """The closed-form fit tracks the library across the MV range, both filtration media."""
    reference = epdl_mu_over_rho[z]
    fitted = parameterization(GRID_MEV)
    deviation = np.abs(fitted / reference - 1.0)
    worst = int(np.argmax(deviation))
    assert deviation[worst] < TOLERANCE, (
        f"{name} mu/rho at {GRID_MEV[worst]} MeV: fit {fitted[worst]:.5f} vs "
        f"EPDL {reference[worst]:.5f} cm^2/g ({deviation[worst]:+.2%})"
    )


def test_annihilation_line_transmission_matches_epdl() -> None:
    """The 511 keV transmission the spectra apply to the C4 line, from library mu instead.

    This is the number the Ali-Rogers fix turns on, so it gets its own assertion rather
    than riding on the grid sweep above: the filtration envelope evaluated at 511 keV
    from EPDL mu must reproduce the one the shipped parameterizations give. Uses the
    Siemens 6 MV coefficients, the mildest filtration of the nine presets and therefore
    the least forgiving relative comparison.
    """
    epdl = build.library_path("epdl", None)
    if not epdl.is_file():
        pytest.skip("EPICS libraries not cached; run python -m pyradmc.data.tabulated.build")
    elements = element_photon_channels(epdl.read_text(encoding="latin-1"), elements=(13, 74))
    line = np.array([_E_ANNIHILATION_MEV])
    mu = {
        z: float(sum(material_mu_over_rho(elements, {z: 1.0}, line).values())[0]) for z in (13, 74)
    }

    c1_sq, c2_sq = 1.184**2, 4.840**2  # siemens-6mv
    from_library = float(np.exp(-mu[74] * c1_sq - mu[13] * c2_sq))
    from_fit = float(
        np.exp(-_mu_over_rho_tungsten(line)[0] * c1_sq - _mu_over_rho_aluminium(line)[0] * c2_sq)
    )
    # The exponent amplifies the mu deviation by C^2, so the gate is on the exponent, not
    # the transmission: a 0.8 percent mu error over 23 g/cm^2 of aluminium is ~16 percent
    # in the ratio while being entirely benign in mu itself.
    assert np.log(from_fit) == pytest.approx(np.log(from_library), rel=0.1)
    assert 0.05 < from_fit < 0.2, f"siemens-6mv 511 keV transmission {from_fit:.4f} off-scale"
