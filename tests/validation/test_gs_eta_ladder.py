r"""The Moliere screening parameter, validated against the EEDL elastic shape.

Under the Goudsmit-Saunderson anchoring the elastic cross-section is
back-derived from the scattering power, so the sampled distribution depends on
the single-scattering law **only through its normalized transport-moment ladder**
``r_l = G_l / G_1`` — absolute moments (and hence any absolute error in ``eta``)
cancel. These tests therefore compare ladders: the screened-Rutherford law at
the production Moliere ``eta`` against the tabulated EEDL MF=26/MT=525 angular
distribution, on the same truncated domain (EEDL stops at ``mu = 0.999999``,
so the fair comparison truncates screened Rutherford identically).

Measured 2026-07-23 (the L0 study this test pins): the Moliere ladder tracks
EEDL within ~3-7 percent through ``l = 32`` for H, O and Ca at 0.26 and 10 MeV,
and the transport tail measure ``P(theta > 0.5) / <1 - mu>`` agrees within
1-13 percent — the worst cases sit at high energy, where EEDL's own forward
truncation moves its ladder by tens of percent (quantified below), so the
reference itself is ambiguous there at the +-10 percent level. Re-fitting
``eta`` to the EEDL ladder was measured **not** to help: a second-moment match
degrades the tail by ~20 percent, and a full-ladder fit is ill-conditioned —
the residual is the screened-Rutherford *family*, not the parameter. Since the
whole GS-vs-Gaussian dose effect is ~0.3 percent, the ladder residual bounds
the eta-related dose uncertainty at ~0.03 percent.

Deterministic (tabulated data against a closed-form law, no sampling): the
bands are documentation of measured agreement, not statistical tolerances.
Needs the EEDL library in the default cache; skips otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.goudsmit_saunderson import moliere_screening
from pyradmc.data.tabulated import build
from pyradmc.data.tabulated.endf import read_mf26_angular_distributions

pytestmark = pytest.mark.validation

X_GRID = np.geomspace(1.0e-6, 2.0, 8192)  # x = 1 - mu; the EEDL truncation domain
L_MAX = 256
L_PLATEAU = 32
X_TAIL = 1.0 - np.cos(0.5)  # theta > 0.5 rad

ELEMENTS = (1, 8, 20)
TARGETS_MEV = (0.256, 10.0)

PLATEAU_BAND = (0.88, 1.05)
TAIL_BAND = (0.85, 1.15)


@pytest.fixture(scope="module")
def distributions():
    eedl = build.library_path("eedl", None)
    if not eedl.is_file():
        pytest.skip("EEDL library not cached; run python -m pyradmc.data.tabulated.build")
    text = eedl.read_text(encoding="latin-1")
    return read_mf26_angular_distributions(text, 525)


def _ladder(x: np.ndarray, p: np.ndarray, l_max: int) -> np.ndarray:
    """Normalized transport ladder r_l = G_l/G_1 by trapezoid in x = 1 - mu."""
    mu = 1.0 - x
    norm = np.trapezoid(p, x)
    g = np.empty(l_max + 1)
    g[0] = 0.0
    p_prev = np.ones_like(mu)
    p_curr = mu.copy()
    for ell in range(1, l_max + 1):
        g[ell] = np.trapezoid((1.0 - p_curr) * p, x) / norm
        p_prev, p_curr = p_curr, ((2 * ell + 1) * mu * p_curr - ell * p_prev) / (ell + 1)
    return g[1:] / g[1]


def _tail_per_strength(x: np.ndarray, p: np.ndarray) -> float:
    norm = np.trapezoid(p, x)
    mean_x = np.trapezoid(x * p, x) / norm
    tail = np.trapezoid(np.where(x >= X_TAIL, p, 0.0), x) / norm
    return float(tail / mean_x)


def _eedl_density(mu_tab: np.ndarray, p_tab: np.ndarray) -> np.ndarray:
    x_tab = 1.0 - mu_tab[::-1]
    p_rev = p_tab[::-1]
    keep = (x_tab > 0.0) & (p_rev > 0.0)
    p = np.exp(np.interp(np.log(X_GRID), np.log(x_tab[keep]), np.log(p_rev[keep])))
    p[x_tab[keep].min() > X_GRID] = 0.0
    p[x_tab[keep].max() < X_GRID] = 0.0
    return p


def _sr_density(eta: float) -> np.ndarray:
    return 2.0 * eta * (1.0 + eta) / (X_GRID + 2.0 * eta) ** 2


def _cases(distributions):
    for z in ELEMENTS:
        entries = distributions[z * 100]
        energies = np.array([e for e, _, _ in entries]) * 1e-6
        for target in TARGETS_MEV:
            idx = int(np.argmin(np.abs(np.log(energies / target))))
            energy_ev, mu_tab, p_tab = entries[idx]
            yield z, float(energy_ev) * 1e-6, _eedl_density(mu_tab, p_tab)


def test_moliere_ladder_tracks_the_eedl_ladder(distributions) -> None:
    """r_l agreement through the plateau, at the production eta, per element.

    The plateau (l <= 32) is where the ladder is insensitive to the EEDL
    forward truncation (verified separately below), so this is the clean part
    of the comparison — and the part that dominates the deflection shape at
    the step sizes transport takes.
    """
    for z, e_mev, p_eedl in _cases(distributions):
        eta = moliere_screening(((z, 1.0),), e_mev)
        r_sr = _ladder(X_GRID, _sr_density(eta), L_PLATEAU)
        r_eedl = _ladder(X_GRID, p_eedl, L_PLATEAU)
        ratios = r_sr[1:] / r_eedl[1:]
        assert PLATEAU_BAND[0] <= ratios.min() and ratios.max() <= PLATEAU_BAND[1], (
            f"Z={z} E={e_mev:.3g} MeV: Moliere/EEDL ladder ratios "
            f"[{ratios.min():.3f}, {ratios.max():.3f}] leave {PLATEAU_BAND}"
        )


def test_moliere_tail_matches_the_eedl_tail_per_unit_strength(distributions) -> None:
    """P(theta > 0.5)/<1-mu>: the anchored transport's tail measure.

    This is the quantity that drove the GS adoption (the large-angle tail the
    Gaussian hinge omits); its agreement is what makes the tail *calibrated*
    rather than merely present.
    """
    for z, e_mev, p_eedl in _cases(distributions):
        eta = moliere_screening(((z, 1.0),), e_mev)
        ratio = _tail_per_strength(X_GRID, _sr_density(eta)) / _tail_per_strength(X_GRID, p_eedl)
        assert TAIL_BAND[0] <= ratio <= TAIL_BAND[1], (
            f"Z={z} E={e_mev:.3g} MeV: tail-per-strength ratio {ratio:.3f} leaves {TAIL_BAND}"
        )


def test_the_plateau_truncation_sensitivity_is_bounded(distributions) -> None:
    """Dropping the most forward decade moves the plateau ladder by a known amount.

    Below ~1 MeV the plateau is truncation-stable to under 3 percent, which is
    what makes the low-energy comparison clean. At 10 MeV the drift reaches
    ~12 percent at ``l = 32`` (and ~40 percent at ``l = 256``) — the EEDL
    ladder there is partly shaped by where the file truncates, which is
    exactly why :data:`PLATEAU_BAND` is as wide as it is rather than tighter.
    Both numbers are measured properties of the file, pinned so a data update
    that changes them is noticed.
    """
    keep = X_GRID >= 1.0e-5
    for z, e_mev, p_eedl in _cases(distributions):
        full = _ladder(X_GRID, p_eedl, L_PLATEAU)
        cut = _ladder(X_GRID[keep], p_eedl[keep], L_PLATEAU)
        drift = float(np.abs(cut[1:] / full[1:] - 1.0).max())
        bound = 0.03 if e_mev < 1.0 else 0.15
        assert drift < bound, (
            f"Z={z} E={e_mev:.3g} MeV: plateau ladder truncation drift {drift:.3f} "
            f"exceeds the measured bound {bound}"
        )
