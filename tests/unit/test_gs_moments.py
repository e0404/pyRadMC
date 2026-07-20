r"""Goudsmit-Saunderson multiple-scattering moments: the L0 limit check.

These tests exist before any transport code that uses GS. They establish two
things that everything downstream rests on:

1. The GS Legendre moments ``<P_l> = exp(-Lambda G_l)`` are correct, checked
   against a **brute-force accumulation of individual single scatters** — an
   independent oracle that assumes nothing about the series.
2. The GS second moment **reduces to the Fermi-Eyges hinge's** ``<theta^2> =
   T rho s`` in the small-step limit. This is the anchoring that makes GS a
   drop-in replacement rather than a different model: ``G_1`` is tied to
   :meth:`~pyRadMC.data.interface.CrossSectionSource.scattering_power` by
   construction, so at small steps the two samplers agree in the second moment
   *exactly*, and any transport difference is the large-angle tail alone.

See AGENTS.md section 4 for the oracle ordering; the brute-force comparison here
is a chi-squared-free moment test because the moments have closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SEED


def _sample_screened_rutherford_cos(xi: np.ndarray, eta: float) -> np.ndarray:
    """Single-scatter cosine by exact inverse CDF, vectorized for the oracle.

    Duplicates :func:`pyRadMC.data.goudsmit_saunderson.screened_rutherford_cos_theta`
    on purpose: an oracle that calls the implementation it checks proves nothing.
    """
    inv = xi / (2.0 * eta * (1.0 + eta)) + 1.0 / (2.0 + 2.0 * eta)
    return 1.0 + 2.0 * eta - 1.0 / inv


def _brute_force_moments(
    lam: float, eta: float, l_max: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """``<P_l>`` after a Poisson(lam) number of composed single scatters."""
    from scipy.special import eval_legendre

    uz = _brute_force_cosines(lam, eta, n, rng)
    return np.array([eval_legendre(ell, uz).mean() for ell in range(l_max + 1)])


def _brute_force_cosines(lam: float, eta: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Deflection cosines from explicitly composing Poisson(lam) single scatters.

    The assumption-free oracle for everything in this module: the GS
    distribution *is* the law of this accumulation, so anything derived from the
    Legendre series must reproduce it.
    """
    # Start every history along +z and compose scatters in its own frame.
    ux = np.zeros(n)
    uy = np.zeros(n)
    uz = np.ones(n)

    counts = rng.poisson(lam, size=n)
    for step in range(int(counts.max())):
        active = counts > step
        k = int(active.sum())
        if k == 0:
            break
        cos_t = _sample_screened_rutherford_cos(rng.random(k), eta)
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
        phi = 2.0 * np.pi * rng.random(k)

        ax, ay, az = ux[active], uy[active], uz[active]
        # Rotate (ax, ay, az) by (cos_t, phi); the standard PENELOPE-style form,
        # with the degenerate near-pole case handled separately.
        sz = np.sqrt(np.maximum(0.0, 1.0 - az * az))
        safe = sz > 1.0e-10
        nx = np.where(
            safe,
            ax * cos_t
            + sin_t * (ax * az * np.cos(phi) - ay * np.sin(phi)) / np.where(safe, sz, 1.0),
            sin_t * np.cos(phi),
        )
        ny = np.where(
            safe,
            ay * cos_t
            + sin_t * (ay * az * np.cos(phi) + ax * np.sin(phi)) / np.where(safe, sz, 1.0),
            sin_t * np.sin(phi),
        )
        nz = np.where(safe, az * cos_t - sin_t * sz * np.cos(phi), np.sign(az) * cos_t)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        ux[active], uy[active], uz[active] = nx / norm, ny / norm, nz / norm

    return uz


class TestScreenedRutherfordMoments:
    """The single-scattering moments underneath the GS series."""

    def test_first_transport_moment_matches_the_closed_form(self) -> None:
        r"""Quadrature ``G_1`` equals ``2 eta (1+eta) ln((1+eta)/eta) - 2 eta``.

        The closed form is what anchors ``G_1`` to the tabulated scattering
        power; if the quadrature and the closed form disagree, the anchoring is
        silently wrong and every GS step is mis-scaled.
        """
        from pyRadMC.data.goudsmit_saunderson import (
            first_transport_moment,
            screened_rutherford_moments,
        )

        for eta in (1.0e-6, 1.0e-4, 1.0e-2, 0.1):
            closed = first_transport_moment(eta)
            quad = screened_rutherford_moments(eta, l_max=1)[1]
            assert quad == pytest.approx(closed, rel=1.0e-9), (
                f"G_1 quadrature {quad:.12e} vs closed form {closed:.12e} at eta={eta}"
            )

    def test_zeroth_moment_vanishes_identically(self) -> None:
        """``G_0 = 0`` by definition, since ``P_0 = 1``; a nonzero value would
        damp the whole GS series and destroy normalization."""
        from pyRadMC.data.goudsmit_saunderson import screened_rutherford_moments

        assert screened_rutherford_moments(1.0e-4, l_max=6)[0] == pytest.approx(0.0, abs=1.0e-12)


class TestGoudsmitSaundersonSeries:
    """``<P_l> = exp(-Lambda G_l)`` — the defining GS result."""

    @pytest.mark.parametrize("lam", [0.5, 3.0, 20.0])
    def test_moments_match_brute_force_single_scattering(self, lam: float) -> None:
        """GS moments reproduce explicit accumulation of individual scatters.

        The independent oracle: draw ``N ~ Poisson(Lambda)`` single screened-
        Rutherford scatters, compose them, and measure ``<P_l>`` directly. This
        assumes nothing about the Legendre series, so it catches a wrong
        normalization, a wrong screening convention, or an off-by-one in ``l``.
        """
        from pyRadMC.data.goudsmit_saunderson import screened_rutherford_moments

        eta = 1.0e-4
        l_max = 4
        n = 200_000
        rng = np.random.default_rng(SEED)

        moments = screened_rutherford_moments(eta, l_max=l_max)
        predicted = np.exp(-lam * moments)
        measured = _brute_force_moments(lam, eta, l_max, n, rng)

        # <P_l> is bounded in [-1, 1], so the standard error of a mean over n
        # histories is at most 1/sqrt(n). k = 4 at a fixed seed (AGENTS 2.4).
        sem = 1.0 / np.sqrt(n)
        for ell in range(1, l_max + 1):
            assert abs(measured[ell] - predicted[ell]) < 4.0 * sem, (
                f"<P_{ell}> at Lambda={lam}: brute force {measured[ell]:.5f} vs "
                f"GS exp(-Lambda G_{ell}) = {predicted[ell]:.5f} (4 sem = {4 * sem:.5f})"
            )


class TestGoudsmitSaundersonSampling:
    """The sampled angular distribution, not just its moments."""

    def test_cumulative_is_normalized_and_monotone(self) -> None:
        """A CDF that is not monotone on ``[0, 1]`` cannot be inverted safely.

        The Legendre series is truncated, so Gibbs ringing near the forward peak
        is the expected failure mode; this pins that the construction controls it.
        """
        from pyRadMC.data.goudsmit_saunderson import gs_cumulative, screened_rutherford_moments

        eta = 1.0e-4
        moments = screened_rutherford_moments(eta, l_max=256)
        mu = np.linspace(-1.0, 1.0, 4001)

        for lam in (50.0, 500.0, 5000.0):
            cdf = gs_cumulative(lam, moments, mu)
            assert cdf[0] == pytest.approx(0.0, abs=1.0e-9), f"CDF(-1) != 0 at Lambda={lam}"
            assert cdf[-1] == pytest.approx(1.0, abs=1.0e-6), f"CDF(+1) != 1 at Lambda={lam}"
            assert np.all(np.diff(cdf) >= -1.0e-12), f"CDF is not monotone at Lambda={lam}"

    @pytest.mark.parametrize(
        ("lam", "n"),
        [
            (50.0, 50_000),
            # Composing ~500 scatters per history is minutes-scale work, so the
            # long-step confirmation runs in the validation tier rather than
            # slowing the every-save loop. It is the more important of the two:
            # long steps are the regime GS exists to unlock.
            pytest.param(500.0, 200_000, marks=pytest.mark.validation),
        ],
    )
    def test_sampled_distribution_matches_brute_force_single_scattering(
        self, lam: float, n: int
    ) -> None:
        """Chi-squared: the series sampler reproduces explicit scatter composition.

        The full-distribution counterpart to the moment test — moments agreeing
        does not imply the shape does, and it is the *shape* (specifically the
        large-angle tail the Gaussian hinge omits) that GS exists to get right.
        Binned in ``1 - cos theta`` on a log grid, which is where the structure
        lives; chi-squared is the AGENTS section 4 detection oracle.
        """
        from scipy import stats

        from pyRadMC.data.goudsmit_saunderson import gs_inverse_cdf, screened_rutherford_moments

        eta = 1.0e-4
        rng = np.random.default_rng(SEED)

        moments = screened_rutherford_moments(eta, l_max=512)
        table = gs_inverse_cdf(lam, moments, n_nodes=4096)
        sampled = np.interp(rng.random(n), np.linspace(0.0, 1.0, table.size), table)

        reference = _brute_force_cosines(lam, eta, n, rng)

        # Log bins in 1 - mu, spanning the sampled range; equal counts expected
        # under the null since both samples have the same size.
        lo = max(1.0e-6, min(float((1.0 - sampled).min()), float((1.0 - reference).min())))
        edges = np.geomspace(lo, 2.0, 25)
        obs, _ = np.histogram(1.0 - sampled, bins=edges)
        exp, _ = np.histogram(1.0 - reference, bins=edges)

        keep = (obs + exp) > 20  # pooled-count floor for the chi-squared approximation
        chi2 = float(np.sum((obs[keep] - exp[keep]) ** 2 / (obs[keep] + exp[keep])))
        dof = int(keep.sum()) - 1
        p = float(stats.chi2.sf(chi2, dof))
        assert p > 0.001, (
            f"series sampler vs brute-force composition at Lambda={lam}: "
            f"chi2={chi2:.1f} dof={dof} p={p:.2e}"
        )


class TestFermiEygesLimit:
    """L0: GS reduces to the Gaussian hinge's second moment at small steps."""

    def test_second_moment_reduces_to_scattering_power_times_path(self) -> None:
        r"""``<theta^2> -> T rho s`` as ``Lambda -> 0``.

        The GS mean-square angle is ``2(1 - exp(-Lambda G_1))``, which tends to
        ``2 Lambda G_1``. With the anchoring ``2 (N/M) sigma_el G_1 = T`` and
        ``Lambda = (N/M) sigma_el rho s``, that is exactly ``T rho s`` — the
        quantity :func:`pyRadMC.physics.msc.sample_hinge_cos_theta` is handed
        today. This is the identity that makes the L1 transport limit check
        meaningful: at small steps the two models share a second moment, so any
        dose difference is the large-angle tail, not a rescaling.
        """
        from pyRadMC.data.goudsmit_saunderson import (
            first_transport_moment,
            mean_square_angle,
        )

        eta = 1.0e-4
        g1 = first_transport_moment(eta)

        for fermi_eyges in (1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2):
            # The transport loop's <theta^2> = T rho s fixes Lambda G_1 = <theta^2>/2.
            lam = 0.5 * fermi_eyges / g1
            gs = mean_square_angle(lam, g1)

            # The departure is not a tolerance to be chosen: with x = <theta^2>/2,
            # the ratio is exactly (1 - e^-x)/x = 1 - x/2 + O(x^2). Asserting the
            # closed form pins both the limit *and* the leading saturation term,
            # which a loose tolerance would let drift unnoticed.
            x = 0.5 * fermi_eyges
            expected_ratio = -np.expm1(-x) / x
            assert gs / fermi_eyges == pytest.approx(expected_ratio, rel=1.0e-12), (
                f"GS <theta^2> saturation departs from (1-e^-x)/x at <theta^2>={fermi_eyges}"
            )
            # And the leading term is the Fermi-Eyges value itself.
            assert gs == pytest.approx(fermi_eyges, rel=fermi_eyges), (
                f"GS <theta^2> is not Fermi-Eyges to O(<theta^2>) at {fermi_eyges}"
            )

    def test_departure_from_the_gaussian_hinge_grows_with_step(self) -> None:
        """Quantify the *legitimate* GS-vs-hinge difference as a function of step.

        Not a pass/fail physics gate: a recorded expectation. The two models
        must agree at small steps and must *disagree* at large ones (GS keeps
        the large-angle tail the Gaussian hinge omits). Pinning the trend here
        stops a later L1 transport difference from being read as a bug when it
        is the improvement being bought.
        """
        from pyRadMC.data.goudsmit_saunderson import first_transport_moment, mean_square_angle

        g1 = first_transport_moment(1.0e-4)
        ratios = []
        for fermi_eyges in (1.0e-4, 1.0e-2, 0.1, 1.0):
            lam = 0.5 * fermi_eyges / g1
            ratios.append(mean_square_angle(lam, g1) / fermi_eyges)

        assert ratios[0] == pytest.approx(1.0, rel=1.0e-4), "must agree at small steps"
        assert ratios == sorted(ratios, reverse=True), "saturation must be monotone in step"
        assert ratios[-1] < 0.95, (
            "must depart once the step is long (GS saturates, Gaussian does not)"
        )
