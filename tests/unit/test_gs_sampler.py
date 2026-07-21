r"""The Goudsmit-Saunderson sampler as the transport loop reaches it.

:mod:`tests.unit.test_gs_moments` pins the distribution's *construction*; this
module pins the *sampler* — the binned table lookup, the scaled-angle rescaling,
and the properties the transport-level (L1) comparison depends on:

- the first moment survives table binning exactly, so an L1 dose difference
  cannot be a mis-scaled deflection;
- the binned table still reproduces the exactly-built distribution, which is what
  justifies the chosen bin density rather than an eyeballed one;
- and the recorded, *expected* departure from the Gaussian hinge at small steps,
  which is the large-angle tail GS exists to add.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SEED


def _draw(source: object, theta2: float, energy: float, material: int, n: int) -> np.ndarray:
    """``n`` deflection cosines through the public source-level sampler."""
    rng = np.random.default_rng(SEED)
    return np.array(
        [source.sample_gs_cos_theta(theta2, energy, material, rng) for _ in range(n)]  # type: ignore[attr-defined]
    )


class TestFirstMomentAnchoring:
    """The property that makes a binned table safe."""

    @pytest.mark.parametrize("theta2", [1.0e-6, 1.0e-4, 1.0e-2, 0.1])
    def test_sampled_first_moment_matches_the_request_exactly(self, theta2: float) -> None:
        r"""``<1 - cos theta>`` of the sample equals ``1 - e^{-<theta^2>/2}``.

        The anchoring is on the *first* moment because that is what
        Goudsmit-Saunderson pins exactly (``<cos> = e^{-Lambda G_1}``) and what
        the scattering power encodes; ``<theta^2>`` is tail-dominated for a
        heavy-tailed law. This must hold *despite* the lookup being binned in
        both ``eta`` and ``<theta^2>``, because the table is normalized in a
        scaled deflection and rescaled by the caller's exact value. If it ever
        fails, the L1 transport comparison stops measuring the angular model and
        starts measuring table resolution.
        """
        from pyRadMC.data.analytic import AnalyticCrossSections
        from pyRadMC.data.materials import WATER

        source = AnalyticCrossSections()
        n = 100_000
        one_minus_cos = 1.0 - _draw(source, theta2, 1.0, WATER, n)
        measured = float(np.mean(one_minus_cos))
        expected = -np.expm1(-0.5 * theta2)

        sem = float(np.std(one_minus_cos)) / np.sqrt(n)
        assert abs(measured - expected) < 4.0 * sem + 1.0e-3 * expected, (
            f"requested <1-cos>={expected:.4e}, sampled {measured:.4e} (4 sem={4 * sem:.2e})"
        )

    def test_deflection_is_forward_for_a_vanishing_step(self) -> None:
        """A zero-length step cannot deflect; the sampler must not consume the
        table (or a uniform) to say so."""
        from pyRadMC.data.analytic import AnalyticCrossSections
        from pyRadMC.data.materials import WATER

        rng = np.random.default_rng(SEED)
        assert AnalyticCrossSections().sample_gs_cos_theta(0.0, 1.0, WATER, rng) == 1.0

    def test_degenerate_bin_yields_a_forward_table_not_nan(self) -> None:
        r"""A near-zero-``Lambda`` bin must build a finite, forward table.

        At the floored key, ``Lambda ~ 1e-9``: every Poisson draw in the
        Monte-Carlo build is zero scatters, every cosine is exactly one, and
        the unit-mean normalization divides zero by zero. The resulting NaN row
        does not even crash downstream — ``max(-1.0, nan)`` is ``-1.0`` under
        both Python and CUDA ``fmaxf`` semantics — it silently turns "no
        deflection" into "certain full backscatter" for every draw from the
        bin. Reachable through the range-out branch: a Moller delta born just
        above ECUT takes one tiny substep, and the hinge fires on it.

        The correct degenerate limit is w = 0 ("always forward"); the residual
        bias against the true distribution is the anchor itself,
        ``1 - e^{-theta2/2} <= ~1e-12``, far below every oracle in the suite.
        """
        from pyRadMC.data.analytic import AnalyticCrossSections
        from pyRadMC.data.goudsmit_saunderson import gs_scaled_deflection_table
        from pyRadMC.data.materials import WATER

        source = AnalyticCrossSections()
        eta = source.elastic_screening(1.0, WATER)

        table = gs_scaled_deflection_table(eta, 1.0e-12, n_u=512)
        assert np.isfinite(table).all(), "degenerate bin built a non-finite table"
        assert np.all(table == 0.0), "no build sample scattered, so w must be identically zero"

        # And through the public path the transport calls, including the
        # sub-floor keying: the deflection must be exactly forward.
        rng = np.random.default_rng(SEED)
        for _ in range(32):
            assert source.sample_gs_cos_theta(1.0e-13, 1.0, WATER, rng) == 1.0


class TestBinningFidelity:
    """The binned lookup must reproduce an exactly-built table."""

    @pytest.mark.validation
    def test_binned_table_matches_the_exact_distribution(self) -> None:
        """Chi-squared: the grid lookup vs a table built at the exact parameters.

        Runs at high build statistics, in the validation tier, because the
        residual this measures is **Monte-Carlo noise in the table build**, not a
        modelling error, and it converges away. Measured chi-squared at
        ``<theta^2> = 0.02 / 0.05`` against build samples: 73.6 / 137.3 at 2^19,
        51.3 / 71.5 at 2^21, 25.8 / 36.2 at 2^23 — the ``1/sqrt(N)`` fall of a
        noise term.

        That convergence is the load-bearing result for the whole table design.
        The two error sources are separable and neither needs a finer grid:
        *interpolation* error is removed by the bilinear lookup (which is why
        ``<theta^2>`` values landing between grid nodes already pass at the
        default statistics), and *build noise* costs only **linearly** in
        samples, where refining the two-dimensional grid would cost
        quadratically — 16 bins per log ran to ~10k tables and ~20 MB, against
        ~1.1 MB for the full grid at the shipped 4 per log.
        """
        from scipy import stats

        import pyRadMC.data.goudsmit_saunderson as gs
        from pyRadMC.data.analytic import AnalyticCrossSections
        from pyRadMC.data.goudsmit_saunderson import gs_scaled_deflection_table
        from pyRadMC.data.materials import WATER

        # Build statistics, not a physics parameter: see the docstring. The
        # default is set for interactive use of the reference backend; tables
        # shipped to a device backend are precomputed once, so they are built
        # here at the statistics the convergence above shows are needed.
        original = gs._RAW_SAMPLES
        gs._RAW_SAMPLES = 1 << 23
        gs._moment_cache.clear()

        source = AnalyticCrossSections()
        energy, theta2, n = 1.0, 0.05, 100_000

        binned = _draw(source, theta2, energy, WATER, n)

        # The same distribution, built at the exact (eta, theta2) with no binning.
        eta = source.elastic_screening(energy, WATER)
        exact_table = gs_scaled_deflection_table(eta, theta2, n_u=512)
        rng = np.random.default_rng(SEED + 1)
        w = exact_table[np.minimum((rng.random(n) * 512).astype(int), 511)]
        exact = np.maximum(-1.0, 1.0 - w * -np.expm1(-0.5 * theta2))

        edges = np.quantile(1.0 - exact, np.linspace(0.0, 1.0, 21))
        edges[0], edges[-1] = 0.0, 2.0
        obs, _ = np.histogram(1.0 - binned, bins=edges)
        exp, _ = np.histogram(1.0 - exact, bins=edges)
        keep = (obs + exp) > 20
        chi2 = float(np.sum((obs[keep] - exp[keep]) ** 2 / (obs[keep] + exp[keep])))
        p = float(stats.chi2.sf(chi2, int(keep.sum()) - 1))
        gs._RAW_SAMPLES = original
        assert p > 0.001, f"binned vs exact GS table: chi2={chi2:.1f} p={p:.2e}"


class TestDepartureFromTheGaussianHinge:
    """What L1 is expected to measure, quantified before transport runs."""

    def test_gs_adds_a_large_angle_tail_at_matched_second_moment(self) -> None:
        r"""At identical ``<theta^2>``, GS puts more probability at wide angles.

        The two samplers agree in the first moment *by construction*, so this
        records the difference that remains — the physical content of the change.
        It is deliberately an inequality, not a tolerance: the tail is the
        improvement being bought, and pinning its direction stops a later
        transport difference from being misread as a defect.
        """
        from pyRadMC.data.analytic import AnalyticCrossSections
        from pyRadMC.data.materials import WATER
        from pyRadMC.physics.msc import sample_hinge_cos_theta

        source = AnalyticCrossSections()
        theta2, n = 0.05, 200_000

        gs = _draw(source, theta2, 1.0, WATER, n)
        rng = np.random.default_rng(SEED)
        hinge = np.array([sample_hinge_cos_theta(theta2, rng) for _ in range(n)])

        # Wide angle: beyond three times the rms deflection.
        wide = np.cos(3.0 * np.sqrt(theta2))
        gs_tail = float(np.mean(gs < wide))
        hinge_tail = float(np.mean(hinge < wide))
        assert gs_tail > hinge_tail, (
            f"GS wide-angle fraction {gs_tail:.5f} is not above the Gaussian hinge's "
            f"{hinge_tail:.5f} at matched <theta^2>={theta2}"
        )

    @pytest.mark.parametrize("theta2", [1.0e-4, 1.0e-3, 1.0e-2, 0.1])
    def test_first_moments_agree_across_the_two_samplers(self, theta2: float) -> None:
        """Both samplers carry the same ``<1 - cos>``, so L1 isolates the shape.

        Runs across the step sizes the engine actually takes; the agreement is
        what makes "the schedule is unchanged, only the angular model moved" a
        true statement at L1.
        """
        from pyRadMC.data.analytic import AnalyticCrossSections
        from pyRadMC.data.materials import WATER
        from pyRadMC.physics.msc import sample_hinge_cos_theta

        source = AnalyticCrossSections()
        n = 50_000

        gs = 1.0 - _draw(source, theta2, 1.0, WATER, n)
        rng = np.random.default_rng(SEED)
        hinge = 1.0 - np.array([sample_hinge_cos_theta(theta2, rng) for _ in range(n)])

        a, b = float(np.mean(gs)), float(np.mean(hinge))
        sem = np.hypot(np.std(gs), np.std(hinge)) / np.sqrt(n)
        assert abs(a - b) < 4.0 * sem, (
            f"<1-cos> differs between samplers at {theta2}: GS {a:.5e} vs hinge {b:.5e}"
        )
