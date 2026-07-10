"""Contract tests for :mod:`pyRadMC.data.interface` and :mod:`pyRadMC.rng.interface`.

These tests encode invariants that, if violated, produce silently wrong dose rather than
a crash. They are the highest-value tests in the repository per line, and they must run
against *every* implementation of each interface.

Written before any implementation exists. Each is skipped until its backend lands.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestMajorantContract:
    """The Woodcock majorant must bound every real macroscopic cross-section."""

    def test_majorant_is_never_exceeded(self) -> None:
        """No material-density combination in the geometry may exceed the majorant.

        If it does, delta scattering acquires a negative probability, histories are
        under-attenuated in the densest medium, and the resulting bias is smooth: it
        will not show up as a crash, an assertion, or a visibly wrong depth-dose curve.
        It will show up as a two percent error in bone that nobody finds for a year.
        """
        pytest.skip("Phase 0: analytic backend not implemented")

        from pyRadMC.data.analytic import AnalyticCrossSections  # noqa: F401

        # for each energy on the transport grid:
        #     mu_max = max over (material, density) of rho * mu_over_rho_total
        #     assert xs.majorant(energy) >= mu_max


class TestUniformContract:
    """``uniform`` returns from the half-open interval [0, 1)."""

    def test_uniform_never_returns_one(self) -> None:
        """An exact 1.0 gives log(1 - u) = 0 and an infinite path length.

        Also divides by zero in the Kahn rejection loop. Draw enough samples that a
        boundary bug in the bit-manipulation of the underlying generator would surface;
        1e7 is cheap and catches the common float32 rounding-to-one case.
        """
        pytest.skip("Phase 0: RNG backends not implemented")

        # for each RNG implementation:
        #     u = draw 1e7 samples
        #     assert np.all(u >= 0.0) and np.all(u < 1.0)

    def test_stream_independent_of_history_partitioning(self) -> None:
        """Counter-based seeding: the stream depends on (seed, history_index) only.

        Not on how histories are grouped into batches or assigned to threads. This is
        the property that makes within-target reproducibility survive a change of
        thread count.
        """
        pytest.skip("Phase 0: RNG backends not implemented")


class TestTableIntegrationContract:
    """Sub-grid integration of products, not products of bin means.

    See AGENTS.md section 2.7. This test may not be weakened.
    """

    def test_product_integration_beats_bin_center_evaluation(self) -> None:
        """Construct a case where the two disagree, and pin the correct answer.

        Take a stopping power S(E) and a looked-up quantity Q(E) that are *correlated*
        within a bin: for instance both monotonically decreasing. Then

            <S * Q>  !=  <S> * <Q>

        and the difference is the intra-bin covariance. Evaluating separately averaged
        bin quantities at bin centers discards it. On a coarse energy grid this is a
        percent-level error in the stopping-power-weighted integral, and it is
        systematic, so it does not average away over histories.

        The reference value is computed by dense-grid quadrature of the product.
        """
        pytest.skip("Phase 5: tabulated backend not implemented")

        # S = lambda E: ...   # decreasing
        # Q = lambda E: ...   # decreasing, correlated with S within each coarse bin
        # exact = quad(lambda E: S(E) * Q(E), lo, hi)          # dense grid
        # naive = mean(S over bin) * mean(Q over bin) * (hi - lo)
        # actual = tabulated_backend.integrate_product(...)
        # assert abs(actual - exact) < abs(naive - exact) / 10


def test_no_cross_target_bit_equality_is_asserted_anywhere() -> None:
    """A meta-test, and a deliberate one.

    Greps the test tree for exact-equality assertions between backends. Cross-target
    bit-reproducibility is unattainable (non-associative atomics, differing
    transcendental intrinsics). A test asserting it will pass on the developer's
    machine and fail in CI, and the usual response is to loosen it until it passes
    vacuously. Better to forbid the shape outright.

    See AGENTS.md section 2.3.
    """
    pytest.skip("implement once more than one backend exists")

    # walk tests/, parse with ast, flag assert_array_equal / == between
    # results tagged as coming from different backends
    _ = np  # keep the import meaningful when the body lands
