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
        from pyRadMC.data.analytic import AnalyticCrossSections
        from pyRadMC.data.materials import WATER

        # Densities bracketing what a Phase 0 water-with-density-scaling geometry can
        # contain, including a >1 g/cm^3 voxel (contoured bolus, wet lung, CT noise).
        geometry = ((WATER, 1.2),)
        xs = AnalyticCrossSections(geometry_densities=geometry)

        for energy in np.geomspace(0.05, 20.0, 300):
            mu_max = max(
                density * xs.mu_over_rho_total(float(energy), material)
                for material, density in geometry
            )
            assert xs.majorant(float(energy)) >= mu_max, (
                f"majorant violated at {energy:.4f} MeV: "
                f"{xs.majorant(float(energy)):.6e} < {mu_max:.6e} 1/cm"
            )


class TestUniformContract:
    """``uniform`` returns from the half-open interval [0, 1)."""

    def test_uniform_never_returns_one(self) -> None:
        """An exact 1.0 gives log(1 - u) = 0 and an infinite path length.

        Also divides by zero in the Kahn rejection loop. Draw enough samples that a
        boundary bug in the bit-manipulation of the underlying generator would surface;
        1e7 is cheap and catches the common float32 rounding-to-one case.
        """
        from pyRadMC.rng.host import HostRNG

        rng = HostRNG()
        state = rng.init_state(seed=1234, history_index=0)

        # Bulk draws through the same underlying generator: 1e7 is cheap vectorized.
        # HostRNG.uniform is documented as scalar-equivalent to Generator.random().
        u = state.random(10_000_000)
        assert np.all(u >= 0.0)
        assert np.all(u < 1.0)

        # And through the actual scalar interface the physics routines call.
        state = rng.init_state(seed=1234, history_index=1)
        draws = np.array([rng.uniform(state) for _ in range(100_000)])
        assert np.all(draws >= 0.0)
        assert np.all(draws < 1.0)

    def test_stream_independent_of_history_partitioning(self) -> None:
        """Counter-based seeding: the stream depends on (seed, history_index) only.

        Not on how histories are grouped into batches or assigned to threads. This is
        the property that makes within-target reproducibility survive a change of
        thread count.
        """
        from pyRadMC.rng.host import HostRNG

        rng = HostRNG()
        seed = 42

        # "Partition A": histories created sequentially, each fully drained in order.
        streams_sequential = {}
        for history in range(8):
            state = rng.init_state(seed, history)
            streams_sequential[history] = [rng.uniform(state) for _ in range(16)]

        # "Partition B": histories created in reverse and drawn interleaved, as a
        # different batch decomposition or thread schedule would.
        states = {history: rng.init_state(seed, history) for history in reversed(range(8))}
        streams_interleaved: dict[int, list[float]] = {h: [] for h in range(8)}
        for _ in range(16):
            for history in range(8):
                streams_interleaved[history].append(rng.uniform(states[history]))

        assert streams_sequential == streams_interleaved

        # Distinct histories must not share a stream.
        assert streams_sequential[0] != streams_sequential[1]


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
