"""Contract tests for :mod:`pyRadMC.data.interface` and :mod:`pyRadMC.rng.interface`.

These tests encode invariants that, if violated, produce silently wrong dose rather than
a crash. They are the highest-value tests in the repository per line, and they must run
against *every* implementation of each interface.

Written before any implementation exists. Each is skipped until its backend lands.
"""

from __future__ import annotations

import ast

import numpy as np


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

        # Densities bracketing what a water-with-density-scaling geometry can
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
        from pyRadMC.data.interface import PhotonProcess
        from pyRadMC.data.tabulated.model import TabulatedData
        from pyRadMC.data.tabulated.source import TabulatedCrossSections

        # Two correlated-decreasing quantities tabulated on a shared grid.
        e_grid = np.geomspace(0.1, 10.0, 40)
        s = 1.0 / e_grid  # a stopping-power-like quantity
        q = 1.0 / np.sqrt(e_grid)  # correlated with s within every interval
        filler = np.ones((1, e_grid.size))
        data = TabulatedData(
            photon_energies=e_grid,
            mu_over_rho={p: filler for p in (PhotonProcess.COMPTON, PhotonProcess.PAIR)},
            electron_energies=e_grid,
            restricted_stopping=s[None, :],
            radiative_stopping=filler,
            moller=filler,
            csda_range=filler,
            scattering_power=q[None, :],
            delta_cut=0.2,
            materials=("water",),
            provenance="contract test",
        )
        src = TabulatedCrossSections(data)

        lo, hi = 0.2, 5.0  # a wide interval spanning many grid nodes
        dense = np.geomspace(lo, hi, 20_001)
        s_dense, q_dense = 1.0 / dense, 1.0 / np.sqrt(dense)
        exact = float(np.trapezoid(s_dense * q_dense, dense))
        # Separately averaged bin quantities, then multiplied: discards the covariance.
        mean_s = float(np.trapezoid(s_dense, dense)) / (hi - lo)
        mean_q = float(np.trapezoid(q_dense, dense)) / (hi - lo)
        naive = mean_s * mean_q * (hi - lo)

        actual = src.integrate_product(
            "restricted_stopping", "scattering_power", 0, lo, hi, n_sub=256
        )
        assert abs(actual - exact) < abs(naive - exact) / 10.0


def _cross_target_equality_violations(tree: ast.Module) -> list[int]:
    """Line numbers of exact-equality assertions between different targets.

    Heuristic by design: within each test function, variables assigned from an
    expression mentioning an engine constructor are tagged with (backend, device
    literal). An equality comparison (``==``, ``np.array_equal``,
    ``assert_array_equal``, ``assert_equal``) whose operands carry different
    backends — or the same backend with different literal devices — is flagged.
    Helper indirection can evade it; the point is to catch the shape a reasonable
    person writes before coffee, not a determined adversary.
    """
    engine_names = {"ReferenceEngine": "ref", "WarpEngine": "warp"}
    equality_calls = {"assert_array_equal", "array_equal", "assert_equal"}

    def tag_of(expr: ast.AST) -> tuple[str, str | None] | None:
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Call):
                name = (
                    sub.func.id if isinstance(sub.func, ast.Name) else getattr(sub.func, "attr", "")
                )
                if name in engine_names:
                    device = None
                    for kw in sub.keywords:
                        if kw.arg == "device" and isinstance(kw.value, ast.Constant):
                            device = str(kw.value.value)
                    return (engine_names[name], device)
        return None

    violations: list[int] = []
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    for func in functions:
        tagged: dict[str, tuple[str, str | None]] = {}
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                tag = tag_of(node.value)
                if isinstance(target, ast.Name) and tag is not None:
                    tagged[target.id] = tag

        def tags_in(
            expr: ast.AST, tagged: dict[str, tuple[str, str | None]] = tagged
        ) -> set[tuple[str, str | None]]:
            # Early-bound default: each function's variable tags, not the last one's.
            found = set()
            direct = tag_of(expr)
            if direct is not None:
                found.add(direct)
            for sub in ast.walk(expr):
                if isinstance(sub, ast.Name) and sub.id in tagged:
                    found.add(tagged[sub.id])
            return found

        def is_cross_target(operands: list[ast.AST], lineno: int, tags_in=tags_in) -> None:
            tags = set().union(*(tags_in(op) for op in operands))
            backends = {backend for backend, _ in tags}
            devices = {device for _, device in tags if device is not None}
            if len(backends) > 1 or (len(devices) > 1):
                violations.append(lineno)

        for node in ast.walk(func):
            if isinstance(node, ast.Compare) and any(isinstance(op, ast.Eq) for op in node.ops):
                is_cross_target([node.left, *node.comparators], node.lineno)
            elif isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else getattr(node.func, "attr", "")
                )
                if name in equality_calls and len(node.args) >= 2:
                    is_cross_target(list(node.args[:2]), node.lineno)
    return violations


def test_no_cross_target_bit_equality_is_asserted_anywhere() -> None:
    """A meta-test, and a deliberate one.

    Walks the test tree's ASTs for exact-equality assertions between backends (or
    between devices of one backend). Cross-target bit-reproducibility is
    unattainable (non-associative atomics, differing transcendental intrinsics). A
    test asserting it will pass on the developer's machine and fail in CI, and the
    usual response is to loosen it until it passes vacuously. Better to forbid the
    shape outright.

    See AGENTS.md section 2.3. Within-target equality (same backend, same device)
    is legitimate and stays unflagged — the Warp backend's determinism tests rely
    on it.
    """
    from pathlib import Path

    # The checker must actually detect the forbidden shapes, or this test is a
    # rubber stamp: prove it on planted violations first.
    planted = ast.parse(
        "def test_bad_backends():\n"
        "    a = ReferenceEngine(grid=g, cross_sections=x, rng=r).run(s, 1, 1, 0)\n"
        "    b = WarpEngine(grid=g, cross_sections=x).run(s, 1, 1, 0)\n"
        "    np.testing.assert_array_equal(a.dose, b.dose)\n"
        "def test_bad_devices():\n"
        "    a = WarpEngine(grid=g, cross_sections=x, device='cpu').run(s, 1, 1, 0)\n"
        "    b = WarpEngine(grid=g, cross_sections=x, device='cuda:0').run(s, 1, 1, 0)\n"
        "    assert (a.dose == b.dose).all()\n"
        "def test_fine_same_device():\n"
        "    a = WarpEngine(grid=g, cross_sections=x, device='cpu').run(s, 1, 1, 0)\n"
        "    b = WarpEngine(grid=g, cross_sections=x, device='cpu').run(s, 1, 1, 0)\n"
        "    np.testing.assert_array_equal(a.dose, b.dose)\n"
    )
    assert len(_cross_target_equality_violations(planted)) == 2

    offenders = []
    for path in sorted(Path(__file__).parent.parent.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders.extend((path.name, line) for line in _cross_target_equality_violations(tree))
    assert offenders == [], (
        f"cross-target exact-equality assertions found: {offenders}. "
        "Compare backends statistically (AGENTS.md 2.3 and 4), never bitwise."
    )
    _ = np  # the oracle-style imports stay meaningful for the module docstring
