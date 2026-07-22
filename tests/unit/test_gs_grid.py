r"""The eager Goudsmit-Saunderson grid: the device backend's table precompute.

The reference backend builds deflection tables lazily, one ``(eta, <theta^2>)``
node at a time, memoized per source (``CrossSectionSource.sample_gs_cos_theta``).
A device backend cannot: a mid-transport host build was measured at 86 s, and a
lazy dict is not safe across the ``devices=[...]`` shard threads. So the Warp
port precomputes the **full rectangle of grid nodes** host-side before the first
launch and uploads it flat.

These tests pin the properties that make that precompute a *re-packaging* of the
reference construction rather than a second implementation:

- every node value is built by the same function, same fixed build seed, as the
  reference's lazy cache — so the two backends draw from bit-identical tables
  (float64, before the device's float32 cast);
- the window covers every ``(eta, <theta^2>)`` key the transport loop can reach,
  which is what justifies the kernel clamping instead of building on miss;
- the grid geometry constants (bins per log, nodes per table, theta2 floor)
  are the *reference sampler's* — one contract, not two.
"""

from __future__ import annotations

import math
import typing

import numpy as np
import pytest

from pyRadMC import ECUT_MEV, PCUT_MEV
from pyRadMC.data.analytic import AnalyticCrossSections
from pyRadMC.data.goudsmit_saunderson import (
    build_gs_grid,
    gs_scaled_deflection_table,
    gs_window_from_tables,
)
from pyRadMC.data.materials import WATER

E_MAX = 21.0

# One-column-pair test window: eta around exp(-11.5) ~ 1e-5, theta2 raised off
# the production floor so the build stays a few dozen cheap Monte-Carlo tables.
LOG_ETA = -11.5
THETA2_MIN = 1.0e-4
THETA2_MAX = 3.0e-3


@pytest.fixture(scope="module")
def small_grid():
    return build_gs_grid(LOG_ETA, LOG_ETA, THETA2_MAX, theta2_min=THETA2_MIN)


@pytest.fixture(scope="module")
def xs() -> AnalyticCrossSections:
    return AnalyticCrossSections()


@pytest.fixture(scope="module")
def tables(xs: AnalyticCrossSections):
    return xs.build_tables(ecut=ECUT_MEV, pcut=PCUT_MEV, e_max=E_MAX, n_points=64)


class TestGridConstruction:
    def test_node_values_match_the_lazy_reference_construction(self, small_grid) -> None:
        """Grid entries equal direct ``gs_scaled_deflection_table`` calls exactly.

        Bit-equality is the right oracle here (AGENTS.md 2.3 forbids it across
        *targets*, not within one host build): the eager builder must call the
        identical constructor at the identical node parameters, so any
        difference — most plausibly a transposed index — is a defect, not
        noise. Checked at an asymmetric pair of nodes so an (i, j) swap cannot
        cancel.
        """
        bins = small_grid.bins_per_log
        nodes = ((small_grid.ix0, small_grid.iy0 + 2), (small_grid.ix0 + 1, small_grid.iy0 + 5))
        for i, j in nodes:
            expected = gs_scaled_deflection_table(
                math.exp(i / bins), math.exp(j / bins), n_u=small_grid.n_u
            )
            got = small_grid.values[i - small_grid.ix0, j - small_grid.iy0]
            np.testing.assert_array_equal(got, expected)

    def test_window_brackets_every_key_in_the_requested_ranges(self, small_grid) -> None:
        """Any (log eta, log theta2) inside the request has all four bracketing nodes.

        The kernel interpolates between ``floor(f)`` and ``floor(f) + 1`` in both
        axes; the builder must therefore extend one node past the floor of each
        upper edge, or the top of the range would read out of bounds (and the
        clamp would silently flatten the interpolation there).
        """
        bins = small_grid.bins_per_log
        rng = np.random.default_rng(0)
        for log_eta in rng.uniform(LOG_ETA, LOG_ETA, 4):
            fx = log_eta * bins
            assert 0 <= math.floor(fx) - small_grid.ix0 <= small_grid.values.shape[0] - 2
        for theta2 in np.exp(rng.uniform(math.log(THETA2_MIN), math.log(THETA2_MAX), 16)):
            fy = math.log(theta2) * bins
            assert 0 <= math.floor(fy) - small_grid.iy0 <= small_grid.values.shape[1] - 2

    def test_rows_are_unit_mean_or_degenerate_zero(self, small_grid) -> None:
        """Each table is the scaled deflection: mean exactly 1, or all-zero.

        Unit mean per row is what makes the bilinear blend anchor-safe on the
        device — the blend of unit-mean rows has unit mean, so the first moment
        stays the caller's exact ``<1 - cos>`` wherever the lookup lands.
        """
        assert np.isfinite(small_grid.values).all()
        means = small_grid.values.mean(axis=2)
        degenerate = np.all(small_grid.values == 0.0, axis=2)
        np.testing.assert_allclose(means[~degenerate], 1.0, rtol=1e-12)

    def test_floor_rows_build_zero_and_finite(self, tmp_path) -> None:
        """At the production theta2 floor the rows are the degenerate forward table.

        Mirrors the reference sampler's NaN-guard pin (test_gs_sampler): near the
        floored key no build sample scatters, and the correct limit is w = 0,
        never NaN — ``max(-1, nan)`` is silent certain backscatter on device.
        (``cache_dir`` keeps a default-constants unit build out of the user's
        real grid cache.)
        """
        grid = build_gs_grid(LOG_ETA, LOG_ETA, 2.0e-12, cache_dir=tmp_path)
        assert np.isfinite(grid.values).all()
        assert np.all(grid.values[:, 0, :] == 0.0)

    def test_parallel_build_is_identical_to_serial(self) -> None:
        """Worker count can never change a value, only the wall clock.

        Each node's table is a pure function of its node parameters and the
        fixed build seed, so the column-parallel build must reproduce the
        serial one bit for bit — this is what makes the thread count a free
        knob rather than a reproducibility hazard (AGENTS.md 2.3: within one
        host build, bit-equality is the right oracle).
        """
        import pyRadMC.data.goudsmit_saunderson as gs

        serial = build_gs_grid(LOG_ETA, LOG_ETA, THETA2_MAX, theta2_min=THETA2_MIN, max_workers=1)
        gs._grid_cache.clear()
        parallel = build_gs_grid(LOG_ETA, LOG_ETA, THETA2_MAX, theta2_min=THETA2_MIN, max_workers=3)
        np.testing.assert_array_equal(serial.values, parallel.values)
        assert (serial.ix0, serial.iy0) == (parallel.ix0, parallel.iy0)

    def test_grid_defaults_match_the_reference_sampler_contract(self, small_grid, tmp_path) -> None:
        """One grid geometry, shared with ``CrossSectionSource.sample_gs_cos_theta``.

        The literals are pinned (not imported) so that a drift in *either* the
        builder's defaults or the reference sampler's constants breaks here: the
        two must move together or the backends stop sampling the same tables.
        """
        from pyRadMC.data.interface import _GS_BINS_PER_LOG, _GS_TABLE_NODES, _GS_THETA2_MIN

        assert small_grid.bins_per_log == _GS_BINS_PER_LOG == 4.0
        assert small_grid.n_u == _GS_TABLE_NODES == 512
        default_floor_grid = build_gs_grid(LOG_ETA, LOG_ETA, 2.0e-12, cache_dir=tmp_path)
        assert default_floor_grid.iy0 == math.floor(math.log(_GS_THETA2_MIN) * 4.0)
        assert _GS_THETA2_MIN == 1.0e-12


class TestDiskCache:
    """The persistent grid cache: load in milliseconds, never rebuild needlessly.

    Node values are pure functions of the node coordinates and the construction
    algorithm — materials, geometry, cuts and sources only choose the *window* —
    so the disk entry is keyed on the construction (an explicit version plus its
    constants and numpy's feature version) and is *extended*, never rebuilt,
    when a request needs a wider window. The tiny windows here sit at the
    production theta2 floor with a microscopic ceiling, so every node is the
    degenerate forward table and builds in milliseconds.
    """

    WINDOW: typing.ClassVar[dict[str, float]] = dict(
        log_eta_min=LOG_ETA, log_eta_max=LOG_ETA, theta2_max=2.0e-12
    )

    @staticmethod
    def _fresh(monkeypatch, tmp_path, **kwargs):
        import pyRadMC.data.goudsmit_saunderson as gs

        gs._grid_cache.clear()
        return build_gs_grid(cache_dir=tmp_path, **kwargs)

    def test_a_second_process_loads_without_building(self, tmp_path, monkeypatch) -> None:
        import pyRadMC.data.goudsmit_saunderson as gs

        first = self._fresh(monkeypatch, tmp_path, **self.WINDOW)

        def refuse(*args, **kwargs):
            raise AssertionError("a cached window must load from disk, not rebuild")

        gs._grid_cache.clear()
        monkeypatch.setattr(gs, "gs_scaled_deflection_table", refuse)
        second = build_gs_grid(cache_dir=tmp_path, **self.WINDOW)
        np.testing.assert_array_equal(first.values, second.values)
        assert (first.ix0, first.iy0) == (second.ix0, second.iy0)

    def test_a_wider_window_extends_incrementally(self, tmp_path, monkeypatch) -> None:
        """Old nodes are reused bit-for-bit; only genuinely new cells build."""
        import pyRadMC.data.goudsmit_saunderson as gs

        first = self._fresh(monkeypatch, tmp_path, **self.WINDOW)

        built = {"n": 0}
        original = gs.gs_scaled_deflection_table

        def counting(*args, **kwargs):
            built["n"] += 1
            return original(*args, **kwargs)

        gs._grid_cache.clear()
        monkeypatch.setattr(gs, "gs_scaled_deflection_table", counting)
        wider = build_gs_grid(
            log_eta_min=LOG_ETA - 0.3, log_eta_max=LOG_ETA, theta2_max=4.0e-12, cache_dir=tmp_path
        )
        n_old = first.values.shape[0] * first.values.shape[1]
        n_new = wider.values.shape[0] * wider.values.shape[1]
        assert built["n"] == n_new - n_old, "extension must build exactly the new cells"
        # The old block, at its offsets inside the union, is the stored data.
        dx = first.ix0 - wider.ix0
        dy = first.iy0 - wider.iy0
        np.testing.assert_array_equal(
            wider.values[dx : dx + first.values.shape[0], dy : dy + first.values.shape[1]],
            first.values,
        )

    def test_a_contained_request_returns_the_stored_superset(self, tmp_path, monkeypatch) -> None:
        """A narrower later request must not shrink or rebuild: the superset works
        transparently through the window offsets the samplers carry."""
        import pyRadMC.data.goudsmit_saunderson as gs

        wide = self._fresh(
            monkeypatch,
            tmp_path,
            log_eta_min=LOG_ETA - 0.3,
            log_eta_max=LOG_ETA,
            theta2_max=4.0e-12,
        )
        gs._grid_cache.clear()
        monkeypatch.setattr(
            gs,
            "gs_scaled_deflection_table",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not rebuild")),
        )
        narrow = build_gs_grid(cache_dir=tmp_path, **self.WINDOW)
        np.testing.assert_array_equal(narrow.values, wide.values)

    def test_a_construction_version_bump_invalidates(self, tmp_path, monkeypatch) -> None:
        """The one legitimate rebuild trigger: the construction algorithm moved."""
        import pyRadMC.data.goudsmit_saunderson as gs

        self._fresh(monkeypatch, tmp_path, **self.WINDOW)
        monkeypatch.setattr(gs, "_GS_GRID_CACHE_VERSION", 999_999)
        gs._grid_cache.clear()
        rebuilt = build_gs_grid(cache_dir=tmp_path, **self.WINDOW)  # must not crash or load
        assert np.isfinite(rebuilt.values).all()
        assert len(list(tmp_path.glob("*.npz"))) == 2, "old and new versions are distinct files"

    def test_a_corrupt_cache_file_rebuilds(self, tmp_path, monkeypatch) -> None:
        import pyRadMC.data.goudsmit_saunderson as gs

        self._fresh(monkeypatch, tmp_path, **self.WINDOW)
        (cache_file,) = tmp_path.glob("*.npz")
        cache_file.write_bytes(b"not an npz")
        gs._grid_cache.clear()
        rebuilt = build_gs_grid(cache_dir=tmp_path, **self.WINDOW)
        assert np.isfinite(rebuilt.values).all()

    def test_overridden_constants_stay_off_disk(self, tmp_path, monkeypatch) -> None:
        """Test-instrument builds (non-default floor/bins/n_u) never pollute the
        production cache file."""
        self._fresh(
            monkeypatch,
            tmp_path,
            log_eta_min=LOG_ETA,
            log_eta_max=LOG_ETA,
            theta2_max=THETA2_MAX,
            theta2_min=THETA2_MIN,
        )
        assert list(tmp_path.glob("*.npz")) == []


class TestWindowFromTables:
    """The reachable-key window, derived from the flattened transport tables."""

    def test_eta_bounds_are_the_log_eta_row_extremes(self, tables) -> None:
        """The kernel reads eta through ``lookup_2d(log_eta, ...)``, which is a
        convex combination of row nodes clamped flat at the grid edges — so the
        row min/max *are* the reachable bounds, exactly."""
        log_eta_min, log_eta_max, _ = gs_window_from_tables(tables, [WATER])
        assert log_eta_min == float(tables.log_eta[WATER].min())
        assert log_eta_max == float(tables.log_eta[WATER].max())

    def test_theta2_ceiling_bounds_the_substep_deflection(self, tables) -> None:
        r"""``theta2 = T(E_hinge) rho s`` with ``s <= restricted_range(E_start)/rho``.

        The range-out clamp bounds every substep by the restricted range
        regardless of ``step_energy_fraction``, and the hinge's scattering power
        is a table lookup bounded by its row maximum — so the product of row
        maxima is a hard, f-independent ceiling. Generous by construction (the
        two maxima sit at opposite ends of the energy grid); the excess costs a
        few cheap series-regime rows, never correctness.
        """
        _, _, theta2_max = gs_window_from_tables(tables, [WATER])
        t_max = float(tables.scattering_power[WATER].max())
        rr_max = float(tables.restricted_range[WATER].max())
        assert theta2_max == pytest.approx(t_max * rr_max, rel=1e-12)

    def test_window_covers_a_dense_energy_probe(self, xs, tables) -> None:
        """Every eta the host sampler would compute lies inside the window."""
        log_eta_min, log_eta_max, _ = gs_window_from_tables(tables, [WATER])
        for e in np.geomspace(ECUT_MEV / 2.0, E_MAX, 200):
            log_eta = math.log(xs.elastic_screening(float(e), WATER))
            assert log_eta_min - 1e-9 <= log_eta <= log_eta_max + 1e-9


class TestLogEtaTable:
    """``CrossSectionTables.log_eta``: the kernel-side face of ``elastic_screening``."""

    def test_log_eta_parity_at_the_grid_nodes(self, xs, tables) -> None:
        """Node values equal the exact host method; the same parity contract as
        every other flattened table (the log is stored so interpolation happens
        in the nearly-linear variable and the kernel skips a log per hinge)."""
        for j in range(0, tables.n_points, 7):
            e = math.exp(tables.electron_log_e_min + j / tables.electron_inv_dlog)
            expected = math.log(xs.elastic_screening(e, WATER))
            assert tables.log_eta[WATER, j] == pytest.approx(expected, abs=1e-12)

    def test_log_eta_shape_follows_declared_materials(self, xs, tables) -> None:
        assert tables.log_eta.shape == (xs.n_materials, tables.n_points)
