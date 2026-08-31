"""Flattened cross-section tables: the kernel-side data path of the Warp backend.

``build_tables`` (AGENTS.md 2.6: all data access through the interface) flattens a
``CrossSectionSource`` onto log-energy grids that kernels interpolate linearly. These
tests pin three things: parity of the interpolated values with the exact host methods,
the Woodcock majorant inequality *in float32 kernel arithmetic* (a violated majorant
is a smooth, silent under-attenuation bias — the worst kind), and clamped behaviour at
the grid edges.

The lookups themselves are single-source functions (``pyradmc.data.tables``): the same
code runs under NumPy here and compiles under ``@wp.func`` for the kernels, so parity
established here transfers to the device up to float32 rounding, which the warp-marked
test at the bottom verifies on each device.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc import ECUT_MEV, PCUT_MEV
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.interface import PhotonProcess
from pyradmc.data.materials import MATERIALS, WATER
from tests.conftest import SEED

E_MAX = 21.0
GEOMETRY = ((WATER, 1.2),)


@pytest.fixture(scope="module")
def xs() -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=GEOMETRY)


@pytest.fixture(scope="module")
def tables(xs: AnalyticCrossSections):
    return xs.build_tables(ecut=ECUT_MEV, pcut=PCUT_MEV, e_max=E_MAX)


def _photon_lookup(tables, values: np.ndarray, material: int, energy: float) -> float:
    from pyradmc.data.tables import lookup_loglinear_2d

    return lookup_loglinear_2d(
        values,
        material,
        tables.photon_log_e_min,
        tables.photon_inv_dlog,
        tables.n_points,
        energy,
    )


def _electron_lookup(tables, values: np.ndarray, material: int, energy: float) -> float:
    from pyradmc.data.tables import lookup_loglinear_2d

    return lookup_loglinear_2d(
        values,
        material,
        tables.electron_log_e_min,
        tables.electron_inv_dlog,
        tables.n_points,
        energy,
    )


class TestGridCoverage:
    def test_photon_grid_covers_pcut_to_e_max(self, tables) -> None:
        """Photons exist in (PCUT, e_max]; the grid must bracket that with margin."""
        assert np.exp(tables.photon_log_e_min) <= PCUT_MEV
        grid_max = tables.photon_log_e_min + (tables.n_points - 1) / tables.photon_inv_dlog
        assert np.exp(grid_max) >= E_MAX * (1.0 - 1e-12)

    def test_electron_grid_covers_below_ecut_to_e_max(self, tables) -> None:
        """Electron lookups happen for kinetic energies in (ECUT, e_max]."""
        assert np.exp(tables.electron_log_e_min) <= ECUT_MEV
        grid_max = tables.electron_log_e_min + (tables.n_points - 1) / tables.electron_inv_dlog
        assert np.exp(grid_max) >= E_MAX * (1.0 - 1e-12)

    def test_all_materials_tabulated(self, xs, tables) -> None:
        """Rows follow the *source's* declared coverage, not the registry size."""
        assert tables.mu_compton.shape == (xs.n_materials, tables.n_points)
        assert tables.majorant.shape == (tables.n_points,)


class TestSourceDeclaredMaterials:
    """Table sizing is source-declared .

    A source flattens exactly the materials it carries data for. The analytic source
    is calibrated for water only and must say so — once the registry grows past
    water, silently answering a bone query with water photoelectric/pair data would
    be the silent-bias failure mode AGENTS.md 2.2/2.8 exist to prevent.
    """

    def test_analytic_declares_water_only(self, xs) -> None:
        assert xs.n_materials == 1

    def test_analytic_rejects_material_beyond_declaration(self, xs) -> None:
        with pytest.raises(ValueError, match="material"):
            xs.mu_over_rho(1.0, xs.n_materials, PhotonProcess.COMPTON)
        with pytest.raises(ValueError, match="material"):
            xs.restricted_stopping_power(1.0, xs.n_materials, ECUT_MEV)

    def test_interface_default_covers_registry(self) -> None:
        """A material-independent source (test instruments) defaults to the registry."""
        from tests.integration.test_channel_complete import CoherentOnlySource

        assert CoherentOnlySource().n_materials == len(MATERIALS)

    def test_tables_sized_by_declared_count(self, xs) -> None:
        tables = xs.build_tables(ecut=ECUT_MEV, pcut=PCUT_MEV, e_max=E_MAX, n_points=16)
        for field in ("mu_compton", "stopping_restricted", "coherent_cumulative"):
            assert getattr(tables, field).shape[0] == xs.n_materials


class TestPhotonParity:
    """Interpolated channel coefficients against the exact host formulas.

    Tolerances are interpolation-error budgets, not statistical: at 2048 points over
    [PCUT/2, 21] MeV the local log-spacing is ~3e-3 and the linear-in-value error is
    second order in it. Errors are scaled against the *total* mu so that a channel
    passing through zero (pair at threshold) cannot inflate a relative comparison.
    """

    CHANNELS = (
        ("mu_compton", PhotonProcess.COMPTON),
        ("mu_photo", PhotonProcess.PHOTOELECTRIC),
        ("mu_pair", PhotonProcess.PAIR),
        ("mu_rayleigh", PhotonProcess.RAYLEIGH),
    )

    def test_channel_lookups_match_host(self, xs, tables) -> None:
        rng = np.random.default_rng(SEED)
        energies = np.exp(rng.uniform(np.log(PCUT_MEV), np.log(E_MAX), 4000))
        for material in range(xs.n_materials):
            for energy in energies:
                e = float(energy)
                total = xs.mu_over_rho_total(e, material)
                for field, process in self.CHANNELS:
                    got = _photon_lookup(tables, getattr(tables, field), material, e)
                    want = xs.mu_over_rho(e, material, process)
                    assert got == pytest.approx(want, rel=2e-4, abs=2e-4 * total), (
                        f"{field} at {e:.4f} MeV: table {got:.6e} vs host {want:.6e}"
                    )

    def test_rayleigh_channel_is_zero_for_analytic_source(self, tables) -> None:
        """No Rayleigh by default (data decision, AGENTS.md 2.10): all-zero table."""
        assert np.all(tables.mu_rayleigh == 0.0)


class TestMajorantContract:
    def test_majorant_bounds_interpolated_total_in_float32(self, xs, tables) -> None:
        """The kernel-arithmetic majorant inequality, checked densely.

        The kernel compares ``rho * sum(channel lookups)`` against the interpolated
        majorant in float32. Both sides interpolate the same grid, and the builder
        gives the majorant nodes headroom above the float64 maximum, so the
        inequality must survive the float32 cast everywhere — including between
        nodes and at the pair threshold kink. A failure here is the silent
        under-attenuation bias the interface docstring warns about.
        """
        rng = np.random.default_rng(SEED)
        energies = np.exp(rng.uniform(np.log(PCUT_MEV), np.log(E_MAX), 20000))
        f32 = {
            field: getattr(tables, field).astype(np.float32)
            for field, _ in TestPhotonParity.CHANNELS
        }
        majorant32 = tables.majorant.astype(np.float32)
        from pyradmc.data.tables import lookup_loglinear_1d

        for energy in energies:
            e = float(energy)
            majorant = lookup_loglinear_1d(
                majorant32, tables.photon_log_e_min, tables.photon_inv_dlog, tables.n_points, e
            )
            for material, rho_max in GEOMETRY:
                mu_real = np.float32(rho_max) * np.float32(
                    sum(
                        _photon_lookup(tables, f32[field], material, e)
                        for field, _ in TestPhotonParity.CHANNELS
                    )
                )
                assert mu_real <= majorant, (
                    f"majorant violated at {e:.4f} MeV: {mu_real:.6e} > {majorant:.6e}"
                )


class TestElectronParity:
    QUANTITIES = (
        ("stopping_restricted", lambda xs, e, m: xs.restricted_stopping_power(e, m, ECUT_MEV)),
        ("stopping_radiative", lambda xs, e, m: xs.radiative_stopping_power(e, m)),
        ("moller", lambda xs, e, m: xs.moller_cross_section(e, m, ECUT_MEV)),
        ("csda_range", lambda xs, e, m: xs.csda_range(e, m)),
        ("scattering_power", lambda xs, e, m: xs.scattering_power(e, m)),
    )

    def test_electron_lookups_match_host(self, xs, tables) -> None:
        """Parity at the energies the transport actually queries: above ECUT.

        Error budget by quantile, not a single tolerance: where the host functions
        are smooth, linear interpolation is second order in the grid spacing (99th
        percentile below 1e-3); where they have derivative kinks — the Moller
        switch-on at 2*ECUT, the density-effect boundaries, the w = 1/2 restriction
        limit — it degrades to first order in the cells containing the kink, which
        the max-error cap bounds. Near the Moller switch-on, where the exact value
        passes through zero, the pointwise comparison is replaced by a
        neighbouring-cell bound.
        """
        rng = np.random.default_rng(SEED)
        energies = np.exp(rng.uniform(np.log(ECUT_MEV * 1.001), np.log(E_MAX), 4000))
        dlog = 1.0 / tables.electron_inv_dlog
        for material in range(xs.n_materials):
            for field, host_fn in self.QUANTITIES:
                relative_errors = []
                for energy in energies:
                    e = float(energy)
                    got = _electron_lookup(tables, getattr(tables, field), material, e)
                    want = host_fn(xs, e, material)
                    near_moller_kink = field == "moller" and (
                        abs(np.log(e) - np.log(2.0 * ECUT_MEV)) < 2.0 * dlog
                    )
                    if near_moller_kink:
                        upper = host_fn(xs, float(np.exp(np.log(e) + 2.0 * dlog)), material)
                        assert 0.0 <= got <= upper * 1.001
                    elif want == 0.0:
                        assert got == 0.0, f"{field} at {e:.4f} MeV: table {got:.6e} vs 0"
                    else:
                        relative_errors.append(abs(got - want) / abs(want))
                errors = np.array(relative_errors)
                assert np.quantile(errors, 0.99) < 1e-3, f"{field}: smooth-region budget"
                assert errors.max() < 5e-3, f"{field}: kink-cell budget ({errors.max():.2e})"

    def test_moller_is_nonnegative_everywhere(self, tables) -> None:
        """A negative interaction coefficient is never acceptable (cf. pair fit)."""
        assert np.all(tables.moller >= 0.0)


class TestLookupEdges:
    def test_lookup_clamps_flat_outside_grid(self, tables) -> None:
        """Below/above the grid the lookup returns the edge node, never extrapolates.

        Linear extrapolation of an E^-3 photoelectric channel below the grid would
        go negative; clamping is the safe, documented behaviour.
        """
        lo = float(np.exp(tables.photon_log_e_min))
        hi = lo * float(np.exp((tables.n_points - 1) / tables.photon_inv_dlog))
        at_lo = _photon_lookup(tables, tables.mu_compton, WATER, lo)
        below = _photon_lookup(tables, tables.mu_compton, WATER, lo * 0.5)
        at_hi = _photon_lookup(tables, tables.mu_compton, WATER, hi)
        above = _photon_lookup(tables, tables.mu_compton, WATER, hi * 2.0)
        assert below == pytest.approx(at_lo, rel=1e-12)
        assert above == pytest.approx(at_hi, rel=1e-12)


@pytest.mark.warp
def test_warp_compiled_lookup_matches_host_lookup(tables) -> None:
    """The same lookup source, compiled under Warp, on every available device.

    float32 tolerance: the device interpolates float32 tables with float32 index
    arithmetic against the host's float64. This is the entire kernel data path;
    everything the transport kernels see flows through these two functions.
    """
    wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp.physics import warp_physics

    p = warp_physics()
    lookup_2d = p.lookup_loglinear_2d
    lookup_1d = p.lookup_loglinear_1d

    @wp.kernel
    def k(
        values2: wp.array2d(dtype=float),
        values1: wp.array(dtype=float),
        log_e_min: float,
        inv_dlog: float,
        n_points: int,
        energies: wp.array(dtype=float),
        out: wp.array2d(dtype=float),
    ):
        tid = wp.tid()
        e = energies[tid]
        out[tid, 0] = lookup_2d(values2, 0, log_e_min, inv_dlog, n_points, e)
        out[tid, 1] = lookup_1d(values1, log_e_min, inv_dlog, n_points, e)

    rng = np.random.default_rng(SEED)
    energies = np.exp(rng.uniform(np.log(PCUT_MEV * 0.4), np.log(E_MAX * 1.1), 2000))

    from pyradmc.data.tables import lookup_loglinear_1d, lookup_loglinear_2d

    compton32 = tables.mu_compton.astype(np.float32)
    majorant32 = tables.majorant.astype(np.float32)
    want = np.array(
        [
            [
                lookup_loglinear_2d(
                    compton32,
                    0,
                    tables.photon_log_e_min,
                    tables.photon_inv_dlog,
                    tables.n_points,
                    float(e),
                ),
                lookup_loglinear_1d(
                    majorant32,
                    tables.photon_log_e_min,
                    tables.photon_inv_dlog,
                    tables.n_points,
                    float(e),
                ),
            ]
            for e in energies
        ]
    )

    devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
    for device in devices:
        values2 = wp.array(compton32, dtype=float, device=device)
        values1 = wp.array(majorant32, dtype=float, device=device)
        e_dev = wp.array(energies.astype(np.float32), dtype=float, device=device)
        out = wp.zeros((energies.size, 2), dtype=float, device=device)
        wp.launch(
            k,
            dim=energies.size,
            inputs=[
                values2,
                values1,
                float(tables.photon_log_e_min),
                float(tables.photon_inv_dlog),
                int(tables.n_points),
                e_dev,
            ],
            outputs=[out],
            device=device,
        )
        wp.synchronize_device(device)
        np.testing.assert_allclose(out.numpy(), want, rtol=3e-5, atol=1e-9)
