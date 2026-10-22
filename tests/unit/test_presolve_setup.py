"""Host-side setup cost of the collimator pre-solve and the collimator geometry objects.

The attenuation tables of the head pre-solve and of the stack transmission model are
memoised per cross-section source (``pyradmc.geometry.mucache``), and the geometry
validators are cheaper than their ``np.allclose``/``np.diff`` originals. None of it
may change a number: every assertion below compares against the *removed* scalar
loops, reproduced here as oracles, with exact (``assert_array_equal``) equality.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.interface import PhotonProcess
from pyradmc.data.materials import WATER
from pyradmc.data.tabulated.model import TabulatedData
from pyradmc.data.tabulated.source import TabulatedCrossSections
from pyradmc.geometry import mucache
from pyradmc.geometry.collimation import (
    MLC,
    BeamFrame,
    BeamLimitingStack,
    JawPair,
    _StackTransmission,
)
from pyradmc.geometry.head import _MuTables

E_MIN, E_MAX, N_TABLE = 0.010, 6.0, 64
PAIRS = ((0, 1.0), (1, 19.3), (1, 17.0), (0, 1.205e-3))


def _synthetic_two_material() -> TabulatedData:
    pe = np.geomspace(0.01, 10.0, 32)
    ee = np.geomspace(0.01, 10.0, 32)
    base = {
        PhotonProcess.COMPTON: 0.15 * np.ones_like(pe),
        PhotonProcess.PHOTOELECTRIC: 1.0e-3 / pe**3,
        PhotonProcess.PAIR: np.maximum(pe - 1.022, 0.0) * 1.0e-3,
        PhotonProcess.RAYLEIGH: 0.01 / pe,
    }
    two = lambda a, f: np.stack([a, f * a])  # noqa: E731
    return TabulatedData(
        photon_energies=pe,
        mu_over_rho={p: two(v, 3.0) for p, v in base.items()},
        electron_energies=ee,
        restricted_stopping=two(2.0 / ee, 0.5),
        radiative_stopping=two(0.01 * ee, 0.5),
        moller=two(0.1 / ee, 0.5),
        csda_range=two(0.5 * ee, 2.0),
        scattering_power=two(1.0 / ee, 0.5),
        delta_cut=0.2,
        materials=("water", "air"),
        provenance="synthetic",
    )


class CountingSource(TabulatedCrossSections):
    """A tabulated source that counts photon queries."""

    queries = 0

    def mu_over_rho(self, energy: float, material: int, process: int) -> float:
        type(self).queries += 1
        return super().mu_over_rho(energy, material, process)


def _oracle(cross_sections, pairs, e_min, e_max, n_table):
    """The removed scalar-loop construction of ``_MuTables``, verbatim."""
    energies = np.geomspace(e_min, max(e_max, e_min * (1.0 + 1e-9)), n_table)
    log_total = np.stack(
        [
            np.log([density * cross_sections.mu_over_rho_total(float(e), m) for e in energies])
            for m, density in pairs
        ]
    )
    rows = []
    for m, density in pairs:
        values = np.array(
            [
                density * cross_sections.mu_over_rho(float(e), m, PhotonProcess.COMPTON)
                for e in energies
            ]
        )
        rows.append(np.log(np.maximum(values, 1e-300)))
    return np.log(energies), log_total, np.stack(rows)


@pytest.fixture(params=["analytic", "tabulated"])
def source(request):
    mucache.clear()
    if request.param == "analytic":
        return AnalyticCrossSections()
    return TabulatedCrossSections(_synthetic_two_material())


def _pairs_for(source) -> tuple[tuple[int, float], ...]:
    limit = source.n_materials
    return tuple((m % limit, d) for m, d in PAIRS)


def test_mu_tables_equal_the_scalar_loops_exactly(source) -> None:
    pairs = _pairs_for(source)
    tables = _MuTables(source, pairs, E_MIN, E_MAX, N_TABLE)
    log_e, log_total, log_compton = _oracle(source, pairs, E_MIN, E_MAX, N_TABLE)
    np.testing.assert_array_equal(tables._log_energies, log_e)
    np.testing.assert_array_equal(tables._log_total, log_total)
    np.testing.assert_array_equal(tables._log_compton, log_compton)
    # A second build (cache hits everywhere) is still the oracle, bit for bit.
    again = _MuTables(source, pairs, E_MIN, E_MAX, N_TABLE)
    np.testing.assert_array_equal(again._log_total, log_total)
    np.testing.assert_array_equal(again._log_compton, log_compton)


def test_degenerate_energy_range_matches_the_scalar_loops(source) -> None:
    pairs = _pairs_for(source)[:2]
    tables = _MuTables(source, pairs, 0.5, 0.5, 8)  # e_max == e_min: nudged upper edge
    _, log_total, log_compton = _oracle(source, pairs, 0.5, 0.5, 8)
    np.testing.assert_array_equal(tables._log_total, log_total)
    np.testing.assert_array_equal(tables._log_compton, log_compton)


def test_repeat_builds_make_no_cross_section_queries() -> None:
    mucache.clear()
    xs = CountingSource(_synthetic_two_material())
    CountingSource.queries = 0
    _MuTables(xs, PAIRS[:2], E_MIN, E_MAX, N_TABLE)
    cold = CountingSource.queries
    assert cold > 0
    CountingSource.queries = 0
    _MuTables(xs, PAIRS[:2], E_MIN, E_MAX, N_TABLE)
    assert CountingSource.queries == 0


def test_a_new_density_reuses_the_material_curves_and_stays_exact() -> None:
    mucache.clear()
    xs = CountingSource(_synthetic_two_material())
    _MuTables(xs, ((0, 1.0), (1, 19.3)), E_MIN, E_MAX, N_TABLE)
    CountingSource.queries = 0
    retuned = ((0, 1.0), (1, 18.7))
    tables = _MuTables(xs, retuned, E_MIN, E_MAX, N_TABLE)
    assert CountingSource.queries == 0
    _, log_total, log_compton = _oracle(xs, retuned, E_MIN, E_MAX, N_TABLE)
    np.testing.assert_array_equal(tables._log_total, log_total)
    np.testing.assert_array_equal(tables._log_compton, log_compton)


def test_distinct_sources_are_never_conflated() -> None:
    mucache.clear()
    data = _synthetic_two_material()
    scaled = TabulatedData(
        **{
            **data.__dict__,
            "mu_over_rho": {p: 2.0 * v for p, v in data.mu_over_rho.items()},
        }
    )
    a = _MuTables(TabulatedCrossSections(data), PAIRS[:2], E_MIN, E_MAX, N_TABLE)
    b = _MuTables(TabulatedCrossSections(scaled), PAIRS[:2], E_MIN, E_MAX, N_TABLE)
    assert not np.array_equal(a._log_total, b._log_total)


def test_the_memo_dies_with_its_source() -> None:
    mucache.clear()
    xs = TabulatedCrossSections(_synthetic_two_material())
    _MuTables(xs, PAIRS[:2], E_MIN, E_MAX, N_TABLE)
    assert len(mucache._entries) == 1
    del xs
    gc.collect()
    assert len(mucache._entries) == 0


def test_the_memoised_tables_are_returned_for_identical_requests() -> None:
    mucache.clear()
    xs = TabulatedCrossSections(_synthetic_two_material())
    first = mucache.mu_curve(xs, 1, E_MIN, E_MAX, N_TABLE, None)
    second = mucache.mu_curve(xs, 1, E_MIN, E_MAX, N_TABLE, None)
    assert first[1] is second[1]
    assert not first[1].flags.writeable


def test_stack_transmission_table_equals_the_scalar_loop() -> None:
    mucache.clear()
    xs = TabulatedCrossSections(_synthetic_two_material())
    stack = BeamLimitingStack(
        frame=BeamFrame(origin=(0.0, 0.0, -100.0)),
        devices=(
            JawPair("u", 40.0, 45.0, -1000.0, 0.5, material=1, density=19.3),
            JawPair("v", 46.0, 50.0, -3.0, 3.0, material=0, density=1.0),
        ),
    )
    for _ in range(2):  # cold, then warm
        model = _StackTransmission(stack, xs, e_max=6.0, e_min=0.01, n_table=64)
        energies = np.geomspace(0.01, 6.0, 64)
        expected = np.stack(
            [np.log([xs.mu_over_rho_total(float(e), m) for e in energies]) for m in stack.materials]
        )
        np.testing.assert_array_equal(model._log_mu, expected)
        np.testing.assert_array_equal(model._log_energies, np.log(energies))


# -- validation: identical semantics -------------------------------------------------


def _allclose_orthonormal(u, v, w) -> bool:
    """The removed check, verbatim."""
    axes = np.array([u, v, w])
    return bool(np.allclose(axes @ axes.T, np.eye(3), atol=1.0e-9))


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def test_beam_frame_orthonormality_decision_equals_allclose_across_the_tolerance() -> None:
    for tilt in (0.0, 1e-12, 1e-10, 3e-10, 9e-10, 1.1e-9, 2e-9, 1e-8, 1e-3):
        for scale in (1.0, -1.0, 0.5):
            v_axis = (0.0, 1.0, 0.0)
            u_axis = (1.0, scale * tilt, 0.0)
            u, v, w = _unit(np.asarray(u_axis)), _unit(np.asarray(v_axis)), (0.0, 0.0, 1.0)
            right_handed = float(np.dot(np.cross(u, v), w)) >= 1.0 - 1.0e-9
            expected_ok = _allclose_orthonormal(u, v, w) and right_handed
            try:
                BeamFrame(origin=(0, 0, 0), u_axis=u_axis, v_axis=v_axis)
                ok = True
            except ValueError:
                ok = False
            assert ok == expected_ok, (tilt, u_axis)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"z_top": -1.0}, "z_top"),
        ({"z_bottom": 45.0}, "z_bottom"),
        ({"tip_radius": 1.0}, "tip_radius"),
        ({"leaf_edges_v": (1.0,)}, "at least two"),
        ({"leaf_edges_v": (-3.0, 1.0, 1.0, 3.0)}, "strictly increasing"),
        ({"leaf_edges_v": (-3.0, 2.0, 1.0, 3.0)}, "strictly increasing"),
        ({"tips_neg": (-1.0, -1.0)}, "tips_neg has 2"),
        ({"tips_pos": (1.0,) * 4}, "tips_pos has 4"),
        ({"tips_neg": (-1.0, 2.0, -1.0)}, "pair 1: tip positions cross"),
        ({"tips_neg": (3.0, 2.0, -1.0), "tips_pos": (1.0, 1.0, 1.0)}, "pair 0: tip"),
        ({"material": -1}, "material"),
        ({"density": 0.0}, "density"),
        ({"leaf_edges_v": (-3.0, float("nan"), 1.0, 3.0)}, "strictly increasing"),
    ],
)
def test_mlc_validation_raises_the_same_errors(overrides, match) -> None:
    settings = {
        "z_top": 46.0,
        "z_bottom": 50.0,
        "leaf_edges_v": (-3.0, -1.0, 1.0, 3.0),
        "tips_neg": (-1.0, -1.0, -1.0),
        "tips_pos": (1.0, 1.0, 1.0),
        "tip_radius": 10.0,
        "material": WATER,
        "density": 1.0,
    }
    settings.update(overrides)
    with pytest.raises(ValueError, match=match):
        MLC(**settings)


def test_mlc_accepts_touching_tips_and_a_single_pair() -> None:
    MLC(46.0, 50.0, (-1.0, 1.0), (0.5,), (0.5,), 10.0, WATER, 1.0)
