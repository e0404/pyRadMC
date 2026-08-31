"""The treatment-head pre-solve: algebraic weight identities and population structure.

``presolve_head`` runs once per field on the host and scores an exit-plane phase
space; these tests pin its deterministic algebra (Beer-Lambert weights per mode),
the population invariants (kinds, cutoffs, forward-going, on-plane), and the mode
contract (``auto`` == ``first_compton``; ``attenuation`` == the wrapper physics).
Water devices on the analytic backend keep the tier EPDL-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.collimation import BeamFrame, BeamLimitingStack, JawPair
from pyradmc.geometry.head import AirColumn, presolve_head
from pyradmc.geometry.source import GaussianSpotBeamSource
from pyradmc.geometry.spectrum import Spectrum
from tests.conftest import SEED

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
FRAME = BeamFrame(origin=FOCAL)
THICKNESS = 7.0
N = 512


def _plane_source(width: float = 6.0) -> GaussianSpotBeamSource:
    """Photons emitted at local w = 30 (z = -70), upstream of the jaws at w in [40, 47]."""
    return GaussianSpotBeamSource(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        center=(8.0, 8.0, -70.0),
        width_u=width,
        width_v=width,
        sigma_u=0.1,
        sigma_v=0.1,
    )


def _jaws(edge_pos: float) -> BeamLimitingStack:
    return BeamLimitingStack(
        frame=FRAME,
        devices=(
            JawPair(
                axis="u",
                z_top=40.0,
                z_bottom=40.0 + THICKNESS,
                edge_neg=-1000.0,
                edge_pos=edge_pos,
                material=WATER,
                density=1.0,
            ),
        ),
    )


def _closed_jaws() -> BeamLimitingStack:
    """Both blocks meet on the axis: every forward ray is blocked."""
    return BeamLimitingStack(
        frame=FRAME,
        devices=(
            JawPair(
                axis="u",
                z_top=40.0,
                z_bottom=40.0 + THICKNESS,
                edge_neg=0.0,
                edge_pos=0.0,
                material=WATER,
                density=1.0,
            ),
        ),
    )


def _xs() -> AnalyticCrossSections:
    return AnalyticCrossSections()


class TestPassThrough:
    def test_no_stack_no_air_reproduces_the_input_rays(self) -> None:
        """The pre-solve with nothing in the beam is the sampled source, exactly:
        same energies and directions, unit weights, positions slid down the rays
        to the exit plane."""
        source = _plane_source()
        result = presolve_head(
            source,
            stack=None,
            frame=FRAME,
            cross_sections=_xs(),
            n_histories=N,
            seed=SEED,
            exit_z=60.0,
        )
        rays = source.sample_batch(SEED, 0, N)
        out = _stored_columns(result)
        assert len(result) == N
        np.testing.assert_array_equal(np.sort(out["energy"]), np.sort(rays["energy"]))
        assert np.all(out["weight"] == 1.0)
        # Positions sit on the exit plane (local w = 60 => z = -40 for this frame)
        # along the original rays: x = x0 + t * ux with t = dz / uz.
        order_out = np.argsort(out["energy"])
        order_in = np.argsort(rays["energy"])
        t = (-40.0 - rays["z"].astype(np.float64)[order_in]) / rays["uz"].astype(np.float64)[
            order_in
        ]
        np.testing.assert_allclose(
            out["x"][order_out],
            rays["x"].astype(np.float64)[order_in] + t * rays["ux"].astype(np.float64)[order_in],
            atol=1e-4,
        )


class TestAttenuationMode:
    def test_weights_are_beer_lambert_exactly(self) -> None:
        """attenuation mode: output photon weights equal exp(-mu rho t) per ray."""
        source = _plane_source()
        stack = _jaws(edge_pos=0.0)  # +u half blocked
        result = presolve_head(
            source,
            stack=stack,
            cross_sections=_xs(),
            n_histories=N,
            seed=SEED,
            exit_z=60.0,
            mode="attenuation",
        )
        rays = source.sample_batch(SEED, 0, N)
        origins = np.stack([rays["x"], rays["y"], rays["z"]], 1).astype(np.float64)
        directions = np.stack([rays["ux"], rays["uy"], rays["uz"]], 1).astype(np.float64)
        thickness = stack.path_lengths(origins, directions)[:, 0]
        mu = np.array(
            [_xs().mu_over_rho_total(float(e), WATER) for e in rays["energy"].astype(np.float64)]
        )
        expected = np.exp(-mu * thickness)
        out = _stored_columns(result)
        # rtol covers the pre-solve's 512-node log-log table against the exact
        # scalar interface call (the same tolerance the wrapper's own gate uses).
        np.testing.assert_allclose(np.sort(out["weight"]), np.sort(expected), rtol=1e-4)
        assert set(np.unique(out["particle_type"])) == {1}

    def test_closed_jaws_transmit_the_bulk_factor(self) -> None:
        source = _plane_source(width=2.0)
        result = presolve_head(
            source,
            stack=_closed_jaws(),
            cross_sections=_xs(),
            n_histories=N,
            seed=SEED,
            exit_z=60.0,
            mode="attenuation",
        )
        out = _stored_columns(result)
        mu_at = {
            float(e): _xs().mu_over_rho_total(float(e), WATER) for e in np.unique(out["energy"])
        }
        for row in range(out["energy"].size):
            e = float(out["energy"][row])
            # Slant paths exceed the nominal thickness slightly (small divergence).
            low = np.exp(-mu_at[e] * 1.05 * THICKNESS)
            high = np.exp(-mu_at[e] * THICKNESS)
            assert low * 0.999 <= out["weight"][row] <= high * 1.001


class TestFirstComptonMode:
    def test_auto_resolves_to_first_compton(self) -> None:
        source = _plane_source()
        kwargs = dict(
            stack=_jaws(edge_pos=0.0),
            cross_sections=_xs(),
            n_histories=N,
            seed=SEED,
            exit_z=60.0,
        )
        auto = _stored_columns(presolve_head(source, mode="auto", **kwargs))
        explicit = _stored_columns(presolve_head(source, mode="first_compton", **kwargs))
        for name in auto:
            np.testing.assert_array_equal(auto[name], explicit[name])

    def test_population_structure(self) -> None:
        """Transmitted + scattered photons, no electrons without air; all forward,
        on the exit plane, above cutoff."""
        source = _plane_source()
        result = presolve_head(
            source,
            stack=_jaws(edge_pos=0.0),
            cross_sections=_xs(),
            n_histories=N,
            seed=SEED,
            exit_z=60.0,
            mode="first_compton",
        )
        out = _stored_columns(result)
        assert set(np.unique(out["particle_type"])) == {1}  # no air => no electrons
        assert out["energy"].size > N  # transmitted + kept scattered photons
        assert np.all(out["energy"] >= 0.05 - 1e-9)  # PCUT
        assert np.all(out["uz"] > 0.0)  # forward-going only (default axes: w = z)
        assert np.all(np.abs(out["z"] - (-100.0 + 60.0)) < 1e-5)  # on the exit plane
        assert np.all(out["weight"] > 0.0)

    def test_population_is_normalized_per_primary(self) -> None:
        """Stored weights carry the N/n factor so runs estimate dose per primary.

        Regression pin for the normalization defect the workstream demo caught
        (2026-07-15): without the factor, uniform resampling of an N-row
        population built from n < N primaries diluted the per-history dose by
        n/N, and the first-Compton pre-solve sat ~40 percent below the
        deterministic wrapper in the open field. The transmitted subpopulation
        (identified by its unchanged energies) must carry exactly
        (N / n) * exp(-tau) per ray.
        """
        source = _plane_source()
        stack = _jaws(edge_pos=0.0)
        result = presolve_head(
            source,
            stack=stack,
            cross_sections=_xs(),
            n_histories=N,
            seed=SEED,
            exit_z=60.0,
            mode="first_compton",
        )
        out = _stored_columns(result)
        rays = source.sample_batch(SEED, 0, N)
        factor = out["energy"].size / N
        origins = np.stack([rays["x"], rays["y"], rays["z"]], 1).astype(np.float64)
        directions = np.stack([rays["ux"], rays["uy"], rays["uz"]], 1).astype(np.float64)
        thickness = stack.path_lengths(origins, directions)[:, 0]
        for i in range(0, N, 37):  # spot-check across the field
            row = np.flatnonzero(out["energy"] == np.float64(rays["energy"][i]))
            assert row.size == 1  # the transmitted copy; scattered energies moved
            mu = _xs().mu_over_rho_total(float(rays["energy"][i]), WATER)
            expected = factor * np.exp(-mu * thickness[i])
            assert out["weight"][row[0]] == pytest.approx(expected, rel=1e-4)

    def test_air_column_adds_contaminant_electrons(self) -> None:
        source = _plane_source()
        result = presolve_head(
            source,
            stack=_jaws(edge_pos=0.0),
            cross_sections=_xs(),
            n_histories=N,
            seed=SEED,
            exit_z=95.0,
            air=AirColumn(z_start=50.0, z_end=95.0, material=WATER, density=1.2e-3),
            mode="first_compton",
        )
        out = _stored_columns(result)
        kinds = set(np.unique(out["particle_type"]))
        assert kinds == {1, 2}
        electrons = out["particle_type"] == 2
        assert np.all(out["energy"][electrons] >= 0.2 - 1e-9)  # ECUT
        assert np.all(out["uz"][electrons] > 0.0)


class TestValidation:
    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            presolve_head(
                _plane_source(),
                stack=_jaws(0.0),
                cross_sections=_xs(),
                n_histories=8,
                seed=SEED,
                exit_z=60.0,
                mode="analog",
            )

    def test_needs_a_frame_or_a_stack(self) -> None:
        with pytest.raises(ValueError, match="frame"):
            presolve_head(
                _plane_source(),
                stack=None,
                cross_sections=_xs(),
                n_histories=8,
                seed=SEED,
                exit_z=60.0,
            )

    def test_air_before_stack_exit_raises(self) -> None:
        with pytest.raises(ValueError, match="air"):
            presolve_head(
                _plane_source(),
                stack=_jaws(0.0),
                cross_sections=_xs(),
                n_histories=8,
                seed=SEED,
                exit_z=60.0,
                air=AirColumn(z_start=30.0, z_end=60.0),
            )


def _stored_columns(result: object) -> dict[str, np.ndarray]:
    """Read the stored population back out of the in-memory phase space."""
    return result.columns()  # type: ignore[attr-defined]


class TestWeightFloorRoulette:
    def test_roulette_preserves_expected_weight(self) -> None:
        """Fair game: the summed weight is unchanged in expectation."""
        from pyradmc.geometry.head import _roulette_below

        generator = np.random.Generator(np.random.PCG64(SEED))
        weights = np.random.default_rng(1).uniform(0.0, 2.0e-3, 200_000)
        out = _roulette_below(weights.copy(), 1.0e-3, generator)
        assert out.sum() == pytest.approx(weights.sum(), rel=2e-2)
        surviving = out > 0.0
        assert np.all(out[surviving] >= 1.0e-3 - 1e-15)
        assert surviving.sum() < weights.size  # some were killed

    def test_zero_floor_disables_the_roulette(self) -> None:
        from pyradmc.geometry.head import _roulette_below

        generator = np.random.Generator(np.random.PCG64(SEED))
        weights = np.random.default_rng(2).uniform(0.0, 2.0e-3, 1_000)
        np.testing.assert_array_equal(_roulette_below(weights.copy(), 0.0, generator), weights)
