"""electron buildup and range behaviour (integration tier).

Expectations are physics-derived from the engine's own data source (CSDA ranges), so
these tests isolate the *transport* from the data, which is ESTAR-pinned in the unit
tier. Tolerances are integration-tier loose; the validation tier repeats the range
checks with more statistics and tighter windows.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine, TransportResult
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.materials import WATER
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import ParallelBeamSource
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED


def _run_photon_beam(transport_electrons: bool) -> tuple[TransportResult, np.ndarray]:
    """6 MeV broad photon beam on a fine-z water slab; returns result and depth axis."""
    grid = VoxelGrid.uniform_water(shape=(8, 8, 30), spacing=(2.0, 2.0, 0.2))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = ParallelBeamSource(energy=6.0, z=-1.0, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    result = engine.run(
        source,
        n_histories=4_000,
        n_batches=8,
        seed=SEED,
        transport_electrons=transport_electrons,
    )
    depth = (np.arange(30) + 0.5) * 0.2
    return result, depth


@pytest.fixture(scope="module")
def transport() -> tuple[TransportResult, np.ndarray]:
    return _run_photon_beam(transport_electrons=True)


@pytest.fixture(scope="module")
def kerma() -> tuple[TransportResult, np.ndarray]:
    return _run_photon_beam(transport_electrons=False)


@pytest.fixture(scope="module")
def beam() -> tuple[TransportResult, np.ndarray, float]:
    """5 MeV broad electron beam on a fine-z water slab."""
    grid = VoxelGrid.uniform_water(shape=(8, 8, 40), spacing=(2.0, 2.0, 0.1))
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    source = ParallelBeamSource(energy=5.0, z=-0.5, x_range=(0.0, 16.0), y_range=(0.0, 16.0))
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    result = engine.run(source, n_histories=1_200, n_batches=8, seed=SEED, primary_kind="electron")
    depth = (np.arange(40) + 0.5) * 0.1
    r_csda = xs.csda_range(5.0, WATER)  # g/cm^2 == cm at unit density
    return result, depth, r_csda


class TestPhotonBeamElectronBuildup:
    """A monoenergetic 6 MeV parallel beam builds up over the secondary-electron range.

    The same beam in KERMA mode is the control: with local deposition there is no
    electron buildup, so the surface slice sits at the plateau. The contrast pins the
    buildup on electron transport rather than on scatter or geometry.
    """

    def test_surface_dose_is_a_small_fraction_of_maximum(
        self, transport: tuple[TransportResult, np.ndarray]
    ) -> None:
        result, _ = transport
        dose_z = result.dose.mean(axis=(0, 1))
        assert dose_z[0] < 0.3 * dose_z.max(), (
            f"no electron buildup: surface/max = {dose_z[0] / dose_z.max():.2f}"
        )

    def test_buildup_reaches_90_percent_within_the_electron_range(
        self, transport: tuple[TransportResult, np.ndarray]
    ) -> None:
        """Dose first reaches 90 percent of max at a depth of order the range of the
        most energetic secondaries (~3 cm for 6 MeV photoelectrons), not at the
        surface (KERMA) and not many ranges deep (over-transported electrons).
        """
        result, depth = transport
        dose_z = result.dose.mean(axis=(0, 1))
        first_90 = depth[np.nonzero(dose_z >= 0.9 * dose_z.max())[0][0]]
        assert 0.6 <= first_90 <= 4.5, f"90 percent of Dmax first reached at {first_90:.1f} cm"

    def test_rise_from_surface_to_plateau(
        self, transport: tuple[TransportResult, np.ndarray]
    ) -> None:
        result, depth = transport
        dose_z = result.dose.mean(axis=(0, 1))
        near_surface = dose_z[depth < 0.6].mean()
        plateau = dose_z[(depth > 1.6) & (depth < 3.0)].mean()
        assert plateau > 2.0 * near_surface

    def test_kerma_control_shows_no_buildup(
        self, kerma: tuple[TransportResult, np.ndarray]
    ) -> None:
        """With local deposition the surface slice already sits near the plateau."""
        result, _ = kerma
        dose_z = result.dose.mean(axis=(0, 1))
        assert dose_z[0] > 0.6 * dose_z.max(), (
            f"KERMA control unexpectedly shows buildup: surface/max = "
            f"{dose_z[0] / dose_z.max():.2f}"
        )


class TestElectronBeamRange:
    """5 MeV broad electron beam: depth dose consistent with the CSDA range."""

    def test_energy_balance_with_electron_primaries(
        self, beam: tuple[TransportResult, np.ndarray, float]
    ) -> None:
        result, _, _ = beam
        balance = result.energy_deposited + result.energy_escaped
        assert balance == pytest.approx(result.energy_emitted, rel=1e-9)

    def test_r50_scales_with_the_csda_range(
        self, beam: tuple[TransportResult, np.ndarray, float]
    ) -> None:
        """R50 sits below R_CSDA by the detour factor, not at it and not far under.

        For broad monoenergetic MeV electron beams R50/R_CSDA is ~0.8; the window is
        integration-tier loose. R50 far above would mean under-scattering or
        under-stopping; far below, the reverse.
        """
        result, depth, r_csda = beam
        dose_z = result.dose.mean(axis=(0, 1))
        above_half = np.nonzero(dose_z >= 0.5 * dose_z.max())[0]
        r50 = float(depth[above_half[-1]])
        assert 0.65 * r_csda <= r50 <= 0.95 * r_csda, (
            f"R50 = {r50:.2f} cm vs R_CSDA = {r_csda:.2f} cm (ratio {r50 / r_csda:.2f})"
        )

    def test_only_the_brems_tail_survives_beyond_the_csda_range(
        self, beam: tuple[TransportResult, np.ndarray, float]
    ) -> None:
        """Past ~1.1 R_CSDA no electron can reach; what remains is bremsstrahlung."""
        result, depth, r_csda = beam
        dose_z = result.dose.mean(axis=(0, 1))
        tail = dose_z[depth > 1.1 * r_csda]
        assert tail.size > 0
        assert np.all(tail < 0.03 * dose_z.max()), (
            f"dose beyond 1.1 R_CSDA up to {tail.max() / dose_z.max():.1%} of Dmax: "
            "electrons are over-ranging"
        )

    def test_buildup_before_the_peak(self, beam: tuple[TransportResult, np.ndarray, float]) -> None:
        """Scatter detour concentrates dose below the surface: D(0) < Dmax."""
        result, _, _ = beam
        dose_z = result.dose.mean(axis=(0, 1))
        assert dose_z[0] < 0.9 * dose_z.max()
        assert int(np.argmax(dose_z)) > 0
