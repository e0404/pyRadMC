"""The primary-fluence virtual source transports consistently and shapes the field.

The reference engine runs ``emit`` (per-history streams); Warp takes the
vectorized pre-sampling batch (its own PCG64 stream) — independent draws, so the
comparison is the batched chi-squared, never bitwise (spectral-source precedent).

The third test is the one that says the model does anything: a fluence table that
stops at a small radius must collimate the field, because the histories outside it
are emitted with weight zero. That is checked on the reference engine alone, which
is cheap enough at this size and needs no backend agreement to be meaningful.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.fluence import RadialFluence
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import PrimaryFluenceBeamSource
from pyradmc.geometry.spectrum import Spectrum
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

pytestmark = pytest.mark.warp

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -80.0)
CENTER = (8.0, 8.0, -2.0)
# A horn plus a roll-off, quoted at the 100 cm reference distance. The emission
# plane sits 78 cm from the spot, so a table radius r maps to 0.78 r on the plane.
HORNED = RadialFluence(
    radii=(0.0, 2.0, 4.0, 5.0, 6.0), values=(1.0, 1.05, 1.10, 0.60, 0.0), reference_distance=100.0
)


def _source(fluence: RadialFluence = HORNED) -> PrimaryFluenceBeamSource:
    return PrimaryFluenceBeamSource(
        spectrum=SPECTRUM,
        fluence=fluence,
        focal_point=FOCAL,
        center=CENTER,
        width_u=6.0,
        width_v=6.0,
        sigma_u=0.2,
        sigma_v=0.2,
    )


def _grid() -> VoxelGrid:
    return VoxelGrid.uniform_water(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))


def test_primary_fluence_agrees_across_backends() -> None:
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    kwargs = dict(n_batches=8, seed=SEED, transport_electrons=False)
    ref = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
        _source(), n_histories=6_000, **kwargs
    )
    warp_result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        _source(), n_histories=24_000, **kwargs
    )
    mask = ref.dose > 0.1 * ref.dose.max()
    mask &= (ref.dose_sigma > 0.0) & (warp_result.dose_sigma > 0.0)
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        warp_result.dose,
        warp_result.dose_sigma,
        warp_result.n_batches,
        mask=mask,
    )


def test_ledgers_close_on_both_backends() -> None:
    """The emitted book must sum weight x energy, not energy — the weighting seam."""
    pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    ref = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG()).run(
        _source(), n_histories=500, n_batches=2, seed=SEED
    )
    assert ref.energy_emitted == pytest.approx(ref.energy_deposited + ref.energy_escaped, rel=1e-9)
    warp_result = WarpEngine(grid=grid, cross_sections=xs, device="cpu").run(
        _source(), n_histories=2_000, n_batches=2, seed=SEED
    )
    assert warp_result.energy_emitted == pytest.approx(
        warp_result.energy_deposited + warp_result.energy_escaped, rel=1e-4
    )


def test_a_narrow_fluence_table_collimates_the_field() -> None:
    """psi = 0 beyond 2 cm at the reference distance must shrink the irradiated width.

    Compared against a table flat over the whole rectangle: same geometry, same
    stream, only the weights differ, so the dose-weighted lateral spread is a
    direct read-out of the fluence shape.
    """
    grid = _grid()
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine = ReferenceEngine(grid=grid, cross_sections=xs, rng=HostRNG())
    kwargs = dict(n_histories=4_000, n_batches=4, seed=SEED, transport_electrons=False)

    wide = engine.run(_source(RadialFluence((0.0, 20.0), (1.0, 1.0))), **kwargs)
    narrow = engine.run(_source(RadialFluence((0.0, 2.0, 2.001), (1.0, 1.0, 0.0))), **kwargs)

    x = (np.arange(grid.shape[0]) + 0.5) * grid.spacing[0] - FOCAL[0]

    def spread(dose: np.ndarray) -> float:
        """Dose-weighted RMS off-axis distance, over the whole phantom."""
        profile = dose.sum(axis=(1, 2))
        return float(np.sqrt((profile * x**2).sum() / profile.sum()))

    assert spread(narrow.dose) < 0.75 * spread(wide.dose)
    # ... and it is a collimation, not a global scaling: total dose falls too.
    assert narrow.dose.sum() < wide.dose.sum()
