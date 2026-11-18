"""Reference against Warp with the pair and positron models forced.

AGENTS.md 2.3: chi-squared, never equality. As in ``test_bethe_heitler_warp.py``,
the comparison is set up so that the oracle's null is measured before its verdict
is trusted: equal history counts on both sides, a pencil beam scored in one
lateral column with depth bins above half the maximum (no sparse bins, no
primary-fluence fluctuation to correlate them), and mutants the oracle must
reject. The numbers quoted in the docstrings below were measured on this tree
(2026-11-14); the scripts are not part of the repository.

Both backends run the Gaussian hinge: the reference backend builds its
Goudsmit-Saunderson tables lazily, which costs minutes, and neither model's arm
depends on the angular law.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest
from scipy import stats

import pyradmc.data.bethe_heitler as bethe_heitler_data
import pyradmc.data.positron as positron_data
from pyradmc import PhysicsOptions
from pyradmc.backends.ref.engine import ReferenceEngine
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.data.bethe_heitler import BetheHeitlerTables, build_bethe_heitler_tables
from pyradmc.data.interface import CrossSectionSource
from pyradmc.data.positron import PositronTables, build_positron_tables
from pyradmc.geometry.grid import VoxelGrid
from pyradmc.geometry.source import BeamletGridSource, Primary, Source
from pyradmc.rng import RNGState, uniform
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED, assert_chi2_consistent_batched

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

COLUMN_SHAPE = (1, 1, 24)
COLUMN_SPACING = (10.0, 10.0, 0.5)
LUNG = slice(6, 10)  # 3 to 5 cm depth at lung-like density
N_BATCHES = 20

MODELS = {
    "bremsstrahlung": ("thin-target", "bethe-heitler"),
    "pair": ("uniform-split", "bethe-heitler"),
    "positron": ("moller-at-rest", "bhabha-in-flight"),
}
COMBINATIONS = [
    PhysicsOptions(bremsstrahlung=b, pair=p, positron=e)
    for b, p, e in itertools.product(*MODELS.values())
]


def _name(physics: PhysicsOptions) -> str:
    return f"{physics.bremsstrahlung}/{physics.pair}/{physics.positron}"


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return str(request.param)


def _column() -> VoxelGrid:
    density = np.ones(COLUMN_SHAPE, dtype=np.float64)
    density[:, :, LUNG] = 0.26
    return VoxelGrid(
        shape=COLUMN_SHAPE,
        spacing=COLUMN_SPACING,
        density=density,
        material=np.zeros(COLUMN_SHAPE, dtype=np.int32),
    )


def _xs(grid: VoxelGrid) -> AnalyticCrossSections:
    return AnalyticCrossSections(geometry_densities=grid.max_density_by_material())


def _ref_engine() -> ReferenceEngine:
    grid = _column()
    return ReferenceEngine(grid=grid, cross_sections=_xs(grid), rng=HostRNG())


def _warp_engine(device: str = "cpu"):  # type: ignore[no-untyped-def]
    from pyradmc.backends.warp.engine import WarpEngine

    grid = _column()
    return WarpEngine(grid=grid, cross_sections=_xs(grid), device=device)


class PositronPencil(Source):
    """12 MeV positrons along +z down the axis of the column."""

    particle_kinds = frozenset({"positron"})

    @property
    def max_energy(self) -> float:
        return 12.0

    def emit(self, rng_state: RNGState) -> Primary:
        return Primary(12.0, 5.0, 5.0, -0.5, 0.0, 0.0, 1.0, kind="positron")


class MixedPencil(Source):
    """A third each of 18 MeV photons, 12 MeV electrons and 12 MeV positrons.

    One source that reaches every model's arm: the photons make pairs, the
    charged primaries radiate photons that make more, and positrons come both
    from the source and from the pairs. The charged two thirds also keep every
    depth bin populated by every batch.
    """

    particle_kinds = frozenset({"photon", "electron", "positron"})

    @property
    def max_energy(self) -> float:
        return 18.0

    def emit(self, rng_state: RNGState) -> Primary:
        u = uniform(rng_state)
        if u < 1.0 / 3.0:
            return Primary(18.0, 5.0, 5.0, -0.5, 0.0, 0.0, 1.0, kind="photon")
        kind = "electron" if u < 2.0 / 3.0 else "positron"
        return Primary(12.0, 5.0, 5.0, -0.5, 0.0, 0.0, 1.0, kind=kind)


def _run(engine, source: Source, physics: PhysicsOptions, n: int, seed: int):  # type: ignore[no-untyped-def]
    return engine.run(
        source, n_histories=n, n_batches=N_BATCHES, seed=seed, msc_model="gaussian", physics=physics
    )


def comparison_mask(a, b) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Depth bins above half the maximum of ``a`` with a batch sigma on both sides."""
    mask = a.dose > 0.5 * a.dose.max()
    mask &= (a.dose_sigma > 0.0) & (b.dose_sigma > 0.0)
    return np.asarray(mask)


def chi2_over_null_mean(a, b) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    """The batched statistic over its null mean, and its nominal p-value.

    The arithmetic of :func:`tests.conftest.assert_chi2_consistent_batched`,
    returned instead of asserted so that a null can be measured with it.
    """
    mask = comparison_mask(a, b)
    va = a.dose_sigma[mask] ** 2
    vb = b.dose_sigma[mask] ** 2
    nu = (va + vb) ** 2 / (va**2 / (a.n_batches - 1) + vb**2 / (b.n_batches - 1))
    chi2 = float(np.sum((a.dose[mask] - b.dose[mask]) ** 2 / (va + vb)))
    mean = float(np.sum(nu / (nu - 2.0)))
    variance = float(np.sum(2.0 * nu**2 * (nu - 1.0) / ((nu - 2.0) ** 2 * (nu - 4.0))))
    return chi2 / mean, float(stats.norm.sf((chi2 - mean) / np.sqrt(variance)))


def _assert_consistent(ref, other, min_bins: int) -> None:  # type: ignore[no-untyped-def]
    mask = comparison_mask(ref, other)
    assert np.count_nonzero(mask) >= min_bins
    # No sparse bins: the oracle has no power where the batch means are not normal.
    assert np.all(ref.dose_sigma[mask] < 0.05 * ref.dose[mask])
    assert_chi2_consistent_batched(
        ref.dose,
        ref.dose_sigma,
        ref.n_batches,
        other.dose,
        other.dose_sigma,
        other.n_batches,
        mask=mask,
    )


# -- the positron model on a positron beam: agreement, and the oracle's power ---------

N_POSITRONS = 3_000
IN_FLIGHT = PhysicsOptions(positron="bhabha-in-flight")
HIGH_ENERGY = PhysicsOptions(
    bremsstrahlung="bethe-heitler", pair="bethe-heitler", positron="bhabha-in-flight"
)


@pytest.fixture(scope="module", params=[IN_FLIGHT, HIGH_ENERGY], ids=_name)
def ref_positrons(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    """One reference run per model set, shared by the device comparisons and the mutants."""
    physics = request.param
    return physics, _run(_ref_engine(), PositronPencil(), physics, N_POSITRONS, SEED)


def test_positron_dose_agrees_across_backends(device: str, ref_positrons) -> None:  # type: ignore[no-untyped-def]
    """Reference and Warp agree under the model, at equal history counts.

    12 MeV positrons, 13 depth bins above half the maximum, each carrying under
    3 percent. The null of this comparison, from 1000 same-backend pairs
    (independent CUDA runs at this size, geometry and mask), has the right mean
    (chi-squared over its null mean 1.011, sd 0.48) and the heavier upper tail
    the bremsstrahlung test found: 5.3 % of pairs fall below a nominal p of 0.01
    and 9.5 % below 0.05; the 95 % and 99 % quantiles of the ratio are 1.97 and
    2.41, the largest of the 1000 is 4.2. Reference against Warp sits at 0.67 on
    CPU and CUDA with only the positron model changed (nominal p = 0.79,
    empirical p = 0.72). The second parameter is all three detailed models
    together, which is what ``"auto"`` runs above the threshold: null mean 0.987,
    4.1 % below a nominal 0.01, 99 % quantile 2.36, largest 3.5; reference against
    Warp 1.07 and 1.08 (nominal p = 0.43, empirical p = 0.35).

    What the comparison is blind to, measured with data-level mutants on the Warp
    side: the electron's range map in place of the positron's leaves the statistic
    at 0.65 and 1.27 (the two restricted ranges differ by a percent and the mask
    keeps the bins above half the maximum), and the electron's Moller rate and
    range map together give 2.2 and 3.5, above 98 % and 99.9 % of the null pairs
    but not cleanly beyond them. So this size of dose comparison does not certify
    the positron's collision data on the device; that is pinned node by node
    against the interface (``test_positron_in_flight_warp.py::TestTables``) and
    through the reference loop's delta-ray rate, spectrum and stopping depth.
    """
    physics, ref = ref_positrons
    other = _run(_warp_engine(device), PositronPencil(), physics, N_POSITRONS, SEED + 1)
    assert other.provenance is not None
    assert _name(other.provenance.physics).endswith("bhabha-in-flight")
    _assert_consistent(ref, other, min_bins=12)


def _no_annihilation(
    source: CrossSectionSource, ecut: float, e_max: float, n_points: int | None = None
) -> PositronTables:
    """Mutant: no annihilation channel; every positron stops and annihilates at rest."""
    tables = build_positron_tables(source, ecut, e_max, n_points)
    return dataclasses.replace(tables, annihilation=np.zeros_like(tables.annihilation))


def test_the_oracle_rejects_a_dropped_annihilation_channel(  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch, ref_positrons
) -> None:
    """Power: the comparison fails when the kernel's positrons never annihilate in flight.

    The mutant replaces the tables the engine uploads; the kernel and the
    reference run are untouched, and energy stays conserved (the books close), so
    only the dose comparison can see it. Measured: 7.9 and 8.6 times the null
    mean for the two model sets, beyond every one of the 1000 null pairs.
    """
    physics, ref = ref_positrons
    monkeypatch.setattr(positron_data, "build_positron_tables", _no_annihilation)
    broken = _run(_warp_engine(), PositronPencil(), physics, N_POSITRONS, SEED + 1)

    balance = broken.energy_deposited + broken.energy_escaped
    assert balance == pytest.approx(broken.energy_emitted, rel=1e-6)
    with pytest.raises(AssertionError, match="not statistically consistent"):
        _assert_consistent(ref, broken, min_bins=12)


def test_the_oracle_rejects_the_clinical_model_in_the_models_place(ref_positrons) -> None:  # type: ignore[no-untyped-def]
    """Power: an electron-like positron on the Warp side is 20 and 18 times the null mean."""
    physics, ref = ref_positrons
    clinical = dataclasses.replace(physics, positron="moller-at-rest")
    other = _run(_warp_engine(), PositronPencil(), clinical, N_POSITRONS, SEED + 1)
    with pytest.raises(AssertionError, match="not statistically consistent"):
        _assert_consistent(ref, other, min_bins=12)


def _no_discrete_bremsstrahlung(
    source: CrossSectionSource, ecut: float, pcut: float, e_max: float, n_points: int | None = None
) -> BetheHeitlerTables:
    """Mutant: no photon above PCUT is ever emitted (as in ``test_bethe_heitler_warp.py``)."""
    tables = build_bethe_heitler_tables(source, ecut, pcut, e_max, n_points)
    return dataclasses.replace(tables, cross_section=np.zeros_like(tables.cross_section))


def test_the_oracle_rejects_a_positron_that_never_radiates(  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch, ref_positrons
) -> None:
    """Power, for the kernel that carries both discrete arms: the three-way channel pick.

    Under ``bethe-heitler`` bremsstrahlung *and* ``bhabha-in-flight`` a positron's
    one uniform chooses among annihilation, a photon and Bhabha. A pick that never
    chose the photon would look exactly like this mutant, whose bremsstrahlung
    cross-section is zero: 7.0 times the null mean, beyond every null pair (and
    67 with the total radiative stopping power charged on top of the photons).
    """
    physics, ref = ref_positrons
    if physics.bremsstrahlung != "bethe-heitler":
        pytest.skip("the thin-target model has no discrete bremsstrahlung channel")
    monkeypatch.setattr(
        bethe_heitler_data, "build_bethe_heitler_tables", _no_discrete_bremsstrahlung
    )
    broken = _run(_warp_engine(), PositronPencil(), physics, N_POSITRONS, SEED + 1)
    with pytest.raises(AssertionError, match="not statistically consistent"):
        _assert_consistent(ref, broken, min_bins=12)


# -- every combination of the three options -------------------------------------------


@pytest.mark.parametrize("physics", COMBINATIONS, ids=_name)
class TestEveryCombination:
    """2 x 2 x 2 models, on a source that reaches every arm (see ``MixedPencil``).

    Books here; the dose comparison of all eight against the reference, which
    costs eight reference runs, is
    ``tests/validation/test_physics_model_matrix.py``.
    """

    def test_reference_books_close(self, physics: PhysicsOptions) -> None:
        result = _run(_ref_engine(), MixedPencil(), physics, 120, SEED)
        assert result.provenance is not None
        assert _name(result.provenance.physics) == _name(physics)
        balance = result.energy_deposited + result.energy_escaped
        assert balance == pytest.approx(result.energy_emitted, rel=1e-12)

    def test_warp_books_close(self, device: str, physics: PhysicsOptions) -> None:
        result = _run(_warp_engine(device), MixedPencil(), physics, 6_000, SEED)
        assert result.provenance is not None
        assert _name(result.provenance.physics) == _name(physics)
        balance = result.energy_deposited + result.energy_escaped
        assert balance == pytest.approx(result.energy_emitted, rel=1e-6)

    def test_warp_dij_books_close(self, device: str, physics: PhysicsOptions) -> None:
        beamlets = BeamletGridSource(
            energy=18.0, z=-0.5, x_range=(3.0, 7.0), y_range=(4.0, 6.0), n_x=2, n_y=1
        )
        engine = _warp_engine(device)
        common = dict(n_histories_per_beamlet=1_000, n_batches=4, seed=SEED, physics=physics)
        host = engine.run_dij(beamlets, **common)
        on_device = engine.run_dij_device(beamlets, **common).to_host()
        balance = host.energy_deposited + host.energy_escaped
        assert balance == pytest.approx(host.energy_emitted, rel=1e-6)
        assert (host.dose_csc() != on_device.dose_csc()).nnz == 0
