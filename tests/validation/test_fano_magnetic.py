"""Fano cavity test in a static magnetic field, and field-free down to air contrast.

Fano's theorem fails in a magnetic field in general — the Lorentz force in
mass-thickness coordinates depends on the density, so the charged-particle fluence
is no longer density independent. Bouchard, de Pooter, Bielajew and Duane (2015,
PMB 60, 6639, doi:10.1088/0031-9155/60/17/6639) prove the one condition under which
it still holds for *any* field: a spatially uniform **and isotropic** charged-particle
source, per unit mass, in a medium of uniform atomic properties. Under it the fluence
is uniform and isotropic for any B, and the dose per unit mass is the same everywhere,
field or not. The recipe is the one Malkov & Rogers (2016, Med. Phys. 43, 4447,
doi:10.1118/1.4954318, sec. 2.G) used to validate EGSnrc: two dense walls with a gap
of the same material at a thousandth of the density, the field parallel to the
interfaces (the harder orientation), monoenergetic electrons emitted isotropically and
uniformly per unit mass, and the dose per unit mass compared between gap and walls.
Their cylinder is sized to the electron range *in the gas*; here the light slab is a
cavity embedded laterally in water instead (see ``fano_phantom``), which keeps the
statistics on a small grid.

This is the test that bites where the stepping approximations do: electrons entering
the gap gyrate back on a radius far below the gap's electron range, the gyration cap
binds on every substep there, and any bias in the chord geometry, the step cap or the
boundary handling shows up as a gap-to-wall dose ratio off unity (Simiele & DeWerd 2018,
PMB 63, 235012, found 7-11 % under-response in exactly this configuration before fixing
Geant4's step limits).

**Three gates, because two things are under test.**

1. *The field adds no violation*: the gap-to-wall ratio with the field on is within 1 %
   of the field-free ratio. Measured 2026-10-18 at 0.35 and 1.5 T: the field moves it by
   at most 0.4 % at a tenth and a hundredth of the wall density and by 0.6-0.8 % in a
   2 cm air-density gap. The one exception is the 0.2 cm air gap, whose *field-free*
   residual (gate 2) is itself a step-length effect: the field shortens the substeps in
   air and the residual shrinks (0.994 / 0.996 at 1.5 T against 0.979 / 0.975 at 0 T),
   so the "movement" reads up to +2 %. That case is held to the field-free gate instead,
   and gate 3 pins it once the substep is converged.
2. *The field-free engine at the shipped GS substep fraction* (0.20) is within 1 % of unity
   at a tenth of the wall density and within 3 % down to air density. The residual is a
   thin-gap, step-length effect of the random hinge at an interface, measured after the
   boundary crossing was made exact: 0.2 cm gap -1.7 to -2.5 % at air density, -0.9 % at
   a hundredth, +0.9 % at a tenth; a 2 cm gap is within 0.4 % at every contrast; and it
   halves at a substep fraction of 0.03 (-1.1 / -0.9 %) and is within 1 % at 0.02
   (-0.2 / -0.6 %). Before the fix the same engine read 2.9x at air contrast, +7.9 % at a
   hundredth and +4.7 % at air even with 0.02 steps. (The baselines were taken at the
   shipped GS fraction; an earlier draft labelled it 0.05, the Gaussian hinge's default.)
3. *Convergence*: at a substep fraction of 0.02 the 0.2 cm air-density gap is within
   1.5 % of unity with and without the field, and the field moves it by less than 1.5 %
   (measured -0.2 / -0.6 % at 0 T and -0.8 / -0.4 % at 1.5 T, 0.5 / 1 MeV, +- 0.3 %).
   Together with gate 2 this pins the residual as step-size convergence rather than a
   bookkeeping error, and it is the number to re-check if the shipped fraction is ever
   changed (AGENTS.md section 8).

**Which path.** These cavities are sealed in water, so they are not exterior air and the
shipped default leaves them on the current stepping with no opt-out needed; the gates above
are gates of that stepping, and it is pinned explicitly. Gates 1, 2 and 3 are also run, with the
same numbers, on the long-step path by declaring the 2 cm air gap exterior. The 0.2 cm gap
cannot take long steps at all (no voxel has the minimum safety), and the 0.1 and 0.01
densities are not exterior air.

**Test instrument: ECUT 0.05 MeV.** Fano holds for any cutoff, but the dose in a gas
voxel is carried to 40 % by 0.2 MeV terminations and so is nearly all of its variance;
at 0.05 MeV the gap sigma is 3.5 times smaller for the same histories, which is what
makes a 1 % gate at air contrast affordable (60 million histories per field instead of
seven hundred). The bremsstrahlung leak that kept this gate at 1 % rather than EGSnrc's
0.1 % — photons are transported analogously and leave the phantom, lowering every
region's dose by a fraction of the radiative yield — still stands; tightening it needs
a photon-killing instrument (AGENTS.md 2.10), a maintainer decision.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.backends.results import TransportResult
from pyradmc.data.analytic import AnalyticCrossSections
from pyradmc.geometry.exterior import EXTERIOR_AIR_DENSITY_MAX, ExteriorAirMap
from pyradmc.geometry.field import MagneticField
from tests.conftest import SEED
from tests.fano_fixtures import FanoElectronSource, fano_phantom

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.validation, pytest.mark.warp]

FIELD_GATE = 0.01  # |ratio(B) / ratio(0) - 1|
FIELD_FREE_GATE = {0.1: 0.01, 0.01: 0.03, 1.0e-3: 0.03}  # |ratio(0) - 1| at the shipped step
THIN_GAP_CM = 0.2  # the field shortens this gap's substeps and shrinks its field-free residual
CONVERGED_GATE = 0.015  # |ratio - 1| and the field's movement at the converged substep
CONVERGED_STEP_FRACTION = 0.02
CONVERGED_HISTORIES = 100_000_000
AIR_LIKE = 1.0e-3
ECUT_INSTRUMENT = 0.05
# The light slab one voxel in from its water surround (margin 1 cm = 2 voxels of 0.5 cm):
# averaging the whole layer rather than the central 2x2 cm is what the gas statistics need.
CAVITY = slice(3, 13)


def _device() -> str:
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _region_dose(result: TransportResult, lateral: slice, depth: slice) -> tuple[float, float]:
    """Mean dose per unit mass over a region and its (quadrature) sigma."""
    d = result.dose[lateral, lateral, depth]
    s = result.dose_sigma[lateral, lateral, depth]
    return float(d.mean()), float(np.sqrt((s * s).sum()) / d.size)


def _gap_to_wall(
    gap_cm: float,
    gap_density: float,
    energy: float,
    field: MagneticField | None,
    step_energy_fraction: float | None = None,
    n_histories: int | None = None,
    long_steps: bool = False,
) -> tuple[float, float, float]:
    """(gap/wall ratio, its sigma, upstream/downstream wall ratio) on CUDA.

    Current stepping unless ``long_steps``: then the gap is declared exterior air, the only way
    to put these sealed cavities through the long-step path (see the module docstring).
    """
    from pyradmc.backends.warp.engine import WarpEngine

    grid, core, upstream, gap, downstream = fano_phantom(gap_cm, gap_density)
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    if long_steps and field is not None:
        air = ExteriorAirMap.from_region(grid, np.asarray(grid.density) < EXTERIOR_AIR_DENSITY_MAX)
        assert np.any(air.safety > 0.0), "this gap takes no long steps; the case tests nothing"
        with pytest.warns(UserWarning, match="exterior-air"):
            engine = WarpEngine(
                grid=grid,
                cross_sections=xs,
                device=_device(),
                magnetic_field=field,
                exterior_air=air,
            )
    else:
        engine = WarpEngine(
            grid=grid, cross_sections=xs, device=_device(), magnetic_field=field, exterior_air=None
        )
    if n_histories is None:
        n_histories = 40_000_000 if gap_density >= 0.01 else 60_000_000
    result = engine.run(
        FanoElectronSource(grid, energy),
        n_histories=n_histories,
        n_batches=10,
        seed=SEED,
        ecut=ECUT_INSTRUMENT,
        step_energy_fraction=step_energy_fraction,
    )
    wall_up, s_up = _region_dose(result, core, upstream)
    in_gap, s_gap = _region_dose(result, CAVITY, gap)
    wall_down, s_down = _region_dose(result, core, downstream)
    wall = 0.5 * (wall_up + wall_down)
    ratio = in_gap / wall
    sigma = ratio * np.hypot(s_gap / in_gap, 0.5 * np.hypot(s_up, s_down) / wall)
    return ratio, sigma, wall_up / wall_down


def _needs_cuda() -> None:
    if not _device().startswith("cuda"):
        pytest.skip("the Fano gates need ~1e8 electron histories per case; CUDA only")


@pytest.mark.parametrize(
    ("gap_cm", "gap_density", "fields_tesla"),
    [
        (2.0, 0.1, (0.0, 0.35, 1.5)),
        (2.0, 0.01, (0.0, 1.5)),
        (0.2, AIR_LIKE, (0.0, 1.5)),
        (2.0, AIR_LIKE, (0.0, 1.5)),
    ],
    ids=["2cm-rho0.1", "2cm-rho0.01", "0.2cm-air", "2cm-air"],
)
@pytest.mark.parametrize("energy", [0.5, 1.0])
def test_dose_per_mass_is_uniform_across_the_gap(
    gap_cm: float, gap_density: float, fields_tesla: tuple[float, ...], energy: float
) -> None:
    """Gates 1 and 2: the field moves the ratio by < 1 %; field-free within its gate."""
    _gates_1_and_2(gap_cm, gap_density, fields_tesla, energy, long_steps=False)


@pytest.mark.parametrize("fields_tesla", [(0.0, 0.35), (0.0, 1.5)], ids=["0.35T", "1.5T"])
@pytest.mark.parametrize("energy", [0.5, 1.0])
def test_dose_per_mass_is_uniform_through_long_steps(
    fields_tesla: tuple[float, ...], energy: float
) -> None:
    """Gates 1 and 2, same numbers, the 2 cm air gap on the long-step path."""
    _gates_1_and_2(2.0, AIR_LIKE, fields_tesla, energy, long_steps=True)


def _gates_1_and_2(
    gap_cm: float,
    gap_density: float,
    fields_tesla: tuple[float, ...],
    energy: float,
    *,
    long_steps: bool,
) -> None:
    _needs_cuda()
    gate_0 = FIELD_FREE_GATE[gap_density]
    gate_b = gate_0 if gap_cm == THIN_GAP_CM and gap_density < 0.1 else FIELD_GATE
    report = []
    ratio_0 = None
    for b in fields_tesla:
        fld = None if b == 0.0 else MagneticField.uniform((b, 0.0, 0.0))
        ratio, sigma, walls = _gap_to_wall(gap_cm, gap_density, energy, fld, long_steps=long_steps)
        report.append(f"B={b} T: gap/wall={ratio:.4f} +- {sigma:.4f}, up/down walls={walls:.4f}")
        where = f"{report[-1]} (gap {gap_cm} cm, rho {gap_density}, {energy} MeV)"
        assert abs(walls - 1.0) < FIELD_GATE, f"walls differ: {where}"
        if b == 0.0:
            ratio_0 = ratio
            assert sigma < gate_0 / 3.0, f"statistics too poor to gate at {gate_0}: {where}"
            assert abs(ratio - 1.0) < gate_0, f"Fano violated without a field: {where}"
        else:
            assert ratio_0 is not None
            moved = ratio / ratio_0 - 1.0
            assert sigma < gate_b / 3.0, f"statistics too poor to gate at {gate_b}: {where}"
            assert abs(moved) < gate_b, f"the field moved the ratio by {moved:+.4f}: {where}"
    print("\n".join(report))


@pytest.mark.parametrize("energy", [0.5, 1.0])
def test_thin_air_gap_converges_with_the_substep(energy: float) -> None:
    """Gate 3: at a substep fraction of 0.02 the 0.2 cm air gap is within 1.5 % of Fano,
    with and without the field, and the field moves it by less than 1.5 %."""
    _gate_3(THIN_GAP_CM, energy, long_steps=False)


@pytest.mark.parametrize("energy", [0.5, 1.0])
def test_air_gap_converges_through_long_steps(energy: float) -> None:
    """Gate 3, same numbers, on the long-step path: a 2 cm gap, since 0.2 cm has no safety."""
    _gate_3(2.0, energy, long_steps=True)


def _gate_3(gap_cm: float, energy: float, *, long_steps: bool) -> None:
    _needs_cuda()
    ratios = {}
    for b in (0.0, 1.5):
        fld = None if b == 0.0 else MagneticField.uniform((b, 0.0, 0.0))
        ratio, sigma, walls = _gap_to_wall(
            gap_cm,
            AIR_LIKE,
            energy,
            fld,
            step_energy_fraction=CONVERGED_STEP_FRACTION,
            n_histories=CONVERGED_HISTORIES,
            long_steps=long_steps,
        )
        where = (
            f"B={b} T: gap/wall={ratio:.4f} +- {sigma:.4f}, "
            f"up/down walls={walls:.4f} ({energy} MeV)"
        )
        assert sigma < CONVERGED_GATE / 3.0, (
            f"statistics too poor to gate at {CONVERGED_GATE}: {where}"
        )
        assert abs(ratio - 1.0) < CONVERGED_GATE, (
            f"not converged at step {CONVERGED_STEP_FRACTION}: {where}"
        )
        assert abs(walls - 1.0) < FIELD_GATE, f"walls differ: {where}"
        ratios[b] = ratio
    moved = ratios[1.5] / ratios[0.0] - 1.0
    assert abs(moved) < CONVERGED_GATE, f"the field moved the converged ratio by {moved:+.4f}"
