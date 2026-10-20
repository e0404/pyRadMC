"""The random hinge must not carry one voxel's energy loss into a lighter neighbour.

Each substep is capped at the next voxel face along the direction it *starts* with.
The hinge then deflects the particle, and the second chord can reach a face the cap
never saw. Before 2026-10-18 the chord simply crossed: the substep's energy loss —
computed with the start voxel's density — was filed along it into the neighbour, a
range-out termination landed there too, and the nudge that carries a particle across a
face was transported path charged at the old voxel's density. At CT contrast this was a
one-voxel interface blemish; at air contrast (a thousandfold) it dominated the light
voxel's dose: a 0.2 cm air-density gap between water walls received 2.9 times the
Fano-expected dose per unit mass with no field at all. Now a chord that reaches a face
stops on it, the particle is carried across by a nudge that the next substep charges at
the new voxel's density, and the chords split only the path inside the voxel.

The oracle is Fano's theorem: a uniform-per-mass isotropic electron source in a medium
of uniform atomic properties gives the same dose per unit mass everywhere, whatever the
density pattern. The light slab is water at air density (same cross-sections), embedded
laterally in water so that its long-range electrons are replaced rather than lost. ECUT
is lowered to 0.05 MeV as a test instrument (Fano holds for any cutoff; 0.2 MeV
terminations carry most of a gas voxel's variance), which makes a 5 % gate affordable
in the fast tier: the step-size residual of the hinge at this contrast and the shipped
substep fraction is -2.0 +- 0.5 % (the validation tier pins it and its convergence), the
old defect was +190 %.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.data.analytic import AnalyticCrossSections
from tests.conftest import SEED
from tests.fano_fixtures import FanoElectronSource, fano_phantom

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.warp, pytest.mark.gpu]

GAP_CM = 0.2
AIR_DENSITY = 1.205e-3
ENERGY = 0.5
ECUT_INSTRUMENT = 0.05
GATE = 0.05
# Dose in a 25 ug air-density voxel is a handful of terminations, so even the physical
# answer is heavy-tailed: the gate averages the whole cavity layer and needs tens of
# millions of histories. CUDA only; the CPU would take minutes.
N_HISTORIES = 20_000_000


def test_air_gap_receives_the_fano_dose_per_mass() -> None:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from pyradmc.backends.warp.engine import WarpEngine

    grid, core, upstream, gap, downstream = fano_phantom(GAP_CM, AIR_DENSITY)
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    engine = WarpEngine(grid=grid, cross_sections=xs, device="cuda:0")
    result = engine.run(
        FanoElectronSource(grid, ENERGY),
        n_histories=N_HISTORIES,
        n_batches=8,
        seed=SEED,
        ecut=ECUT_INSTRUMENT,
    )

    cavity = slice(3, 13)  # the light slab, one voxel in from its water surround
    in_gap = result.dose[cavity, cavity, gap]
    wall = 0.5 * (
        result.dose[core, core, upstream].mean() + result.dose[core, core, downstream].mean()
    )
    ratio = float(in_gap.mean() / wall)
    sigma = float(np.sqrt((result.dose_sigma[cavity, cavity, gap] ** 2).sum()) / in_gap.size / wall)
    assert abs(ratio - 1.0) < GATE, (
        f"air-density gap dose per mass is {ratio:.3f} +- {sigma:.3f} of the walls': "
        "the hinge is carrying wall energy loss across the face"
    )
    assert sigma < GATE / 3.0, f"statistics too poor to gate at {GATE}: {ratio:.3f} +- {sigma:.3f}"
