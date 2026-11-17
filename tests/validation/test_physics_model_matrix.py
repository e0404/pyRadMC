"""Reference against Warp for every combination of the three physics options.

Two bremsstrahlung, two pair and two positron models make eight combinations, each
its own pair of compiled kernels. The fast tier closes the energy books of all
eight on every target and compares two of them against the reference on a
positron beam (``tests/integration/test_pair_positron_cross_backend.py``); this
file compares all eight, on a source that reaches every arm at once: a third each
of 18 MeV photons, 12 MeV electrons and 12 MeV positrons down one lateral column
(``MixedPencil``).

What the comparison is worth, measured (2026-11-14, 3000 histories a side, 13
depth bins above half the maximum): the mixed source makes the batched
chi-squared more anticonservative than a single-kind pencil, because the number
of charged primaries in a batch fluctuates and moves every bin together. From
same-backend pairs of independent CUDA runs (500 for the first and last
combination, 100 for the others) the statistic has the right mean, 0.98 to 1.04,
but 8 to 11 % of pairs fall below a nominal p of 0.01 (4 to 5 % on the positron
pencil), with a 99 % quantile of 2.7 to 3.8 times the null mean. Reference against
Warp at these seeds: 0.47 to 0.76 times the null mean for the eight combinations
on CPU and CUDA (nominal p 0.72 to 0.90, empirical p 0.48 to 0.78; the eight are
not independent draws: they share their seeds). So a pass here says the eight
kernels are not grossly different from the reference loop; it
does not resolve the difference between two models of one
process, which at this size is below the noise for bremsstrahlung and pair
production (measured in the same document).
"""

from __future__ import annotations

import pytest

from pyradmc import PhysicsOptions
from tests.conftest import SEED
from tests.integration.test_pair_positron_cross_backend import (
    COMBINATIONS,
    MixedPencil,
    _assert_consistent,
    _name,
    _ref_engine,
    _run,
    _warp_engine,
)

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = [pytest.mark.validation, pytest.mark.warp]

N_HISTORIES = 3_000


@pytest.mark.parametrize("physics", COMBINATIONS, ids=_name)
def test_every_combination_agrees_with_the_reference(physics: PhysicsOptions) -> None:
    ref = _run(_ref_engine(), MixedPencil(), physics, N_HISTORIES, SEED)
    devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
    for device in devices:
        other = _run(_warp_engine(device), MixedPencil(), physics, N_HISTORIES, SEED + 1)
        assert other.provenance is not None
        assert _name(other.provenance.physics) == _name(physics)
        _assert_consistent(ref, other, min_bins=12)
