"""The Fano source must sample the same histories regardless of chunk size."""

import numpy as np

from tests.conftest import SEED
from tests.fano_fixtures import FanoElectronSource, fano_phantom


def test_fano_presampling_is_chunk_invariant() -> None:
    grid, *_ = fano_phantom(0.2, 1.0e-3)
    source = FanoElectronSource(grid, 1.0)
    whole = source.sample_batch(SEED, 37, 101)
    parts = [source.sample_batch(SEED, 37, 29), source.sample_batch(SEED, 66, 72)]
    for name, column in whole.items():
        np.testing.assert_array_equal(column, np.concatenate([p[name] for p in parts]))
