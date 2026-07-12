"""Compiled cross-section format: round-trip and versioning (Phase 5, tabulated, slice A).

The precompiler writes authoritative source data (EPDL/ESTAR) into this internal
format; the loader reads it back with no knowledge of where it came from. This test
pins the round-trip and the version guard, independent of any real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyRadMC.data.interface import PhotonProcess
from pyRadMC.data.tabulated.format import FORMAT_VERSION, load_tables, save_tables
from pyRadMC.data.tabulated.model import TabulatedData

_PROCESSES = (
    PhotonProcess.COMPTON,
    PhotonProcess.PHOTOELECTRIC,
    PhotonProcess.PAIR,
    PhotonProcess.RAYLEIGH,
)


def synthetic_tables(n_mat: int = 1, n_p: int = 64, n_e: int = 48) -> TabulatedData:
    """A small, fully-populated table with reproducible arbitrary values."""
    rng = np.random.default_rng(20260712)
    return TabulatedData(
        photon_energies=np.geomspace(0.01, 10.0, n_p),
        mu_over_rho={p: rng.random((n_mat, n_p)) for p in _PROCESSES},
        electron_energies=np.geomspace(0.01, 10.0, n_e),
        restricted_stopping=rng.random((n_mat, n_e)),
        radiative_stopping=rng.random((n_mat, n_e)),
        moller=rng.random((n_mat, n_e)),
        csda_range=rng.random((n_mat, n_e)),
        scattering_power=rng.random((n_mat, n_e)),
        delta_cut=0.2,
        materials=("water",),
        provenance="synthetic test data",
    )


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    # float64 isolates container fidelity from the (separately tested) precision knob.
    data = synthetic_tables()
    back = load_tables(save_tables(data, tmp_path / "w", dtype=np.float64))

    assert back.provenance == data.provenance
    assert back.delta_cut == data.delta_cut
    assert back.materials == data.materials
    np.testing.assert_array_equal(back.photon_energies, data.photon_energies)
    np.testing.assert_array_equal(back.electron_energies, data.electron_energies)
    assert set(back.mu_over_rho) == set(data.mu_over_rho)
    for proc, arr in data.mu_over_rho.items():
        np.testing.assert_array_equal(back.mu_over_rho[proc], arr)
    for name in (
        "restricted_stopping",
        "radiative_stopping",
        "moller",
        "csda_range",
        "scattering_power",
    ):
        np.testing.assert_array_equal(getattr(back, name), getattr(data, name))


def test_dtype_is_selectable_at_compile_time(tmp_path: Path) -> None:
    """Recompiling the value tables at a reduced precision is a one-argument change.

    The dtype knob compresses the value arrays only; the energy grids stay float64
    (they are interpolation abscissae — see the format module docstring).
    """
    data = synthetic_tables()
    p = save_tables(data, tmp_path / "w16", dtype=np.float16)
    back = load_tables(p)
    assert back.mu_over_rho[PhotonProcess.COMPTON].dtype == np.float16
    assert back.photon_energies.dtype == np.float64
    assert back.electron_energies.dtype == np.float64
    # float16 round-trips the values to ~1e-3 relative — enough to prove the knob works.
    np.testing.assert_allclose(
        back.mu_over_rho[PhotonProcess.COMPTON],
        data.mu_over_rho[PhotonProcess.COMPTON],
        rtol=2e-3,
    )


def test_version_mismatch_is_rejected(tmp_path: Path) -> None:
    p = save_tables(synthetic_tables(), tmp_path / "w")
    # Corrupt the stored version.
    with np.load(p, allow_pickle=False) as npz:
        arrays = dict(npz.items())
    arrays["_manifest"] = np.array(
        arrays["_manifest"].item().replace(f'"version": {FORMAT_VERSION}', '"version": 999'),
    )
    with p.open("wb") as f:
        np.savez(f, **arrays)
    with pytest.raises(ValueError, match="format version"):
        load_tables(p)
