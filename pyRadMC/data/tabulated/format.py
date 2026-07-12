"""The internal compiled-table format: a versioned NumPy ``.npz`` container.

Small by nature (a material is tens of kilobytes), so the goal is not compactness
but a fast, dependency-light load that leaves the runtime with ready arrays and no
physics to assemble. The float dtype is chosen at *compile* time — recompiling the
same source at reduced precision is one argument — while the loader is agnostic to
which dtype it finds. A JSON manifest carries the format version, provenance,
``delta_cut`` and the material/process lists needed to rebuild the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyRadMC.data.tabulated.model import TabulatedData

__all__ = ["FORMAT_VERSION", "load_tables", "save_tables"]

FORMAT_VERSION = 1

_ELECTRON_FIELDS = (
    "restricted_stopping",
    "radiative_stopping",
    "moller",
    "csda_range",
    "scattering_power",
)


def save_tables(data: TabulatedData, path: Path | str, *, dtype: type = np.float32) -> Path:
    """Write ``data`` to ``path`` as a versioned ``.npz``; return the path written.

    Every float array is stored at ``dtype``; the manifest records everything
    needed to reconstruct the :class:`TabulatedData` without pickling.
    """
    path = Path(path)
    processes = sorted(data.mu_over_rho)
    arrays: dict[str, np.ndarray] = {
        "photon_energies": data.photon_energies.astype(dtype),
        "electron_energies": data.electron_energies.astype(dtype),
    }
    for proc in processes:
        arrays[f"mu_{proc}"] = data.mu_over_rho[proc].astype(dtype)
    for field in _ELECTRON_FIELDS:
        arrays[field] = getattr(data, field).astype(dtype)
    manifest = {
        "version": FORMAT_VERSION,
        "provenance": data.provenance,
        "delta_cut": data.delta_cut,
        "materials": list(data.materials),
        "processes": processes,
    }
    arrays["_manifest"] = np.array(json.dumps(manifest))
    with path.open("wb") as handle:
        # numpy's savez stub mistypes the **kwargs of named arrays; the call is valid.
        np.savez(handle, **arrays)  # type: ignore[arg-type]
    return path


def load_tables(path: Path | str) -> TabulatedData:
    """Read a compiled-table file back into :class:`TabulatedData`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        manifest = json.loads(str(npz["_manifest"].item()))
        if manifest["version"] != FORMAT_VERSION:
            raise ValueError(
                f"unsupported compiled-table format version {manifest['version']}; "
                f"this build reads version {FORMAT_VERSION}. Recompile from source."
            )
        return TabulatedData(
            photon_energies=npz["photon_energies"],
            mu_over_rho={proc: npz[f"mu_{proc}"] for proc in manifest["processes"]},
            electron_energies=npz["electron_energies"],
            restricted_stopping=npz["restricted_stopping"],
            radiative_stopping=npz["radiative_stopping"],
            moller=npz["moller"],
            csda_range=npz["csda_range"],
            scattering_power=npz["scattering_power"],
            delta_cut=float(manifest["delta_cut"]),
            materials=tuple(manifest["materials"]),
            provenance=str(manifest["provenance"]),
        )
