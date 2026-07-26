"""The top-level ``pyRadMC`` namespace is the released API contract.

Three properties are pinned here, because each breaks silently otherwise:

1. **Every name in ``__all__`` resolves.** The lazy re-export map is a dict of
   strings, so a typo or a moved class is invisible until a user hits it.
2. **``__all__`` and the lazy map agree.** Adding one without the other either
   hides a name from ``dir()``/``from pyRadMC import *`` or promises a name that
   cannot be imported.
3. **``import pyRadMC`` stays cheap and dependency-free.** The constants must be
   readable without NumPy, and a core-only install (no ``warp`` extra) must not
   fail at import merely because ``WarpEngine`` is a public name. That is the
   whole reason the re-exports are PEP 562 lazy rather than plain imports.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import pyRadMC

# Names that are module-level values here, not lazy re-exports from elsewhere.
_EAGER = {
    "DIJ_TRUNCATION_RELATIVE",
    "ECUT_MEV",
    "ELECTRON_MASS_MEV",
    "GY_PER_MEV_PER_G",
    "PCUT_MEV",
    "PHOTON_ROULETTE_MEV",
    "PHOTON_ROULETTE_SURVIVAL",
    "PHOTON_ROULETTE_WEIGHT_CAP",
    "PHOTON_SPLIT_N",
    "RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV",
    "__version__",
}


@pytest.mark.parametrize("name", sorted(pyRadMC.__all__))
def test_every_public_name_resolves(name: str) -> None:
    """Each exported name imports and is not None."""
    if name == "WarpEngine":
        pytest.importorskip("warp", reason="warp-lang optional dependency not installed")
    assert getattr(pyRadMC, name) is not None


def test_all_matches_the_lazy_map() -> None:
    """``__all__`` is exactly the eager values plus the lazily re-exported names."""
    assert set(pyRadMC.__all__) == _EAGER | set(pyRadMC._LAZY_EXPORTS)


def test_all_has_no_duplicates() -> None:
    """Ordering is ruff's job (RUF022); uniqueness is not checked by anything else."""
    assert len(pyRadMC.__all__) == len(set(pyRadMC.__all__))


def test_dir_reports_the_public_api() -> None:
    """Tab completion must see names that have not been imported yet."""
    assert sorted(pyRadMC.__all__) == dir(pyRadMC)


def test_unknown_attribute_raises_attribute_error() -> None:
    """The lazy hook must not turn a typo into an ImportError or a KeyError."""
    with pytest.raises(AttributeError, match="no attribute 'NotAThing'"):
        pyRadMC.NotAThing  # noqa: B018


def test_py_typed_marker_ships() -> None:
    """PEP 561: without this file, downstream type checkers ignore our annotations."""
    assert (Path(pyRadMC.__file__).parent / "py.typed").is_file()


_IMPORT_IS_CHEAP = """
import sys

import pyRadMC

assert pyRadMC.ECUT_MEV > 0.0, "constants must be readable with no dependencies"

eager = sorted(
    name
    for name in ("numpy", "scipy", "warp", "SimpleITK", "matplotlib")
    if name in sys.modules
)
assert not eager, f"import pyRadMC eagerly imported: {eager}"
"""


_MISSING_WARP = """
import sys

sys.modules["warp"] = None  # any 'import warp' below now raises ImportError(name='warp')

import pyRadMC

try:
    pyRadMC.WarpEngine
except ImportError as exc:
    assert "pyRadMC[warp]" in str(exc), f"unhelpful message: {exc}"
else:
    raise AssertionError("WarpEngine resolved with warp unavailable")
"""


def test_missing_warp_names_the_extra() -> None:
    """A core-only install must be told which extra to install, not just 'No module'.

    Runs in a subprocess because it poisons ``sys.modules`` to simulate the
    core-only install; warp is installed in the dev environment.
    """
    result = subprocess.run(
        [sys.executable, "-c", _MISSING_WARP],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bare_import_pulls_no_third_party_module() -> None:
    """``import pyRadMC`` must not drag in NumPy, SciPy, Warp or SimpleITK.

    Runs in a subprocess: the suite has long since imported all of these, so an
    in-process check would pass no matter how the re-exports were written.
    """
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_IS_CHEAP],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
