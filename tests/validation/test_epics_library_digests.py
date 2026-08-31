"""Validation tier: the pinned EPICS digests are the libraries we validated against.

:data:`~pyradmc.data.tabulated.build.LIBRARY_SHA256` is the only thing standing
between a silently altered cross-section file and every dose this engine computes.
The unit tier exercises the *mechanism* (mismatch raises, a bad transfer never
becomes a cache entry) against synthetic bytes; only this tier can check the pins
themselves, because doing so requires the real ~120 MB libraries.

Two failure modes this catches:

- **The pin was edited without re-validating.** Changing a digest changes the
  cross-sections. This test fails unless the edited value matches a library actually
  present, which forces the rest of the validation tier to be run against it.
- **The IAEA revised a library in place.** The other validation tests would then
  shift by small amounts and might still pass their tolerances; this one names the
  cause instead of leaving a mystery drift.

Libraries are located exactly as the rest of the tier does: ``PYRADMC_EPDL_PATH`` /
``PYRADMC_EEDL_PATH``, falling back to the build tool's own cache directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pyradmc.data.tabulated import build

pytestmark = pytest.mark.validation

_ENV_VAR = {"epdl": "PYRADMC_EPDL_PATH", "eedl": "PYRADMC_EEDL_PATH"}

# Byte sizes of the pinned libraries, recorded alongside the digests. Cheap to check
# first: a truncated file is the common corruption, and the size names it immediately
# instead of reporting an opaque digest mismatch.
_EXPECTED_SIZE = {"epdl": 91_508_720, "eedl": 27_576_928}


def _library(which: str) -> Path:
    raw = os.environ.get(_ENV_VAR[which])
    candidate = Path(raw) if raw else build.library_path(which)
    if not candidate.is_file():
        pytest.skip(
            f"set {_ENV_VAR[which]} to a local {build.LIBRARY_FILES[which]}, or populate "
            f"{build.DEFAULT_CACHE_DIR}, to run this validation"
        )
    return candidate


@pytest.mark.parametrize("which", ["epdl", "eedl"])
def test_library_matches_its_pinned_size(which: str) -> None:
    """Truncation is the likeliest corruption; report it as such."""
    assert _library(which).stat().st_size == _EXPECTED_SIZE[which]


@pytest.mark.parametrize("which", ["epdl", "eedl"])
def test_library_matches_its_pinned_digest(which: str) -> None:
    """The real library hashes to the value this release's tables were built from."""
    assert build._file_digest(_library(which)) == build.LIBRARY_SHA256[which]


@pytest.mark.parametrize("which", ["epdl", "eedl"])
def test_download_library_accepts_the_real_cache_entry(which: str) -> None:
    """End to end through the shipping code path, verification enabled.

    The digest tests above could pass while ``download_library`` still rejected the
    file — a wrong ``which`` key, a cache-hit branch that skips verification — so the
    public entry point is exercised too.
    """
    library = _library(which)
    returned = build.download_library(which, library.parent)
    assert returned == library
