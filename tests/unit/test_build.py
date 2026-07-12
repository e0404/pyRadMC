"""Build-tool path and cache logic (Phase 5, tabulated, slice C).

The actual download and full compile hit the network and the ~120 MB libraries, so
they live in the validation flow; here we pin only the offline behaviour: where a
library caches, and that an already-cached file is returned without a download.
"""

from __future__ import annotations

from pathlib import Path

from pyRadMC.data.tabulated import build


def test_library_path_uses_the_cache_dir(tmp_path: Path) -> None:
    assert build.library_path("epdl", tmp_path) == tmp_path / "EPDL2023.ALL"
    assert build.library_path("eedl", tmp_path) == tmp_path / "EEDL2023.ALL"


def test_download_returns_cached_file_without_network(tmp_path: Path) -> None:
    """An existing cache entry is returned as-is (no download attempted)."""
    cached = tmp_path / "EPDL2023.ALL"
    cached.write_text("already here")
    # If this reached the network it would fail offline; it must short-circuit instead.
    result = build.download_library("epdl", tmp_path)
    assert result == cached
    assert result.read_text() == "already here"
