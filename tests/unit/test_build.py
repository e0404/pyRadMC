"""Build-tool path and cache logic (Phase 5, tabulated, slice C).

The actual download and full compile hit the network and the ~120 MB libraries, so
they live in the validation flow; here we pin only the offline behaviour: where a
library caches, and that an already-cached file is returned without a download.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

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


def test_main_compiles_the_whole_registry_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI routes through ``compile_materials`` (BLD slice 0: tungsten row).

    Default is the whole registry (``n_materials=None``) so a compiled table can
    serve CT media and the beam-limiting devices at once; ``--n-materials`` narrows
    it to a registry prefix (1 reproduces the historical water-only build).
    """
    from types import SimpleNamespace

    (tmp_path / "EPDL2023.ALL").write_text("epdl text")
    (tmp_path / "EEDL2023.ALL").write_text("eedl text")
    calls: dict[str, object] = {}

    def fake_compile(epdl_text: str, eedl_text: str, **kwargs: object) -> object:
        calls["n_materials"] = kwargs.get("n_materials", "missing")
        calls["texts"] = (epdl_text, eedl_text)
        return SimpleNamespace(provenance="fake")

    monkeypatch.setattr(build, "compile_materials", fake_compile)
    monkeypatch.setattr(build, "save_tables", lambda data, path, dtype: Path(path))

    common = ["--output", str(tmp_path / "out.npz"), "--cache-dir", str(tmp_path)]
    build.main(common)
    assert calls["n_materials"] is None  # whole registry
    assert calls["texts"] == ("epdl text", "eedl text")

    build.main([*common, "--n-materials", "1"])
    assert calls["n_materials"] == 1  # the historical water-only prefix


def test_download_progress_goes_through_logging_not_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Library output policy: progress is a logger INFO record, never a print.

    The consumer (pyRadPlan, a notebook) controls verbosity via the standard
    ``logging`` hierarchy, so an un-configured import must stay silent on stdout.
    """
    monkeypatch.setattr(
        build.urllib.request, "urlopen", lambda request: io.BytesIO(b"library bytes")
    )
    with caplog.at_level(logging.INFO, logger="pyRadMC"):
        result = build.download_library("epdl", tmp_path, force=True)
    assert result.read_bytes() == b"library bytes"
    assert any(
        record.levelno == logging.INFO and "downloading" in record.getMessage()
        for record in caplog.records
    )
    assert capsys.readouterr().out == ""
