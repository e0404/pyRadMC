"""Build-tool path and cache logic.

The actual download and full compile hit the network and the ~120 MB libraries, so
they live in the validation flow; here we pin only the offline behaviour: where a
library caches, and that an already-cached file is returned without a download.
"""

from __future__ import annotations

import hashlib
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
    # verify=False because the fixture bytes are not the real library.
    result = build.download_library("epdl", tmp_path, verify=False)
    assert result == cached
    assert result.read_text() == "already here"


def _pin_digest(monkeypatch: pytest.MonkeyPatch, which: str, payload: bytes) -> None:
    """Pin ``which``'s expected digest to that of ``payload``."""
    monkeypatch.setitem(build.LIBRARY_SHA256, which, hashlib.sha256(payload).hexdigest())


def test_cached_library_is_rehashed_and_rejected_when_corrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache hit is not trusted: the file may have rotted since it was written.

    This is the case that matters most in practice — the download succeeded once,
    and nothing would ever look at those bytes again.
    """
    cached = tmp_path / "EPDL2023.ALL"
    cached.write_bytes(b"corrupted")
    _pin_digest(monkeypatch, "epdl", b"the real library")

    with pytest.raises(build.LibraryChecksumError, match="does not match the pinned SHA-256"):
        build.download_library("epdl", tmp_path)


def test_cached_library_passes_when_it_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The happy path still returns the cache entry untouched."""
    payload = b"the real library"
    cached = tmp_path / "EPDL2023.ALL"
    cached.write_bytes(payload)
    _pin_digest(monkeypatch, "epdl", payload)

    assert build.download_library("epdl", tmp_path).read_bytes() == payload


def test_bad_download_never_becomes_a_cache_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification happens before the rename, and the partial file is cleaned up.

    Otherwise a single corrupted transfer would poison the cache permanently: the
    next run would take the cache-hit branch and compile from the bad bytes.
    """
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda request: io.BytesIO(b"truncated"))
    _pin_digest(monkeypatch, "epdl", b"the whole library")

    with pytest.raises(build.LibraryChecksumError, match="downloaded"):
        build.download_library("epdl", tmp_path, force=True)

    assert not (tmp_path / "EPDL2023.ALL").exists()
    assert list(tmp_path.glob("*.partial")) == []


def test_verify_false_accepts_unpinned_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented escape hatch for a legitimately revised upstream library."""
    monkeypatch.setattr(
        build.urllib.request, "urlopen", lambda request: io.BytesIO(b"revised library")
    )
    result = build.download_library("epdl", tmp_path, force=True, verify=False)
    assert result.read_bytes() == b"revised library"


def test_pinned_digests_are_well_formed() -> None:
    """Both libraries are pinned, as 64 lowercase hex characters."""
    assert set(build.LIBRARY_SHA256) == set(build.LIBRARY_FILES)
    for digest in build.LIBRARY_SHA256.values():
        assert len(digest) == 64
        assert digest == digest.lower()
        int(digest, 16)  # raises unless it is hex


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

    common = [
        "--output",
        str(tmp_path / "out.npz"),
        "--cache-dir",
        str(tmp_path),
        "--allow-unverified-library",  # the fixture bytes are not the real libraries
    ]
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
        result = build.download_library("epdl", tmp_path, force=True, verify=False)
    assert result.read_bytes() == b"library bytes"
    assert any(
        record.levelno == logging.INFO and "downloading" in record.getMessage()
        for record in caplog.records
    )
    assert capsys.readouterr().out == ""
