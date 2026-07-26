"""Download the EPICS libraries and compile a tabulated cross-section table.

A small build tool, not part of the runtime: it fetches the EPDL and EEDL 2023 files
from the IAEA (public data; the download and its licence stay user-side, and the
compiled ``.npz`` is source-agnostic), caches them, compiles a registry prefix via
:func:`~pyRadMC.data.tabulated.precompile.compile_materials`, and writes the result.

    python -m pyRadMC.data.tabulated.build --output materials.npz

The default compiles the whole registry (water through tungsten); ``--n-materials 1``
reproduces the historical water-only table. The cached libraries and the compiled
table are never committed (stdlib ``urllib`` only, so no dependency is added).
Re-running reuses the cache unless ``--force-download``.

Every library — freshly downloaded or served from the cache — is checked against a
pinned SHA-256 (:data:`LIBRARY_SHA256`) before it is used. These files *are* the
cross-sections: a truncated transfer, a corrupted cache entry or an upstream revision
would otherwise propagate silently into every dose this engine computes. The check is
therefore fail-closed; ``--allow-unverified-library`` exists for the case where the
IAEA has legitimately revised a library, and using it means the resulting tables are
not the ones this release validated.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import urllib.request
from pathlib import Path

import numpy as np

from pyRadMC import ECUT_MEV, PCUT_MEV, __version__
from pyRadMC.data.tables import TABLE_POINTS
from pyRadMC.data.tabulated.format import save_tables
from pyRadMC.data.tabulated.precompile import ElectronStoppingStrategy, compile_materials

__all__ = ["LibraryChecksumError", "download_library", "library_path", "main"]

EPICS_BASE_URL = "https://www-nds.iaea.org/epics/ENDF2023/"
LIBRARY_FILES = {"epdl": "EPDL2023.ALL", "eedl": "EEDL2023.ALL"}
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pyRadMC" / "epics"
_USER_AGENT = f"pyRadMC/{__version__} (EPICS library fetch)"

LIBRARY_SHA256: dict[str, str] = {
    "epdl": "b4888d0d0e3b7672f56232f1c6ef8da369a67ad2f2d773168fc26275fac82fca",
    "eedl": "1cf45f070fe31d5c0753930499f9e04b59c04285108b6e704f8693c25362ff57",
}
"""SHA-256 of the EPICS 2023 libraries this release's cross-sections are built from.

Recorded 2026-07-26 from the copies that produced the tables behind the XCOM
(sub-percent), ESTAR and EGSnrc PDD validation results, and re-confirmed against a
second independent transfer from ``https://www-nds.iaea.org/epics/ENDF2023/``.
Sizes: EPDL 91_508_720 bytes, EEDL 27_576_928 bytes.

Updating a digest is not a maintenance chore: it changes the cross-sections. Re-run
the validation tier against the new library and record the outcome before changing a
value here.
"""

_HASH_BLOCK = 1 << 20

logger = logging.getLogger(__name__)


class LibraryChecksumError(RuntimeError):
    """A downloaded or cached EPICS library does not match its pinned SHA-256."""


def library_path(which: str, cache_dir: Path | None = None) -> Path:
    """Return the cached path a library resolves to (whether or not it exists yet)."""
    return (cache_dir or DEFAULT_CACHE_DIR) / LIBRARY_FILES[which]


def _file_digest(path: Path) -> str:
    """SHA-256 of a file, read in blocks so a 92 MB library never lands in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(which: str, actual: str, *, source: str) -> None:
    """Raise unless ``actual`` is the pinned digest for ``which``."""
    expected = LIBRARY_SHA256[which]
    if actual == expected:
        return
    raise LibraryChecksumError(
        f"{LIBRARY_FILES[which]} ({source}) does not match the pinned SHA-256.\n"
        f"  expected {expected}\n"
        f"  actual   {actual}\n"
        "These bytes are the cross-sections, so pyRadMC refuses to compile tables from "
        "them. Delete the cache entry and retry in case the transfer was corrupted. If "
        "the IAEA has revised the library, re-run the validation tier against it and "
        "update LIBRARY_SHA256 deliberately; --allow-unverified-library bypasses this "
        "check but yields tables this release has not validated."
    )


def download_library(
    which: str,
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    verify: bool = True,
) -> Path:
    """Return the cached ``which`` library ('epdl'/'eedl'), downloading it if absent.

    Writes to a temporary file and renames, so an interrupted download never leaves a
    truncated file masquerading as a complete cache entry. The bytes are hashed as
    they stream past and checked against :data:`LIBRARY_SHA256` *before* the rename,
    so a bad transfer never becomes a cache entry; a cache hit is re-hashed for the
    same reason, since the file may have been corrupted since it was written.

    Parameters
    ----------
    verify
        Check the pinned SHA-256. ``False`` accepts whatever bytes arrive and is only
        appropriate when deliberately building against a revised upstream library.

    Raises
    ------
    LibraryChecksumError
        The library does not match its pinned digest.
    """
    destination = library_path(which, cache_dir)
    if destination.is_file() and not force:
        if verify:
            _require_digest(which, _file_digest(destination), source="cached")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = EPICS_BASE_URL + LIBRARY_FILES[which]
    temporary = destination.with_suffix(destination.suffix + ".partial")
    logger.info("downloading %s -> %s", url, destination)
    # An explicit User-Agent is required: the IAEA site is behind Cloudflare, which
    # rejects urllib's default "Python-urllib/x.y" agent with HTTP 403.
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        while block := response.read(_HASH_BLOCK):
            digest.update(block)
            handle.write(block)
    if verify:
        try:
            _require_digest(which, digest.hexdigest(), source="downloaded")
        except LibraryChecksumError:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)
    return destination


def main(argv: list[str] | None = None) -> None:
    """CLI: download the libraries (as needed) and compile a materials table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--output", type=Path, required=True, help="output .npz path")
    parser.add_argument("--cache-dir", type=Path, default=None, help="library cache directory")
    parser.add_argument(
        "--n-materials",
        type=int,
        default=None,
        help="compile only the first N registry materials (default: whole registry)",
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ElectronStoppingStrategy],
        default=ElectronStoppingStrategy.BERGER_SELTZER.value,
    )
    parser.add_argument("--e-max", type=float, default=30.0, help="upper grid edge in MeV")
    parser.add_argument("--pcut", type=float, default=PCUT_MEV)
    parser.add_argument("--ecut", type=float, default=ECUT_MEV)
    parser.add_argument("--n-points", type=int, default=TABLE_POINTS)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--allow-unverified-library",
        action="store_true",
        help="skip the pinned SHA-256 check; the tables are then unvalidated",
    )
    args = parser.parse_args(argv)

    verify = not args.allow_unverified_library
    if not verify:
        logger.warning(
            "SHA-256 verification disabled: the compiled tables are built from "
            "unverified cross-section data and are not what this release validated"
        )
    epdl = download_library("epdl", args.cache_dir, force=args.force_download, verify=verify)
    eedl = download_library("eedl", args.cache_dir, force=args.force_download, verify=verify)
    data = compile_materials(
        epdl.read_text(encoding="latin-1"),
        eedl.read_text(encoding="latin-1"),
        n_materials=args.n_materials,
        strategy=ElectronStoppingStrategy(args.strategy),
        pcut=args.pcut,
        ecut=args.ecut,
        e_max=args.e_max,
        n_points=args.n_points,
    )
    written = save_tables(data, args.output, dtype=getattr(np, args.dtype))
    logger.info("compiled %s", data.provenance)
    logger.info("wrote %s", written)


if __name__ == "__main__":
    # Handler configuration belongs to the application; this CLI entry is the
    # application. Library import paths never call basicConfig.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
