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
"""

from __future__ import annotations

import argparse
import logging
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from pyRadMC import ECUT_MEV, PCUT_MEV, __version__
from pyRadMC.data.tables import TABLE_POINTS
from pyRadMC.data.tabulated.format import save_tables
from pyRadMC.data.tabulated.precompile import ElectronStoppingStrategy, compile_materials

__all__ = ["download_library", "library_path", "main"]

EPICS_BASE_URL = "https://www-nds.iaea.org/epics/ENDF2023/"
LIBRARY_FILES = {"epdl": "EPDL2023.ALL", "eedl": "EEDL2023.ALL"}
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pyRadMC" / "epics"
_USER_AGENT = f"pyRadMC/{__version__} (EPICS library fetch)"

logger = logging.getLogger(__name__)


def library_path(which: str, cache_dir: Path | None = None) -> Path:
    """Return the cached path a library resolves to (whether or not it exists yet)."""
    return (cache_dir or DEFAULT_CACHE_DIR) / LIBRARY_FILES[which]


def download_library(which: str, cache_dir: Path | None = None, *, force: bool = False) -> Path:
    """Return the cached ``which`` library ('epdl'/'eedl'), downloading it if absent.

    Writes to a temporary file and renames, so an interrupted download never leaves a
    truncated file masquerading as a complete cache entry.
    """
    destination = library_path(which, cache_dir)
    if destination.is_file() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = EPICS_BASE_URL + LIBRARY_FILES[which]
    temporary = destination.with_suffix(destination.suffix + ".partial")
    logger.info("downloading %s -> %s", url, destination)
    # An explicit User-Agent is required: the IAEA site is behind Cloudflare, which
    # rejects urllib's default "Python-urllib/x.y" agent with HTTP 403.
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
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
    args = parser.parse_args(argv)

    epdl = download_library("epdl", args.cache_dir, force=args.force_download)
    eedl = download_library("eedl", args.cache_dir, force=args.force_download)
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
