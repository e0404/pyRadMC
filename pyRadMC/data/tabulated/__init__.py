"""Tabulated cross-section backend (Phase 5).

A compile-once / load-fast split: authoritative source data (EPDL/ESTAR, or
user-supplied) is parsed into the canonical :class:`~pyRadMC.data.tabulated.model.TabulatedData`,
the precompiler writes it to an internal format
(:mod:`~pyRadMC.data.tabulated.format`), and the loader
(:class:`~pyRadMC.data.tabulated.source.TabulatedCrossSections`) reads that format
with no knowledge of its provenance — so reduced-precision recompiles and
user-defined tables are the same code path.
"""
