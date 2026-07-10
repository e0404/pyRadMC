"""Material definitions for the analytic cross-section backend.

Phase 0 transports photons in water only. The registry is nevertheless a registry, not a
constant, so that adding lung/bone-like media in later phases is a data change and not a
code change.

Materials are referred to everywhere by integer index (see
:class:`pyRadMC.data.interface.CrossSectionSource`); the index crosses unchanged into
kernels, where a Python object cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MATERIALS", "WATER", "MaterialData"]

AVOGADRO: float = 6.022_140_76e23
"""Avogadro constant in 1/mol (exact, SI 2019)."""


@dataclass(frozen=True)
class MaterialData:
    """Host-side physical data for one material.

    Attributes
    ----------
    name
        Human-readable identifier.
    density
        Reference mass density in g/cm^3. Voxel densities in the geometry scale
        macroscopic cross-sections relative to this via the mass quantities, so this
        value is informational for water-like transport, not accuracy-defining.
    electrons_per_gram
        Electron density per unit mass in 1/g, i.e. ``N_A * sum(Z_i) / M``. Multiplies
        per-electron cross-sections (Klein-Nishina) into mass attenuation coefficients.
    """

    name: str
    density: float
    electrons_per_gram: float


WATER: int = 0
"""Material index of liquid water."""

# H2O: 10 electrons per molecule, M = 18.01528 g/mol.
_WATER_ELECTRONS_PER_GRAM = 10.0 * AVOGADRO / 18.01528

MATERIALS: tuple[MaterialData, ...] = (
    MaterialData(name="water", density=1.0, electrons_per_gram=_WATER_ELECTRONS_PER_GRAM),
)
"""Material registry, indexed by the ``WATER``-style integer constants."""
