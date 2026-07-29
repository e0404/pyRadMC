"""Measured radial primary photon fluence, the input of the virtual source model.

A :class:`RadialFluence` is a commissioning curve: the relative primary photon
fluence psi(r) of a flattened MV beam, radially symmetric about the central axis
and tabulated against off-axis radius at one reference distance from the target
(conventionally isocentre, hence the 100 cm default). It carries the flattening
filter's horn and the primary collimator's circular field edge in one table.

Like :class:`~pyRadMC.geometry.spectrum.Spectrum` this is *source data* — what
the machine emits, measured once — not a material property, so it does not go
through :class:`~pyRadMC.data.interface.CrossSectionSource`. It is consumed by
:class:`~pyRadMC.geometry.source.PrimaryFluenceBeamSource` and its beamlet twin.

Two extrapolation rules apply outside the tabulated range, and both are
deliberate:

- **inward** of the first radius the first value is held flat (a table starting
  at r > 0 makes no statement about the axis, where the fluence is smooth);
- **outward** of the last radius the fluence is **zero**, not the last value. A
  measured curve runs past the primary collimator's field edge and ends at zero,
  so this is a no-op for real data; for a truncated table it is a hard field
  edge, which is the safe reading — a held-constant tail would irradiate the
  whole phantom.

Values are used **as given**: the source multiplies them onto each history's
statistical weight, so scaling the table scales dose per history linearly. The
table is a shape, and its overall normalization is the caller's calibration.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["RadialFluence"]


class RadialFluence:
    """A radially symmetric primary fluence psi(r), linearly interpolated.

    Parameters
    ----------
    radii
        Off-axis radii in **cm**, non-negative and strictly increasing; at least
        two values. They are distances *at* :paramref:`reference_distance`, not
        at the plane the source emits on — the source projects between the two.
    values
        Relative fluence at each radius; one per radius, non-negative, at least
        one positive. Used as given (see the module docstring on normalization).
    reference_distance
        Distance from the focal spot, in cm, at which ``radii`` are quoted. The
        default 100 cm is the isocentre convention of a commissioning curve.
    """

    def __init__(
        self,
        radii: Sequence[float] | npt.NDArray[np.floating[Any]],
        values: Sequence[float] | npt.NDArray[np.floating[Any]],
        reference_distance: float = 100.0,
    ) -> None:
        r = np.asarray(radii, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        if r.ndim != 1 or r.size < 2:
            raise ValueError(f"a fluence table needs at least two radii, got shape {r.shape}")
        if v.shape != r.shape:
            raise ValueError(f"need one value per radius: {r.size} radii but {v.size} values")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            raise ValueError("fluence radii and values must be finite")
        if r[0] < 0.0:
            raise ValueError(f"fluence radii must be non-negative, got {r[0]} cm")
        if not np.all(np.diff(r) > 0.0):
            raise ValueError("fluence radii must be strictly increasing")
        if np.any(v < 0.0):
            raise ValueError("fluence values must be non-negative")
        if not np.any(v > 0.0):
            raise ValueError("fluence values must include at least one positive value")
        if reference_distance <= 0.0:
            raise ValueError(f"reference_distance must be positive, got {reference_distance} cm")

        self._radii = r
        self._values = v
        self._reference_distance = float(reference_distance)
        self._radii.flags.writeable = False
        self._values.flags.writeable = False

    @property
    def radii(self) -> npt.NDArray[np.float64]:
        """Tabulated off-axis radii in cm at :attr:`reference_distance` (read-only)."""
        return self._radii

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Tabulated relative fluence, one per radius (read-only)."""
        return self._values

    @property
    def reference_distance(self) -> float:
        """Distance from the focal spot, cm, at which :attr:`radii` are quoted."""
        return self._reference_distance

    @property
    def max_radius(self) -> float:
        """Largest tabulated radius in cm; beyond it the fluence reads zero."""
        return float(self._radii[-1])

    def at_radius(self, radius: float) -> float:
        """Interpolate psi at one radius (cm at :attr:`reference_distance`).

        The magnitude is taken, so a signed off-axis coordinate reads correctly.
        See the module docstring for the two extrapolation rules.
        """
        return float(
            np.interp(abs(radius), self._radii, self._values, left=self._values[0], right=0.0)
        )

    def at_radii(self, radii: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
        """Vectorized :meth:`at_radius`, for the pre-sampling batch route."""
        return np.asarray(
            np.interp(np.abs(radii), self._radii, self._values, left=self._values[0], right=0.0),
            dtype=np.float64,
        )

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        radius_scale: float = 0.1,
        reference_distance: float = 100.0,
    ) -> RadialFluence:
        """Read a two-column whitespace table of ``radius fluence`` (``#`` comments).

        This is the layout of the PPBKC ``primflu.dat`` commissioning file, whose
        radii are in **mm** at isocentre; ``radius_scale`` converts them to the
        engine's cm (pass 1.0 for a table already in cm).
        """
        table = np.loadtxt(path, comments="#", ndmin=2)
        if table.shape[1] < 2:
            raise ValueError(f"{path}: expected two columns of 'radius fluence'")
        return cls(
            radii=table[:, 0] * radius_scale,
            values=table[:, 1],
            reference_distance=reference_distance,
        )
