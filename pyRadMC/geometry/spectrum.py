"""Photon energy spectra for the spectral beam sources.

A :class:`Spectrum` is a photon-number histogram — bin ``edges`` (n + 1) plus per-bin
``weights`` (n), normalized internally — sampled by CDF inversion on the per-history
RNG stream, exactly like the mixture selection in
:class:`~pyRadMC.geometry.source.CompositeSource`. It is the *source's own data*
(what the machine emits), not a material property, so it does not go through
:class:`~pyRadMC.data.interface.CrossSectionSource`.

Two input conventions are accepted (the pyRadMC <-> pyRadPlan seam): photon-**number**
content per bin (what sampling natively consumes) or energy-**fluence** content per
bin, converted by dividing by the bin midpoint energy.

:func:`ali_rogers_mv` builds a Spectrum from the analytic MV bremsstrahlung form of
Ali and Rogers (2012), either from explicit :class:`AliRogersMV` parameters or from
the paper's fitted parameters for nine benchmark linac beams
(:data:`ALI_ROGERS_BEAMS`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from pyRadMC.rng import RNGState, uniform

__all__ = ["ALI_ROGERS_BEAMS", "AliRogersMV", "Spectrum", "ali_rogers_mv"]

_CONVENTIONS = ("number", "energy_fluence")


class Spectrum:
    """A histogram photon spectrum sampled by CDF inversion.

    Parameters
    ----------
    edges
        Bin edges in MeV, strictly increasing, first edge positive; ``n + 1`` values.
        The last edge is :attr:`max_energy`, which sizes cross-section tables.
    weights
        Per-bin **content** (an integral over the bin, not a density); ``n``
        non-negative values, at least one positive. Normalized internally, so only
        ratios matter.
    convention
        ``"number"`` (default): weights are photons per bin — what CDF-inversion
        sampling natively consumes. ``"energy_fluence"``: weights are energy fluence
        per bin, converted to photon number by dividing by the bin **midpoint**
        energy. Stated approximation: the midpoint conversion and the uniform
        within-bin sampling below are exact only in the narrow-bin limit; supply
        number weights directly if the distinction matters at your bin width.
    """

    def __init__(
        self,
        edges: Sequence[float] | npt.NDArray[np.floating[Any]],
        weights: Sequence[float] | npt.NDArray[np.floating[Any]],
        convention: str = "number",
    ) -> None:
        e = np.asarray(edges, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if e.ndim != 1 or e.size < 2:
            raise ValueError(f"a spectrum needs at least two edges, got shape {e.shape}")
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(w)):
            raise ValueError("spectrum edges and weights must be finite")
        if e[0] <= 0.0:
            raise ValueError(f"the first edge must be positive, got {e[0]} MeV")
        if not np.all(np.diff(e) > 0.0):
            raise ValueError("spectrum edges must be strictly increasing")
        if w.shape != (e.size - 1,):
            raise ValueError(f"need one weight per bin: {e.size - 1} bins but {w.size} weights")
        if np.any(w < 0.0):
            raise ValueError("spectrum weights must be non-negative")
        if convention not in _CONVENTIONS:
            raise ValueError(f"unknown convention {convention!r}; expected {_CONVENTIONS}")
        if convention == "energy_fluence":
            w = w / (0.5 * (e[:-1] + e[1:]))
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("spectrum weights must include at least one positive value")

        self._edges = e
        self._p = w / total
        self._cdf = np.cumsum(self._p)
        self._edges.flags.writeable = False
        self._p.flags.writeable = False
        self._cdf.flags.writeable = False

    @property
    def max_energy(self) -> float:
        """Highest sampleable energy in MeV (the top bin edge)."""
        return float(self._edges[-1])

    @property
    def bin_probabilities(self) -> npt.NDArray[np.float64]:
        """Normalized per-bin photon emission probabilities (read-only)."""
        return self._p

    @property
    def edges(self) -> npt.NDArray[np.float64]:
        """Bin edges in MeV (read-only); ``n + 1`` values."""
        return self._edges

    @property
    def cdf(self) -> npt.NDArray[np.float64]:
        """Cumulative bin probabilities (read-only); ``n`` values ending at 1.

        The inversion table a device backend uploads to sample energies in-kernel
        with the same ``searchsorted`` convention as :meth:`sample_energy`.
        """
        return self._cdf

    @property
    def mean_energy(self) -> float:
        """Photon-number-weighted mean energy, MeV, on the bin-midpoint approximation."""
        return float(np.sum(self._p * 0.5 * (self._edges[:-1] + self._edges[1:])))

    def sample_energy(self, rng_state: RNGState) -> float:
        """Sample one photon energy by CDF inversion; consumes exactly two uniforms.

        The first uniform selects the bin from the cumulative weights (the
        :class:`~pyRadMC.geometry.source.CompositeSource` selection idiom); the
        second places the energy uniformly within the bin.
        """
        k = min(int(np.searchsorted(self._cdf, uniform(rng_state))), self._p.size - 1)
        lo = float(self._edges[k])
        hi = float(self._edges[k + 1])
        return lo + (hi - lo) * uniform(rng_state)

    def sample_energies(
        self, u_bin: npt.NDArray[np.float64], u_within: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Vectorized CDF inversion from caller-supplied uniforms.

        The same inversion as :meth:`sample_energy` — ``u_bin`` selects the bin,
        ``u_within`` the position inside it — for the vectorized pre-sampling batch
        of the spectral sources, which draws its uniforms from its own stream.
        """
        k = np.minimum(np.searchsorted(self._cdf, u_bin), self._p.size - 1)
        lo = self._edges[k]
        return np.asarray(lo + (self._edges[k + 1] - lo) * u_within, dtype=np.float64)


class AliRogersMV(NamedTuple):
    """Parameters of the Ali and Rogers (2012) proposed spectral form.

    The paper's function 12 (or 13 with ``c4 > 0``); see :func:`ali_rogers_mv` for
    the governing equation. Except for ``e_e``, the parameters are fit coefficients,
    not physical quantities (paper, section 3).
    """

    e_e: float
    """Endpoint energy in MeV: the mean incident electron kinetic energy."""
    c1: float
    """Effective tungsten filtration sqrt-thickness; C1^2 in g/cm^2 attenuates by mu_W."""
    c2: float
    """Effective aluminium filtration sqrt-thickness; C2^2 in g/cm^2 attenuates by mu_Al."""
    c3: float
    """Dimensionless shape coefficient of the thin-target term."""
    c4: float = 0.0
    """Integral-energy-fluence coefficient of the 511 keV delta term, MeV.

    Function 13 places this term inside the same tungsten/aluminium filtration
    envelope as ``psi_thin``; 0 omits the line (the four-parameter form).
    """


ALI_ROGERS_BEAMS: dict[str, AliRogersMV] = {
    "varian-4mv": AliRogersMV(e_e=3.75, c1=3.824, c2=3.522, c3=-1.222, c4=0.00308),
    "varian-6mv": AliRogersMV(e_e=5.76, c1=1.222, c2=5.147, c3=-1.186, c4=0.00881),
    "varian-10mv": AliRogersMV(e_e=10.46, c1=0.702, c2=6.226, c3=-1.285, c4=0.03891),
    "varian-15mv": AliRogersMV(e_e=14.58, c1=4.614, c2=3.804, c3=-1.060, c4=0.14034),
    "varian-18mv": AliRogersMV(e_e=18.33, c1=3.347, c2=5.847, c3=-1.228, c4=0.43160),
    "elekta-6mv": AliRogersMV(e_e=6.49, c1=1.320, c2=5.072, c3=-1.109, c4=0.02526),
    "elekta-25mv": AliRogersMV(e_e=19.02, c1=0.000, c2=7.504, c3=-1.274, c4=0.57864),
    "siemens-6mv": AliRogersMV(e_e=6.83, c1=1.184, c2=4.840, c3=-1.161, c4=0.01416),
    "siemens-18mv": AliRogersMV(e_e=14.94, c1=1.213, c2=6.142, c3=-1.126, c4=0.12522),
}
"""Fitted parameters for the nine benchmark linac beams of Sheikh-Bagheri and Rogers
(2002), with flattening filters, from Ali and Rogers (2012) table 5. Keyed
``"<vendor>-<nominal MV>mv"``; the endpoint ``e_e`` is the *fitted* electron energy,
which differs from the nominal MV (e.g. 5.76 MeV for the Varian 6 MV beam)."""

_MU_W_FLOOR_MEV = 0.0695
"""Validity floor of the tungsten mu(E) parameterization (its K-edge), paper table 3.

Tungsten's K edge is a discontinuity — mu/rho jumps by roughly 4.6x across it, from
about 2.4 to about 11 cm^2/g — and a single smooth form cannot straddle it, so
:func:`_mu_over_rho_tungsten` represents the **above-edge** branch. It tracks EPDL 2023
to within 1.4 percent from 0.070 MeV upward (validated in
``tests/validation/test_ali_rogers_mu_parameterization.py``).

The paper states the range inclusively, and the guard in :func:`ali_rogers_mv` follows
it, so ``e_min`` exactly at this floor is accepted and evaluates the above-edge branch a
few hundred eV *below* where the library places the jump. The affected sliver is far
narrower than one bin, sits below ``PCUT``, and is 80 keV below the 0.15 MeV default
``e_min``; it is recorded here because it looks like a bug when first encountered, not
because it moves dose.
"""

_E_ANNIHILATION_MEV = 0.511
"""Photon energy of the positron annihilation line."""


def _mu_over_rho_tungsten(e: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Tungsten mass attenuation, cm^2/g, parameterized per Ali and Rogers table 3.

    ``mu_W(E) = exp[(a0 + a1 E + a2 E^2) / (b0 + b1 E + b2 E^2 + b3 E^3)]`` for
    69.5 keV <= E <= 30 MeV (typical local error 0.5% against the NIST grid).
    """
    num = 6.575 + -3.623e1 * e + -1.578 * e**2
    den = 1.0 + 9.667 * e + 7.132e-1 * e**2 + -3.778e-4 * e**3
    return np.exp(num / den)


def _mu_over_rho_aluminium(e: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Aluminium mass attenuation, cm^2/g, parameterized per Ali and Rogers table 3.

    ``mu_Al(E) = sum a_j (ln E)^j / sum b_j (ln E)^j`` (j = 0..5) for
    1.56 keV <= E <= 30 MeV.
    """
    log_e = np.log(e)
    a = (6.107e-2, 1.694e-2, -2.390e-3, 3.116e-4, 5.286e-4, 2.507e-5)
    b = (1.0, 7.769e-1, 2.434e-1, 3.838e-2, 3.042e-3, 9.686e-5)
    num = sum(aj * log_e**j for j, aj in enumerate(a))
    den = sum(bj * log_e**j for j, bj in enumerate(b))
    return np.asarray(num / den, dtype=np.float64)


def _psi_continuum(e: npt.NDArray[np.float64], p: AliRogersMV) -> npt.NDArray[np.float64]:
    """Differential energy fluence psi(E) of the proposed form (function 12).

    psi(E) = [1 + C3 (E/Ee) + (E/Ee)^2] [ln(Ee (Ee - E) / E + 1.65) - 0.5]
             * exp(-mu_W(E) C1^2 - mu_Al(E) C2^2),   energies in MeV,

    Ali and Rogers, Phys. Med. Biol. 57 (2012) 31-50, doi:10.1088/0031-9155/57/1/31,
    table 2. The 1.65 = exp(0.5) constant imposes psi(Ee) ~ 0 (their figure 1); the
    squared thickness parameters guarantee positivity of the filtration term.
    """
    ratio = e / p.e_e
    thin = (1.0 + p.c3 * ratio + ratio**2) * (np.log(p.e_e * (p.e_e - e) / e + 1.65) - 0.5)
    return np.asarray(
        thin * np.exp(-_mu_over_rho_tungsten(e) * p.c1**2 - _mu_over_rho_aluminium(e) * p.c2**2),
        dtype=np.float64,
    )


def ali_rogers_mv(beam: str | AliRogersMV, e_min: float = 0.15, n_bins: int = 100) -> Spectrum:
    """Build a Spectrum from the Ali and Rogers (2012) analytic MV form.

    The governing equation is in :func:`_psi_continuum`; per-bin photon numbers are
    the sub-grid integrals of ``psi(E) / E`` (the product is integrated, not bin
    means — the AGENTS.md 2.7 rule), and a ``c4 > 0`` adds the 511 keV annihilation
    line inside the common filtration envelope of function 13. Its photon content is
    ``c4 * exp(-mu_W C1^2 - mu_Al C2^2) / 0.511`` at 511 keV.

    Parameters
    ----------
    beam
        A key of :data:`ALI_ROGERS_BEAMS` (e.g. ``"varian-6mv"``) or explicit
        :class:`AliRogersMV` parameters.
    e_min
        Lower spectrum edge in MeV. Must stay at or above the 69.5 keV validity
        floor of the tungsten attenuation parameterization; the default 0.15 MeV
        is far above ``PCUT`` and cuts only a negligible fluence tail.
    n_bins
        Histogram resolution; the default matches the paper's 100-bin spectra.
    """
    if isinstance(beam, str):
        if beam not in ALI_ROGERS_BEAMS:
            raise KeyError(
                f"unknown beam {beam!r}; available: {', '.join(sorted(ALI_ROGERS_BEAMS))}"
            )
        beam = ALI_ROGERS_BEAMS[beam]
    if e_min < _MU_W_FLOOR_MEV:
        raise ValueError(
            f"e_min={e_min} MeV is below the 69.5 keV validity floor of the "
            "tungsten attenuation parameterization (Ali and Rogers 2012, table 3)"
        )
    if e_min >= beam.e_e:
        raise ValueError(f"e_min={e_min} MeV is not below the endpoint {beam.e_e} MeV")
    if n_bins < 1:
        raise ValueError(f"need at least one bin, got {n_bins}")

    edges = np.linspace(e_min, beam.e_e, n_bins + 1)
    # Per-bin photon number: integrate psi(E)/E on a sub-grid of each bin (trapezoid,
    # 8 panels per bin). The endpoint bin's upper limit is Ee itself, where the form
    # is finite (psi(Ee) ~ 0), so no special casing is needed.
    sub = 8
    fine = np.linspace(edges[:-1], edges[1:], sub + 1, axis=1)
    values = _psi_continuum(fine.reshape(-1), beam).reshape(fine.shape) / fine
    weights = np.asarray(np.trapezoid(values, fine, axis=1), dtype=np.float64)
    if beam.c4 > 0.0:
        if not e_min <= _E_ANNIHILATION_MEV < beam.e_e:
            raise ValueError(
                f"c4={beam.c4} books a 511 keV line outside the spectrum range "
                f"[{e_min}, {beam.e_e}) MeV"
            )
        k = int(np.searchsorted(edges, _E_ANNIHILATION_MEV, side="right") - 1)
        line_energy = np.array([_E_ANNIHILATION_MEV], dtype=np.float64)
        line_transmission = float(
            np.exp(
                -_mu_over_rho_tungsten(line_energy)[0] * beam.c1**2
                - _mu_over_rho_aluminium(line_energy)[0] * beam.c2**2
            )
        )
        weights[k] += beam.c4 * line_transmission / _E_ANNIHILATION_MEV
    return Spectrum(edges, weights)
