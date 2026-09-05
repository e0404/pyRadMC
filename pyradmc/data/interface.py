"""Cross-section and stopping-power access interface.

This module defines the *only* permitted access path to interaction data. No physics
routine may hardcode a cross-section formula; see AGENTS.md section 2.6.

Two implementations are planned:

``analytic``
    Closed-form parameterizations. Fast to stand up, adequate for development and
    for water-only work, insufficient for final accuracy.

``tabulated``
    Interpolated tables (e.g. EPDL/EEDL-derived, or NIST XCOM/ESTAR-derived). Must
    integrate *products* over sub-grid intervals rather than evaluating separately
    averaged bin quantities at bin centers; see AGENTS.md section 2.7 and
    ``tests/unit/test_interface_contracts.py``.

All energies are in MeV. All macroscopic cross-sections are in 1/cm. All mass
attenuation and mass stopping-power quantities are per unit mass density, i.e. in
cm^2/g and MeV cm^2/g respectively.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from pyradmc.data.tables import CrossSectionTables

__all__ = ["CrossSectionSource", "PhotonProcess"]

_RANGE_GRID_E_MAX: float = 64.0
"""Upper edge (MeV) of the cached restricted-range grid: comfortably above the
1-20 MeV engine scope; queries clamp to it."""

_GS_BINS_PER_LOG: float = 4.0
"""Goudsmit-Saunderson grid nodes per natural log of ``eta`` and ``<theta^2>``.

Coarse on purpose, and affordable because the lookup **interpolates bilinearly**
between the four bracketing nodes rather than snapping to the nearest. The
tables are normalized in a scaled deflection, so binning cannot move
``<1 - cos theta>`` — only the shape around it — and refining this
two-dimensional grid instead costs quadratically (measured: 16 per log ran to
~10k tables and ~20 MB, against ~1.1 MB for the full grid at 4). The residual
shape error is pinned by a validation-tier test against exactly-built tables,
not chosen by eye.
"""

_GS_TABLE_NODES: int = 512
"""Equal-probability bins per Goudsmit-Saunderson deflection table."""

_GS_THETA2_MIN: float = 1.0e-12
"""Floor on the binned ``<theta^2>``; below this a substep deflects immeasurably
and the table would be built for a distribution indistinguishable from forward."""

_RANGE_GRID_POINTS: int = 4096
"""Nodes of the cached range grid; trapezoid error at this log spacing sits far
below the 1e-4 pins of the derivative and round-trip tests."""


class PhotonProcess:
    """Enumeration of photon interaction channels.

    Integer-valued rather than a ``enum.Enum`` so that the values can cross into
    Warp kernels and the reference backend unchanged.
    """

    COMPTON = 0
    PHOTOELECTRIC = 1
    PAIR = 2
    RAYLEIGH = 3


class CrossSectionSource(ABC):
    """Abstract source of interaction data for photons and electrons.

    Implementations are constructed once, on the host, and then expose flat array
    handles to the kernels. The abstract methods below are the *host-side* query
    API used for construction, validation, and the reference backend. Kernel-side
    access goes through the flattened tables that ``build_tables`` returns.
    """

    @property
    def n_materials(self) -> int:
        """How many registry materials this source can answer for.

        Tables are flattened for material indices ``0 .. n_materials - 1``
        (:func:`~pyradmc.data.tables.build_cross_section_tables` sizes its rows by
        this), and a query beyond it must raise rather than approximate: a source
        silently answering for a material it has no data for is a silent transport
        bias. The default — the full registry — is for material-*independent*
        sources (test instruments); a calibrated source overrides it with its real
        coverage (the analytic source: water only; the tabulated source: the
        compiled row count).
        """
        from pyradmc.data.materials import MATERIALS

        return len(MATERIALS)

    @property
    def provenance(self) -> str:
        """Human-readable citation for the data this source answers from.

        Carried into every result's
        :class:`~pyradmc.backends.results.RunProvenance` so an archived dose says
        which cross-sections produced it — the single most consequential thing about
        a run and the one least recoverable from the dose array. The default names
        the class, which is honest but uninformative; a source built from compiled
        data overrides it with that data's own provenance string.
        """
        return type(self).__name__

    # -- photons ------------------------------------------------------------

    @abstractmethod
    def mu_over_rho(self, energy: float, material: int, process: int) -> float:
        """Mass attenuation coefficient for one process, in cm^2/g.

        Parameters
        ----------
        energy
            Photon energy in MeV.
        material
            Material index into the material table.
        process
            One of the :class:`PhotonProcess` constants.
        """

    @abstractmethod
    def mu_over_rho_total(self, energy: float, material: int) -> float:
        """Total mass attenuation coefficient, in cm^2/g.

        Must equal the sum over enabled processes. This redundancy is deliberate:
        it is a contract test (``tests/unit/test_xs_contract.py``).
        """

    @abstractmethod
    def majorant(self, energy: float) -> float:
        """Woodcock majorant: the maximum macroscopic total cross-section, in 1/cm.

        Taken over all materials and densities present in the geometry, at the given
        energy. Delta (fictitious) scattering makes up the difference. The majorant
        must never be exceeded by any real macroscopic cross-section in the geometry;
        this is a contract test, and violating it silently biases the transport.

        See Woodcock et al. (1965), ANL-7050.
        """

    def sample_coherent_cos_theta(self, energy: float, material: int, rng_state: object) -> float:
        """Sample the coherent (Rayleigh) polar scattering cosine.

        The default is the Thomson distribution (zero-momentum-transfer, flat form
        factor); a source with real atomic form-factor data (the tabulated backend)
        overrides this to sample the forward-peaked coherent distribution. Kept on the
        source, not as a free function, because the angular shape *is* cross-section data
        (AGENTS.md 2.6): the sampling math lives in :mod:`pyradmc.physics.rayleigh`, the
        data that selects it lives here.
        """
        from pyradmc.physics.rayleigh import sample_rayleigh_cos_theta

        return sample_rayleigh_cos_theta(rng_state)

    def coherent_cumulative(
        self, x_grid: npt.NDArray[np.float64], material: int
    ) -> npt.NDArray[np.float64]:
        r"""Return the coherent form-factor cumulative ``A(x)=\int_0^x F^2 x' dx'`` on ``x_grid``.

        This is the flattened, kernel-consumable face of :meth:`sample_coherent_cos_theta`:
        :func:`~pyradmc.data.tables.build_cross_section_tables` calls it per material to
        fill the table the Warp kernel inverts. The default is the flat form factor,
        ``A(x) = x^2/2``, which the sampler inverts to the Thomson distribution — so an
        analytic (zero-coherent) source needs no override, and a form-factor source
        (tabulated) resamples its compiled cumulative onto ``x_grid``.
        """
        return 0.5 * np.asarray(x_grid, dtype=np.float64) ** 2

    # -- electrons ----------------------------------------------------------

    @abstractmethod
    def restricted_stopping_power(self, energy: float, material: int, delta_cut: float) -> float:
        """Restricted collision stopping power, in MeV cm^2/g.

        Energy losses above ``delta_cut`` are excluded, being handled explicitly as
        discrete Moller (or Bhabha) events in the Class II scheme.

        Berger-Seltzer formulation; see ICRU Report 37 (1984).
        """

    @abstractmethod
    def radiative_stopping_power(self, energy: float, material: int) -> float:
        """Radiative (bremsstrahlung) stopping power, in MeV cm^2/g."""

    @abstractmethod
    def moller_cross_section(self, energy: float, material: int, delta_cut: float) -> float:
        """Restricted Moller cross-section per unit mass, in cm^2/g.

        Total cross-section for a discrete knock-on collision transferring more than
        ``delta_cut`` (MeV, kinetic) to a delta ray. Zero when ``energy`` is at or
        below ``2 * delta_cut``: by indistinguishability the delta is the *lower*
        energy outgoing electron, so it can carry at most half the kinetic energy.

        Consistency contract (tested): the energy moment of the Moller differential
        cross-section above ``delta_cut`` equals the difference between the
        unrestricted and restricted collision stopping powers. Moller (1932),
        doi:10.1002/andp.19324060506.
        """

    @abstractmethod
    def csda_range(self, energy: float, material: int) -> float:
        """Continuous-slowing-down-approximation range, in g/cm^2.

        Used for range rejection. An overestimate is safe (it rejects less); an
        underestimate biases the dose. Implementations must document which side they
        err on.
        """

    @abstractmethod
    def scattering_power(self, energy: float, material: int, delta_cut: float) -> float:
        """Mass angular scattering power, in rad^2 cm^2/g.

        Drives the multiple-elastic-scattering hinge deflection: ``T rho s`` is
        the small-step ``<theta^2>`` handed to the sampler. Because
        Goudsmit-Saunderson pins ``<cos theta> = exp(-s N sigma_tr)`` exactly,
        this must be the *first transport moment* ``2 (N_A/A) sigma_el G_1`` of
        the screened scattering law — a core-width fit such as Highland's is not the same
        quantity and under-scatters at high energy (the analytic source's
        docstring records the approximation). ``delta_cut`` partitions the
        electron-electron moment: transfers below it remain condensed here,
        while the moment of above-cutoff Moller events is excluded because those
        deflections are transported explicitly by the Class-II loop.
        """

    def elastic_screening(self, energy: float, material: int) -> float:
        r"""Moliere screening parameter of the elastic scattering law, dimensionless.

        Where :meth:`scattering_power` fixes the *strength* of multiple
        scattering, this fixes the *shape*: it is the parameter of the
        screened-Rutherford single-scattering law whose Legendre moments drive
        the Goudsmit-Saunderson angular distribution
        (:mod:`pyradmc.data.goudsmit_saunderson`). The two are consistent by
        construction — the elastic cross-section is back-derived from
        ``T = 2 (N_A/M) sigma_el G_1(eta)`` — so the GS mean-square deflection
        over a short step reproduces the Fermi-Eyges ``T rho s`` exactly. That
        anchoring is a contract test
        (``tests/unit/test_elastic_screening.py``): it is what makes GS a
        refinement of the shipped hinge rather than a rescaling of it.

        Concrete on the interface, computed from the material's elemental
        composition, so no implementation can silently omit it and the analytic
        and tabulated backends cannot drift apart on the angular shape. A source
        carrying real elastic differential data may override it.
        """
        from pyradmc.data.goudsmit_saunderson import moliere_screening
        from pyradmc.data.materials import MATERIALS

        if not 0 <= material < self.n_materials:
            raise ValueError(f"material index {material} beyond this source")
        return moliere_screening(MATERIALS[material].composition, energy)

    def sample_gs_cos_theta(
        self, mean_square_angle: float, energy: float, material: int, rng_state: object
    ) -> float:
        """Sample the Goudsmit-Saunderson multiple-scattering deflection cosine.

        The exact multiple-scattering angle for the substep, in place of the
        small-angle Gaussian of :func:`pyradmc.physics.msc.sample_hinge_cos_theta`.
        Takes the *same* ``mean_square_angle = T rho s`` the Gaussian hinge takes
        and consumes the same single uniform, so the two are drop-in
        alternatives that do not shift the random stream relative to each other —
        which is what lets a transport comparison isolate the angular model.

        Kept on the source rather than as a free function for the reason the
        coherent sampler is (see :meth:`sample_coherent_cos_theta`): the angular
        shape *is* cross-section data. The sampling math lives in
        :mod:`pyradmc.physics.gs`; the table that selects it lives here.

        Tables are memoized on a log-spaced ``(eta, <theta^2>)`` grid. Binning is
        safe because the table is normalized in a scaled deflection: the
        rescaling by this call's exact ``<1 - cos theta>`` restores the first
        moment exactly, so the grid resolution perturbs only the shape.
        """
        from pyradmc.data.goudsmit_saunderson import gs_scaled_deflection_table
        from pyradmc.physics.gs import sample_gs_cos_theta_bilinear

        if mean_square_angle <= 0.0:
            return 1.0
        eta = self.elastic_screening(energy, material)

        cache: dict[tuple[int, int], npt.NDArray[np.float64]] | None = getattr(
            self, "_gs_table_cache", None
        )
        if cache is None:
            cache = {}
            self._gs_table_cache = cache

        # Grid in log space: both parameters span decades over the transported
        # range (eta ~ 3, <theta^2> ~ 2), and the distribution varies smoothly
        # in their logarithms. The lookup interpolates between the four
        # bracketing nodes rather than snapping to the nearest, because the
        # shape is irreducibly two-dimensional and refining a nearest-bin grid
        # to the same fidelity costs quadratically more table.
        fx = math.log(eta) * _GS_BINS_PER_LOG
        fy = math.log(max(mean_square_angle, _GS_THETA2_MIN)) * _GS_BINS_PER_LOG
        ix, iy = math.floor(fx), math.floor(fy)

        def row(i: int, j: int) -> npt.NDArray[np.float64]:
            table = cache.get((i, j))
            if table is None:
                table = gs_scaled_deflection_table(
                    math.exp(i / _GS_BINS_PER_LOG),
                    math.exp(j / _GS_BINS_PER_LOG),
                    n_u=_GS_TABLE_NODES,
                )
                cache[(i, j)] = table
            return table

        return sample_gs_cos_theta_bilinear(
            row(ix, iy),
            row(ix + 1, iy),
            row(ix, iy + 1),
            row(ix + 1, iy + 1),
            fx - ix,
            fy - iy,
            _GS_TABLE_NODES,
            mean_square_angle,
            rng_state,
        )

    # -- restricted-collision range (concrete: derived from the queries above) --

    def restricted_range(self, energy: float, material: int, delta_cut: float) -> float:
        r"""Restricted-collision range down to ``delta_cut``, in g/cm^2.

        .. math::

            r(E) = \int_{\Delta}^{E} \frac{dE'}{S_{col}(E', \Delta)}

        with the *restricted collision* stopping power of
        :meth:`restricted_stopping_power` — radiative and discrete-Moller losses
        are booked separately by the Class II scheme, so this is exactly the mass
        path over which the transport loop's continuous loss takes ``E`` to the
        cutoff. Together with :meth:`energy_after_mass_path` it defines the
        exact-energy-loss substep (DPM; Sempau et al. 2000,
        doi:10.1088/0031-9155/45/8/315), replacing the first-order
        ``S(E_start) * rho * s`` linearization.

        Concrete on the interface: implementations answer through
        :meth:`restricted_stopping_power`, so the range can never disagree with
        the stopping power it integrates (trapezoid on a dense log grid, cached
        per ``(material, delta_cut)``; node count pinned by the derivative and
        round-trip tests).
        """
        log_e, cumulative = self._range_grid(material, delta_cut)
        e = min(max(energy, delta_cut), _RANGE_GRID_E_MAX)
        return float(np.interp(np.log(e), log_e, cumulative))

    def energy_after_mass_path(
        self, energy: float, material: int, delta_cut: float, mass_path: float
    ) -> float:
        """Energy after a continuous-loss mass path, ``r^-1(r(E) - mass_path)``.

        Clamped to ``[delta_cut, energy]``: a path at or beyond the remaining
        range returns exactly ``delta_cut`` (the loop's range-out branch), and
        interpolation wiggle can never *gain* energy.
        """
        log_e, cumulative = self._range_grid(material, delta_cut)
        remaining = self.restricted_range(energy, material, delta_cut) - mass_path
        if remaining <= 0.0:
            return delta_cut
        e_end = float(np.exp(np.interp(remaining, cumulative, log_e)))
        return min(max(e_end, delta_cut), energy)

    def _range_grid(
        self, material: int, delta_cut: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the cached ``(log E nodes, cumulative range)`` grid for one key."""
        cache: (
            dict[tuple[int, float], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None
        ) = getattr(self, "_restricted_range_cache", None)
        if cache is None:
            cache = {}
            self._restricted_range_cache = cache
        key = (material, delta_cut)
        grids = cache.get(key)
        if grids is None:
            energies = np.geomspace(delta_cut, _RANGE_GRID_E_MAX, _RANGE_GRID_POINTS)
            inv_s = np.array(
                [
                    1.0 / self.restricted_stopping_power(float(e), material, delta_cut)
                    for e in energies
                ]
            )
            cumulative = np.concatenate(
                ([0.0], np.cumsum(np.diff(energies) * 0.5 * (inv_s[1:] + inv_s[:-1])))
            )
            grids = (np.log(energies), cumulative)
            cache[key] = grids
        return grids

    # -- construction -------------------------------------------------------

    def build_tables(
        self, ecut: float, pcut: float, e_max: float, n_points: int | None = None
    ) -> CrossSectionTables:
        """Flatten host-side data into kernel-consumable array handles.

        Generic over implementations: everything flows through the abstract query
        methods above, so a source never flattens itself differently from how it
        answers the host API (that equality is what the table parity tests pin).
        The result is host NumPy; kernel backends upload and cast it. Its contents
        are frozen after construction.

        Parameters
        ----------
        ecut, pcut
            Electron and photon cutoffs in MeV the tables are built at
            (accuracy-defining, AGENTS.md section 2.8).
        e_max
            Upper grid edge in MeV; must cover the highest primary energy.
        n_points
            Grid nodes; defaults to :data:`pyradmc.data.tables.TABLE_POINTS`.
        """
        from pyradmc.data.tables import TABLE_POINTS, build_cross_section_tables

        return build_cross_section_tables(
            self,
            ecut=ecut,
            pcut=pcut,
            e_max=e_max,
            n_points=TABLE_POINTS if n_points is None else n_points,
        )
