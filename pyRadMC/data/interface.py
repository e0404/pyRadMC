"""Cross-section and stopping-power access interface.

This module defines the *only* permitted access path to interaction data. No physics
routine may hardcode a cross-section formula; see AGENTS.md section 2.6.

Two implementations are planned:

``analytic``
    Closed-form parameterizations. Fast to stand up, adequate for Phase 0 and 1,
    insufficient for final accuracy.

``tabulated``
    Interpolated tables (e.g. EPDL/EEDL-derived, or NIST XCOM/ESTAR-derived). Must
    integrate *products* over sub-grid intervals rather than evaluating separately
    averaged bin quantities at bin centers; see AGENTS.md section 2.7 and
    ``tests/unit/test_table_integration.py``.

All energies are in MeV. All macroscopic cross-sections are in 1/cm. All mass
attenuation and mass stopping-power quantities are per unit mass density, i.e. in
cm^2/g and MeV cm^2/g respectively.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["CrossSectionSource", "PhotonProcess"]


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
    def scattering_power(self, energy: float, material: int) -> float:
        """Mass angular scattering power, in rad^2 cm^2/g.

        Drives the multiple-elastic-scattering hinge deflection.
        """

    # -- construction -------------------------------------------------------

    @abstractmethod
    def build_tables(self) -> object:
        """Flatten host-side data into kernel-consumable array handles.

        The returned object is backend-specific (NumPy arrays for ``ref``, Warp arrays
        or textures for ``warp``). Its contents are frozen after construction.
        """
