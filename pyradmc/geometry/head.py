"""Treatment-head pre-solve: one-shot host MC through the beam-limiting devices.

``presolve_head`` runs **once per field** on the host: it samples primaries from a
planar source, traces them through the :class:`~pyradmc.geometry.collimation.
BeamLimitingStack` (and optionally an air column), and scores an exit-plane
phase space that the main engines consume as an
:class:`~pyradmc.geometry.phasespace.InMemoryPhaseSpaceSource`. Full transport
starts behind the exit plane; the head physics cost is paid once, and the main
transport loops and their RNG contracts are untouched — this module uses a plain
seeded :class:`numpy.random.Generator` (bit-reproducible per seed on the host,
outside the per-history stream machinery).

Fidelity ``mode`` (the BLD workstream's configuration 1):

- ``"attenuation"`` — deterministic Beer-Lambert weights through stack and air,
  photons only; exactly the physics of
  :class:`~pyradmc.geometry.collimation.CollimatedSource` (test-pinned), as a
  phase space.
- ``"first_compton"`` — additionally forces one Compton interaction per primary:
  in the **stack**, the scattered photon is kept (Klein-Nishina, exit-attenuated
  through the remaining devices) — the collimator-scatter component; in the
  **air column**, both the scattered photon and the recoil **electron** are kept
  — the contaminant-electron component.
- ``"auto"`` — resolves to ``"first_compton"`` (the pre-solve exists to buy
  fidelity; the deterministic wrappers are the cheap path).

Stated approximations, v1 (each with its governing reason):

- **Forced first interaction with weight splitting** (the standard forced-collision
  estimator): each primary contributes a transmitted copy at ``w e^(-tau)`` and an
  interacting copy at ``w (1 - e^(-tau))``; unbiased in expectation for the first
  interaction, second and later scatters in the head are neglected.
- **Non-Compton interactions absorb**: photoelectric absorption in tungsten emits
  no fluorescence (consistent with :mod:`pyradmc.transport.photon`, where the
  W K-edge caveat at 69.5 keV is stated) and the coherent remainder is absorbed
  rather than re-aimed (conservative; coherent scatter is forward-peaked).
- **Electrons born in the devices are neglected**: the CSDA range of a ~1 MeV
  electron in tungsten is ~0.5 g/cm^2, i.e. ~0.3 mm at 19.3 g/cm^3 (NIST ESTAR)
  — escape is a surface effect far below the kept components.
- **The interaction site within a device** is placed at the ray's mid-plane
  crossing of the sampled device (the device is sampled by its share of the
  optical depth; the exponential depth *within* the chord is not resolved). The
  placement error is bounded by the half-thickness and moves only the scattered
  photon's origin and exit attenuation.
- **Air interactions are forced on primaries only** (not on stack-scattered
  photons), and air attenuation is not applied to stack-scattered photons: both
  are sub-percent effects on a ~1 percent component.
- **Kept particles fly straight and lossless to the exit plane**: a 2 MeV
  electron loses ~2.2 keV/cm in air (~0.1 MeV over 50 cm), well inside the
  contaminant component's own modelling error.

Place ``exit_z`` at (or near) the patient surface / transport-grid entry: the
engines fly vacuum outside the grid, so any gap between the exit plane and the
grid is silently interaction-free (under-counted, never double-counted).

Measured throughput (2026-07-15, one host core): ~5e5 rays/s through two jaw
pairs + air, ~1e5 rays/s with a 10-pair rounded-tip MLC added — the MLC's
per-leaf inclusion-exclusion dominates. For a many-field or high-statistics
workload, :func:`pyradmc.backends.warp.presolve.presolve_head_device` runs this
same physics as a Warp kernel on cpu or cuda (validated against this host oracle);
this module stays the reference and the Warp-free host path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyradmc import ECUT_MEV, ELECTRON_MASS_MEV, PCUT_MEV
from pyradmc.data.interface import CrossSectionSource, PhotonProcess
from pyradmc.data.materials import AIR
from pyradmc.geometry.collimation import BeamFrame, BeamLimitingStack
from pyradmc.geometry.phasespace import (
    IAEA_ELECTRON,
    IAEA_PHOTON,
    InMemoryPhaseSpaceSource,
)
from pyradmc.geometry.source import Source

__all__ = ["AirColumn", "presolve_head"]

logger = logging.getLogger(__name__)

_F64 = NDArray[np.float64]

_MODES = ("auto", "first_compton", "attenuation")

# Below this transverse magnitude squared the rotation's local frame degenerates;
# mirrors pyradmc.physics.direction._POLE_TRANSVERSE2_FLOOR.
_POLE_TRANSVERSE2_FLOOR = 1.0e-20


@dataclass(frozen=True)
class AirColumn:
    """The air gap between the last device and the exit plane (local ``w``, cm).

    ``material``/``density`` default to registry air; tests on the analytic
    water-only backend substitute low-density water. The column must start at or
    below the stack's exit face and end at or above ``exit_z`` is *not* required
    — the un-modelled remainder is vacuum, stated in the module docstring.
    """

    z_start: float
    z_end: float
    material: int = AIR
    density: float = 1.205e-3

    def __post_init__(self) -> None:
        """Validate the column extent."""
        if not 0.0 < self.z_start < self.z_end:
            raise ValueError(f"need 0 < z_start < z_end, got {self.z_start}, {self.z_end}")
        if self.density <= 0.0:
            raise ValueError(f"density must be positive, got {self.density}")


class _MuTables:
    """Per-(material, density) log-log tables of macroscopic mu (1/cm).

    Total and Compton channels, built once through the
    :class:`~pyradmc.data.interface.CrossSectionSource` interface (AGENTS.md 2.6).
    """

    def __init__(
        self,
        cross_sections: CrossSectionSource,
        pairs: tuple[tuple[int, float], ...],
        e_min: float,
        e_max: float,
        n_table: int,
    ) -> None:
        if pairs and max(material for material, _ in pairs) >= cross_sections.n_materials:
            raise ValueError(
                "a device or air material index is beyond this cross-section source; "
                "tungsten devices need the tabulated backend"
            )
        energies = np.geomspace(e_min, max(e_max, e_min * (1.0 + 1e-9)), n_table)
        self._log_energies = np.log(energies)
        self._log_total = np.stack(
            [
                np.log(
                    [
                        density * cross_sections.mu_over_rho_total(float(e), material)
                        for e in energies
                    ]
                )
                for material, density in pairs
            ]
        )
        compton_rows = []
        for material, density in pairs:
            values = np.array(
                [
                    density * cross_sections.mu_over_rho(float(e), material, PhotonProcess.COMPTON)
                    for e in energies
                ]
            )
            compton_rows.append(np.log(np.maximum(values, 1e-300)))
        self._log_compton = np.stack(compton_rows)

    def total(self, row: int | NDArray[np.int64], energies: _F64) -> _F64:
        """Macroscopic total mu (1/cm), row-wise when ``row`` is an array."""
        return self._lookup(self._log_total, row, energies)

    def compton(self, row: int | NDArray[np.int64], energies: _F64) -> _F64:
        """Macroscopic Compton mu (1/cm), row-wise when ``row`` is an array."""
        return self._lookup(self._log_compton, row, energies)

    def _lookup(self, table: _F64, row: int | NDArray[np.int64], energies: _F64) -> _F64:
        log_e = np.log(np.asarray(energies, dtype=np.float64))
        if isinstance(row, (int, np.integer)):
            return np.asarray(np.exp(np.interp(log_e, self._log_energies, table[int(row)])))
        out = np.empty(log_e.shape, dtype=np.float64)
        for index in np.unique(row):
            mask = row == index
            out[mask] = np.exp(np.interp(log_e[mask], self._log_energies, table[int(index)]))
        return out


def _sample_compton_ratios(energies: _F64, generator: np.random.Generator) -> _F64:
    """Vectorized Kahn rejection for the Klein-Nishina energy ratio r = E'/E.

    The array-rate sibling of
    :func:`pyradmc.physics.compton.sample_compton_energy_ratio` (Kahn 1956;
    PENELOPE-2018 sec. 2.3, doi:10.1787/32da5043-en) — deliberate, contained
    duplication (AGENTS.md 2.5 keeps ``physics/`` scalar), test-pinned against
    the scalar oracle in ``tests/physics/test_presolve_compton.py``. Undecided
    lanes redraw until every ray has accepted.
    """
    energies = np.asarray(energies, dtype=np.float64)
    alpha = energies / ELECTRON_MASS_MEV
    two_alpha = 2.0 * alpha
    branch_probability = (1.0 + two_alpha) / (9.0 + two_alpha)
    ratios = np.empty_like(energies)
    undecided = np.ones(energies.shape, dtype=bool)
    while undecided.any():
        lanes = np.flatnonzero(undecided)
        u1, u2, u3 = generator.random((3, lanes.size))
        ta = two_alpha[lanes]
        first_branch = u1 <= branch_probability[lanes]
        x = np.where(first_branch, 1.0 + ta * u2, (1.0 + ta) / (1.0 + ta * u2))
        cos_theta = 1.0 - (x - 1.0) / alpha[lanes]
        accept = np.where(
            first_branch,
            u3 <= 4.0 * (x - 1.0) / (x * x),
            u3 <= 0.5 * (cos_theta * cos_theta + 1.0 / x),
        )
        accepted = lanes[accept]
        ratios[accepted] = 1.0 / x[accept]
        undecided[accepted] = False
    return ratios


def _compton_cos_thetas(energies: _F64, ratios: _F64) -> _F64:
    """Vectorized Compton relation; see :func:`pyradmc.physics.compton.compton_cos_theta`."""
    alpha = np.asarray(energies, dtype=np.float64) / ELECTRON_MASS_MEV
    cos_theta = 1.0 + 1.0 / alpha - 1.0 / (alpha * np.asarray(ratios, dtype=np.float64))
    return np.minimum(1.0, np.maximum(-1.0, cos_theta))


def _compton_electron_cos_thetas(energies: _F64, ratios: _F64) -> _F64:
    """Vectorized recoil-electron angle.

    See :func:`pyradmc.physics.compton.compton_electron_cos_theta`.
    """
    energies = np.asarray(energies, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)
    recoil = energies * (1.0 - ratios)
    p_electron = np.sqrt(recoil * (recoil + 2.0 * ELECTRON_MASS_MEV))
    cos_gamma = _compton_cos_thetas(energies, ratios)
    cos_electron = (energies - ratios * energies * cos_gamma) / p_electron
    return np.minimum(1.0, np.maximum(-1.0, cos_electron))


def _rotate_directions(directions: _F64, cos_theta: _F64, phi: _F64) -> _F64:
    """Vectorized direction-cosine update.

    The array sibling of :func:`pyradmc.physics.direction.rotate_direction`
    (PENELOPE-2018 eq. 1.131), including the polar-degeneracy branch.
    """
    d = np.asarray(directions, dtype=np.float64)
    ux, uy, uz = d[:, 0], d[:, 1], d[:, 2]
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    transverse2 = ux * ux + uy * uy
    polar = transverse2 <= _POLE_TRANSVERSE2_FLOOR
    transverse = np.sqrt(np.where(polar, 1.0, transverse2))
    inv = sin_theta / transverse
    vx = ux * cos_theta + inv * (ux * uz * cos_phi - uy * sin_phi)
    vy = uy * cos_theta + inv * (uy * uz * cos_phi + ux * sin_phi)
    vz = uz * cos_theta - transverse * sin_theta * cos_phi
    sign = np.where(uz > 0.0, 1.0, -1.0)
    out = np.stack(
        [
            np.where(polar, sin_theta * cos_phi, vx),
            np.where(polar, sign * sin_theta * sin_phi, vy),
            np.where(polar, sign * cos_theta, vz),
        ],
        axis=1,
    )
    return out


class _Population:
    """Column accumulator for the kept exit-plane particles."""

    def __init__(self) -> None:
        self._parts: list[dict[str, np.ndarray]] = []

    def add(
        self,
        code: int,
        energy: _F64,
        points: _F64,
        directions: _F64,
        weight: _F64,
        keep: NDArray[np.bool_],
    ) -> float:
        """Append the kept rows; return the dropped weight-energy for the log."""
        dropped = float(np.sum(weight[~keep] * energy[~keep]))
        if keep.any():
            self._parts.append(
                {
                    "particle_type": np.full(int(keep.sum()), code, dtype=np.int32),
                    "energy": energy[keep],
                    "x": points[keep, 0],
                    "y": points[keep, 1],
                    "z": points[keep, 2],
                    "ux": directions[keep, 0],
                    "uy": directions[keep, 1],
                    "uz": directions[keep, 2],
                    "weight": weight[keep],
                }
            )
        return dropped

    def build(self, n_primaries: int) -> InMemoryPhaseSpaceSource:
        """Assemble the source, normalized **per original pre-solve primary**.

        The population holds ``N`` rows for ``n_primaries`` sampled primaries
        (forced-interaction secondaries make ``N > n``); uniform resampling of
        the rows would dilute a run's per-history dose by ``n / N``. Scaling
        every stored weight by ``N / n`` makes ``run(phase_space, M)`` estimate
        the dose per original primary — directly comparable, history for
        history, to transporting the planar source itself.
        """
        if not self._parts:
            raise ValueError("the pre-solve kept no particles; check the geometry and cutoffs")
        merged = {
            name: np.concatenate([part[name] for part in self._parts]) for name in self._parts[0]
        }
        merged["weight"] = merged["weight"] * (merged["weight"].size / n_primaries)
        return InMemoryPhaseSpaceSource(**merged)


def _roulette_below(weight: _F64, floor: float, generator: np.random.Generator) -> _F64:
    """Fair Russian roulette on secondaries below the weight floor.

    A particle at ``w < floor`` survives with probability ``w / floor`` at weight
    ``floor`` — expected weight preserved exactly, killed rows set to zero (the
    population accumulator drops them). Forced air interactions in particular
    produce one secondary pair per primary at ~0.1-percent weights; without this
    the phase space fills with rows that uniform resampling then wastes transport
    on. Host-side pre-solve RNG only — no per-history draw contract exists here.
    """
    if floor <= 0.0:
        return weight
    playing = (weight > 0.0) & (weight < floor)
    if not playing.any():
        return weight
    survives = generator.random(weight.shape[0]) < weight / floor
    out = np.where(playing & survives, floor, weight)
    out[playing & ~survives] = 0.0
    return out


def resolve_presolve(
    stack: BeamLimitingStack | None,
    frame: BeamFrame | None,
    air: AirColumn | None,
    exit_z: float,
    mode: str,
) -> tuple[str, BeamFrame]:
    """Validate the pre-solve geometry and resolve ``mode`` and the beam frame.

    Shared by the host :func:`presolve_head` and the Warp device pre-solve so the
    two entry points enforce one contract: a valid ``mode``, a frame (the stack's
    if given), and an ``exit_z`` at or below the stack exit and the air-column end.
    Returns the resolved mode (``auto`` -> ``first_compton``) and the frame.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    resolved = "first_compton" if mode == "auto" else mode
    if stack is not None:
        frame = stack.frame
    if frame is None:
        raise ValueError("without a stack, pass the beam frame explicitly")
    if stack is not None and exit_z < stack.exit_z:
        raise ValueError(f"exit_z {exit_z} is above the stack exit {stack.exit_z}")
    if air is not None:
        if stack is not None and air.z_start < stack.exit_z:
            raise ValueError(
                f"the air column starts at {air.z_start}, inside the stack (exit {stack.exit_z})"
            )
        if exit_z < air.z_end:
            raise ValueError(f"exit_z {exit_z} is above the air column end {air.z_end}")
    return resolved, frame


def presolve_head(
    source: Source,
    *,
    cross_sections: CrossSectionSource,
    n_histories: int,
    seed: int,
    exit_z: float,
    stack: BeamLimitingStack | None = None,
    frame: BeamFrame | None = None,
    air: AirColumn | None = None,
    mode: str = "auto",
    pcut: float = PCUT_MEV,
    ecut: float = ECUT_MEV,
    weight_floor: float = 1.0e-3,
    e_min: float = 0.010,
    n_table: int = 512,
) -> InMemoryPhaseSpaceSource:
    """Pre-solve the head once and return the exit-plane phase space.

    ``exit_z`` and the :class:`AirColumn` extents are local ``w`` coordinates in
    the beam frame (the stack's frame; pass ``frame`` explicitly when running
    without a stack). The source must emit photons only (the planar Gaussian-spot
    source is the intended input). Physics, modes and stated approximations: the
    module docstring. Sub-cutoff and backward-going particles are dropped with
    their weight-energy logged; the kept population sizes are logged too.

    ``weight_floor`` plays fair Russian roulette (:func:`_roulette_below`) on
    forced-interaction **secondaries** below that weight — population control for
    the stored phase space, never applied to the transmitted primaries, which are
    the deterministic backbone. Zero disables it.
    """
    resolved, frame = resolve_presolve(stack, frame, air, exit_z, mode)

    columns = source.sample_batch(seed, 0, n_histories)
    if not np.all(columns["particle_type"] == IAEA_PHOTON):
        raise ValueError("the pre-solve transports photon sources only")
    energy = columns["energy"].astype(np.float64)
    points = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
    directions = np.stack([columns["ux"], columns["uy"], columns["uz"]], axis=1).astype(np.float64)
    weight = columns["weight"].astype(np.float64)

    # A dedicated stream, decoupled from the source's own PCG64(seed) batch stream.
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence((seed, 0x48454144))))

    pairs: tuple[tuple[int, float], ...] = ()
    if stack is not None:
        pairs = tuple(zip(stack.materials, stack.densities, strict=True))
    air_row = len(pairs)
    if air is not None:
        pairs = (*pairs, (air.material, air.density))
    tables = _MuTables(cross_sections, pairs, e_min, source.max_energy, n_table) if pairs else None

    population = _Population()
    dropped = 0.0

    # --- the beam-limiting stack -------------------------------------------------
    if stack is not None and tables is not None:
        thickness = stack.path_lengths(points, directions)  # (n, n_devices), full line
        taus = np.stack(
            [tables.total(i, energy) * thickness[:, i] for i in range(len(stack.devices))],
            axis=1,
        )
        tau_total = taus.sum(axis=1)
        transmitted_weight = weight * np.exp(-tau_total)

        if resolved == "first_compton":
            interacting = tau_total > 0.0
            if interacting.any():
                idx = np.flatnonzero(interacting)
                tau_i = taus[idx]
                share = np.cumsum(tau_i, axis=1) / tau_i.sum(axis=1, keepdims=True)
                device = np.argmax(generator.random(idx.size)[:, None] <= share, axis=1)
                # Interaction at the sampled device's mid-plane crossing (stated
                # approximation): t to the mid-plane along the ray, in the frame.
                z_mid = np.array(
                    [(d.z_top + d.z_bottom) / 2.0 for d in stack.devices], dtype=np.float64
                )
                p_local, d_local = frame.to_local(points[idx], directions[idx])
                t_mid = (z_mid[device] - p_local[:, 2]) / d_local[:, 2]
                site = points[idx] + t_mid[:, None] * directions[idx]

                e_in = energy[idx]
                compton_fraction = tables.compton(device, e_in) / tables.total(device, e_in)
                interact_weight = weight[idx] * (1.0 - np.exp(-tau_total[idx]))
                ratios = _sample_compton_ratios(e_in, generator)
                e_out = ratios * e_in
                phi = 2.0 * np.pi * generator.random(idx.size)
                d_out = _rotate_directions(directions[idx], _compton_cos_thetas(e_in, ratios), phi)
                exit_thickness = stack.path_lengths(site, d_out, from_origin=True)
                tau_out = np.stack(
                    [
                        tables.total(i, e_out) * exit_thickness[:, i]
                        for i in range(len(stack.devices))
                    ],
                    axis=1,
                ).sum(axis=1)
                scatter_weight = _roulette_below(
                    interact_weight * compton_fraction * np.exp(-tau_out),
                    weight_floor,
                    generator,
                )
                dropped += _propagate_and_add(
                    population,
                    IAEA_PHOTON,
                    e_out,
                    site,
                    d_out,
                    scatter_weight,
                    frame,
                    exit_z,
                    pcut,
                )
    else:
        transmitted_weight = weight.copy()

    # --- the air column (primaries only; stated approximation) --------------------
    if air is not None and tables is not None:
        _, d_local = frame.to_local(points, directions)
        forward = d_local[:, 2] > 0.0
        length = np.where(
            forward, (air.z_end - air.z_start) / np.where(forward, d_local[:, 2], 1.0), 0.0
        )
        mu_air = tables.total(air_row, energy)
        tau_air = mu_air * length
        pre_air_weight = transmitted_weight.copy()
        transmitted_weight = transmitted_weight * np.exp(-tau_air)

        if resolved == "first_compton":
            interacting = forward & (tau_air > 0.0)
            if interacting.any():
                idx = np.flatnonzero(interacting)
                e_in = energy[idx]
                interact_weight = pre_air_weight[idx] * (1.0 - np.exp(-tau_air[idx]))
                compton_fraction = tables.compton(air_row, e_in) / tables.total(air_row, e_in)
                pair_weight = interact_weight * compton_fraction

                # Exponential depth within the column, inverted from the truncated CDF.
                u_depth = generator.random(idx.size)
                depth = -np.log(1.0 - u_depth * (1.0 - np.exp(-tau_air[idx]))) / mu_air[idx]
                p_local, d_local_i = frame.to_local(points[idx], directions[idx])
                t_start = (air.z_start - p_local[:, 2]) / d_local_i[:, 2]
                site = points[idx] + (t_start + depth)[:, None] * directions[idx]

                ratios = _sample_compton_ratios(e_in, generator)
                e_photon = ratios * e_in
                e_electron = e_in - e_photon
                phi = 2.0 * np.pi * generator.random(idx.size)
                d_photon = _rotate_directions(
                    directions[idx], _compton_cos_thetas(e_in, ratios), phi
                )
                d_electron = _rotate_directions(
                    directions[idx], _compton_electron_cos_thetas(e_in, ratios), phi + np.pi
                )
                # The photon and electron arms are rouletted independently: each
                # arm stays unbiased in expectation (the stored phase space is a
                # fluence estimator, not a per-event ledger).
                dropped += _propagate_and_add(
                    population,
                    IAEA_PHOTON,
                    e_photon,
                    site,
                    d_photon,
                    _roulette_below(pair_weight, weight_floor, generator),
                    frame,
                    exit_z,
                    pcut,
                )
                dropped += _propagate_and_add(
                    population,
                    IAEA_ELECTRON,
                    e_electron,
                    site,
                    d_electron,
                    _roulette_below(pair_weight, weight_floor, generator),
                    frame,
                    exit_z,
                    ecut,
                )

    dropped += _propagate_and_add(
        population,
        IAEA_PHOTON,
        energy,
        points,
        directions,
        transmitted_weight,
        frame,
        exit_z,
        pcut,
    )

    result = population.build(n_histories)
    logger.info(
        "pre-solve: %d histories -> %d exit-plane particles (mode=%s); "
        "dropped weight-energy %.4g MeV",
        n_histories,
        len(result),
        resolved,
        dropped,
    )
    return result


def _propagate_and_add(
    population: _Population,
    code: int,
    energy: _F64,
    points: _F64,
    directions: _F64,
    weight: _F64,
    frame: BeamFrame,
    exit_z: float,
    cutoff: float,
) -> float:
    """Slide particles down their rays to the exit plane and keep the viable ones.

    Kept: forward-going (positive local ``w`` direction), at or above the cutoff,
    positive weight. Returns the dropped weight-energy.
    """
    p_local, d_local = frame.to_local(points, directions)
    forward = d_local[:, 2] > 0.0
    t_exit = np.where(
        forward, (exit_z - p_local[:, 2]) / np.where(forward, d_local[:, 2], 1.0), 0.0
    )
    exit_points = points + t_exit[:, None] * directions
    keep = forward & (t_exit >= 0.0) & (energy >= cutoff) & (weight > 0.0)
    return population.add(code, energy, exit_points, directions, weight, keep)
