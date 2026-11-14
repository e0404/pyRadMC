"""Selectable physics models, one choice per process (AGENTS.md section 2.10).

Three processes have a model that a run may choose:

``bremsstrahlung``
    How an electron's radiative loss becomes photons.
``pair``
    How a pair-producing photon's kinetic energy and direction are shared between
    the electron and the positron.
``positron``
    How a positron differs from an electron in flight, and where it annihilates.
    One option, because the discrete cross-section and the annihilation channel
    are only consistent together.

A run names its choice with :class:`PhysicsOptions`. Every field defaults to
``"auto"``, which :func:`resolve_physics` turns into a concrete model from the run
configuration — the source's maximum energy and primary kinds — by reading
:data:`PHYSICS_MODELS`. ``"auto"`` is the certified path: it is what the validation
tier runs. An explicit model name overrides it for that one process and is the
caller's responsibility.

**This module is the registration point.** :data:`PHYSICS_MODELS` is the only place
a model is named, and the only place ``"auto"`` is decided. Registering a model is
one edit to its process row; the same change must then implement it where the
process is sampled, on both backends — the reference loops raise on a registered
model they do not implement, and the Warp engine refuses one its kernels have no
code for (``KERNEL_PHYSICS_CODES`` in :mod:`pyradmc.backends.warp.kernels`), so a
half-registered model cannot run silently as another.

Nothing here samples anything: this is configuration, resolved once per run on the
host before any history starts. The engines log the resolved choice once at INFO.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from pyradmc import HIGH_ENERGY_PHYSICS_THRESHOLD_MEV

__all__ = [
    "AUTO",
    "CLINICAL_PHYSICS",
    "PHYSICS_MODELS",
    "PRIMARY_KINDS",
    "PhysicsOptions",
    "ProcessModels",
    "ResolvedPhysics",
    "require_implemented",
    "resolve_physics",
    "run_primary_kinds",
]

AUTO = "auto"
"""The field value that leaves a process to :func:`resolve_physics`."""

PRIMARY_KINDS: tuple[str, ...] = ("photon", "electron", "positron")
"""The particle kinds a run may start; the names ``Primary.kind`` uses."""


@dataclass(frozen=True)
class ProcessModels:
    """One process's row in :data:`PHYSICS_MODELS`.

    ``models`` are the registered names, in registration order. ``auto_clinical``
    is what ``"auto"`` resolves to when the run's maximum energy is at or below
    :data:`pyradmc.HIGH_ENERGY_PHYSICS_THRESHOLD_MEV`, ``auto_high_energy`` what it
    resolves to above it. Both must be registered names.
    """

    models: tuple[str, ...]
    auto_clinical: str
    auto_high_energy: str

    def __post_init__(self) -> None:
        """Refuse a row whose ``"auto"`` choices could not be selected by name."""
        if AUTO in self.models:
            raise ValueError(f"{AUTO!r} is reserved and cannot be a model name")
        for choice in (self.auto_clinical, self.auto_high_energy):
            if choice not in self.models:
                raise ValueError(
                    f"auto choice {choice!r} is not a registered model; "
                    f"registered: {_quoted(self.models)}"
                )


PHYSICS_MODELS: Mapping[str, ProcessModels] = MappingProxyType(
    {
        # "thin-target": 1/k spectrum, collinear emission (pyradmc.physics.brems).
        # "bethe-heitler": screened Bethe-Heitler spectrum, 1/gamma emission angle
        # (pyradmc.physics.brems_bh). Both carry the same radiative stopping power.
        "bremsstrahlung": ProcessModels(
            models=("thin-target", "bethe-heitler"),
            auto_clinical="thin-target",
            auto_high_energy="bethe-heitler",
        ),
        # "uniform-split": kinetic energy shared uniformly, both leptons forward
        # (pyradmc.transport.photon). "bethe-heitler": screened Bethe-Heitler
        # sharing, each lepton at its own leading-order angle
        # (pyradmc.physics.pair). The pair cross-section is the same for both.
        "pair": ProcessModels(
            models=("uniform-split", "bethe-heitler"),
            auto_clinical="uniform-split",
            auto_high_energy="bethe-heitler",
        ),
        # "moller-at-rest": transported as an electron (Moller, no Bhabha),
        # annihilating at rest. "bhabha-in-flight": the positron's own collision
        # stopping power and range, Bhabha delta rays, and two-photon annihilation
        # in flight as a competing channel (pyradmc.physics.bhabha,
        # pyradmc.physics.annihilation); one option, because the discrete
        # cross-sections are only consistent together (AGENTS.md 2.10).
        "positron": ProcessModels(
            models=("moller-at-rest", "bhabha-in-flight"),
            auto_clinical="moller-at-rest",
            auto_high_energy="bhabha-in-flight",
        ),
    }
)
"""The registration table: process name -> its models and its ``"auto"`` rule.

The keys are the field names of :class:`PhysicsOptions` and
:class:`ResolvedPhysics`, in the same order.

Every process has two models: the one every clinical run has always computed, which
``"auto"`` keeps at or below the threshold, and a more detailed one that ``"auto"``
takes above it. Adding a model appends its name to ``models``; making it the
above-threshold default changes ``auto_high_energy``. Changing ``auto_clinical``
changes what every clinical run computes: that is replacing a default, and needs
the dosimetric evidence AGENTS.md 2.8 asks for.
"""


def _quoted(names: Collection[str]) -> str:
    return ", ".join(repr(name) for name in names)


def _check_model(process: str, model: str, *, allow_auto: bool) -> None:
    registered = PHYSICS_MODELS[process].models
    if model in registered or (allow_auto and model == AUTO):
        return
    expected = f"{AUTO!r} or one of {_quoted(registered)}" if allow_auto else _quoted(registered)
    raise ValueError(f"unknown {process} model {model!r}; expected {expected}")


@dataclass(frozen=True)
class PhysicsOptions:
    """A run's requested physics models: ``"auto"`` or a model name per process.

    Pass it as an engine's ``physics`` argument. The default — every field
    ``"auto"`` — is the certified configuration; :func:`resolve_physics` picks each
    model from the run. Naming a model pins that process whatever the run looks
    like, and leaves the others on ``"auto"``.

    Registered models (:data:`PHYSICS_MODELS`):

    ``bremsstrahlung``
        ``"thin-target"`` — a ``1/k`` photon spectrum, emitted along the electron's
        direction (:mod:`pyradmc.physics.brems`). What ``"auto"`` selects at or
        below the high-energy threshold.

        ``"bethe-heitler"`` — the screened Bethe-Heitler photon spectrum with the
        Coulomb correction, emitted at an angle of order ``1/gamma`` to the
        electron (:mod:`pyradmc.physics.brems_bh`, which lists its
        approximations). Normalized to the same radiative stopping power as
        ``"thin-target"``: the two differ in how the radiated energy is shared
        among photons, not in how much is radiated. What ``"auto"`` selects above
        the threshold.
    ``pair``
        ``"uniform-split"`` — the kinetic energy is shared uniformly between the
        electron and the positron, both emitted forward
        (:mod:`pyradmc.transport.photon`). What ``"auto"`` selects at or below
        the high-energy threshold.

        ``"bethe-heitler"`` — the sharing follows the screened Bethe-Heitler
        differential cross-section (uniform below 2 MeV, where that expression
        fails), and each lepton leaves at its own angle of order ``1/gamma`` to
        the photon, with independent azimuths (:mod:`pyradmc.physics.pair`;
        approximations in :mod:`pyradmc.data.pair`). The pair cross-section is
        the source's either way: the two differ in how a pair's energy and
        direction are distributed, not in how often a pair is made. What
        ``"auto"`` selects above the threshold.
    ``positron``
        ``"moller-at-rest"`` — a positron is transported as an electron (Moller
        scattering, no Bhabha distinction) and annihilates at rest
        (:mod:`pyradmc.transport.electron`). What ``"auto"`` selects at or below
        the high-energy threshold.

        ``"bhabha-in-flight"`` — a positron loses energy by its own restricted
        collision stopping power, throws Bhabha delta rays (up to its whole
        kinetic energy) and may annihilate in flight into two photons that share
        its kinetic energy and both rest masses; one that reaches the cutoff
        still annihilates at rest (:mod:`pyradmc.physics.bhabha`,
        :mod:`pyradmc.physics.annihilation`; what remains the electron's is
        listed in :mod:`pyradmc.transport.electron`). What ``"auto"`` selects
        above the threshold.

    An unknown name raises ``ValueError`` listing the valid ones.
    """

    bremsstrahlung: str = AUTO
    pair: str = AUTO
    positron: str = AUTO

    def __post_init__(self) -> None:
        """Refuse a name that is not ``"auto"`` or registered for its process."""
        for process in PHYSICS_MODELS:
            _check_model(process, getattr(self, process), allow_auto=True)


@dataclass(frozen=True)
class ResolvedPhysics:
    """The physics models a run actually uses: one registered name per process.

    What :func:`resolve_physics` returns, what the transport loops and kernels are
    handed, and what a result's provenance records. Never ``"auto"``.

    ``reason`` says why these models were chosen, for the log. It is commentary:
    two resolutions that name the same models are equal and hash alike.
    """

    bremsstrahlung: str
    pair: str
    positron: str
    reason: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        """Refuse anything but a registered model name in every field."""
        for process in PHYSICS_MODELS:
            _check_model(process, getattr(self, process), allow_auto=False)

    def summary(self) -> str:
        """One-line digest of the models, for logs and provenance."""
        return " ".join(f"{process}={getattr(self, process)}" for process in PHYSICS_MODELS)


CLINICAL_PHYSICS = ResolvedPhysics(
    **{process: row.auto_clinical for process, row in PHYSICS_MODELS.items()},
    reason="the at-or-below-threshold models, taken without a run configuration",
)
"""What ``"auto"`` resolves to at or below the high-energy threshold.

The default of the transport loops in :mod:`pyradmc.transport`, for callers that
drive a loop directly rather than through an engine (single-interaction tests).
Engines never use it: they resolve from the run.
"""

_PRESETS: Mapping[str, PhysicsOptions] = MappingProxyType({AUTO: PhysicsOptions()})
# The bare strings an engine's ``physics`` argument accepts. Only ``"auto"``: it is
# the spelling of the default. A model name is not a preset — it belongs to one
# process, and PhysicsOptions is where a process is named.


def _coerce(physics: PhysicsOptions | str) -> PhysicsOptions:
    if isinstance(physics, PhysicsOptions):
        return physics
    if isinstance(physics, str):
        preset = _PRESETS.get(physics)
        if preset is None:
            raise ValueError(
                f"unknown physics preset {physics!r}; expected {_quoted(_PRESETS)} or a "
                "PhysicsOptions naming a model per process"
            )
        return preset
    raise TypeError(
        f"physics must be a PhysicsOptions or a preset string, got {type(physics).__name__}"
    )


def resolve_physics(
    physics: PhysicsOptions | str = AUTO,
    *,
    max_energy_mev: float,
    primary_kinds: Collection[str],
) -> ResolvedPhysics:
    """Resolve every ``"auto"`` field from the run configuration.

    Pure: the same arguments give the same result, nothing is logged and no state
    is read beyond :data:`PHYSICS_MODELS` and the threshold constant. An engine
    calls it once per run, before any history starts.

    The rule, per process:

    - A field naming a model resolves to that model.
    - An ``"auto"`` field resolves to the row's ``auto_high_energy`` when
      ``max_energy_mev`` exceeds
      :data:`pyradmc.HIGH_ENERGY_PHYSICS_THRESHOLD_MEV`, and to its
      ``auto_clinical`` otherwise.

    The threshold is on the *maximum* energy the source can emit, not a mean: it
    is the same bound the cross-section tables are sized to, and a model is
    needed wherever its regime is reached at all. It is provisional (see the
    constant) and accuracy-defining — moving it moves which physics a run near it
    computes.

    ``primary_kinds`` is part of the configuration and is named in the reason, but
    today's rule keys on energy alone: a secondary of any kind can reach an energy
    near the primary's, so the kind a history starts as does not bound the regime
    its family explores. A rule that reads it belongs here, next to the energy
    test, and nowhere else.

    Parameters
    ----------
    physics
        A :class:`PhysicsOptions`, or the preset string ``"auto"`` (all fields
        ``"auto"``).
    max_energy_mev
        Highest primary energy the source can emit, in MeV
        (``Source.max_energy``).
    primary_kinds
        The kinds of particle the run starts: any of ``"photon"``,
        ``"electron"``, ``"positron"``; see :func:`run_primary_kinds`.

    Raises
    ------
    ValueError
        For an unknown preset, a non-positive or non-finite energy, or an empty or
        unknown set of kinds.
    TypeError
        If ``physics`` is neither a :class:`PhysicsOptions` nor a string.
    """
    options = _coerce(physics)
    if not (math.isfinite(max_energy_mev) and max_energy_mev > 0.0):
        raise ValueError(f"max_energy_mev must be positive and finite, got {max_energy_mev}")
    kinds = frozenset(primary_kinds)
    if not kinds:
        raise ValueError("need at least one primary kind")
    unknown = sorted(kinds.difference(PRIMARY_KINDS))
    if unknown:
        raise ValueError(
            f"unknown primary kind(s) {_quoted(unknown)}; expected any of {_quoted(PRIMARY_KINDS)}"
        )

    high_energy = max_energy_mev > HIGH_ENERGY_PHYSICS_THRESHOLD_MEV
    models: dict[str, str] = {}
    explicit: list[str] = []
    for process, row in PHYSICS_MODELS.items():
        requested = getattr(options, process)
        if requested == AUTO:
            models[process] = row.auto_high_energy if high_energy else row.auto_clinical
        else:
            models[process] = requested
            explicit.append(f"{process}={requested}")

    started = "/".join(kind for kind in PRIMARY_KINDS if kind in kinds)
    side = "above" if high_energy else "at or below"
    reason = (
        f"{started} primaries up to {max_energy_mev:g} MeV, {side} the "
        f"{HIGH_ENERGY_PHYSICS_THRESHOLD_MEV:g} MeV high-energy threshold"
    )
    if explicit:
        reason += "; " + ", ".join(explicit) + " set explicitly"
    return ResolvedPhysics(**models, reason=reason)


def require_implemented(process: str, model: str, implemented: Collection[str], where: str) -> None:
    """Raise unless ``where`` implements ``model`` for ``process``.

    The guard every implementation site runs on the model it is handed. A name in
    :data:`PHYSICS_MODELS` is a promise that both backends implement it; this is
    what turns a broken promise into an error instead of a run that silently
    used a different model.
    """
    if model not in implemented:
        raise NotImplementedError(
            f"{process} model {model!r} is registered but not implemented in {where}; "
            f"implemented there: {_quoted(implemented)}"
        )


def run_primary_kinds(source: object, fallback: str) -> frozenset[str]:
    """Return the primary kinds a run starts, as far as its configuration declares them.

    A source that sets ``Primary.kind`` per record declares the kinds it holds
    through ``particle_kinds`` (:attr:`pyradmc.geometry.source.Source.particle_kinds`).
    One that leaves the kind to the engine declares ``None``, and the run starts
    ``fallback`` — the engine's ``primary_kind``.
    """
    declared: Collection[str] | None = getattr(source, "particle_kinds", None)
    return frozenset({fallback}) if declared is None else frozenset(declared)
