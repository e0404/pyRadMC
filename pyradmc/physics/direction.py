"""Direction sampling and rotation on the unit sphere.

Pure scalar functions (AGENTS.md section 2.5). Directions are three floats, never an
array: the tuple return compiles to a multiple-value return on the Warp targets
"""

from __future__ import annotations

import math

from pyradmc.rng import RNGState, uniform

__all__ = ["rotate_direction", "rotate_direction_one_minus_cos", "sample_isotropic_direction"]

# Below this transverse magnitude squared the local-frame construction divides by ~0;
# the direction is then treated as exactly polar. sqrt(1e-20) = 1e-10 bounds the
# angular error of that substitution, far below any physical tolerance in this code.
_POLE_TRANSVERSE2_FLOOR = 1.0e-20


def sample_isotropic_direction(rng_state: RNGState) -> tuple[float, float, float]:
    """Sample a direction uniformly on the unit sphere.

    cos(theta) = 1 - 2u is uniform on (-1, 1] and phi = 2 pi v uniform on [0, 2 pi);
    uniformity of the solid-angle element d(cos theta) d(phi) is the standard inversion
    (e.g. Salvat et al., PENELOPE-2018, sec. 1.4; doi:10.1787/32da5043-en).
    """
    cos_theta = 1.0 - 2.0 * uniform(rng_state)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * uniform(rng_state)
    return (sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta)


def rotate_direction(
    ux: float, uy: float, uz: float, cos_theta: float, phi: float
) -> tuple[float, float, float]:
    r"""Deflect the unit vector (ux, uy, uz) by polar angle theta and azimuth phi.

    The standard direction-cosine update of Monte Carlo transport (e.g. Salvat et al.,
    PENELOPE-2018, eq. 1.131):

    .. math::

        u_x' = u_x \cos\theta
             + \frac{\sin\theta}{\sqrt{1 - u_z^2}}
               (u_x u_z \cos\phi - u_y \sin\phi)

    and cyclic counterparts, with the polar singularity replaced by the exact
    axis-aligned rotation when the transverse magnitude underflows the pole floor.

    :math:`1 - u_z^2` is evaluated as :math:`u_x^2 + u_y^2` — identical for a unit
    vector, but free of the catastrophic cancellation that costs the naive form half
    its digits near the pole (percent-level norm errors at
    :math:`|u_z| \approx 1 - 5 \cdot 10^{-15}`, where the test tier checks it).

    Parameters
    ----------
    ux, uy, uz
        Incident unit direction.
    cos_theta
        Cosine of the polar deflection angle.
    phi
        Azimuthal angle in radians, measured in the plane perpendicular to the
        incident direction.
    """
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    transverse2 = ux * ux + uy * uy
    if transverse2 > _POLE_TRANSVERSE2_FLOOR:
        transverse = math.sqrt(transverse2)
        inv = sin_theta / transverse
        vx = ux * cos_theta + inv * (ux * uz * cos_phi - uy * sin_phi)
        vy = uy * cos_theta + inv * (uy * uz * cos_phi + ux * sin_phi)
        vz = uz * cos_theta - transverse * sin_theta * cos_phi
        return (vx, vy, vz)

    # Travelling (anti)parallel to z: rotate about the axis directly.
    sign = 1.0 if uz > 0.0 else -1.0
    return (sin_theta * cos_phi, sign * sin_theta * sin_phi, sign * cos_theta)


def rotate_direction_one_minus_cos(
    ux: float, uy: float, uz: float, one_minus_cos_theta: float, phi: float
) -> tuple[float, float, float]:
    r"""Deflect (ux, uy, uz) by a polar angle given as :math:`1 - \cos\theta`.

    The rotation of :func:`rotate_direction` (same formula, same azimuth convention,
    same pole treatment), for deflections too small for their cosine to carry:
    with :math:`x = 1 - \cos\theta`,

    .. math::

        \sin\theta = \sqrt{x (2 - x)},

    which is exact to rounding at any angle, whereas
    :math:`\sqrt{1 - \cos^2\theta}` loses the angle entirely once
    :math:`x` falls below the spacing of floats near one — :math:`6 \cdot 10^{-8}`
    in the single precision the kernels run in. That is the regime of
    bremsstrahlung emission angles :math:`\sim 1/\gamma` above a few tens of MeV
    (:func:`pyradmc.physics.brems_bh.sample_bremsstrahlung_one_minus_cos_theta`).

    Parameters
    ----------
    ux, uy, uz
        Incident unit direction.
    one_minus_cos_theta
        :math:`1 - \cos\theta` of the polar deflection, in [0, 2].
    phi
        Azimuthal angle in radians, as in :func:`rotate_direction`.
    """
    cos_theta = 1.0 - one_minus_cos_theta
    sin_theta = math.sqrt(max(0.0, one_minus_cos_theta * (2.0 - one_minus_cos_theta)))
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    transverse2 = ux * ux + uy * uy
    if transverse2 > _POLE_TRANSVERSE2_FLOOR:
        transverse = math.sqrt(transverse2)
        inv = sin_theta / transverse
        vx = ux * cos_theta + inv * (ux * uz * cos_phi - uy * sin_phi)
        vy = uy * cos_theta + inv * (uy * uz * cos_phi + ux * sin_phi)
        vz = uz * cos_theta - transverse * sin_theta * cos_phi
        return (vx, vy, vz)

    # Travelling (anti)parallel to z: rotate about the axis directly.
    sign = 1.0 if uz > 0.0 else -1.0
    return (sin_theta * cos_phi, sign * sin_theta * sin_phi, sign * cos_theta)
