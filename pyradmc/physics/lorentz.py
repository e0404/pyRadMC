r"""Charged-particle motion in a static magnetic field.

Pure scalar functions (AGENTS.md section 2.5): the kinematic scale of the motion
(momentum, gyroradius), the exact rotation of a direction about a fixed lab axis, and
the signed angle a chord of a gyration orbit turns through. Directions are three
floats, never an array, and the tuple return compiles to a multiple-value return on
the Warp targets.

The equation of motion for charge :math:`q` with momentum :math:`\mathbf{p}` in a
static field :math:`\mathbf{B}` is

.. math::

    \frac{d\mathbf{p}}{dt} = q\,\mathbf{v} \times \mathbf{B},

whose solution is a helix: the component of the velocity along
:math:`\hat{\mathbf{B}}` is constant and the transverse component rotates at the
cyclotron frequency. The Lorentz force is perpendicular to the velocity and **does no
work**: the kinetic energy, and therefore the path length per unit energy loss, is
unchanged by the field. In a condensed-history step this decouples the field from the
continuous energy loss and from multiple scattering: the step is transported as usual
and the direction is rotated about :math:`\hat{\mathbf{B}}` by the angle the chord
subtends on the gyration circle. Bielajew (1988), "Electron transport in E and B
fields", doi:10.1007/978-1-4613-1059-4_19; Malkov & Rogers (2016), Med. Phys. 43,
4447, doi:10.1118/1.4954318, sec. 2.A.

The transport loops own the step-length cap that keeps the chord-versus-arc error
small; nothing here depends on it.
"""

from __future__ import annotations

import math

from pyradmc import ELECTRON_MASS_MEV, MEV_C_PER_TESLA_CM

__all__ = [
    "gyroradius_cm",
    "helix_chord",
    "lorentz_rotation_angle",
    "momentum_mev_c",
    "rotate_about_axis",
]


def momentum_mev_c(e_kin_mev: float) -> float:
    r"""Momentum of an electron or positron of kinetic energy ``e_kin_mev``, in MeV/c.

    From the relativistic energy-momentum relation
    :math:`(pc)^2 = (T + m_e c^2)^2 - (m_e c^2)^2`,

    .. math::

        pc = \sqrt{T\,(T + 2 m_e c^2)}.

    Parameters
    ----------
    e_kin_mev
        Kinetic energy T in MeV, non-negative.
    """
    return math.sqrt(e_kin_mev * (e_kin_mev + 2.0 * ELECTRON_MASS_MEV))


def gyroradius_cm(e_kin_mev: float, b_mag_tesla: float) -> float:
    r"""Gyroradius (Larmor radius) of an electron or positron, in cm.

    Balancing the Lorentz force against the centripetal acceleration of the
    transverse motion gives the magnetic rigidity :math:`p = |q| B r`, i.e.

    .. math::

        r_g[\mathrm{cm}] = \frac{p[\mathrm{MeV}/c]}{2.99792458\; B[\mathrm{T}]},

    the constant being ``c / 1e8`` exactly (:data:`pyradmc.MEV_C_PER_TESLA_CM`). This is
    the radius of the full gyration circle, i.e. for motion perpendicular to the
    field; for a general direction the transverse component of the motion follows a
    circle of radius :math:`r_g \sin\alpha` with :math:`\alpha` the pitch angle, but
    the angle a chord of length s turns through about :math:`\hat{\mathbf{B}}` is
    :math:`s / r_g` regardless of pitch (see :func:`lorentz_rotation_angle`). A 1 MeV
    electron has :math:`r_g` = 3.2 mm at 1.5 T and 14 mm at 0.35 T.

    Parameters
    ----------
    e_kin_mev
        Kinetic energy in MeV.
    b_mag_tesla
        Field magnitude in tesla. **Precondition: strictly positive**; the caller
        guards the field-free case (where the radius is infinite and no rotation is
        applied) so that B = 0 executes no floating-point operation here.
    """
    return momentum_mev_c(e_kin_mev) / (MEV_C_PER_TESLA_CM * b_mag_tesla)


def rotate_about_axis(
    ux: float,
    uy: float,
    uz: float,
    ax: float,
    ay: float,
    az: float,
    theta: float,
) -> tuple[float, float, float]:
    r"""Rotate the unit vector u about the unit axis a by the angle theta (Rodrigues).

    .. math::

        \mathbf{v} = \mathbf{u}\cos\theta
                   + (\mathbf{a} \times \mathbf{u})\sin\theta
                   + \mathbf{a}\,(\mathbf{a}\cdot\mathbf{u})(1 - \cos\theta),

    a right-handed rotation: positive theta about +z takes +x toward +y. It is an
    isometry fixing the axis, so :math:`|\mathbf{v}| = |\mathbf{u}|` and
    :math:`\mathbf{a}\cdot\mathbf{v} = \mathbf{a}\cdot\mathbf{u}` — the component of
    the motion along the field is conserved exactly, as the helix requires, and a
    vector parallel to the axis is returned unchanged for any angle.

    ``theta == 0.0`` returns the input floats untouched by an explicit guard rather
    than through ``cos(0) == 1``: under the Warp targets' fast-math the trigonometric
    intrinsics are not guaranteed exact, and the field-free run is pinned bit-identical
    to the pre-field code.

    Parameters
    ----------
    ux, uy, uz
        Direction to rotate; a unit vector.
    ax, ay, az
        Rotation axis. **Must be a unit vector**: the caller normalises
        :math:`\mathbf{B}` once per step, so no normalisation happens here.
    theta
        Rotation angle in radians, right-handed about the axis.
    """
    if theta == 0.0:
        return (ux, uy, uz)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    axial = (ax * ux + ay * uy + az * uz) * (1.0 - cos_t)

    # a x u
    cx = ay * uz - az * uy
    cy = az * ux - ax * uz
    cz = ax * uy - ay * ux

    vx = ux * cos_t + cx * sin_t + ax * axial
    vy = uy * cos_t + cy * sin_t + ay * axial
    vz = uz * cos_t + cz * sin_t + az * axial
    return (vx, vy, vz)


def lorentz_rotation_angle(s_cm: float, gyroradius_cm: float, is_positron: bool) -> float:
    r"""Signed angle, about :math:`\hat{\mathbf{B}}`, that a chord of length s turns through.

    On the gyration circle a path length s subtends :math:`\theta = s / r_g`
    (independent of the pitch angle: the arc length of the transverse projection is
    :math:`s \sin\alpha` on a circle of radius :math:`r_g \sin\alpha`). The sense
    follows from :math:`\mathbf{F} = q\,\mathbf{v} \times \mathbf{B}`: for an
    electron (q = -e) moving along +x in a field along +z,
    :math:`\mathbf{v} \times \mathbf{B} = -\hat{\mathbf{y}}` and the force is toward
    **+y**, which is the right-handed (positive) rotation about +z. Hence the electron
    turns by :math:`+s/r_g` about :math:`\hat{\mathbf{B}}` and a positron by
    :math:`-s/r_g`. Pinned in ``tests/physics/test_lorentz.py``.

    The charge sign is taken as the ``is_positron`` bool both transport loops already
    carry (``transport.electron.electron_steps`` and the Warp ``electron_kernel``,
    where it is the result of a kind comparison); Warp compiles a ``bool`` parameter
    directly, so no int/float encoding is needed. No positron physics beyond the sign
    of the charge is implied (AGENTS.md section 8: positrons stay Moller-approximated).

    Parameters
    ----------
    s_cm
        Chord (path) length in cm.
    gyroradius_cm
        Gyroradius in cm from :func:`gyroradius_cm`; strictly positive.
    is_positron
        ``True`` for a positron (q = +e), ``False`` for an electron (q = -e).
    """
    angle = s_cm / gyroradius_cm
    if is_positron:
        return -angle
    return angle


def helix_chord(
    ux: float,
    uy: float,
    uz: float,
    ax: float,
    ay: float,
    az: float,
    half_angle: float,
) -> tuple[float, float, float]:
    r"""Exact helix displacement per unit path, from the direction at the path's midpoint.

    Over a path :math:`s` turning through :math:`\theta = s / r_g` at constant radius, the
    motion along the unit field axis :math:`\hat{\mathbf{a}}` is unchanged and the
    transverse motion is a chord of the gyration circle, of length
    :math:`2 r \sin(\theta/2)` for the transverse path, pointing along the transverse
    direction at the midpoint. With :math:`\mathbf{u}` the direction already turned by
    :math:`\theta/2`,

    .. math::

        \frac{\Delta\mathbf{p}}{s} = k\,\mathbf{u}
            + (1 - k)\,(\hat{\mathbf{a}}\cdot\mathbf{u})\,\hat{\mathbf{a}},
        \qquad k = \frac{\sin(\theta/2)}{\theta/2}.

    The small-step scheme of the transport loops drifts the full :math:`s` along
    :math:`\mathbf{u}`, which is this with :math:`k = 1`: an error of
    :math:`s\,\theta^2/24` that the curvature cap bounds and an arbitrary angle does not.
    The result has length at most 1, which is what a safety bound on the path leans on.
    Bielajew (1988), doi:10.1007/978-1-4613-1059-4_19.

    Pure geometry: the sign of the turn does not enter (``k`` is even), so electrons and
    positrons differ only through the direction they were turned to, and no random number
    is consumed.

    Parameters
    ----------
    ux, uy, uz
        Direction at the midpoint of the path; a unit vector.
    ax, ay, az
        Field axis; a unit vector (the caller normalises once per step).
    half_angle
        Half the signed gyration angle of the path, radians.
    """
    h = abs(half_angle)
    # Series below 0.01 rad: sin(h)/h is exact there to float64 rounding (next term h^6/5040), and a
    # fast-math sine is not guaranteed to keep the ratio at 1 for tiny arguments.
    h2 = h * h
    k = 1.0 - h2 / 6.0 * (1.0 - h2 / 20.0)
    if h >= 0.01:
        k = math.sin(h) / h
    along = (ax * ux + ay * uy + az * uz) * (1.0 - k)
    return (k * ux + along * ax, k * uy + along * ay, k * uz + along * az)
