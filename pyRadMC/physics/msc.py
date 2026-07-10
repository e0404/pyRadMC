r"""Multiple-scattering hinge deflection.

Pure scalar functions (AGENTS.md section 2.5). Phase 1 uses the Gaussian small-angle
(Fermi-Eyges) model, named here where it is implemented: over a condensed-history
substep the accumulated deflection has

.. math::

    \langle \theta^2 \rangle = (T/\rho) \, \rho \, s,

with the mass scattering power from
:meth:`pyRadMC.data.interface.CrossSectionSource.scattering_power`. The projected
angles are independent Gaussians, so :math:`\\theta^2` is exponentially distributed
(Rayleigh in :math:`\\theta`) and inverts with a single uniform. Eyges, Phys. Rev. 74,
1534 (1948), doi:10.1103/PhysRev.74.1534.

Stated approximations: no large-angle (Rutherford tail) events, no Moliere/Goudsmit-
Saunderson shape correction, deflection applied at the random hinge point rather than
continuously. Adequate for depth-dose observables; penumbra work should revisit all
three.
"""

from __future__ import annotations

import math

from pyRadMC.rng import RNGState, uniform

__all__ = ["sample_hinge_cos_theta"]


def sample_hinge_cos_theta(mean_square_angle: float, rng_state: RNGState) -> float:
    """Sample the polar deflection cosine for one hinge.

    theta^2 ~ Exp(``mean_square_angle``), inverted with one uniform; theta is clamped
    to pi for the (pathological) tail where the sampled angle exceeds a half turn,
    which only occurs if the caller passed a step far too long for the Gaussian model
    to hold in the first place.

    Parameters
    ----------
    mean_square_angle
        Fermi-Eyges accumulated mean square deflection over the substep, in rad^2:
        scattering power times mass path length.
    rng_state
        Per-history RNG state; consumes one uniform.
    """
    theta = math.sqrt(-mean_square_angle * math.log1p(-uniform(rng_state)))
    return math.cos(min(theta, math.pi))
