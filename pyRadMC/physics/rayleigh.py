r"""Coherent (Rayleigh) scattering: angular sampling.

Pure scalar functions (AGENTS.md section 2.5). The transport loops are
channel-complete (AGENTS.md section 2.10): they sample this channel whenever the data
source reports a nonzero coherent cross-section. Every current source keeps that
column at zero — **no Rayleigh by default** is a statement of data, and with it this
sampler is unreachable; it exists so that enabling the channel (Phase 5) is purely a
data change.

Angular model, named here where it is implemented: the Thomson differential
cross-section

.. math::

    \frac{d\sigma}{d\Omega} \propto \frac{1 + \cos^2\theta}{2},

the zero-momentum-transfer limit :math:`F(q, Z) \to Z` of the coherent form-factor
DCS (Hubbell et al., J. Phys. Chem. Ref. Data 4, 471 (1975), doi:10.1063/1.555523;
Salvat et al., PENELOPE-2018, sec. 2.1, doi:10.1787/32da5043-en). **Stated
approximation:** the atomic form factor is data, not physics. The Phase 5 tabulated
source must bring form-factor angular sampling in the same change that turns the
coherent channel on — together with the pair-channel recalibration that AGENTS.md 7.2
already flags for whoever enables Rayleigh.
"""

from __future__ import annotations

from pyRadMC.rng import RNGState, uniform

__all__ = ["sample_rayleigh_cos_theta"]


def sample_rayleigh_cos_theta(rng_state: RNGState) -> float:
    r"""Sample the coherent polar scattering cosine from the Thomson distribution.

    Rejection from a uniform envelope on [-1, 1]: a candidate :math:`\mu` is
    accepted with probability :math:`(1 + \mu^2)/2` (the pdf's maximum is at the
    endpoints), giving a mean acceptance of 2/3. Exact for the Thomson form; see
    the module docstring for what "Thomson" approximates.

    Parameters
    ----------
    rng_state
        Per-history RNG state; two uniforms per rejection round.
    """
    while True:
        mu = 1.0 - 2.0 * uniform(rng_state)
        if 2.0 * uniform(rng_state) <= 1.0 + mu * mu:
            return mu
