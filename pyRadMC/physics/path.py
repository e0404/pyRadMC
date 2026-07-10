"""Free-path sampling.

Pure scalar functions (AGENTS.md section 2.5).
"""

from __future__ import annotations

import math

from pyRadMC.rng import RNGState, uniform

__all__ = ["sample_path_length"]


def sample_path_length(mu: float, rng_state: RNGState) -> float:
    r"""Sample a free path from the exponential attenuation law.

    .. math::

        s = -\frac{\ln(1 - u)}{\mu}, \qquad u \sim U[0, 1),

    the inversion of :math:`P(S > s) = e^{-\\mu s}` (e.g. Salvat et al.,
    PENELOPE-2018, sec. 1.4.5; doi:10.1787/32da5043-en). ``log1p(-u)`` keeps full
    precision for small u, and u < 1 (the RNG interface contract) keeps the argument
    strictly positive, so the result is always finite.

    Parameters
    ----------
    mu
        Macroscopic attenuation coefficient in 1/cm; with Woodcock tracking, the
        majorant. Must be positive.
    rng_state
        Per-history RNG state; consumes one uniform.
    """
    return -math.log1p(-uniform(rng_state)) / mu
