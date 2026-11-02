"""A Dij that stays on the device it was computed on.

:meth:`pyradmc.backends.warp.engine.WarpEngine.run_dij_device` returns a
:class:`DeviceDijResult`: the same CSC matrix as :class:`~pyradmc.scoring.dij.DijResult`,
with the four arrays left as Warp arrays on the computing device. A consumer that
keeps its own working copy of the dose matrix on that device (an optimizer that
recomputes the Dij every iteration) then pays no device-to-host copy and no host
assembly; one that wants the host result calls :meth:`DeviceDijResult.to_host`,
which is bit-identical to ``run_dij`` with the same arguments on the same device
(test-pinned).

Warp-only by construction: the reference engine computes on the host and has no
device to leave anything on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyradmc.scoring.dij import DijResult

if TYPE_CHECKING:
    import warp as wp

    from pyradmc.backends.results import RunProvenance

__all__ = ["DeviceDijResult"]


@dataclass(frozen=True, eq=False)
class DeviceDijResult:
    """Sparse Dij in CSC layout by column, resident on one Warp device.

    The fields and their meaning are those of :class:`~pyradmc.scoring.dij.DijResult`
    (doses per emitted history in MeV/g, flat C-order voxel indices over
    ``grid_shape``, each column's rows ascending, ``sigma`` the per-entry standard
    error, the same correlated-sampling caveat on combining sigmas across columns),
    except that ``indptr`` (``wp.int64``, ``n_beamlets + 1``), ``indices``
    (``wp.int64``), ``dose`` and ``sigma`` (``wp.float64``) are Warp arrays on
    :attr:`device`. The engine synchronizes the device before returning, so the
    arrays are complete on any stream.

    The arrays are the engine's output buffers, handed over: the engine keeps no
    reference to them, and the caller owns them. Identity, not value, is equality
    here (``eq=False``): comparing device arrays elementwise would hide a readback
    behind ``==``; compare :meth:`to_host` results instead.
    """

    device: wp.context.Device
    grid_shape: tuple[int, int, int]
    n_beamlets: int
    n_histories_per_beamlet: int
    n_batches: int
    truncation: float
    indptr: wp.array
    indices: wp.array
    dose: wp.array
    sigma: wp.array
    energy_emitted: float
    energy_deposited: float
    energy_escaped: float
    energy_unscored: float = 0.0
    correlated: bool = False
    scoring_mode: str = "dose_to_medium"
    provenance: RunProvenance | None = None

    @property
    def n_voxels(self) -> int:
        """Number of voxels (rows of the matrix)."""
        return int(np.prod(self.grid_shape))

    @property
    def nnz(self) -> int:
        """Number of stored entries, from the array length (no device readback)."""
        return int(self.indices.shape[0])

    def to_host(self) -> DijResult:
        """Copy the matrix to the host as a :class:`~pyradmc.scoring.dij.DijResult`.

        Equal, field for field and bit for bit, to what ``run_dij`` returns for the
        same arguments on the same device. The host arrays are copies: on the
        ``cpu`` device ``numpy()`` is a view of the Warp buffer, and the host result
        must not change if the caller later reuses that buffer.
        """
        return DijResult(
            grid_shape=self.grid_shape,
            n_beamlets=self.n_beamlets,
            n_histories_per_beamlet=self.n_histories_per_beamlet,
            n_batches=self.n_batches,
            truncation=self.truncation,
            indptr=self.indptr.numpy().copy(),
            indices=self.indices.numpy().copy(),
            dose=self.dose.numpy().copy(),
            sigma=self.sigma.numpy().copy(),
            energy_emitted=self.energy_emitted,
            energy_deposited=self.energy_deposited,
            energy_escaped=self.energy_escaped,
            energy_unscored=self.energy_unscored,
            correlated=self.correlated,
            scoring_mode=self.scoring_mode,
            provenance=self.provenance,
        )
