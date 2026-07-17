"""Device batch-reduction kernels vs the host scorer, on synthetic fixed-point maps.

The reduction that turns per-batch fixed-point energy maps into per-voxel dose mean
and sigma moved onto the device (so only the reduced maps read back, not the dense
per-group buffer). These kernels must reproduce ``BatchedDoseScorer`` /
``BatchedBeamletScorer`` — the host oracle the whole suite is calibrated against — to
float64 rounding, including the zero-mass (uncovered) masking and the single-batch
degenerate sigma. Cross host/device that is agreement to rounding, not bit equality
(AGENTS 2.3); the *within-device* bit coupling between run and run_dij is pinned in
the Dij/backend integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

from pyRadMC.backends.warp import kernels  # noqa: E402
from pyRadMC.backends.warp.kernels import ENERGY_QUANTUM_MEV  # noqa: E402
from pyRadMC.scoring.dij import BatchedBeamletScorer  # noqa: E402
from pyRadMC.scoring.dose import BatchedDoseScorer  # noqa: E402
from pyRadMC.scoring.grid import ScoringGrid  # noqa: E402

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return request.param


def _scoring_grid() -> ScoringGrid:
    """A grid with a genuine density spread and one uncovered (zero-mass) voxel."""
    from pyRadMC.data.materials import WATER
    from pyRadMC.geometry.grid import VoxelGrid

    shape = (2, 2, 3)
    rng = np.random.default_rng(4242)
    density = rng.uniform(0.3, 1.8, size=shape)
    grid = VoxelGrid(
        shape=shape,
        spacing=(1.0, 1.5, 2.0),
        density=density,
        material=np.full(shape, WATER, dtype=np.int32),
    )
    # A scoring grid whose last voxel row juts past the CT -> zero mass there.
    return ScoringGrid.rebin(grid, shape=(2, 2, 4), spacing=(1.0, 1.5, 1.5), origin=grid.origin)


def _synthetic_quanta(n_cols: int, n_batches: int, n_voxels: int) -> np.ndarray:
    rng = np.random.default_rng(20260714)
    # Span a wide magnitude range, including empty voxels, like real deposits.
    q = rng.integers(0, 5_000_000, size=(n_cols, n_batches, n_voxels), dtype=np.int64)
    q[rng.random((n_cols, n_batches, n_voxels)) < 0.3] = 0
    return q


@pytest.mark.parametrize("n_batches", [1, 5])
def test_dij_batch_accumulation_matches_host_beamlet_scorer(device: str, n_batches: int) -> None:
    """The streamed Dij fold: one accumulate launch per batch, then a finalize.

    The engine never materializes the batch axis — it reuses one ``group * n_voxels``
    quanta map per batch and folds it into running sums — so the kernels are driven
    here exactly as the engine drives them, batch by batch, and compared against the
    host scorer fed the same blocks in the same order.
    """
    sg = _scoring_grid()
    n_voxels = sg.n_voxels
    group = 3
    per_batch = 250
    quanta = _synthetic_quanta(group, n_batches, n_voxels)

    # Host oracle: feed the equivalent per-batch energy blocks to the scorer.
    scorer = BatchedBeamletScorer(sg, n_batches=n_batches, n_beamlets=group)
    for b in range(n_batches):
        scorer.deposit_block(quanta[:, b, :].astype(np.float64) * ENERGY_QUANTUM_MEV)
        scorer.end_batch(per_batch)
    block = scorer.finalize()

    mass = wp.array(sg.voxel_mass.reshape(n_voxels), dtype=wp.float64, device=device)
    s1 = wp.zeros(group * n_voxels, dtype=wp.float64, device=device)
    s2 = wp.zeros(group * n_voxels, dtype=wp.float64, device=device)
    total_q = wp.zeros(1, dtype=wp.int64, device=device)
    for b in range(n_batches):
        # One batch's (local, voxel) map, tagged by beamlet alone.
        edep = wp.array(quanta[:, b, :].reshape(-1), dtype=wp.int64, device=device)
        wp.launch(
            kernels.accumulate_dij_batch,
            dim=group * n_voxels,
            inputs=[
                edep,
                mass,
                n_voxels,
                float(per_batch),
                float(ENERGY_QUANTUM_MEV),
                s1,
                s2,
                total_q,
            ],
            device=device,
        )
    # The engine aliases the outputs onto the sums; do the same here so the aliasing
    # is covered, not just the arithmetic.
    wp.launch(
        kernels.finalize_run,
        dim=group * n_voxels,
        inputs=[s1, s2, n_batches, s1, s2],
        device=device,
    )
    wp.synchronize_device(device)

    mean = s1.numpy().reshape(group, n_voxels)
    sigma = s2.numpy().reshape(group, n_voxels)
    np.testing.assert_allclose(mean, block.dose, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(sigma, block.sigma, rtol=1e-12, atol=0.0)
    assert int(total_q.numpy()[0]) == int(quanta.sum())
    # Uncovered voxels (zero mass) stay exactly zero, never NaN.
    uncovered = sg.voxel_mass.reshape(n_voxels) == 0.0
    assert np.all(mean[:, uncovered] == 0.0)


@pytest.mark.parametrize("n_batches", [1, 4])
def test_run_accumulate_finalize_matches_host_dose_scorer(device: str, n_batches: int) -> None:
    sg = _scoring_grid()
    n_voxels = sg.n_voxels
    per_batch = 400
    quanta = _synthetic_quanta(1, n_batches, n_voxels)[0]  # (n_batches, n_voxels)

    scorer = BatchedDoseScorer(sg, n_batches=n_batches)
    for b in range(n_batches):
        scorer.deposit_grid(quanta[b].reshape(sg.shape).astype(np.float64) * ENERGY_QUANTUM_MEV)
        scorer.end_batch(per_batch)
    ref = scorer.finalize()

    mass = wp.array(sg.voxel_mass.reshape(n_voxels), dtype=wp.float64, device=device)
    s1 = wp.zeros(n_voxels, dtype=wp.float64, device=device)
    s2 = wp.zeros(n_voxels, dtype=wp.float64, device=device)
    total_q = wp.zeros(1, dtype=wp.int64, device=device)
    for b in range(n_batches):
        edep = wp.array(quanta[b], dtype=wp.int64, device=device)
        wp.launch(
            kernels.accumulate_run_batch,
            dim=n_voxels,
            inputs=[edep, mass, float(per_batch), float(ENERGY_QUANTUM_MEV), s1, s2, total_q],
            device=device,
        )
    mean_out = wp.zeros(n_voxels, dtype=wp.float64, device=device)
    sigma_out = wp.zeros(n_voxels, dtype=wp.float64, device=device)
    wp.launch(
        kernels.finalize_run,
        dim=n_voxels,
        inputs=[s1, s2, n_batches, mean_out, sigma_out],
        device=device,
    )
    wp.synchronize_device(device)

    np.testing.assert_allclose(mean_out.numpy().reshape(sg.shape), ref.dose, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        sigma_out.numpy().reshape(sg.shape), ref.dose_sigma, rtol=1e-12, atol=0.0
    )
    assert int(total_q.numpy()[0]) == int(quanta.sum())


@pytest.mark.parametrize("truncation", [0.0, 1.0e-3])
def test_device_truncation_compaction_matches_host_assembler(
    device: str, truncation: float
) -> None:
    """Device count + compact builds the exact CSC the host DijAssembler would.

    The device operates on the same dense dose maps the host would truncate, ``col_max``
    is an arithmetic-free maximum, and the keep test shares the float64 threshold
    product — so the sparse pattern and values must be byte-identical, and the compacted
    indices sorted ascending per column (AGENTS 2.8 truncation, 2.3 determinism).
    """
    from pyRadMC.scoring.dij import DijAssembler

    group = 4
    grid_shape = (2, 2, 4)
    n_voxels = int(np.prod(grid_shape))
    rng = np.random.default_rng(99)
    mean = rng.uniform(0.0, 1.0, size=(group, n_voxels))
    mean[rng.random((group, n_voxels)) < 0.4] = 0.0  # empty voxels
    mean[1, :] = 0.0  # a wholly-empty column
    sigma = rng.uniform(0.0, 0.1, size=(group, n_voxels))

    # Host reference: truncate the dense maps through the assembler.
    host = DijAssembler(
        grid_shape=grid_shape,
        n_beamlets=group,
        n_histories_per_beamlet=100,
        n_batches=2,
        truncation=truncation,
    )
    host.add_block(0, mean, sigma)
    host_dij = host.finalize(energy_emitted=1.0, energy_deposited=1.0, energy_escaped=0.0)

    # Device path: count -> exclusive prefix -> compact, then add_sparse_block.
    mean_dev = wp.array(mean.reshape(-1), dtype=wp.float64, device=device)
    sigma_dev = wp.array(sigma.reshape(-1), dtype=wp.float64, device=device)
    counts_dev = wp.zeros(group, dtype=wp.int32, device=device)
    wp.launch(
        kernels.count_kept_per_column,
        dim=group,
        inputs=[mean_dev, n_voxels, float(truncation), counts_dev],
        device=device,
    )
    wp.synchronize_device(device)
    counts = counts_dev.numpy()
    offsets = np.zeros(group, dtype=np.int32)
    np.cumsum(counts[:-1], out=offsets[1:])
    nnz = int(counts.sum())
    out_indices = wp.zeros(nnz, dtype=wp.int64, device=device)
    out_dose = wp.zeros(nnz, dtype=wp.float64, device=device)
    out_sigma = wp.zeros(nnz, dtype=wp.float64, device=device)
    wp.launch(
        kernels.compact_column,
        dim=group,
        inputs=[
            mean_dev,
            sigma_dev,
            n_voxels,
            float(truncation),
            wp.array(offsets, dtype=wp.int32, device=device),
            out_indices,
            out_dose,
            out_sigma,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    dev = DijAssembler(
        grid_shape=grid_shape,
        n_beamlets=group,
        n_histories_per_beamlet=100,
        n_batches=2,
        truncation=truncation,
    )
    dev.add_sparse_block(0, counts, out_indices.numpy(), out_dose.numpy(), out_sigma.numpy())
    dev_dij = dev.finalize(energy_emitted=1.0, energy_deposited=1.0, energy_escaped=0.0)

    np.testing.assert_array_equal(dev_dij.indptr, host_dij.indptr)
    np.testing.assert_array_equal(dev_dij.indices, host_dij.indices)
    np.testing.assert_array_equal(dev_dij.dose, host_dij.dose)
    np.testing.assert_array_equal(dev_dij.sigma, host_dij.sigma)
