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
from pyRadMC.backends.warp.kernels import ENERGY_QUANTUM_MEV, _mean_sigma  # noqa: E402
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


# --- chunk-parallel truncation + compaction ---------------------------------------
#
# The one-thread-per-column scan above is correct but serialises the whole n_voxels
# sweep inside a single thread (measured: 48 percent of Dij wall time at a
# CT-resolution scoring grid, and *rising* as the group shortens, since a trailing
# group runs the same per-thread work on fewer threads). The chunked kernels below
# split each column's sweep across many threads. They must produce byte-identical
# output, so these tests pin them against both the host assembler and the serial
# kernels they replace.


def _truncate_serial(device: str, mean_dev, sigma_dev, group: int, n_voxels: int, trunc: float):
    """The one-thread-per-column path: count -> host prefix -> compact."""
    counts_dev = wp.zeros(group, dtype=wp.int32, device=device)
    wp.launch(
        kernels.count_kept_per_column,
        dim=group,
        inputs=[mean_dev, n_voxels, float(trunc), counts_dev],
        device=device,
    )
    wp.synchronize_device(device)
    counts = counts_dev.numpy().copy()
    offsets = np.zeros(group, dtype=np.int32)
    np.cumsum(counts[:-1], out=offsets[1:])
    nnz = int(counts.sum())
    out_i = wp.zeros(nnz, dtype=wp.int64, device=device)
    out_d = wp.zeros(nnz, dtype=wp.float64, device=device)
    out_s = wp.zeros(nnz, dtype=wp.float64, device=device)
    wp.launch(
        kernels.compact_column,
        dim=group,
        inputs=[
            mean_dev,
            sigma_dev,
            n_voxels,
            float(trunc),
            wp.array(offsets, dtype=wp.int32, device=device),
            out_i,
            out_d,
            out_s,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return counts, out_i.numpy().copy(), out_d.numpy().copy(), out_s.numpy().copy()


def _truncate_chunked(device: str, mean_dev, sigma_dev, group: int, n_voxels: int, trunc: float):
    """The chunk-parallel path: column max -> per-chunk counts -> chunked compact."""
    n_chunks, chunk = kernels.truncation_chunk_layout(n_voxels)
    col_max = wp.zeros(group, dtype=wp.float64, device=device)
    wp.launch(
        kernels.column_max,
        dim=group * n_voxels,
        inputs=[mean_dev, n_voxels, col_max],
        device=device,
    )
    chunk_counts = wp.zeros(group * n_chunks, dtype=wp.int32, device=device)
    wp.launch(
        kernels.count_kept_per_chunk,
        dim=group * n_chunks,
        inputs=[mean_dev, n_voxels, n_chunks, chunk, float(trunc), col_max, chunk_counts],
        device=device,
    )
    wp.synchronize_device(device)

    per_chunk = chunk_counts.numpy().reshape(group, n_chunks)
    counts = per_chunk.sum(axis=1).astype(np.int32)
    column_base = np.zeros(group, dtype=np.int64)
    np.cumsum(counts[:-1], out=column_base[1:])
    # Write cursor for each chunk: its column's base plus the exclusive prefix of the
    # chunks before it *within* that column, so ascending voxel order is preserved.
    within = np.zeros((group, n_chunks), dtype=np.int64)
    np.cumsum(per_chunk[:, :-1], axis=1, out=within[:, 1:])
    chunk_base = (column_base[:, None] + within).reshape(-1).astype(np.int32)

    nnz = int(counts.sum())
    out_i = wp.zeros(nnz, dtype=wp.int64, device=device)
    out_d = wp.zeros(nnz, dtype=wp.float64, device=device)
    out_s = wp.zeros(nnz, dtype=wp.float64, device=device)
    wp.launch(
        kernels.compact_column_chunked,
        dim=group * n_chunks,
        inputs=[
            mean_dev,
            sigma_dev,
            n_voxels,
            n_chunks,
            chunk,
            float(trunc),
            col_max,
            wp.array(chunk_base, dtype=wp.int32, device=device),
            out_i,
            out_d,
            out_s,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return counts, out_i.numpy().copy(), out_d.numpy().copy(), out_s.numpy().copy()


def _awkward_maps(group: int, n_voxels: int, seed: int):
    """Dose/sigma maps exercising the column shapes truncation has to survive."""
    rng = np.random.default_rng(seed)
    mean = rng.uniform(0.0, 1.0, size=(group, n_voxels))
    mean[rng.random((group, n_voxels)) < 0.4] = 0.0
    mean[0, :] = 0.0  # wholly empty column
    mean[1, :] = 0.25  # constant column: every voxel is the maximum
    mean[2, :] = 0.0
    mean[2, 0] = 1.0  # maximum at the very first voxel
    mean[3, :] = 0.0
    mean[3, -1] = 1.0  # maximum at the very last voxel
    sigma = rng.uniform(0.0, 0.1, size=(group, n_voxels))
    return mean, sigma


@pytest.mark.parametrize("truncation", [0.0, 1.0e-3, 0.5])
def test_chunked_truncation_matches_host_assembler(device: str, truncation: float) -> None:
    """Chunk-parallel truncation builds the exact CSC the host DijAssembler would.

    Same contract as the serial kernels (AGENTS 2.8 truncation, 2.3 determinism):
    ``col_max`` is an arithmetic-free maximum, the keep test shares the float64
    threshold product, and indices come out sorted ascending per column — the chunk
    write cursors are the exclusive prefix within each column, so splitting the sweep
    cannot reorder it.
    """
    from pyRadMC.scoring.dij import DijAssembler

    group = 6
    grid_shape = (7, 11, 13)  # 1001 voxels: not a multiple of the chunk count
    n_voxels = int(np.prod(grid_shape))
    mean, sigma = _awkward_maps(group, n_voxels, seed=99)

    host = DijAssembler(
        grid_shape=grid_shape,
        n_beamlets=group,
        n_histories_per_beamlet=100,
        n_batches=2,
        truncation=truncation,
    )
    host.add_block(0, mean, sigma)
    host_dij = host.finalize(energy_emitted=1.0, energy_deposited=1.0, energy_escaped=0.0)

    mean_dev = wp.array(mean.reshape(-1), dtype=wp.float64, device=device)
    sigma_dev = wp.array(sigma.reshape(-1), dtype=wp.float64, device=device)
    counts, idx, dose, sig = _truncate_chunked(
        device, mean_dev, sigma_dev, group, n_voxels, truncation
    )

    dev = DijAssembler(
        grid_shape=grid_shape,
        n_beamlets=group,
        n_histories_per_beamlet=100,
        n_batches=2,
        truncation=truncation,
    )
    dev.add_sparse_block(0, counts, idx, dose, sig)
    dev_dij = dev.finalize(energy_emitted=1.0, energy_deposited=1.0, energy_escaped=0.0)

    np.testing.assert_array_equal(dev_dij.indptr, host_dij.indptr)
    np.testing.assert_array_equal(dev_dij.indices, host_dij.indices)
    np.testing.assert_array_equal(dev_dij.dose, host_dij.dose)
    np.testing.assert_array_equal(dev_dij.sigma, host_dij.sigma)


@pytest.mark.parametrize("n_voxels", [1, 7, 1001, 4096])
def test_chunked_truncation_is_byte_identical_to_the_serial_kernels(
    device: str, n_voxels: int
) -> None:
    """The chunked kernels replace the serial ones with no change to any output byte.

    Swept across voxel counts that land on both sides of the chunk layout: fewer
    voxels than chunks (chunk size 1, idle threads), a partial trailing chunk, and an
    exact multiple.
    """
    group = 6
    mean, sigma = _awkward_maps(group, n_voxels, seed=7)
    mean_dev = wp.array(mean.reshape(-1), dtype=wp.float64, device=device)
    sigma_dev = wp.array(sigma.reshape(-1), dtype=wp.float64, device=device)

    trunc = 1.0e-3
    s_counts, s_idx, s_dose, s_sig = _truncate_serial(
        device, mean_dev, sigma_dev, group, n_voxels, trunc
    )
    c_counts, c_idx, c_dose, c_sig = _truncate_chunked(
        device, mean_dev, sigma_dev, group, n_voxels, trunc
    )

    np.testing.assert_array_equal(c_counts, s_counts)
    np.testing.assert_array_equal(c_idx, s_idx)
    np.testing.assert_array_equal(c_dose, s_dose)
    np.testing.assert_array_equal(c_sig, s_sig)


def test_chunked_truncation_keeps_indices_sorted_within_each_column(device: str) -> None:
    """Ascending row order per column — what CSC consumers and add_sparse_block assume."""
    group, n_voxels = 5, 1001
    mean, sigma = _awkward_maps(group, n_voxels, seed=11)
    mean_dev = wp.array(mean.reshape(-1), dtype=wp.float64, device=device)
    sigma_dev = wp.array(sigma.reshape(-1), dtype=wp.float64, device=device)

    counts, idx, _, _ = _truncate_chunked(device, mean_dev, sigma_dev, group, n_voxels, 1.0e-3)

    start = 0
    for column, n in enumerate(counts):
        column_idx = idx[start : start + int(n)]
        assert np.all(np.diff(column_idx) > 0), f"column {column} indices not strictly ascending"
        start += int(n)
    assert start == idx.size


# --- the fold must not depend on visiting empty voxels -----------------------------
#
# A Dij group's dense quanta map is overwhelmingly empty: measured 99.66 percent zero
# per batch on a TG-119-shaped 3 mm run (0.345 percent occupancy, 36 interior 5 mm
# beamlets over a 1e6-voxel grid). ``accumulate_dij_batch`` therefore skips empty
# entries rather than paying a float64 divide on each one. That is an exact
# equivalence, not an approximation — ``_batch_dose`` returns 0.0 for zero quanta, and
# ``s += 0.0`` / ``atomic_add(..., 0)`` are no-ops — and this reference kernel, which
# does the arithmetic unconditionally, pins it to the bit.


@wp.kernel
def _accumulate_dij_batch_unguarded(
    edep: wp.array(dtype=wp.int64),
    voxel_mass: wp.array(dtype=wp.float64),
    n_voxels: int,
    nhist: wp.float64,
    quantum: wp.float64,
    s1: wp.array(dtype=wp.float64),
    s2: wp.array(dtype=wp.float64),
    total_quanta: wp.array(dtype=wp.int64),
):
    """``accumulate_dij_batch`` with every entry folded, empty or not."""
    tid = wp.tid()
    local = tid // n_voxels
    vox = tid - local * n_voxels
    q = edep[tid]
    energy = wp.float64(q) * quantum
    d = wp.float64(0.0)
    if voxel_mass[vox] > wp.float64(0.0):
        d = energy / (voxel_mass[vox] * nhist)
    s1[tid] = s1[tid] + d
    s2[tid] = s2[tid] + d * d
    wp.atomic_add(total_quanta, 0, q)


def _sparse_quanta(group: int, n_voxels: int, n_batches: int) -> np.ndarray:
    """Production-shaped quanta: ~0.3 percent occupancy, with the awkward batches."""
    rng = np.random.default_rng(31337)
    q = np.zeros((n_batches, group, n_voxels), dtype=np.int64)
    n_hit = max(1, int(0.003 * group * n_voxels))
    for b in range(n_batches):
        flat = rng.choice(group * n_voxels, size=n_hit, replace=False)
        q[b].reshape(-1)[flat] = rng.integers(1, 5_000_000, size=n_hit, dtype=np.int64)
    q[0, :, 0] = 7  # voxel 0 carries quanta in the first batch ...
    if n_batches > 1:
        q[1] = 0  # ... a wholly empty batch ...
    if n_batches > 2:
        q[2, :, 0] = 0  # ... and a voxel empty here but not in batch 0
    return q


@pytest.mark.parametrize("n_batches", [1, 4])
def test_dij_fold_is_bit_identical_when_empty_entries_are_skipped(
    device: str, n_batches: int
) -> None:
    """Folding only the nonzero entries reproduces folding all of them, to the bit.

    Covers the cases skipping could plausibly break: a wholly empty batch, a voxel
    empty in one batch and not another, and uncovered (zero-mass) voxels that carry
    quanta anyway — where the dose is defined as exactly zero, so the guard and the
    masked divide must agree there too.
    """
    group, n_voxels = 3, 4096
    quanta = _sparse_quanta(group, n_voxels, n_batches)
    rng = np.random.default_rng(5)
    mass = rng.uniform(0.01, 0.05, size=n_voxels)
    mass[:8] = 0.0  # uncovered voxels
    quanta[0, :, 3] = 11  # ... carrying quanta regardless
    per_batch, quantum = 250.0, float(ENERGY_QUANTUM_MEV)

    def fold(kernel) -> tuple[np.ndarray, np.ndarray, int]:
        mass_dev = wp.array(mass, dtype=wp.float64, device=device)
        s1 = wp.zeros(group * n_voxels, dtype=wp.float64, device=device)
        s2 = wp.zeros(group * n_voxels, dtype=wp.float64, device=device)
        total_q = wp.zeros(1, dtype=wp.int64, device=device)
        for b in range(n_batches):
            edep = wp.array(quanta[b].reshape(-1), dtype=wp.int64, device=device)
            wp.launch(
                kernel,
                dim=group * n_voxels,
                inputs=[edep, mass_dev, n_voxels, per_batch, quantum, s1, s2, total_q],
                device=device,
            )
        wp.synchronize_device(device)
        return s1.numpy().copy(), s2.numpy().copy(), int(total_q.numpy()[0])

    got_s1, got_s2, got_q = fold(kernels.accumulate_dij_batch)
    ref_s1, ref_s2, ref_q = fold(_accumulate_dij_batch_unguarded)

    np.testing.assert_array_equal(got_s1, ref_s1)
    np.testing.assert_array_equal(got_s2, ref_s2)
    assert got_q == ref_q == int(quanta.sum())
    assert np.all(np.isfinite(got_s1)) and np.all(np.isfinite(got_s2))
    # Uncovered voxels stay exactly zero even where they carried quanta.
    uncovered = np.tile(mass == 0.0, group)
    assert np.all(got_s1[uncovered] == 0.0)


# --- finalize must not depend on visiting empty voxels either ----------------------
#
# ``s1`` is still ~97 percent zero after all batches are folded, and ``_mean_sigma``
# costs a float64 divide and a sqrt on a part where FP64 runs at 1/64 rate. An empty
# voxel's mean and standard error are both exactly zero, so ``finalize_run`` writes
# those zeros directly instead of computing them. It still *writes* them: the Dij
# aliases the outputs onto the sums (already zero there) but ``run`` passes separate
# arrays, so skipping the store would leave whatever those held.


@wp.kernel
def _finalize_run_unguarded(
    s1: wp.array(dtype=wp.float64),
    s2: wp.array(dtype=wp.float64),
    n_batches: int,
    mean_out: wp.array(dtype=wp.float64),
    sigma_out: wp.array(dtype=wp.float64),
):
    """``finalize_run`` with every voxel reduced, empty or not."""
    vox = wp.tid()
    mean, sigma = _mean_sigma(s1[vox], s2[vox], n_batches)
    mean_out[vox] = mean
    sigma_out[vox] = sigma


@pytest.mark.parametrize("n_batches", [1, 4])
@pytest.mark.parametrize("alias", [True, False])
def test_finalize_is_bit_identical_when_empty_voxels_are_skipped(
    device: str, n_batches: int, alias: bool
) -> None:
    """Reducing only the nonzero voxels reproduces reducing all of them, to the bit.

    Run under both call shapes the engine uses: ``run_dij`` aliases the outputs onto
    the sums, ``run`` passes separate arrays. The non-aliased arrays are seeded with
    junk here, which pins the contract that an empty voxel is *written* zero rather
    than left untouched.
    """
    n = 4096
    rng = np.random.default_rng(2718)
    s1_h = np.zeros(n)
    hit = rng.choice(n, size=max(1, n // 40), replace=False)  # ~2.5 percent occupancy
    s1_h[hit] = rng.uniform(1e-9, 1.0, size=hit.size)
    s2_h = np.zeros(n)
    s2_h[hit] = s1_h[hit] ** 2 * rng.uniform(0.2, 1.5, size=hit.size)

    def finalize(kernel) -> tuple[np.ndarray, np.ndarray]:
        s1 = wp.array(s1_h, dtype=wp.float64, device=device)
        s2 = wp.array(s2_h, dtype=wp.float64, device=device)
        if alias:
            mean_out, sigma_out = s1, s2
        else:
            junk = np.full(n, -12345.75)
            mean_out = wp.array(junk, dtype=wp.float64, device=device)
            sigma_out = wp.array(junk, dtype=wp.float64, device=device)
        wp.launch(kernel, dim=n, inputs=[s1, s2, n_batches, mean_out, sigma_out], device=device)
        wp.synchronize_device(device)
        return mean_out.numpy().copy(), sigma_out.numpy().copy()

    got_mean, got_sigma = finalize(kernels.finalize_run)
    ref_mean, ref_sigma = finalize(_finalize_run_unguarded)

    np.testing.assert_array_equal(got_mean, ref_mean)
    np.testing.assert_array_equal(got_sigma, ref_sigma)
    empty = s1_h == 0.0
    assert np.all(got_mean[empty] == 0.0), "empty voxels must be written zero, not left stale"
    assert np.all(got_sigma[empty] == 0.0)
    assert np.all(np.isfinite(got_mean)) and np.all(np.isfinite(got_sigma))


# --- the fold consumes its map, so the engine clears it once per group -------------
#
# ``edep`` was cleared before every batch — a full ``group * n_voxels`` int64 write
# ten times per group, to erase a map that is 99.66 percent zero already. Since the
# fold visits every entry anyway, it resets the ones it consumes, and the engine then
# only has to clear once at group start. The clear is kept (rather than dropped
# entirely, relying on ``wp.zeros`` plus the reset) because a *shorter* group only
# resets its own prefix: today a short group is always last, but that ordering is not
# an invariant the buffer should depend on.


@wp.kernel
def _add_quanta(edep: wp.array(dtype=wp.int64), src: wp.array(dtype=wp.int64)):
    """Deposit like transport does: atomic accumulation into the live map."""
    tid = wp.tid()
    if src[tid] != wp.int64(0):
        wp.atomic_add(edep, tid, src[tid])


def test_dij_fold_leaves_its_quanta_map_zeroed(device: str) -> None:
    """The fold's new postcondition: every entry it consumed is reset to zero.

    This is what lets the engine clear ``edep`` once per group instead of once per
    batch; without it the second batch would fold the first batch's quanta again.
    """
    group, n_voxels = 3, 4096
    quanta = _sparse_quanta(group, n_voxels, 1)[0].reshape(-1)
    mass = wp.array(np.full(n_voxels, 0.03), dtype=wp.float64, device=device)
    edep = wp.array(quanta, dtype=wp.int64, device=device)
    s1 = wp.zeros(group * n_voxels, dtype=wp.float64, device=device)
    s2 = wp.zeros(group * n_voxels, dtype=wp.float64, device=device)
    total_q = wp.zeros(1, dtype=wp.int64, device=device)

    assert int(quanta.sum()) > 0, "the fixture must actually deposit something"
    wp.launch(
        kernels.accumulate_dij_batch,
        dim=group * n_voxels,
        inputs=[edep, mass, n_voxels, 250.0, float(ENERGY_QUANTUM_MEV), s1, s2, total_q],
        device=device,
    )
    wp.synchronize_device(device)

    assert not np.any(edep.numpy()), "fold must leave the quanta map zeroed"


@pytest.mark.parametrize("n_batches", [2, 4])
def test_clearing_the_quanta_map_once_per_group_matches_clearing_every_batch(
    device: str, n_batches: int
) -> None:
    """The engine's new pattern reproduces the old one to the bit.

    Old: clear, deposit, fold — every batch. New: clear once, then deposit and fold
    per batch, relying on the fold to have reset the map.
    """
    group, n_voxels = 3, 4096
    quanta = _sparse_quanta(group, n_voxels, n_batches)
    mass_h = np.full(n_voxels, 0.03)
    mass_h[:8] = 0.0  # uncovered voxels must not disturb the reset either
    per_batch, quantum = 250.0, float(ENERGY_QUANTUM_MEV)

    def run(clear_every_batch: bool) -> tuple[np.ndarray, np.ndarray, int]:
        mass = wp.array(mass_h, dtype=wp.float64, device=device)
        edep = wp.zeros(group * n_voxels, dtype=wp.int64, device=device)
        s1 = wp.zeros(group * n_voxels, dtype=wp.float64, device=device)
        s2 = wp.zeros(group * n_voxels, dtype=wp.float64, device=device)
        total_q = wp.zeros(1, dtype=wp.int64, device=device)
        for b in range(n_batches):
            if clear_every_batch:
                wp.launch(
                    kernels.fill_int64,
                    dim=group * n_voxels,
                    inputs=[edep, 0],
                    device=device,
                )
            src = wp.array(quanta[b].reshape(-1), dtype=wp.int64, device=device)
            wp.launch(_add_quanta, dim=group * n_voxels, inputs=[edep, src], device=device)
            wp.launch(
                kernels.accumulate_dij_batch,
                dim=group * n_voxels,
                inputs=[edep, mass, n_voxels, per_batch, quantum, s1, s2, total_q],
                device=device,
            )
        wp.synchronize_device(device)
        return s1.numpy().copy(), s2.numpy().copy(), int(total_q.numpy()[0])

    old_s1, old_s2, old_q = run(clear_every_batch=True)
    new_s1, new_s2, new_q = run(clear_every_batch=False)

    np.testing.assert_array_equal(new_s1, old_s1)
    np.testing.assert_array_equal(new_s2, old_s2)
    assert new_q == old_q == int(quanta.sum())


def test_truncation_chunk_layout_covers_every_voxel_exactly_once() -> None:
    """The layout must tile [0, n_voxels) with no gap and no overlap."""
    for n_voxels in (1, 2, 7, 255, 256, 257, 1001, 4096, 1_000_000):
        n_chunks, chunk = kernels.truncation_chunk_layout(n_voxels)
        assert n_chunks >= 1 and chunk >= 1
        assert n_chunks * chunk >= n_voxels, "layout leaves a tail uncovered"
        assert (n_chunks - 1) * chunk < n_voxels, "layout has a wholly empty trailing chunk"
