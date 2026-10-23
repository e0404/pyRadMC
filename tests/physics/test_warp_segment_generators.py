"""The segmented-fan generators, read back from the queue: deterministic pins.

Statistical dose tests cannot see a wrong stream key, a wrong beamlet tag or a weight
taken from the wrong segment cheaply; these read the generated primaries directly.

- ``generate_segmented_fan`` writes, history for history, the energy, direction,
  position and RNG slot ``generate_spectral_beamlets`` writes for the same fan and
  the same ``(seed, key)`` on the same device — bit for bit, both correlated and
  independent, with ``group_start > 0`` so the key mapping is exercised; its column
  tag is ``t // per_batch``; its weight is the host transmission of the tagged
  segment along that ray at float32 tolerance; ``emitted`` books weight x energy.
- ``generate_segmented_plan`` draws its segment from the *first* uniform of the
  history's stream (replayed here by a probe kernel) against the share CDF, never a
  zero-share segment, then the fan from the next four uniforms in the host order
  (:func:`pyradmc.geometry.fan.fan_primary`, run on the host on the replayed
  uniforms), weighted by that segment's transmission.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SEED
from tests.physics.test_warp_segment_transmission import _source

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

# The device chord tolerance pinned in test_warp_compiled_stack (float32 geometry):
# a weight is checked through its optical depth, |dtau| <= sum_j mu_j (RTOL L_j + ATOL).
RTOL = 2.0e-3
ATOL = 2.0e-3
PER_BATCH = 1_500
N_PER = 6_000
GROUP_START = 1
GROUP = 3  # segments 1..3 of 4
BATCH = 2


def _require(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")


def _fan_scalars(fan) -> list[float]:
    return [*fan.focal_point, *fan.center, *fan.u_axis, *fan.v_axis, fan.width_u, fan.width_v]


def _read(q, n: int) -> dict[str, np.ndarray]:
    names = ("kind", "beamlet", "primary", "energy", "weight", "x", "y", "z", "ux", "uy", "uz")
    return {name: getattr(q, name).numpy()[:n].copy() for name in (*names, "rng")}


def _assert_weights(source, segments, cols) -> None:
    """Device weights equal the host transmission within the device chord tolerance."""
    origins = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)
    directions = np.stack([cols["ux"], cols["uy"], cols["uz"]], axis=1).astype(np.float64)
    energy = cols["energy"].astype(np.float64)
    tau = np.zeros(segments.size)
    bound = np.zeros(segments.size)
    for s in np.unique(segments):
        rows = segments == s
        chords = source.segment_stack(int(s)).path_lengths(origins[rows], directions[rows])
        for j in range(chords.shape[1]):
            mu = source.attenuation_tables.total(j, energy[rows])
            tau[rows] += mu * chords[:, j]
            bound[rows] += mu * (RTOL * chords[:, j] + ATOL)
    host = np.exp(-tau)
    assert host.min() < 0.05 and host.max() > 0.9  # blocked and open rays both present
    dtau = np.abs(np.log(cols["weight"].astype(np.float64)) + tau)
    worst = int(np.argmax(dtau - bound))
    assert np.all(dtau <= bound), (
        f"ray {worst}: device tau {-np.log(cols['weight'][worst]):.6g} vs host "
        f"{tau[worst]:.6g}, beyond the chord tolerance {bound[worst]:.3g}"
    )
    # And the bulk agrees far tighter than the worst-case edge rays.
    assert np.median(np.abs(cols["weight"] / host - 1.0)) < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlated", [0, 1])
def test_segment_generator_is_the_spectral_beamlet_fan_plus_its_weight(
    device: str, correlated: int
) -> None:
    _require(device)
    from pyradmc.backends.warp import kernels
    from pyradmc.backends.warp.engine import _upload_queue
    from pyradmc.backends.warp.segments import generate_segmented_fan, upload_segmented_fan

    source = _source()
    fan = source.fan
    upload = upload_segmented_fan(source, device)
    n = GROUP * PER_BATCH
    slots = wp.zeros(n, dtype=wp.uint32, device=device)
    common = [SEED, GROUP_START, N_PER, PER_BATCH, BATCH, 0, correlated]
    nb = int(upload.cdf.shape[0])
    f = _fan_scalars(fan)

    q_seg = _upload_queue(n, device)
    emitted = wp.zeros(1, dtype=wp.int64, device=device)
    wp.launch(
        generate_segmented_fan,
        dim=n,
        inputs=[
            *common,
            upload.edges,
            upload.cdf,
            nb,
            upload.stack,
            upload.segments,
            upload.mu,
            slots,
            emitted,
            q_seg,
        ],
        device=device,
    )
    got = _read(q_seg, n)

    centre = [wp.array(np.full(GROUP, c, np.float32), dtype=float, device=device) for c in f[3:6]]
    q_ref = _upload_queue(n, device)
    wp.launch(
        kernels.generate_spectral_beamlets,
        dim=n,
        inputs=[
            *common,
            upload.edges,
            upload.cdf,
            nb,
            *f[0:3],
            *f[6:14],
            *centre,
            slots,
            wp.zeros(1, dtype=wp.int64, device=device),
            q_ref,
        ],
        device=device,
    )
    want = _read(q_ref, n)

    # (i) the fan photon and its stream, bit for bit.
    for name in ("kind", "primary", "energy", "x", "y", "z", "ux", "uy", "uz", "rng"):
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)
    # (iii) the column tag is the group-local segment.
    np.testing.assert_array_equal(got["beamlet"], np.arange(n) // PER_BATCH)
    np.testing.assert_array_equal(want["beamlet"], got["beamlet"])
    # (ii) the weight is the tagged segment's transmission along that ray.
    _assert_weights(source, GROUP_START + got["beamlet"], got)
    book = float(emitted.numpy()[0]) * 1.0e-9  # ENERGY_QUANTUM_MEV
    expected = float(np.sum(got["weight"].astype(np.float64) * got["energy"]))
    assert book == pytest.approx(expected, rel=1e-6)


@wp.kernel
def _first_uniforms(seed: int, offset: int, out: wp.array2d(dtype=float)):
    """Replay the first five uniforms of history ``offset + tid``'s stream."""
    tid = wp.tid()
    state = wp.rand_init(seed, offset + tid)
    for k in range(5):
        out[tid, k] = wp.randf(state)


class _Replay:
    """A host RNG state handing out recorded uniforms (``uniform`` calls ``random``)."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = iter(values.tolist())

    def random(self) -> float:
        return next(self._values)


@pytest.mark.parametrize("device", DEVICES)
def test_plan_generator_selects_by_the_first_uniform(device: str) -> None:
    _require(device)
    from pyradmc.backends.warp.engine import _upload_queue
    from pyradmc.backends.warp.segments import generate_segmented_plan, upload_segmented_fan
    from pyradmc.geometry.fan import fan_primary

    source = _source()
    plan = source.plan([1.0, 0.0, 2.0, 3.0])
    fan = source.fan
    upload = upload_segmented_fan(source, device)
    selection = wp.array(plan.selection_cdf.astype(np.float32), dtype=float, device=device)
    n = 4_000
    offset = 700
    slots = wp.zeros(n, dtype=wp.uint32, device=device)
    q = _upload_queue(n, device)
    emitted = wp.zeros(1, dtype=wp.int64, device=device)
    nb = int(upload.cdf.shape[0])
    wp.launch(
        generate_segmented_plan,
        dim=n,
        inputs=[
            SEED,
            offset,
            selection,
            4,
            upload.edges,
            upload.cdf,
            nb,
            upload.stack,
            upload.segments,
            upload.mu,
            slots,
            emitted,
            q,
        ],
        device=device,
    )
    got = _read(q, n)
    uniforms = wp.zeros((n, 5), dtype=float, device=device)
    wp.launch(_first_uniforms, dim=n, inputs=[SEED, offset, uniforms], device=device)
    u = uniforms.numpy().astype(np.float64)

    cdf32 = plan.selection_cdf.astype(np.float32)
    segment = np.minimum(np.searchsorted(cdf32, u[:, 0].astype(np.float32), side="right"), 3)
    assert 1 not in set(segment.tolist())  # the zero-share segment is never drawn
    np.testing.assert_allclose(
        np.bincount(segment, minlength=4) / n, [1 / 6, 0.0, 2 / 6, 3 / 6], atol=0.03
    )
    np.testing.assert_array_equal(got["beamlet"], 0)
    edges = fan.spectrum.edges
    cdf = fan.spectrum.cdf
    host = np.array(
        [fan_primary(_Replay(u[i, 1:]), edges, cdf, cdf.size, *_fan_scalars(fan)) for i in range(n)]
    )
    np.testing.assert_allclose(got["energy"], host[:, 0], rtol=1e-6)
    for k, name in enumerate(("ux", "uy", "uz"), start=1):
        np.testing.assert_allclose(got[name], host[:, k], rtol=0.0, atol=1e-5)
    _assert_weights(source, segment, got)
