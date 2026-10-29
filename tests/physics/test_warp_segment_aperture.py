"""Aperture-sampled segment generators, read back from the queue: deterministic pins.

With ``aperture_sampling`` on, ``generate_segmented_fan`` and
``generate_segmented_plan`` draw each segment primary through
:func:`pyradmc.geometry.fan.fan_primary_aperture` on the uploaded per-segment
table. Replaying each history's uniforms (the kernel's ``(seed, key)`` stream) and
running the host sampler on them, on the float32 table the device reads:

- energy and direction agree to float32 precision, the column tag is
  ``t // per_batch`` and the stream key is the Dij block mapping (correlated: the
  within-segment index; independent: ``segment * n_per + r``), with
  ``group_start > 0``;
- the importance factor is the host's (the component choice and the region test
  agree except within float32 rounding of a region edge), and the weight is that
  factor times the tagged segment's host transmission, at the device chord
  tolerance; ``emitted`` books weight x energy;
- the plan generator's first uniform selects the segment and the next five feed
  that segment's aperture sampler;
- off, the upload carries no table and the generators are pinned elsewhere bit for
  bit to the uniform route (``test_warp_segment_generators``).

Sources: the single fan and the arc (collimator-rotated, non-coplanar, off-axis
segments) of ``tests/unit/test_segmented_aperture_sampling``, dense "water"
devices so that open and blocked rays are both present.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradmc.geometry.fan import fan_primary_aperture
from pyradmc.transport.particles import PHOTON
from tests.conftest import SEED
from tests.unit.test_segmented_aperture_sampling import _arc, _source

wp = pytest.importorskip("warp", reason="warp-lang optional dependency not installed")

pytestmark = pytest.mark.warp

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]

# The device chord tolerance pinned in test_warp_compiled_stack (float32 geometry).
RTOL = 2.0e-3
ATOL = 2.0e-3
PER_BATCH = 1_000
N_PER = 5_000
BATCH = 2


def _require(device: str) -> None:
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")


def _sources():
    return {"fan": _source(density=6.0), "arc": _arc(density=6.0)}


@wp.kernel
def _replay(seed: int, keys: wp.array(dtype=int), out: wp.array2d(dtype=float)):
    """The first six uniforms of the stream of history key ``keys[tid]``."""
    tid = wp.tid()
    state = wp.rand_init(seed, keys[tid])
    for k in range(6):
        out[tid, k] = wp.randf(state)


class _Replay:
    def __init__(self, values) -> None:
        self._values = iter(np.asarray(values, dtype=np.float64).tolist())

    def random(self) -> float:
        return next(self._values)


def _uniforms(device: str, keys: np.ndarray) -> np.ndarray:
    out = wp.zeros((keys.size, 6), dtype=float, device=device)
    key_array = wp.array(keys.astype(np.int32), dtype=int, device=device)
    wp.launch(_replay, dim=keys.size, inputs=[SEED, key_array, out], device=device)
    return out.numpy().astype(np.float64)


def _read(q, n: int) -> dict[str, np.ndarray]:
    names = ("kind", "beamlet", "primary", "energy", "weight", "x", "y", "z", "ux", "uy", "uz")
    return {name: getattr(q, name).numpy()[:n].copy() for name in names}


def _host(source, segments: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The host sampler on the replayed uniforms, on the float32 tables the device reads."""
    table = source.aperture_table.astype(np.float32).astype(np.float64)
    stride = table.shape[1]
    flat = table.reshape(-1)
    spectrum = source.spectrum
    edges = spectrum.edges.astype(np.float32).astype(np.float64)
    cdf = spectrum.cdf.astype(np.float32).astype(np.float64)
    rows = []
    for i, s in enumerate(segments):
        fan = source.fans[int(s)]
        geometry = [
            *fan.focal_point,
            *fan.center,
            *fan.u_axis,
            *fan.v_axis,
            fan.width_u,
            fan.width_v,
        ]
        geometry = np.asarray(geometry, dtype=np.float32).astype(np.float64).tolist()
        rows.append(
            fan_primary_aperture(
                _Replay(u[i]),
                edges,
                cdf,
                cdf.size,
                *geometry,
                flat,
                int(s) * stride,
                source.aperture_bands,
            )
        )
    return np.array(rows, dtype=np.float64)


def _device_transmission(upload, segments: np.ndarray, got, device: str) -> np.ndarray:
    """The device's own transmission of each queued ray through its tagged segment.

    ``segment_transmission`` is the function the generator weights by, so the
    device factor ``weight / transmission`` is isolated from the float32 chord
    error (pinned against the host in ``test_warp_segment_transmission``), which
    is largest exactly at the leaf tips this sampler concentrates on.
    """
    from pyradmc.backends.warp.segments import segment_transmission_kernel

    n = segments.size
    columns = [
        wp.array(got[name].astype(np.float32), dtype=float, device=device)
        for name in ("energy", "x", "y", "z", "ux", "uy", "uz")
    ]
    out = wp.zeros((n, upload.segments.n_segments), dtype=float, device=device)
    wp.launch(
        segment_transmission_kernel,
        dim=(n, upload.segments.n_segments),
        inputs=[upload.stack, upload.segments, upload.mu, *columns, out],
        device=device,
    )
    return out.numpy()[np.arange(n), segments].astype(np.float64)


def _assert_primaries(source, upload, segments: np.ndarray, got, host, device: str) -> None:
    np.testing.assert_allclose(got["energy"], host[:, 0], rtol=2e-6)
    for k, name in enumerate(("ux", "uy", "uz"), start=1):
        np.testing.assert_allclose(got[name], host[:, k], rtol=0.0, atol=2e-5)
    np.testing.assert_array_equal(got["kind"], PHOTON)
    np.testing.assert_array_equal(got["primary"], 1)

    # The importance factor: the device weight over the device transmission of the
    # same ray, against the host sampler's factor.
    transmission = _device_transmission(upload, segments, got, device)
    assert transmission.min() < 0.05 and transmission.max() > 0.9  # blocked and open rays
    factor = got["weight"].astype(np.float64) / transmission
    factor_host = host[:, 4]
    tables = source.aperture_table.astype(np.float32).astype(np.float64)
    out, inside = tables[segments, 0], tables[segments, 1]
    assert np.all(
        np.isclose(factor_host, out, rtol=1e-12) | np.isclose(factor_host, inside, rtol=1e-12)
    )
    assert np.unique(np.round(factor_host, 5)).size >= 3  # inside and outside, several segments
    agree = np.isclose(factor, factor_host, rtol=1e-5, atol=0.0)
    # A disagreement can only be a whole-field draw on a region edge whose test
    # flipped in float32 (the device then used the other factor): a handful at most.
    flipped = ~agree
    assert flipped.sum() <= max(2, segments.size // 5_000), f"{flipped.sum()} factors differ"
    other = np.where(np.isclose(factor_host, out, rtol=1e-12), inside, out)
    np.testing.assert_allclose(factor[flipped], other[flipped], rtol=1e-5)

    # And the weight is the factor times the host transmission, in bulk (the
    # worst-case edge rays are the chord pin's business, see above).
    origins = np.stack([got["x"], got["y"], got["z"]], axis=1).astype(np.float64)
    directions = np.stack([got["ux"], got["uy"], got["uz"]], axis=1).astype(np.float64)
    energy = got["energy"].astype(np.float64)
    host_transmission = np.zeros(segments.size)
    for s in np.unique(segments):
        rows = segments == s
        host_transmission[rows] = source.transmission(
            int(s), energy[rows], origins[rows], directions[rows]
        )
    ratio = got["weight"][agree] / (factor_host[agree] * host_transmission[agree])
    assert np.median(np.abs(ratio - 1.0)) < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlated", [0, 1])
@pytest.mark.parametrize("which", ["fan", "arc"])
def test_aperture_generator_is_the_host_sampler(device: str, correlated: int, which: str) -> None:
    _require(device)
    from pyradmc.backends.warp.engine import _upload_queue
    from pyradmc.backends.warp.segments import generate_segmented_fan, upload_segmented_fan

    source = _sources()[which]
    group_start = 1
    group = source.n_segments - group_start
    upload = upload_segmented_fan(source, device)
    assert upload.segments.aperture == 1
    n = group * PER_BATCH
    slots = wp.zeros(n, dtype=wp.uint32, device=device)
    q = _upload_queue(n, device)
    emitted = wp.zeros(1, dtype=wp.int64, device=device)
    wp.launch(
        generate_segmented_fan,
        dim=n,
        inputs=[
            SEED,
            group_start,
            N_PER,
            PER_BATCH,
            BATCH,
            0,
            correlated,
            upload.edges,
            upload.cdf,
            int(upload.cdf.shape[0]),
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
    local = np.arange(n) // PER_BATCH
    np.testing.assert_array_equal(got["beamlet"], local)
    segments = group_start + local
    r = BATCH * PER_BATCH + np.arange(n) % PER_BATCH
    keys = r if correlated else segments * N_PER + r
    u = _uniforms(device, keys)
    host = _host(source, segments, u[:, :5])
    _assert_primaries(source, upload, segments, got, host, device)
    for s in np.unique(segments):
        fan = source.fans[int(s)]
        rows = segments == s
        for name, value in zip(("x", "y", "z"), fan.focal_point, strict=True):
            np.testing.assert_array_equal(got[name][rows], np.float32(value))
    book = float(emitted.numpy()[0]) * 1.0e-9  # ENERGY_QUANTUM_MEV
    expected = float(np.sum(got["weight"].astype(np.float64) * got["energy"]))
    assert book == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("which", ["fan", "arc"])
def test_aperture_plan_generator_selects_then_samples_the_aperture(device: str, which: str) -> None:
    _require(device)
    from pyradmc.backends.warp.engine import _upload_queue
    from pyradmc.backends.warp.segments import generate_segmented_plan, upload_segmented_fan

    source = _sources()[which]
    weights = [1.0, 0.0, 2.0, 3.0][: source.n_segments]
    plan = source.plan(weights)
    upload = upload_segmented_fan(source, device)
    selection = wp.array(plan.selection_cdf.astype(np.float32), dtype=float, device=device)
    n = 4_000
    offset = 300
    slots = wp.zeros(n, dtype=wp.uint32, device=device)
    q = _upload_queue(n, device)
    emitted = wp.zeros(1, dtype=wp.int64, device=device)
    wp.launch(
        generate_segmented_plan,
        dim=n,
        inputs=[
            SEED,
            offset,
            selection,
            source.n_segments,
            upload.edges,
            upload.cdf,
            int(upload.cdf.shape[0]),
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
    np.testing.assert_array_equal(got["beamlet"], 0)
    u = _uniforms(device, offset + np.arange(n))
    cdf32 = plan.selection_cdf.astype(np.float32)
    segments = np.minimum(
        np.searchsorted(cdf32, u[:, 0].astype(np.float32), side="right"), source.n_segments - 1
    )
    assert 1 not in set(segments.tolist())  # the zero-share segment is never drawn
    host = _host(source, segments, u[:, 1:6])
    _assert_primaries(source, upload, segments, got, host, device)


def test_the_upload_carries_no_table_when_off() -> None:
    from pyradmc.backends.warp.segments import upload_segmented_fan

    upload = upload_segmented_fan(_source(delta=None), "cpu")
    assert upload.segments.aperture == 0
    on = upload_segmented_fan(_source(), "cpu")
    np.testing.assert_array_equal(
        on.segments.aperture_table.numpy(),
        _source().aperture_table.reshape(-1).astype(np.float32),
    )
    assert on.segments.aperture_bands == _source().aperture_bands
