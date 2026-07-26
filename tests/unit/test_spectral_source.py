"""Divergent polyenergetic sources: geometry and spectrum contracts.

A :class:`~pyRadMC.geometry.source.SpectralBeamletSource` fans from a focal point
through per-bixel apertures at a reference plane, all **in the engine frame** —
gantry/couch rotation is the caller's business (the pyRadPlan adapter maps world
coordinates exactly like the CT adapter does). These tests pin the analytic
geometry (rays originate at the focal spot and pass through their bixel aperture;
uniform aperture sampling), the per-beamlet energy spectrum (chi-squared), the
correlated-sampling stream contract (same stream => same within-aperture offset in
every beamlet), the pencil-beam degeneracy, and the default pre-sampling batch.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pyRadMC.geometry.source import Primary, SpectralBeamletSource, SpectralBeamSource
from pyRadMC.geometry.spectrum import Spectrum
from pyRadMC.rng.host import HostRNG
from tests.conftest import SEED

SPECTRUM = Spectrum((0.5, 1.0, 2.0, 4.0, 6.0), (1.0, 3.0, 4.0, 2.0))
FOCAL = (8.0, 8.0, -100.0)
CENTERS = ((6.0, 8.0, 0.0), (8.0, 8.0, 0.0), (10.0, 8.0, 0.0))
WIDTH = 2.0


def _source(**overrides: object) -> SpectralBeamletSource:
    kwargs: dict[str, object] = dict(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        centers=CENTERS,
        width_u=WIDTH,
        width_v=WIDTH,
    )
    kwargs.update(overrides)
    return SpectralBeamletSource(**kwargs)  # type: ignore[arg-type]


def _emit_many(source: SpectralBeamletSource, beamlet: int, n: int, stream: int) -> list[Primary]:
    rng = HostRNG()
    return [source.emit(beamlet, rng.init_state(SEED, stream * n + i)) for i in range(n)]


# ---------------------------------------------------------------------------
# Interface and validation
# ---------------------------------------------------------------------------


def test_beamlet_count_and_max_energy() -> None:
    """n_beamlets is the aperture count; max_energy is the spectrum's top edge."""
    source = _source()
    assert source.n_beamlets == 3
    assert source.max_energy == SPECTRUM.max_energy


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"centers": ()}, "at least one"),
        ({"width_u": -1.0}, "non-negative"),
        ({"centers": ((8.0, 8.0, -100.0),)}, "focal point"),
        ({"u_axis": (0.0, 0.0, 0.0)}, "zero"),
        ({"v_axis": (2.0, 0.0, 0.0)}, "parallel"),
    ],
)
def test_invalid_geometry_is_rejected(overrides: dict[str, object], match: str) -> None:
    """Degenerate aperture geometry fails loudly at construction."""
    with pytest.raises(ValueError, match=match):
        _source(**overrides)


# ---------------------------------------------------------------------------
# Divergent geometry
# ---------------------------------------------------------------------------


def test_rays_originate_at_the_focal_spot() -> None:
    """Every primary starts exactly at the focal point with a unit direction."""
    for p in _emit_many(_source(), beamlet=1, n=200, stream=0):
        assert (p.x, p.y, p.z) == FOCAL
        assert p.ux**2 + p.uy**2 + p.uz**2 == pytest.approx(1.0, abs=1e-12)
        assert p.kind is None and p.weight == 1.0


@pytest.mark.parametrize("beamlet", [0, 1, 2])
def test_rays_pass_through_their_bixel_aperture(beamlet: int) -> None:
    """Extending each ray to the aperture plane lands inside its own bixel.

    With the default axes the plane is z = 0; the in-plane offsets from the bixel
    centre must stay within +/- width/2, and cover it roughly uniformly (both
    halves populated on each axis).
    """
    center = np.array(CENTERS[beamlet])
    focal = np.array(FOCAL)
    offsets = []
    for p in _emit_many(_source(), beamlet, n=400, stream=1):
        d = np.array([p.ux, p.uy, p.uz])
        t = (center[2] - focal[2]) / d[2]
        q = focal + t * d
        offsets.append(q - center)
    du, dv = np.array(offsets)[:, 0], np.array(offsets)[:, 1]
    assert np.all(np.abs(du) <= WIDTH / 2 + 1e-9)
    assert np.all(np.abs(dv) <= WIDTH / 2 + 1e-9)
    # Uniform aperture sampling: both halves of each axis are populated.
    assert np.count_nonzero(du > 0) > 100 and np.count_nonzero(du < 0) > 100
    assert np.count_nonzero(dv > 0) > 100 and np.count_nonzero(dv < 0) > 100


def test_correlated_streams_replay_the_same_aperture_offset() -> None:
    """The same stream gives every beamlet the same energy and in-bixel offset.

    This is what makes correlated sampling effective for this
    source: corresponding histories differ only by the bixel centre.
    """
    rng = HostRNG()
    source = _source()
    reference = None
    for beamlet in range(source.n_beamlets):
        p = source.emit(beamlet, rng.init_state(SEED, 7))
        center = np.array(CENTERS[beamlet])
        focal = np.array(FOCAL)
        d = np.array([p.ux, p.uy, p.uz])
        q = focal + (center[2] - focal[2]) / d[2] * d
        offset = (p.energy, *(q - center))
        if reference is None:
            reference = offset
        else:
            assert offset == pytest.approx(reference, abs=1e-9)


def test_degenerate_source_reduces_to_a_pencil_beam() -> None:
    """One bixel, zero widths, one narrow bin: the divergent source is a pencil beam."""
    source = SpectralBeamletSource(
        spectrum=Spectrum((5.999, 6.001), (1.0,)),
        focal_point=FOCAL,
        centers=((8.0, 8.0, 0.0),),
        width_u=0.0,
        width_v=0.0,
    )
    expected = np.array([8.0, 8.0, 0.0]) - np.array(FOCAL)
    expected /= np.linalg.norm(expected)
    for p in _emit_many(source, beamlet=0, n=50, stream=2):
        assert (p.x, p.y, p.z) == FOCAL
        np.testing.assert_allclose([p.ux, p.uy, p.uz], expected, atol=1e-12)
        assert 5.999 <= p.energy < 6.001


# ---------------------------------------------------------------------------
# Spectrum through emit
# ---------------------------------------------------------------------------


def test_per_beamlet_sampled_spectrum_matches_input() -> None:
    """Chi-squared of one beamlet's emitted energies against the input histogram."""
    n = 20_000
    energies = np.array([p.energy for p in _emit_many(_source(), beamlet=2, n=n, stream=3)])
    counts, _ = np.histogram(energies, bins=np.array((0.5, 1.0, 2.0, 4.0, 6.0)))
    expected = n * SPECTRUM.bin_probabilities
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    p_value = float(stats.chi2.sf(chi2, counts.size - 1))
    assert p_value > 0.01, f"chi2={chi2:.1f}, p={p_value:.2e}"


# ---------------------------------------------------------------------------
# Vectorized pre-sampling batch (the simple Warp route)
#
# Like the phase-space source (docs/decisions.md), the spectral sources override the
# default per-history pre-sampling with a vectorized batch on its own PCG64 stream:
# the backends draw *independent* primaries and agree statistically, never bit-wise.
# What is pinned here is the stream contract, not equality with emit.
# ---------------------------------------------------------------------------


def test_sample_beamlet_batch_is_chunk_invariant() -> None:
    """Draws are keyed on (seed, history index): chunked calls concatenate exactly."""
    source = _source()
    whole = source.sample_beamlet_batch(SEED, 0, 64, beamlet=1)
    parts = [
        source.sample_beamlet_batch(SEED, 0, 24, beamlet=1),
        source.sample_beamlet_batch(SEED, 24, 40, beamlet=1),
    ]
    for name in whole:
        np.testing.assert_array_equal(
            whole[name], np.concatenate([p[name] for p in parts]), err_msg=name
        )


def test_sample_beamlet_batch_replays_draws_across_beamlets() -> None:
    """The stream depends on the history key only, never the beamlet.

    Correlated Dij sampling hands every beamlet the same history keys; the batch
    then replays the same energies and in-aperture offsets in every beamlet, so the
    correlation the study relies on survives the vectorized route.
    """
    source = _source()
    a = source.sample_beamlet_batch(SEED, 0, 32, beamlet=0)
    b = source.sample_beamlet_batch(SEED, 0, 32, beamlet=2)
    np.testing.assert_array_equal(a["energy"], b["energy"])
    focal = np.array(FOCAL)
    offsets = []
    for batch, center in ((a, np.array(CENTERS[0])), (b, np.array(CENTERS[2]))):
        d = np.stack([batch["ux"], batch["uy"], batch["uz"]], axis=1).astype(np.float64)
        t = (center[2] - focal[2]) / d[:, 2]
        offsets.append(focal + t[:, None] * d - center)
    np.testing.assert_allclose(offsets[0], offsets[1], atol=1e-5)  # float32 columns


def test_sample_beamlet_batch_geometry_and_spectrum() -> None:
    """Batched primaries obey the same contract as emit: focal origin, unit
    directions through the aperture, energies distributed per the histogram."""
    source = _source()
    n = 20_000
    batch = source.sample_beamlet_batch(SEED, 0, n, beamlet=1)
    assert np.all(batch["particle_type"] == 1)  # photons
    assert np.all(batch["weight"] == 1.0)
    assert batch["x"] == pytest.approx(FOCAL[0]) and batch["z"] == pytest.approx(FOCAL[2])

    d = np.stack([batch["ux"], batch["uy"], batch["uz"]], axis=1).astype(np.float64)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-6)
    focal, center = np.array(FOCAL), np.array(CENTERS[1])
    t = (center[2] - focal[2]) / d[:, 2]
    q = focal + t[:, None] * d - center
    assert np.all(np.abs(q[:, :2]) <= WIDTH / 2 + 1e-4)

    counts, _ = np.histogram(batch["energy"], bins=np.array((0.5, 1.0, 2.0, 4.0, 6.0)))
    expected = n * SPECTRUM.bin_probabilities
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    p_value = float(stats.chi2.sf(chi2, counts.size - 1))
    assert p_value > 0.01, f"chi2={chi2:.1f}, p={p_value:.2e}"


def test_open_field_sample_batch_is_chunk_invariant() -> None:
    """The open-field variant shares the vectorized, chunk-invariant batch."""
    source = SpectralBeamSource(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        center=(8.0, 8.0, 0.0),
        width_u=6.0,
        width_v=6.0,
    )
    whole = source.sample_batch(SEED, 0, 48)
    parts = [source.sample_batch(SEED, 0, 16), source.sample_batch(SEED, 16, 32)]
    for name in whole:
        np.testing.assert_array_equal(
            whole[name], np.concatenate([p[name] for p in parts]), err_msg=name
        )


# ---------------------------------------------------------------------------
# Open-field variant
# ---------------------------------------------------------------------------


def test_open_field_source_shares_the_geometry_contract() -> None:
    """SpectralBeamSource: one aperture, same focal-spot fan, same spectrum."""
    source = SpectralBeamSource(
        spectrum=SPECTRUM,
        focal_point=FOCAL,
        center=(8.0, 8.0, 0.0),
        width_u=6.0,
        width_v=6.0,
    )
    assert source.max_energy == SPECTRUM.max_energy
    rng = HostRNG()
    center, focal = np.array((8.0, 8.0, 0.0)), np.array(FOCAL)
    for i in range(300):
        p = source.emit(rng.init_state(SEED, i))
        assert (p.x, p.y, p.z) == FOCAL
        d = np.array([p.ux, p.uy, p.uz])
        q = focal + (center[2] - focal[2]) / d[2] * d
        assert np.all(np.abs(q - center)[:2] <= 3.0 + 1e-9)
