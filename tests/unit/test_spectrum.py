"""Histogram spectrum sampling and the Ali-Rogers analytic MV form.

The :class:`~pyradmc.geometry.spectrum.Spectrum` is the energy model of the spectral
beam sources: a photon-number histogram sampled by CDF inversion. These tests pin
(a) the input validation and the number/energy-fluence conversion as exact algebra,
(b) the sampled energies against the input histogram with the chi-squared detection
oracle (AGENTS.md section 4), including within-bin uniformity, and (c) the semantic
invariants of the Ali & Rogers (2012) analytic MV builder — endpoint behaviour,
positivity, filtration hardening, and the 511 keV annihilation line.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pyradmc.geometry.spectrum import ALI_ROGERS_BEAMS, AliRogersMV, Spectrum, ali_rogers_mv
from pyradmc.rng.host import HostRNG
from tests.conftest import SEED

EDGES = (0.5, 1.0, 2.0, 4.0, 6.0)
WEIGHTS = (1.0, 3.0, 4.0, 2.0)


def _samples(spectrum: Spectrum, n: int, stream: int) -> np.ndarray:
    state = HostRNG().init_state(SEED, stream)
    return np.array([spectrum.sample_energy(state) for _ in range(n)])


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_max_energy_is_top_edge() -> None:
    """max_energy feeds the cross-section table upper edge: it is the last bin edge."""
    assert Spectrum(EDGES, WEIGHTS).max_energy == EDGES[-1]


def test_bin_probabilities_are_normalized() -> None:
    """Weights normalize internally to per-bin emission probabilities."""
    p = Spectrum(EDGES, WEIGHTS).bin_probabilities
    assert p == pytest.approx(np.asarray(WEIGHTS) / sum(WEIGHTS), rel=1e-14)
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize(
    ("edges", "weights", "match"),
    [
        ((1.0, 2.0, 2.0), (1.0, 1.0), "increasing"),
        ((2.0, 1.0), (1.0,), "increasing"),
        ((0.0, 1.0), (1.0,), "positive"),
        ((-1.0, 1.0), (1.0,), "positive"),
        ((1.0, 2.0, 3.0), (1.0,), "one weight per bin"),
        ((1.0, 2.0), (-1.0,), "non-negative"),
        ((1.0, 2.0, 3.0), (0.0, 0.0), "at least one"),
        ((1.0,), (), "at least two edges"),
    ],
)
def test_invalid_histograms_are_rejected(
    edges: tuple[float, ...], weights: tuple[float, ...], match: str
) -> None:
    """Malformed histograms fail loudly at construction, not during transport."""
    with pytest.raises(ValueError, match=match):
        Spectrum(edges, weights)


def test_unknown_convention_is_rejected() -> None:
    """The convention flag admits exactly 'number' and 'energy_fluence'."""
    with pytest.raises(ValueError, match="convention"):
        Spectrum(EDGES, WEIGHTS, convention="fluence")


def test_energy_fluence_conversion_is_exact_algebra() -> None:
    """Energy-fluence weights convert by dividing by the bin midpoint energy.

    Two bins with midpoints 1.0 and 2.5 MeV and equal energy-fluence content carry
    photon numbers in the ratio 2.5 : 1 — exact, no sampling involved.
    """
    s = Spectrum((0.5, 1.5, 3.5), (1.0, 1.0), convention="energy_fluence")
    expected = np.array([1.0 / 1.0, 1.0 / 2.5])
    assert s.bin_probabilities == pytest.approx(expected / expected.sum(), rel=1e-14)


def test_number_convention_is_the_default() -> None:
    """The default convention treats weights as photon-number content per bin."""
    assert Spectrum(EDGES, WEIGHTS).bin_probabilities == pytest.approx(
        Spectrum(EDGES, WEIGHTS, convention="number").bin_probabilities, rel=0.0
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sampled_energies_match_histogram() -> None:
    """Chi-squared of sampled energies on a sub-bin grid against the input pdf.

    Four sub-bins per spectrum bin test the bin selection *and* the uniform
    within-bin distribution in one detection-oracle pass.
    """
    spectrum = Spectrum(EDGES, WEIGHTS)
    n = 30_000
    e = _samples(spectrum, n, stream=0)

    sub = 4
    sub_edges = np.concatenate(
        [np.linspace(EDGES[b], EDGES[b + 1], sub + 1)[:-1] for b in range(len(WEIGHTS))]
        + [np.array([EDGES[-1]])]
    )
    counts, _ = np.histogram(e, bins=sub_edges)
    expected = n * np.repeat(spectrum.bin_probabilities / sub, sub)
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    p = float(stats.chi2.sf(chi2, counts.size - 1))
    assert p > 0.01, f"chi2={chi2:.1f} on {counts.size - 1} dof, p={p:.2e}"


def test_samples_stay_inside_the_histogram_support() -> None:
    """Every sampled energy lies within [first edge, last edge)."""
    e = _samples(Spectrum(EDGES, WEIGHTS), 5_000, stream=1)
    assert np.all(e >= EDGES[0])
    assert np.all(e < EDGES[-1])


def test_single_narrow_bin_is_quasi_monoenergetic() -> None:
    """A single narrow bin confines every sample to it (the pencil-beam degeneracy)."""
    e = _samples(Spectrum((5.999, 6.001), (1.0,)), 500, stream=2)
    assert np.all((e >= 5.999) & (e < 6.001))


def test_zero_weight_bin_is_never_sampled() -> None:
    """A zero-weight bin emits nothing; its neighbours split the probability."""
    e = _samples(Spectrum((1.0, 2.0, 3.0, 4.0), (1.0, 0.0, 1.0)), 5_000, stream=3)
    assert not np.any((e >= 2.0) & (e < 3.0))


# ---------------------------------------------------------------------------
# Ali & Rogers (2012) analytic MV builder
# ---------------------------------------------------------------------------


def test_ali_rogers_presets_cover_the_nine_benchmark_beams() -> None:
    """Table 5 of the paper: nine validated linac beams, endpoint = fitted Ee."""
    assert len(ALI_ROGERS_BEAMS) == 9
    assert ali_rogers_mv("varian-6mv").max_energy == pytest.approx(5.76)
    assert ali_rogers_mv("elekta-25mv").max_energy == pytest.approx(19.02)


def test_ali_rogers_unknown_beam_is_rejected() -> None:
    """A typo in the preset name fails loudly with the available names."""
    with pytest.raises(KeyError, match="varian-6mv"):
        ali_rogers_mv("varian-6x")


def test_ali_rogers_is_positive_and_vanishes_at_the_endpoint() -> None:
    """psi(Ee) = 0 by construction (C''3 = exp(0.5)): the top bin is empty in the limit.

    The proposed form imposes psi(Ee) = 0 through the 1.65 constant; the top bin's
    photon-number probability must be far below the modal bin's.
    """
    for name in ALI_ROGERS_BEAMS:
        p = ali_rogers_mv(name).bin_probabilities
        assert np.all(p >= 0.0), name
        assert p[-1] < 0.02 * p.max(), name


def test_ali_rogers_mean_energy_is_physical_for_6mv() -> None:
    """The Varian 6 MV fit gives a mean photon energy in the accepted ~1.4-1.9 MeV band."""
    assert 1.4 < ali_rogers_mv("varian-6mv").mean_energy < 1.9


def test_ali_rogers_filtration_hardens_the_beam() -> None:
    """More aluminium filtration (larger C2) removes soft photons: mean energy rises."""
    base = AliRogersMV(e_e=5.76, c1=1.222, c2=5.147, c3=-1.186)
    harder = AliRogersMV(e_e=5.76, c1=1.222, c2=7.0, c3=-1.186)
    assert ali_rogers_mv(harder).mean_energy > ali_rogers_mv(base).mean_energy


def _line_to_continuum(params: AliRogersMV, n_bins: int = 100) -> tuple[float, float, float]:
    """Recover the 511 keV line's strength from public spectra alone.

    Building the same beam with and without C4 isolates the line: the off-line bins
    are rescaled by one common factor r = S / (S + L), with S the continuum photon
    total and L the line's photon number. Returns ``(L / S, 1 - r, p_continuum[k])``
    — the line-to-continuum ratio, the line's share of all emitted photons, and the
    normalized continuum content of the bin holding 511 keV.

    Nothing here reaches into :mod:`pyradmc.geometry.spectrum` internals: the point
    of these tests is to constrain ``ali_rogers_mv`` from the outside, so that a wrong
    mu(E) parameterization or a wrong quadrature cannot satisfy them by construction.
    """
    p0 = ali_rogers_mv(params._replace(c4=0.0), n_bins=n_bins).bin_probabilities
    p1 = ali_rogers_mv(params, n_bins=n_bins).bin_probabilities
    edges = np.linspace(0.15, params.e_e, n_bins + 1)
    k = int(np.searchsorted(edges, 0.511, side="right") - 1)
    off_ratio = np.delete(p1, k) / np.delete(p0, k)
    r = float(off_ratio[0])
    # The common-rescale claim is itself an assertion: the line must land in bin k only.
    assert off_ratio == pytest.approx(r, rel=1e-12)
    return (1.0 - r) / r, 1.0 - r, float(p0[k])


def test_ali_rogers_annihilation_line_rides_the_same_filtration_envelope() -> None:
    """Function 13 puts C4 inside the common W/Al envelope, so the line is filtered.

    Ali and Rogers (2012), table 2, replaces ``psi_thin`` in function 12 by
    ``psi_thin + C4 delta(E - E511)`` *before* multiplying by
    ``exp(-mu_W C1^2 - mu_Al C2^2)``. The observable consequence, and the only one
    that survives the spectrum's internal normalization, is a double ratio: between
    two beams differing only in filtration, the line's strength relative to the
    continuum must change by the same factor as the 511 keV continuum bin's own
    share does. The unknown continuum totals cancel, so this holds without knowing
    mu_W, mu_Al, or the quadrature — it fails only if the line skips the envelope.

    Tolerance: the line is a delta at exactly 511 keV while the continuum bin carries
    a bin-averaged transmission, an irreducible discretization mismatch measured at
    0.70 percent for this pair at the shipped n_bins=100. The pre-fix behaviour
    (line added after the envelope) misses by a factor of 11.1, so 3 percent
    separates them by roughly three orders of magnitude.
    """
    heavy = AliRogersMV(e_e=5.76, c1=1.222, c2=5.147, c3=-1.186, c4=0.00881)
    light = heavy._replace(c1=0.200, c2=0.500)

    ratio_heavy, _, continuum_heavy = _line_to_continuum(heavy)
    ratio_light, _, continuum_light = _line_to_continuum(light)

    assert ratio_heavy / ratio_light == pytest.approx(continuum_heavy / continuum_light, rel=3e-2)


def test_ali_rogers_annihilation_line_stays_a_perturbation_on_every_preset() -> None:
    """The 511 keV line is a small feature on the continuum, never a dominant one.

    Positron annihilation contributes a visible spike to an MV spectrum, not a
    significant fraction of the fluence. Leaving the line outside the filtration
    envelope inflated it to 36.1 percent of all photons for ``varian-18mv`` and
    48.8 percent for ``elekta-25mv`` — a physically impossible spectrum that the
    6 MV-only mean-energy check above was too weak to see. Filtered, the largest
    share across the nine presets is 0.82 percent, so the 5 percent bound holds
    with six-fold headroom while still excluding the unfiltered treatment outright.
    """
    for name, params in ALI_ROGERS_BEAMS.items():
        assert params.c4 > 0.0, name
        _, share, _ = _line_to_continuum(params)
        assert 0.0 < share < 0.05, f"{name}: line carries {share:.1%} of emitted photons"


def test_ali_rogers_low_edge_respects_the_mu_parameterization_floor() -> None:
    """The tungsten mu(E) parameterization is valid above 69.5 keV; below raises."""
    with pytest.raises(ValueError, match=r"69\.5"):
        ali_rogers_mv("varian-6mv", e_min=0.05)
