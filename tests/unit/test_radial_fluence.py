"""The radial primary-fluence table: interpolation, extrapolation, validation.

:class:`~pyRadMC.geometry.fluence.RadialFluence` is source *data* in the same
sense as :class:`~pyRadMC.geometry.spectrum.Spectrum` — what the machine emits,
measured once at commissioning — so it is validated at construction and read by
pure interpolation afterwards.

The two extrapolation rules are the ones that bite silently and are therefore
pinned here: **flat inward** of the first tabulated radius (a table that starts
at r > 0 says nothing about the axis, and the primary fluence is smooth there),
and **zero outward** of the last (the table is expected to run past the primary
collimator's field edge, as a measured one does).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.geometry.fluence import RadialFluence

# A miniature stand-in for a measured curve: flat horn, then a roll-off to zero.
RADII = (0.0, 1.0, 2.0, 3.0, 4.0)
VALUES = (1.0, 1.05, 1.08, 0.50, 0.0)


def _fluence(**overrides: object) -> RadialFluence:
    kwargs: dict[str, object] = dict(radii=RADII, values=VALUES)
    kwargs.update(overrides)
    return RadialFluence(**kwargs)  # type: ignore[arg-type]


class TestInterpolation:
    def test_tabulated_radii_return_tabulated_values(self) -> None:
        f = _fluence()
        for r, value in zip(RADII, VALUES, strict=True):
            assert f.at_radius(r) == pytest.approx(value)

    def test_between_points_is_linear(self) -> None:
        f = _fluence()
        assert f.at_radius(1.5) == pytest.approx(0.5 * (1.05 + 1.08))
        assert f.at_radius(2.25) == pytest.approx(1.08 + 0.25 * (0.50 - 1.08))

    def test_inside_the_first_radius_is_flat(self) -> None:
        """A table starting at r > 0 says nothing about the axis; hold its first value."""
        f = _fluence(radii=(1.0, 2.0, 3.0), values=(0.9, 0.8, 0.0))
        assert f.at_radius(0.0) == pytest.approx(0.9)
        assert f.at_radius(0.5) == pytest.approx(0.9)

    def test_outside_the_last_radius_is_zero(self) -> None:
        """Beyond the tabulated field there is no primary fluence — not the last value."""
        f = _fluence()
        assert f.at_radius(4.0) == 0.0
        assert f.at_radius(4.001) == 0.0
        assert f.at_radius(1.0e6) == 0.0

    def test_vectorized_matches_the_scalar_path(self) -> None:
        f = _fluence()
        r = np.linspace(-0.0, 5.0, 57)
        np.testing.assert_allclose(
            f.at_radii(r), [f.at_radius(float(value)) for value in r], atol=0.0, rtol=1e-15
        )

    def test_negative_radius_is_read_as_its_magnitude(self) -> None:
        """A signed off-axis coordinate is a common caller slip; |r| is the only
        reading a radially symmetric table can have."""
        f = _fluence()
        assert f.at_radius(-1.5) == pytest.approx(f.at_radius(1.5))
        np.testing.assert_allclose(f.at_radii(np.array([-2.0, 2.0])), f.at_radius(2.0))


class TestProperties:
    def test_max_radius_is_the_last_edge(self) -> None:
        assert _fluence().max_radius == 4.0

    def test_reference_distance_defaults_to_isocentre(self) -> None:
        """Commissioning curves are quoted at isocentre; 100 cm is the common case."""
        assert _fluence().reference_distance == 100.0
        assert _fluence(reference_distance=22.5).reference_distance == 22.5

    def test_tables_are_read_only(self) -> None:
        """The source holds the table for the life of a run; a mutated table would
        silently change the model mid-run (the Spectrum precedent)."""
        f = _fluence()
        with pytest.raises(ValueError):
            f.radii[0] = 9.0
        with pytest.raises(ValueError):
            f.values[0] = 9.0


class TestValidation:
    def test_non_increasing_radii_raise(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            _fluence(radii=(0.0, 2.0, 1.0, 3.0, 4.0))

    def test_negative_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _fluence(radii=(-1.0, 1.0, 2.0, 3.0, 4.0))

    def test_negative_value_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _fluence(values=(1.0, 1.05, -0.1, 0.5, 0.0))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="one value per radius"):
            _fluence(values=(1.0, 1.05, 1.08))

    def test_all_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _fluence(values=(0.0,) * 5)

    def test_too_few_points_raise(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            _fluence(radii=(0.0,), values=(1.0,))

    def test_non_finite_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            _fluence(values=(1.0, 1.05, float("nan"), 0.5, 0.0))

    def test_non_positive_reference_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="reference_distance"):
            _fluence(reference_distance=0.0)


class TestFileLoader:
    def test_reads_a_two_column_table_and_converts_mm_to_cm(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The PPBKC/pyvsm ``primflu.dat`` layout: comments, then radius_mm value."""
        path = tmp_path / "primflu.dat"
        path.write_text(
            "# Date: 2008/01/08\n# Artiste 2 (M5166) kernel input data\n"
            "0.0\t1.0\n14.14\t1.001290323\n350.0   0.0\n"
        )
        f = RadialFluence.from_file(path)
        np.testing.assert_allclose(f.radii, [0.0, 1.414, 35.0])
        np.testing.assert_allclose(f.values, [1.0, 1.001290323, 0.0])
        assert f.max_radius == pytest.approx(35.0)

    def test_radius_scale_is_overridable(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "cm.dat"
        path.write_text("0.0 1.0\n10.0 0.0\n")
        f = RadialFluence.from_file(path, radius_scale=1.0)
        assert f.max_radius == pytest.approx(10.0)
