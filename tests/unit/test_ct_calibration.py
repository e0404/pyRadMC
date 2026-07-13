"""HU -> (density, material) calibration and CT array -> VoxelGrid (Phase 5 CT adapter).

The pure, dependency-free half of the CT adapter: a Schneider-like Hounsfield
calibration that a scanner-independent default provides and a site overrides. Maps CT
numbers to a per-voxel mass density (a piecewise-linear ramp) and to a registry
material index (HU threshold bins into the ICRP media), then assembles a VoxelGrid. No
SimpleITK here — that is the image-reading slice; this is the physics of the mapping.

The default numbers are illustrative Schneider-style values (Schneider, Bortfeld &
Schlegel, Phys. Med. Biol. 45, 459 (2000), doi:10.1088/0031-9155/45/2/314; the bilinear
density ramp of Schneider, Pedroni & Lomax, Phys. Med. Biol. 41, 111 (1996)); a real
plan supplies the scanner's measured calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRadMC.adapters.ct import DEFAULT_CALIBRATION, HounsfieldCalibration, grid_from_hu
from pyRadMC.data.materials import ADIPOSE, AIR, CORTICAL_BONE, LUNG, MATERIALS, WATER


class TestDefaultDensity:
    def test_canonical_anchors(self) -> None:
        """Air, water and a bone point sit where the ramp is anchored."""
        assert DEFAULT_CALIBRATION.density(-1000.0) == pytest.approx(0.00121, rel=1e-3)
        assert DEFAULT_CALIBRATION.density(0.0) == pytest.approx(1.0, rel=1e-6)
        assert DEFAULT_CALIBRATION.density(1400.0) == pytest.approx(1.85, abs=0.05)

    def test_monotonic_and_clamped(self) -> None:
        """Density rises with HU and clamps flat outside the ramp (never non-positive)."""
        hu = np.linspace(-2000.0, 5000.0, 500)
        rho = DEFAULT_CALIBRATION.density(hu)
        assert np.all(np.diff(rho) >= 0.0)
        assert np.all(rho > 0.0)  # VoxelGrid rejects non-positive density
        assert rho[0] == pytest.approx(DEFAULT_CALIBRATION.density(-2000.0))
        # Below the lowest and above the highest node it holds flat.
        assert DEFAULT_CALIBRATION.density(-5000.0) == DEFAULT_CALIBRATION.density(-1000.0)

    def test_vectorized(self) -> None:
        out = DEFAULT_CALIBRATION.density(np.array([-1000.0, 0.0]))
        assert out.shape == (2,)
        assert out[1] == pytest.approx(1.0)


class TestDefaultMaterial:
    @pytest.mark.parametrize(
        ("hu", "expected"),
        [
            (-1000.0, AIR),
            (-500.0, LUNG),
            (-60.0, ADIPOSE),
            (0.0, WATER),
            (60.0, WATER),
            (800.0, CORTICAL_BONE),
        ],
    )
    def test_segmentation_bins(self, hu: float, expected: int) -> None:
        assert DEFAULT_CALIBRATION.material(hu) == expected

    def test_vectorized_returns_int_indices(self) -> None:
        mats = DEFAULT_CALIBRATION.material(np.array([-1000.0, 0.0, 800.0]))
        assert mats.dtype.kind == "i"
        assert list(mats) == [AIR, WATER, CORTICAL_BONE]

    def test_every_bin_is_a_registry_material(self) -> None:
        mats = DEFAULT_CALIBRATION.material(np.linspace(-2000.0, 4000.0, 1000))
        assert np.all((mats >= 0) & (mats < len(MATERIALS)))


class TestValidation:
    def test_density_nodes_must_be_ascending(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            HounsfieldCalibration(
                density_hu=(0.0, -1000.0),
                density_values=(1.0, 0.0012),
                material_thresholds=(0.0,),
                material_indices=(WATER, CORTICAL_BONE),
            )

    def test_density_values_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            HounsfieldCalibration(
                density_hu=(-1000.0, 0.0),
                density_values=(0.0, 1.0),
                material_thresholds=(0.0,),
                material_indices=(WATER, CORTICAL_BONE),
            )

    def test_material_indices_length_matches_bins(self) -> None:
        """n thresholds define n+1 bins, so n+1 indices are required."""
        with pytest.raises(ValueError, match="bins"):
            HounsfieldCalibration(
                density_hu=(-1000.0, 0.0),
                density_values=(0.0012, 1.0),
                material_thresholds=(-950.0, 100.0),
                material_indices=(AIR, WATER),
            )

    def test_material_thresholds_must_be_ascending(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            HounsfieldCalibration(
                density_hu=(-1000.0, 0.0),
                density_values=(0.0012, 1.0),
                material_thresholds=(100.0, -950.0),
                material_indices=(AIR, WATER, CORTICAL_BONE),
            )

    def test_material_indices_must_be_in_the_registry(self) -> None:
        with pytest.raises(ValueError, match="registry"):
            HounsfieldCalibration(
                density_hu=(-1000.0, 0.0),
                density_values=(0.0012, 1.0),
                material_thresholds=(0.0,),
                material_indices=(WATER, len(MATERIALS)),
            )


class TestGridFromHu:
    def test_builds_a_voxel_grid_from_a_hu_volume(self) -> None:
        """A HU array plus spacing becomes a VoxelGrid with mapped density and material."""
        hu = np.array(
            [[[-1000.0, 0.0], [-500.0, 800.0]]],  # air, water | lung, bone
            dtype=np.float64,
        )
        grid = grid_from_hu(hu, spacing=(0.3, 0.3, 0.3))
        assert grid.shape == hu.shape
        assert grid.spacing == (0.3, 0.3, 0.3)
        assert grid.material[0, 0, 0] == AIR
        assert grid.material[0, 0, 1] == WATER
        assert grid.material[0, 1, 0] == LUNG
        assert grid.material[0, 1, 1] == CORTICAL_BONE
        assert grid.density[0, 0, 1] == pytest.approx(1.0)
        assert grid.density[0, 0, 0] == pytest.approx(0.00121, rel=1e-3)

    def test_origin_is_carried_through(self) -> None:
        hu = np.zeros((2, 2, 2))
        grid = grid_from_hu(hu, spacing=(0.2, 0.2, 0.2), origin=(-1.0, -2.0, -3.0))
        assert grid.origin == (-1.0, -2.0, -3.0)

    def test_material_and_density_arrays_have_grid_dtypes(self) -> None:
        hu = np.zeros((3, 3, 3))
        grid = grid_from_hu(hu, spacing=(0.3, 0.3, 0.3))
        assert grid.density.dtype == np.float64
        assert grid.material.dtype.kind == "i"

    def test_custom_calibration_overrides_default(self) -> None:
        """A site calibration replaces the default ramp and bins."""
        flat = HounsfieldCalibration(
            density_hu=(-1000.0, 1000.0),
            density_values=(0.5, 1.5),
            material_thresholds=(0.0,),
            material_indices=(WATER, CORTICAL_BONE),
        )
        hu = np.array([[[-1000.0, 1000.0]]])
        grid = grid_from_hu(hu, spacing=(0.3, 0.3, 0.3), calibration=flat)
        assert grid.density[0, 0, 0] == pytest.approx(0.5)
        assert grid.material[0, 0, 0] == WATER
        assert grid.material[0, 0, 1] == CORTICAL_BONE
