# API reference

Everything on this page is importable directly from the top-level `pyradmc` namespace and
is **supported and versioned**. Anything not documented here is an implementation detail
that may move between releases, even if it is importable.

```python
from pyradmc import VoxelGrid, ReferenceEngine, WarpEngine  # etc.
```

Re-exports are lazy, so importing `pyradmc` costs nothing and does not pull in NumPy,
SciPy, Warp or SimpleITK. A core-only install can name `WarpEngine` without having warp
installed; touching it then raises an error naming the extra to install.

## Engines

::: pyradmc.backends.ref.engine.ReferenceEngine

::: pyradmc.backends.warp.engine.WarpEngine

## Results

::: pyradmc.backends.results.TransportResult

::: pyradmc.backends.results.RunProvenance

::: pyradmc.scoring.dij.DijResult

## Geometry and scoring

::: pyradmc.geometry.grid.VoxelGrid

::: pyradmc.scoring.grid.ScoringGrid

::: pyradmc.scoring.cylinder.CylindricalScoringGrid

::: pyradmc.scoring.cylinder.uniform_edges

::: pyradmc.scoring.cylinder.geometric_edges

::: pyradmc.scoring.cylinder.graded_edges

::: pyradmc.scoring.cylinder.common_bin_divisor

## Cross-sections and materials

::: pyradmc.data.interface.CrossSectionSource

::: pyradmc.data.interface.PhotonProcess

::: pyradmc.data.analytic.AnalyticCrossSections

::: pyradmc.data.tabulated.source.TabulatedCrossSections

::: pyradmc.data.materials.MaterialData

## Sources

::: pyradmc.geometry.source.Source

::: pyradmc.geometry.source.BeamletSource

::: pyradmc.geometry.source.Primary

::: pyradmc.geometry.source.PencilBeamSource

::: pyradmc.geometry.source.ParallelBeamSource

::: pyradmc.geometry.source.BeamletGridSource

::: pyradmc.geometry.source.GaussianSpotBeamSource

::: pyradmc.geometry.source.GaussianSpotBeamletSource

::: pyradmc.geometry.source.PrimaryFluenceBeamSource

::: pyradmc.geometry.source.PrimaryFluenceBeamletSource

::: pyradmc.geometry.source.SpectralBeamSource

::: pyradmc.geometry.source.SpectralBeamletSource

::: pyradmc.geometry.source.CompositeSource

::: pyradmc.geometry.source.CompositeBeamletSource

::: pyradmc.geometry.phasespace.PhaseSpaceSource

::: pyradmc.geometry.phasespace.InMemoryPhaseSpaceSource

## Spectra

::: pyradmc.geometry.spectrum.Spectrum

::: pyradmc.geometry.spectrum.ali_rogers_mv

## Primary fluence

::: pyradmc.geometry.fluence.RadialFluence

## Random numbers

::: pyradmc.rng.host.HostRNG

## Accuracy-defining constants

These live in one place so that a change to any of them is visible in a diff. They are
not tuning knobs: changing one requires a test demonstrating the dosimetric effect.

::: pyradmc
    options:
      members:
        - ECUT_MEV
        - PCUT_MEV
        - DIJ_TRUNCATION_RELATIVE
        - PHOTON_ROULETTE_MEV
        - PHOTON_ROULETTE_SURVIVAL
        - PHOTON_ROULETTE_WEIGHT_CAP
        - PHOTON_SPLIT_N
        - ELECTRON_MASS_MEV
        - GY_PER_MEV_PER_G
        - RAYLEIGH_MOMENTUM_TRANSFER_PER_MEV
      show_root_heading: false

## Subsystems documented at module level

These are coherent subsystems with their own vocabulary rather than names a first script
reaches for. They are supported, but imported from their modules:

| Module | What it provides |
|---|---|
| `pyradmc.geometry.collimation` | Jaw pairs, rounded-tip MLC, collimated and transmission-mask source wrappers |
| `pyradmc.geometry.head` | Treatment-head pre-solve producing an exit-plane phase space |
| `pyradmc.adapters.ct` | Hounsfield calibration and CT image reading |
| `pyradmc.data.tabulated` | Table precompiler and the EPICS build tool |
| `pyradmc.study` | Toy fluence optimizer and DVH endpoints, used by the noise/bias study |
