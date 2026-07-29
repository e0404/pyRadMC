# API reference

Everything on this page is importable directly from the top-level `pyRadMC` namespace and
is **supported and versioned**. Anything not documented here is an implementation detail
that may move between releases, even if it is importable.

```python
from pyRadMC import VoxelGrid, ReferenceEngine, WarpEngine  # etc.
```

Re-exports are lazy, so importing `pyRadMC` costs nothing and does not pull in NumPy,
SciPy, Warp or SimpleITK. A core-only install can name `WarpEngine` without having warp
installed; touching it then raises an error naming the extra to install.

## Engines

::: pyRadMC.backends.ref.engine.ReferenceEngine

::: pyRadMC.backends.warp.engine.WarpEngine

## Results

::: pyRadMC.backends.results.TransportResult

::: pyRadMC.backends.results.RunProvenance

::: pyRadMC.scoring.dij.DijResult

## Geometry and scoring

::: pyRadMC.geometry.grid.VoxelGrid

::: pyRadMC.scoring.grid.ScoringGrid

## Cross-sections and materials

::: pyRadMC.data.interface.CrossSectionSource

::: pyRadMC.data.interface.PhotonProcess

::: pyRadMC.data.analytic.AnalyticCrossSections

::: pyRadMC.data.tabulated.source.TabulatedCrossSections

::: pyRadMC.data.materials.MaterialData

## Sources

::: pyRadMC.geometry.source.Source

::: pyRadMC.geometry.source.BeamletSource

::: pyRadMC.geometry.source.Primary

::: pyRadMC.geometry.source.PencilBeamSource

::: pyRadMC.geometry.source.ParallelBeamSource

::: pyRadMC.geometry.source.BeamletGridSource

::: pyRadMC.geometry.source.GaussianSpotBeamSource

::: pyRadMC.geometry.source.GaussianSpotBeamletSource

::: pyRadMC.geometry.source.PrimaryFluenceBeamSource

::: pyRadMC.geometry.source.PrimaryFluenceBeamletSource

::: pyRadMC.geometry.source.SpectralBeamSource

::: pyRadMC.geometry.source.SpectralBeamletSource

::: pyRadMC.geometry.source.CompositeSource

::: pyRadMC.geometry.source.CompositeBeamletSource

::: pyRadMC.geometry.phasespace.PhaseSpaceSource

::: pyRadMC.geometry.phasespace.InMemoryPhaseSpaceSource

## Spectra

::: pyRadMC.geometry.spectrum.Spectrum

::: pyRadMC.geometry.spectrum.ali_rogers_mv

## Primary fluence

::: pyRadMC.geometry.fluence.RadialFluence

## Random numbers

::: pyRadMC.rng.host.HostRNG

## Accuracy-defining constants

These live in one place so that a change to any of them is visible in a diff. They are
not tuning knobs: changing one requires a test demonstrating the dosimetric effect.

::: pyRadMC
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
| `pyRadMC.geometry.collimation` | Jaw pairs, rounded-tip MLC, collimated and transmission-mask source wrappers |
| `pyRadMC.geometry.head` | Treatment-head pre-solve producing an exit-plane phase space |
| `pyRadMC.adapters.ct` | Hounsfield calibration and CT image reading |
| `pyRadMC.data.tabulated` | Table precompiler and the EPICS build tool |
| `pyRadMC.study` | Toy fluence optimizer and DVH endpoints, used by the noise/bias study |
