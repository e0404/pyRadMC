# Changelog

All notable changes to pyRadMC are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a dose engine, one extra rule applies: **any change that can move a
computed dose is listed under `Changed` or `Fixed` even when it is an improvement**, with
the measured size of the effect where one was taken. A user re-running last month's plan
must be able to find out from this file whether the numbers should have moved.

## [Unreleased]

## [0.1.0] — unreleased

First public release.

### Added

- **Beamlet-resolved dose influence matrix (Dij)** in sparse CSC layout with per-entry
  statistical uncertainty, on CPU and CUDA. Scheduling — beamlet grouping, chunking, batch
  merging, multi-device sharding — is bit-inert; a 1x1-beamlet column reproduces the
  open-field run bit for bit on a given device.
- **Two backends from one physics source.** A pure-NumPy reference engine (the correctness
  oracle) and a Warp backend compiling the identical scalar physics to CPU and CUDA.
- **Condensed-history electron transport** with a random hinge and a Goudsmit-Saunderson
  multiple-scattering model as the shipped default; a Gaussian small-angle model is
  retained as a test instrument. Woodcock delta tracking for photons.
- **Cross-sections behind an interface.** An analytic closed-form parameterization, and a
  tabulated backend compiled from IAEA EPICS 2023 (EPDL photons with coherent form
  factors, EEDL scattering, ICRU-37 stopping).
- **Materials.** ICRP/ICRU reference media (water, air, lung, adipose, cortical bone) plus
  tungsten, each carrying elemental composition, I-value and Sternheimer coefficients.
- **Sources.** Pencil, parallel and beamlet-lattice beams; polyenergetic spectra including
  the Ali & Rogers MV parameterization; divergent spectral beams and beamlets; Gaussian
  focal spots; IAEA phase-space files (streaming, mmap-backed) and in-memory phase spaces;
  composite/mixture sources; and public `Source`/`BeamletSource` base classes so user
  sources are first-class on both backends.
- **Beam-limiting devices.** Jaw pairs (straight or focused edges) and a rounded-tip MLC as
  parametric ray-attenuation geometry, usable three ways: a treatment-head pre-solve
  scoring an exit-plane phase space, deterministic transmission wrappers that consume no
  random numbers (so correlated Dij sampling survives), or bare divergent fans.
- **Decoupled dose grid.** Scoring on any origin/spacing/shape, including subregions, with
  exact separable voxel-overlap mass rebinning and an unscored energy ledger bucket, so
  `emitted == deposited + unscored + escaped` stays exact. Optional `dose_to_water`
  scoring via restricted stopping-power ratios.
- **CT adapter** (`pyRadMC.adapters.ct`): Hounsfield calibration to density and material,
  and a SimpleITK reader for DICOM/NIfTI/MetaImage.
- **Correlated sampling across beamlets**, the shipped Dij default, which roughly halves
  the renormalized plan-dose error at matched per-beamlet sigma.
- **Result provenance.** Every engine result carries a `RunProvenance` record — version,
  backend, device, seed, cutoffs, multiple-scattering model, resolved substep fraction and
  the cross-section citation — so an archived dose still says what produced it.
- **Public API.** Everything reachable from the top-level `pyRadMC` namespace is supported
  and versioned; re-exports are lazy, so `import pyRadMC` pulls in no third-party module
  and a core-only install does not fail on the names of optional backends. The package
  ships a PEP 561 `py.typed` marker.
- **Cross-section library integrity.** `python -m pyRadMC.data.tabulated.build` verifies
  every downloaded or cached EPICS library against a pinned SHA-256 before compiling
  tables, and refuses to proceed on a mismatch.

### Known limitations

See `AGENTS.md` section 8 for the full list. The two most likely to bite:

- Under correlated sampling — the default — Dij column sigmas are **not** independent and
  must never be combined in quadrature to form a plan-dose sigma. `variance_csc()` exports
  per-column variance and warns.
- The depth-dose validation gate is 5 %/3 mm, not 2 %/2 mm. The limiter is the reference
  benchmark's missing geometry and transport metadata, not the cross-section data.

[Unreleased]: https://github.com/e0404/pyRadMC/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/e0404/pyRadMC/releases/tag/v0.1.0
