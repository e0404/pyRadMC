# Changelog

All notable changes to pyRadMC are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a dose engine, one extra rule applies: **any change that can move a
computed dose is listed under `Changed` or `Fixed` even when it is an improvement**, with
the measured size of the effect where one was taken. A user re-running last month's plan
must be able to find out from this file whether the numbers should have moved.

## [Unreleased]

### Added

- **Measured-primary-fluence virtual source model** (`PrimaryFluenceBeamSource`,
  `PrimaryFluenceBeamletSource`, and the `RadialFluence` table they read), after Tacke et
  al., *Med. Phys.* **33** (2006) 1125-1132, doi:10.1118/1.2181298. Photons are sampled on a
  plane in the treatment head, given a direction from a finite Gaussian focal spot and an
  energy from a `Spectrum`, and weighted by the machine's measured radial primary fluence at
  that point — so a commissioning curve's horn and the primary collimator's field edge enter
  the transport directly. `RadialFluence.from_file` reads the two-column PPBKC `primflu.dat`
  layout. Both sources run on the reference and Warp backends through the vectorized
  pre-sampling route, and the beamlet variant is `run_dij`-ready.

  The geometry is exactly `GaussianSpotBeamSource`'s and the fluence enters **as the
  per-history statistical weight, not as importance sampling**, so the draw count stays at
  six uniforms and correlated Dij sampling, chunk-invariance and the vectorized batch route
  are unchanged; a table flat over the emission rectangle reproduces the Gaussian-spot source
  bit for bit. Stated approximations: one spectrum at every off-axis radius (no off-axis
  softening), and zero fluence beyond the last tabulated radius. No existing engine behaviour
  changes and no shipped dose moves.

- **Virtual-source example** (`examples/virtual_source_demo.py`). Runs the model on a
  synthetic Artiste-like commissioning curve (or a real `primflu.dat` via `--fluence-file`)
  and shows the input fluence, the horn transferring into a 30 x 30 cm profile against a
  flat-fluence control on the same seed, and a field-edge penumbra with a point-spot control
  run. That last panel records a measurement worth knowing: the usual quadrature
  back-calculation of focal-spot size from a penumbra (`P_total^2 = P_geom^2 +
  P_transport^2`) **over-estimates the spot**, because the 80-20 widths of a Gaussian spot
  and of the electron-transport kernel do not add in quadrature. On this geometry a spot
  derived for a 5 mm penumbra measured 6.2 mm, with the transport-only control at 3.6 mm
  against the 3.5 mm the derivation assumed.

- **Concurrent forward batches on CUDA.** `WarpEngine.run(...,
  concurrent_batches=N)` can overlap independent statistical batches on private CUDA
  streams while preserving the same-device result bit for bit by folding batch dose maps
  in order. The reference backend and Warp CPU accept the same interface and remain
  sequential. Each CUDA lane owns another queue set and dose map, so memory grows roughly
  linearly with `N`.
- **Device-native spectral transmission masks.** An exact
  `TransmissionMaskSource(SpectralBeamSource(...))` composition now uploads its spectrum
  and mask once and performs source sampling, bilinear mask evaluation and zero-weight
  compaction on the device. `TransmissionMaskSource` exposes its inner source and mask
  plane geometry through accessor properties for backend generators and user inspection.

- Aluminium (Z = 13) in `STANDARD_ATOMIC_WEIGHT`, so the EPDL/EEDL parsers can build it.
  It is not a registered transport material; the entry exists because aluminium is half of
  the Ali–Rogers flattening-filter attenuation and the validation tier now checks those
  parameterizations against EPICS 2023.

### Changed

- Warp's general host-pre-sampled source route no longer seeds exactly-zero-weight
  primaries into transport queues. The attempted-history denominator and global history
  keys are unchanged; reference-vs-Warp agreement remains statistical, and repeated runs
  on one target remain bit-reproducible.

### Fixed

- **The Ali–Rogers 511 keV annihilation line is now filtered with the continuum.** The
  `C4` delta term belongs inside the same tungsten/aluminium envelope as `psi_thin` in
  the paper's function 13; it was previously added after the envelope, leaving the line
  unattenuated. Transmission at 511 keV spans 0.115 (`siemens-6mv`, an 8.7x reduction of
  the line) to 0.0087 (`elekta-25mv`, 115x), so **all nine presets move and the
  high-energy beams move most**. The line's share of emitted photons falls from 36.1% to
  0.71% (`varian-18mv`) and from 48.8% to 0.82% (`elekta-25mv`), raising their mean
  photon energies from 3.05 to 4.51 MeV and from 2.95 to 5.30 MeV — the previous values
  were unphysically soft for those nominal energies. No released version is affected, so
  no published result needs re-running; any dose computed from a preset on `main` before
  this commit is wrong by the amounts above.

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
