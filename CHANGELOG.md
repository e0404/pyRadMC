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

- **Jawless dual-layer MLC commissioning example** (`examples/halcyon_commissioning_demo.py`,
  a jupytext percent notebook that also runs as a plain script). The same beam-data session as
  `commissioning_demo.py` against a Halcyon-like head: no jaws, two stacked MLC layers of 29
  and 28 leaf pairs at 1 cm projected pitch — the leaf counts *are* the half-pitch stagger, and
  the aperture is their intersection — and an unflattened 6 MV FFF beam. Produces depth doses
  and both principal profiles at five depths for a sweep of square fields, diagonals of the
  largest, and a rectangular x-by-y output-factor matrix. No engine behaviour changes and no
  shipped dose moves; this exercises existing collimation, pre-solve, source and scoring code.

  Three things it establishes that the flattened notebook does not. **Field widths need no
  fitted offset**: every edge in the swept set, square and rectangular alike, is realized to
  within 0.1 mm at the isocentre with nothing tuned to a measurement, from the rounded-end
  tangent construction evaluated at each bank's own mid-plane (the two differ by 0.16 mm at a
  10 x 10) plus a solved radiation-edge correction. That correction was missing at first, and
  the way it surfaced is the useful part: the tangent construction places the ray that
  *grazes* the tip arc, which passes through zero tungsten and so marks full transmission
  rather than the 50 % radiation edge. The two coincide while the ray is near-normal to the
  arc and separate as it turns oblique — under 0.02 mm out to a 10 x 10 field, 0.16 mm per
  side at 20 x 20, 0.47 mm at 28 x 28 — so the largest field came out **0.84 mm narrow** while
  every smaller one was exact, which is why it went unnoticed. Three independent runs measured
  it to within 0.14 mm of each other and a transport-free ray trace reproduced it to 0.05 mm,
  making it the one discrepancy in this example whose noise floor is far below its size.
  Correcting it needs the attenuation and so has no closed form, but nothing in it is fitted. **The field edge on an
  unflattened beam is the profile's inflection point, not 50 % of the axis** (Fogliata et al.,
  *Med. Phys.* **39** (2012) 6455): the classical construction reports a 40 mm penumbra and a
  4 mm-narrow field at 28 x 28, describing the cone rather than the collimator; both
  conventions are computed and the classical ones kept in the CSV. And **a jawless head leaks
  where a jawed one cannot** — leaf banks were built only over the strips the source fan
  reaches, leaving no material outside them, and the wide extra-focal source floods through
  that hole. Caught by the extra-focal component being non-monotone in field size (a 2 x 2
  field read 3.3x the extra-focal of a 2 x 10); closed by fixed shielding at the fan boundary,
  after which it falls 6.7x and the matrix's Scp(x,y)/Scp(y,x) reciprocity residual goes from
  7 % to under 1 %.

  That shielding closed the flood but not the hole. A 5.25 cm slab is what the space between
  the source plane and the leaves allows, and it transmits 0.9 % of this spectrum where the
  leaf stack it stands in for transmits nothing — so a narrow field was still shielded past
  the fan by a twentieth of the tungsten the machine has there. Because leaf bodies are
  unbounded along travel and strips are not, the residue existed on one axis only: a traced
  28 x 1 came out **5.5 % hotter than a 1 x 28** over ±5 cm, entirely from leakage, and the
  giveaway was that the excess grew with the integration window instead of sitting in the
  aperture. **Each bank is now built over its full nominal width always**, which is what the
  machine has — all 29 and 28 pairs present whatever the field, parked closed. The two
  apertures then sit 0.40 % apart at every window rather than drifting to 5.55 %, and the
  0.9 % plateau at 2 to 4 cm off axis goes to zero. It is free: interleaved A/B at full
  output-factor settings puts 3 strips against 28 at 0.99x, with a null control that builds
  34 either way reading 1.01x.

  Two things worth recording about how it was measured, since both cut against the result.
  The cost was first taken as a cross-run comparison — the same eighteen fields timed 1630 s
  and 5720 s — which looked like a 3.5x regression and was this card's clocks; only the
  interleaved A/B, alternating geometries inside one process against a null control, is
  trustworthy on this hardware. And the fix **does not** explain the effect it was reached
  for: the machine gives Scp(1 x Y) / Scp(Y x 1) = 1.032 where this model gives 1.000, and
  removing the leakage moved that by +0.25 ± 1.00 points, because the leakage sat off axis
  and reached the measuring point only through phantom scatter. Leaf-bank leakage is
  therefore ruled out as the cause of that asymmetry, which remains open.

  That asymmetry is now reproduced, and identifying its mechanism took two eliminations.
  The measured signal is oddly specific: +3.14 %, flat to 0.08 % across every wide side
  from 2 to 28 cm, and zero on every pair with both sides at 2 cm or more. A fixed
  tongue-and-groove loss on the leaf sides is **excluded by shape** — a fixed aperture
  loss must imply the same offset on every row of the matrix, and finite differences on
  the vendor's own grid give 3.06 mm from the 10 mm row against 0.11 mm from the 20 mm
  row, a factor of 28 apart. Source occlusion **saturates** exactly right: calibrated on
  the 10 mm row it has nothing left free and predicts 0.00 % at 20 and 40 mm, where the
  sheet reads 0.03 and −0.06. So the leaf-side focal-spot sigma is now its own parameter
  (`SPOT_SIGMA_V`), fitted in transport over all nine transposed pairs — three
  calibration points bracket the target and the shipped 0.09 cm is their interpolation, a
  **declared one-number fit** that every regenerated matrix re-verifies for free (latest:
  1.0284 ± 0.0068 against the machine's 1.0319). Two cautions are recorded with it: the
  response is steep and convex, so calibrating on a single field pair mis-reads it by 2x,
  and no inplane profile exists in this data set to check the implied 1.8:1 spot, so the
  parameter is honest but unfalsifiable here.

  **Off-axis spectral softening is now modelled, and it was the largest systematic left.**
  The evidence was the in-field residual's *trend across depth* — axis-normalization
  cancels everything depth-independent, and what remained ran −1 to −4 % across 5–30 cm,
  growing with radius, too hard a beam everywhere off axis. The primary now passes through
  radial spectral shells (`SOFTENING_SHELLS`, a `ShellSpectrumSource` wrapper that
  re-draws only the energy of each history by its fan radius — no engine change, no
  statistics waste, and the reference shell is sized so the on-axis PDD column never sees
  a softened spectrum). The dc2 values are **fitted, not derived**: a one-scale quadratic
  in radius — the clean small-angle bremsstrahlung shape — is rejected by the data, so
  this ships as a stepwise table with the same standing as the fluence table. Two fit
  passes were enough, and the second measured its own floor: a control shell whose value
  did not change drifted by +0.11 in dc2 between passes, so iteration stopped there
  rather than chase one realization of noise. The fluence table was then refit jointly
  (+0.8 % at 2 cm to +15 % at the corner — the photon fluence the first fit had bent to
  absorb the missing softening, given back). Net, against the vendor data — both runs
  with the locally fitted fluence table in place, which is the configuration these
  numbers require: diagonal gamma 2 %/2 mm rose from 76–88 % to 88–93 % at every depth,
  the depth trend collapsed to the ±0.7 % measurement floor, and PDDs, output factors and
  field widths stayed put. The honest cost: the ~1-sigma-of-floor overcorrection at mid
  radii redistributed which 30 cm-depth crossline cells fail gamma (five before, five
  after).

  A finding about the vendor data itself, needed to interpret any of the above: **the
  sheet's small-field scans are already detector-corrected, whatever its header says.**
  Against the vendor's own penumbrae, simulated profiles match best convolved with the
  CC13's chord kernel at 6 cm and up but as raw point dose at 4 cm and below — the mean
  penumbra error swaps from 1.2/0.3 mm (convolved/point) at 2 cm to 0.5/1.9 mm at 6 cm,
  and convolving the 2 cm field costs its in-field RMS a factor three. The comparison
  applies the kernel by field size accordingly. The same undocumented processing is the
  best available reading of the 1 cm output-factor row and column, which sit +2 % above
  the model on *both* axes (after `SPOT_SIGMA_V` closes the between-axis asymmetry) and
  are reported apart from the headline block for that reason.

  History allocation is *derived* rather than fitted, since no measured per-field sigma table
  exists for this geometry. One quantity drives it: how dilute a field's measurement is — the
  fan spreads a fixed history count over an area that grows with the field, and the measuring
  volume collapses eighteenfold once the field is too small to hold a chamber. Histories scale
  with it and the phase-space reuse cap inversely to it. Measured against a flat allocation,
  the on-axis error over a 2 to 28 cm range went from **21x spread (0.45 % to 9.5 %) to 2.3x**;
  adding the measuring-volume term then took the fields with a 1 cm side from 5-15 % to a
  comparable band. Most of the machine's geometry is unpublished; the opening cell is an
  explicit inventory of what is stated, what is inferred, and what is a placeholder to be
  refitted against measured data.

  **The off-axis cone can be fitted rather than invented**, through the example's existing
  `PRIMARY_FLUENCE_FILE` hook. The fitted table itself **does not ship** — it is derived
  from the vendor's measured profiles, so it stays local with the workbook and the shipped
  default remains the analytic Gaussian; what ships is the hook, the method (below), and
  the fitted-configuration results quoted in this entry, all reproducible against any
  machine's own beam data. The fit is a first-order correction of `exp(-(r/19.5 cm)^2)`
  against the vendor's
  open-field crosslines and diagonals, and the reason it is a separate table rather than a
  retuned sigma is that no Gaussian fits: the residual peaks at r = 8 cm and returns through
  zero by r = 12, so narrowing sigma to correct the peak over-corrects r = 12 by 11 %. Two
  method notes that cost a first attempt. The correction must be fitted against the **fan**
  radius r * SAD / (SSD + depth), not the physical off-axis distance — in the latter the field
  edge sits at a different radius at every depth, and the fit reads the penumbra as a fluence
  error (the residual at r = 12 cm swings from -4.0 % at 1.3 cm depth to +2.5 % at 30 cm from
  that alone). And a **diagonal must be renormalized against the crossline**, not against its
  own axis sample, which is one voxel and carries a whole-curve offset of a few per cent; in
  fan coordinates all twelve curves then collapse onto one band, with rescale factors of
  0.991 to 1.018. Fitted at 5, 10, 20 and 30 cm depth; 1.3 cm is excluded because it sits in
  the build-up region, where the model has a known deficit that would be pushed into the
  fluence. (A first version of this fit also had to absorb off-axis spectral softening into
  its depth-independent shape, at ~1.5 percentage points of depth residual; since
  `SOFTENING_SHELLS` the softening lives in the spectrum where it belongs, and the table is
  refit jointly on top — when the simulated curves were produced with a fitted table
  already in place, the correction must multiply *that* table, not the analytic cone,
  which is what the fit script's `--base` argument is for.)

  **What the shipped default gives, so nobody is surprised**: on the analytic cone —
  everything else identical, shells and elliptical spot included — the depth doses still
  pass gamma 2 %/2 mm at 100 % on all seven fields, the trusted output-factor block reads
  mean +0.33 %, RMS 1.33 % (on-axis ratios barely see the radial shape), and the
  transposed-pair asymmetry holds at 1.020 ± 0.006 against the machine's 1.032. What the
  invented cone costs is the large-field *shapes*: the 28 x 28 crossline runs −1 to −1.9 %
  in-field (gamma 61–71 %) and the diagonals −3.3 to −4.7 % (gamma 26–44 %), against
  88–93 % on the diagonals with a fitted table. That gap **is** the fluence commissioning,
  and reproducing the fitted numbers on another machine's data is a one-script step.

- **Commissioning example** (`examples/commissioning_demo.py`, a jupytext percent notebook
  that also runs as a plain script). Drives the existing
  collimation, head pre-solve, spectral-source and scoring pieces through a beam-data
  measurement session: a water phantom at a stated SSD, a sweep of square fields set on
  jaws and/or a rounded-tip MLC, and the resulting depth doses, tissue-phantom ratios both
  from real SAD-setup runs and as full curves converted from the depth doses (inverse-square
  plus a phantom-scatter correction; each route checks the other), total scatter factors,
  field widths and penumbrae, plus a
  deterministic ray-traced primary fluence in the measurement plane along both axes and
  both diagonals. Uncertainties come from independent replicates, so nonlinear quantities
  (dmax, field width, penumbra) carry error bars too. No engine behaviour changes.

  It is driven by a **two-component virtual source model** against one machine's measured
  data (a Siemens Artiste): a `PrimaryFluenceBeamSource` carrying the measured radial
  primary fluence, plus a wide Gaussian **extra-focal source** at the flattening-filter
  plane for head scatter, transported separately and combined linearly. The extra-focal
  geometry is taken from the machine files (plane from `params.dat`; width from the
  primary-collimator opening the fluence curve implies) and only its weight is fitted.
  Together they take the total-scatter-factor residual against measured output factors
  from **3.4 % to 0.58 % RMS over 20 to 400 mm**, and the depth-dose-to-TPR consistency
  check from +1.6 % to +0.32 %. Two findings recorded in the notebook: the focal spot and
  the extra-focal weight must be fitted **jointly**, because the spot governs source
  occlusion and therefore moves output factors as well as penumbra; and the notebook's
  penumbra column is **not grid-converged** — at 2.5 mm voxels it reads up to 0.9 mm wider
  than the same physics at 1.25 mm, so it is comparative only.

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
