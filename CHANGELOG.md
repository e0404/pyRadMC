# Changelog

All notable changes to pyRadMC are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a dose engine, one extra rule applies: **any change that can move a
computed dose is listed under `Changed` or `Fixed` even when it is an improvement**, with
the measured size of the effect where one was taken. A user re-running last month's plan
must be able to find out from this file whether the numbers should have moved.

## [Unreleased]

### Fixed

- Reference electron transport now reuses and incrementally extends the persisted
  Goudsmit-Saunderson grid shared with Warp. Fresh processes no longer rebuild
  covered nodes. Table construction and dose values are unchanged.
  Brackets extend in one save, and both builders share column parallelism and
  recover from cache write failures. Cache identity includes inverse-CDF resolution;
  compatible existing 4096-bin grids remain readable.
- CI caches physics data by construction identity, source hash, OS and Python version.

## [0.2.0] — 2026-09-05

First public release: the first version published to PyPI and tagged on GitHub.

### Changed

- **Electron multiple-scattering strength is now the Class-II transport moment of
  the Moliere-screened Rutherford law** (`AnalyticCrossSections.scattering_power`,
  via `goudsmit_saunderson.transport_moment_scattering_power`), replacing the
  Rossi-Greisen/Highland core width `(14.1/pv)^2 / X_0`. The strength is
  `2(N_A/A)[Z^2 sigma_tr + Z(sigma_tr - sigma_tr,M^hard(E, ECUT))]`: nuclear elastic
  scattering plus the atomic-electron moment below ECUT, with the moment of explicit
  hard Moller events removed. Highland lacks the energy-growing logarithm of the true
  moment; the restricted result is 1.25x Highland at 1 MeV, 1.68x at 10 MeV and 1.77x
  at 15 MeV in water; the hard-event correction is 0.4-4.2 % of the unrestricted
  `Z(Z+1)` strength over 0.5-15 MeV. **Dose moves in the build-up and laterally.**
  Measured on monoenergetic pencil kernels in water against EGSnrc/TOPAS (1 mm
  voxels, 1 mm axial shell, 1e6 histories): the laterally integrated build-up at
  half dmax goes from 0.95-0.97 to 0.99-1.01 of EGSnrc (1 / 6 / 15 MeV), the
  central-ray excess from +4 / +14 / +19 % to +3 / +2 / +7 % (TOPAS and EGSnrc
  themselves differ by 3-4 % there), and dmax moves shallower (6 MeV: 2.95 -> 2.75 cm,
  onto EGSnrc). Beyond dmax on the central axis nothing moves (CPE). Electron-beam
  R50 falls onto EGS4 (Rogers & Bielajew 1986, Table III): 1.969 / 4.157 / 8.50 cm at
  5 / 10 / 20 MeV vs 1.952 / 4.138 / 8.451, where Highland read 8-9 % long; the R50
  validation gate is re-derived against that table (3 / 5 / 10 MeV, lateral-equilibrium
  geometry, +-4 %) in place of the old R50/R_CSDA detour window. **The tabulated
  source moves too**: it now adds the same restricted soft-electron term to EEDL's
  nuclear-only elastic moment (build-up +1-1.5 % at half dmax, central ray -0.5 / -2 /
  -3 % at 1 / 6 / 15 MeV; 15 MeV axial excess +11 % -> +8 %), and compiled table
  format version 2 prevents older nuclear-only tables from loading silently.
  The moment is evaluated per substep and costs runtime: electron transport on the
  reference backend measured **~12 % slower** than the previous Highland width
  (interleaved A/B on `tests/integration/test_boundary_truncation.py`, 149-155 s
  before against 169.9 s after).
- **The import name is now lowercase: `import pyradmc`**, following PEP 8 package
  naming. Every module path changes with it (`pyradmc.data`, `pyradmc.geometry`, ...),
  as do the default EPICS cache directory (`~/.cache/pyradmc/epics`) and the leading
  token of the provenance summary string. The distribution keeps its display name, so
  `pip install pyRadMC` and `pip install pyradmc` are the same package (PyPI normalizes
  case), and the repository stays `e0404/pyRadMC`. Made before the first PyPI release,
  while no released import site exists to break. The `PYRADMC_*` environment variables
  are unchanged. No physics, numerics or dose moves.

### Added

- **Tungsten heavy-alloy material** (`TUNGSTEN_ALLOY`: W 0.95 / Ni 0.035 / Cu 0.015 by
  mass at 18.0 g/cm^3), the published composition surrogate for Halcyon-class dual-layer
  MLC leaves. This is the alloy entry the `TUNGSTEN` docstring anticipated, retiring the
  pure-W-at-caller-density approximation (~1-2 % in mu/rho at MV energies, and about a
  factor two in single-bank transmission, which is exponential in rho t) for callers that
  use it. Provenance: I = 692.5 eV and the radiative anchors are NIST ESTAR's
  user-defined-material output for exactly this composition and density; the Sternheimer
  density-effect row is not in SBS-1984, so cbar comes from the exact plasma-energy
  relation (which reproduces the published tungsten row) and a/m/x0/x1 are fitted to
  ESTAR's exact delta column over its full grid (max |delta error| 0.028). Gated against
  ESTAR like the ICRP media in `tests/unit/test_tungsten_alloy.py`, with an extra
  absolute pin on the fitted density-effect parameterization. No engine behaviour changes
  and no shipped dose moves; nothing used the entry before it existed.

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

- **Jawless dual-layer head fed from a fitted virtual-source phase space**
  (now the phase-space route of `examples/commissioning_vsm_example.py`). The commissioning example's collimator with its own source model removed: photons
  and contaminant electrons come from an uncollimated IAEA phase space sampled from a virtual
  source model fitted elsewhere against the same vendor beam data, scored 2 mm above the
  proximal bank, so spectrum, focal spot, off-axis fluence and softening, head scatter and
  contamination are all in the particles and nothing about the beam is fitted here. The file
  is memory-mapped and scanned once (78 M records in ~6 s); each field samples only the
  records aimed into its fan (0.98 M for a 2 x 2, 67 M for a 28 x 28), photons through the
  head pre-solve and electrons through the attenuation route, combined by the file's own
  counts. Same summary/curve tables as the commissioning example; the output-factor matrix is
  deferred (the source model's monitor-backscatter term is a per-field MU factor no phase
  space carries). No engine behaviour changes and no shipped dose moves.

  Measured against the vendor workbook at the shipped statistics (40 replicates x 50 M
  histories per field at the reference size, 3.4 h on an RTX 4070 laptop; the phase space
  does not ship): **PDD gamma 2 %/2 mm 100 % on all seven fields**, mean |residual| over
  3-30 cm depth 0.12 points and PDD10 within 0.35 points; crossline in-field RMS
  0.3-1.1 % for fields >= 4 cm (1.8-2.8 % at 2 x 2, where the vendor scan is point dose);
  widths within 1.3 mm at every field and depth with no fitted leaf offset; diagonals
  in-field RMS 0.6-1.4 %, gamma 84-100 %. Achieved on-axis error 0.29-0.53 % per field.
  The phase space's own contaminant electrons carry 0.06 % (2 x 2) to 4.64 % (28 x 28) of
  the dose at 0.5 cm depth. Those figures and the tables beside them were produced at
  `ELECTRON_HISTORY_FRACTION = 0.02`, since lowered to 0.005: the electron component's
  own error doubles, which is under 0.1 % of the total dose and moves nothing reported
  here, while the sweep gets most of its wall clock back (electrons take the host-side
  attenuation route, and it was making the run CPU-bound with the GPU idle).

  The statistics were raised fourfold (`REPLICATES` 10 -> 40 with `HISTORIES` scaled in
  step) and the error fell as 1/sqrt(N) on every channel -- median improvement 2.34x on the
  on-axis sigma, 2.16x on depth-dose noise, 2.14x on profile-core noise, against the ideal
  2.0x. **The run is therefore transport-limited, not limited by re-use of the stored phase
  space**, which was the open question: at 8-40x re-use of the pre-solved head, more
  histories still buy accuracy at the full rate.

  Two residuals survive that increase and are therefore the source model's, not noise.
  The **build-up deficit is systematic**: mean |residual| over 0.4-1.0 cm is 1.09 points at
  4x statistics against 1.11 at 1x, i.e. unmoved. And **`dmax` does not shift with field
  size**: the simulated peak sits within 0.04 cm of one depth for every field from 2 to
  28 cm, while the measured one shallows steadily as the aperture opens, so the model is
  exact at 6 x 6 and 1.1 mm too deep at the largest field. Both point the same way -- shallow dose
  that should grow with aperture and does not -- and both corroborate the source model's own
  stated health warning, that its contamination component sits pinned at its upper bound and
  "always wants more surface dose than it is allowed". Everything from the peak downward is
  unaffected: the deep residual of 0.12 points is the best channel in the comparison.

  Two data-handling notes recorded in the notebook, neither fixed in the library. The pyvsm
  export writes the IAEA ``W`` flag as 0 with a ``RECORD_CONSTANT``, which
  :func:`read_iaea_header` rejects (correctly -- ``w``'s sign rides on the particle type,
  which is a stored-flag convention), so the notebook builds the header itself. And the
  notebook now **refuses to transport a phase space that disagrees with its own stated
  spectrum** by more than 50 keV in on-axis mean energy: the v46 export was 0.17 MeV from
  its readme and truncated 0.9 MeV below its own endpoint, and that guard is what would
  have caught it before a two-hour run rather than after.

- **Jawless dual-layer MLC commissioning example** (`examples/commissioning_vsm_example.py`,
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

  **What the shipped default gives, so nobody is surprised** (numbers from the shipped
  artifacts' own regeneration, contaminant electrons, alloy leaves and surrogate bank
  geometry included): on the analytic cone — everything else identical, shells and
  elliptical spot included — the depth doses pass gamma 2 %/2 mm at 100 % on all seven
  fields, the trusted output-factor block reads mean +0.06 %, RMS 1.40 %, median 0.7
  simulated sigma (on-axis ratios barely see the radial shape; the boosted reference
  keeps the block mean's common-mode term near 0.3 %), and the transposed-pair asymmetry
  holds at 1.035 ± 0.006 against the machine's 1.032. What the invented cone costs is
  the large-field *shapes*: the 28 x 28 crossline runs −1 to −1.7 % in-field (gamma
  60–71 %) and the diagonals −3.0 to −4.5 % (gamma 28–45 %), against
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

- **Sub-voxel deposit resolution**, `deposit_resolution_cm` on `ReferenceEngine.run`,
  `WarpEngine.run` and both `run_dij`. A condensed-history substep is capped at the
  transport voxel face and its collision loss is filed at the half-step midpoint, so
  once a step is longer than a voxel every deposit lands near the voxel centre and
  the *sub-voxel* dose profile becomes a tent — peaked at the centre, starved at the
  faces. Measured on a 6 MeV pencil beam scored at a tenth of the voxel: 76 % peak-to-
  trough at the voxel period, rising with energy as the substep outgrows the voxel
  (1 % at 1 MeV, 21 % at 2, 78 % at 6, 104 % at 15). Setting this splits each
  half-step into pieces no longer than it, which removes the modulation.

  **The value must divide the scoring bin width.** Deposits sit at a fixed spacing
  from the step start and step starts are pinned to voxel faces, so the point set is
  locked to the lattice; a spacing that does not divide the bin width beats against
  it, and the fixed phase makes that beat a standing ripple rather than noise.
  Measured at 6 / 15 MeV over 0.025 cm bins: 0.010 cm (ratio 2.5) leaves 5.7 / 12.7 %,
  0.005 cm (ratio 5) leaves 1.3 / 1.7 %, 0.0025 cm leaves 0.7 / 0.8 %. The scoring
  geometries therefore suggest **half the largest common divisor of their bin
  widths** as `deposit_resolution_cm`, not simply their finest bin.

  **The default is unchanged and this fixes nothing on its own.** `None` is the
  single midpoint deposit, byte-identical to every result produced before it
  existed, and the artifact remains the default behaviour. The option is opt-in
  because it costs roughly 1.4x (broad field) to 2x (pencil beam) at four pieces per
  half-step, and buys nothing when dose is scored at voxel resolution — which is
  every result the engine has produced so far. Energy is only ever moved *within* a
  voxel: the books, and dose coarse-grained back to the voxel pitch, are unchanged
  (test-pinned). `RunProvenance` records it, since it is not recoverable from the
  dose array.

- **Mono-energetic pencil-beam kernels.** A new cylindrical scoring geometry,
  `CylindricalScoringGrid`, bins dose by depth and by radial shell about a beam axis —
  the geometry a pencil-beam kernel is defined on. It is accepted by the `scoring_grid`
  argument of `ReferenceEngine.run` and `WarpEngine.run`, so it works on the reference,
  Warp CPU and Warp CUDA backends through one code path, with per-shell sigma from the
  usual batch statistics. Shell edges are the caller's (`uniform_edges`,
  `geometric_edges`, or any increasing array); per-shell mass is the analytic annulus
  volume, and `for_grid` refuses a binned region the phantom does not cover or does not
  fill uniformly rather than approximating it. Electron primaries work through the
  existing `primary_kind="electron"` range instrument; electron beams as a clinical
  modality remain out of scope. Demonstrated by `examples/pencil_kernel_demo.py`.

  Both axes take arbitrary bin edges: `geometric_edges` for equal-ratio radial
  shells, `graded_edges` for a depth schedule that is fine through the build-up
  region and coarse in the tail (0.010/0.025/0.25/1.00 cm over 0-32 cm is 152 bins
  where a uniform fine grid would need 3200). One `edge_bin_index` primitive serves
  both, on host and device.
- **Pencil-beam kernel database script**, `examples/pencil_kernel_database.py`:
  sweeps a photon energy series and writes the kernels to a single `.npz` with bin
  edges, per-bin sigma and run provenance. Reports achieved uncertainty per dose
  band rather than as one global number, because the outermost shells are many
  orders below the peak and no history count fixes that.

  **This moves no dose.** Transport still runs on the rectilinear `VoxelGrid` and never
  sees the scoring geometry: for a run scored on a grid, emitted and escaped energy are
  bit-identical to before, and the Warp rectilinear deposit path is byte-identical
  (both test-pinned).
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

- **The commissioning examples are one example.** `examples/commissioning_demo.py` (a
  flattened head with jaws) and `examples/halcyon_vsm_phasespace_demo.py` (a phase-space
  variant of the jawless head) are removed, and
  `examples/commissioning_vsm_example.py` becomes
  **`examples/commissioning_vsm_example.py`**, which now carries both source routes: the
  analytic model it always had, and — through `PHASESPACE`, or the
  `HALCYON_VSM_PHASESPACE` environment variable — an IAEA phase space that replaces the
  whole source model with already-sampled particles.

  The phase-space route is the one the deleted variant contributed, and it arrives intact
  rather than as the crude hook that stood there before: the file is memory-mapped and
  scanned once for every record's isocentre aim point, each field draws only the records
  aimed into its own fan, photons and electrons are separated by kind, and the stated
  spectrum is checked against the records before anything is transported. Because the
  photons are now kind-pure they go through the head pre-solve like the analytic ones,
  where the old hook had to fall back to deterministic attenuation.

  Nothing is lost from the surviving example and no shipped dose moves: the default route
  is byte-for-byte the configuration that produced the previous figures. What goes is the
  flattened-head session — jaws, TPRs and the `Sc`/`Sp` conversion — which was a second
  worked example of the same exercise on a machine whose geometry the collimation tests
  already cover.

- **The Halcyon example carries a contaminant-electron source** (`CONTAMINANT_ENERGY`,
  `CONTAMINANT_WEIGHTS`; the example's shallow doses move, the engine does not change —
  the component re-badges an existing photon source's emissions as electrons and rides
  the attenuation route). Derived from the vendor's own 1 mm build-up columns by
  inverting the depth-dose normalization per field: the implied component is
  electron-like on range (spent by ~1.2 cm), magnitude (1-4 % of peak at 0.5 cm) and
  field-size scaling. Mono-energetic 2.2 MeV (fitted on a 1.5/2.2/3.0 grid), emitted
  from the extra-focal source's plane with weight 4.3e-4 per unit on-axis photon
  fluence. **The companion beam-fan term was built, fitted, and came out zero** — its
  basis is 4.3x steeper in field size than the two-population argument assumed, and the
  residual it was invented for *decreases* with field size, which no contamination
  mechanism does; that small-field shallow deficit (+1.5 % of peak at 2 x 2, gone by
  8 x 8) stays an open item rather than being absorbed into a fitted source. Validated
  against pre-committed targets on the fitted-fluence configuration: the large-field
  dmax trend turns over and moves toward the measured curve (28 x 28 error 2.6 mm ->
  0.9 mm), the 0.4-2.5 cm build-up residual at 20/28 cm collapses (4.2/4.4 % -> 0.2/
  0.9 % of peak), and the 1.3 cm-depth profiles and diagonal — which no part of the fit
  ever saw — improved or held (diagonal gamma 88.2 -> 93.2 %). Output factors are
  immune by construction (electrons are spent centimetres above the 5 cm scoring
  depth). Stated approximations recorded at the definition, including that the fitted
  head term is an *effective* source absorbing the unmodelled carbon-fibre bore cover's
  electrons.

- **The Halcyon example's leaves are now `tungsten_alloy`, on the literature-surrogate
  bank positions** (`examples/commissioning_vsm_example.py`; the example's numbers move,
  the engine's do not). Three coupled updates from checking the model against a
  literature-informed geometry specification: the leaf material changes from pure W at
  19.30 to the alloy at 18.0 (single-bank narrow-beam transmission roughly doubles to
  0.19 % by energy fluence; in-field dose is unaffected), the bank positions move to the
  surrogate 68.8-61.1 / 60.8-53.1 cm SID with a 3 mm interlayer gap (was 69.0/60.0 with
  ~1.3 cm; traced radiation edges move by at most 0.011 mm at the isocentre, so widths
  and penumbrae are preserved by construction), and a new transmission check cell prints
  the narrow-beam single- and dual-layer transmission against the published ~0.4-0.5 % /
  ~0.01 % dose figures, with the unmodelled contributions (interleaf leakage, in-leaf
  build-up, phantom scatter) stated. The dependency-free `DEVICES = "water"` stand-in
  follows the leaves to 18.0, so it keeps reproducing the shipped geometry's areal
  density rather than pure tungsten's. A stale tip-retraction comment (0.67/0.70 mm) was
  corrected to the measured 0.76/0.62 mm — the values this file's own 0.16 mm
  differential always implied.

- **The Halcyon example's reference field gets twelve times the histories, not three**
  (`REFERENCE_HISTORY_BOOST`; the example's quoted output-factor errors move, the engine
  does not). Every output factor is a ratio against the one reference field, so that
  field's own draw lands on all hundred entries at once and cancels nowhere. Two full
  regenerations whose raw doses agreed to 0.04 ± 0.2 per cent still differed by 1.3 per
  cent in *every* output factor simultaneously, because the block mean inherits the
  reference draw (~0.6-0.9 per cent at boost 3) and the per-entry sigmas share the shift
  rather than flagging it. At 12 the common mode falls to ~0.3 per cent, for about an
  hour more on a ten-hour matrix.

- **The Halcyon example's figures and tables are regenerated** from the shipped
  configuration with the alloy leaves, the surrogate bank geometry, the contaminant
  electrons and the boosted reference in place, and the "what the shipped default gives"
  baseline above quotes that run rather than the previous one. The run also re-verified
  the nine-pair transposed-field asymmetry on the new geometry: 1.035 ± 0.006 simulated
  against the machine's 1.032, the third consistent read of the fitted
  `SPOT_SIGMA_V = 0.09`.

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

## 0.1.0 — never released

Prepared as the first release and never tagged, so no `v0.1.0` exists to install or
compare against. Everything below ships in 0.2.0; it is kept as its own section because
it is the record of what the engine could already do before that release's changes.

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
- **CT adapter** (`pyradmc.adapters.ct`): Hounsfield calibration to density and material,
  and a SimpleITK reader for DICOM/NIfTI/MetaImage.
- **Correlated sampling across beamlets**, the shipped Dij default, which roughly halves
  the renormalized plan-dose error at matched per-beamlet sigma.
- **Result provenance.** Every engine result carries a `RunProvenance` record — version,
  backend, device, seed, cutoffs, multiple-scattering model, resolved substep fraction and
  the cross-section citation — so an archived dose still says what produced it.
- **Public API.** Everything reachable from the top-level `pyradmc` namespace is supported
  and versioned; re-exports are lazy, so `import pyradmc` pulls in no third-party module
  and a core-only install does not fail on the names of optional backends. The package
  ships a PEP 561 `py.typed` marker.
- **Cross-section library integrity.** `python -m pyradmc.data.tabulated.build` verifies
  every downloaded or cached EPICS library against a pinned SHA-256 before compiling
  tables, and refuses to proceed on a mismatch.

### Known limitations

See `AGENTS.md` section 8 for the full list. The two most likely to bite:

- Under correlated sampling — the default — Dij column sigmas are **not** independent and
  must never be combined in quadrature to form a plan-dose sigma. `variance_csc()` exports
  per-column variance and warns.
- The depth-dose validation gate is 5 %/3 mm, not 2 %/2 mm. The limiter is the reference
  benchmark's missing geometry and transport metadata, not the cross-section data.

[Unreleased]: https://github.com/e0404/pyRadMC/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/e0404/pyRadMC/releases/tag/v0.2.0
