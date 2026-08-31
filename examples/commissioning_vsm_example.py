# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
#
# ruff: noqa: D100
# (jupytext "percent" notebook; also an ordinary script --
# `python examples/halcyon_commissioning_demo.py` runs it top to bottom.)

# %% [markdown]
# # Commissioning a virtual source model against beam data
#
# A full commissioning session: build a head, transport it, and produce the data sets you
# would put beside a water-tank scan. It is written to be read as a worked example of
# *how* a source model is commissioned, not as a model of one particular machine.
#
# **The geometry is a jawless dual-layer head, built from published data about the Varian
# Halcyon**: no jaws at all, and two stacked, staggered multi-leaf collimators doing all
# of the collimation in both directions. Leaf ends define the x (inplane) edge; leaf
# *sides* define the y (crossplane) edge, which is therefore quantized to leaf
# boundaries. That machine was chosen because it makes the collimator do everything, so
# nothing important hides behind a jaw — but every number describing it is inventoried
# below as stated, derived, fitted, bounded or invented, and several are surrogates.
#
# **The source comes one of two ways**, and switching between them is the point of the
# example:
#
# * the **analytic model built here** — a spectrum, a focal spot, a radial fluence, an
#   extra-focal term and a contaminant-electron term, each fitted or stated in the cell
#   below. This is the default and needs no external data.
# * an **IAEA phase space**, set through `PHASESPACE`. Then the whole source model arrives
#   already sampled into the particles and this notebook supplies only the collimator —
#   which is the interesting comparison, because a virtual source model fitted through
#   some other code's simplified collimator can then be pushed through an explicit one.
#
# Either way it produces the four data sets a comparison against measured beam data needs:
#
# * **Percentage depth doses** at SSD 90 cm for every square field.
# * **Lateral profiles** in both principal directions at 1.3, 5, 10, 20 and 30 cm depth,
#   SSD 90 cm, for every square field. The *leaf-travel* direction is `profile_x`.
# * **Diagonal profiles** of the largest field at the same depths and SSD.
# * **Output factors** at SSD 95 cm (point at the isocentre, 5 cm deep) over the full
#   **x-by-y rectangular field matrix**. Neither route models **monitor backscatter** —
#   a field-size-dependent scaling of the monitor response, i.e. a per-beam MU factor
#   rather than a property of any particle — so an output factor here is a dose ratio and
#   a machine's would carry that factor on top.
#
# ## Read this before quoting a number: what is measured and what is guessed
#
# The geometry of this machine is not published in the detail a Monte Carlo model needs.
# The table below is the honest inventory. The second block began as placeholders; each
# row now records what has since been fitted, bounded, pinned or excluded against the
# vendor beam data, and by which measurement — being the thing those rows are refit
# against is the point of this notebook.
#
# | Quantity | Value | Source |
# |---|---|---|
# | SAD | 100 cm | stated |
# | leaf height (each layer) | 7.7 cm | stated |
# | leaf end radius | 23.4 cm | stated |
# | leaf pitch, projected | 1.0 cm per layer, 0.5 cm effective | stated |
# | leaf pairs | 29 proximal, 28 distal | stated |
# | maximum field | 28 x 28 cm^2 | stated |
# | proximal layer top | 68.8 cm SID | literature surrogate (pre-MLC phase space at 69.4 cm) |
# | distal layer top | 60.8 cm SID | literature surrogate (3 mm interlayer gap) |
# | leaf material | W95/Ni3.5/Cu1.5, 18.0 g/cm^3 | published heavy-alloy surrogate |
# | | | |
# | spectrum | 6 MV **FFF** | see `BEAM` — *derived, not fitted* |
# | off-axis cone | Gaussian, sigma 19.5 cm | *invented*; `PRIMARY_FLUENCE_FILE` takes a fit |
# | spot sigma, leaf travel | 0.5 mm | guessed, now **held**: crossline penumbrae match ~0.25 mm |
# | spot sigma, leaf sides | 0.9 mm | **fitted**, one number to one statistic; see `SPOT_SIGMA_V` |
# | leaf-position offset | 0 | tangent + solved radiation-edge gives widths to 0.1 mm, unfitted |
# | head-scatter weight | 0.03 | **bounded, not fitted**: four runs give 0.037 +/- 0.010 |
# | leaf sides | single-focused, flat | *assumed*; a fixed tongue-and-groove loss is excluded |
# | off-axis softening | 4 shells to dc2 -0.62 | **fitted** to depth trends; `SOFTENING_SHELLS` |
#
# ### The stagger falls out of the leaf counts
#
# 29 leaves of 1 cm span -14.5 to +14.5 and 28 span -14 to +14, so the two layers'
# strip boundaries are offset by exactly half a pitch and a boundary of *one* layer
# lands every 0.5 cm. The aperture is the **intersection** of the two layers, so a y
# edge on a half-integer is set by the proximal bank and one on an integer by the
# distal bank, and the effective resolution is 0.5 cm with 1 cm leaves. That is the
# whole design, and it is reproduced here by construction rather than modelled: the
# leaf-open rule is the same "does this strip meet the field" test in both layers.
#
# Read that as a statement about *edges*, not about field sizes. A **symmetric** field
# puts both edges on the lattice at once and therefore steps in 1.0 cm, not 0.5 — a
# symmetric 3.5 cm field does not exist and opens to 4.0. Every y size swept below was
# chosen to be exactly realizable; the aperture figure near the end shows what happens to
# one that is not.
#
# ### There are no jaws, so the leaves also make the y penumbra
#
# In the original notebook a jaw pair defined the y edge and the leaf bank was opened
# past it deliberately. Here the leaf sides *are* that edge. Two consequences worth
# watching in the output: the crossplane field width steps in 1 cm for a symmetric field,
# and the crossplane penumbra is a leaf-side penumbra (two of them, at two different
# source distances) rather than a jaw penumbra.
#
# There *is* fixed tungsten above the leaves in this model, sized to the source fan rather
# than to the field — see `backup_collimator`. It stands in for the machine's own fixed
# secondary collimator, which is what closes the head beyond the edge of the banks; the
# always-closed outer leaf pairs are carried explicitly. It never comes within a centimetre
# of a field edge, so nothing reported here is shaped by it.
#
# ### The two layers sharpen each other
#
# Both banks are set to the same isocentre projection, so near an edge the transmitted
# fluence is the *product* of two partial-transmission profiles and falls off faster
# than either alone. Whether that reproduces the measured 3.8 mm (2 x 2) and 5.0 mm
# (10 x 10) penumbrae is one of the things this notebook is for; `SPOT_SIGMA` and the
# extra-focal weight are the two knobs, and they must be fitted jointly (see the
# original notebook's discussion, which applies unchanged).
#
# ### Everything the original notebook warns about still applies
#
# The penumbra is grid-limited at `SPACING = 0.25` and is good for comparing fields
# with each other, not as an absolute; the device pre-solve appends with atomics so a
# fixed `SEED` does not pin the last digit; and every error bar comes from the spread of
# `REPLICATES` independent runs, head model included.

# %% [markdown]
# ## Parameters
#
# The whole configuration. Lengths in cm, energies in MeV.

# %%
# --- the beam ---------------------------------------------------------------------
# Halcyon-class heads are flattening-filter-free, and `ALI_ROGERS_BEAMS` holds only
# *flattened* fits, so the spectrum is derived rather than looked up: take the Varian
# 6 MV fit and remove the flattening filter from its filtration term. C2^2 is an
# effective aluminium thickness in g/cm^2; the fitted flattened beam carries 26.5, a
# 6 MV filter is of order 16 g/cm^2 (roughly 1.8 cm of copper on axis), which leaves
# 10.5 and so C2 = 3.24. Independently, published mean energies put 6X-FFF at about
# 0.77 of the flattened beam; this fit's flattened mean is 1.601 MeV, so the target is
# about 1.23 MeV, and C2 = 3.25 gives 1.229. The two routes agreeing is the reason to
# start here -- it is still a derivation, not a fit.
#
# **This is the first thing to refit against the measured depth doses.** C2 is a
# single monotone knob on beam quality: raising it hardens the beam and lifts PDD(10).
from pyradmc.geometry.spectrum import ALI_ROGERS_BEAMS, AliRogersMV

_FLATTENED = ALI_ROGERS_BEAMS["varian-6mv"]
BEAM = AliRogersMV(
    e_e=_FLATTENED.e_e,  # 5.76 MeV endpoint; the target is unchanged by removing a filter
    c1=_FLATTENED.c1,  # tungsten filtration: the target itself, also unchanged
    c2=3.25,  # aluminium-equivalent filtration with the flattening filter removed
    c3=_FLATTENED.c3,
    c4=_FLATTENED.c4,
)
# --- driving the head from a phase space instead of the model below --------------------
# Leave `PHASESPACE` None and the source is the analytic model built further down: a
# spectrum, a focal spot, a radial fluence, an extra-focal term and a contaminant-electron
# term, each fitted or stated here. Name an IAEA pair instead (either extension, or the
# bare stem) and all of that arrives in the particles, already sampled; the only thing this
# notebook still supplies is the collimator. That is the interesting comparison: a source
# model fitted through some other code's collimator, pushed through an explicit one.
#
# `HALCYON_VSM_PHASESPACE` in the environment is read when this is None, so the example can
# be pointed at a file without being edited.
PHASESPACE: str | None = None
PHASESPACE_PLANE_Z = 31.0  # cm from the target: where the records sit, checked against them
CLAMP_RADIUS = 20.0  # cm at the isocentre; the aiming boundary baked into the file
# Electron histories per replicate, as a fraction of the photon count, for the electrons
# carried by the file. They are a fraction of a per cent of its records and at most a few
# per cent of the dose, and they take the deterministic attenuation route (which re-traces
# every leaf pair on the host), so oversampling them is what makes a sweep CPU-bound.
ELECTRON_HISTORY_FRACTION = 0.005
# The spectrum the file's own documentation claims, used only to check the file against it
# before transporting anything -- see the guard below. None skips the check.
VSM_BEAM: AliRogersMV | None = AliRogersMV(e_e=5.76, c1=1.66276, c2=3.25, c3=-1.186, c4=0.0)
VSM_GRID = (0.25, 60)  # (e_min, n_bins) of that stated spectrum; both ends are EDGES
SPOT_SIGMA = 0.05  # focal spot sigma (0.5 mm); this is what makes the geometric penumbra.
#                    Carried over from the other notebook, where it was fitted against a
#                    different machine's penumbra. Refit it here jointly with
#                    EXTRAFOCAL_WEIGHT against the measured 3.8 mm (2 x 2) and 5.0 mm
#                    (10 x 10) penumbrae -- the two are not separable, see the original.
SPOT_SIGMA_V: float | None = 0.09  # along leaf sides; None keeps the spot circular.
# Separate because the measured output factors say the two axes are not the same. The
# vendor matrix is not symmetric under transposition: Scp(1 x Y) exceeds Scp(Y x 1) by
# 3.17 %, flat to 0.08 % across every Y from 2 to 28 cm, while every pair with both sides
# at 2 cm or more is symmetric to the sheet's own rounding. Two mechanisms can put an
# asymmetry on one axis; they differ in how it dies with field size, and the matrix
# separates them without any fitting.
#
# A tongue-and-groove step on the leaf sides is a *fixed* loss of aperture, so the offset
# it implies must be the same on every row. Measured by finite difference on the sheet's
# own grid it implies 3.06 mm at iso on the 10 mm row and 0.11 mm on the 20 mm row, a
# factor of 28 apart. It is not a fixed aperture loss.
#
# Source occlusion *saturates*: the on-axis primary through a gap w is erf(w / (2 sqrt2
# sigma)), which is 3.17 % short at 10 mm and 0.00 % short at 20 mm for the same sigma.
# Calibrating on the 10 mm row leaves nothing free, and the prediction lands -- 0.00 %
# against a measured 0.03 % at 20 mm and -0.06 % at 40 mm.
#
# The erf argument is a one-dimensional model of a point measurement in a scattering
# phantom behind two staggered banks, so it sets the direction, not the value. Measured in
# transport over all nine transposed pairs the response is convex: the nine-pair mean
# reads 1.0026 +/- 0.0050 at sigma_v = 0.05 (circular), 1.0173 +/- 0.0048 at 0.08 and
# 1.0435 +/- 0.0046 at 0.10 -- 0.49 ratio-points per 0.01 cm on the first interval, 1.31
# on the second -- against the machine's 1.0319. **The shipped 0.09 is the interpolation
# of that bracket, a declared one-number fit** (statistical error about 0.005, and the
# calibrating measurement itself carries the vendor-processing caveat above). Every
# output-factor matrix re-measures the nine pairs, so each full run re-verifies it.
#
# Calibrate on the *nine* pairs, never on one. The 28 cm pair alone reads 1.0317 where the
# nine-pair mean is 1.0026, and a sweep run against it gave 0.99 points per 0.01 cm at the
# working point -- twice the true response there. One pair's noise is worth 0.01 to
# 0.02 cm of sigma_v.
#
# Read that as "the mechanism saturates like occlusion", not as proof of an elliptical
# target, because **nothing in this data set can check it**: the vendor supplies crossline
# profiles only, so the leaf-side penumbra that would constrain it was never measured.
# This is one parameter fitted to one number. `SPOT_SIGMA` itself is *not* free the same
# way -- the crossline penumbra pins it to 0.25 mm once the small-field scans are compared
# as point dose, which is why the residual +1.94 % common to both 1 cm axes cannot be a
# spot and is left open rather than absorbed here.

# --- the off-axis primary fluence ---------------------------------------------------
# An FFF head emits the unflattened bremsstrahlung cone: peaked on axis and falling
# monotonically, with no horn and no filter edge. Two shapes are available here.
#
# The analytic one is radially symmetric,
#
#     psi(r) = exp(-(r / CONE_SIGMA)^2)   for r <= PRIMARY_COLLIMATOR_RADIUS, else 0,
#
# with the radius quoted at the isocentre. CONE_SIGMA = 19.5 cm puts psi(14 cm) at 0.60,
# which is where a 6 MV FFF in-air off-axis ratio sits at the corner of the largest
# field. **It is invented**, and it is what this example used until the shape below
# replaced it.
#
# `halcyon_primary_fluence.dat` is that replacement, and it is **fitted, not derived**:
# a first-order correction of the analytic cone against the vendor's own open-field
# crossline and diagonal profiles, produced by `local_fit_halcyon_fluence.py` (not
# shipped -- it needs the vendor workbook). The table's own header states its
# provenance and its three approximations. Setting this back to None restores the
# analytic cone and makes the example self-contained again, at the cost of about
# 3 per cent in the off-axis ratio at mid radius and considerably more at the corner.
#
# **The fitted table does not ship** (maintainer decision, 2026-08-11): it is derived
# from the vendor workbook, so it stays local alongside the workbook and the fit script,
# and the shipped default is the analytic cone. Fitting your own table against your own
# beam data and pointing this at it is expected — it is worth about 3 per cent in the
# off-axis ratio at mid radius and considerably more at the 28 x 28 corner, and every
# number quoted in this notebook's tables that depends on the off-axis shape improves
# accordingly (the fitted-table results are recorded in the CHANGELOG).
#
# The spectrum's own radial dependence is `SOFTENING_SHELLS` below. An earlier note here
# argued an FFF beam barely softens off axis and that the target's angular *hardening*
# was "the opposite sign and smaller"; the vendor data refuted both claims -- see the
# shell comment.
#
# **The fluence is radial on measurement, not assumption** (2026-08-11). A non-radial
# "fixed-collimator envelope" C(x,y) was the one degree of freedom the literature
# suggests adding if diagonals demand it, and the measured data does show the diagonal
# reading above the crossline at equal radius (+0.2 to +1.0 per cent over r = 7-11 cm).
# But that differential is *flat in measurement-plane coordinates and grows with depth
# at fixed fan angle* -- the opposite of a fan-fixed fluence structure, which would
# scale with divergence and dilute with depth -- and the simulation, whose fluence is
# radial by construction, already reproduces it (4.0 vs 4.3 per cent at the strongest
# bin): it is phantom scatter asymmetry near the crossline's field edge, transported
# for free in both channels. The model-minus-measurement azimuthal residual is not
# coherent above its ~1 per cent noise floor, so there is no C(x,y) to fit. The
# diagnostic is worth keeping: plane-coordinate flatness plus depth growth reads
# "scatter", fan-coordinate constancy would read "fluence".
PRIMARY_FLUENCE_FILE: str | None = None  # "radius_mm value", at iso; see the note below
CONE_SIGMA = 19.5  # cm at the isocentre; only used when PRIMARY_FLUENCE_FILE is None
PRIMARY_COLLIMATOR_RADIUS = 21.0  # cm at iso; must clear the 28 x 28 corner at 19.8 cm

# --- off-axis softening ----------------------------------------------------------------
# The primary's spectrum softens with fan radius: each shell below is the axis beam with
# only its aluminium-equivalent filtration `c2` changed by `dc2`, and a photon gets the
# shell its isocentre-plane radius lands in. **The dc2 values are fitted, not derived**
# (2026-08-11), the same standing as the fluence table: normalizing every measured and
# simulated profile to its own axis cancels everything depth-independent, and what is
# left -- the in-field residual's *trend across 5 to 30 cm depth* -- is a spectral
# signature. Fitted per 1 cm radius bin over the 20 x 20 and 28 x 28 crosslines and the
# diagonals, the trend reads as dc2 falling from ~0 near the axis to about -0.6 by
# r = 14 cm (mean energy -2 to -7 per cent vs axis, the expected order for an
# unflattened beam's target-driven softening). A single quadratic in r is *rejected*
# (chi2/dof ~ 5 even after doubling the errors for correlated MC noise), so this ships
# as a stepwise table rather than a law, and the first-principles reading stays a
# hypothesis.
#
# The values below are the *second* pass, and the second is the last. The first pass
# applied the kernel-implied dc2 directly and removed ~70 per cent of the trend (transport
# responds at ~0.7 of the narrow-beam kernel -- phantom scatter diluting spectral
# contrast, the kernel's one stated bias), so the residual was folded back in at that
# response. The second pass then measured residuals of +0.09/+0.12/-0.10/-0.15 -- but its
# own control says stop: the innermost softened shell, whose value did not change between
# passes, drifted -0.02 -> +0.09, so **the residual estimator's floor is about +/-0.1 in
# dc2** (pre-solve nondeterminism plus shared-curve MC noise; the per-bin errors are
# optimistic). Every remaining residual is within ~1.5 sigma of that floor, the signs do
# not cohere, and iterating further would fit one realization of noise -- the same
# mistake the extra-focal weight saga above records. What the shells leave behind is a
# depth trend of at most ~0.7 per cent across 5-30 cm, against -1 to -4 before.
#
# Three stated approximations. The reference shell reaches to 2 cm fan radius so the
# on-axis PDD column (1.5 cm physical, up to ~1.6 cm fan at this SSD) never sees a
# softened spectrum -- the axis beam stays exactly the derived `BEAM`. The extra-focal
# source keeps the axis spectrum everywhere (a ~3 per cent component whose true spectrum
# is unknown anyway). And the deterministic tip traces (`radiation_edge_correction`)
# also keep the axis spectrum; at 14 cm off axis the softer shell would move the traced
# 50 per cent edge by well under the 0.1 mm the widths are quoted to.
#
# The energy redraw preserves each history's *weight*, i.e. photon fluence, so softening
# lowers the off-axis energy fluence on top of the fluence table's own shape. The two are
# therefore refit jointly: this table was fitted with the shells off, and the fluence
# refit that follows it absorbs the depth-average the shells introduce.
SOFTENING_SHELLS: tuple[tuple[float, float], ...] | None = (
    (2.0, 0.0),  # fan radius at iso (cm, outer edge), dc2 -- the axis reference shell
    (6.0, -0.17),
    (10.0, -0.51),
    (14.0, -0.80),
    (float("inf"), -1.23),
)

# --- the extra-focal (head-scatter) source ---------------------------------------------
# Same wide-Gaussian second source as the flattened notebook, and the same fitted-weight
# story -- but a much smaller effect, because the dominant scatterer in a flattened head
# is the flattening filter and there is not one. What is left is the primary collimator,
# the monitor chamber and the target assembly, so the source plane moves up and the
# weight comes down.
#
# `w` does not need a re-run to refit: the output-factor CSV carries each component's
# dose per unit on-axis emitted fluence separately, so
#
#     Scp(w) = [(1-w) P(field) + w E(field)] / [(1-w) P(ref) + w E(ref)]
#
# can be fitted against the measured matrix directly from the file. Set it to None only
# if you want the primary-only beam; that also loses the E column and with it the
# ability to refit.
#
# Fit `w` against the **whole matrix at once**, not entry by entry. `E` is by far the
# noisiest column here — the extra-focal fan is spread over thousands of square
# centimetres at the isocentre, so only a small fraction of its histories reach any
# aperture (which is what `extrafocal_histories` exists to compensate) — and individual
# entries at the shipped statistics are good to tens of per cent, while the aggregate that
# a least-squares fit uses is not. `E` contributes only a few per cent of the dose, so
# this costs the fit far less than it looks.
#
# **Fit the trend, not the chi-square**, if you refit it. What a wrong `w` produces is a
# *slope* of the residual against field size, not an offset, because E/P is not constant:
# it runs from about 1 per cent at 2 x 2 to 126 per cent at 28 x 28, so `w` reweights
# small and large fields differently rather than scaling them together. The chi-square is
# nearly flat over w = 0.02 to 0.05 because the per-entry scatter is Monte Carlo noise
# rather than model error (chi2/n about 1.9, i.e. the scatter already exceeds the quoted
# errors), so least squares barely discriminates while the slope does.
#
# **`w` cannot be fitted against this matrix at these statistics, so it is left at its
# placeholder.** Four runs of this configuration put the slope's zero crossing at 0.049,
# 0.042, 0.029 and 0.027 -- w = 0.037 +/- 0.010 run to run, which is a 1.8x spread on the
# quantity being fitted. The runs differ only in `w` itself, which is applied after both
# components are scored, so P and E are the same measurement four times; three share a
# seed and one does not, and that makes no difference to the spread.
#
# The reason is that **the residual's slope is itself not reproducible here**. At the
# shipped w = 0.03 it reads -2.12, -1.64 and +0.12 per cent across 2 to 28 cm on three
# runs, against a formal error of +/- 0.5. So the field-size trend in the output factors
# is not a model error to chase: it is noise, and any `w` fitted to remove it is fitted to
# one realization of that noise. Confirmed the expensive way -- raising `w` to the 0.047
# one run preferred moved the next run's trend to +2.0 per cent.
#
# Do not read 0.03 as fitted. What can be said is 0.037 +/- 0.010, which contains it.
#
# Two things about the *analysis* came out of that and are worth not repeating:
#
# - **Put a floor under the per-entry sigma before weighting by it.** It is the spread of
#   `OUTPUT_FACTOR_REPLICATES` replicates, so with six of them the estimate itself carries
#   about 1/sqrt(2(n-1)) = 32 per cent error; across 81 entries several come out
#   spuriously small and 1/sigma^2 weighting then leans on exactly those. One run's
#   chi2/n went from 4.40 to 1.90 on flooring sigma at its median, and its fitted crossing
#   moved by 0.007 -- the same size as the quantity being fitted.
# - **The extra-focal column is not the limit, despite reproducing far worse.** E moves by
#   55 to 66 per cent run to run against the primary's 1.9 per cent (which is exactly its
#   quoted sigma), but E carries only about 3 per cent of the dose, so it injects about
#   0.5 per cent into an output factor against the primary's 1.6 per cent.
#
# The likely cause of the run-to-run scatter is the head pre-solve rather than the history
# count: `presolve_head_device` is order-nondeterministic (atomic append reorders the exit
# phase space), each replicate reuses one pre-solve up to `PHASE_SPACE_REUSE` times, and
# there are only six replicates -- so an entry rests on six pre-solve draws however many
# histories are transported through them. If this needs to be pinned down, raise
# `OUTPUT_FACTOR_REPLICATES` at fixed total histories before raising the histories.
#
# If you do refit, smooth E first. It is the share of a Gaussian source that a rectangular
# aperture sees, so it is separable and saturating -- `A g(x) g(y)` with
# `g(s) = 1 - exp(-(s/s0)^p)` fits all 81 entries with s0 about 7 cm and p about 1.1, and
# those parameters reproduce between runs to 5 per cent where the raw column does not.
#
# `EXTRAFOCAL_SIGMA` was swept against the same matrix (0.75 to 12 cm, weight refitted at
# each) and is **not identifiable from this data**: with `w` free, every size leaves the
# same residual to within its error. A reseed of the extra-focal arm alone moved the
# fitted trend by up to 2 percentage points, the size of the effect. It stays at its
# placeholder; do not read it as fitted.
EXTRAFOCAL_WEIGHT: float | None = 0.03  # on-axis emitted fluence fraction; 0.037 +/- 0.010
EXTRAFOCAL_Z = 12.0  # source plane, cm from the target: the primary collimator region
EXTRAFOCAL_SIGMA = 1.5  # Gaussian sigma there (cm); swept, not identifiable, placeholder
EXTRAFOCAL_FIELD = 44.0  # emitting square at the isocentre, cm: field-independent, and
#                          wide enough to illuminate the 28 x 28 corners
# Histories per replicate relative to the primary's, *per unit dilution ratio* -- see
# `extrafocal_histories`, which is where the failure this fixes is recorded. 0.005 puts a
# 10 x 10 field near the flat 0.2 the flattened notebook used; the cap stops a 1 x 1 field
# from spending three times its primary count on a few per cent of its dose.
EXTRAFOCAL_HISTORY_SCALE = 0.005
EXTRAFOCAL_HISTORY_MAX = 1.0

# --- the contaminant-electron source ---------------------------------------------------
# The third linear component: electrons reaching the phantom with the beam. The evidence
# is the vendor's own build-up columns, which nothing above uses -- inverting
# M = (S + e)/max(S + e) per field leaves an implied component e(d) that is electron-like
# on three independent counts: it is spent by ~1.2 cm depth with no deep tail (practical
# range of ~2 MeV electrons), it is 1-2 % of the photon peak at 0.5 cm for small fields
# rising to 4.4 % at 28 x 28 (the published range for 6 MV FFF contamination), and its
# field-size scaling splits into a *floor* present already at 2 x 2 plus a *saturating*
# rise -- the signature of two populations. Hence two sources, both mono-energetic at
# `CONTAMINANT_ENERGY` and both re-badged photons-to-electrons over existing geometry:
#
#   beam: emitted from the primary's own fan rectangle, aimed from the focal spot --
#         electrons travelling with the beam, aperture-proportional, the floor term.
#   head: emitted from the extra-focal source's wide plane at `EXTRAFOCAL_Z` --
#         electrons born around the primary collimator, progressively uncovered by the
#         aperture, the saturating term. Reuses that source's geometry deliberately:
#         one fewer invented knob, and the same physical region of the head.
#
# Both are clipped by the collimator through the attenuation route, never the photon
# pre-solve. Stated approximations: the clip weights by *photon* attenuation, so a
# millimetre of grazing tungsten transmits ~90 per cent where an electron would stop --
# wrong at the aperture edge, irrelevant to the bulk build-up dose this exists for; air
# scatter between the head and the phantom's transported air column is not modelled and
# the fitted energy partly stands in for it; and one mono-energy serves every field,
# which the derivation's marginal collapse test (large fields reach slightly deeper)
# already strains -- if the fit fails, it fails there first.
#
# The head also carries a carbon-fibre bore cover (~50 cm SID) that this model omits
# entirely, and the literature's ordering rule -- model the cover before fitting
# electron contamination, or the fitted source absorbs the cover's own electrons -- is
# knowingly violated (maintainer decision, 2026-08-11). The fitted head term is
# therefore an *effective* source that includes whatever the cover generates; the
# cover's photon attenuation is field-independent and vanishes into the per-fluence
# normalization. This beam data cannot separate the two (both are shallow, aperture-
# scaled electron fluxes), so an explicit cover would add an invented thickness and a
# forced refit with nothing to constrain them. Revisit only with surface-dose or
# electron-range data that can actually see the cover.
#
# The weights combine linearly on top of the photon components, so like the extra-focal
# weight they are refittable from the saved component columns without re-running; only
# the energy needs a re-run to change.
CONTAMINANT_ENERGY = 2.2  # MeV; fitted on a 1.5/2.2/3.0 grid, 2.2 wins on depth shape
CONTAMINANT_WEIGHTS: tuple[float, float] | None = (0.0, 0.00043)  # (beam, head) on-axis
#                     emitted electron fluence per unit on-axis photon fluence; None = off.
#
# **The beam term was built, fitted, and came out zero** -- keep it zero. Two independent
# reasons, both measured. Its basis is not the flat floor the two-population argument
# assumed: per unit weight it contributes 4.3x more at 8 x 8 than at 2 x 2 (the clip
# survival and the per-history normalization do not cancel the way the back-of-envelope
# said), so any weight large enough to matter at 2 x 2 wrecks the large fields. And the
# residual it was invented to carry *decreases* with field size (+1.5 % of peak at
# 2 x 2, +0.7 at 4 x 4, ~0 by 8 x 8 at 0.5 cm depth), which no contamination mechanism
# of any geometry does -- electrons only grow with aperture. That leftover is a
# small-field shallow-depth deficit of the model or the measurement (lateral electron
# disequilibrium, or the chamber at its build-up worst), and it stays on the open list
# rather than being laundered through a fitted source.
CONTAMINANT_HISTORY_SCALE = 0.003  # per unit dilution ratio, like the extra-focal rule
CONTAMINANT_HISTORY_MAX = 0.02  # electrons deposit locally: a build-up column smooth to
#                                 ~1 % costs a few per cent of the photon count, not more

# --- the treatment head -------------------------------------------------------------
# Device slabs in the beam frame: cm downstream of the focal spot. The frame origin is
# the focal spot, beam-frame w == engine z, and the isocentre sits at (0, 0, SAD).
# Machine data is quoted as SID (cm from the isocentre), so it is converted below.
SAD = 100.0
LEAF_HEIGHT = 7.7  # stated, both layers
MLC_PROXIMAL_TOP_SID = 68.8  # upstream face of the upper bank, cm from iso. A published
#                              model of this head scored a phase space at 69.4 cm, after
#                              the secondary collimator and before the leaves, so the
#                              first tungsten cannot be far below that. 68.8-61.1 is the
#                              literature-informed surrogate (2026-08-11, replacing this
#                              notebook's earlier 69.0 estimate); still a surrogate, not
#                              vendor geometry.
MLC_DISTAL_TOP_SID = 60.8  # upstream face of the lower bank: the same surrogate's
#                            60.8-53.1, a 3 mm interlayer gap where the earlier estimate
#                            guessed ~1.3 cm of interlayer shielding (60.0). Both banks
#                            still clear the 50 cm bore comfortably. The distal bank
#                            moved 8 mm upstream, which slightly changes each tip's
#                            tangent retraction and the leaf-side occlusion response --
#                            the traced widths and the nine-pair sigma_v read are the
#                            re-verification, and every full matrix run repeats it.
MLC_PROXIMAL_PAIRS = 29  # stated. Odd -> a leaf is centred on the axis.
MLC_DISTAL_PAIRS = 28  # stated. Even -> a strip *boundary* is on the axis.
MLC_PITCH = 1.0  # leaf width projected to the isocentre, per layer. 29 and 28 leaves of
#                  this pitch are offset by exactly half of it, which is the stagger:
#                  the effective resolution is 0.5 cm. Nothing else imposes it.
MLC_TIP_RADIUS = 23.4  # stated. Large next to the 7.7 cm leaf height, so the rounded end
#                        is a shallow arc and its transmission tail is long.
# The leaf-position calibration, cm at the isocentre added to each tip's retraction.
# Two parts, and only the second is free.
#
# The geometric part is the tangent offset: a rounded end does not put its edge under the
# tip's nominal position, the edge follows the ray from the focal spot that grazes the
# arc. It is computed per layer from that layer's own mid-plane (see `tangent_offset`),
# so the two banks get slightly different retractions for the same field -- with the
# spectral radiation-edge solve included, 0.76 mm proximal against 0.62 mm distal at a
# 10 x 10, the nearer bank retracting more -- and nothing about it is fitted. (An
# earlier revision of this comment quoted 0.67/0.70 mm; re-measured 2026-08-11, those
# numbers matched neither the values nor the ordering of the shipped functions.)
#
# The free part is one constant: the light-field/radiation-field offset every vendor
# quotes and calibrates out. It is **unknown for this machine and left at zero**, which
# is why the simulated field widths below are an independent check rather than a
# tautology. If they come out uniformly long or short, this is the number that absorbs
# it; if the residual has a *slope* in field size, the tangent geometry above is what is
# wrong and a table is needed instead (the flattened notebook ships one).
MLC_TIP_OFFSET = 0.0
MLC_FOCUSED_SIDES = True  # trapezoid leaf cross-section focused on the spot. Assumed:
#                           with parallel sides a strip shadows a field from ~12 % inside
#                           its edge, which for a jawless head would eat the crossplane
#                           field width outright. Real single-focusing designs are built
#                           this way. Set False to see what it would cost.
CLOSED_PAIR_PARK = 1.0  # cm at iso by which a closed pair's abutment is parked outside
#                         the field, in *opposite* directions in the two layers, so the
#                         two leaf-end gaps never line up. On a real dual-layer head that
#                         non-alignment is a design feature; here it is the only thing
#                         standing in for it, since interleaf gaps are not modelled.
EMISSION_Z = 20.0  # the source plane, above the first device
# The fixed shielding that closes the head outside the source fan -- see
# `backup_collimator`, which is where this is justified and where its one stated
# approximation is recorded. Two slabs, one per axis, between the source plane and the
# leaves; it is set to the fan, never to the field, and defines no field edge.
BACKUP_U_Z = (20.25, 25.5)
BACKUP_V_Z = (25.5, 30.75)
# How far past the field the primary fan reaches, per side at the isocentre. Everything
# aimed outside is absorbed in tungsten, so a wide margin is mostly wasted histories at
# small fields and necessary at large ones; the ramp is the flattened notebook's,
# measured there to be free of in-field bias. The minimum is 1.0 rather than that
# notebook's 0.75 because the fan boundary is real tungsten here: it has to clear the
# smallest field's penumbra with room to spare, and 0.75 does not.
FAN_MARGIN_MAX = 3.0
FAN_MARGIN_MIN = 1.0
FAN_MARGIN_SLOPE = 0.25

HEAD = "presolve"  # "presolve" (collimator scatter + contaminant electrons) or
#                    "attenuation" (deterministic Beer-Lambert weights, far slower here:
#                    it re-traces 60-odd leaf pairs on the host for every chunk)
DEVICES = "tungsten"  # "tungsten" needs the EPICS tables; "water" is the stand-in

# --- the measurement setups -------------------------------------------------------------
# Two of them, because the measured data uses two.
PROFILE_SSD = 90.0  # depth doses, lateral profiles, diagonals
PROFILE_DEPTHS = (1.3, 5.0, 10.0, 20.0, 30.0)  # cm; profiles are interpolated in depth
OUTPUT_FACTOR_SSD = 95.0  # output factors; with the depth below the point is at the iso
OUTPUT_FACTOR_DEPTH = SAD - OUTPUT_FACTOR_SSD

PHANTOM_DEPTH = 45.0  # deep enough that the 30 cm plane still has full backscatter
LATERAL_MARGIN = 15.0  # water beyond the field at the deepest plane. 15 cm carries the
#                        scatter that reaches the axis and the diagonal sampling below.
AIR_GAP = 25  # centimetres of *transported* air above the surface; the rest of the
#               modelled air path is the pre-solve's analytic column, and the total is
#               held the same in every setup so the build-up regions stay comparable
SPACING = 0.25  # voxel edge, all three axes
# The output-factor runs are read for a single on-axis number, so they transport in a
# smaller phantom and score only a box around the point (a `ScoringGrid`). Nothing about
# the dose at that point changes: the transport grid still carries the scatter volume
# that matters, and the deposits outside the box are simply not accumulated.
OUTPUT_FACTOR_PHANTOM_DEPTH = 25.0
OUTPUT_FACTOR_LATERAL_MARGIN = 12.0
OUTPUT_FACTOR_SCORING_HALF = 2.0  # half-width of the scored box, cm

# --- the field sets ------------------------------------------------------------------
PROFILE_FIELDS_MM = (20, 40, 60, 80, 100, 200, 280)  # square, at the isocentre
DIAGONAL_FIELD_MM = 280  # diagonals are only asked for at the largest field
# The output-factor matrix: every x against every y. 10 x 10 = 100 fields, which is the
# bulk of this notebook's wall clock; trim either tuple to cut it.
OUTPUT_FACTOR_SIDES_MM = (10, 20, 40, 60, 80, 100, 140, 200, 240, 280)
REFERENCE_FIELD_MM = 100  # the (square) field output factors are normalized to

# --- statistics -------------------------------------------------------------------------
# **This is the expensive cell, and the output-factor matrix is where the time goes.**
# Extrapolated from measured field timings on an RTX 4070 laptop: the seven-field profile
# sweep runs about 16 minutes and the 100-field matrix about 50, so budget a bit over an
# hour. The matrix cost is roughly linear in the number of fields — trimming
# `OUTPUT_FACTOR_SIDES_MM` is the first lever, and raising `OUTPUT_FACTOR_HISTORIES` the
# second, at 1/sqrt(N).
#
# What that buys, extrapolated from the achieved errors at reduced statistics: the on-axis
# error on the matrix lands near **1 to 2 %** across the whole field range, and the run
# prints what it actually achieved. That is enough to see a 3 % model error and not enough
# to argue about a 1 % one; raise the histories fourfold for a final comparison. The depth
# doses and profiles are converged well past that — they are read from column averages
# rather than from a single measuring volume.
#
# Equal histories per field do **not** buy equal precision, and over a 1 to 28 cm range
# that is not a detail. The flattened notebook levelled it with a measured per-field sigma
# table; there is no such table for this geometry, and inventing one would be worse than
# not having it, so the two allocations below are *derived* instead — both from one
# quantity, `dilution`: the source fan spreads a fixed number of histories over an area
# that grows with the field, and the volume they are counted in collapses by eighteen once
# the field is too small to hold a chamber.
#
# Both were checked against a flat allocation before being adopted, and both were needed.
# Flat, at equal histories, the achieved on-axis error ran 0.45 % at 2 x 2 and **9.5 %** at
# 28 x 28 — a factor of 21, and the largest fields unusable. The measuring-volume half of
# `dilution` was added after the fan-area half, when the fields with a 1 cm side were still
# coming out at 5 to 15 %; it costs 1.56x on the matrix and is what makes those 19 entries
# worth reading.
HISTORY_BASE_SCALE = int(1e8)
HISTORIES = (
    5 * HISTORY_BASE_SCALE
)  # per profile field at the reference size, split over the replicates
REPLICATES = 10  # independent runs per field; their spread is every error bar
OUTPUT_FACTOR_HISTORIES = int(1.5 * HISTORY_BASE_SCALE)  # per matrix field at the reference size
OUTPUT_FACTOR_REPLICATES = 6
# (1) Histories in proportion to the fan area, so every field is simulated at the same
# incident fluence per unit area at the isocentre. Over the shipped matrix that costs only
# 1.34x a flat allocation, because most of the fields it starves are small ones that were
# over-resourced. The clamps stop a 1 x 1 field from being allocated 3 % of the reference
# (its curves, not its on-axis value, would fall apart first) and a 28 x 28 from running
# away.
HISTORY_SCALE_FLOOR = 0.20
HISTORY_SCALE_MAX = 6.0
# Every output factor is a ratio against the reference field, so its error lands on all
# hundred at once and cancels nowhere. It gets a boost the others do not.
#
# Raised 3 -> 12 (2026-08-13) after the effect was seen at full size: two regenerations
# whose raw doses agreed to 0.04 +/- 0.2 per cent still differed by 1.3 per cent in
# every output factor at once, because each run's block mean inherits its own
# reference draw (~0.6-0.9 per cent at boost 3, and the quoted per-entry sigma cannot
# warn about a shift it shares). At 12 the common mode drops to ~0.3 per cent for
# about an hour more on a ten-hour matrix -- the block-mean quote was
# reference-limited, nothing else was.
REFERENCE_HISTORY_BOOST = 12.0
# (2) The pre-solved phase space is resampled by the transport, so a small one puts a
# floor under the error that no amount of transport can lift — and the floor is *also* a
# density: a fixed head-history count spread over a larger fan leaves fewer distinct
# records able to reach the measuring volume. Measured directly on the 28 x 28 output
# factor at fixed transport, the on-axis error went 4.76 % at 40x reuse to 3.07 % at 8x
# for 1.46x the wall clock — a 1.6x gain in sigma^2 x time — while the 10 x 10 field, which
# is transport-limited rather than phase-space-limited, barely moved (1.54 % to 1.47 %).
# So the reuse cap is scaled inversely with the fan area, which lands the 28 x 28 field on
# 8 by construction.
PHASE_SPACE_REUSE = 40  # at the reference field; ~40 is the measured knee of sigma^2 x time
PHASE_SPACE_REUSE_MIN = 8  # the measured point above; below it the pre-solve dominates
PRESOLVE_HISTORIES_MIN = 200_000
SEED = 20260803
BACKEND = "auto"  # "auto" prefers Warp (CUDA if present); "ref" forces the oracle
SHOW_PLOTS = False

# %% [markdown]
# ## Imports, and what we are running on

# %%
import csv  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402
import mmap  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, NamedTuple  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pyradmc.backends.ref.engine import ReferenceEngine  # noqa: E402
from pyradmc.data.materials import AIR, TUNGSTEN_ALLOY, WATER  # noqa: E402
from pyradmc.geometry.collimation import (  # noqa: E402
    MLC,
    BeamFrame,
    BeamLimitingStack,
    CollimatedSource,
    JawPair,
    project_between_planes,
)
from pyradmc.geometry.fluence import RadialFluence  # noqa: E402
from pyradmc.geometry.grid import VoxelGrid  # noqa: E402
from pyradmc.geometry.head import AirColumn, presolve_head  # noqa: E402
from pyradmc.geometry.phasespace import (  # noqa: E402
    IAEA_ELECTRON,
    IAEA_PHOTON,
    IAEAHeader,
    InMemoryPhaseSpaceSource,
    _build_struct,
    _decode_record,
    _iaea_path,
    _n_particles,
    _parse_sections,
    _record_dtype,
    _sample_indices,
)
from pyradmc.geometry.source import (  # noqa: E402
    GaussianSpotBeamSource,
    Primary,
    PrimaryFluenceBeamSource,
    Source,
)
from pyradmc.geometry.spectrum import Spectrum, ali_rogers_mv  # noqa: E402
from pyradmc.rng import RNGState, uniform  # noqa: E402
from pyradmc.rng.host import HostRNG  # noqa: E402
from pyradmc.scoring.grid import ScoringGrid  # noqa: E402

try:
    import warp as wp

    from pyradmc.backends.warp.engine import WarpEngine
    from pyradmc.backends.warp.presolve import presolve_head_device

    WARP_DEVICE: str | None = "cuda:0" if wp.is_cuda_available() else "cpu"
except ImportError:  # pragma: no cover - the example degrades to the oracle
    WarpEngine = None  # type: ignore[assignment,misc]
    WARP_DEVICE = None

if BACKEND == "ref":
    WARP_DEVICE = None
print(f"transport on {WARP_DEVICE or 'ref (reference engine)'}")

# %% [markdown]
# ## Resolved configuration

# %%
PROFILE_FIELDS = tuple(mm / 10.0 for mm in PROFILE_FIELDS_MM)
DIAGONAL_FIELD = DIAGONAL_FIELD_MM / 10.0
OUTPUT_FACTOR_SIDES = tuple(mm / 10.0 for mm in OUTPUT_FACTOR_SIDES_MM)
REFERENCE_FIELD = REFERENCE_FIELD_MM / 10.0
# The phase-space route splits the file by particle kind before transport, so its
# photons are kind-pure and can go through the pre-solve like the analytic ones. An
# earlier revision forced "attenuation" here because a mixed-kind source cannot enter
# the photon pre-solve; `fan_clip` removes that constraint.
HEAD_MODEL = HEAD
PER_REPLICATE = max(1, HISTORIES // REPLICATES)
PHASESPACE_PATH = PHASESPACE or os.environ.get("HALCYON_VSM_PHASESPACE")
OUTPUT_FACTOR_PER_REPLICATE = max(1, OUTPUT_FACTOR_HISTORIES // OUTPUT_FACTOR_REPLICATES)

# Beam-frame slab extents, converted from the quoted SIDs.
MLC_PROXIMAL_Z = (SAD - MLC_PROXIMAL_TOP_SID, SAD - MLC_PROXIMAL_TOP_SID + LEAF_HEIGHT)
MLC_DISTAL_Z = (SAD - MLC_DISTAL_TOP_SID, SAD - MLC_DISTAL_TOP_SID + LEAF_HEIGHT)
if MLC_PROXIMAL_Z[1] > MLC_DISTAL_Z[0]:
    raise ValueError(
        f"the two leaf banks overlap in z: proximal ends at {MLC_PROXIMAL_Z[1]:g} cm, "
        f"distal starts at {MLC_DISTAL_Z[0]:g} cm (from the focal spot)"
    )
if MLC_PROXIMAL_Z[0] <= EMISSION_Z:
    raise ValueError(
        f"the emission plane ({EMISSION_Z:g} cm) must sit above the first device "
        f"({MLC_PROXIMAL_Z[0]:g} cm)"
    )

SPECTRUM = ali_rogers_mv(BEAM)


def _shell_spectra() -> tuple[tuple[float, Spectrum | None], ...]:
    """Resolve `SOFTENING_SHELLS` into (outer fan-radius edge, spectrum) pairs.

    ``None`` in place of a spectrum means "keep the wrapped source's own draw", which
    keeps the reference shell bit-identical to the unwrapped source rather than
    re-drawing the same distribution through a different stream.
    """
    if SOFTENING_SHELLS is None:
        return ()
    edges = [edge for edge, _ in SOFTENING_SHELLS]
    if edges != sorted(edges) or edges[-1] != float("inf") or len(set(edges)) != len(edges):
        raise ValueError(
            f"SOFTENING_SHELLS edges must increase strictly and end at inf, got {edges}"
        )
    return tuple(
        (
            edge,
            None
            if dc2 == 0.0
            else ali_rogers_mv(
                AliRogersMV(e_e=BEAM.e_e, c1=BEAM.c1, c2=BEAM.c2 + dc2, c3=BEAM.c3, c4=BEAM.c4)
            ),
        )
        for edge, dc2 in SOFTENING_SHELLS
    )


SHELL_SPECTRA = _shell_spectra()


class ShellSpectrumSource(Source):
    """Give a wrapped beam source a radius-dependent spectrum, in fan-radius shells.

    Each emitted photon keeps its position, direction and weight; only its energy is
    re-drawn from the spectrum of the shell its own ray crosses the isocentre plane in.
    This is the zero-waste alternative to the annular `CompositeSource` the engine's
    `PrimaryFluenceBeamSource` docstring suggests: no history is ever zero-weighted for
    landing in the wrong annulus, and the fluence shape is untouched by construction.

    The vectorized batch route draws its redraw uniforms from a stream seeded by
    ``(seed, history_offset)``, so replicates stay independent and a re-run is
    reproducible; the per-history ``emit`` route consumes two extra uniforms from the
    engine stream instead, so the two routes agree statistically but not bit for bit
    (they already do not for the wrapped sources).
    """

    def __init__(self, inner: Source, shells: tuple[tuple[float, Spectrum | None], ...]) -> None:
        if not shells:
            raise ValueError("ShellSpectrumSource needs at least one shell")
        self._inner = inner
        self._edges = np.asarray([edge for edge, _ in shells[:-1]], dtype=np.float64)
        self._spectra = tuple(spectrum for _, spectrum in shells)

    @property
    def max_energy(self) -> float:
        """Highest energy any shell (or the wrapped source) can emit."""
        return max(
            self._inner.max_energy,
            *(spectrum.max_energy for spectrum in self._spectra if spectrum is not None),
        )

    def _shell_of(self, x: Any, y: Any, z: Any, ux: Any, uy: Any, uz: Any) -> Any:
        """Shell index for rays, from where they cross the isocentre plane."""
        t = (SAD - z) / uz
        return np.digitize(np.hypot(x + t * ux, y + t * uy), self._edges)

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit from the wrapped source, then re-draw the energy for the ray's shell."""
        p = self._inner.emit(rng_state)
        spectrum = self._spectra[int(self._shell_of(p.x, p.y, p.z, p.ux, p.uy, p.uz))]
        return p if spectrum is None else p._replace(energy=spectrum.sample_energy(rng_state))

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, Any]:
        """Sample the wrapped source's batch, re-drawing energies shell by shell."""
        columns = dict(self._inner.sample_batch(seed, history_offset, n))
        shell = self._shell_of(
            *(columns[k].astype(np.float64) for k in ("x", "y", "z", "ux", "uy", "uz"))
        )
        rng = np.random.default_rng((seed, history_offset, 0x50F7))
        # One pair of uniforms per history whether used or not, so a history's redraw
        # does not depend on how many others share its shell.
        u_bin, u_within = rng.random((2, n))
        energy = columns["energy"].astype(np.float64)
        for index, spectrum in enumerate(self._spectra):
            if spectrum is None:
                continue
            chosen = shell == index
            if chosen.any():
                energy[chosen] = spectrum.sample_energies(u_bin[chosen], u_within[chosen])
        columns["energy"] = energy.astype(columns["energy"].dtype)
        return columns


class ElectronSource(Source):
    """Re-badge a beam source's emissions as electrons; geometry and weights untouched.

    The engine's host-sampling route seeds its transport queues from each record's own
    particle kind (the phase-space convention), and the plain beam sources stamp their
    batches as photons unconditionally -- so an electron component is the same geometry
    with the kind column rewritten, which is all this does. IAEA code 2 is an electron.
    """

    def __init__(self, inner: Source) -> None:
        self._inner = inner

    @property
    def max_energy(self) -> float:
        """The wrapped source's top energy; kind does not change table sizing."""
        return self._inner.max_energy

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit from the wrapped source, stamped as an electron."""
        return self._inner.emit(rng_state)._replace(kind="electron")

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, Any]:
        """Sample the wrapped batch with every record's particle type set to electron."""
        columns = dict(self._inner.sample_batch(seed, history_offset, n))
        columns["particle_type"] = np.full(n, 2, dtype=np.int32)
        return columns


def cone_fluence() -> RadialFluence:
    """Analytic unflattened cone psi(r), tabulated as a `RadialFluence` at the isocentre.

    A Gaussian in the off-axis radius out to the primary collimator's opening and zero
    beyond it, which is the `RadialFluence` convention for a table that ends. Sampled
    finely enough that the linear interpolation between nodes is not the limiting
    approximation. See the `CONE_SIGMA` cell: the shape is invented, and refitting it
    against the largest field's measured profile is expected.
    """
    radii = np.linspace(0.0, PRIMARY_COLLIMATOR_RADIUS, 106)
    return RadialFluence(
        radii=radii,
        values=np.exp(-((radii / CONE_SIGMA) ** 2)),
        reference_distance=SAD,
    )


# Where the figures and tables are written; the notebook has no __file__ of its own.
STEM = Path(__file__).with_suffix("") if "__file__" in dir() else Path("halcyon_commissioning_demo")

# The fluence table lives beside this notebook, so it is found whatever the working
# directory is. A bare name is resolved there; an absolute path is used as given.
FLUENCE = (
    RadialFluence.from_file(STEM.parent / PRIMARY_FLUENCE_FILE, reference_distance=SAD)
    if PRIMARY_FLUENCE_FILE is not None
    else cone_fluence()
)


def measuring_voxels(field_x: float, field_y: float) -> int:
    """Voxels in the on-axis measuring volume — the count `roi_dose` averages over.

    Mirrors `roi_dose`'s choice of half-width, and the phantom lattice it lands on: an odd
    lateral voxel count centred on the axis, and depth planes at half-voxel offsets from
    the surface. It is **18 voxels** for a field that can hold the chamber-like 0.5 cm cube
    and **1** below 2 cm, and that factor of eighteen is why it has to enter the allocation
    — see `dilution`. `simulate` checks this against the selector actually used, so the two
    cannot drift apart.
    """
    half = 0.25 if min(field_x, field_y) >= 2.0 else 0.4 * SPACING
    lateral = 1 + 2 * math.floor(half / SPACING + 1.0e-9)
    depth = max(1, 2 * math.floor((half + 0.5 * SPACING) / SPACING + 1.0e-9))
    return lateral * lateral * depth


def dilution(area_iso: float, voxels: int) -> float:
    """How dilute one field's measurement is, against the reference field's.

    The single quantity both allocations below are driven by. What a measuring volume sees
    is a **density**: the source fan spreads a fixed number of histories over an area that
    grows with the field, and the volume they are counted in shrinks by eighteen once the
    field is too small to hold a chamber. `area_iso / voxels` is that density inverted, and
    a field whose dilution is 3 needs three times the histories to reach the same error.
    """
    return (area_iso / voxels) / REFERENCE_DILUTION


def histories_per_replicate(diluted: float, base: int, *, boost: float = 1.0) -> int:
    """Per-replicate histories for a field of this `dilution`, in proportion to it.

    Clamped at both ends; see the statistics cell for the measured spread this replaces
    and the cost it carries.
    """
    scale = min(HISTORY_SCALE_MAX, max(HISTORY_SCALE_FLOOR, diluted))
    return max(1, int(boost * scale * base))


def phase_space_reuse(diluted: float) -> float:
    """Transport histories per pre-solved head history, inversely to the `dilution`.

    The pre-solved phase space is resampled by the transport, so a small one puts a floor
    under the error that no amount of transport can lift — and that floor is a density
    too: a fixed head-history count leaves fewer distinct records able to reach a diluted
    measuring volume. Measured on this geometry; see the statistics cell.
    """
    return min(PHASE_SPACE_REUSE, max(PHASE_SPACE_REUSE_MIN, PHASE_SPACE_REUSE / diluted))


print(
    f"6 MV FFF (derived): endpoint {BEAM.e_e:g} MeV, C2 {BEAM.c2:g} "
    f"(flattened fit: {_FLATTENED.c2:g}), mean energy {SPECTRUM.mean_energy:.3f} MeV "
    f"against {ali_rogers_mv(_FLATTENED).mean_energy:.3f} MeV flattened"
)
print(
    f"SAD {SAD:g} cm; profiles at SSD {PROFILE_SSD:g} cm, output factors at "
    f"SSD {OUTPUT_FACTOR_SSD:g} cm / {OUTPUT_FACTOR_DEPTH:g} cm deep; head {HEAD_MODEL!r}, "
    f"devices {DEVICES!r}"
)
print(
    f"proximal MLC: {MLC_PROXIMAL_PAIRS} pairs, SID "
    f"{MLC_PROXIMAL_TOP_SID:g}-{MLC_PROXIMAL_TOP_SID - LEAF_HEIGHT:g} cm; "
    f"distal MLC: {MLC_DISTAL_PAIRS} pairs, SID "
    f"{MLC_DISTAL_TOP_SID:g}-{MLC_DISTAL_TOP_SID - LEAF_HEIGHT:g} cm; "
    f"pitch {MLC_PITCH:g} cm at iso, effective {MLC_PITCH / 2:g} cm"
)
if PRIMARY_FLUENCE_FILE is None:
    print(
        f"primary fluence: analytic FFF cone, sigma {CONE_SIGMA:g} cm at iso "
        f"(psi(14 cm) = {FLUENCE.at_radius(14.0):.3f}), zero beyond "
        f"{PRIMARY_COLLIMATOR_RADIUS:g} cm"
    )
else:
    print(
        f"primary fluence: {PRIMARY_FLUENCE_FILE} (fitted), "
        f"psi(14 cm) = {FLUENCE.at_radius(14.0):.3f}, "
        f"psi(19.8 cm) = {FLUENCE.at_radius(19.8):.3f}, zero beyond "
        f"{FLUENCE.max_radius:g} cm"
    )

# %% [markdown]
# ## The phase space, when one is configured
#
# Everything in this cell is inert unless `PHASESPACE` names a file. With one, the head's
# whole source model — spectrum, focal spot, off-axis fluence and softening, head scatter
# and contamination — arrives in the particles instead of being built below, and the only
# thing this notebook still supplies is the collimator.
#
# The file is memory-mapped and scanned once: every record's ray is projected to the
# isocentre plane and that aim point kept, so each field can select the records aimed into
# its own fan by index and sample them straight from the map. Nothing is copied out until a
# batch is drawn.
#
# Two file-format notes, both handled here rather than in the library. Some exporters write
# the IAEA `W` flag as 0 with a "constant" of 0.0 — the sign of `w` rides on the particle
# type, which is a *stored*-flag convention — so the header is parsed with that flag read
# as 1 and its record length checked against the file size. And `z` is taken from the
# records rather than from any accompanying document, because the two have been known to
# disagree.


# %%
def vsm_header(path: str) -> IAEAHeader:
    """Parse the pyvsm header, reading its off-spec `W` flag as the standard 1."""
    header_path, _ = _iaea_path(path)
    sections = _parse_sections(header_path.read_text())
    flags = [int(token) for token in sections["RECORD_CONTENTS"][:7]]
    flags[5] = 1
    stored = tuple(bool(flags[i]) for i in (0, 1, 2, 3, 4, 6))
    if not all(stored):
        raise ValueError("this notebook expects x, y, z, u, v and weight stored per record")
    n_extra_floats, n_extra_ints = (
        int(sections["RECORD_CONTENTS"][7]),
        int(sections["RECORD_CONTENTS"][8]),
    )
    record_length = 1 + 4 + 4 * sum(stored) + 4 * n_extra_floats + 4 * n_extra_ints
    if int(sections["RECORD_LENGTH"][0]) != record_length:
        raise ValueError(
            f"RECORD_LENGTH {sections['RECORD_LENGTH'][0]} disagrees with the "
            f"{record_length} bytes implied by RECORD_CONTENTS"
        )
    return IAEAHeader(
        byte_order="<" if sections["BYTE_ORDER"][0] == "1234" else ">",
        record_length=record_length,
        n_particles=_n_particles(sections, header_path, record_length),
        n_extra_floats=n_extra_floats,
        n_extra_ints=n_extra_ints,
        stored=stored,
        constants={},
    )


class VSMPhaseSpace:
    """The whole file, memory-mapped, with every record's isocentre aim point scanned."""

    def __init__(self, path: str, *, chunk: int = 8_000_000) -> None:
        self.header = vsm_header(path)
        _, self._phsp_path = _iaea_path(path)
        self.dtype = _record_dtype(self.header)
        self.struct = _build_struct(self.header)
        self._file = self._phsp_path.open("rb")
        self.mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        n = self.header.n_particles
        self.kind = np.empty(n, dtype=np.int8)
        self.energy = np.empty(n, dtype=np.float32)
        self.x_iso = np.empty(n, dtype=np.float32)
        self.y_iso = np.empty(n, dtype=np.float32)
        self.weight_sum = 0.0
        z_seen: list[float] = []
        records = self.records()
        for start in range(0, n, chunk):
            block = records[start : start + chunk]
            u = block["u"].astype(np.float64)
            v = block["v"].astype(np.float64)
            w = np.sign(block["typ"].astype(np.float64)) * np.sqrt(
                np.clip(1.0 - u * u - v * v, 0.0, None)
            )
            if np.any(w <= 0.0):
                raise ValueError("the phase space carries backward-going particles")
            z = block["z"].astype(np.float64)
            z_seen += [float(z.min()), float(z.max())]
            t = (SAD - z) / w
            stop = start + block.size
            self.x_iso[start:stop] = block["x"] + t * u
            self.y_iso[start:stop] = block["y"] + t * v
            self.kind[start:stop] = np.abs(block["typ"])
            self.energy[start:stop] = np.abs(block["e"])
            self.weight_sum += float(block["weight"].astype(np.float64).sum())
            del block, u, v, w, z, t
        del records
        if max(abs(min(z_seen) - PHASESPACE_PLANE_Z), abs(max(z_seen) - PHASESPACE_PLANE_Z)) > 1e-3:
            raise ValueError(
                f"the records sit at z = {min(z_seen):g}..{max(z_seen):g}, not at the stated "
                f"plane {PHASESPACE_PLANE_Z:g} cm from the target; check the frame"
            )

    def records(self) -> np.ndarray:
        """Return a structured view of every record on the map (no copy)."""
        return np.frombuffer(self.mmap, dtype=self.dtype, count=self.header.n_particles)

    def __len__(self) -> int:
        """Return the number of records in the file."""
        return int(self.header.n_particles)

    def close(self) -> None:
        """Release the map; every source built from it must be dead first."""
        self.mmap.close()
        self._file.close()


class ClippedPhaseSpaceSource(Source):
    """The file's records at a given index set, sampled with replacement.

    The same contract as the library's `PhaseSpaceSource` (one uniform per `emit`;
    chunk-invariant `sample_batch` on its own PCG64 stream; per-record kind and
    weight), restricted to a subset of the file by index -- here the photons aimed
    into one field's fan. Records are gathered from the memory map only when drawn.
    """

    def __init__(self, phsp: VSMPhaseSpace, indices: np.ndarray) -> None:
        if indices.size == 0:
            raise ValueError("no records selected")
        self._phsp = phsp
        self._indices = np.ascontiguousarray(indices, dtype=np.int64)
        self._max_energy = float(phsp.energy[self._indices].max())

    def __len__(self) -> int:
        """Return the number of records available to sample."""
        return int(self._indices.size)

    @property
    def max_energy(self) -> float:
        """Highest energy in the selection, in MeV (for table sizing)."""
        return self._max_energy

    def emit(self, rng_state: RNGState) -> Primary:
        """Emit one selected record, uniformly; consumes one uniform."""
        k = min(int(uniform(rng_state) * len(self)), len(self) - 1)
        offset = int(self._indices[k]) * self._phsp.header.record_length
        raw = self._phsp.mmap[offset : offset + self._phsp.header.record_length]
        record = _decode_record(raw, self._phsp.header, self._phsp.struct)
        return Primary(
            energy=record.energy,
            x=record.x,
            y=record.y,
            z=record.z,
            ux=record.u,
            uy=record.v,
            uz=record.w,
            kind={IAEA_PHOTON: "photon", IAEA_ELECTRON: "electron"}.get(
                record.particle_type, "positron"
            ),
            weight=record.weight,
        )

    def sample_batch(self, seed: int, history_offset: int, n: int) -> dict[str, np.ndarray]:
        """Column arrays for histories ``[history_offset, history_offset + n)``."""
        k = _sample_indices(seed, history_offset, n, len(self))
        return gather_columns(self._phsp, self._indices[k])


def gather_columns(phsp: VSMPhaseSpace, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Decode the records at `indices` into the engine's upload columns (float32)."""
    recs = phsp.records()[indices]
    typ = recs["typ"].astype(np.int32)
    u = np.asarray(recs["u"], dtype=np.float32)
    v = np.asarray(recs["v"], dtype=np.float32)
    tmp = u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2
    sign_w = np.where(typ < 0, -1.0, 1.0)
    w = np.where(tmp <= 1.0, sign_w * np.sqrt(np.maximum(0.0, 1.0 - tmp)), 0.0).astype(np.float32)
    over = tmp > 1.0
    if over.any():
        scale = np.sqrt(tmp[over]).astype(np.float32)
        u, v = u.copy(), v.copy()
        u[over] /= scale
        v[over] /= scale
    return {
        "particle_type": np.abs(typ),
        "energy": np.abs(np.asarray(recs["e"], dtype=np.float32)),
        "x": np.asarray(recs["x"], dtype=np.float32),
        "y": np.asarray(recs["y"], dtype=np.float32),
        "z": np.asarray(recs["z"], dtype=np.float32),
        "ux": u,
        "uy": v,
        "uz": w,
        "weight": np.asarray(recs["weight"], dtype=np.float32),
    }


if PHASESPACE_PATH is None:
    PHSP: Any = None
    AXIS_SPECTRUM: Any = None
    R_ISO: Any = None
else:
    scan_start = time.perf_counter()
    PHSP = VSMPhaseSpace(PHASESPACE_PATH)
    N_PHOTONS = int(np.count_nonzero(PHSP.kind == IAEA_PHOTON))
    N_ELECTRONS = int(np.count_nonzero(PHSP.kind == IAEA_ELECTRON))
    if len(PHSP) != N_PHOTONS + N_ELECTRONS:
        raise ValueError("the file carries particles other than photons and electrons")
    R_ISO = np.hypot(PHSP.x_iso.astype(np.float64), PHSP.y_iso.astype(np.float64))
    print(
        f"phase space: {len(PHSP):,} records ({N_PHOTONS:,} photons, {N_ELECTRONS:,} electrons, "
        f"{100.0 * N_ELECTRONS / len(PHSP):.2f} %), scanned in "
        f"{time.perf_counter() - scan_start:.0f} s"
    )
    print(
        f"  photon aim points reach r = {R_ISO[PHSP.kind == IAEA_PHOTON].max():.2f} cm at the "
        f"isocentre (clamp {CLAMP_RADIUS:g} cm), electrons "
        f"{R_ISO[PHSP.kind == IAEA_ELECTRON].max():.1f} cm; "
        f"mean weight {PHSP.weight_sum / len(PHSP):.3f}; photon energies to "
        f"{PHSP.energy[PHSP.kind == IAEA_PHOTON].max():.2f} MeV, electrons to "
        f"{PHSP.energy[PHSP.kind == IAEA_ELECTRON].max():.2f} MeV"
    )
    # The readme's verification table, reproduced from the records: mean weight tracks the
    # fitted I(r) and the mean energy falls off axis. Weight bands read from the file.
    _weights_all = PHSP.records()["weight"]
    print(f"  {'r at iso':>10} {'mean weight':>12} {'<E>/MeV':>9}   (readme: I(r) and <E>)")
    for lo, hi in ((0.0, 2.0), (8.0, 10.0), (16.0, 18.0), (18.0, 20.0)):
        band = (lo <= R_ISO) & (hi > R_ISO) & (PHSP.kind == IAEA_PHOTON)
        print(
            f"  {lo:4.0f}-{hi:<5.0f} {float(_weights_all[band].mean()):12.4f} "
            f"{float(PHSP.energy[band].mean()):9.3f}"
        )
    del _weights_all

    # The on-axis photon spectrum, read back from the particles (r < 2 cm at the isocentre,
    # which is the other notebook's unsoftened reference shell): what the tip radiation-edge
    # solve and the transmission check attenuate.
    _axis = (R_ISO < 2.0) & (PHSP.kind == IAEA_PHOTON)
    _edges = np.linspace(0.0, float(PHSP.energy[_axis].max()) + 1e-3, 61)[1:]
    _edges = np.concatenate([[0.01], _edges])
    _counts, _ = np.histogram(PHSP.energy[_axis], bins=_edges)
    AXIS_SPECTRUM = Spectrum(edges=_edges, weights=_counts.astype(np.float64))
    # The stated form, on its own grid, against what the records actually carry. Both
    # numbers are photon-number-weighted means over the same population; the records also
    # carry the ~4 % extra-focal component, which is slightly softer, so the file is
    # expected to sit a shade below the primary-only model rather than exactly on it.
    _STATED = (
        None if VSM_BEAM is None else ali_rogers_mv(VSM_BEAM, e_min=VSM_GRID[0], n_bins=VSM_GRID[1])
    )
    _axis_energies = PHSP.energy[_axis].astype(np.float64)
    print(
        f"  on-axis (r < 2 cm) photon mean energy: {_axis_energies.mean():.4f} MeV from the "
        f"records"
        + ("" if _STATED is None else f", {_STATED.mean_energy:.4f} MeV from the stated form")
    )
    print(
        f"  highest photon energy in the file: {PHSP.energy[PHSP.kind == IAEA_PHOTON].max():.4f} "
        f"MeV against the stated {VSM_BEAM.e_e:g} MeV endpoint"
    )
    if _STATED is not None and abs(_axis_energies.mean() - _STATED.mean_energy) > 0.05:
        raise ValueError(
            f"the file's on-axis mean energy ({_axis_energies.mean():.4f} MeV) disagrees with "
            f"its own stated spectrum ({_STATED.mean_energy:.4f} MeV) by more than 50 keV; "
            "the export and its documentation describe different beams -- do not "
            "transport this file"
        )


class FanClip(NamedTuple):
    """One field's slice of the file: the photon source, the electron source, counts."""

    photons: ClippedPhaseSpaceSource
    electrons: InMemoryPhaseSpaceSource | None
    n_photons: int
    n_electrons: int
    area_iso: float


def fan_clip(field_x: float, field_y: float) -> FanClip:
    """Select the file's records aimed into this field's fan, by kind."""
    half_x = 0.5 * field_x + fan_margin(field_x)
    half_y = 0.5 * field_y + fan_margin(field_y)
    inside = (np.abs(PHSP.x_iso) <= half_x) & (np.abs(PHSP.y_iso) <= half_y)
    photon_index = np.flatnonzero(inside & (PHSP.kind == IAEA_PHOTON))
    electron_index = np.flatnonzero(inside & (PHSP.kind == IAEA_ELECTRON))
    electrons = None
    if electron_index.size:
        columns = gather_columns(PHSP, electron_index)
        electrons = InMemoryPhaseSpaceSource(**columns)
    return FanClip(
        photons=ClippedPhaseSpaceSource(PHSP, photon_index),
        electrons=electrons,
        n_photons=int(photon_index.size),
        n_electrons=int(electron_index.size),
        area_iso=4.0 * half_x * half_y,
    )


# The spectrum the geometry checks below attenuate: the file's own on-axis photons when a
# phase space is driving the head, the analytic model otherwise. Only the deterministic
# checks use it -- the tip radiation-edge solve, the leaf-transmission table and the
# aperture trace. Transport always uses each history's own energy.
EDGE_SPECTRUM = SPECTRUM if AXIS_SPECTRUM is None else AXIS_SPECTRUM
MAX_ENERGY = SPECTRUM.max_energy if PHSP is None else float(PHSP.energy.max())

# %% [markdown]
# ## Materials and cross-sections
#
# `tungsten` is the real thing and needs the compiled EPICS tables (cached after the
# first download): the leaves are `tungsten_alloy`, the published W95/Ni3.5/Cu1.5
# heavy-alloy surrogate at 18.0 g/cm^3, which is what actual dual-layer leaves are
# made of (pure W at 19.30 was this notebook's earlier stand-in; the alloy transmits
# roughly half again as much through a single bank, which is why the transmission
# check below exists). `water` is the dependency-free stand-in: water at the leaf
# density, plus water at air's density above the phantom. Its low-energy behaviour is
# not tungsten's, so leakage and the deep transmission tail are indicative only — and
# on a jawless head, where every edge is tungsten, that caveat is worth more than it
# was on the flattened machine.

# %%
if DEVICES == "tungsten":
    from pyradmc.data.tabulated.build import download_library
    from pyradmc.data.tabulated.precompile import compile_materials
    from pyradmc.data.tabulated.source import TabulatedCrossSections

    DEVICE_MATERIAL, DEVICE_DENSITY = TUNGSTEN_ALLOY, 18.0
    AIR_MATERIAL, AIR_DENSITY = AIR, 1.205e-3
    print("compiling the tabulated table (EPICS cache/download) ...")
    _tables = compile_materials(
        download_library("epdl").read_text(encoding="latin-1"),
        download_library("eedl").read_text(encoding="latin-1"),
        e_max=max(7.0, MAX_ENERGY + 1.0),
    )
    XS: Any = TabulatedCrossSections(_tables, geometry_densities=((WATER, 1.0), (AIR, AIR_DENSITY)))
else:
    from pyradmc.data.analytic import AnalyticCrossSections

    # 18.0, not pure tungsten's 19.30: the stand-in exists to reproduce the shipped
    # geometry's areal density, and transmission is exponential in rho t.
    DEVICE_MATERIAL, DEVICE_DENSITY = WATER, 18.0
    AIR_MATERIAL, AIR_DENSITY = WATER, 1.205e-3
    XS = AnalyticCrossSections(geometry_densities=((WATER, 1.0),))

print(XS.provenance)

# %% [markdown]
# ## Leaf transmission against the published values
#
# The one direct constraint on the leaf material this head offers: published Halcyon
# transmission is ~0.4-0.5 per cent through a single layer and ~0.01 per cent (i.e.
# effectively nothing) through both. The check below is the *narrow-beam spectral*
# transmission — the axis spectrum attenuated through the full leaf height at the leaf
# density, number- and energy-fluence weighted — which is the material-data part of
# that measurement. It is not the measured quantity itself: a chamber under a closed
# bank also collects in-leaf build-up and hardened scatter plus phantom scatter, and
# the published figure averages over *interleaf leakage*, which this model does not
# have at all (no interleaf gaps; see `CLOSED_PAIR_PARK`). All of those raise the
# measured number, so the spectral value should land *below* the published range --
# it reads about half of it (0.19 vs 0.4-0.5 per cent single-layer with the alloy;
# pure W at 19.30 would halve it again, which is the measurable content of the alloy
# switch). Order-of-magnitude disagreement would mean the material or the leaf height
# is wrong; landing under the band by a factor consistent with the unmodelled
# contributions is as much as this check can claim. With `DEVICES = "water"` the
# numbers are the stand-in's, not tungsten's, and say nothing about the machine.

# %%
_edges = EDGE_SPECTRUM.edges
_e_mid = 0.5 * (_edges[:-1] + _edges[1:])
_mu = np.array([XS.mu_over_rho_total(float(e), DEVICE_MATERIAL) for e in _e_mid])
for _layers, _published in ((1, "0.4-0.5 %"), (2, "~0.01 %")):
    _path = np.exp(-_mu * DEVICE_DENSITY * (_layers * LEAF_HEIGHT))
    _by_number = float(np.sum(EDGE_SPECTRUM.bin_probabilities * _path))
    _by_energy = float(np.sum(EDGE_SPECTRUM.bin_probabilities * _e_mid * _path)) / float(
        np.sum(EDGE_SPECTRUM.bin_probabilities * _e_mid)
    )
    print(
        f"narrow-beam transmission, {_layers} layer(s) ({_layers * LEAF_HEIGHT:g} cm at "
        f"{DEVICE_DENSITY:g} g/cm^3): {100.0 * _by_number:.3f} % by number, "
        f"{100.0 * _by_energy:.3f} % by energy fluence (published dose: {_published})"
    )

# %% [markdown]
# ## The beam-limiting stack: two banks, and no field-defining jaws
#
# A field is stated at the isocentre as `(field_x, field_y)` — leaf travel is along x,
# strips tile y — and `project_between_planes` carries it to each bank's mid-plane. Every
# field edge is made by a leaf: ends in x, sides in y.
#
# Each bank is always built over its whole nominal width, and over more than that when the
# source fan reaches past it. The machine carries all 29 and 28 pairs whatever the field,
# parked closed, and for a while this did not: strips were built only as far as the fan
# reached, so a 1 cm field had three of them and beyond that the head was open but for the
# backup collimator's 5.25 cm of tungsten against the leaf stack's 15.4. Because leaf
# bodies are unbounded along travel and strips are not, that hole existed in y and not in
# x, and a traced 28 x 1 came out 5.5 % hotter than a 1 x 28 over ±5 cm from leakage alone.
# Building the full bank costs nothing measurable — 4 pairs to 31 on a narrow-y pre-solve
# sat inside the run-to-run scatter of cells that did identical work — so it is not worth
# the asymmetry. Strips outside the *real* bank are still built when the fan needs them,
# and always closed.
#
# Above them sit two slabs that are **not** field-defining and are the one piece of this
# geometry invented for the model rather than taken from the machine: they close the head
# beyond the bank edge, which the leaves themselves cannot. See
# `backup_collimator` for why a jawless head needs that and what it approximates. They
# never sit closer than `FAN_MARGIN_MIN` to a field edge, so no width or penumbra below
# depends on them.
#
# Two details that decide whether this reproduces the machine.
#
# **Which bank sets the y edge is a property of the edge, not of the bank.** Both banks
# open every strip that meets the field, so the aperture is their intersection; a y edge
# on a half-integer lands on a proximal boundary and one on an integer on a distal
# boundary. Nothing selects a "controlling" layer — it falls out.
#
# **The tips carry two retractions, per bank.** The first is the tangent offset: a rounded
# end's edge follows the focal-spot ray that grazes the arc, which is not the tip's own
# coordinate, and the discrepancy depends on the bank's distance from the source.
#
# The second exists because the grazing ray is the *zero-chord* ray, so it marks full
# transmission rather than the radiation edge, which is conventionally the 50 per cent
# point and sits inside it. That gap is negligible while the ray is near-normal to the arc
# and grows as it turns oblique: below 0.02 mm out to a 10 x 10 field, 0.16 mm per side at
# 20 x 20 and 0.47 mm at 28 x 28. Left uncorrected it makes the largest field come out
# **0.84 mm narrow**, which three independent runs measured to within 0.14 mm of each
# other and a transport-free ray trace reproduced to 0.05 mm. `radiation_edge_correction`
# closes it. It needs the attenuation and so has no closed form, but nothing in it is
# fitted -- it is the tip's own arc and the beam's own spectrum.
#
# `MLC_TIP_OFFSET` is the one free constant on top of both, and it is still zero.


# %%
def fan_margin(field: float) -> float:
    """Fan reach past one edge of a field, cm at the isocentre."""
    return min(FAN_MARGIN_MAX, max(FAN_MARGIN_MIN, FAN_MARGIN_SLOPE * field))


def tangent_offset(half: float, z_mid: float) -> float:
    """Extra tip retraction that puts a rounded end's radiation edge at `half` (cm at iso).

    The ray from the focal spot tangent to the tip cylinder — radius `MLC_TIP_RADIUS`,
    axis on the bank mid-plane `z_mid` — crosses the isocentre plane inside the tip's own
    projection, by

        t = half * z_mid / SAD,  c = t + R,
        u = (c z_mid - R sqrt(z_mid^2 + c^2 - R^2)) / (z_mid^2 - R^2),

    so the offset is `half - u * SAD`. Pure geometry: nothing here is fitted, and it is
    evaluated at each bank's own mid-plane rather than shared.
    """
    radius = MLC_TIP_RADIUS
    t = half * z_mid / SAD
    c = t + radius
    denominator = z_mid * z_mid - radius * radius
    if denominator <= 0.0:
        raise ValueError(
            f"the tip radius {radius:g} cm reaches the focal spot from a bank at "
            f"{z_mid:g} cm; the tangent construction is undefined"
        )
    u = (c * z_mid - radius * math.sqrt(z_mid * z_mid + c * c - radius * radius)) / denominator
    return half - u * SAD


def _traced_edge(reach_iso: float, z_range: tuple[float, float]) -> float:
    """Isocentre x at which one leaf pair's narrow-beam primary falls to half.

    A single wide pair, so only the tips are in the beam, traced from a point focal spot
    with no scatter and no source size: the number this returns is the collimator's edge
    and nothing else. That is the same physics as the aperture figure at the end of this
    notebook, restricted to one line and one device.
    """
    z_mid = 0.5 * (z_range[0] + z_range[1])

    def to_mid(value: float) -> float:
        return float(project_between_planes(value, SAD, z_mid))

    wide = 10.0 * (abs(reach_iso) + 1.0)
    pair = MLC(
        z_top=z_range[0],
        z_bottom=z_range[1],
        leaf_edges_v=(to_mid(-wide), to_mid(wide)),
        tips_neg=(to_mid(-reach_iso),),
        tips_pos=(to_mid(reach_iso),),
        tip_radius=MLC_TIP_RADIUS,
        material=DEVICE_MATERIAL,
        density=DEVICE_DENSITY,
        focused_sides=MLC_FOCUSED_SIDES,
    )
    stack = BeamLimitingStack(frame=BeamFrame(origin=(0.0, 0.0, 0.0)), devices=(pair,))
    # The tip's shadow reaches further inward the more oblique the ray, so the search span
    # has to scale with the retraction rather than be a fixed window -- at a 28 x 28 field
    # a 6 mm window does not even contain the open plateau, and normalizing inside the
    # penumbra put the solved edge 3 mm wrong. The axis is prepended as the open reference
    # because it is inside the opening whatever the field is.
    window = max(1.0, 0.15 * abs(reach_iso))
    span = np.linspace(max(0.0, reach_iso - window), reach_iso + window, 3001)
    probe = np.concatenate([[0.0], span])
    points = np.stack([probe, np.zeros_like(probe), np.full(probe.size, SAD)], axis=1)
    directions = points / np.linalg.norm(points, axis=1)[:, None]
    chord = stack.path_lengths(np.zeros_like(points), directions)[:, 0]

    energies = 0.5 * (EDGE_SPECTRUM.edges[:-1] + EDGE_SPECTRUM.edges[1:])
    total = np.zeros(probe.size)
    for energy, weight in zip(energies, EDGE_SPECTRUM.bin_probabilities, strict=True):
        if weight <= 0.0:
            continue
        mu = DEVICE_DENSITY * XS.mu_over_rho_total(float(energy), DEVICE_MATERIAL)
        total += weight * energy * np.exp(-mu * chord)
    open_value, curve = total[0], total[1:] / total[0]
    if open_value <= 0.0 or curve[0] < 0.9 or curve[-1] > 0.1:
        raise ValueError(
            f"the search window of {window:g} cm does not bracket the edge of a tip at "
            f"{reach_iso:g} cm (it runs from {curve[0]:.3f} to {curve[-1]:.3f} of the open "
            "value); the 50 % crossing cannot be located"
        )
    # Monotone decreasing across the tip, so reversing it gives np.interp its ascending x.
    return float(np.interp(0.5, curve[::-1], span[::-1]))


_EDGE_CORRECTION: dict[tuple[float, float, float], float] = {}


def radiation_edge_correction(half: float, z_range: tuple[float, float]) -> float:
    """Retraction beyond the tangent offset that lands the 50 % edge on `half`.

    `tangent_offset` places the ray that grazes the tip arc. That ray passes through zero
    tungsten, so it is the *full transmission* ray, not the radiation edge; the 50 % point
    lies inside it by an amount that grows with the ray's obliquity. See the cell above for
    the size of it and for how it was measured.

    Solved rather than derived, because the 50 % condition involves the attenuation and
    has no closed form. It is still not a *fitted* offset: the only inputs are the tip
    radius, the bank's position and the beam's own spectrum, and nothing is tuned against
    a measurement. Newton converges in three or four steps from the closed form, since the
    edge moves one for one with the retraction; results are memoized because a sweep asks
    for the same handful of field sizes many times over.

    What is left afterwards is a constant 0.08 mm, at every field and in a single bank as
    much as in the pair. That is not geometry: it is the gap between the 50 % of the open
    value solved for here and the inflection construction the profiles are measured with,
    which do not coincide on an edge as asymmetric as a rounded tip's. The width in the
    *other* direction, where the geometry is exact, carries a constant bias of the same
    kind. Both are far below the 0.14 mm the transported runs reproduce to.
    """
    key = (round(half, 9), z_range[0], z_range[1])
    if key in _EDGE_CORRECTION:
        return _EDGE_CORRECTION[key]
    z_mid = 0.5 * (z_range[0] + z_range[1])
    correction = 0.0
    for _ in range(8):
        residual = _traced_edge(half + tangent_offset(half, z_mid) + correction, z_range) - half
        if abs(residual) < 1.0e-6:
            break
        correction -= residual
    else:
        raise ValueError(
            f"the radiation-edge correction did not converge for a half-width of "
            f"{half:g} cm at a bank spanning {z_range}; the last residual was {residual:g} cm"
        )
    _EDGE_CORRECTION[key] = correction
    return correction


def strip_edges(bank_pairs: int, reach: float) -> np.ndarray:
    """Strip boundaries covering `+-reach` at the isocentre, in this bank's own phase.

    A bank of `n` leaves has boundaries at `(i - n/2) * MLC_PITCH`, so the *parity* of
    `n` is the whole stagger: 28 puts a boundary on the axis, 29 puts a leaf there.
    Growing or shrinking the built strip count by two preserves that parity and keeps
    the set symmetric, which is why the loop steps by two.
    """
    n = bank_pairs % 2 or 2
    while 0.5 * n * MLC_PITCH < reach:
        n += 2
    return (np.arange(n + 1) - 0.5 * n) * MLC_PITCH


def leaf_bank(
    z_range: tuple[float, float],
    bank_pairs: int,
    field_x: float,
    field_y: float,
    park_sign: float,
) -> MLC:
    """One layer of the stack, set for a rectangular field stated at the isocentre."""
    half_x, half_y = 0.5 * field_x, 0.5 * field_y
    z_mid = 0.5 * (z_range[0] + z_range[1])
    bank_half = 0.5 * bank_pairs * MLC_PITCH
    # Only the y fan needs strips: in x the leaves are unbounded and the tips do the work.
    # The whole bank is always built, even where the fan cannot reach it: a real machine
    # carries all its pairs whatever the field, and the ones outside the field are closed
    # tungsten, not air. Building only to the fan left a narrow field shielded out there by
    # the backup collimator's 5.25 cm instead of the leaf stack's 15.4, which showed up as
    # a ~0.9 per cent leakage plateau 3 to 4 cm off axis in the leaf-travel direction only
    # -- an asymmetry between 28 x 1 and 1 x 28 that no real geometry has.
    #
    # It costs nothing. Interleaved A/B at full output-factor settings, alternating the two
    # geometries back to back inside one process: 28 x 1 goes from 3 strips to 28 at 0.99x,
    # against a 1 x 28 null control that builds 34 either way and reads 1.01x with the same
    # +-20 per cent scatter. Do not trust a cross-run comparison here instead -- the same
    # eighteen fields timed 1630 s one day and 5720 s another at unchanged geometry, which
    # is this card's clocks, not the leaf count.
    #
    # It also buys nothing measurable in dose. The transposed-field ratio it was meant to
    # explain moved +0.25 +/- 1.00 points, i.e. not at all: the leakage sat off axis and
    # reached the measuring point only through phantom scatter. The machine's +3.2 per cent
    # Scp(1 x Y) / Scp(Y x 1) asymmetry is still unexplained, and this rules out leaf-bank
    # leakage as its cause. Kept because the geometry is right, not because it fixed a gap.
    edges = strip_edges(bank_pairs, max(half_y + fan_margin(field_y), bank_half))

    reach_x = (
        half_x
        + tangent_offset(half_x, z_mid)
        + radiation_edge_correction(half_x, z_range)
        + MLC_TIP_OFFSET
    )
    park = park_sign * (half_x + CLOSED_PAIR_PARK)
    tips_neg, tips_pos = [], []
    for lo, hi in itertools.pairwise(edges):
        # A real leaf (inside the nominal bank) that meets the field opens; everything
        # else is closed tungsten, parked away from the other layer's abutment.
        real = abs(lo) <= bank_half + 1.0e-9 and abs(hi) <= bank_half + 1.0e-9
        if real and lo < half_y - 1.0e-9 and hi > -half_y + 1.0e-9:
            tips_neg.append(-reach_x)
            tips_pos.append(reach_x)
        else:
            tips_neg.append(park)
            tips_pos.append(park)

    def to_mid(values: Any) -> tuple[float, ...]:
        return tuple(float(project_between_planes(v, SAD, z_mid)) for v in values)

    return MLC(
        z_top=z_range[0],
        z_bottom=z_range[1],
        leaf_edges_v=to_mid(edges),
        tips_neg=to_mid(tips_neg),
        tips_pos=to_mid(tips_pos),
        tip_radius=MLC_TIP_RADIUS,
        material=DEVICE_MATERIAL,
        density=DEVICE_DENSITY,
        focused_sides=MLC_FOCUSED_SIDES,
    )


def backup_collimator(axis: str, z_range: tuple[float, float], half_iso: float) -> JawPair:
    """One slab of the fixed shielding that closes the head outside the source fan.

    **Not a field-defining device**, and it is the one component here that is a modelling
    convenience rather than a stated part of the machine. It exists because a jawless head
    has a hole in it: past the edge of the leaf banks there is no material at all, and the
    wide extra-focal source — which spreads over thousands of square centimetres at the
    isocentre, most of it nowhere near any leaf — would flood the phantom through that hole
    unattenuated.

    What it stands in for is the fixed secondary collimator above the leaves, which is
    real, tungsten, and upstream of the phantom, so the dosimetric outcome is the same.
    Putting the absorber at the fan boundary rather than out at the bank edge costs one
    stated approximation — the scatter it generates is born a little higher in the head
    than it should be. Inside the bank that misplacement is harmless, because the closed
    leaf pairs there already stop everything; it is only beyond the bank that this slab is
    doing the work alone.

    It is set to the *fan*, never to the field, and the fan clears the field by at least
    `FAN_MARGIN_MIN` per side, so it never touches a penumbra.

    Second stated approximation: each slab is only as thick as the space between the source
    plane and the leaves allows, 5.25 cm, which transmits about 1 % of this spectrum. The
    fixed collimator it stands in for is several times thicker, so **leakage beyond the
    bank is over-predicted**. That matters to the extra-focal component and to far
    out-of-field dose, and to nothing reported in the tables below. Measured on the
    geometry along the leaf-travel-perpendicular axis, a 2 x 2 cm field transmits 0.12 %
    in the 0.5 cm band where the stagger leaves only one bank closed (which is real), 0 %
    from there out to the bank edge at 14 cm, and 0.9 % beyond it, which is this. Along
    leaf travel it is 0 % everywhere past the tips, since a leaf body has no outer end.
    """
    return JawPair(
        axis=axis,
        z_top=z_range[0],
        z_bottom=z_range[1],
        edge_neg=float(project_between_planes(-half_iso, SAD, sum(z_range) / 2.0)),
        edge_pos=float(project_between_planes(half_iso, SAD, sum(z_range) / 2.0)),
        material=DEVICE_MATERIAL,
        density=DEVICE_DENSITY,
        focused=True,
    )


def beam_limiting_stack(field_x: float, field_y: float) -> BeamLimitingStack:
    """Build the jawless dual-layer head for one rectangular field stated at the isocentre."""
    return BeamLimitingStack(
        frame=BeamFrame(origin=(0.0, 0.0, 0.0)),
        devices=(
            backup_collimator("u", BACKUP_U_Z, 0.5 * field_x + fan_margin(field_x)),
            backup_collimator("v", BACKUP_V_Z, 0.5 * field_y + fan_margin(field_y)),
            leaf_bank(MLC_PROXIMAL_Z, MLC_PROXIMAL_PAIRS, field_x, field_y, park_sign=+1.0),
            leaf_bank(MLC_DISTAL_Z, MLC_DISTAL_PAIRS, field_x, field_y, park_sign=-1.0),
        ),
    )


# A quick look at what that produced for the reference field, and at the stagger.
for device in beam_limiting_stack(REFERENCE_FIELD, REFERENCE_FIELD).devices:
    if not isinstance(device, MLC):
        at_iso = abs(device.edge_pos) * SAD / (0.5 * (device.z_top + device.z_bottom))
        print(
            f"  backup   z = {device.z_top:5.2f} .. {device.z_bottom:5.2f} cm, "
            f"{device.axis} axis at +-{at_iso:.2f} cm at iso (the fan, not the field)"
        )
        continue
    z_mid = 0.5 * (device.z_top + device.z_bottom)
    at_iso = np.asarray(device.leaf_edges_v) * SAD / z_mid
    tips = zip(device.tips_neg, device.tips_pos, strict=True)
    opened = [i for i, (neg, pos) in enumerate(tips) if neg < pos]
    print(
        f"  MLC z = {device.z_top:5.2f} .. {device.z_bottom:5.2f} cm, "
        f"{device.n_pairs} strips built, {len(opened)} open; boundaries at iso "
        f"{at_iso[0]:+.2f} .. {at_iso[-1]:+.2f} cm, open span "
        f"{at_iso[opened[0]]:+.2f} .. {at_iso[opened[-1] + 1]:+.2f} cm"
    )
for _label, _field in (("reference", REFERENCE_FIELD), ("largest", 0.1 * DIAGONAL_FIELD_MM)):
    _half = 0.5 * _field
    print(
        f"  tip retraction at the {_label} {_field:g} cm field, proximal / distal: tangent "
        f"{10 * tangent_offset(_half, 0.5 * sum(MLC_PROXIMAL_Z)):.3f} / "
        f"{10 * tangent_offset(_half, 0.5 * sum(MLC_DISTAL_Z)):.3f} mm, "
        f"radiation edge {10 * radiation_edge_correction(_half, MLC_PROXIMAL_Z):+.3f} / "
        f"{10 * radiation_edge_correction(_half, MLC_DISTAL_Z):+.3f} mm"
    )

# %% [markdown]
# ## A consistent air path
#
# Build-up dose is dominated by contaminant electrons born in the air above the phantom,
# so the air path is pinned to one length for every simulation here — the longest that
# fits in the shallowest setup, which is the SSD 90 profile geometry. Above it is vacuum,
# which contributes nothing and can therefore differ between setups. The path is split
# into a transported part (`AIR_GAP` centimetres of real air voxels) and the pre-solve's
# analytic column above it, which reaches further for the same cost.
#
# The two setups here differ by only 5 cm of SSD, so this matters less than it did in the
# flattened notebook (where the TPR runs moved the surface by 20 cm). It is kept because
# the output factors and the depth doses are read against each other.

# %%
STACK_EXIT = beam_limiting_stack(REFERENCE_FIELD, REFERENCE_FIELD).exit_z
SHALLOWEST_SSD = min(PROFILE_SSD, OUTPUT_FACTOR_SSD)
AIR_PATH = SHALLOWEST_SSD - STACK_EXIT
# Quantized *down* to the voxel grid: the phantoms lay out their air as
# `round(TRANSPORTED_AIR / SPACING)` voxels, and a value that is not a multiple of
# SPACING can round up and put the grid ceiling above the device exit.
TRANSPORTED_AIR = (int(min(AIR_GAP, AIR_PATH) / SPACING)) * SPACING
PRESOLVE_AIR = AIR_PATH - TRANSPORTED_AIR
if abs(TRANSPORTED_AIR / SPACING - round(TRANSPORTED_AIR / SPACING)) > 1.0e-9:
    raise ValueError(
        f"transported air {TRANSPORTED_AIR} cm is not a multiple of the {SPACING} cm voxel;"
        " the phantom grid will round up past the device exit"
    )
if AIR_PATH <= 0.0:
    raise ValueError(
        f"the shallowest phantom surface (SSD {SHALLOWEST_SSD:g} cm) is at or above the "
        f"device exit ({STACK_EXIT:g} cm); there is no room for the beam to leave the head"
    )
print(
    f"air path {AIR_PATH:.1f} cm in every setup: {TRANSPORTED_AIR:.1f} cm transported"
    f" + {PRESOLVE_AIR:.1f} cm pre-solved, vacuum above"
)

# %% [markdown]
# ## The source
#
# Photons are born on a plane above the leaves, aimed from a Gaussian focal spot, over a
# rectangle that is the field plus `fan_margin` per side projected back from the
# isocentre — so every field is simulated with the same fluence per unit area at the
# isocentre plane, which is what makes output factors comparable across fields. Each
# history's weight carries the FFF cone, and the area normalization is divided by the
# on-axis fluence so "per unit incident fluence" still means *per unit fluence on the
# axis*.


# %%
def head_input_source(field_x: float, field_y: float, clip: Any = None) -> tuple[Any, float]:
    """Give the raw source for one field and its aperture area at the isocentre.

    With a phase space driving the head the source is that file restricted to the records
    aimed into this field's fan, and the area is the fan rectangle -- the same quantity the
    analytic route reports, so the history allocation below is identical either way.
    """
    if PHSP is not None:
        if clip is None:
            raise ValueError("the phase-space route needs the field's fan clip")
        return clip.photons, clip.area_iso
    # Per axis, not from the larger side: a 1 x 28 field wants a 1 cm field's margin in x
    # and a 28 cm field's in y, and taking the larger for both would put three quarters of
    # its histories where nothing can pass.
    width_x_iso = field_x + 2.0 * fan_margin(field_x)
    width_y_iso = field_y + 2.0 * fan_margin(field_y)
    geometry: dict[str, Any] = dict(
        spectrum=SPECTRUM,
        focal_point=(0.0, 0.0, 0.0),
        center=(0.0, 0.0, EMISSION_Z),
        width_u=float(project_between_planes(width_x_iso, SAD, EMISSION_Z)),
        width_v=float(project_between_planes(width_y_iso, SAD, EMISSION_Z)),
        sigma_u=SPOT_SIGMA,
        sigma_v=SPOT_SIGMA if SPOT_SIGMA_V is None else SPOT_SIGMA_V,
    )
    area = width_x_iso * width_y_iso
    source: Source
    if FLUENCE is None:
        source, normalization = GaussianSpotBeamSource(**geometry), area
    else:
        source = PrimaryFluenceBeamSource(fluence=FLUENCE, **geometry)
        normalization = area / FLUENCE.at_radius(0.0)
    if SHELL_SPECTRA:
        source = ShellSpectrumSource(source, SHELL_SPECTRA)
    return source, normalization


# Both allocation rules above are quoted relative to the reference square field.
REFERENCE_AREA = (
    4.0 * (0.5 * REFERENCE_FIELD + fan_margin(REFERENCE_FIELD)) ** 2
    if PHSP is not None
    else head_input_source(REFERENCE_FIELD, REFERENCE_FIELD)[1]
)
REFERENCE_DILUTION = REFERENCE_AREA / measuring_voxels(REFERENCE_FIELD, REFERENCE_FIELD)


# %% [markdown]
# ## The extra-focal source
#
# The same planar geometry with the focal spot moved down to the collimator region and
# widened. A point in the field sees only the part of that extended source the leaves
# leave unobstructed, so opening the collimator admits more of it and head scatter rises
# with field size — with no scattering geometry modelled anywhere. The emitting square is
# the primary collimator's opening and is therefore field-independent; the leaves cut it.


# %%
def on_axis_emitted_area(source: Any, n: int = 4_000_000, half: float = 1.0) -> float:
    """Effective isocentre-plane area per emitted history, measured from the source.

    A source's on-axis emitted fluence is ``histories / this area``, the normalization
    every output factor is quoted against. Measured rather than assumed: the naive
    ``(emitting square)^2`` is only right for rays emanating from the target, and the
    extra-focal rays come from far below it, so they diverge faster and cover several
    times that area. Assuming otherwise inflates the fitted weight by exactly that ratio.
    """
    columns = source.sample_batch(SEED, 0, n)
    z = columns["z"].astype(np.float64)
    uz = columns["uz"].astype(np.float64)
    t = (SAD - z) / uz
    x = columns["x"].astype(np.float64) + t * columns["ux"].astype(np.float64)
    y = columns["y"].astype(np.float64) + t * columns["uy"].astype(np.float64)
    inside = (np.abs(x) <= half) & (np.abs(y) <= half)
    landed = float(columns["weight"].astype(np.float64)[inside].sum())
    if landed <= 0.0:
        raise ValueError("no sampled primary reached the central square; cannot normalize")
    return 4.0 * half * half * n / landed


def extrafocal_source() -> tuple[Any, float]:
    """Build the wide-Gaussian head-scatter source and measure its isocentre area."""
    width_plane = float(project_between_planes(EXTRAFOCAL_FIELD, SAD, EMISSION_Z))
    source = GaussianSpotBeamSource(
        spectrum=SPECTRUM,
        focal_point=(0.0, 0.0, EXTRAFOCAL_Z),
        center=(0.0, 0.0, EMISSION_Z),
        width_u=width_plane,
        width_v=width_plane,
        sigma_u=EXTRAFOCAL_SIGMA,
        sigma_v=EXTRAFOCAL_SIGMA,
    )
    return source, on_axis_emitted_area(source)


def extrafocal_histories(ratio: float, primary: int) -> int:
    """Extra-focal histories for one replicate, given its dilution `ratio`.

    **Not** a fixed fraction of the primary count, which is what the flattened notebook
    used and what fails here. The extra-focal fan is spread over `ratio` times the
    isocentre area the primary's is, so the same number of histories puts `ratio` times
    fewer photons through the aperture — and `ratio` runs from about 8 at the largest
    field to over a thousand at 1 x 1 cm^2, where a flat fraction was measured to put
    *zero* extra-focal events in the measuring voxel and to report the component as
    exactly 0.

    Scaling the allocation with `ratio` levels that out. The scale is set so a 10 x 10
    field keeps roughly the flattened notebook's 0.2, and the cap keeps a 1 x 1 field —
    where the ratio would ask for eight times the primary count — from dominating the
    wall clock for a component that is a few per cent of its dose. These are cheap
    histories: almost all of them are aimed outside the collimator and die in the
    pre-solve.
    """
    wanted = int(EXTRAFOCAL_HISTORY_SCALE * ratio * primary)
    return max(1, min(wanted, int(EXTRAFOCAL_HISTORY_MAX * primary)))


EXTRAFOCAL_RAW, EXTRAFOCAL_AREA = (
    (None, 0.0) if (EXTRAFOCAL_WEIGHT is None or PHSP is not None) else extrafocal_source()
)
if EXTRAFOCAL_WEIGHT is None:
    print("extra-focal source: off (and the refit column will not be written)")
else:
    print(
        f"extra-focal source: w = {EXTRAFOCAL_WEIGHT:.3f} (placeholder), sigma "
        f"{EXTRAFOCAL_SIGMA:g} cm at z = {EXTRAFOCAL_Z:g} cm, effective "
        f"{EXTRAFOCAL_AREA:.0f} cm^2 at the isocentre plane per history"
    )


def contaminant_spectrum() -> Spectrum:
    """One narrow bin at `CONTAMINANT_ENERGY`: the mono-energetic electron line."""
    return Spectrum(edges=(CONTAMINANT_ENERGY - 0.025, CONTAMINANT_ENERGY + 0.025), weights=(1.0,))


def contaminant_beam_source(field_x: float, field_y: float) -> tuple[Any, float]:
    """Electrons on the primary's own fan: the aperture-proportional floor term.

    A flat rectangle aimed from the focal spot -- no fluence cone and no spectral
    shells, which belong to the photons -- so its analytic isocentre area is exact.
    """
    width_x_iso = field_x + 2.0 * fan_margin(field_x)
    width_y_iso = field_y + 2.0 * fan_margin(field_y)
    source = ElectronSource(
        GaussianSpotBeamSource(
            spectrum=contaminant_spectrum(),
            focal_point=(0.0, 0.0, 0.0),
            center=(0.0, 0.0, EMISSION_Z),
            width_u=float(project_between_planes(width_x_iso, SAD, EMISSION_Z)),
            width_v=float(project_between_planes(width_y_iso, SAD, EMISSION_Z)),
            sigma_u=SPOT_SIGMA,
            sigma_v=SPOT_SIGMA if SPOT_SIGMA_V is None else SPOT_SIGMA_V,
        )
    )
    return source, width_x_iso * width_y_iso


def contaminant_head_source() -> tuple[Any, float]:
    """Electrons from the extra-focal plane: the saturating wide term."""
    width_plane = float(project_between_planes(EXTRAFOCAL_FIELD, SAD, EMISSION_Z))
    source = ElectronSource(
        GaussianSpotBeamSource(
            spectrum=contaminant_spectrum(),
            focal_point=(0.0, 0.0, EXTRAFOCAL_Z),
            center=(0.0, 0.0, EMISSION_Z),
            width_u=width_plane,
            width_v=width_plane,
            sigma_u=EXTRAFOCAL_SIGMA,
            sigma_v=EXTRAFOCAL_SIGMA,
        )
    )
    return source, on_axis_emitted_area(source)


def contaminant_histories(ratio: float, primary: int) -> int:
    """Electron histories for one replicate; the extra-focal dilution rule, own scale."""
    wanted = int(CONTAMINANT_HISTORY_SCALE * ratio * primary)
    return max(1, min(wanted, int(CONTAMINANT_HISTORY_MAX * primary)))


CONTAMINANT_HEAD_RAW, CONTAMINANT_HEAD_AREA = (
    (None, 0.0) if (CONTAMINANT_WEIGHTS is None or PHSP is not None) else contaminant_head_source()
)
if CONTAMINANT_WEIGHTS is None:
    print("contaminant electrons: off (and the refit columns will not be written)")
else:
    print(
        f"contaminant electrons: {CONTAMINANT_ENERGY:g} MeV, w = "
        f"({CONTAMINANT_WEIGHTS[0]:.4f} beam, {CONTAMINANT_WEIGHTS[1]:.4f} head), "
        f"head plane effective {CONTAMINANT_HEAD_AREA:.0f} cm^2 at iso per history"
    )

# %% [markdown]
# ## The phantom
#
# Air over water, centred on the beam axis, with an odd number of lateral voxels so that
# one voxel sits on the axis. Each lateral axis is sized to its own field, which matters
# for the output-factor matrix: a 1 x 28 field needs no more water in x than a 1 x 1.


# %%
def water_phantom(
    field_x: float,
    field_y: float,
    ssd: float,
    phantom_depth: float,
    lateral_margin: float,
) -> VoxelGrid:
    """Build the air-over-water phantom for one rectangular field and placement."""

    def lateral_voxels(field: float) -> int:
        half = 0.5 * field * (ssd + phantom_depth) / SAD + lateral_margin
        n = math.ceil(2.0 * half / SPACING)
        return n + (n % 2 == 0)  # odd, so a voxel is centred on the axis

    n_x, n_y = lateral_voxels(field_x), lateral_voxels(field_y)
    n_air = round(TRANSPORTED_AIR / SPACING)
    shape = (n_x, n_y, n_air + round(phantom_depth / SPACING))
    density = np.ones(shape, dtype=np.float64)
    material = np.full(shape, WATER, dtype=np.int32)
    if n_air:
        density[:, :, :n_air] = AIR_DENSITY
        material[:, :, :n_air] = AIR_MATERIAL
    return VoxelGrid(
        shape=shape,
        spacing=(SPACING, SPACING, SPACING),
        density=density,
        material=material,
        origin=(-0.5 * n_x * SPACING, -0.5 * n_y * SPACING, ssd - n_air * SPACING),
    )


def point_scoring_grid(grid: VoxelGrid, ssd: float, depth: float, half: float) -> ScoringGrid:
    """Build a small scored box around the measurement point, on the transport lattice.

    The output-factor runs read one on-axis number, so accumulating the whole phantom
    costs memory and host time for nothing. Deposits outside the box go to the engine's
    unscored ledger rather than being clamped into an edge voxel, so the box is exact
    where it exists — the transport, and therefore the scatter that reaches the point,
    is unchanged.
    """
    n = math.ceil(2.0 * half / SPACING)
    n += n % 2 == 0  # odd, so a voxel is centred on the axis
    n_z = math.ceil(2.0 * half / SPACING)
    origin_z = ssd + depth - 0.5 * n_z * SPACING
    # Snap onto the transport lattice so the mass rebin is a clean 1:1 overlap.
    origin_z = grid.origin[2] + round((origin_z - grid.origin[2]) / SPACING) * SPACING
    return ScoringGrid.rebin(
        grid,
        shape=(n, n, n_z),
        spacing=(SPACING, SPACING, SPACING),
        origin=(-0.5 * n * SPACING, -0.5 * n * SPACING, origin_z),
    )


# %% [markdown]
# ## Reading a dose distribution
#
# A `Scan` is one dose array plus what is needed to interpret it. `grid` is whatever the
# dose was accumulated on — the transport grid for the profile runs, the small scoring
# box for the output-factor runs — since both carry the origin, shape and spacing these
# readers need.
#
# Curves are averaged over a column a fifth of the field wide, capped at 1.5 cm: across a
# square field that is nearly distortion-free (the edges are straight, so the orthogonal
# direction is flat wherever the profile is not) and it buys a factor of several in
# variance. The output factor deliberately does not use that column — a reported dose is
# a point measurement, a plotted curve is not.


# %%
class Scan(NamedTuple):
    """One dose distribution and the geometry needed to read it."""

    field_x: float
    field_y: float
    ssd: float
    depth: float
    grid: Any  # VoxelGrid or ScoringGrid; both carry origin / shape / spacing
    dose: np.ndarray


def coordinates(grid: Any, axis: int) -> np.ndarray:
    """Voxel-centre coordinates along one grid axis (cm, engine frame)."""
    return grid.origin[axis] + (np.arange(grid.shape[axis]) + 0.5) * grid.spacing[axis]


def within(grid: Any, axis: int, half: float) -> np.ndarray:
    """Mask of voxels whose centres lie within ``half`` of the axis."""
    return np.abs(coordinates(grid, axis)) <= half + 1.0e-9


def column_half(scan: Scan, axis: int) -> float:
    """Half-width of the averaging column used for curves (never for the ROI)."""
    field = scan.field_y if axis == 1 else scan.field_x
    return max(min(0.2 * field, 1.5), 0.4 * scan.grid.spacing[axis])


def roi_selector(
    grid: Any, ssd: float, depth: float, field_x: float, field_y: float, half: float | None = None
) -> tuple[Any, ...]:
    """Select the on-axis measuring volume: the index tuple `roi_dose` averages over.

    A ~0.5 cm cube, chamber-like, for fields that can hold one; below 2 cm the single
    central voxel, which is then volume-averaged over one voxel and noisier. See the
    flattened notebook's output-factor section for what that costs below ~15 mm: this
    is a small-field detector-response regime and the model is not a check there.

    Exposed rather than inlined because the history allocation has to know how big this
    volume is before the field is run — see `measuring_voxels`.
    """
    if half is None:
        half = 0.25 if min(field_x, field_y) >= 2.0 else 0.4 * grid.spacing[0]
    z = coordinates(grid, 2) - ssd
    in_z = np.abs(z - depth) <= half + 1.0e-9
    if not in_z.any():
        in_z = np.zeros_like(z, dtype=bool)
        in_z[int(np.argmin(np.abs(z - depth)))] = True
    return np.ix_(within(grid, 0, half), within(grid, 1, half), in_z)


def roi_dose(scan: Scan, depth: float, half: float | None = None) -> float:
    """Mean dose in the on-axis measuring volume at a depth; see `roi_selector`."""
    selector = roi_selector(scan.grid, scan.ssd, depth, scan.field_x, scan.field_y, half)
    return float(scan.dose[selector].mean())


def depth_dose(scan: Scan) -> tuple[np.ndarray, np.ndarray]:
    """Central-axis depth and dose, averaged over the on-axis column."""
    depth = coordinates(scan.grid, 2) - scan.ssd
    inside = depth >= 0.0
    columns = np.ix_(
        within(scan.grid, 0, column_half(scan, 0)),
        within(scan.grid, 1, column_half(scan, 1)),
        inside,
    )
    return depth[inside], scan.dose[columns].mean(axis=(0, 1))


def plane_at_depth(scan: Scan, depth: float) -> np.ndarray:
    """Transverse dose plane at a depth, linearly interpolated between voxel planes.

    The measured depths are not multiples of the voxel (1.3 cm is not), and taking the
    nearest plane would misplace a build-up profile by up to half a voxel. Interpolating
    costs nothing and removes the question.
    """
    z = coordinates(scan.grid, 2) - scan.ssd
    if depth <= z[0]:
        return np.asarray(scan.dose[:, :, 0])
    if depth >= z[-1]:
        return np.asarray(scan.dose[:, :, -1])
    k = int(np.searchsorted(z, depth) - 1)
    t = (depth - z[k]) / (z[k + 1] - z[k])
    return np.asarray((1.0 - t) * scan.dose[:, :, k] + t * scan.dose[:, :, k + 1])


def lateral_profile(scan: Scan, axis: int, depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Off-axis coordinate and dose along one principal axis at a depth."""
    plane = plane_at_depth(scan, depth)
    other = 1 - axis
    mask = within(scan.grid, other, column_half(scan, other))
    profile = plane[:, mask].mean(axis=1) if axis == 0 else plane[mask, :].mean(axis=0)
    return coordinates(scan.grid, axis), profile


def sample_plane(values: np.ndarray, grid: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinearly sample a transverse slice at engine-frame (x, y) points (cm)."""
    fx = np.clip((x - grid.origin[0]) / grid.spacing[0] - 0.5, 0.0, grid.shape[0] - 1.0)
    fy = np.clip((y - grid.origin[1]) / grid.spacing[1] - 0.5, 0.0, grid.shape[1] - 1.0)
    i = np.minimum(fx.astype(int), grid.shape[0] - 2)
    j = np.minimum(fy.astype(int), grid.shape[1] - 2)
    tx, ty = fx - i, fy - j
    return (
        values[i, j] * (1.0 - tx) * (1.0 - ty)
        + values[i + 1, j] * tx * (1.0 - ty)
        + values[i, j + 1] * (1.0 - tx) * ty
        + values[i + 1, j + 1] * tx * ty
    )


def both_diagonals(scan: Scan, depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Radial coordinate and dose along the field diagonals, at a depth.

    Deliberately *not* averaged over a perpendicular band the way the axis scans are:
    across a diagonal a square field is not flat — the band runs in and out through the
    corner — and averaging one measurably moves the 50 % point inward from the geometric
    corner distance. Averaging the *two* diagonals instead is free and distortion-less,
    since a square field is four-fold symmetric.

    The sampled radius runs to `sqrt(2)` times the phantom half-width, not to a half of
    it: the legs are what has to stay inside the grid, so the radius may be longer than
    the half-width by exactly that factor.
    """
    plane = plane_at_depth(scan, depth)
    grid = scan.grid
    leg_max = 0.5 * min(grid.shape[0] * grid.spacing[0], grid.shape[1] * grid.spacing[1])
    leg_max -= grid.spacing[0]
    radius_max = leg_max * math.sqrt(2.0)
    radius = np.linspace(-radius_max, radius_max, 2 * int(radius_max / grid.spacing[0]) + 1)
    leg = radius / math.sqrt(2.0)
    positive = sample_plane(plane, grid, leg, leg)
    negative = sample_plane(plane, grid, leg, -leg)
    return radius, 0.5 * (positive + negative)


# %% [markdown]
# ## Commissioning metrics from one distribution
#
# Two details that matter more than they look.
#
# The peak of a depth-dose curve is taken from a **parabola fitted through the five
# samples around the largest one**, not from `argmax`. On a Monte Carlo curve `argmax`
# lands on whichever voxel fluctuated highest, which biases `dmax` around and the
# normalizing dose *upward*, dragging every PDD down by several percent at demo
# statistics.
#
# **The field edge is the inflection point, not 50 % of the axis.** This is the one place
# where an unflattened beam cannot borrow the flattened convention, and it is not a small
# correction. On this machine's 28 x 28 field the profile has already fallen to about 60 %
# of the axis *inside* the field, so the classical construction puts the 80 % level 7 cm
# in from the edge and reports a penumbra of 40 mm and a field 4 mm too narrow — numbers
# that describe the cone, not the collimator. The FFF convention (Fogliata et al., *Med.
# Phys.* **39** (2012) 6455; IEC 60976 as amended for unflattened beams) takes the field
# edge at the **steepest point** of each shoulder and renormalizes the profile so that
# point reads 50 %; widths and 80/20 penumbrae are then read off the renormalized curve
# and mean what they mean on a flattened beam.
#
# Both are computed. `width` and `penumbra` are the inflection construction and are what
# the tables print; `width_ax` and `penumbra_ax` are the classical ones and are kept in the
# CSV, because a measured data set may well have been analysed either way and the two only
# agree at small fields. If the measured penumbrae were taken with a flattened-beam
# analysis at 28 x 28, `penumbra_ax` is the column to compare them against.


# %%
def crossing(x: np.ndarray, y: np.ndarray, level: float, rising: bool) -> float:
    """Interpolated abscissa where a profile crosses a level; NaN if it never does."""
    above = y >= level
    edges = np.flatnonzero(~above[:-1] & above[1:] if rising else above[:-1] & ~above[1:])
    if edges.size == 0:
        return float("nan")
    i = int(edges[0] if rising else edges[-1])
    y0, y1 = float(y[i]), float(y[i + 1])
    if y1 == y0:
        return float(x[i])
    return float(x[i] + (level - y0) * (x[i + 1] - x[i]) / (y1 - y0))


def inflection_point(offsets: np.ndarray, values: np.ndarray, side: int) -> float:
    """Off-axis position of the steepest point of one shoulder (NaN if there is none).

    The extremum of the numerical gradient, refined below one voxel by a parabola through
    it and its two neighbours — the same sub-sample trick `measure_profiles` uses for
    `dmax`, and for the same reason: on a Monte Carlo curve the raw argmax lands on
    whichever sample fluctuated steepest.
    """
    centre = int(np.argmin(np.abs(offsets)))
    gradient = np.gradient(values, offsets)
    if side < 0:
        window = slice(1, max(2, centre))
        local = np.argmax(gradient[window])
    else:
        window = slice(min(centre + 1, gradient.size - 2), gradient.size - 1)
        local = np.argmin(gradient[window])
    i = int(window.start + local)
    if not 1 <= i < gradient.size - 1:
        return float("nan")
    g0, g1, g2 = (float(gradient[i + k]) for k in (-1, 0, 1))
    denominator = g0 - 2.0 * g1 + g2
    shift = 0.5 * (g0 - g2) / denominator if denominator != 0.0 else 0.0
    step = float(offsets[1] - offsets[0])
    return float(offsets[i] + max(-1.0, min(1.0, shift)) * step)


def profile_metrics(offsets: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Field width, penumbra, symmetry and off-axis ratios from one profile.

    `width` and `penumbra` follow the FFF convention: the edges are the two inflection
    points and the profile is renormalized so their (mean) dose reads 50 %, after which
    the 80/20 penumbra is the ordinary construction on that curve. `width_ax` and
    `penumbra_ax` are the classical on-axis-normalized versions, kept for comparison
    against a data set analysed the flattened way. `oar_50` and `oar_80` are the off-axis
    ratios at half and at 80 % of the field half-width — the shape numbers an unflattened
    beam is compared on, since flatness in the flattened sense is not defined for it.
    """
    keys = ("width", "penumbra", "width_ax", "penumbra_ax", "symmetry", "oar_50", "oar_80")
    centre = float(np.interp(0.0, offsets, values))
    if centre <= 0.0 or values.size < 5:
        return dict.fromkeys(keys, float("nan"))

    def construction(level: float) -> tuple[float, float]:
        """Width and mean 80/20 penumbra against a stated 100 % level."""
        edges = {
            f"{s}{pct}": crossing(offsets, values, 0.01 * pct * level, rising=s == "l")
            for pct in (20, 50, 80)
            for s in "lr"
        }
        return (
            edges["r50"] - edges["l50"],
            0.5 * ((edges["l80"] - edges["l20"]) + (edges["r20"] - edges["r80"])),
        )

    width_ax, penumbra_ax = construction(centre)
    left, right = inflection_point(offsets, values, -1), inflection_point(offsets, values, +1)
    if not (np.isfinite(left) and np.isfinite(right)) or right <= left:
        width, penumbra, level = float("nan"), float("nan"), float("nan")
    else:
        # The renormalization level: twice the mean dose at the two inflection points, so
        # that both read 50 % of it by construction.
        level = float(np.interp(left, offsets, values) + np.interp(right, offsets, values))
        width, penumbra = construction(level)
    half_width = 0.5 * (width if np.isfinite(width) else width_ax)

    ratios: dict[str, float] = {}
    for fraction, name in ((0.5, "oar_50"), (0.8, "oar_80")):
        if not np.isfinite(half_width):
            ratios[name] = float("nan")
            continue
        at = fraction * half_width
        ratios[name] = float(
            0.5 * (np.interp(at, offsets, values) + np.interp(-at, offsets, values)) / centre
        )

    band = np.abs(offsets) <= 0.8 * half_width if np.isfinite(half_width) else np.zeros(0, bool)
    if np.count_nonzero(band) < 3:
        symmetry = float("nan")
    else:
        mirrored = np.interp(-offsets[band], offsets, values)
        symmetry = 100.0 * float(np.max(np.abs(values[band] - mirrored)) / centre)
    return {
        "width": width,
        "penumbra": penumbra,
        "width_ax": width_ax,
        "penumbra_ax": penumbra_ax,
        "symmetry": symmetry,
        **ratios,
    }


def measure_profiles(scan: Scan) -> dict[str, float]:
    """Every profile metric of one distribution, keyed `<name>_<axis>_d<depth>`."""
    metrics: dict[str, float] = {}
    depth_cm, curve = depth_dose(scan)
    peak = int(np.argmax(curve))
    lo, hi = max(0, peak - 2), min(curve.size, peak + 3)
    dmax, reference = float(depth_cm[peak]), float(curve[peak])
    if hi - lo >= 3:
        a, b, c = np.polyfit(depth_cm[lo:hi], curve[lo:hi], 2)
        vertex = -b / (2.0 * a) if a < 0.0 else float("nan")
        if depth_cm[lo] <= vertex <= depth_cm[hi - 1]:
            dmax, reference = float(vertex), float(a * vertex * vertex + b * vertex + c)
    metrics["dmax"] = dmax
    for at in (5.0, 10.0, 20.0, 30.0):
        metrics[f"pdd{at:g}"] = 100.0 * float(np.interp(at, depth_cm, curve)) / reference
    for depth in PROFILE_DEPTHS:
        for axis, name in ((0, "x"), (1, "y")):
            for key, value in profile_metrics(*lateral_profile(scan, axis, depth)).items():
                metrics[f"{key}_{name}_d{depth:g}"] = value
    return metrics


def mean_sem(values: list[float]) -> tuple[float, float]:
    """Mean and standard error of the mean over independent replicate values."""
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size < 2:
        return (float(finite[0]) if finite.size else float("nan")), float("nan")
    return float(finite.mean()), float(finite.std(ddof=1) / math.sqrt(finite.size))


# %% [markdown]
# ## Simulating one field
#
# The replicates share nothing but the geometry: each reseeds the head model *and* the
# transport. Their pooled mean is the reported dose; their spread is the reported
# uncertainty. Reseeding the pre-solve matters — replicates sharing one phase space would
# share its latent variance and report a spread that is too small — and it is also what
# lets each replicate cheaply resample a small phase space many times.


# %%
class FieldResult(NamedTuple):
    """One field: the pooled scan, its per-replicate metrics, and its normalization.

    ``component_roi`` carries each source component's own dose per unit on-axis emitted
    fluence at the measurement point, *before* the extra-focal weight is applied.
    Transport is linear in the source, so that is what lets the weight be refitted from
    the saved table without re-running anything.
    """

    scan: Scan
    stack: BeamLimitingStack
    replicates: list[dict[str, float]]
    area_iso: float
    seconds: float
    component_roi: dict[str, float]

    def spread(self, key: str) -> tuple[float, float]:
        """Mean and standard error over the replicates for one metric."""
        return mean_sem([replicate[key] for replicate in self.replicates])


def transport_component(
    engine: Any,
    raw: Any,
    stack: BeamLimitingStack,
    grid: VoxelGrid,
    scoring: ScoringGrid | None,
    n: int,
    reuse: float,
    seed: int,
    model: str | None = None,
) -> tuple[np.ndarray, float]:
    """Run the head model and transport for one source component and one replicate.

    Returns the dose per emitted history of *that component* and the seconds spent in the
    head model. Every component goes through this identical path, which is what makes
    their doses linearly combinable further down. `reuse` is how many transport histories
    each pre-solved head history is resampled by; see `phase_space_reuse`. `model`
    overrides `HEAD_MODEL` for this component: the electron components pass
    "attenuation", because the pre-solve is photon physics and must never see them.
    """
    mark = time.perf_counter()
    if (model or HEAD_MODEL) == "attenuation":
        source: Any = CollimatedSource(raw, stack, XS)
    else:
        exit_z = float(grid.origin[2])
        arguments = {
            "stack": stack,
            "cross_sections": XS,
            "n_histories": min(n, max(PRESOLVE_HISTORIES_MIN, int(n / reuse))),
            "seed": seed,
            "exit_z": exit_z,
            # Fixed length, not "from the device exit": the air path must be the same in
            # every setup or the build-up regions are not comparable.
            "air": (
                AirColumn(
                    z_start=exit_z - PRESOLVE_AIR,
                    z_end=exit_z,
                    material=AIR_MATERIAL,
                    density=AIR_DENSITY,
                )
                if PRESOLVE_AIR > 0.0
                else None
            ),
            "mode": "first_compton",
        }
        if WARP_DEVICE is not None:
            # Runs as a Warp kernel and stays on the device: no copy back.
            source = presolve_head_device(
                raw, device=WARP_DEVICE, return_device_source=True, **arguments
            )
        else:
            source = presolve_head(raw, **arguments)
    head_seconds = time.perf_counter() - mark
    result = engine.run(source, n_histories=n, n_batches=1, seed=seed + 1, scoring_grid=scoring)
    return np.asarray(result.dose), head_seconds


def simulate(
    field_x: float,
    field_y: float,
    ssd: float,
    depth: float,
    *,
    phantom_depth: float,
    lateral_margin: float,
    base_histories: int,
    replicates: int,
    history_boost: float = 1.0,
    scoring_half: float | None = None,
    measure: Any = None,
    announce: bool = True,
) -> FieldResult:
    """Transport one rectangular field as independent replicates and pool them.

    With an extra-focal component the two sources are transported *separately* and
    combined as ``(1 - w) D_primary + w (A_e / A_p) D_extrafocal``. Transport is linear in
    the source, so that is exact rather than an approximation — and it keeps each
    component's own dose available for refitting `w`.

    `base_histories` is the per-replicate count *at the reference field*; this field's own
    count and its phase-space reuse both follow from its fan area (see the statistics
    cell). `scoring_half` restricts accumulation to a box around the measurement point;
    `measure` is the per-replicate metric function, or None to record only the ROI dose,
    which is all the output-factor matrix reads.
    """
    grid = water_phantom(field_x, field_y, ssd, phantom_depth, lateral_margin)
    scoring = None if scoring_half is None else point_scoring_grid(grid, ssd, depth, scoring_half)
    read_on = grid if scoring is None else scoring
    stack = beam_limiting_stack(field_x, field_y)
    clip = None if PHSP is None else fan_clip(field_x, field_y)
    raw, area_iso = head_input_source(field_x, field_y, clip)
    if WARP_DEVICE is not None:
        engine: Any = WarpEngine(grid=grid, cross_sections=XS, device=WARP_DEVICE)
    else:
        engine = ReferenceEngine(grid=grid, cross_sections=XS, rng=HostRNG())

    # With a phase space the head scatter is already in the particles, so the analytic
    # extra-focal term is off whatever it is configured to.
    weight = 0.0 if (EXTRAFOCAL_WEIGHT is None or PHSP is not None) else float(EXTRAFOCAL_WEIGHT)
    # Ratio that puts the extra-focal dose on the primary's per-history footing; the
    # (1 - w) and w factors then make the sum a dose per unit *total* on-axis fluence.
    # It is also exactly the factor by which the extra-focal fan is more dilute at the
    # isocentre, hence the history allocation below.
    ratio = EXTRAFOCAL_AREA / area_iso
    # The measuring volume is known before the run, and the allocation needs it. Checked
    # against the selector actually used, so `measuring_voxels` cannot drift out of sync
    # with `roi_selector` unnoticed.
    voxels = measuring_voxels(field_x, field_y)
    actual = int(np.prod([a.size for a in roi_selector(read_on, ssd, depth, field_x, field_y)]))
    if voxels != actual:
        raise ValueError(
            f"measuring_voxels says {voxels} voxels for a {field_x:g} x {field_y:g} cm "
            f"field but roi_selector selects {actual}; the history allocation is being "
            "computed for a different measuring volume than the one that is read"
        )
    diluted = dilution(area_iso, voxels)
    per_replicate = histories_per_replicate(diluted, base_histories, boost=history_boost)
    extra_n = extrafocal_histories(ratio, per_replicate)
    reuse = phase_space_reuse(diluted)
    extra_reuse = (
        phase_space_reuse(dilution(EXTRAFOCAL_AREA, voxels)) if EXTRAFOCAL_AREA > 0.0 else 1.0
    )

    electron_parts: list[tuple[Any, float, float, float, int, int, str]] = []
    if clip is not None:
        # The file's own electrons, on the file's own footing: their share of the dose is
        # the ratio of the two clipped populations, not a fitted weight.
        if clip.electrons is not None:
            electron_parts = [
                (
                    clip.electrons,
                    1.0,
                    clip.n_electrons / clip.n_photons,
                    1.0,
                    max(1, int(ELECTRON_HISTORY_FRACTION * per_replicate)),
                    700_003,
                    "contaminant_head",
                )
            ]
    elif CONTAMINANT_WEIGHTS is not None:
        beam_raw, beam_area = contaminant_beam_source(field_x, field_y)
        candidates = [
            # (source, area, ratio to the primary footing, weight, histories, seed offset, tag)
            (
                beam_raw,
                beam_area,
                beam_area / area_iso,
                CONTAMINANT_WEIGHTS[0],
                contaminant_histories(beam_area / area_iso, per_replicate),
                700_003,
                "contaminant_beam",
            ),
            (
                CONTAMINANT_HEAD_RAW,
                CONTAMINANT_HEAD_AREA,
                CONTAMINANT_HEAD_AREA / area_iso,
                CONTAMINANT_WEIGHTS[1],
                contaminant_histories(CONTAMINANT_HEAD_AREA / area_iso, per_replicate),
                900_007,
                "contaminant_head",
            ),
        ]
        # A weight fitted to zero stays configured (the record above says why) but is
        # not transported: its column reads 0 by decision, not by accident.
        electron_parts = [part for part in candidates if part[3] > 0.0]

    if announce:
        voxels = int(np.prod(grid.shape))
        extra_note = f" + {extra_n:,} extra-focal" if weight > 0.0 else ""
        electron_note = (
            " + " + "/".join(f"{n_e:,}" for *_, n_e, _, _ in electron_parts) + " e-"
            if electron_parts
            else ""
        )
        print(
            f"  {field_x * 10:5.0f} x {field_y * 10:<5.0f} mm: {grid.shape} "
            f"({voxels / 1e6:.1f} M voxels)"
            + (f", scoring {read_on.shape}" if scoring is not None else "")
            + f", {replicates} x {per_replicate:,}{extra_note}{electron_note} histories, "
            f"{reuse:.0f}x reuse"
        )
    shape = read_on.shape
    pooled = np.zeros(shape, dtype=np.float64)
    per_replicate_metrics: list[dict[str, float]] = []
    component_totals = {
        "primary": 0.0,
        "extrafocal": 0.0,
        "contaminant_beam": 0.0,
        "contaminant_head": 0.0,
    }
    head_seconds = 0.0
    start = time.perf_counter()
    for index in range(replicates):
        seed = SEED + 1013 * index
        primary_dose, spent = transport_component(
            engine, raw, stack, grid, scoring, per_replicate, reuse, seed
        )
        head_seconds += spent
        combined = (1.0 - weight) * primary_dose
        component_totals["primary"] += (
            roi_dose(Scan(field_x, field_y, ssd, depth, read_on, primary_dose), depth) * area_iso
        )
        if weight > 0.0:
            # A separate seed stream: the two components must not share focal-spot draws.
            extra_dose, spent = transport_component(
                engine, EXTRAFOCAL_RAW, stack, grid, scoring, extra_n, extra_reuse, seed + 500_003
            )
            head_seconds += spent
            combined += weight * ratio * extra_dose
            component_totals["extrafocal"] += (
                roi_dose(Scan(field_x, field_y, ssd, depth, read_on, extra_dose), depth)
                * EXTRAFOCAL_AREA
            )
        for raw_e, area_e, ratio_e, weight_e, n_e, offset, tag in electron_parts:
            # Never through the photon pre-solve: clipped by the collimator instead.
            electron_dose, spent = transport_component(
                engine, raw_e, stack, grid, scoring, n_e, 1.0, seed + offset, model="attenuation"
            )
            head_seconds += spent
            combined += weight_e * ratio_e * electron_dose
            component_totals[tag] += (
                roi_dose(Scan(field_x, field_y, ssd, depth, read_on, electron_dose), depth) * area_e
            )
        pooled += combined
        scan = Scan(field_x, field_y, ssd, depth, read_on, combined)
        record = {"roi_dose": roi_dose(scan, depth)}
        if measure is not None:
            record.update(measure(scan))
        per_replicate_metrics.append(record)
    seconds = time.perf_counter() - start
    if announce:
        note = (
            f", {head_seconds:.1f} s of it pre-solving the head" if HEAD_MODEL == "presolve" else ""
        )
        print(f"          {seconds:.1f} s{note}")
    return FieldResult(
        scan=Scan(field_x, field_y, ssd, depth, read_on, pooled / replicates),
        stack=stack,
        replicates=per_replicate_metrics,
        area_iso=area_iso,
        seconds=seconds,
        component_roi={name: total / replicates for name, total in component_totals.items()},
    )


# %% [markdown]
# ## The profile sweep
#
# SSD 90 cm, square fields, the full phantom. Everything in the depth-dose, profile and
# diagonal sets comes from these runs.

# %%
print(f"profile sweep at SSD {PROFILE_SSD:g} cm:")
sweep_start = time.perf_counter()
runs: dict[float, FieldResult] = {
    field: simulate(
        field,
        field,
        PROFILE_SSD,
        PROFILE_DEPTHS[2],
        phantom_depth=PHANTOM_DEPTH,
        lateral_margin=LATERAL_MARGIN,
        base_histories=PER_REPLICATE,
        replicates=REPLICATES,
        measure=measure_profiles,
    )
    for field in PROFILE_FIELDS
}
print(f"  profile sweep: {time.perf_counter() - sweep_start:.1f} s")

if DIAGONAL_FIELD not in runs:
    raise ValueError(
        f"DIAGONAL_FIELD_MM = {DIAGONAL_FIELD_MM} is not in PROFILE_FIELDS_MM; the "
        "diagonals are read off that field's run"
    )

# %% [markdown]
# ## Depth doses
#
# Normalized to each field's own maximum, which is the definition of a PDD. Larger fields
# fall off more slowly: more phantom scatter reaching the axis.

# %%
COLOURS = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(PROFILE_FIELDS)))

print(f"{'field':>7} {'dmax':>6} {'PDD(5)':>8} {'PDD(10)':>8} {'PDD(20)':>8} {'PDD(30)':>8}")
for field in PROFILE_FIELDS:
    pooled = measure_profiles(runs[field].scan)
    print(
        f"{field * 10:6.0f}  {pooled['dmax']:6.2f} {pooled['pdd5']:8.2f} "
        f"{pooled['pdd10']:8.2f} {pooled['pdd20']:8.2f} {pooled['pdd30']:8.2f}"
    )

# %% [markdown]
# ## Field widths and penumbrae
#
# The two directions are different devices here, and this table is where that shows.
# `x` is the leaf-travel direction — rounded ends, a continuous edge. `y` is the leaf-side
# direction — no jaws behind it, and the edge can only sit on a 0.5 cm boundary of the
# staggered pair. A crossplane width that is quantized while the inplane one is not is the
# model working, not a defect.
#
# Widths and penumbrae are the FFF inflection construction (see above), `nominal` is the
# field projected to that depth, and `OAR(80%)` is the off-axis ratio at 80 % of the field
# half-width — the unflattened cone's depth, which is what the primary
# fluence table sets.
#
# The width column is an *unfitted* check: `MLC_TIP_OFFSET` is zero, so nothing here was
# calibrated against a field size and the agreement is the rounded-end tangent geometry
# alone. The penumbra column is not a check in the same way — it is grid-limited at
# `SPACING = 0.25` and reads systematically wide by a few tenths of a millimetre.

# %%
print(
    f"{'field':>6} {'depth':>6} {'width_x':>8} {'width_y':>8} {'pen_x':>7} {'pen_y':>7} "
    f"{'nominal':>8} {'OAR(80%)':>9}"
)
for field in PROFILE_FIELDS:
    pooled = measure_profiles(runs[field].scan)
    for depth in PROFILE_DEPTHS:
        nominal = 10.0 * field * (PROFILE_SSD + depth) / SAD
        print(
            f"{field * 10:6.0f} {depth:6.1f} "
            f"{10 * pooled[f'width_x_d{depth:g}']:8.2f} {10 * pooled[f'width_y_d{depth:g}']:8.2f} "
            f"{10 * pooled[f'penumbra_x_d{depth:g}']:7.2f} "
            f"{10 * pooled[f'penumbra_y_d{depth:g}']:7.2f} {nominal:8.2f} "
            f"{pooled[f'oar_80_x_d{depth:g}']:9.3f}"
        )

# %% [markdown]
# ## The output-factor matrix
#
# SSD 95 cm with the point at the isocentre, every x against every y. `Scp` is the dose
# per unit incident fluence at the point relative to the reference square field — which
# is why each field's dose-per-history is multiplied by its aperture area first.
#
# Both source components are reported separately (`dose_per_fluence_primary` and
# `..._extrafocal`), so the extra-focal weight can be refitted against the measured matrix
# straight from the CSV without re-running anything.
#
# **Below ~15 mm this stops being a check on the model.** A reported output factor at
# 1 x 1 cm^2 is a detector measurement with a detector-specific correction (TRS-483), and
# nothing here models the detector; the flattened notebook measured this explicitly and
# found several per cent of the small-field residual accounted for by the size of the
# measuring volume alone. Treat 20 mm and above as the comparison.

# %%
print(
    f"output-factor matrix at SSD {OUTPUT_FACTOR_SSD:g} cm "
    f"({len(OUTPUT_FACTOR_SIDES) ** 2} fields):"
)
matrix_start = time.perf_counter()
output_runs: dict[tuple[float, float], FieldResult] = {}
for field_x in OUTPUT_FACTOR_SIDES:
    for field_y in OUTPUT_FACTOR_SIDES:
        output_runs[field_x, field_y] = simulate(
            field_x,
            field_y,
            OUTPUT_FACTOR_SSD,
            OUTPUT_FACTOR_DEPTH,
            phantom_depth=OUTPUT_FACTOR_PHANTOM_DEPTH,
            lateral_margin=OUTPUT_FACTOR_LATERAL_MARGIN,
            base_histories=OUTPUT_FACTOR_PER_REPLICATE,
            replicates=OUTPUT_FACTOR_REPLICATES,
            # The reference field's own error lands on every ratio and cancels nowhere.
            history_boost=(
                REFERENCE_HISTORY_BOOST
                if (field_x, field_y) == (REFERENCE_FIELD, REFERENCE_FIELD)
                else 1.0
            ),
            scoring_half=OUTPUT_FACTOR_SCORING_HALF,
            announce=False,
        )
    print(f"  x = {field_x * 10:5.0f} mm done ({time.perf_counter() - matrix_start:.1f} s elapsed)")

if (REFERENCE_FIELD, REFERENCE_FIELD) not in output_runs:
    raise ValueError(
        f"REFERENCE_FIELD_MM = {REFERENCE_FIELD_MM} is not in OUTPUT_FACTOR_SIDES_MM; "
        "every output factor is a ratio against that field"
    )
reference = output_runs[REFERENCE_FIELD, REFERENCE_FIELD]
reference_roi = roi_dose(reference.scan, OUTPUT_FACTOR_DEPTH)
reference_norm = reference_roi * reference.area_iso
reference_rel = reference.spread("roi_dose")[1] / reference_roi

output_table: list[dict[str, Any]] = []
for (field_x, field_y), run in output_runs.items():
    value = roi_dose(run.scan, OUTPUT_FACTOR_DEPTH)
    normalized = value * run.area_iso
    relative = run.spread("roi_dose")[1] / value
    factor = normalized / reference_norm
    output_table.append(
        {
            "field_x_mm": field_x * 10.0,
            "field_y_mm": field_y * 10.0,
            # Sterling's equivalent square, the field size a beam model indexes on.
            "equivalent_square_mm": 10.0 * 2.0 * field_x * field_y / (field_x + field_y),
            "dose_per_fluence": normalized,
            "dose_rel_sigma": relative,
            "output_factor": factor,
            # Independent runs, so the two relative errors combine in quadrature. The
            # reference field's own error is common to every ratio and cancels nowhere.
            "output_factor_sigma": factor * math.hypot(relative, reference_rel),
            "dose_per_fluence_primary": run.component_roi["primary"],
            "dose_per_fluence_extrafocal": run.component_roi["extrafocal"],
            "dose_per_fluence_contaminant_beam": run.component_roi["contaminant_beam"],
            "dose_per_fluence_contaminant_head": run.component_roi["contaminant_head"],
            "seconds": run.seconds,
        }
    )

SCP = np.array(
    [
        [
            next(
                r["output_factor"]
                for r in output_table
                if r["field_x_mm"] == fx * 10.0 and r["field_y_mm"] == fy * 10.0
            )
            for fy in OUTPUT_FACTOR_SIDES
        ]
        for fx in OUTPUT_FACTOR_SIDES
    ]
)

print(f"  matrix: {time.perf_counter() - matrix_start:.1f} s")

# Read this before the matrix itself. Both allocations above are *derived*, so this is
# where they are checked: if the spread across the square fields is still large, the
# levelling is not working and the clamps are the place to look. `OUTPUT_FACTOR_HISTORIES`
# is the lever, at 1/sqrt(N).
achieved = [row["dose_rel_sigma"] for row in output_table]
print(
    f"achieved on-axis error: median {100 * float(np.median(achieved)):.2f} %, "
    f"worst {100 * max(achieved):.2f} % (error scales as 1/sqrt(histories))"
)
print(
    "  by square field (mm): "
    + ", ".join(
        f"{row['field_x_mm']:.0f} {100 * row['dose_rel_sigma']:.2f} %"
        for row in output_table
        if row["field_x_mm"] == row["field_y_mm"]
    )
)

# A self-check that caught a real defect while this notebook was being written — when the
# head leaked outside the built leaf strips the worst pair disagreed by 7 %, several
# sigma, and the extra-focal component was not even monotone in field size. Since
# `SPOT_SIGMA_V` it is no longer a pure symmetry check: an elliptical spot makes
# Scp(x, y) != Scp(y, x) *by design*, about +3 % on the pairs with a 1 cm side (which is
# the machine's own measured asymmetry, the thing sigma_v was fitted to) and next to
# nothing on pairs of 2 cm and up. So read it as: the median over all pairs sits near the
# 1 cm rows' intended asymmetry; a *worst* pair several sigma beyond its error bars, or
# any asymmetry among the wide pairs, still means geometry before statistics.
pairs = [
    abs(SCP[i, j] / SCP[j, i] - 1.0)
    for i in range(SCP.shape[0])
    for j in range(i + 1, SCP.shape[1])
    if SCP[j, i] > 0.0
]
if pairs:
    print(
        f"reciprocity |Scp(x,y)/Scp(y,x) - 1|: median {100 * float(np.median(pairs)):.2f} %, "
        f"worst {100 * max(pairs):.2f} %"
    )

print("Scp, rows = x (leaf travel), columns = y (leaf sides), mm at iso")
print("       " + "".join(f"{side * 10:8.0f}" for side in OUTPUT_FACTOR_SIDES))
for i, field_x in enumerate(OUTPUT_FACTOR_SIDES):
    print(f"{field_x * 10:6.0f} " + "".join(f"{value:8.4f}" for value in SCP[i]))

# %% [markdown]
# ## The summary figure
#
# Depth doses and the output-factor matrix, saved beside this notebook.

# %%
fig, (ax_pdd, ax_of) = plt.subplots(1, 2, figsize=(11.5, 4.6))
for colour, field in zip(COLOURS, PROFILE_FIELDS, strict=True):
    depth_cm, curve = depth_dose(runs[field].scan)
    ax_pdd.plot(depth_cm, 100.0 * curve / curve.max(), color=colour, lw=1.5, label=f"{field:g} cm")
ax_pdd.set_title(f"Central-axis depth dose (SSD {PROFILE_SSD:g} cm)")
ax_pdd.set_xlabel("depth (cm)")
ax_pdd.set_ylabel("dose (% of own maximum)")
ax_pdd.grid(alpha=0.25, lw=0.6)
ax_pdd.legend(fontsize=8, ncol=2, title="field at iso")

image = ax_of.imshow(SCP, origin="lower", cmap="cividis", aspect="auto")
fig.colorbar(image, ax=ax_of, label=f"Scp (ref {REFERENCE_FIELD_MM:g} mm square)")
ax_of.set_xticks(range(len(OUTPUT_FACTOR_SIDES)), [f"{s * 10:g}" for s in OUTPUT_FACTOR_SIDES])
ax_of.set_yticks(range(len(OUTPUT_FACTOR_SIDES)), [f"{s * 10:g}" for s in OUTPUT_FACTOR_SIDES])
ax_of.set_xlabel("y field at iso, leaf sides (mm)")
ax_of.set_ylabel("x field at iso, leaf travel (mm)")
ax_of.set_title(f"Output factors (SSD {OUTPUT_FACTOR_SSD:g} cm, {OUTPUT_FACTOR_DEPTH:g} cm deep)")
for i in range(SCP.shape[0]):
    for j in range(SCP.shape[1]):
        ax_of.text(j, i, f"{SCP[i, j]:.3f}", ha="center", va="center", fontsize=5.5, color="w")
fig.tight_layout()
fig.savefig(f"{STEM}.png", dpi=120)
if SHOW_PLOTS:
    plt.show()

# %% [markdown]
# ## Profiles
#
# One panel per measured depth, every field, in the leaf-travel direction. The
# unflattened cone is what makes the large fields peak on the axis, and these are the
# curves the shipped fluence table was fitted against — so read them as a fit residual,
# not as a prediction. The diagonals below are the only data beyond the 14 cm half-width,
# and the fit leans on them there.

# %%
fig, axes = plt.subplots(
    1, len(PROFILE_DEPTHS), figsize=(3.1 * len(PROFILE_DEPTHS), 4.2), sharey=True
)
for ax, depth in zip(np.atleast_1d(axes), PROFILE_DEPTHS, strict=True):
    for colour, field in zip(COLOURS, PROFILE_FIELDS, strict=True):
        offsets, values = lateral_profile(runs[field].scan, 0, depth)
        ax.plot(
            offsets,
            100.0 * values / float(np.interp(0.0, offsets, values)),
            color=colour,
            lw=1.2,
            label=f"{field:g} cm",
        )
    ax.set_title(f"{depth:g} cm depth")
    ax.set_xlabel("off-axis x (cm)")
    ax.set_xlim(-17.0, 17.0)
    ax.set_ylim(0.0, 125.0)
    ax.grid(alpha=0.25, lw=0.6)
np.atleast_1d(axes)[0].set_ylabel("dose (% of central axis)")
np.atleast_1d(axes)[-1].legend(fontsize=7, title="field at iso")
fig.suptitle(f"Inplane (leaf-travel) profiles, SSD {PROFILE_SSD:g} cm")
fig.tight_layout()
fig.savefig(f"{STEM}_profiles.png", dpi=120)
if SHOW_PLOTS:
    plt.show()

# %% [markdown]
# ## Diagonals, and the two principal directions against each other
#
# Left: the largest field's diagonals at every depth. The diagonal reaches further —
# half-width `a/sqrt(2)` at the corner — and falls off more gradually, because at a corner
# two field edges are crossed at once, so the dose at the exact corner is about a quarter
# of the axis and not half.
#
# Right: the three scan directions of that field at one depth. Any systematic difference
# between the inplane and crossplane curves here is the two collimation mechanisms — leaf
# ends against leaf sides — and is the thing this head is unusual for.

# %%
diagonal_run = runs[DIAGONAL_FIELD]
fig, (ax_diag, ax_scan) = plt.subplots(1, 2, figsize=(12.0, 4.6))
depth_colours = plt.get_cmap("magma")(np.linspace(0.15, 0.8, len(PROFILE_DEPTHS)))
for colour, depth in zip(depth_colours, PROFILE_DEPTHS, strict=True):
    radius, values = both_diagonals(diagonal_run.scan, depth)
    ax_diag.plot(
        radius,
        100.0 * values / float(np.interp(0.0, radius, values)),
        color=colour,
        lw=1.3,
        label=f"{depth:g} cm",
    )
ax_diag.axvline(DIAGONAL_FIELD * math.sqrt(2.0) / 2.0, color="0.6", lw=0.8, ls="--")
ax_diag.set_title(f"Diagonal profiles, {DIAGONAL_FIELD:g} x {DIAGONAL_FIELD:g} cm")
ax_diag.set_xlabel("distance from the axis along the diagonal (cm)")
ax_diag.set_ylabel("dose (% of central axis)")
ax_diag.set_ylim(0.0, 125.0)
ax_diag.grid(alpha=0.25, lw=0.6)
ax_diag.legend(fontsize=8, title="depth")

scan_depth = PROFILE_DEPTHS[2]
for label, colour, (offsets, values) in (
    ("inplane x (leaf ends)", "#2b5fa8", lateral_profile(diagonal_run.scan, 0, scan_depth)),
    ("crossplane y (leaf sides)", "#3f8f6b", lateral_profile(diagonal_run.scan, 1, scan_depth)),
    ("diagonal", "#d8853b", both_diagonals(diagonal_run.scan, scan_depth)),
):
    centre = float(np.interp(0.0, offsets, values))
    ax_scan.plot(offsets, 100.0 * values / centre, color=colour, lw=1.4, label=label)
ax_scan.set_title(f"Scan directions, {DIAGONAL_FIELD:g} cm at {scan_depth:g} cm depth")
ax_scan.set_xlabel("distance from the axis (cm)")
ax_scan.set_ylabel("% of central axis")
ax_scan.set_ylim(0.0, 125.0)
ax_scan.grid(alpha=0.25, lw=0.6)
ax_scan.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{STEM}_diagonals.png", dpi=120)
if SHOW_PLOTS:
    plt.show()

# %% [markdown]
# ## The aperture the two banks actually make
#
# Zero-variance and analytic: for each spectral bin the ray from the focal spot to the
# point is attenuated through both banks' chords and the modelled air, divided by the
# inverse square of its length and scaled by the source's own radial fluence. It traces
# from a **point** focal spot and keeps only the narrow-beam primary, so it carries no
# source penumbra and no scatter — which is exactly what makes it a geometry check.
#
# This is the figure to look at first if a field comes out the wrong size: the leaf
# quantization in y, the rounded-end softening in x, and the two banks' agreement are all
# visible in it directly.
#
# The right-hand panel is worth reading carefully, because it is the one thing about this
# head that will bite a plan. Each *edge* can sit on a 0.5 cm boundary — that is what the
# stagger buys — but a **symmetric** field therefore steps in 1.0 cm, and a requested
# 3.5 cm opens to 4.0. Every field size in this notebook was chosen to be exactly
# realizable (all the half-widths land on the 0.5 cm lattice); 3.5 is drawn to show what
# happens when one does not. An asymmetric 3.5 cm field, -1.5 to +2.0, is available.


# %%
def primary_fluence(
    stack: BeamLimitingStack,
    x: np.ndarray,
    y: np.ndarray,
    ssd: float = PROFILE_SSD,
    depth: float = 0.0,
    *,
    through_phantom: bool = False,
) -> np.ndarray:
    """Primary energy fluence at points in a plane below the head (relative)."""
    points = np.stack([x.ravel(), y.ravel(), np.full(x.size, ssd + depth)], axis=1)
    radius = np.linalg.norm(points, axis=1)
    directions = points / radius[:, None]
    thickness = stack.path_lengths(np.zeros_like(points), directions)  # (n, n_devices)
    air_length = (ssd - stack.exit_z) / directions[:, 2]
    water_length = depth / directions[:, 2] if through_phantom else 0.0

    edges = EDGE_SPECTRUM.edges
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    weights = EDGE_SPECTRUM.bin_probabilities
    total = np.zeros(points.shape[0], dtype=np.float64)
    for energy, weight in zip(midpoints, weights, strict=True):
        if weight <= 0.0:
            continue
        tau = np.zeros(points.shape[0], dtype=np.float64)
        for column, device in enumerate(stack.devices):
            tau += (
                device.density
                * XS.mu_over_rho_total(float(energy), device.material)
                * thickness[:, column]
            )
        tau += AIR_DENSITY * XS.mu_over_rho_total(float(energy), AIR_MATERIAL) * air_length
        tau += XS.mu_over_rho_total(float(energy), WATER) * water_length
        total += weight * energy * np.exp(-tau)
    if FLUENCE is not None:
        iso_radius = np.hypot(points[:, 0], points[:, 1]) * SAD / (ssd + depth)
        total *= FLUENCE.at_radii(iso_radius) / FLUENCE.at_radius(0.0)
    return (total * (SAD / radius) ** 2 / float(np.sum(weights * midpoints))).reshape(x.shape)


APERTURE_FIELDS = ((3.0, 4.0), (DIAGONAL_FIELD, DIAGONAL_FIELD))
fig, axes = plt.subplots(1, len(APERTURE_FIELDS) + 1, figsize=(15.0, 4.4))
for ax, (field_x, field_y) in zip(axes[:-1], APERTURE_FIELDS, strict=True):
    reach = 0.75 * max(field_x, field_y) + 2.0
    axis_cm = np.linspace(-reach, reach, 321)
    gx, gy = np.meshgrid(axis_cm, axis_cm, indexing="ij")
    # Traced at the isocentre plane, in air, so the picture is the aperture and nothing
    # else; the profiles above are what the transported dose does with it.
    field = primary_fluence(
        beam_limiting_stack(field_x, field_y),
        gx * SAD / PROFILE_SSD,
        gy * SAD / PROFILE_SSD,
        ssd=SAD,
    )
    picture = ax.imshow(
        field.T,
        origin="lower",
        extent=(axis_cm[0], axis_cm[-1], axis_cm[0], axis_cm[-1]),
        cmap="cividis",
    )
    fig.colorbar(picture, ax=ax, label="primary energy fluence (rel.)")
    ax.set_title(f"{field_x:g} x {field_y:g} cm at the isocentre")
    ax.set_xlabel("x, leaf travel (cm)")
    ax.set_ylabel("y, leaf sides (cm)")

# The y edge is the intersection of two staggered banks, so it lands on a 0.5 cm lattice.
# A symmetric field therefore steps in 1.0 cm: 3.5 opens to 4.0, and the two curves lie on
# top of each other.
ax_cut = axes[-1]
span = np.linspace(-4.0, 4.0, 1601)
for field_y, colour, style in (
    (3.0, "#2b5fa8", "-"),
    (3.5, "#3f8f6b", "-"),
    (4.0, "#d8853b", "--"),
):
    values = primary_fluence(beam_limiting_stack(3.0, field_y), np.zeros_like(span), span, ssd=SAD)
    ax_cut.plot(
        span,
        values / values.max(),
        color=colour,
        lw=1.3,
        ls=style,
        label=f"requested {field_y:g} cm",
    )
    ax_cut.axvline(0.5 * field_y, color=colour, lw=0.7, ls=":")
ax_cut.set_title("Crossplane edge: 0.5 cm edges, 1.0 cm symmetric fields")
ax_cut.set_xlabel("y, leaf sides (cm)")
ax_cut.set_ylabel("primary fluence (rel.)")
ax_cut.grid(alpha=0.25, lw=0.6)
ax_cut.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{STEM}_aperture.png", dpi=120)
if SHOW_PLOTS:
    plt.show()

# %% [markdown]
# ## Save the tables
#
# Three files. `_summary.csv` is one row per profile field with `dmax`, the PDDs and the
# per-depth widths, penumbrae and symmetries, each with the replicate standard error
# beside it. `_curves.csv` is every curve in long format — depth doses, both principal
# profiles at every depth, and the largest field's diagonals. `_output_factors.csv` is
# the matrix, with both source components kept separate so the extra-focal weight can be
# refitted from it. Floats are written to six significant figures, well past any Monte
# Carlo digit.

# %%
summary: list[dict[str, Any]] = []
for field in PROFILE_FIELDS:
    run = runs[field]
    pooled = measure_profiles(run.scan)
    row: dict[str, Any] = {
        "field_mm": field * 10.0,
        "ssd_cm": PROFILE_SSD,
        "dose_rel_sigma": run.spread("roi_dose")[1] / roi_dose(run.scan, PROFILE_DEPTHS[2]),
        "seconds": run.seconds,
    }
    for key, value in pooled.items():
        scale = 10.0 if key.startswith(("width_", "penumbra_")) else 1.0
        row[key] = scale * value
        row[f"{key}_sem"] = scale * run.spread(key)[1]
    summary.append(row)

curves: list[dict[str, Any]] = []
for field in PROFILE_FIELDS:
    scan = runs[field].scan
    depth_cm, curve = depth_dose(scan)
    curves += [
        {
            "field_mm": field * 10.0,
            "kind": "pdd",
            "depth_cm": float(d),
            "coord_cm": float(d),
            "value": float(v),
        }
        for d, v in zip(depth_cm, curve, strict=True)
    ]
    for depth in PROFILE_DEPTHS:
        named = [
            (f"profile_{n}", *lateral_profile(scan, a, depth)) for a, n in ((0, "x"), (1, "y"))
        ]
        if field == DIAGONAL_FIELD:
            named.append(("diagonal", *both_diagonals(scan, depth)))
        for kind, offsets, values in named:
            curves += [
                {
                    "field_mm": field * 10.0,
                    "kind": kind,
                    "depth_cm": depth,
                    "coord_cm": float(c),
                    "value": float(v),
                }
                for c, v in zip(offsets, values, strict=True)
            ]

for name, records in (
    (f"{STEM.name}_summary.csv", summary),
    (f"{STEM.name}_curves.csv", curves),
    (f"{STEM.name}_output_factors.csv", output_table),
):
    with STEM.with_name(name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(
            {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in record.items()}
            for record in records
        )
    print(f"saved {name}")
