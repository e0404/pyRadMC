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
# (This file is a jupytext "percent" notebook: the narrative lives in the markdown
# cells below rather than in a module docstring. It is still an ordinary script --
# `python examples/commissioning_demo.py` runs it top to bottom.)

# %% [markdown]
# # Commissioning a modelled treatment head
#
# This is the beam-data measurement session a physicist runs on a new machine,
# assembled from pieces pyRadMC already has: a water phantom at a stated SSD, a set
# of square fields set on the beam-limiting devices, and the four things
# commissioning reports.
#
# * **Percentage depth dose** on the central axis, with `dmax`, PDD(10), PDD(20) and
#   D20/D10.
# * **Tissue-phantom ratios**, twice over: at a couple of depths from real SAD-setup
#   runs — the phantom moved so the point of measurement stays at the isocentre — and
#   as full curves converted from the depth doses for every field. The first is what
#   `TPR(20,10)` claims to be; the second is what a beam model is actually loaded
#   with, and running both lets each check the other.
# * **Total scatter (output) factors** `Scp` against a reference field, with the
#   square-to-circle equivalences a beam model wants alongside them.
# * **Primary fluence** in the measurement plane: deterministic, zero-variance,
#   ray-traced from the focal spot through the device chords — in air and through
#   the phantom, along both principal axes and both diagonals.
#
# Open it as a notebook (jupytext, or VS Code, which reads the header above) and
# work down the cells, or just run the file. Everything you would change lives in
# the **Parameters** cell; nothing below it takes arguments.
#
# ## Read these before quoting a number
#
# **The flattening filter enters as a measured fluence, not as geometry.** With
# `PRIMARY_FLUENCE` set, photons are sampled on the source plane and weighted by the
# machine's measured radial primary fluence — the virtual source model of Tacke et
# al., *Med. Phys.* **33** (2006) 1125-1132 — so the filter's off-axis *shaping* and
# the primary collimator's circular field edge are both present, and profiles are
# flat because the machine's own commissioning curve says they should be. What is
# still missing is the filter's off-axis **softening**: the Ali & Rogers spectra are
# on-axis spectra for flattened beams and this source uses one spectrum at every
# radius, so the beam does not harden toward the axis. That shows up as a residual
# in the outer part of large fields and at depth, not in the depth doses or output
# factors. Set `PRIMARY_FLUENCE = None` to fall back to a uniform fan, which has
# neither effect and whose profiles come out visibly rounded; point `PHASESPACE` at a
# file scored under a real filter to get both.
#
# **Head scatter is a fitted second source, not modelled geometry.** Scatter out of
# the flattening filter and primary collimator is represented by a wide Gaussian
# source at the filter plane (`EXTRAFOCAL_*`), occluded by the same jaws and MLC as
# the primary — the standard extra-focal model. Its *shape* is machine data and its
# *strength* is one fitted number; without it `Scp` is essentially the phantom-scatter
# component alone and comes out about 10 % too flat over 30 to 400 mm. Set
# `EXTRAFOCAL_WEIGHT = None` to see that. What is still absent either way is scatter
# off the far jaw faces below the filter, and monitor-chamber backscatter, which on a
# Siemens head changes small-field output per MU and is an empirical correction rather
# than a geometric one. `HEAD = "presolve"` does carry collimator scatter and
# the contaminant electrons generated in the air column (one forced Compton — see
# `pyRadMC.geometry.head` for the stated approximations). `HEAD = "attenuation"` is
# the deterministic Beer-Lambert baseline instead, and it is far slower per history:
# it re-samples and re-traces the stack on the host for every chunk, which an MLC's
# per-leaf inclusion-exclusion dominates, where the pre-solve pays that cost once
# per replicate on the device.
#
# **The air path is the same length in every setup, on purpose.** `AIR_GAP`
# centimetres above the phantom are real air voxels in the transport grid and the
# pre-solve's analytic column reaches further above that, but the *total* is pinned to
# one value for the depth doses and the TPR runs alike — otherwise their build-up
# regions would carry different contaminant-electron components and could not be
# converted into one another. Above the path is vacuum, as everywhere else in the
# engine, which is harmless precisely because it does nothing. See "A consistent air
# path".
#
# **`SEED` does not pin the answer on the device pre-solve route.** Transport is
# bit-reproducible per device, and so is the host pre-solve — both checked. The Warp
# *device* pre-solve is not: it appends its exit population with atomics, so the
# phase-space record **order** varies between runs and uniform resampling then draws
# a different subset. The physics is unchanged and the scatter sits inside the error
# bars, but two runs will not print identical digits. Use `HEAD = "attenuation"`, or
# the reference engine, when you need a fixed answer.
#
# **The penumbra column is grid-limited and is not a quotable number.** At
# `SPACING = 0.25` the 10 x 10 penumbra reads about 0.8 mm wider than the identical
# physics does at 0.125 cm — 5.4 mm against 4.6 mm — because a ~4.5 mm penumbra cannot
# be resolved on 2.5 mm voxels. The penumbrae below are therefore usable for comparing
# fields with each other but not as absolute values; measure those on a finer grid, as
# `SPOT_SIGMA` was set. Field widths, depth doses and output factors are unaffected —
# they are not sub-voxel quantities.
#
# **Every uncertainty comes from replicates.** Each field is simulated `REPLICATES`
# times independently, head model included, and the spread of those runs is the
# error bar on *every* quantity — including the nonlinear ones (dmax, field width,
# penumbra) that a per-voxel sigma cannot address at all. Read the `_sem` columns
# before believing a third digit.

# %% [markdown]
# ## Parameters
#
# The whole configuration. Lengths in cm, energies in MeV.

# %%
# --- the beam ---------------------------------------------------------------------
BEAM = "siemens-6mv"  # any key of pyRadMC.geometry.spectrum.ALI_ROGERS_BEAMS
PHASESPACE = None  # or a path to an IAEA header, which then replaces the spectrum
SPOT_SIGMA = 0.05  # focal spot sigma (0.5 mm); this is what makes the geometric penumbra.
#                    Fitted *jointly* with EXTRAFOCAL_WEIGHT against this machine's
#                    measured 10 x 10 penumbra (4.4-4.5 mm at 5 cm depth) and its measured
#                    output factors, both on a converged grid. The 1 mm this notebook
#                    shipped with was too wide on both counts.
# Measured radial primary fluence: a two-column "radius_mm relative_fluence" table at
# the isocentre (the PPBKC `primflu.dat` layout). None emits a uniform fan instead --
# see "There is no flattening filter" above for what that costs.
PRIMARY_FLUENCE = "D:/Python/pyvsm/example_data/primflu.dat"

# --- the extra-focal (head-scatter) source ---------------------------------------------
# Head scatter -- photons scattered out of the flattening filter and the primary
# collimator -- as a second, wide Gaussian source near the filter plane (Jaffray et al.,
# Med. Phys. 20 (1993) 1417; Sharpe et al., Med. Phys. 22 (1995) 2065). Sc rises with
# field size because a wider collimator opening lets the measurement point see more of
# that extended source, so the field-size dependence comes out of the *same* jaws and MLC
# that collimate the primary -- nothing extra is modelled. Set EXTRAFOCAL_WEIGHT to None
# to switch it off and recover the primary-only beam.
# The weight is the one fitted parameter, and it must be fitted *jointly with*
# `SPOT_SIGMA`: the focal spot sets how much of the source a small aperture occludes, so
# it moves output factors as well as the penumbra, and a weight fitted at one spot size
# does not transfer to another. Fitted at sigma = 1 mm this reached 0.95 % and then
# degraded to 2.0 % when the spot was reduced — the fit is two-dimensional.
#
# Re-fitted at sigma = 0.5 mm, least squares against this machine's measured output
# factors (`of.dat`) gives w = 0.100, an RMS relative residual of **0.36 %** over 20 to
# 400 mm against 3.4 % with no extra-focal source at all. That w ~ 0.10 comes back
# independently at both spot sizes is the reason to read it as a property of the machine
# rather than an artefact of the fit.
#
# It is not free. The extra-focal source lies under the field edge as a broad pedestal —
# its blur at the isocentre is centimetres wide, so its penumbra cost is set by how much
# of it there is and barely at all by its width — and at this weight the 10 x 10
# penumbra comes out 4.84 mm inplane against a measured 4.4-4.5 mm (crossplane, 4.51 mm,
# is exact). That 0.4 mm is conceded deliberately: w = 0.05 recovers it but costs a 5.5x
# worse output-factor fit.
#
# A radial profile on the emitting square is **not** the fix, which is worth stating
# because it is the obvious idea and an earlier revision of this cell recommended it.
# Measured on the geometry itself — build the extra-focal fluence at the isocentre through
# the real 10 x 10 stack, blur both components by the Gaussian that reproduces the sweep's
# own 4.84 mm, and vary the shape at fixed on-axis contribution:
#
#     no extra-focal source        4.39 mm      (measured: 4.4-4.5)
#     uniform, w = 0.05            4.61
#     uniform, w = 0.10 (shipped)  4.84
#     radial, r0 = 15 cm           4.83
#     radial, r0 = 7 cm            4.82
#
# The strength lever is worth 0.44 mm and the shape lever 0.02 mm, and r0 = 7 cm is
# already an absurd source that could not illuminate a 40 x 40 field at all. The reason is
# the sentence above, taken seriously: the extra-focal fluence at the isocentre is a
# ~95 cm plateau with 11 cm Gaussian edges, and no shaping of the emitting square imposes
# a 5 cm-scale gradient behind an 11 cm blur. What reaches the edge of a small field is set
# by how much extra-focal there is, full stop.
#
# So the two fitted requirements genuinely conflict, and the way out is not another free
# parameter in this source. The likelier account is in the paragraph above about what is
# absent: on a Siemens head, monitor-chamber backscatter changes output per MU with field
# size, it is an empirical correction rather than a geometric one, and it is not modelled
# here. If part of the measured `Sc` slope is really backscatter, then `w` is over-fitted —
# it is absorbing a non-geometric effect through the only geometric knob available — and
# being too strong for the penumbra is exactly the symptom that would produce.
EXTRAFOCAL_WEIGHT: float | None = 0.10  # fraction of the on-axis emitted fluence
EXTRAFOCAL_Z = 10.6  # source plane, cm from the target (params.dat source_ff_distance)
EXTRAFOCAL_SIGMA = 1.32  # Gaussian sigma there (cm); see the resolved-configuration cell
EXTRAFOCAL_FIELD = 50.0  # emitting square at the isocentre, cm: the primary collimator's
#                          opening, so it is field-independent and the jaws do the cutting
EXTRAFOCAL_HISTORY_FRACTION = 0.20  # histories relative to the primary's, per replicate

# --- the treatment head -------------------------------------------------------------
# Device slabs in the beam frame: cm downstream of the focal spot. The frame origin is
# the focal spot and its axes are the engine axes, so beam-frame w == engine z and the
# isocentre sits at (0, 0, SAD). Devices must be ordered and must not overlap in z.
SAD = 100.0
NOZZLE = 60.0
JAW_U_Z = None  # (22.5, 29.9)  # the jaw pair limiting x
JAW_V_Z = (22.5, 29.9)  # the jaw pair limiting y
MLC_Z = (36.35, 45.85)  # the leaf bank; leaves travel along x, strips tile y. The real
#                         leaves are 95 mm deep and *alternate*, spanning 35.7-45.2 and
#                         37.0-46.5, so their mid-planes are 40.45 and 41.75 -- mean
#                         41.10, which is also the mid-plane of the 35.7-46.5 envelope.
#                         Modelling one 95 mm slab there therefore corrects the tungsten
#                         thickness (the envelope gave every leaf 108 mm, no real leaf
#                         has more than 95) without moving any tip projection. Stated
#                         approximation: the stagger itself is dropped. A vendor
#                         calibrates leaf position to project correctly to the isocentre,
#                         so it puts no alternating edge in the radiation field; what
#                         survives is that neighbouring tips sit at slightly different
#                         source distances, worth 0.06 mm of penumbra spread here (the
#                         spot's projection factor differs by 5 % between the two planes)
#                         inside a 4.4 mm penumbra dominated by electron transport.
#                         Leakage is the real casualty and is under-predicted either way,
#                         since the gaps and tongue-and-groove are not modelled at all.
MLC_TIP_RADIUS = 11.5  # rounded leaf end; must cover the leaf half-height
# The leaf-position calibration: cm at the isocentre added to each tip's retraction, as a
# function of where that tip sits. A rounded end does not put its 50 % dose edge under the
# tip's nominal position -- the edge follows the ray that grazes the arc -- and how far off
# it sits depends on the tip's own off-axis distance, because a divergent ray meets the
# cylinder at a steeper angle the further out it is. One constant is therefore wrong
# everywhere but where it was fitted. Real machines ship exactly this as a table.
#
# Fitted to the width residuals of this notebook's own sweep, so understand what it is: a
# calibration, not geometry, and once it is applied the field width stops being an
# independent check on the model. Penumbra, output factors and the TPR conversion still
# are -- none of them was used here.
#
# Two passes. The first took the width residual at face value, which assumes the edge
# follows the tip 1:1; it does not, because a retracting rounded end presents a different
# chord of its arc, and the measured gain is ~1.3 in the middle of the range, 1.08 at
# 2.5 mm off axis and 0.81 at 10 cm. The second pass divides each residual by the gain
# measured at that position. Iterating further is not worth it: what is left is at the
# 0.1 mm level, which is where the grid dependence above already sits.
#
# The last two nodes are not fitted. Subtracting the analytic tangent offset -- the ray
# from the spot that grazes the tip cylinder, at radius `MLC_TIP_RADIUS` and the mid-plane
# below, projected to the isocentre:
#
#     t = half * z_mid / SAD, c = t + R
#     u = (c * z_mid - R * sqrt(z_mid^2 + c^2 - R^2)) / (z_mid^2 - R^2)
#     delta_tangent(half) = half - u * SAD
#
# leaves a residual of -0.0569 +- 0.0081 cm over a 40x span of tip positions, 0.25 to
# 10 cm. That is flat: the fitted table *is* the tangent geometry plus one constant, and
# the constant is an ordinary light-field/radiation-field offset of the kind a vendor
# quotes. So the 300 and 400 mm entries are `delta_tangent + (-0.0569)` evaluated, not
# measured, and they land on what the sweep independently wanted (+0.2298 and +0.4702)
# to 0.14 mm and 0.01 mm. This is why they are here at all: their excess over the trend
# the small fields sit on used to be unexplained, and clamping was the honest response.
# It is now predicted by geometry that was never fitted to them, so extrapolating on it
# is no longer burying a centimetre inside something labelled "calibration".
#
# Simulated back, the two nodes deliver: `width_x` goes 297.54 -> 300.20 mm at 300 and
# 392.30 -> 399.02 mm at 400, so residuals of -2.46 and -7.70 mm become +0.20 and
# -0.98 mm and both fields join the +-0.35 mm band the rest of the sweep sits in. The
# 400 mm field is still the worst in the sweep and is left that way: its last 0.5 mm per
# side would have to come from fitting the node, and a fitted node here would forfeit the
# only reason to trust it out there, which is that nothing was fitted.
#
# The smaller nodes are left fitted rather than replaced by the same formula. Their
# 0.08 cm scatter about it is at the grid-dependence level and re-deriving them
# analytically would trade a measured agreement for a modelled one at no gain.
MLC_TIP_CALIBRATION = (
    (0.25, -0.0788),
    (0.50, -0.0556),
    (1.00, -0.0504),
    (1.50, -0.0467),
    (2.00, -0.0474),
    (2.50, -0.0450),
    (3.00, -0.0455),
    (4.00, -0.0376),
    (5.00, -0.0237),
    (7.50, +0.0282),
    (10.00, +0.0781),
    (15.00, +0.2438),  # analytic; see above
    (20.00, +0.4688),  # analytic; see above
)
# The radius and the offset are fitted *together*, against this machine's measured 10 x 10
# penumbra (4.4-4.5 mm) and its nominal field width, on a converged 1.25 mm grid. They must
# be: the offset is not a pure translation. Retracting a rounded tip presents a different
# chord of the arc, so it moves the penumbra too (0.13 mm per 0.034 cm here, several sigma),
# and a radius fitted with the offset pinned at zero comes out at 10.5 cm -- an artefact.
# Solving the 2 x 2 instead (d width / d offset ~ +24.8 mm/cm, d pen / d offset ~ -3.9)
# gives R = 11.5, which also sits where published radii for comparable MLCs do. Measured
# at the reference field: penumbra 4.45 +- 0.07 mm, width 100.02 +- 0.06 mm.
#
# A *single* offset cannot be right at every field size, and this one is not: the residual
# runs +0.54 mm at 40 mm to -1.55 mm at 200 mm. That slope is the rounded end behaving
# correctly, not a defect -- a divergent ray meets the tip cylinder more obliquely the
# further off axis the leaf sits, so the effective edge shifts with leaf position (Boyer &
# Li, Med. Phys. 24 (1997) 757). It is exactly why vendors calibrate leaf position with a
# position-dependent table rather than one number, and why `MLC_TIP_CALIBRATION` above is
# one: that residual slope is what it was fitted to. The numbers in this paragraph are the
# *pre-calibration* state, kept because they are how the slope was found and because they
# are what a reader gets by emptying the table.
# Note this is *not* a primary-fluence effect: `primflu.dat` horns up to 1.082
# at 14 cm radius and is still 1.062 at the 200 mm field edge, which pushes that edge out
# by ~0.3 mm, so the geometric deficit there is larger than the residual shows.
MLC_PITCH = 0.5  # leaf width projected to the isocentre plane
MLC_FOCUSED_SIDES = True  # trapezoid leaf cross-section, focused on the spot. Real
#                           single-focusing MLCs are built this way; parallel sides
#                           shadow a jaw-bounded field from ~12 % inside its edge.
MLC_Y_MARGIN = 0.5  # cm at iso by which the bank is opened past the field, so the y
#                     jaws alone define that edge. 0 parks the leaves flush with the
#                     field, which is the real leaf pattern; with focused sides the two
#                     agree, so this is belt-and-braces rather than a correction.
EMISSION_Z = 20.0  # the source plane, above the first device
FOCUSED_JAWS = True  # jaw edge faces pivot through the focal spot
# How far past the field the primary fan -- and with it the MLC bank -- reaches, per side
# at the isocentre. Everything aimed outside the aperture is absorbed in the collimator, so
# a wide margin is mostly wasted histories: at a 2 cm margin only 1.2 % of a 5 mm field's
# histories are aimed inside the field, against 82.6 % at 400 mm.
#
# Tightening it is measurably free of bias. At 0.75 cm against 2.0 the dose per unit
# incident fluence moved -0.34 to +0.16 % over 5 to 100 mm with no trend in field size --
# smaller than this notebook's own run-to-run scatter -- and the widths and penumbrae held
# to 0.06 mm, so the fan is not being clipped. The prior was already good: tripling the
# collimator transmission (the contribution a wide margin exists to capture) moved the
# 10 mm output factor by 0.3 sigma.
#
# It is *not* free of cost, and a flat tightening is a net loss. Histories redirected out of
# the collimator and into the aperture get fully transported instead of cheaply absorbed, so
# runs get slower per history. Measured as sigma^2 x time -- the time to reach a fixed error
# bar -- 0.75 cm against 2.0 gains 5.8x at 5 mm, 3.0x at 10 mm, 1.4x at 20 mm, then *loses*:
# 0.71x at 40 mm and 0.45x at 100 mm. The crossover is near 30 mm, which is where this ramp
# reaches the wide value.
FAN_MARGIN_MAX = 2.0
FAN_MARGIN_MIN = 0.75  # must stay clear of MLC_Y_MARGIN, or the bank ends inside the fan
FAN_MARGIN_SLOPE = 0.25

COLLIMATOR = "jaw-mlc"  # "jaws" (x+y), "jaw-mlc" (y jaws + x MLC), or "jaws-mlc" (both)
HEAD = "presolve"  # "presolve" (collimator scatter + contaminant electrons) or
#                    "attenuation" (deterministic Beer-Lambert weights)
DEVICES = "tungsten"  # "tungsten" needs the EPICS tables; "water" is the stand-in

# --- the measurement setup ------------------------------------------------------------
SSD = 95.0
DEPTH = SAD - SSD  # measurement depth; the default puts the point at the isocentre
PHANTOM_DEPTH = 45.0  # 30 cm truncated the largest fields: at 400 mm the dose at 30 cm
#                       depth ran -10.0 % against the measurement purely for want of
#                       backscatter material, and went to +0.0 % at 45 cm deep with a
#                       25 cm margin. Deep enough that the deepest scored plane still has
#                       full backscatter behind it.
AIR_GAP = 25  # SSD - NOZZLE  # centimetres of *transported* air above the surface; the
#               rest of the modelled air path is the pre-solve's analytic column, and the
#               total is held the same in every setup (see "A consistent air path")
LATERAL_MARGIN = 25.0  # phantom beyond the field at the deepest plane; see PHANTOM_DEPTH
SPACING = 0.25  # voxel edge, all three axes
TPR_DEPTHS = (10.0, 20.0)  # SAD-setup depths; () skips the TPR runs entirely
# Run the SAD setup for every field, not just the reference. These are the only TPR numbers
# here that involve no conversion at all, and the converted curves are known to drift high
# below ~40 mm (see the TPR section), so at small fields they are the ones to believe.
# Costs one run per field per depth.
TPR_ALL_FIELDS = True
TPR_HISTORY_FRACTION = 0.5  # of PER_REPLICATE. These runs are read for one on-axis value,
#                             which converges long before the maps the sweep scores; at 0.5
#                             the TPR(20,10) sem lands near 0.5 %, well inside the 1-2 %
#                             conversion bias they exist to measure.
# The in-air setup `Sc` is measured on: a water cube centred on the measurement point,
# air elsewhere. 4 cm puts 2 cm of water above the point -- past dmax, so there is
# charged-particle equilibrium -- while staying far too small to build up phantom
# scatter. The absolute value depends on this size; `Sc` is only ever used here as a
# *ratio* between field sizes, and the small residual scatter is common to all of them,
# so the ratio is insensitive to it. Set MINI_PHANTOM_SIDE = None to skip the in-air
# runs and fall back to Sp ~ Scp (see the TPR section for what that costs).
MINI_PHANTOM_SIDE: float | None = 4.0
MINI_PHANTOM_ROI = 0.5  # half-width of the in-air measuring cube, cm, capped below by
#                         MINI_PHANTOM_ROI_FRACTION of the field: the cube has to sit
#                         inside the radiation field, and a fixed 0.5 cm does not for the
#                         smallest ones. At 5 mm it reached past the penumbra and gave
#                         Sc = 0.14, hence Sp = 3.2 -- impossible, and the symptom to
#                         watch for if these numbers are ever changed.
MINI_PHANTOM_ROI_FRACTION = 0.15
# Of PER_REPLICATE. Larger than 1 looks wasteful for a single number and is not: the fan
# is sized to the field, so the fraction of histories that even reach a 4 cm cube falls
# as 1/field^2 -- at 400 mm it is ~5e-4, against ~4e-3 at 40 mm. The in-air runs are the
# statistics-starved ones, not the cheap ones. They stay affordable because the grid is
# tiny and air barely interacts. At 0.15 with a 0.5 cm ROI the resulting Sc was visibly
# non-monotonic in field size, which is how this was found. 0.6 puts the sigma on Sc at a
# few parts in a thousand at the shipped HISTORIES, which is well under the Scp error it
# divides into; it is a fraction of PER_REPLICATE, so raise it if you lower those.
MINI_PHANTOM_HISTORY_FRACTION = 0.6

# --- the field sweep ------------------------------------------------------------------
FIELDS_MM = (
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
    150,
    200,
    300,
    400,
)  # square field side at the isocentre
# The full commissioning sweep; expect several minutes and raise HISTORIES with it:
# FIELDS_MM = (5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400)
REFERENCE_FIELD_MM = 100  # the field output factors are normalized to

# --- statistics -------------------------------------------------------------------------
HISTORIES = int(1e10)  # per field at TARGET_SIGMA, split over the replicates
REPLICATES = 20  # independent runs per field; their spread is every error bar
# Equal histories per field do *not* buy equal precision. Measured on this geometry at
# HISTORIES each, the on-axis relative error runs 0.44 % at 5 mm, dips to 0.19 % at 20 mm,
# and climbs to 1.00 % at 400 mm -- a factor of five spread, because a large field's fan is
# mostly outside the aperture while a small field's scoring volume is a single voxel. Both
# ends are wasteful in opposite directions: the middle is over-resourced and the large
# fields are under-resourced.
#
# So histories are allocated to a *target* error instead. Error scales as 1/sqrt(N), so the
# per-field count is HISTORIES x (measured sigma / TARGET_SIGMA)^2, using the table below.
# Levelling at 0.7 % takes roughly 40 % off the sweep against equal allocation, and makes
# every error bar mean the same thing -- which matters more than the time, since the bars
# are what the comparisons are read against.
#
# MEASURED_SIGMA must be re-derived if the geometry changes, and the fan margin above is
# exactly such a change: at 5 mm it took the error from 0.44 % to 0.14 %. Re-derive by
# running one sweep with TARGET_SIGMA = None (equal allocation) and reading `dose_rel_sigma`.
#
# Read what the target actually binds before tuning it. With the table below, nine of the
# thirteen fields land on HISTORY_SCALE_FLOOR rather than on the target -- their on-axis
# error is so far inside 0.7 % that the allocation would starve their *curves* first, which
# is what the floor exists to stop. So `TARGET_SIGMA` only really governs 100 mm and up,
# and lowering it buys precision on those five fields at close to linear cost in wall
# clock. It does not speed anything up: the floor is already the cheaper constraint.
TARGET_SIGMA: float | None = 0.007
# The reference field is not like the others: every output factor is a ratio against it, so
# its error lands on all thirteen at once and does not cancel anywhere. Left on the same
# target it was the second-noisiest field in the sweep at 0.73 %, and the whole Scp column
# duly shifted by about that much against the previous run -- every field the same way,
# which is the giveaway. It gets its own, tighter target; the extra histories buy thirteen
# comparisons, not one.
REFERENCE_TARGET_SIGMA = 0.002
# Floor on the per-field scale. The target is set on the *on-axis* error, but the same
# histories also buy the depth doses, the profiles and the penumbrae, and those do not
# converge at the same rate -- a 10 mm depth dose is read from a single-voxel column and
# already carries ~1 % point noise at the flat allocation. Without a floor the small fields
# would drop to 0.03-0.08x and their curves would fall apart while their ROI number looked
# fine. The floor costs almost nothing (0.48x -> 0.52x of the flat allocation) because the
# sweep's wall clock is dominated by the large fields either way.
HISTORY_SCALE_FLOOR = 0.15
MEASURED_SIGMA = {  # field_mm -> dose_rel_sigma at HISTORIES, under the margins above
    5: 0.00100,
    10: 0.00125,
    20: 0.00099,
    30: 0.00095,
    40: 0.00146,
    50: 0.00168,
    60: 0.00200,
    80: 0.00197,
    100: 0.00292,
    150: 0.00347,
    200: 0.00336,
    300: 0.00452,
    400: 0.00487,
}
# Head primaries per replicate, as a *reuse cap* on the transport count rather than a fixed
# number. The transport resamples the pre-solved phase space, so a small phase space puts a
# floor under the error that no amount of transport can lift: at 100 mm with 2e9 histories,
# growing it from 1e6 (167x reuse) to 16e6 (10x) took the on-axis error from 0.73 % to
# 0.35 %, at fixed transport. Measured as sigma^2 x time the optimum is broad -- 120 at
# 167x, 87 at 42x, 80 at 10x -- so ~40x sits at the knee and 10x buys little for 2x the
# wall clock. A fixed 1e6 also decoupled the error from the history allocation entirely,
# which is why levelling on TARGET_SIGMA barely worked before this.
PHASE_SPACE_REUSE = 40
PRESOLVE_HISTORIES_MIN = 200_000  # keeps the smallest runs from pre-solving a stub
SEED = 20260727
BACKEND = "auto"  # "auto" prefers Warp (CUDA if present); "ref" forces the oracle
SHOW_PLOTS = False  # True calls plt.show(). Under an interactive backend (tkagg here)
#                     that BLOCKS the script until each window is closed; the figures are
#                     saved either way, and a notebook displays them inline regardless.

# %% [markdown]
# ## Imports, and what we are running on
#
# The Warp backend is optional; without it this falls back to the reference engine,
# which is the correctness oracle and is not optimized — expect it to be far too slow
# for the full sweep.

# %%
import csv  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, NamedTuple  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pyRadMC.backends.ref.engine import ReferenceEngine  # noqa: E402
from pyRadMC.data.materials import AIR, TUNGSTEN, WATER  # noqa: E402
from pyRadMC.geometry.collimation import (  # noqa: E402
    MLC,
    BeamFrame,
    BeamLimitingStack,
    CollimatedSource,
    JawPair,
    project_between_planes,
)
from pyRadMC.geometry.fluence import RadialFluence  # noqa: E402
from pyRadMC.geometry.grid import VoxelGrid  # noqa: E402
from pyRadMC.geometry.head import AirColumn, presolve_head  # noqa: E402
from pyRadMC.geometry.source import (  # noqa: E402
    GaussianSpotBeamSource,
    PrimaryFluenceBeamSource,
)
from pyRadMC.geometry.spectrum import ali_rogers_mv  # noqa: E402
from pyRadMC.rng.host import HostRNG  # noqa: E402

try:
    import warp as wp

    from pyRadMC.backends.warp.engine import WarpEngine
    from pyRadMC.backends.warp.presolve import presolve_head_device

    WARP_DEVICE: str | None = "cuda:0" if wp.is_cuda_available() else "cpu"
except ImportError:  # pragma: no cover - the example degrades to the oracle
    WarpEngine = None  # type: ignore[assignment,misc]
    WARP_DEVICE = None

if BACKEND == "ref":
    WARP_DEVICE = None
print(f"transport on {WARP_DEVICE or 'ref (reference engine)'}")

# %% [markdown]
# ## Resolved configuration
#
# A phase space may carry electrons and the pre-solve transports photons only, so a
# phase-space input takes the deterministic wrapper.

# %%
FIELDS = sorted({*(mm / 10.0 for mm in FIELDS_MM), REFERENCE_FIELD_MM / 10.0})
REFERENCE_FIELD = REFERENCE_FIELD_MM / 10.0
HEAD_MODEL = "attenuation" if PHASESPACE is not None else HEAD
PER_REPLICATE = max(1, HISTORIES // REPLICATES)


def histories_per_replicate(field: float) -> int:
    """Per-replicate histories for one field (cm), allocated to `TARGET_SIGMA`.

    `None` restores the flat allocation, which is what to use when re-deriving
    `MEASURED_SIGMA`. Interpolated in log-log between the tabulated fields and clamped
    outside them; the ceiling on the scale is a guard against a stale table sending a run
    to an extreme, not a physical statement. The floor is `HISTORY_SCALE_FLOOR`.

    `MEASURED_SIGMA` pools two independent runs at the shipped geometry: a flat-allocation
    calibration at 2e9 per field, and the sweep itself read back through its own allocation.
    They agree to ~20 % below 50 mm, which is about what a 20-replicate error estimate can
    resolve, but from 60 mm up the sweep came out 1.2-1.5x noisier than the calibration, six
    fields running the same way -- too consistent to be estimator noise, and unexplained.
    Pooling splits the difference; the practical effect of getting it wrong is that a field
    misses the target, which the achieved `dose_rel_sigma` reports every run. Re-derive
    against that column rather than trusting this table if precision starts mattering.
    """
    if TARGET_SIGMA is None:
        return PER_REPLICATE
    fields = sorted(MEASURED_SIGMA)
    sigma = float(
        np.exp(
            np.interp(
                np.log(field * 10.0),
                np.log(fields),
                [math.log(MEASURED_SIGMA[f]) for f in fields],
            )
        )
    )
    target = REFERENCE_TARGET_SIGMA if field == REFERENCE_FIELD else TARGET_SIGMA
    scale = min(20.0, max(HISTORY_SCALE_FLOOR, (sigma / target) ** 2))
    return max(1, int(scale * PER_REPLICATE))


SPECTRUM = ali_rogers_mv(BEAM)
# The table's radii are quoted at the isocentre, so its reference distance is the SAD.
FLUENCE = (
    RadialFluence.from_file(PRIMARY_FLUENCE, reference_distance=SAD)
    if PRIMARY_FLUENCE is not None
    else None
)
# Where the figures and tables are written; the notebook has no __file__ of its own.
STEM = Path(__file__).with_suffix("") if "__file__" in dir() else Path("commissioning_demo")

print(
    f"{BEAM}: SAD {SAD:g} cm, SSD {SSD:g} cm, measurement depth {DEPTH:g} cm; "
    f"collimator {COLLIMATOR!r}, head {HEAD_MODEL!r}, devices {DEVICES!r}"
)
print(f"fields (mm at iso): {[f * 10 for f in FIELDS]}")
if FLUENCE is not None:
    print(
        f"primary fluence: {PRIMARY_FLUENCE} — {FLUENCE.radii.size} points to "
        f"{FLUENCE.max_radius:g} cm at iso, peak {FLUENCE.values.max():.3f} "
        f"at {FLUENCE.radii[int(np.argmax(FLUENCE.values))]:g} cm"
    )
else:
    print("primary fluence: uniform (no flattening-filter shaping)")

if EXTRAFOCAL_WEIGHT is None:
    print("extra-focal source: off (Scp will carry no collimator-scatter component)")
else:
    # A uniform disc of radius R has the same second moment as a Gaussian of
    # sigma = R/2, which is how EXTRAFOCAL_SIGMA is set from the collimator opening.
    _implied_r = 2.0 * EXTRAFOCAL_SIGMA * SAD / EXTRAFOCAL_Z
    print(
        f"extra-focal source: w = {EXTRAFOCAL_WEIGHT:.3f}, sigma {EXTRAFOCAL_SIGMA:g} cm at "
        f"z = {EXTRAFOCAL_Z:g} cm, emitting {EXTRAFOCAL_FIELD:g} cm at iso\n"
        f"  (that sigma is the equal-variance disc of radius {_implied_r:.1f} cm at iso; "
        f"the primary collimator's own opening for comparison)"
    )

# %% [markdown]
# ## Materials and cross-sections
#
# `tungsten` is the real thing and needs the compiled EPICS tables (cached after the
# first download). `water` is the dependency-free stand-in on the analytic,
# water-only source: water at *tungsten's* density, because a water-density block
# would transmit ~70 % and there would be no field to measure, plus water at air's
# density above the phantom. Its low-energy behaviour is not tungsten's — no K edge,
# no photoelectric dominance — so leakage and the deep transmission tail are
# indicative only.

# %%
if DEVICES == "tungsten":
    from pyRadMC.data.tabulated.build import download_library
    from pyRadMC.data.tabulated.precompile import compile_materials
    from pyRadMC.data.tabulated.source import TabulatedCrossSections

    DEVICE_MATERIAL, DEVICE_DENSITY = TUNGSTEN, 19.30
    AIR_MATERIAL, AIR_DENSITY = AIR, 1.205e-3
    print("compiling the tabulated table (EPICS cache/download) ...")
    _tables = compile_materials(
        download_library("epdl").read_text(encoding="latin-1"),
        download_library("eedl").read_text(encoding="latin-1"),
        e_max=max(7.0, SPECTRUM.max_energy + 1.0),
    )
    XS: Any = TabulatedCrossSections(_tables, geometry_densities=((WATER, 1.0), (AIR, AIR_DENSITY)))
else:
    from pyRadMC.data.analytic import AnalyticCrossSections

    DEVICE_MATERIAL, DEVICE_DENSITY = WATER, 19.30
    AIR_MATERIAL, AIR_DENSITY = WATER, 1.205e-3
    # Air is water at air's density here, so water's own maximum is the majorant.
    XS = AnalyticCrossSections(geometry_densities=((WATER, 1.0),))

print(XS.provenance)

# %% [markdown]
# ## The beam-limiting devices
#
# A plan states field settings at the isocentre plane; `project_between_planes`
# carries them to each device mid-plane along the divergent rays.
#
# `jaws` limits x and y with jaw pairs. `jaw-mlc` is the arrangement this head uses:
# the leaves travel along x and their rounded tips define that edge, while the y jaw
# pair defines the other one; leaves beyond the field plus `MLC_Y_MARGIN` are parked
# closed. `jaws-mlc` adds the x jaw pair behind the tips, as a real head has.
#
# Two settings here matter more than they look, both calibration rather than physics.
# `MLC_FOCUSED_SIDES` gives the leaves the trapezoid cross-section of a real
# single-focusing MLC: with parallel sides a strip only clears the whole slab for rays
# under `v / z_bottom`, so a bank parked at the field edge shadows it from about 12 %
# inside and narrows the radiation field proportionally (it measured -2.8 % here).
# `MLC_TIP_RADIUS` sets how gradual the leaf-end falloff is, and the tip coordinate is
# the *physical* apex — the `MLC` class states that any light-field/radiation-field
# offset a vendor quotes is the caller's to apply. `MLC_TIP_CALIBRATION` is this
# notebook's, applied below; before it existed `width_x` ran ~0.7 mm long at every field
# under 100 mm and short above it.


# %%
def tip_calibration(position: float) -> float:
    """Leaf-position calibration at a tip's own off-axis position (cm at the isocentre).

    Linear between the tabulated positions and held flat outside them, which the swept
    fields never reach; see `MLC_TIP_CALIBRATION` for where the table comes from.
    """
    positions = [entry[0] for entry in MLC_TIP_CALIBRATION]
    deltas = [entry[1] for entry in MLC_TIP_CALIBRATION]
    return float(np.interp(abs(position), positions, deltas))


def fan_margin(field: float) -> float:
    """Fan (and MLC bank) reach past one edge of a field, cm at the isocentre.

    Ramps from the tight value to the wide one across the measured crossover; see
    `FAN_MARGIN_MIN` for what tightening buys and where it stops paying.
    """
    return min(FAN_MARGIN_MAX, max(FAN_MARGIN_MIN, FAN_MARGIN_SLOPE * field))


def beam_limiting_stack(field: float) -> BeamLimitingStack:
    """Set the devices for one square field stated at the isocentre."""
    half = 0.5 * field

    def at(setting: float, z_range: tuple[float, float]) -> float:
        return float(project_between_planes(setting, SAD, sum(z_range) / 2.0))

    def jaw(axis: str, z_range: tuple[float, float]) -> JawPair:
        return JawPair(
            axis=axis,
            z_top=z_range[0],
            z_bottom=z_range[1],
            edge_neg=at(-half, z_range),
            edge_pos=at(half, z_range),
            material=DEVICE_MATERIAL,
            density=DEVICE_DENSITY,
            focused=FOCUSED_JAWS,
        )

    devices: list[JawPair | MLC] = []
    if COLLIMATOR in ("jaws", "jaws-mlc"):
        devices.append(jaw("u", JAW_U_Z))
    devices.append(jaw("v", JAW_V_Z))
    if COLLIMATOR in ("jaw-mlc", "jaws-mlc"):
        # The bank spans the field plus the fan margin only: the source fan is sized
        # the same way, so leaves further out would never be crossed. Strip edges are
        # symmetric about the axis (a leaf boundary on the axis, Varian-like).
        n_pairs = 2 * max(1, math.ceil((half + fan_margin(field)) / MLC_PITCH))
        edges = (np.arange(n_pairs + 1) - n_pairs / 2.0) * MLC_PITCH
        # Opened past the field by MLC_Y_MARGIN so the y jaws, not the parked leaves,
        # define that edge -- which is how this head actually collimates.
        reach = half + MLC_Y_MARGIN
        open_pair = [edges[i] < reach and edges[i + 1] > -reach for i in range(n_pairs)]
        # The tips carry the rounded-end calibration; the strip edges above do not. It is
        # a function of each tip's own position, so it carries over to asymmetric banks.
        reach_u = half + tip_calibration(half)
        z_mid = sum(MLC_Z) / 2.0

        def to_mid(values: Any) -> tuple[float, ...]:
            return tuple(float(project_between_planes(v, SAD, z_mid)) for v in values)

        devices.append(
            MLC(
                z_top=MLC_Z[0],
                z_bottom=MLC_Z[1],
                leaf_edges_v=to_mid(edges),
                tips_neg=to_mid([-reach_u if is_open else 0.0 for is_open in open_pair]),
                tips_pos=to_mid([reach_u if is_open else 0.0 for is_open in open_pair]),
                tip_radius=MLC_TIP_RADIUS,
                material=DEVICE_MATERIAL,
                density=DEVICE_DENSITY,
                focused_sides=MLC_FOCUSED_SIDES,
            )
        )
    return BeamLimitingStack(frame=BeamFrame(origin=(0.0, 0.0, 0.0)), devices=tuple(devices))


# A quick look at what that produced for the reference field.
for device in beam_limiting_stack(REFERENCE_FIELD).devices:
    print(f"  {type(device).__name__:8s} z = {device.z_top:5.1f} .. {device.z_bottom:5.1f} cm")

# %% [markdown]
# ## A consistent air path
#
# This is what makes a TPR curve usable in the build-up region, so it is worth being
# explicit about.
#
# Build-up dose is dominated by contaminant electrons born in the air above the
# phantom, and how many arrive depends on how far the beam travelled through air. The
# natural thing — run air from the collimator exit down to whatever surface the setup
# happens to have — gives a *different* air path in every run: 48.5 cm for a 95 cm SSD,
# 33.5 cm for the SAD setup at 20 cm depth. The depth doses and the TPR runs would then
# have systematically different contaminant components, and converting between them in
# build-up would be comparing two different beams.
#
# So the air path is fixed to one length for every simulation: the longest that fits in
# the shallowest setup, which is the one whose phantom surface sits closest to the
# devices. Above it is vacuum, which contributes nothing and can therefore differ
# between setups without consequence. The path is split into a transported part
# (`AIR_GAP` centimetres of real air voxels, full Monte Carlo) and the pre-solve's
# analytic column above it, which reaches further for the same cost.
#
# The stack exit is field-independent, so this is one number for the whole notebook.

# %%
STACK_EXIT = beam_limiting_stack(REFERENCE_FIELD).exit_z
SHALLOWEST_SSD = min([SSD, *(SAD - depth for depth in TPR_DEPTHS)])
AIR_PATH = SHALLOWEST_SSD - STACK_EXIT
# Quantized *down* to the voxel grid, and that matters: the phantoms lay out their air as
# `round(TRANSPORTED_AIR / SPACING)` voxels, so a value that is not a multiple of SPACING
# can round up, put the grid ceiling above the device exit, and fail the pre-solve with
# "exit_z is above the stack exit". It only bites when the air path is both short enough to
# be transported whole and not a multiple of the voxel -- a deep SAD-setup depth does it,
# since that is what makes SHALLOWEST_SSD small. Flooring here and giving the remainder to
# the pre-solved column keeps the total air path exactly `AIR_PATH` either way.
TRANSPORTED_AIR = (int(min(AIR_GAP, AIR_PATH) / SPACING)) * SPACING
PRESOLVE_AIR = AIR_PATH - TRANSPORTED_AIR
# Asserted rather than left implicit. This flooring has been silently dropped once already
# and the symptom surfaces far away, inside the pre-solve, only for TPR_DEPTHS deep enough
# to make the air path short -- which no default run exercises.
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
if TRANSPORTED_AIR < AIR_GAP:
    print(
        f"  (AIR_GAP {AIR_GAP:g} cm did not fit and was reduced; the shallowest setup is "
        f"SSD {SHALLOWEST_SSD:g} cm against a device exit at {STACK_EXIT:g} cm)"
    )

# %% [markdown]
# ## The source
#
# Photons are born on a plane above the devices, aimed from a Gaussian focal spot —
# the finite-source-size model that gives the geometric penumbra. The rectangle is
# the field plus `fan_margin(field)` per side, *projected back* from the isocentre, so every
# field is simulated with the same fluence per unit area at the isocentre plane.
#
# That last point is what makes output factors comparable across fields. The returned
# area is the factor converting dose-per-history into dose per unit incident fluence:
# a small field with a small fan gets more histories per unit fluence, and this is
# what cancels it.
#
# With `PRIMARY_FLUENCE` set, the same geometry carries the measured radial fluence as
# each history's statistical weight, so the fan is no longer uniform: it has the
# filter's horn, and for a large enough field its corners run into the primary
# collimator's roll-off and are emitted at weight zero. The area normalization is
# divided by the on-axis fluence so that "per unit incident fluence" still means *per
# unit fluence on the axis* — a field-independent factor, so output factors are
# unaffected either way, but the absolute `dose_per_fluence` column is.


# %%
def head_input_source(field: float) -> tuple[Any, float]:
    """Give the raw source for one field and its aperture area at the isocentre."""
    if PHASESPACE is not None:
        from pyRadMC.geometry.phasespace import PhaseSpaceSource

        # The file defines the fluence; there is no aperture to normalize by.
        return PhaseSpaceSource(PHASESPACE), 1.0
    width_iso = field + 2.0 * fan_margin(field)
    width_plane = float(project_between_planes(width_iso, SAD, EMISSION_Z))
    geometry: dict[str, Any] = dict(
        spectrum=SPECTRUM,
        focal_point=(0.0, 0.0, 0.0),
        center=(0.0, 0.0, EMISSION_Z),
        width_u=width_plane,
        width_v=width_plane,
        sigma_u=SPOT_SIGMA,
        sigma_v=SPOT_SIGMA,
    )
    if FLUENCE is None:
        return GaussianSpotBeamSource(**geometry), width_iso * width_iso
    source = PrimaryFluenceBeamSource(fluence=FLUENCE, **geometry)
    return source, width_iso * width_iso / FLUENCE.at_radius(0.0)


# %% [markdown]
# ## The extra-focal source
#
# The same planar geometry with two things changed: the focal spot moves down to the
# flattening filter and gets wide. Everything that makes this behave like head scatter
# follows from that. A point in the field sees only the part of the extended source that
# the jaws and MLC leave unobstructed, so opening the collimator admits more of it and
# `Sc` rises with field size — with no scattering geometry modelled anywhere.
#
# Two deliberate choices. The emitting square is the **primary collimator's** opening and
# is therefore *field-independent*: extra-focal radiation illuminates the whole cone
# whatever the jaws are doing, and the jaws are what cut it. And the source carries no
# radial shaping — it is a plain wide Gaussian, the standard first-order model.
#
# The weight is a fitted parameter, the only one here. Everything else comes from the
# machine data: the plane from `params.dat`, the width from the primary collimator
# opening that `primflu.dat` itself implies.


# %%
def on_axis_emitted_area(source: Any, n: int = 4_000_000, half: float = 1.0) -> float:
    """Effective isocentre-plane area per emitted history, measured from the source.

    A source's on-axis emitted fluence is ``histories / this area``, which is the
    normalization every output factor is quoted against. It is measured — project a
    sample of the source's own primaries onto the isocentre plane and count the
    weight landing in a small central square — rather than assumed, because the
    naive ``(emitting square)^2`` is only right for rays that emanate from the
    target. The extra-focal source's rays come from the filter plane instead, so
    they diverge far faster and cover several times the area the emitting square
    projects to; assuming otherwise inflates its fitted weight by that ratio.
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
    """Give the wide-Gaussian head-scatter source and its effective isocentre area.

    Field-independent by construction, so it is built once; only the stack it is
    later collimated by changes between fields.
    """
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


# Field-independent, so it is built and normalized once for the whole notebook.
EXTRAFOCAL_RAW, EXTRAFOCAL_AREA = (None, 0.0) if EXTRAFOCAL_WEIGHT is None else extrafocal_source()
if EXTRAFOCAL_WEIGHT is not None:
    print(
        f"extra-focal normalization: effective {EXTRAFOCAL_AREA:.0f} cm^2 at the isocentre "
        f"plane per history, against {EXTRAFOCAL_FIELD**2:.0f} cm^2 if its rays came from "
        f"the target ({EXTRAFOCAL_AREA / EXTRAFOCAL_FIELD**2:.1f}x wider, because they do not)"
    )


# %% [markdown]
# ## The phantom
#
# Air over water, centred on the beam axis, with an odd number of lateral voxels so
# that one voxel sits on the axis. The lateral extent covers the field projected to
# the deepest plane plus `LATERAL_MARGIN`.


# %%
def mini_phantom(ssd: float, depth: float, side: float) -> VoxelGrid:
    """Air everywhere but a small water cube centred on the measurement point.

    This is the in-air setup that `Sc` is defined on: enough water around the point
    for charged-particle equilibrium, little enough that almost no phantom scatter
    reaches it. The air above starts where `water_phantom` starts it, so the
    transported air path is the one every other setup uses; the extra centimetres
    between that surface and the cube are air the full phantom would have had as
    water, which is the whole point of the geometry.
    """
    half = 0.5 * side
    n_air = round(TRANSPORTED_AIR / SPACING)
    top = ssd - n_air * SPACING  # the transported air starts here, as in water_phantom
    n_z = round((ssd + depth + half - top) / SPACING)
    n_lateral = math.ceil(2.0 * (half + 1.0) / SPACING)
    n_lateral += n_lateral % 2 == 0  # odd, so a voxel is centred on the axis
    shape = (n_lateral, n_lateral, n_z)
    density = np.full(shape, AIR_DENSITY, dtype=np.float64)
    material = np.full(shape, AIR_MATERIAL, dtype=np.int32)
    z = top + (np.arange(n_z) + 0.5) * SPACING
    lateral = (np.arange(n_lateral) + 0.5 - 0.5 * n_lateral) * SPACING
    cube = (
        (np.abs(lateral) <= half)[:, None, None]
        & (np.abs(lateral) <= half)[None, :, None]
        & (np.abs(z - (ssd + depth)) <= half)[None, None, :]
    )
    density[cube] = 1.0
    material[cube] = WATER
    return VoxelGrid(
        shape=shape,
        spacing=(SPACING, SPACING, SPACING),
        density=density,
        material=material,
        origin=(-0.5 * n_lateral * SPACING, -0.5 * n_lateral * SPACING, top),
    )


def water_phantom(field: float, ssd: float, phantom_depth: float) -> VoxelGrid:
    """Build the air-over-water phantom for one field and placement."""
    half = 0.5 * field * (ssd + phantom_depth) / SAD + LATERAL_MARGIN
    n_lateral = math.ceil(2.0 * half / SPACING)
    n_lateral += n_lateral % 2 == 0  # odd, so a voxel is centred on the axis
    n_air = round(TRANSPORTED_AIR / SPACING)
    shape = (n_lateral, n_lateral, n_air + round(phantom_depth / SPACING))
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
        origin=(-0.5 * n_lateral * SPACING, -0.5 * n_lateral * SPACING, ssd - n_air * SPACING),
    )


# %% [markdown]
# ## Reading a dose distribution
#
# A `Scan` is one dose array plus what is needed to interpret it. Curves are read
# from a column a fifth of the field wide, capped at 1.5 cm and floored at one voxel:
# averaging *across* a square field is nearly distortion-free — the edges are
# straight, so the orthogonal direction is flat wherever the profile is not — and it
# buys the factor of several in variance that makes a curve readable. The output
# factor deliberately does not use that column: a reported dose is a point
# measurement, a plotted curve is not.


# %%
class Scan(NamedTuple):
    """One dose distribution and the geometry needed to read it."""

    field: float
    ssd: float
    depth: float
    grid: VoxelGrid
    dose: np.ndarray


def coordinates(grid: VoxelGrid, axis: int) -> np.ndarray:
    """Voxel-centre coordinates along one grid axis (cm, engine frame)."""
    return grid.origin[axis] + (np.arange(grid.shape[axis]) + 0.5) * grid.spacing[axis]


def within(grid: VoxelGrid, axis: int, half: float) -> np.ndarray:
    """Mask of voxels whose centres lie within ``half`` of the axis."""
    return np.abs(coordinates(grid, axis)) <= half + 1.0e-9


def column_half(scan: Scan, axis: int) -> float:
    """Half-width of the averaging column used for curves (never for the ROI)."""
    return max(min(0.2 * scan.field, 1.5), 0.4 * scan.grid.spacing[axis])


def roi_dose(scan: Scan, depth: float, half: float | None = None) -> float:
    """Mean dose in the on-axis measuring volume at a depth.

    A ~0.5 cm cube, chamber-like, for fields that can hold one; below 2 cm the single
    central voxel, which is then volume-averaged over 2.5 mm and noisier. `half`
    overrides that choice, which the in-air runs need: nearly every history misses a
    mini-phantom, so they buy their statistics with a larger measuring volume.
    """
    if half is None:
        half = 0.25 if scan.field >= 2.0 else 0.4 * scan.grid.spacing[0]
    z = coordinates(scan.grid, 2) - scan.ssd
    in_z = np.abs(z - depth) <= half + 1.0e-9
    if not in_z.any():
        in_z = np.zeros_like(z, dtype=bool)
        in_z[int(np.argmin(np.abs(z - depth)))] = True
    block = scan.dose[np.ix_(within(scan.grid, 0, half), within(scan.grid, 1, half), in_z)]
    return float(block.mean())


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


def lateral_profile(scan: Scan, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Off-axis coordinate and dose along one principal axis at the measurement depth."""
    z = coordinates(scan.grid, 2) - scan.ssd
    plane = scan.dose[:, :, int(np.argmin(np.abs(z - scan.depth)))]
    other = 1 - axis
    mask = within(scan.grid, other, column_half(scan, other))
    profile = plane[:, mask].mean(axis=1) if axis == 0 else plane[mask, :].mean(axis=0)
    return coordinates(scan.grid, axis), profile


def sample_plane(values: np.ndarray, grid: VoxelGrid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def diagonal_profile(scan: Scan, sign: float) -> tuple[np.ndarray, np.ndarray]:
    """Radial coordinate and dose along one field diagonal.

    Deliberately *not* averaged over a perpendicular band the way the axis scans are:
    across a diagonal a square field is not flat — the band runs in and out through
    the corner — and averaging one measurably moved the 50 % point inward from the
    geometric corner distance. Averaging the two diagonals instead is free.
    """
    grid = scan.grid
    z = coordinates(grid, 2) - scan.ssd
    plane = scan.dose[:, :, int(np.argmin(np.abs(z - scan.depth)))]
    reach = 0.5 * min(grid.shape[0] * grid.spacing[0], grid.shape[1] * grid.spacing[1])
    reach /= math.sqrt(2.0)
    radius = np.linspace(-reach, reach, 2 * int(reach / grid.spacing[0]) + 1)
    leg = radius / math.sqrt(2.0)
    return radius, sample_plane(plane, grid, leg, sign * leg)


def both_diagonals(scan: Scan) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the two diagonals — independent samples of the same physical curve.

    A square field is four-fold symmetric, so this is a free factor of sqrt(2) in
    noise with no geometric distortion at all.
    """
    radius, positive = diagonal_profile(scan, 1.0)
    _, negative = diagonal_profile(scan, -1.0)
    return radius, 0.5 * (positive + negative)


# %% [markdown]
# ## Commissioning metrics from one distribution
#
# Two details that matter more than they look.
#
# The peak of a depth-dose curve is taken from a **parabola fitted through the five
# samples around the largest one**, not from `argmax`. On a Monte Carlo curve
# `argmax` lands on whichever voxel fluctuated highest, which biases `dmax` around
# and the normalizing dose *upward* — dragging every PDD down by several percent at
# demo statistics. The fit uses the curve's own local shape and resolves `dmax` below
# one voxel.
#
# The profile reference level is the **on-axis** dose, not the profile maximum: with
# no flattening filter the maximum can sit in a shoulder, and a 50 % level taken from
# it would misreport the field width.


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


def measure(scan: Scan) -> dict[str, float]:
    """Read every commissioning metric off one dose distribution.

    Called on the pooled scan for the reported values and on each replicate for their
    uncertainties, so a metric and its error bar always come from the same code path.
    """
    depth, curve = depth_dose(scan)
    peak = int(np.argmax(curve))
    lo, hi = max(0, peak - 2), min(curve.size, peak + 3)
    dmax, reference = float(depth[peak]), float(curve[peak])
    if hi - lo >= 3:
        a, b, c = np.polyfit(depth[lo:hi], curve[lo:hi], 2)
        vertex = -b / (2.0 * a) if a < 0.0 else float("nan")
        if depth[lo] <= vertex <= depth[hi - 1]:
            dmax, reference = float(vertex), float(a * vertex * vertex + b * vertex + c)
    d10 = float(np.interp(10.0, depth, curve))
    d20 = float(np.interp(20.0, depth, curve))
    metrics = {
        "roi_dose": roi_dose(scan, scan.depth),
        "dmax": dmax,
        "pdd10": 100.0 * d10 / reference,
        "pdd20": 100.0 * d20 / reference,
        "d20_d10": d20 / d10 if d10 > 0.0 else float("nan"),
    }
    for axis, name in ((0, "x"), (1, "y")):
        offsets, values = lateral_profile(scan, axis)
        centre = float(np.interp(0.0, offsets, values))
        levels = {
            f"{side}{pct}": crossing(offsets, values, 0.01 * pct * centre, rising=side == "l")
            for pct in (20, 50, 80)
            for side in "lr"
        }
        width = levels["r50"] - levels["l50"]
        metrics[f"width_{name}"] = width
        metrics[f"penumbra_{name}"] = 0.5 * (
            (levels["l80"] - levels["l20"]) + (levels["r20"] - levels["r80"])
        )
        band = np.abs(offsets) <= 0.4 * width if np.isfinite(width) else np.zeros(0, bool)
        if np.count_nonzero(band) < 3 or centre <= 0.0:
            metrics[f"flatness_{name}"] = metrics[f"symmetry_{name}"] = float("nan")
        else:
            inside = values[band]
            mirrored = np.interp(-offsets[band], offsets, values)
            metrics[f"flatness_{name}"] = (
                100.0 * (inside.max() - inside.min()) / (inside.max() + inside.min())
            )
            metrics[f"symmetry_{name}"] = 100.0 * float(np.max(np.abs(inside - mirrored)) / centre)
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
# The replicates share nothing but the geometry: each reseeds the head model *and*
# the transport. Their pooled mean is the reported dose; their spread is the reported
# uncertainty. Reseeding the pre-solve matters — replicates sharing one phase space
# would share its latent variance and report a spread that is too small — and it is
# also what lets each replicate cheaply resample a small phase space many times.


# %%
class FieldResult(NamedTuple):
    """One field: the pooled scan, its per-replicate metrics, and its normalization.

    ``component_roi`` carries each source component's own dose per unit on-axis
    emitted fluence at the measurement point, *before* the extra-focal weight is
    applied. Because transport is linear in the source, that is what lets the
    weight be re-fitted from the saved table without re-running anything.
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
    engine: Any, raw: Any, stack: BeamLimitingStack, grid: VoxelGrid, n: int, seed: int
) -> tuple[np.ndarray, float]:
    """Run the head model and transport for one source component and one replicate.

    Returns the dose per emitted history of *that component* and the seconds spent
    in the head model. Both components go through this identical path, which is
    what makes their doses linearly combinable further down.
    """
    mark = time.perf_counter()
    if HEAD_MODEL == "attenuation":
        source: Any = CollimatedSource(raw, stack, XS)
    else:
        exit_z = float(grid.origin[2])
        arguments = {
            "stack": stack,
            "cross_sections": XS,
            "n_histories": min(n, max(PRESOLVE_HISTORIES_MIN, n // PHASE_SPACE_REUSE)),
            "seed": seed,
            "exit_z": exit_z,
            # Fixed length, not "from the device exit": the air path must be the
            # same in every setup or the build-up regions are not comparable.
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
    result = engine.run(source, n_histories=n, n_batches=1, seed=seed + 1)
    return np.asarray(result.dose), head_seconds


def simulate(
    field: float,
    ssd: float,
    depth: float,
    phantom_depth: float,
    history_fraction: float = 1.0,
) -> FieldResult:
    """Transport one field as `REPLICATES` independent runs and pool them.

    With an extra-focal component the two sources are transported *separately* and
    combined as ``(1 - w) D_primary + w (A_e / A_p) D_extrafocal``. Transport is
    linear in the source, so that is exact rather than an approximation — and it
    keeps each component's own dose available for re-fitting `w`.

    `history_fraction` scales the per-replicate histories. The SAD-setup runs use it:
    they are read for a single on-axis value, which converges far sooner than the map
    the fixed-SSD runs are scored for.
    """
    per_replicate = max(1, int(history_fraction * histories_per_replicate(field)))
    grid = water_phantom(field, ssd, phantom_depth)
    stack = beam_limiting_stack(field)
    raw, area_iso = head_input_source(field)
    if WARP_DEVICE is not None:
        engine: Any = WarpEngine(grid=grid, cross_sections=XS, device=WARP_DEVICE)
    else:
        engine = ReferenceEngine(grid=grid, cross_sections=XS, rng=HostRNG())

    weight = 0.0 if EXTRAFOCAL_WEIGHT is None else float(EXTRAFOCAL_WEIGHT)
    extra_raw, extra_area = EXTRAFOCAL_RAW, EXTRAFOCAL_AREA
    extra_n = max(1, int(EXTRAFOCAL_HISTORY_FRACTION * per_replicate))
    # Ratio that puts the extra-focal dose on the primary's per-history footing; the
    # (1 - w) and w factors then make the sum a dose per unit *total* on-axis fluence.
    ratio = extra_area / area_iso

    voxels = int(np.prod(grid.shape))
    extra_note = f" + {extra_n:,} extra-focal" if weight > 0.0 else ""
    print(
        f"  {field * 10:6.0f} mm: {grid.shape} ({voxels / 1e6:.1f} M voxels), "
        f"{len(stack.devices)} devices, {REPLICATES} x {per_replicate:,}{extra_note} histories"
    )
    pooled = np.zeros(grid.shape, dtype=np.float64)
    replicates: list[dict[str, float]] = []
    component_totals = {"primary": 0.0, "extrafocal": 0.0}
    head_seconds = 0.0
    start = time.perf_counter()
    for index in range(REPLICATES):
        seed = SEED + 1013 * index
        primary_dose, spent = transport_component(engine, raw, stack, grid, per_replicate, seed)
        head_seconds += spent
        combined = (1.0 - weight) * primary_dose
        component_totals["primary"] += (
            roi_dose(Scan(field, ssd, depth, grid, primary_dose), depth) * area_iso
        )
        if weight > 0.0:
            # A separate seed stream: the two components must not share focal-spot draws.
            extra_dose, spent = transport_component(
                engine, extra_raw, stack, grid, extra_n, seed + 500_003
            )
            head_seconds += spent
            combined += weight * ratio * extra_dose
            component_totals["extrafocal"] += (
                roi_dose(Scan(field, ssd, depth, grid, extra_dose), depth) * extra_area
            )
        pooled += combined
        replicates.append(measure(Scan(field, ssd, depth, grid, combined)))
    seconds = time.perf_counter() - start
    component_roi = {name: total / REPLICATES for name, total in component_totals.items()}
    # Only the pre-solve is a separable head cost. The attenuation wrapper's cost sits
    # inside engine.run: it host-samples and ray-traces the stack for every chunk.
    note = f", {head_seconds:.1f} s of it pre-solving the head" if HEAD_MODEL == "presolve" else ""
    print(f"          {seconds:.1f} s{note}")
    return FieldResult(
        scan=Scan(field, ssd, depth, grid, pooled / REPLICATES),
        stack=stack,
        replicates=replicates,
        area_iso=area_iso,
        seconds=seconds,
        component_roi=component_roi,
    )


# %% [markdown]
# ## The in-air output ratio `Sc`
#
# Same source, same head, same point — only the phantom changes, to the small cube
# `mini_phantom` builds. The number that comes back is the collimator-scatter half of
# `Scp`, and dividing it out is what leaves the phantom-scatter factor `Sp` the TPR
# conversion actually asks for.
#
# It is a single on-axis value rather than a distribution, so it needs far fewer
# histories than a field run and is cheap next to the sweep.


# %%
def in_air_output(field: float) -> tuple[float, float]:
    """On-axis dose per unit incident fluence in the mini-phantom, and its sigma.

    Combines the two source components exactly as `simulate` does, so the result
    divides into that field's `Scp` without any normalization left over.
    """
    assert MINI_PHANTOM_SIDE is not None
    grid = mini_phantom(SSD, DEPTH, MINI_PHANTOM_SIDE)
    stack = beam_limiting_stack(field)
    raw, area_iso = head_input_source(field)
    engine: Any = (
        WarpEngine(grid=grid, cross_sections=XS, device=WARP_DEVICE)
        if WARP_DEVICE is not None
        else ReferenceEngine(grid=grid, cross_sections=XS, rng=HostRNG())
    )
    weight = 0.0 if EXTRAFOCAL_WEIGHT is None else float(EXTRAFOCAL_WEIGHT)
    ratio = EXTRAFOCAL_AREA / area_iso
    n = max(1, int(MINI_PHANTOM_HISTORY_FRACTION * histories_per_replicate(field)))
    extra_n = max(1, int(EXTRAFOCAL_HISTORY_FRACTION * n))
    # Well inside the radiation field. Below one voxel it collapses to the central voxel
    # alone, as `roi_dose` does for small fields in the full phantom.
    wanted = MINI_PHANTOM_ROI_FRACTION * field
    roi_half = min(MINI_PHANTOM_ROI, wanted) if wanted >= SPACING else 0.4 * SPACING

    values: list[float] = []
    for index in range(REPLICATES):
        seed = SEED + 7919 * index
        primary_dose, _ = transport_component(engine, raw, stack, grid, n, seed)
        combined = (1.0 - weight) * primary_dose
        if weight > 0.0:
            extra_dose, _ = transport_component(
                engine, EXTRAFOCAL_RAW, stack, grid, extra_n, seed + 500_003
            )
            combined += weight * ratio * extra_dose
        values.append(roi_dose(Scan(field, SSD, DEPTH, grid, combined), DEPTH, roi_half) * area_iso)
    mean = float(np.mean(values))
    return mean, float(np.std(values, ddof=1) / math.sqrt(REPLICATES))


# %% [markdown]
# ## Run the sweep
#
# Fixed SSD, one placement, every field.

# %%
print("field sweep:")
runs = {field: simulate(field, SSD, DEPTH, PHANTOM_DEPTH) for field in FIELDS}

# %% [markdown]
# ## Depth doses
#
# Normalized to each field's own maximum, which is the definition of a PDD. Larger
# fields fall off more slowly: more phantom scatter reaching the axis.

# %%
COLOURS = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(FIELDS)))

for colour, field in zip(COLOURS, FIELDS, strict=True):
    depth, curve = depth_dose(runs[field].scan)
    plt.plot(depth, 100.0 * curve / curve.max(), color=colour, lw=1.5, label=f"{field:g} cm")
plt.title(f"Central-axis depth dose (SSD {SSD:g} cm)")
plt.xlabel("depth (cm)")
plt.ylabel("dose (% of own maximum)")
plt.grid(alpha=0.25, lw=0.6)
plt.legend(fontsize=8, ncol=2, title="field at iso")
if SHOW_PLOTS:
    plt.show()

for field in FIELDS:
    pooled = measure(runs[field].scan)
    print(
        f"  {field * 10:6.0f} mm: dmax {pooled['dmax']:5.2f} cm, "
        f"PDD(10) {pooled['pdd10']:5.2f} +- {runs[field].spread('pdd10')[1]:.2f} %, "
        f"D20/D10 {pooled['d20_d10']:.3f}"
    )

# %% [markdown]
# ## Tissue-phantom ratios
#
# A TPR is measured with the point of measurement fixed at the isocentre and the
# overlying depth varied, so each depth is its own phantom placement and its own
# simulation at `SSD = SAD - depth`. That is what makes `TPR(20,10)` a beam-quality
# index independent of the inverse square law, and why it cannot be read off the
# fixed-SSD depth-dose curve above without a phantom-scatter correction this notebook
# does not invent.

# %%
tpr_dose: dict[tuple[float, float], tuple[float, float]] = {}
if TPR_DEPTHS:
    which = FIELDS if TPR_ALL_FIELDS else [REFERENCE_FIELD]
    note = "every field" if TPR_ALL_FIELDS else f"the {REFERENCE_FIELD * 10:g} mm field"
    print(f"SAD-setup runs for {note} ({TPR_HISTORY_FRACTION:g} x histories):")
    for tpr_field in which:
        for tpr_depth in TPR_DEPTHS:
            run = simulate(
                tpr_field,
                SAD - tpr_depth,
                tpr_depth,
                max(PHANTOM_DEPTH, tpr_depth + 10.0),
                history_fraction=TPR_HISTORY_FRACTION,
            )
            tpr_dose[tpr_field, tpr_depth] = (
                roi_dose(run.scan, tpr_depth),
                run.spread("roi_dose")[1],
            )


def simulated_tpr_20_10(field: float) -> tuple[float, float]:
    """Directly simulated TPR(20,10) for one field, or NaN if it was not run."""
    if (field, 20.0) not in tpr_dose or (field, 10.0) not in tpr_dose:
        return float("nan"), float("nan")
    (deep, deep_sem), (shallow, shallow_sem) = tpr_dose[field, 20.0], tpr_dose[field, 10.0]
    ratio = deep / shallow
    return ratio, ratio * math.hypot(deep_sem / deep, shallow_sem / shallow)


TPR_20_10, TPR_20_10_SEM = simulated_tpr_20_10(REFERENCE_FIELD)
if not math.isnan(TPR_20_10):
    print(f"  TPR(20,10) = {TPR_20_10:.4f} +- {TPR_20_10_SEM:.4f}")

# %% [markdown]
# ## Output factors
#
# `Scp` is the dose per unit incident fluence at the measurement point, relative to
# the reference field — which is why each field's dose-per-history is multiplied by
# its aperture area first.
#
# The two equivalent radii are for a beam model parameterized on circular fields: the
# equal-area circle (`a / sqrt(pi)`) preserves the geometric field size, the
# equal-A/P circle (`a / 2`) preserves Sterling's scatter equivalence, which is the
# one that reproduces output factors.
#
# **Below ~15 mm this stops being a check on the model.** Against `of.dat` the sweep sits
# within ±0.94 % from 20 to 400 mm — eleven fields, RMS 0.48 %, none over 1.2 sigma — and
# then the 10 mm field runs +4.5 % at 11 sigma. Simulating `of.dat`'s own field sizes, so
# that nothing is interpolated:
#
#     of.dat mm     5.7433   10.1112   12.0144   19.8299
#     simulated    +19.47     +4.78     +2.12     +1.08   %
#     n sigma        46.7      10.2       5.9       1.5
#
# That is not the interpolation (10 mm was the only swept field landing in a gap of
# `of.dat`, which was worth checking and is worth 0.7 points of the 4.5). It is a steep,
# monotonic rise as the field shrinks, flat above ~15 mm.
#
# It is the shape of a measuring-volume effect, and the sweep's own profiles say how big
# one would have to be. Differentiating the reading with respect to the diameter of the
# averaging volume gives -6.7 %/mm at 5 mm, -1.3 %/mm at 10 mm, and |0.19| %/mm or less at
# every field from 20 mm up — the same collapse, at the same place. Closing the residuals
# above takes 2.9 mm of extra averaging at 5.7 mm and 3.8 mm at 10.1: *one* number, near
# 3 mm, for residuals that differ by a factor of four. `roi_dose` already averages over a
# 2.5 mm voxel at these fields, so that puts the measurement's effective diameter near
# 5-6 mm — a small ion chamber, and exactly the case TRS-483's output correction factors
# exist for, being worth 5-10 % at 1 x 1 cm^2 for chambers of that size.
#
# What that does *not* establish is that the model is right here. `of.dat` states no
# detector and no small-field correction, so this is a consistent account and not a
# demonstration; the alternative — that the transport over-predicts small-field output —
# is not excluded by anything above. Four candidate mechanisms on the model side have been
# eliminated by measurement (leaf-end transmission, the extra-focal weight, source
# occlusion, and the field width, which is now within 0.07 mm at 10 mm), which is what
# leaves detector response as the leading explanation rather than the only one considered.
#
# The practical reading: treat 20 mm and above as the commissioning check, and treat 15 mm
# and below as agreeing to a few per cent at best until someone supplies the detector and
# its corrections. Do not tune the model to close it.

# %%
reference = runs[REFERENCE_FIELD]
reference_roi = roi_dose(reference.scan, DEPTH)
reference_norm = reference_roi * reference.area_iso
reference_rel = reference.spread("roi_dose")[1] / reference_roi

if MINI_PHANTOM_SIDE is None:
    in_air: dict[float, tuple[float, float]] = {}
else:
    print(f"in-air runs for Sc ({MINI_PHANTOM_SIDE:g} cm mini-phantom):")
    mark = time.perf_counter()
    in_air = {field: in_air_output(field) for field in FIELDS}
    print(f"  {time.perf_counter() - mark:.1f} s")

table: list[dict[str, Any]] = []
for field in FIELDS:
    run = runs[field]
    pooled = measure(run.scan)
    normalized = pooled["roi_dose"] * run.area_iso
    relative = run.spread("roi_dose")[1] / pooled["roi_dose"]
    factor = normalized / reference_norm
    row: dict[str, Any] = {
        "field_mm": field * 10.0,
        "equiv_radius_area_mm": 10.0 * field / math.sqrt(math.pi),
        "equiv_radius_ap_mm": 10.0 * field / 2.0,
        "dose_per_fluence": normalized,
        "dose_rel_sigma": relative,
        "output_factor": factor,
        # Independent runs, so the two relative errors combine in quadrature. The
        # reference field's own error is common to every ratio and does not cancel.
        "output_factor_sigma": factor * math.hypot(relative, reference_rel),
        # Each component's own dose per unit on-axis emitted fluence, unweighted.
        # Scp for any other extra-focal weight w is
        #   [(1-w) P(field) + w E(field)] / [(1-w) P(ref) + w E(ref)]
        # so the weight can be re-fitted from this file without re-running.
        "dose_per_fluence_primary": run.component_roi["primary"],
        "dose_per_fluence_extrafocal": run.component_roi["extrafocal"],
    }
    if in_air:
        # Sc is the in-air ratio against the reference field; Sp = Scp / Sc is then
        # the phantom-scatter factor, which is what the TPR conversion wants.
        row["in_air_dose_per_fluence"] = in_air[field][0]
        row["sc"] = in_air[field][0] / in_air[REFERENCE_FIELD][0]
        row["sc_sigma"] = row["sc"] * math.hypot(
            in_air[field][1] / in_air[field][0],
            in_air[REFERENCE_FIELD][1] / in_air[REFERENCE_FIELD][0],
        )
        row["sp"] = factor / row["sc"]
    else:
        row["sc"] = 1.0
        row["sc_sigma"] = 0.0
        row["sp"] = factor
    # Conversion-free, where the SAD setup was run for this field.
    row["tpr_20_10_sim"], row["tpr_20_10_sim_sem"] = simulated_tpr_20_10(field)
    for key, scale in (
        ("dmax", 1.0),
        ("pdd10", 1.0),
        ("pdd20", 1.0),
        ("d20_d10", 1.0),
        ("width_x", 10.0),
        ("width_y", 10.0),
        ("penumbra_x", 10.0),
        ("penumbra_y", 10.0),
        ("flatness_x", 1.0),
        ("symmetry_x", 1.0),
    ):
        row[key] = scale * pooled[key]
        row[f"{key}_sem"] = scale * run.spread(key)[1]
    row["seconds"] = run.seconds
    table.append(row)

print(
    f"{'field':>7} {'Scp':>8} {'+-':>7} {'Sc':>7} {'Sp':>7} "
    f"{'dmax':>6} {'PDD10':>7} {'width_x':>8} {'pen_x':>6}"
)
for row in table:
    print(
        f"{row['field_mm']:7.0f} {row['output_factor']:8.4f} {row['output_factor_sigma']:7.4f} "
        f"{row['sc']:7.4f} {row['sp']:7.4f} "
        f"{row['dmax']:6.2f} {row['pdd10']:7.2f} {row['width_x']:8.1f} {row['penumbra_x']:6.1f}"
    )

# %% [markdown]
# ## TPR curves converted from the depth doses
#
# A full TPR curve costs one simulation per depth, which is why only two were run
# above. The rest of the curve is normally *converted* from the fixed-SSD depth dose
# instead (Khan, *The Physics of Radiation Therapy*, ch. 10). Both setups measure the
# same point in the same phantom; they differ only in the incident fluence, which is
# inverse-square, and in the field size *at the point*, which is what the phantom
# scatter factor `Sp` corrects:
#
# $$\mathrm{TPR}(d) = \frac{D_{\mathrm{SSD}}(d)}{D_{\mathrm{SSD}}(d_{\mathrm{ref}})}
#   \left(\frac{f+d}{f+d_{\mathrm{ref}}}\right)^{2}
#   \frac{S_p(r_{\mathrm{ref}})}{S_p(r_d)}$$
#
# with `f = SSD` and `r_d = a (f + d) / SAD` the field at depth `d` for a field `a`
# set at the isocentre. The reference depth is `DEPTH`, so the curve is
# `TPR(d, DEPTH)` and `TPR(20,10)` can be read straight off it when `DEPTH = 10`.
# `Sp(a)` — the field a beam model actually asks for — cancels out of the ratio.
#
# **`Sp` is simulated, not assumed.** `Sp = Scp / Sc`, and `Sc` comes from running each
# field again into the small in-air phantom above. Earlier revisions of this notebook took
# `Sp ~ Scp` on the grounds that `Sc` was close to 1 by construction — the source is
# normalized per unit fluence at the isocentre plane, so a head that adds no
# field-dependent scatter leaves `Sc` flat. That argument died with the extra-focal
# source: a wider opening lets the point see more of the extended source, which is
# precisely a field-dependent `Sc`, and it is no small correction. The residual against
# the directly simulated `TPR(20,10)` below is the check — and it is a check the
# conversion earns, since nothing in it was fitted to that number.
#
# **The curve runs from the surface, build-up included.** TPR/TMR tables conventionally
# begin at `dmax`, for two reasons — and only one of them still applies here.
#
# The one that did apply was the air path. Build-up dose is contaminant-electron
# dominated, and letting the air run from the device exit to whatever surface each setup
# happens to have would give the depth doses and the TPR runs different contaminant
# components; converting between them would then be comparing two different beams. That
# is fixed by construction above — one air path length for every simulation — so the
# build-up regions of the two setups are now the same beam, and the conversion carries
# through them.
#
# **The converted curves drift high at small fields, and the drift grows with depth.** At
# 10 mm the converted curve runs +2.0 % beyond dmax on average and reaches **+9 % by 30 cm**;
# 20 to 40 mm show a weaker version of the same; everything at 60 mm and above is flat to
# ±1 %. It reproduces across independent sweeps, so it is not noise.
#
# **It is the conversion, not the dose.** Running the 10 mm field's SAD setup directly at
# six depths -- `TPR_DEPTHS = (5, 10, 15, 20, 25, 30)`, no conversion anywhere in it --
# separates the two cleanly, against this machine's `tpr.dat`:
#
#     depth/cm      5      10      15      20      25      30
#     simulated  +0.00   -0.98   -1.84   -1.00   +0.58   +1.78   %
#     converted  -0.00   +0.85   +1.39   +4.83   +6.00   +8.73   %
#
# The simulated column holds to ±1.8 % with mixed signs and no trend; the converted one
# runs away monotonically. So the transported depth dose is right and the arithmetic that
# turns it into a TPR is what fails.
#
# The mechanism is the `Sp` term. It corrects for the field size *at* the point and says
# nothing about the cone above it, which is what actually differs between the two setups:
# at 20 cm depth the fixed-SSD field runs 0.95a at the surface to 1.15a at the point, the
# SAD setup 0.80a to 1.00a, so the SAD setup has less water scattering in over the whole
# path. Path-averaged, the two setups differ by a factor 1.22 in field size at 30 cm, and
# at a small field's sensitivity to field size that is worth several per cent -- which is
# the size of the drift. At 60 mm and above `Sp` has gone flat and the same 1.22 buys
# almost nothing, which is why only the small fields show it.
#
# An earlier revision dismissed this on the `TPR(20,10)` table below, where converted and
# simulated agreed to about +1 % with no trend in field size. That dismissal was wrong, and
# so is the obvious explanation for it. The ratio does *not* cancel the drift: at 10 mm the
# converted curve is +0.85 % high at 10 cm and +4.83 % at 20 cm, so `TPR(20,10)` keeps
# about +3.9 % of it. The table has plenty of power. It was simply being read off a sweep
# whose *simulated* arm carried ~1 % error bars, which is the same size as the effect at
# every field except the smallest.
#
# Re-run with the simulated arm at 0.2-0.6 %, it shows the drift plainly -- against
# `tpr.dat`, the conversion-free column runs +0.21 % at 10 mm while the converted one runs
# +3.93 %, and every field from 20 mm up sits inside +-0.8 % on both. So `TPR(20,10)` is a
# fine test after all, provided the SAD runs are not the noisy half of it. The six-depth
# run is still the better one, because it shows the drift growing with depth rather than
# only that it is there.
#
# What this costs the reader: below ~60 mm, treat the converted curve past ~20 cm depth as
# biased high by a known mechanism rather than as beam data.
#
# **`TPR_DEPTHS` is the fix for small fields only.** An earlier revision of this cell said
# it was the fix for any field, which a control falsified. Running 10, 20, 30, 40 and
# 100 mm at six depths each, against `tpr.dat`, mean and RMS of the residual beyond the
# anchor depth:
#
#     field/mm       10        20        30        40       100
#     direct     +0.01/1.05  -0.80/1.23  -0.51/0.69  -0.33/0.99  -2.52/2.94
#     converted  +3.61/4.79  +0.86/1.14  +0.88/1.34  +0.62/0.93  +0.25/0.49
#
# At 10 mm the direct runs are right and the conversion drifts to +9 % by 30 cm. At 100 mm
# it reverses: the conversion sits at 0.49 % RMS while the direct runs go -4.8 % at 25 cm.
# Somewhere between 40 and 100 mm they cross, and where is not known.
#
# The 100 mm behaviour is not an artefact of that study. The `TPR(20,10)` table below shows
# the same sign in the shipped sweep -- the directly simulated column runs -0.4 to -2.1 %
# against measurement at every field *except* 10 mm, while the converted one holds inside
# +-0.8 %. So the SAD-setup runs carry a systematic deficit at depth that the fixed-SSD
# runs do not, and it is unexplained. A second candidate is that `tpr.dat` was itself
# produced by this same conversion rather than measured at SAD, which would make the
# converted curve agree with it by construction -- but that cannot be the whole story,
# because 10 mm goes the other way.
#
# Practically: below ~40 mm, add the depths you need to `TPR_DEPTHS` and trust those over
# the converted curve. At 100 mm and above, trust the converted curve. In between, they
# agree to about 1 % and the choice does not matter much.
#
# Watch the reference field when reading any of this. `Scp` is normalized at
# `REFERENCE_FIELD`, so one ~1 sigma excursion there shifts every other field together and
# reads as a systematic drift that is nothing of the kind. It happened: with the reference
# on the same target as everything else, its error sat at 0.73 % and a sweep duly landed
# 0.8 % high across the board, every field the same way. `REFERENCE_TARGET_SIGMA` is the
# answer -- at 0.28 % achieved, the per-field `Scp` bars roughly halved and the signs went
# back to mixed. If a future change makes the whole `Scp` column move one way, suspect that
# field's statistics before believing the beam moved.
#
# The device pre-solve does append its exit phase space with atomics, so a fixed seed does
# not give a bit-identical dose. Measured, that is worth **0.1 %** on the on-axis dose --
# an order of magnitude below the statistical bars, and not what the shifts above are.
#
# The one that remains is `Sp`. It is a charged-particle-equilibrium quantity measured at
# a reference depth, and it does not describe how a build-up curve's *shape* varies with
# field size, so the field-size correction is being extrapolated into a region it was not
# built for. The effect is second order — it is a ratio of two `Sp` values at nearby
# field sizes, and it is exactly 1 on the axis of the reference field — but it is real,
# and it is why the shaded region below `dmax` in the figure is marked. Treat the curve
# there as an interpolation guide rather than beam data; if you need the build-up region
# to be right, simulate the SAD setup at those depths directly by adding them to
# `TPR_DEPTHS`, which costs one run each and involves no conversion at all.
#
# One more limit throughout: the `Sp` interpolation clamps at the ends of the field
# sweep, so a field whose projection leaves the measured range is held at the edge value.

# %%
SCP_FIELDS = np.array([row["field_mm"] / 10.0 for row in table])
SP_VALUES = np.array([row["sp"] for row in table])


def phantom_scatter(field_at_point: Any) -> np.ndarray:
    """Relative Sp for a field size at the point of measurement (log-log interpolated)."""
    return np.exp(
        np.interp(
            np.log(np.asarray(field_at_point, dtype=np.float64)),
            np.log(SCP_FIELDS),
            np.log(SP_VALUES),
        )
    )


DMAX = {row["field_mm"] / 10.0: row["dmax"] for row in table}

if max(DMAX.values()) > DEPTH:
    print(
        f"WARNING: the reference depth DEPTH = {DEPTH:g} cm is inside the build-up region "
        f"of at least one field (deepest dmax {max(DMAX.values()):.2f} cm). The converted "
        "TPR curves are normalized at a depth where the conversion does not hold; move the "
        "measurement depth deeper, or read the simulated SAD-setup runs instead."
    )


def tpr_from_pdd(run: FieldResult) -> tuple[np.ndarray, np.ndarray]:
    """Convert one fixed-SSD depth dose into a TPR curve referenced to `DEPTH`.

    Runs the full depth range, build-up included; see above for what that costs.
    """
    depth, curve = depth_dose(run.scan)
    at_reference = float(np.interp(DEPTH, depth, curve))
    scatter = phantom_scatter(run.scan.field * (SSD + DEPTH) / SAD) / phantom_scatter(
        run.scan.field * (SSD + depth) / SAD
    )
    inverse_square = ((SSD + depth) / (SSD + DEPTH)) ** 2
    return depth, (curve / at_reference) * inverse_square * scatter


converted_tpr = {field: tpr_from_pdd(runs[field]) for field in FIELDS}


def converted_tpr_20_10(field: float) -> float:
    """TPR(20,10) read off this field's converted curve."""
    depth, curve = converted_tpr[field]
    return float(np.interp(20.0, depth, curve) / np.interp(10.0, depth, curve))


if any(np.isfinite(row["tpr_20_10_sim"]) for row in table):
    # The conversion is textbook, but Sp only describes the field size *at* the point and
    # not the cone above it, so the two disagree where scatter has not saturated. This is
    # the measurement of that: both are ours, so nothing about the beam model enters it.
    print("\nTPR(20,10): conversion-free against converted")
    print(f"{'field':>7} {'simulated':>10} {'+-':>7} {'converted':>10} {'conv-sim %':>11}")
    for row in table:
        simulated = row["tpr_20_10_sim"]
        if not np.isfinite(simulated):
            continue
        ratio = converted_tpr_20_10(row["field_mm"] / 10.0)
        print(
            f"{row['field_mm']:7.0f} {simulated:10.4f} {row['tpr_20_10_sim_sem']:7.4f} "
            f"{ratio:10.4f} {100.0 * (ratio / simulated - 1.0):+11.2f}"
        )

# %%
fig, ax_tpr = plt.subplots(figsize=(6.6, 4.6))
# Below the deepest dmax the Sp correction is extrapolated (see above): drawn, shaded.
ax_tpr.axvspan(0.0, max(DMAX.values()), color="0.85", alpha=0.5, lw=0)
ax_tpr.text(
    max(DMAX.values()),
    0.02,
    " build-up: Sp extrapolated ",
    transform=ax_tpr.get_xaxis_transform(),
    fontsize=7,
    color="0.35",
    va="bottom",
)
for colour, field in zip(COLOURS, FIELDS, strict=True):
    ax_tpr.plot(*converted_tpr[field], color=colour, lw=1.5, label=f"{field:g} cm")
reference_tpr_dose = {d: v for (f, d), v in tpr_dose.items() if f == REFERENCE_FIELD}
if reference_tpr_dose:
    # Only the reference field is drawn: every field's points would need its own curve to
    # sit against, and the per-field comparison is the TPR(20,10) column in the CSV.
    # The simulated points are dose, not a ratio, so they need the same reference the
    # converted curve uses. When DEPTH was simulated that reference is measured and the
    # overlay is a clean test. Otherwise the set is anchored to the converted curve at
    # its shallowest depth, and only the *shape* — the ratios between the points — is
    # being compared; the printed TPR(20,10) check above is reference-free either way.
    depth, curve = converted_tpr[REFERENCE_FIELD]
    anchor_depth = min(reference_tpr_dose)
    anchored = DEPTH not in reference_tpr_dose
    at_reference = reference_tpr_dose[anchor_depth if anchored else DEPTH][0]
    scale = float(np.interp(anchor_depth, depth, curve)) if anchored else 1.0
    ax_tpr.errorbar(
        list(reference_tpr_dose),
        [scale * value / at_reference for value, _ in reference_tpr_dose.values()],
        yerr=[scale * sem / at_reference for _, sem in reference_tpr_dose.values()],
        marker="o",
        ls="none",
        color="#d8853b",
        capsize=3,
        label="simulated (SAD setup)" + (f", anchored at {anchor_depth:g} cm" if anchored else ""),
    )
ax_tpr.set_title(f"TPR(d, {DEPTH:g} cm) converted from the depth doses")
ax_tpr.set_xlabel("depth (cm)")
ax_tpr.set_ylabel(f"TPR referenced to {DEPTH:g} cm")
ax_tpr.grid(alpha=0.25, lw=0.6)
ax_tpr.legend(fontsize=8, ncol=2, title="field at iso")
fig.tight_layout()
fig.savefig(f"{STEM}_tpr.png", dpi=120)
if SHOW_PLOTS:
    plt.show()

# %% [markdown]
# ## The summary figure
#
# The classic commissioning pair — depth doses and the output-factor curve — saved
# beside this notebook.

# %%
fig, (ax_pdd, ax_of) = plt.subplots(1, 2, figsize=(11.0, 4.4))
for colour, field in zip(COLOURS, FIELDS, strict=True):
    depth, curve = depth_dose(runs[field].scan)
    ax_pdd.plot(depth, 100.0 * curve / curve.max(), color=colour, lw=1.5, label=f"{field:g} cm")
ax_pdd.set_title(f"Central-axis depth dose (SSD {SSD:g} cm)")
ax_pdd.set_xlabel("depth (cm)")
ax_pdd.set_ylabel("dose (% of own maximum)")
ax_pdd.grid(alpha=0.25, lw=0.6)
ax_pdd.legend(fontsize=8, ncol=2, title="field at iso")

ax_of.errorbar(
    [row["field_mm"] for row in table],
    [row["output_factor"] for row in table],
    yerr=[row["output_factor_sigma"] for row in table],
    marker="o",
    color="#2b5fa8",
    capsize=3,
)
ax_of.set_xscale("log")
ax_of.set_title(f"Total scatter factor Scp (ref {REFERENCE_FIELD * 10:g} mm)")
ax_of.set_xlabel("field side at isocentre (mm)")
ax_of.set_ylabel("Scp")
ax_of.grid(alpha=0.25, lw=0.6, which="both")
quality = next(row for row in table if row["field_mm"] == REFERENCE_FIELD * 10)
note = f"PDD(10) = {quality['pdd10']:.1f} %\nD20/D10 = {quality['d20_d10']:.3f}"
if np.isfinite(TPR_20_10):
    note += f"\nTPR(20,10) = {TPR_20_10:.3f} +- {TPR_20_10_SEM:.3f}"
ax_of.text(
    0.04,
    0.96,
    note,
    transform=ax_of.transAxes,
    va="top",
    fontsize=8,
    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
)
fig.tight_layout()
fig.savefig(f"{STEM}.png", dpi=120)
if SHOW_PLOTS:
    plt.show()

# %% [markdown]
# ## Primary fluence in the measurement plane
#
# Zero-variance and analytic: for each spectral bin the ray from the focal spot to
# the point is attenuated through every device chord, through the modelled air and —
# optionally — through the overlying water, then divided by the inverse square of its
# length and scaled by the source's own radial primary fluence at that ray's isocentre
# radius. Normalized to 1 for an unattenuated on-axis ray at the isocentre.
#
# Two things this deliberately is not. It is the *narrow-beam primary* only: the
# total attenuation coefficient removes every scattered photon and none is put back,
# exactly as `CollimatedSource` does. And it traces from a **point** focal spot, so
# its edge carries no source penumbra — with focused jaws it is essentially a step.
# The gap between it and the transported profile is therefore everything the Monte
# Carlo adds: scatter, the finite focal spot, and lateral electron transport.


# %%
def primary_fluence(
    stack: BeamLimitingStack,
    x: np.ndarray,
    y: np.ndarray,
    ssd: float = SSD,
    depth: float = DEPTH,
    *,
    through_phantom: bool = True,
) -> np.ndarray:
    """Primary energy fluence at points in the measurement plane (relative)."""
    points = np.stack([x.ravel(), y.ravel(), np.full(x.size, ssd + depth)], axis=1)
    radius = np.linalg.norm(points, axis=1)
    directions = points / radius[:, None]
    thickness = stack.path_lengths(np.zeros_like(points), directions)  # (n, n_devices)
    air_length = (ssd - stack.exit_z) / directions[:, 2]
    water_length = depth / directions[:, 2] if through_phantom else 0.0

    edges = SPECTRUM.edges
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    weights = SPECTRUM.bin_probabilities
    total = np.zeros(points.shape[0], dtype=np.float64)
    for energy, weight in zip(midpoints, weights, strict=True):
        if weight <= 0.0:
            continue
        tau = np.zeros(points.shape[0], dtype=np.float64)
        for column, device in enumerate(stack.devices):
            tau += (
                device.density
                * XS.mu_over_rho_total(float(energy), device.material)
                * (thickness[:, column])
            )
        tau += AIR_DENSITY * XS.mu_over_rho_total(float(energy), AIR_MATERIAL) * air_length
        tau += XS.mu_over_rho_total(float(energy), WATER) * water_length
        total += weight * energy * np.exp(-tau)
    if FLUENCE is not None:
        # The source's own radial shaping, on the same ray: the table is quoted at the
        # isocentre, so project each point's off-axis distance back to that plane. The
        # on-axis normalization below is unchanged because psi(0) divides out of it.
        iso_radius = np.hypot(points[:, 0], points[:, 1]) * SAD / (ssd + depth)
        total *= FLUENCE.at_radii(iso_radius) / FLUENCE.at_radius(0.0)
    return (total * (SAD / radius) ** 2 / float(np.sum(weights * midpoints))).reshape(x.shape)


reach = 0.75 * REFERENCE_FIELD + 2.0
axis = np.linspace(-reach, reach, 241)
gx, gy = np.meshgrid(axis, axis, indexing="ij")

fig, (ax_map, ax_dose) = plt.subplots(1, 2, figsize=(11.5, 4.6))
image = ax_map.imshow(
    primary_fluence(reference.stack, gx, gy).T,
    origin="lower",
    extent=(axis[0], axis[-1], axis[0], axis[-1]),
    cmap="cividis",
)
fig.colorbar(image, ax=ax_map, label="primary energy fluence (rel.)")
for sign in (1.0, -1.0):
    ax_map.plot([-reach, reach], [-sign * reach, sign * reach], color="w", lw=0.8, ls="--")
ax_map.set_title(f"Primary fluence, measurement plane ({REFERENCE_FIELD:g} cm)")
ax_map.set_xlabel("x (cm)")
ax_map.set_ylabel("y (cm)")

# Normalize the dose map on the water, not the whole slab: dose is energy per unit
# *mass*, so the air voxels above the surface carry a ~800x larger per-history deposit
# and a wild variance with it — they would own the colour scale outright.
grid = reference.scan.grid
slab = reference.scan.dose[:, grid.shape[1] // 2, :]
z = coordinates(grid, 2) - SSD
mesh = ax_dose.pcolormesh(
    z,
    coordinates(grid, 0),
    np.clip(slab / float(slab[:, z >= 0.0].max()), 0.0, 1.0),
    cmap="magma",
    shading="nearest",
    vmin=0.0,
    vmax=1.0,
)
fig.colorbar(mesh, ax=ax_dose, label="dose (rel.)")
ax_dose.axvline(0.0, color="w", lw=0.8, ls=":")
ax_dose.axvline(DEPTH, color="w", lw=0.8, ls="--")
ax_dose.set_title(f"Transported dose, x-z plane ({REFERENCE_FIELD:g} cm)")
ax_dose.set_xlabel("depth from surface (cm)")
ax_dose.set_ylabel("x (cm)")
fig.tight_layout()
fig.savefig(f"{STEM}_fluence.png", dpi=120)
if SHOW_PLOTS:
    plt.show()

# %% [markdown]
# ## Profiles
#
# Left: inplane profiles for every field, normalized to the central axis. With
# `PRIMARY_FLUENCE` set these carry the machine's own horn; with it `None` the
# rounding away from the axis on the larger fields is the missing flattening filter,
# not the machine.
#
# Right: the three scan directions of the reference field against the primary-fluence
# step. The diagonal reaches further — half-width `a/sqrt(2)` at the corner — and
# falls off more gradually, because at a corner two field edges are crossed at once:
# the dose at the exact corner is about a quarter of the axis, not half.

# %%
fig, (ax_all, ax_scan) = plt.subplots(1, 2, figsize=(12.0, 4.6))
for colour, field in zip(COLOURS, FIELDS, strict=True):
    offsets, values = lateral_profile(runs[field].scan, 0)
    ax_all.plot(
        offsets,
        100.0 * values / float(np.interp(0.0, offsets, values)),
        color=colour,
        lw=1.3,
        label=f"{field:g} cm",
    )
ax_all.set_title(f"Inplane dose profiles at {DEPTH:g} cm depth")
ax_all.set_xlabel("off-axis distance (cm)")
ax_all.set_ylabel("dose (% of central axis)")
ax_all.set_ylim(0.0, 120.0)
ax_all.grid(alpha=0.25, lw=0.6)
ax_all.legend(fontsize=8, ncol=2, title="field at iso")

diagonal_reach = reach * math.sqrt(2.0)
leg = np.linspace(-diagonal_reach, diagonal_reach, 401) / math.sqrt(2.0)
fluence = primary_fluence(reference.stack, leg, leg)
ax_scan.plot(
    leg * math.sqrt(2.0),
    100.0 * fluence / float(np.interp(0.0, leg * math.sqrt(2.0), fluence)),
    color="#7a7a7a",
    lw=1.4,
    ls="--",
    label="primary fluence, diagonal",
)
for label, colour, (offsets, values) in (
    ("dose, inplane", "#2b5fa8", lateral_profile(reference.scan, 0)),
    ("dose, crossplane", "#3f8f6b", lateral_profile(reference.scan, 1)),
    ("dose, diagonal", "#d8853b", both_diagonals(reference.scan)),
):
    centre = float(np.interp(0.0, offsets, values))
    ax_scan.plot(offsets, 100.0 * values / centre, color=colour, lw=1.4, label=label)
ax_scan.set_title(f"Scan directions vs. primary fluence ({REFERENCE_FIELD:g} cm)")
ax_scan.set_xlabel("distance from the axis (cm)")
ax_scan.set_ylabel("% of central axis")
ax_scan.set_ylim(0.0, 120.0)
ax_scan.grid(alpha=0.25, lw=0.6)
ax_scan.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{STEM}_profiles.png", dpi=120)
if SHOW_PLOTS:
    plt.show()

# %% [markdown]
# ## Save the tables
#
# `commissioning_demo.csv` is the per-field summary; `commissioning_demo_curves.csv`
# is every curve in long format — depth doses, the TPR curves converted from them,
# both principal profiles, both diagonals, and the deterministic fluence scans in air
# and through the phantom, the off-axis ratios a beam model is fitted against. Floats
# are written to six significant figures, well past any Monte Carlo digit.

# %%
curves: list[dict[str, Any]] = []
for field in FIELDS:
    scan = runs[field].scan
    named = [("pdd", *depth_dose(scan)), ("tpr_from_pdd", *converted_tpr[field])]
    named += [(f"profile_{n}", *lateral_profile(scan, a)) for a, n in ((0, "x"), (1, "y"))]
    named += [
        (f"diagonal_{n}", *diagonal_profile(scan, s)) for s, n in ((1.0, "pos"), (-1.0, "neg"))
    ]

    span = np.arange(-(0.75 * field + 3.0), 0.75 * field + 3.0 + 0.5 * SPACING, SPACING)
    zero, leg = np.zeros_like(span), span / math.sqrt(2.0)
    for direction, x, y in (
        ("inplane", span, zero),
        ("crossplane", zero, span),
        ("diagonal_pos", leg, leg),
        ("diagonal_neg", leg, -leg),
    ):
        for medium, through in (("air", False), ("phantom", True)):
            values = primary_fluence(runs[field].stack, x, y, through_phantom=through)
            named.append((f"fluence_{medium}_{direction}", span, values))

    for kind, xs_, ys_ in named:
        curves += [
            {"field_mm": field * 10.0, "kind": kind, "coord_cm": float(c), "value": float(v)}
            for c, v in zip(xs_, ys_, strict=True)
        ]

for (tpr_field, tpr_depth), (value, _) in tpr_dose.items():
    curves.append(
        {
            "field_mm": tpr_field * 10.0,
            "kind": "tpr",
            "coord_cm": tpr_depth,
            "value": value,
        }
    )

for name, records in ((f"{STEM.name}.csv", table), (f"{STEM.name}_curves.csv", curves)):
    with STEM.with_name(name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(
            {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in record.items()}
            for record in records
        )
    print(f"saved {name}")
