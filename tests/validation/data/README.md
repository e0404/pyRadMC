# Validation benchmark data

## center_ray_dose_EGSNrc_intRadius6.0mm.txt

Central-axis depth dose in water from **EGSnrc (full physics)**, supplied by the
maintainer on 2026-07-10. Stored verbatim.

**Known provenance** (maintainer statement):

- Wide uniform photon beam (lateral scatter equilibrium) in a large water cylinder.
- Scored on the central cylinder of radius 6 mm; depth bins of 0.25 cm to ~50 cm.
- Monoenergetic columns from 0.05 to 7 MeV; dose in arbitrary units, no per-point
  uncertainties provided.

**Missing metadata** (recorded so nobody mistakes the gate for tighter than it is):

- EGSnrc transport parameters (ECUT/PCUT, Rayleigh on/off, brems model, EII).
- Exact cylinder radius and length; source normalization.

**How the gate uses it** (`test_ranges_and_pdd.py`): for a laterally-equilibrated
uniform beam, the central-axis dose equals the laterally *integrated* dose of a pencil
beam (pencil-kernel superposition), so the test runs a pencil beam and scores slice
sums — every history contributes at every depth. Normalization is free (least squares)
because the benchmark units are arbitrary.

Columns below ~1 MeV are not gated in Phase 1: the analytic cross-section layer is an
explicitly few-percent parameterization for the soft spectrum (Klein-Nishina without
binding, crude photoelectric power law, no Rayleigh), which is also why the Phase 1
gamma criterion is 5%/3mm rather than the 2%/2mm the tabulated backend (Phase 5) must
meet against this same file.
