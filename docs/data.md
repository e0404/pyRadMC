# Cross-section data

pyRadMC ships **no evaluated cross-section libraries**. There are two sources of
interaction data, and which one you use is a deliberate accuracy decision.

## Analytic

Closed-form parameterizations, water only, no data files, correct to a few percent
between 0.05 and 20 MeV. Klein-Nishina without electron binding, an `E^-3` photoelectric
power law, a near-threshold pair shape least-squares calibrated so the *total* attenuation
reproduces the NIST/XCOM water table, and no coherent channel.

Good for standing up a calculation, development, and anything where a few percent in the
soft spectrum does not matter. It reports itself honestly in `result.provenance`.

```python
from pyradmc import AnalyticCrossSections
xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
```

## Tabulated

Compiled from the IAEA EPICS 2023 evaluations: EPDL photon cross-sections (sub-percent
against NIST XCOM) including MF=27 coherent form factors, EEDL elastic scattering, and
ICRU-37 Berger-Seltzer electron stopping. Covers the whole material registry.

```bash
python -m pyradmc.data.tabulated.build --output materials.npz
```

```python
from pyradmc import TabulatedCrossSections
from pyradmc.data.tabulated.format import load_tables

xs = TabulatedCrossSections(load_tables("materials.npz"),
                            geometry_densities=grid.max_density_by_material())
```

The build downloads `EPDL2023.ALL` and `EEDL2023.ALL` (~120 MB) from
[www-nds.iaea.org/epics](https://www-nds.iaea.org/epics/ENDF2023/) and caches them under
`~/.cache/pyradmc/epics`.

## Integrity

**Every library is verified against a pinned SHA-256 before it is used** — freshly
downloaded or served from the cache. These bytes *are* the cross-sections: a truncated
transfer, a corrupted cache entry, or a silent upstream revision would otherwise
propagate into every dose the engine computes, and would surface as unexplained drift in
a dose gate rather than as an error.

The check is fail-closed and deliberately awkward to bypass:

- Downloads are hashed **as they stream** and verified **before** the temporary file is
  renamed, so a bad transfer never becomes a cache entry.
- Cache hits are re-hashed, because a file that was correct when written may not be
  correct now.
- `--allow-unverified-library` exists for the case where the IAEA has legitimately
  revised a library. Using it means your tables are not the ones this release validated,
  and the tool says so.

Updating a pinned digest is not maintenance — it changes the cross-sections. Re-run the
validation tier against the new library and record the outcome first.

## Multiple-scattering table cache

Reference and Warp transport share the deterministic Goudsmit-Saunderson grid at
`~/.cache/pyradmc/gs-grid`. Warp precomputes its reachable rectangle; the reference
sampler reads that file on its first node request and retains it in memory. With no
stored coverage, it builds the smallest rectangle containing the requested nodes and
the stored window, reusing every existing node. Short reference runs therefore do
not initialize the full reachable grid. A rectangle can include unrequested interior
nodes; the tradeoff preserves the existing dense format and its eager-grid reader.
The reference sampler requests its four interpolation rows together, extending and
saving once per bracket. Both builders use the same column workers when an extension
adds at least 256 nodes; smaller extensions stay serial.

The cache changes startup cost only: node construction, transport seeds, cutoffs and
dose values are unchanged. The construction version, build constants and NumPy feature
version identify the file, including the inverse-CDF grid resolution. Existing version-1
files that implicitly used 4096 inverse-CDF bins remain readable at that resolution;
extensions save under the explicit identity. Geometry and source changes extend its
window. Builds made
through `build_gs_grid` with explicit grid-constant overrides remain in memory. A
cache write failure in either backend logs a warning and retains the tables in memory.

Writes share the eager builder's thread lock and use atomic replacement, with temporary
files removed after failed saves. Concurrent processes can duplicate deterministic
work, with the last completed file winning;
this affects cache coverage, never node values. CI caches `~/.cache/pyradmc` under the
construction identity and source hash, separated by OS and Python version.
Ordinary reference tests also use this home-directory cache; only tests that redirect
the private cache directory are isolated. Warm-suite timings therefore depend on its
stored coverage.
The [startup measurements](decisions.md#reference-gs-startup) document coverage and
the tradeoff between lazy and eager initialization.

## Licence and provenance

The EPICS libraries are the IAEA's. Their terms of use apply to you as the person who
downloads them, and are not granted by pyRadMC's licence. Nothing derived from them is
redistributed here; the compiled `.npz` is produced on your machine.

Published reference data reproduced in the source — NIST XCOM and ESTAR anchors, ICRU-37
formulations, ICRP/ICRU tissue compositions, Sternheimer coefficients, IUPAC atomic
weights, the Ali & Rogers MV spectrum parameters — is cited at each point of use, and
collected in the repository's `NOTICE` file.

## Materials

The registry carries water, air, lung, adipose tissue, cortical bone (the ICRP/ICRU
reference formulations, so every number is independently checkable), tungsten, and the
W95/Ni3.5/Cu1.5 heavy alloy that MLC leaves are actually made of. Each entry has
elemental mass fractions, an ICRU-37 I-value, exact Sternheimer density-effect
coefficients, and per-material ESTAR radiative anchors.

```python
from pyradmc import MATERIALS, WATER, LUNG, CORTICAL_BONE
```

The analytic source answers for water only and raises for anything else — a source that
silently answered for a material it has no data for would be a silent transport bias.
