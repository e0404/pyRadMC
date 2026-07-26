# CLAUDE.md

**Read `AGENTS.md` first, in full.** It is the governing document. This file adds only
session-level workflow notes.

## Before you write anything

1. Check `AGENTS.md` section 6 (out of scope) and section 7 (how work is accepted). Work
   outside that scope is out of scope; say so rather than doing it.
2. Check `AGENTS.md` section 8 for a standing constraint covering what you are about to
   touch. Several current defaults are the result of a measurement that came out negative;
   re-deriving one by guesswork is the most expensive mistake available here.
3. Check that a failing test exists for what you are about to write. If not, write it.

## The five things most often gotten wrong here

1. Adjusting the `ref` backend to match a faster backend. Never.
2. Asserting bit-equality across CPU and GPU. Impossible; do not try.
3. Loosening a tolerance to silence a statistical flake. Fix the seed instead.
4. Writing a sampling routine inside `backends/` instead of `physics/`.
5. Hardcoding a cross-section instead of going through `CrossSectionSource`.

## Commands

There is no Makefile; these are the commands, identical on every platform.

```bash
pytest                            # fast tiers: unit, physics, integration, dij
pytest -m ""                      # every tier, adding validation and perf
pytest -m validation              # validation tier alone (slow; needs EPICS data)
pytest -m perf --benchmark-only   # perf tier against recorded baselines
ruff check . && ruff format .     # lint and format
mypy pyRadMC                      # types (--strict; kernels exempt)
pre-commit run --all-files        # everything the commit hook gates on
mkdocs build --strict             # docs, as CI and ReadTheDocs build them
```

The commit hook runs the fast gates only. Tests are CI's job — run `pytest` yourself.

## When you are uncertain

State the uncertainty and stop. On a stochastic code, a plausible-looking guess that happens to
pass a loose test is worse than no code, because it will be trusted later.

Specifically, ask rather than decide when:

- A change would alter `ECUT`, `PCUT`, or the Dij truncation threshold.
- A new dependency is needed (state its license).
- A test tolerance would need to change.
- An approximation is being introduced that is not already named in `AGENTS.md`.
