# CLAUDE.md

**Read `AGENTS.md` first, in full.** It is the governing document. This file adds only
session-level workflow notes.

## Before you write anything

1. Check `AGENTS.md` section 7 for the current phase. Work outside the current phase is
   out of scope; say so rather than doing it.
2. Check that a failing test exists for what you are about to write. If not, write it.

## The five things most often gotten wrong here

1. Adjusting the `ref` backend to match a faster backend. Never.
2. Asserting bit-equality across CPU and GPU. Impossible; do not try.
3. Loosening a tolerance to silence a statistical flake. Fix the seed instead.
4. Writing a sampling routine inside `backends/` instead of `physics/`.
5. Hardcoding a cross-section instead of going through `CrossSectionSource`.

## Commands

```bash
make test        # fast tiers: unit, physics, integration, dij
make test-all    # adds validation and perf
make lint        # ruff check + format --check
make types       # mypy --strict on non-kernel code
make bench       # perf tier against recorded baselines
```

## When you are uncertain

State the uncertainty and stop. On a stochastic code, a plausible-looking guess that happens to
pass a loose test is worse than no code, because it will be trusted later.

Specifically, ask rather than decide when:

- A change would alter `ECUT`, `PCUT`, or the Dij truncation threshold.
- A new dependency is needed (state its license).
- A test tolerance would need to change.
- An approximation is being introduced that is not already named in `AGENTS.md`.
