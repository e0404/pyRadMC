# Contributing to pyRadMC

Thanks for your interest. This is a dose engine, so the bar for a change is a little
different from most Python projects: a plausible-looking patch that passes a loose test is
worse than no patch, because it will be trusted later.

**Read [`AGENTS.md`](https://github.com/e0404/pyRadMC/blob/main/AGENTS.md) in full before
writing code.** It is the governing
document — written for human and agentic contributors alike — and it explains the rules
that this file only summarises.

## Setup

```bash
pip install -e ".[warp,dev,examples]"
pre-commit install    # once per clone
```

## Commands

There is no Makefile; every task is a single command that works the same on Linux, macOS
and Windows.

| Task | Command |
|---|---|
| Run the gates over the whole tree | `pre-commit run --all-files` |
| Fast test tiers (unit, physics, integration, dij) | `pytest` |
| Every tier, including validation and perf | `pytest -m ""` |
| Validation tier only (slow; needs EPICS data) | `pytest -m validation` |
| Benchmarks against recorded baselines | `pytest -m perf --benchmark-only` |
| Lint and format | `ruff check .` / `ruff format .` |
| Types | `mypy pyRadMC` |
| Build the docs | `mkdocs build --strict` |
| Preview the docs | `mkdocs serve` |

The pre-commit hook runs `ruff`, `ruff format`, `mypy --strict` and a few file-hygiene
checks — all fast and file-scoped. It deliberately does **not** run the tests: the suite
takes minutes, and a gate slow enough to be resented is a gate that gets bypassed with
`--no-verify`. CI runs the fast tiers on every push to `main` and `develop` (GitHub
Actions and GitLab CI both), and the validation tier on a schedule. **Run `pytest`
yourself before pushing anything you care about.**

## Branches and pull requests

The branch model is `AGENTS.md` section 9. The short version:

- Cut a **task branch from `develop`** — one task per branch, descriptive name. Nothing
  is committed directly to `develop` or `main`; `main` carries releases only.
- Open a pull request onto `develop`. After review and green CI it is **squash-merged**,
  so **the PR title becomes the commit subject** and must follow the commit convention:

  ```
  type(scope): imperative description        # description ≤ 50; whole line ≤ 72
  ```

  Types: `feat` `fix` `perf` `refactor` `docs` `test` `build` `ci` `chore` `revert`.
  Scope is optional, usually a module name (`transport`, `data`, `warp`, `dij`, ...).
  The PR description becomes the commit body: state the physics and the why, wrapped at
  72 columns after a blank line, with footers (`Co-Authored-By:`, `Refs: #123`) last.
- On the task branch itself the convention is *encouraged, not enforced*; the PR title
  is what is held to it in review.
- Releases are cut by the maintainer from `rc/X.Y.Z` branches (`AGENTS.md` 9.4).

## The workflow

1. **Write the failing test first.** Every change starts red. On a stochastic code this is
   not dogma: it is the only way to know a test can fail at all.
2. **Make it pass** without touching the reference backend to accommodate a faster one.
3. **Run the tiers you affected.** `pytest` for the fast ones; `pytest -m ""` if you
   touched physics, data or anything the validation tier gates.
4. **Update the docs in the same change** — README capability list, `docs/`, docstrings,
   and any comment or skip-reason your change made stale.
5. **Add a `CHANGELOG.md` entry.** If the change can move a computed dose, say so and give
   the measured size of the effect.

## The five things most often gotten wrong here

1. Adjusting the `ref` backend to match a faster backend. Never — it is the oracle.
2. Asserting bit-equality across CPU and GPU. Atomic float accumulation is
   non-associative; this is impossible, not merely difficult. Test statistically.
3. Loosening a tolerance to silence a statistical flake. Fix the seed instead, and if the
   tolerance really is wrong, say so explicitly and show the measurement.
4. Writing a sampling routine inside `backends/` instead of `physics/`.
5. Hardcoding a cross-section instead of going through `CrossSectionSource`.

## Ask, don't guess

Open an issue before writing code when a change would:

- alter `ECUT`, `PCUT`, the Dij truncation threshold, or the electron substep fraction —
  these are accuracy-defining (`AGENTS.md` 2.8) and need a measured dosimetric effect on
  realistic spectral, heterogeneous geometry;
- need a new runtime dependency (state its licence);
- require a test tolerance to change;
- introduce an approximation not already named in `AGENTS.md`;
- touch anything listed in `AGENTS.md` section 8. Several current defaults are there
  because a measurement came out negative. Re-deriving one by guesswork is the most
  expensive mistake available in this repository — `docs/decisions.md` records what was
  already tried and what it cost.

## Adding a capability

A change that adds something a user would reach for ships with a runnable script under
`examples/` that exercises it on a small but physically meaningful problem and saves an
intuitive figure beside itself. A depth-dose curve reads; a printed array does not. Keep it
under a minute on a laptop; examples run manually, not in CI.

## Licence

By contributing you agree that your contribution is licensed under Apache-2.0, the same
terms as the project.
