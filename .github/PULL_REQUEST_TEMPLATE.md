<!--
The PR title becomes the squash-commit subject on develop. It should follow
AGENTS.md 9.2:  type(scope): imperative description
(description <= 50 chars; whole line <= 72)
The description below becomes the commit body: wrap lines at 72.
-->

## What and why

<!-- The physics / the reasoning, not the diff. Measurements go here. -->

## Checklist

- [ ] A failing test motivated this change (AGENTS.md 2.1)
- [ ] `pre-commit run --all-files` and `pytest` pass locally
- [ ] Docs, README and docstrings this change made stale are updated
- [ ] If a user-visible capability: a runnable example under `examples/`
- [ ] If it can move a computed dose: `CHANGELOG.md` entry + the measurement
- [ ] If it touches `ECUT`/`PCUT`/truncation/step defaults: the dosimetric-effect
      test of AGENTS.md 2.8 is included
