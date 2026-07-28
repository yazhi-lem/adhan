# CI Workflow Cleanup Proposal

Review of `.github/workflows/{lint,test,type-check}.yml` against `pyproject.toml`.
Small, safe fixes have already been applied directly (see "Done now" below);
everything else is a proposal for a follow-up PR since it changes CI structure
and is worth a second pair of eyes before merging.

## Done now (applied in this pass)

1. **Dependency drift between YAML and `pyproject.toml`** — `lint.yml` ran
   `pip install black isort ruff` and `type-check.yml` ran
   `pip install mypy numpy pyyaml`, both hardcoding *unpinned* versions of
   tools that are already declared (with version floors) in `pyproject.toml`'s
   `dev` extra. Two sources of truth for the same tool versions means the
   workflow can silently drift from what contributors install locally via
   `pip install -e ".[dev]"`. Both jobs now install `-e ".[dev]"` instead,
   matching `test.yml`'s pattern.

2. **Security scan result was unconditionally discarded** — `bandit -r src/ -ll || true`
   makes the step exit 0 no matter what bandit finds, so a real finding never
   shows up as anything but a green check — not even a warning annotation.
   Moved the "don't block the build" intent to `continue-on-error: true` at
   the step level instead, so a finding now shows as a visible
   "failed but continued" annotation in the Actions UI rather than being
   invisible.

3. **Double-suppressed mypy exit code** — `type-check.yml` piped mypy through
   `head -100 || true` *and* set `continue-on-error: true` on the same step.
   Piping through `head` already discards mypy's real exit code (the step's
   exit status becomes `head`'s), so the trailing `|| true` was redundant on
   top of a redundancy. Dropped `|| true`; `continue-on-error: true` alone
   already keeps the job non-blocking, and it's now clearer that
   *that* line is what makes it non-blocking.

## Proposed for a follow-up (not applied — structural change)

4. **Three workflows are near-duplicates of each other.** `lint.yml`,
   `test.yml`, and `type-check.yml` each repeat the identical `on:` block
   (`push: [main, claude/**]`, `pull_request: [main]`) and the identical
   checkout + `actions/setup-python@v4` boilerplate. Every push/PR currently
   spins up 4 separate runners (lint, security, test×3-matrix, mypy) that all
   redo `pip install --upgrade pip` from scratch. Consolidating into one
   `ci.yml` with `lint`, `security`, `test`, and `typecheck` jobs sharing a
   single trigger block would:
   - cut the maintenance surface from 3 files to 1
   - make it obvious at a glance which jobs run on which trigger
   - let `test` reuse pip cache warmed by `lint`'s setup-python step less
     redundantly (each job still gets its own runner, but the YAML itself
     stops repeating)

   This is a structural change (touches CI topology), so it's proposed here
   rather than applied inline — worth confirming no one depends on the
   current per-file check names in branch protection rules before merging.

5. **`actions/setup-python@v4` is behind current (`v5`).** Cosmetic/low-risk;
   worth bumping in the same follow-up PR rather than as a one-off, so the
   version bump gets tested alongside the consolidation.

6. **`ruff` config already ignores `E501`** (`pyproject.toml` `[tool.ruff]`)
   because black owns line-length — this is *not* redundant, just noting it
   was checked: black and ruff are intentionally divided (formatting vs.
   lint rules), no overlap to clean up there.
