# Releasing ocr-bench

A release is one merged PR. The PR bumps `version` in `pyproject.toml` and adds a dated section for that version to `CHANGELOG.md`. When the PR is merged to `main`, `release.yml`:

1. checks that the version has no published GitHub release yet, and that `CHANGELOG.md` has a `## [X.Y.Z] - YYYY-MM-DD` section for it;
2. runs the CI suite (`ci.yml`);
3. builds the wheel and sdist, checks that the wheel imports and reports version X.Y.Z, and publishes both to PyPI with trusted publishing;
4. tags the merge commit `vX.Y.Z` and creates the GitHub release. The release notes are the CHANGELOG section, and the built files are attached.

Do not push a tag by hand. The workflow creates it. A push to `main` that changes `pyproject.toml` but keeps an already-released version does nothing.

## Prerequisites

- Push access to `davanstrien/ocr-bench` on GitHub
- [Trusted publishing](https://docs.pypi.org/trusted-publishers/) configured on PyPI:
  - PyPI project: `ocr-bench`
  - Repository: `davanstrien/ocr-bench`
  - Workflow: `release.yml`
  - Environment: `pypi`
- GitHub environment `pypi` exists (Settings > Environments)
- Branch protection on `main` (require PR + passing CI)

## Release steps

1. **Create a release branch**:

   ```bash
   git checkout -b release/0.0.6
   ```

2. **Bump the version** in `pyproject.toml` and update the lockfile:

   ```bash
   uv lock
   ```

3. **Date the CHANGELOG section.** In `CHANGELOG.md`, add `## [0.0.6] - YYYY-MM-DD` below `## [Unreleased]` and move the unreleased entries under it. Use the planned merge date. The heading must match that pattern exactly, with nothing after the date.

4. **Run checks locally**:

   ```bash
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   uv run ty check src/
   uv run --extra viewer pytest tests/ -x -q
   ```

5. **Commit, push and open a PR** against `main`:

   ```bash
   git add pyproject.toml uv.lock CHANGELOG.md
   git commit -m "Release 0.0.6"
   git push -u origin release/0.0.6
   gh pr create --title "Release 0.0.6"
   ```

6. **Merge the PR** once CI is green. This starts the release. Merge one release PR at a time, and wait for its release run to finish before you merge the next. GitHub keeps only one pending run in the `release` concurrency group, so a newer pending run replaces an older one.

7. **Watch the release workflow**:

   ```bash
   gh run list --workflow release.yml --limit 1
   gh run watch <run-id>
   ```

8. **Verify**:

   ```bash
   gh release view v0.0.6
   uv run --isolated --no-project --with ocr-bench==0.0.6 ocr-bench --help
   ```

## If a release stops part-way

- **Re-run all jobs** of the run (Actions > Release > Re-run all jobs), not only the failed ones. The `check` job then runs again: it stops if the release was already published, and it stops with a message if an earlier run left a draft release. `uv publish --check-url` skips files that PyPI already has, so a re-run after a successful upload goes on to create the tag and the GitHub release. This works because `uv_build` produces byte-identical files when it rebuilds the same commit with the same `uv_build` version.
- **An unfinished draft release** (`gh release create` makes a draft, uploads the files, then publishes): delete the draft with `gh release delete vX.Y.Z --yes`, then re-run all jobs.
- **The fix needs a code change** (for example a missing CHANGELOG section): merge the fix in a new PR, then start the workflow by hand (Actions > Release > Run workflow, on `main`). A PR that does not change `pyproject.toml` does not start the release by itself. A manual run tags the current head of `main`. A manual run on another branch does nothing.
- **PyPI never accepts the same version twice with different files.** If a different build of the version is already on PyPI, bump to a new version.

## CI workflows

### `ci.yml` — runs on PRs and pushes to main, and from `release.yml`

Lint (ruff), type check (ty) and tests (pytest). This is the check required by branch protection.

### `release.yml` — runs on pushes to main that change `pyproject.toml`, or by hand on main

1. **check** job: reads the version; stops if it is already released; fails if a draft release is left over or the CHANGELOG section is missing
2. **test** job: the `ci.yml` suite
3. **publish** job: build > smoke test the wheel > publish via trusted publishing > tag and GitHub release

## Branch protection setup

Go to **Settings > Branches > Add rule** for `main`:

- Require a pull request before merging
- Require status checks to pass: select `test` (from the CI workflow)
- Do not allow bypassing the above settings (optional, but recommended)

## Troubleshooting

- **Lockfile out of sync**: CI uses `--locked`, so always run `uv lock` after changing `pyproject.toml`
- **Missing dev tool in CI**: Dev dependencies (ruff, pytest) must be in `[dependency-groups] dev`
- **test_web.py fails in CI**: Needs `--extra viewer` since FastAPI is optional

## Versioning

[SemVer](https://semver.org/). Currently `0.0.x` (proof of concept). Bump to `0.1.0` when the API stabilises.
