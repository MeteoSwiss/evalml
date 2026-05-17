# CI

EvalML's GitHub Actions live under `.github/workflows/`.

## `ci.yaml`

The main CI workflow runs on every push to `main` and every PR targeting
`main`.

### Job: `test`

Matrix:

- `os`: `ubuntu-latest`
- `python-version`: `3.11`, `3.12`, `3.13`
- `anemoi-dev`: `false` today; the leg that's `true` would install
  `anemoi-inference` and `anemoi-datasets` from `main`. Keep an eye on
  this if you change anything inference-adjacent.

Steps:

1. `astral-sh/setup-uv@v2` (with caching).
2. `actions/setup-python@v5` to install the matrix Python.
3. `uv sync --all-extras --dev` to install the project + dev tools +
   the `kerchunk` extra.
4. `uv run pytest tests/`.

### Job: `lint`

Runs `pre-commit run --all-files --verbose`, which exercises:

- `trailing-whitespace`, `end-of-file-fixer`.
- `ruff` (auto-fix) and `ruff-format`.
- `snakefmt workflow/`.
- The local `pydantic-schema` hook that regenerates
  `workflow/tools/config.schema.json` and asserts no diff.

## `docs.yaml`

Triggered on push to `main`, on PR open/reopen/sync/close, and via
`workflow_dispatch`. It has three jobs:

### `build`

1. Installs `uv` and Python 3.12.
2. Runs `uv sync --group docs`.
3. Builds the docs with
   `sphinx-build -W --keep-going -b html docs/source docs/build/html`.
4. Uploads the result as a workflow artifact (`html-docs`, 7-day
   retention) so reviewers can download the rendered HTML and so the
   deploy jobs can pick it up.

`-W` turns warnings into errors, so any broken cross-reference,
malformed directive, or missing module fails the build. If the build
fails on something genuinely unfixable on the docs side (e.g. a heavy
runtime dependency that can't be installed in CI), add the offending
import to `autodoc_mock_imports` in `docs/source/conf.py`.

The build is skipped on PR close — only the preview-cleanup needs to
run at that point.

### `deploy-main`

Only runs on push to `main`. Downloads the `html-docs` artifact and
publishes it to the root of the `gh-pages` branch via
[`JamesIves/github-pages-deploy-action`](https://github.com/JamesIves/github-pages-deploy-action).
The `clean-exclude: pr-preview/` setting preserves PR preview
directories so a main-branch deploy doesn't wipe open previews.

### `preview`

Runs on every PR event. Uses
[`rossjrw/pr-preview-action`](https://github.com/rossjrw/pr-preview-action)
to deploy the built site to
`gh-pages/pr-preview/pr-<number>/`, and to remove that directory when
the PR is closed. The action also posts and updates a comment on the
PR with the preview URL.

## GitHub Pages

The rendered docs are hosted on GitHub Pages from the `gh-pages` branch.
To enable Pages on a fresh fork or new repository:

1. Go to **Settings → Pages**.
2. Under **Build and deployment** → **Source**, select
   **Deploy from a branch**.
3. Set **Branch** to `gh-pages` and **Folder** to `/ (root)`.
4. Save.

The first push to `main` after that publishes the site at
`https://<owner>.github.io/<repo>/`. PR previews appear at
`https://<owner>.github.io/<repo>/pr-preview/pr-<number>/`.

Note: the `gh-pages` branch is created automatically by the deploy
action on the first run; you don't need to create it manually.
