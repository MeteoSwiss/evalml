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

Triggered on push to `main`, on PRs that touch `docs/**`, `src/**`,
`pyproject.toml`, or the workflow itself, and via `workflow_dispatch`.
It has two jobs:

### `build`

1. Installs `uv` and Python 3.12.
2. Runs `uv sync --group docs`.
3. Builds the docs with
   `sphinx-build -W --keep-going -b html docs/source docs/build/html`.
4. On `push` to `main`, uploads the result as a GitHub Pages artifact.
5. On PRs, uploads the result as a regular workflow artifact
   (`html-docs`, 7-day retention) so reviewers can preview the rendered
   pages without merging.

`-W` turns warnings into errors, so any broken cross-reference,
malformed directive, or missing module fails the build. If the build
fails on something genuinely unfixable on the docs side (e.g. a heavy
runtime dependency that can't be installed in CI), add the offending
import to `autodoc_mock_imports` in `docs/source/conf.py`.

### `deploy`

Only runs on `push` to `main`. Uses
[`actions/deploy-pages`](https://github.com/actions/deploy-pages) to
publish the Pages artifact uploaded by the `build` job.

## GitHub Pages

The rendered docs are hosted on GitHub Pages. To enable Pages on a
fresh fork or new repository:

1. Go to **Settings → Pages**.
2. Under **Source**, select **GitHub Actions**.

The first push to `main` after that will publish the site at
`https://<owner>.github.io/<repo>/`. Subsequent pushes redeploy
automatically.

The `concurrency: pages` group in the workflow ensures that overlapping
deploys are serialised — a fast-following push won't race with an
in-flight deploy. PRs intentionally do not deploy; reviewers download
the `html-docs` artifact from the workflow run instead.
