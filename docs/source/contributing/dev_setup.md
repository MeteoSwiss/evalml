# Development setup

EvalML's development workflow centres on `uv` and `pre-commit`. Get them
working once and the rest of the toolchain (Sphinx, `snakefmt`,
`ruff`, schema validation) plugs in via `pre-commit`.

## Prerequisites

- Python 3.11+ (managed by `uv`).
- A clone of the repo:

  ```bash
  git clone https://github.com/MeteoSwiss/evalml.git
  cd evalml
  ```

## Install dev dependencies

```bash
uv sync --dev
source .venv/bin/activate
```

`uv sync --dev` adds the project's runtime dependencies *and* the
`dev` group from `pyproject.toml` (currently `pre-commit`,
`snakefmt`).

For docs work, also install the `docs` group:

```bash
uv sync --dev --group docs
```

## Install pre-commit hooks

```bash
pre-commit install
```

Now every `git commit` runs:

- Trailing-whitespace and end-of-file fixes.
- `ruff` (lint + auto-fix) and `ruff-format`.
- `snakefmt workflow/`.
- A regenerate-and-diff check on `workflow/tools/config.schema.json` —
  if your changes to `src/evalml/config.py` would alter the schema and
  you didn't regenerate it, the hook fails.

To run the full hook set on demand:

```bash
pre-commit run --all-files
```

## Building the docs locally

```bash
uv sync --group docs
sphinx-build -W --keep-going -b html docs/source docs/build/html
open docs/build/html/index.html
```

For live preview during authoring:

```bash
sphinx-autobuild docs/source docs/build/html
```

The published site is hosted on GitHub Pages and rebuilt automatically
on every push to `main` by `.github/workflows/docs.yaml`. Open PRs get
a live preview at
`https://<owner>.github.io/<repo>/pr-preview/pr-<number>/`; the URL is
posted as a comment on the PR by `rossjrw/pr-preview-action` and
updates on every push.
