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
on every push to `main` by `.github/workflows/docs.yaml`. PR builds
upload an `html-docs` artifact that you can download from the workflow
run page to preview changes before merging.

## Running the workflow on a developer laptop

The workflow is designed for SLURM but will run locally with reduced
parallelism. Pick a small config, force a single core, and disable
SLURM-only resources by passing extra Snakemake args after `--`:

```bash
evalml experiment config/forecasters-co2.yaml --cores 1 -- --executor local
```

For most rule-level debugging, prefer `evalml make CONFIG TARGET --dry-run`
first to confirm the DAG, then drop `--dry-run`.
